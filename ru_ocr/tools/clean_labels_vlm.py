"""Clean recognition label files with the model predictions and a VLM third opinion.

A line whose label differs from our model's prediction (`pred`, exact compare, spaces count) is kept only when
the VLM reading confirms one side: VLM == label -> the label stays, VLM == pred -> the label becomes pred. When
label and pred differ only in spaces, the VLM's space positions alone decide (its characters may be misread, see
vlm_verdict). Every other disputed line is dropped. Only the VLM text is normalized (whitespace runs -> one
space, ends stripped); label and pred are compared as they are. Lines the model got right and lines without a
prediction (e.g. other datasets) are written unchanged.

    python ru_ocr/tools/clean_labels_vlm.py \
        --preds output/stroyinf_preds/preds_all.tsv.gz output/stroyinf_modern_preds/preds_all.tsv.gz \
        --vlm output/stroyinf_preds/review_vlm.tsv output/stroyinf_preds/review_spaces_vlm.tsv \
              output/stroyinf_modern_preds/review_vlm.tsv \
        --labels data/ru_labels/all_train_w_h.txt data/ru_labels/all_val_w_h.txt

writes <label file>_clean next to each label file (all_train_w_h.txt -> all_train_clean_w_h.txt) and never
overwrites an existing file. --preds and --vlm take several tables (one per prediction / VLM run); the review
tables' columns: path, label, pred, vlm.
"""

import argparse
import csv
import gzip
import os
import re
import sys
from collections import Counter


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_table(path, columns):
    """Rows of a TSV with a header as tuples of `columns` (no quoting: texts hold quotes)."""
    with open_text(path) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [tuple(row[c] for c in columns) for row in reader]


# Cyrillic and Latin letters drawn the same: a crop cannot tell them apart, neither the model nor the VLM can
CYR_TWINS = "АВЕКМНОРСТХУаеорсух"
LAT_TWINS = "ABEKMHOPCTXYaeopcyx"
TO_CYR = str.maketrans(LAT_TWINS, CYR_TWINS)
TO_LAT = str.maketrans(CYR_TWINS, LAT_TWINS)
WORD = re.compile(r"[^\W\d_]+")


def alphabet(ch):
    if "Ѐ" <= ch <= "ӿ":
        return "cyr"
    if ch.isascii() and ch.isalpha():
        return "lat"
    return None


def unify_words(text):
    """A word mixing Cyrillic and Latin gets its look-alike letters in the alphabet of most of its letters
    (Cyrillic on a tie)."""
    def fix(match):
        word = match.group(0)
        n = Counter(alphabet(ch) for ch in word)
        if not (n["cyr"] and n["lat"]):
            return word
        return word.translate(TO_LAT if n["lat"] > n["cyr"] else TO_CYR)

    return WORD.sub(fix, text)


def lookalike_verdict(label, pred):
    """label and pred differ only in look-alike letters: the side written in the alphabet of the line's letters
    that have no twin wins ('OK' -> 'ОК' in a Russian line); with no such letters (a lone 'Е') the label stays,
    the text layer holds the real character."""
    n = Counter(alphabet(ch) for ch in label if ch not in CYR_TWINS + LAT_TWINS)
    if n["cyr"] == n["lat"]:
        return "label_ok"
    twins = CYR_TWINS if n["cyr"] > n["lat"] else LAT_TWINS
    in_alphabet = [sum(ch in twins for ch in text) for text in (label, pred)]
    return "pred_ok" if in_alphabet[1] > in_alphabet[0] else "label_ok"


def space_layout(text):
    """(characters without spaces, positions in that string before which a space stands)."""
    chars, gaps = [], set()
    for ch in text:
        if ch == " ":
            gaps.add(len(chars))
        else:
            chars.append(ch)
    return len(chars), frozenset(gaps)


def vlm_verdict(label, pred, vlm):
    """'label_ok', 'pred_ok' or None; only the VLM text is normalized.

    An exact match decides first. When label and pred differ only in spaces (the characters are confirmed by
    both), the VLM decides by where it puts its spaces alone, its characters may be misread: the same number of
    characters and the same space positions as the label -> label_ok, as pred -> pred_ok. A pred with a space at
    an end (a model artifact, labels never have one) is compared without it but never chosen. When they differ
    only in look-alike letters, the VLM cannot tell either: the line's alphabet decides (lookalike_verdict).
    """
    if label != pred and label.translate(TO_CYR) == pred.translate(TO_CYR):
        return lookalike_verdict(label, pred)
    text = " ".join(vlm.split())
    if text == label:
        return "label_ok"
    if text == pred:
        return "pred_ok"
    if label.replace(" ", "") != pred.replace(" ", ""):
        return None
    vlm_layout = space_layout(text)
    if vlm_layout == space_layout(label):
        return "label_ok"
    if vlm_layout == space_layout(pred.strip(" ")) and pred == pred.strip(" "):
        return "pred_ok"
    return None


def clean_path(label_file):
    stem, ext = os.path.splitext(label_file)
    if stem.endswith("_w_h"):
        return stem[: -len("_w_h")] + "_clean_w_h" + ext
    return stem + "_clean" + ext


def clean_file(label_file, preds, verdicts):
    out_file = clean_path(label_file)
    if os.path.exists(out_file):
        sys.exit(f"{out_file} exists, not overwritten")
    stats = Counter()
    with open(label_file, encoding="utf-8") as src, open(out_file, "w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            parts = line.rstrip("\r\n").split("\t")
            path, label = parts[0], parts[1]
            pred = preds.get(path)
            if pred is None:
                stats["no prediction, kept"] += 1
            elif pred == label:
                stats["model == label, kept"] += 1
            else:
                verdict = verdicts.get(path)
                if verdict == "label_ok":
                    stats["VLM confirms label, kept"] += 1
                elif verdict == "pred_ok":
                    stats["VLM confirms pred, label := pred"] += 1
                    parts[1] = pred
                else:
                    stats["disputed, dropped"] += 1
                    continue
            if pred is not None:
                unified = unify_words(parts[1])
                if unified != parts[1]:
                    stats["  of the kept: a word unified to one alphabet"] += 1
                    parts[1] = unified
            dst.write("\t".join(parts) + "\n")
    return out_file, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", nargs="+", required=True, help="TSVs with path, pred (preds_all.tsv.gz)")
    parser.add_argument("--vlm", nargs="+", required=True, help="review tables with path, label, pred, vlm")
    parser.add_argument("--labels", nargs="+", required=True, help="label files to clean")
    args = parser.parse_args(argv)

    preds = {}
    for table in args.preds:
        preds.update(read_table(table, ("path", "pred")))
    verdicts = {}
    for table in args.vlm:
        for path, label, pred, vlm in read_table(table, ("path", "label", "pred", "vlm")):
            verdicts[path] = vlm_verdict(label, pred, vlm)
    print(f"{len(preds)} predictions, {len(verdicts)} VLM readings")
    for label_file in args.labels:
        out_file, stats = clean_file(label_file, preds, verdicts)
        kept = sum(n for k, n in stats.items() if not k.startswith(" ")) - stats["disputed, dropped"]
        print(f"{label_file} -> {out_file}: {kept} lines")
        for key, n in stats.most_common():
            print(f"  {n:8d}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
