"""Clean recognition label files with the model predictions and a VLM third opinion.

A line whose label differs from our model's prediction (`pred`, exact compare, spaces count) is kept only when
the VLM reading confirms one side exactly: VLM == label -> the label stays, VLM == pred -> the label becomes pred.
Every other disputed line is dropped. Only the VLM text is normalized (whitespace runs -> one space, ends
stripped); label and pred are compared as they are. Lines the model got right and lines without a prediction
(e.g. other datasets) are written unchanged.

    python ru_ocr/tools/clean_labels_vlm.py --preds output/stroyinf_preds/preds_all.tsv.gz \
        --vlm output/stroyinf_preds/review_vlm.tsv \
        --labels data/ru_labels/all_train_w_h.txt data/ru_labels/all_val_w_h.txt

writes <label file>_clean next to each label file (all_train_w_h.txt -> all_train_clean_w_h.txt) and never
overwrites an existing file. --vlm takes several review tables (e.g. the spaces-only lines checked later);
their columns: path, label, pred, vlm.
"""

import argparse
import csv
import gzip
import os
import sys
from collections import Counter


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_table(path, columns):
    """Rows of a TSV with a header as tuples of `columns` (no quoting: texts hold quotes)."""
    with open_text(path) as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [tuple(row[c] for c in columns) for row in reader]


def vlm_verdict(label, pred, vlm):
    """'label_ok', 'pred_ok' or None; only the VLM text is normalized."""
    text = " ".join(vlm.split())
    if text == label:
        return "label_ok"
    if text == pred:
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
                    stats["VLM == label, kept"] += 1
                elif verdict == "pred_ok":
                    stats["VLM == pred, label := pred"] += 1
                    parts[1] = pred
                else:
                    stats["disputed, dropped"] += 1
                    continue
            dst.write("\t".join(parts) + "\n")
    return out_file, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", required=True, help="TSV with path, pred (preds_all.tsv.gz)")
    parser.add_argument("--vlm", nargs="+", required=True, help="review tables with path, label, pred, vlm")
    parser.add_argument("--labels", nargs="+", required=True, help="label files to clean")
    args = parser.parse_args()

    preds = dict(read_table(args.preds, ("path", "pred")))
    verdicts = {}
    for table in args.vlm:
        for path, label, pred, vlm in read_table(table, ("path", "label", "pred", "vlm")):
            verdicts[path] = vlm_verdict(label, pred, vlm)
    print(f"{len(preds)} predictions, {len(verdicts)} VLM readings")
    for label_file in args.labels:
        out_file, stats = clean_file(label_file, preds, verdicts)
        print(f"{label_file} -> {out_file}: {sum(stats.values()) - stats['disputed, dropped']} lines")
        for key, n in stats.most_common():
            print(f"  {n:8d}  {key}")


if __name__ == "__main__":
    main()
