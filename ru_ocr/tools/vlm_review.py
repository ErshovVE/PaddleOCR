"""Attach the VLM readings to the review tables and give every line a verdict.

The VLM notebook (ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb) reads the crops of the review tables
(review.tsv, review_spaces.tsv from build_review.py) and appends `index<TAB>text<TAB>score` lines to
vlm_part{k}.tsv, where index is the line number in the review tables read one after another, in the order
given here. This tool joins them back:

    python ru_ocr/tools/vlm_review.py --reviews review.tsv review_spaces.tsv \
        --parts vlm_part0.tsv vlm_part1.tsv --out review_vlm.tsv

review_vlm.tsv = the review columns + review (source table), vlm, vlm_score, verdict. The verdict is
clean_labels_vlm.vlm_verdict (label_ok, pred_ok, or disputed when it gives None), the same rule
clean_labels_vlm.py applies when it writes the cleaned label files.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_labels_vlm import vlm_verdict  # noqa: E402


def read_reviews(names):
    rows = []
    for name in names:
        with open(name, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for r in reader:
                if None in r or None in r.values():  # a tab inside a text, or a cut file: columns would shift
                    raise ValueError(f"{name}, line {reader.line_num}: expected {len(reader.fieldnames)} fields")
                rows.append(dict(r, review=os.path.basename(name)))
    return rows


def read_parts(names):
    """{index: (text, score)}; a line cut by an upload taken mid-append is skipped."""
    vlm = {}
    for name in names:
        with open(name, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) == 3:
                    vlm[int(fields[0])] = (fields[1], fields[2])
    return vlm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", nargs="+", required=True, help="review tables, in the order of the VLM run")
    parser.add_argument("--parts", nargs="+", required=True, help="vlm_part*.tsv (and vlm_earlier.tsv) of that run")
    parser.add_argument("--out", required=True, help="review_vlm.tsv")
    args = parser.parse_args(argv)

    rows = read_reviews(args.reviews)
    vlm = read_parts(args.parts)
    missing = [i for i in range(len(rows)) if i not in vlm]
    if missing:
        parser.error(f"{len(missing)} of {len(rows)} review lines have no VLM reading, e.g. line {missing[0]}")
    counts = {}
    columns = list(rows[0]) + ["vlm", "vlm_score", "verdict"] if rows else []
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for i, row in enumerate(rows):
            text, score = vlm[i]
            verdict = vlm_verdict(row["label"], row["pred"], text) or "disputed"
            counts[(row["review"], verdict)] = counts.get((row["review"], verdict), 0) + 1
            f.write("\t".join(list(row.values()) + [text, score, verdict]) + "\n")
    for (review, verdict), n in sorted(counts.items()):
        print(f"{review:20s} {verdict:10s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
