"""Join model predictions with the label files and split out the lines where they disagree.

The prediction notebook (ru_ocr/kaggle/ru_export_eval_predict.ipynb) writes preds_partial.tsv: one line per
predicted label line, `index<TAB>path<TAB>pred<TAB>score`, where index is the line number in the label files
read one after another. This tool writes three tables with a header next to each other:

- preds_all.tsv.gz  every predicted line: path, label, pred, score, dist, dist_ns
- review.tsv        lines whose label and prediction differ in characters (dist_ns > 0),
                    most edits first, then the most confident prediction first
- review_spaces.tsv lines that differ only in spaces (dist > 0, dist_ns == 0), same order by dist

dist is the edit distance between label and prediction, dist_ns the same with all spaces removed (a text
layer often splits words, "п р и я т и я"). The review tables are what the VLM notebook checks
(ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb).

    python ru_ocr/tools/build_review.py --preds preds_partial.tsv \
        --labels data/ru_labels/stroyinf_modern_train_w_h.txt data/ru_labels/stroyinf_modern_val_w_h.txt \
        --out output/stroyinf_modern_preds

The label files must be the same files, in the same order, as in the prediction run; a prediction whose path
does not match its label line is an error. Several --preds files (sessions of one run) are merged.
"""

import argparse
import gzip
import os
import sys

from rapidfuzz.distance import Levenshtein

COLUMNS = ["path", "label", "pred", "score", "dist", "dist_ns"]


def read_rows(label_files):
    """[(path, label)] of all non-empty lines of the label files, in order."""
    rows = []
    for name in label_files:
        with open(name, encoding="utf-8") as f:
            rows += [line.rstrip("\r\n").split("\t")[:2] for line in f if line.strip()]
    return rows


def read_preds(pred_files, rows):
    """{index: (pred, score)}; a line cut by an upload taken mid-append is skipped."""
    preds = {}
    for name in pred_files:
        with open(name, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 4:
                    continue
                index, path, pred, score = fields
                try:
                    float(score)
                except ValueError:  # cut right after the last tab
                    continue
                index = int(index)
                if index >= len(rows) or rows[index][0] != path:
                    raise ValueError(f"{name}: line {index} is {path!r}, the label files have "
                                     f"{rows[index][0] if index < len(rows) else 'no such line'!r}")
                preds[index] = (pred, score)
    return preds


def build(rows, preds):
    """(all, review, review_spaces) as lists of rows in COLUMNS order."""
    table = []
    for index, (path, label) in enumerate(rows):
        if index not in preds:
            continue
        pred, score = preds[index]
        dist = Levenshtein.distance(label, pred)
        dist_ns = Levenshtein.distance(label.replace(" ", ""), pred.replace(" ", ""))
        table.append([path, label, pred, score, dist, dist_ns])
    review = sorted((r for r in table if r[5] > 0), key=lambda r: (r[5], float(r[3])), reverse=True)
    spaces = sorted((r for r in table if r[4] > 0 and r[5] == 0), key=lambda r: (r[4], float(r[3])), reverse=True)
    return table, review, spaces


def write(path, table):
    opener = gzip.open if path.endswith(".gz") else open
    # plain TSV, no quoting: texts hold quotes, and labels never hold tabs or newlines
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        f.writelines("\t".join(map(str, row)) + "\n" for row in [COLUMNS] + table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", nargs="+", required=True, help="preds_partial.tsv files of one prediction run")
    parser.add_argument("--labels", nargs="+", required=True, help="the label files of that run, in its order")
    parser.add_argument("--out", required=True, help="folder for preds_all.tsv.gz, review.tsv, review_spaces.tsv")
    args = parser.parse_args(argv)

    rows = read_rows(args.labels)
    preds = read_preds(args.preds, rows)
    table, review, spaces = build(rows, preds)
    os.makedirs(args.out, exist_ok=True)
    write(os.path.join(args.out, "preds_all.tsv.gz"), table)
    write(os.path.join(args.out, "review.tsv"), review)
    write(os.path.join(args.out, "review_spaces.tsv"), spaces)
    exact = sum(r[4] == 0 for r in table)
    print(f"{len(rows)} label lines, {len(table)} predicted, exact {exact / max(len(table), 1):.4f}, "
          f"differ in characters {len(review)}, only in spaces {len(spaces)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
