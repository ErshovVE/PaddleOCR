"""Merge label files of several generator output folders into one train/val pair.

Every folder (ocr_markup / doc-generator output) has its own crops/ with the same
names (crops/0/image_00000.webp), so the merged lines get paths relative to a common
root: <folder relative to root>/crops/... -- then one Train.dataset.data_dir reads all.

    python ru_ocr/tools/merge_labels.py --root ../ocr_markup/data \
        --inputs ../ocr_markup/data/stroyinf_textlayer_v2 ../ocr_markup/data/stroyinf_textlayer_v2_part* \
        --val stroyinf_textlayer_v2_part1 --drop-vertical --out-prefix data/stroyinf/stroyinf_all

Lines must have 4 columns (path, text, w, h: append_crop_size in the generators).
Validation is made of whole folders: crops inside a folder go document by document,
so a tail split would cut a document in two.
"""

import argparse
import os
import sys
from pathlib import Path


def read_rows(label_file):
    """[(path, text, w, h)] from a 4-column label file; other lines are counted as bad."""
    rows, bad = [], 0
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if (
                len(parts) != 4
                or not parts[1]
                or not parts[2].isdigit()
                or not parts[3].isdigit()
            ):
                bad += 1 if line.strip() else 0
                continue
            rows.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    return rows, bad


def is_vertical(text, w, h):
    """Rotated text on page margins: taller than wide, more than one character."""
    return h > w and len(text) > 1


def merge(root, inputs, val_names, label_name="good.txt", drop_vertical=False):
    """Returns (train_lines, val_lines, stats); lines are 'path\\ttext\\tw\\th'."""
    root = Path(root).resolve()
    train, val = [], []
    stats = {"folders": 0, "lines": 0, "vertical": 0, "bad": 0}
    for folder in inputs:
        folder = Path(folder).resolve()
        rel = folder.relative_to(root).as_posix()
        rows, bad = read_rows(folder / label_name)
        stats["folders"] += 1
        stats["bad"] += bad
        target = val if folder.name in val_names else train
        for path, text, w, h in rows:
            if drop_vertical and is_vertical(text, w, h):
                stats["vertical"] += 1
                continue
            target.append("{}/{}\t{}\t{}\t{}".format(rel, path, text, w, h))
            stats["lines"] += 1
    return train, val, stats


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="common data_dir for training")
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="generator output folders"
    )
    parser.add_argument(
        "--val", nargs="*", default=[], help="folder names that go to validation"
    )
    parser.add_argument(
        "--label", default="good.txt", help="label file name inside each folder"
    )
    parser.add_argument("--drop-vertical", action="store_true")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args(argv)

    names = {Path(p).name for p in args.inputs}
    missing = sorted(set(args.val) - names)
    if missing:
        parser.error("--val folders not among --inputs: {}".format(missing))
    for folder in args.inputs:
        if not (Path(folder) / args.label).is_file():
            parser.error("{} has no {}".format(folder, args.label))

    train, val, stats = merge(
        args.root, args.inputs, set(args.val), args.label, args.drop_vertical
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    write_lines(args.out_prefix + "_train_w_h.txt", train)
    write_lines(args.out_prefix + "_val_w_h.txt", val)
    print(
        "folders {folders}: train {train}, val {val}; dropped vertical {vertical}, "
        "bad lines {bad}".format(train=len(train), val=len(val), **stats)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
