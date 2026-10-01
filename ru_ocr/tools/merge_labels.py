"""Merge label files of several generator output folders into one train/val pair.

Every folder (ocr_markup / doc-generator output) has its own crops/ with the same
names (crops/0/image_00000.webp), so the merged lines get paths relative to a common
root: <folder relative to root>/crops/... -- then one Train.dataset.data_dir reads all.

    python ru_ocr/tools/merge_labels.py --root ../ocr_markup/data \
        --inputs ../ocr_markup/data/stroyinf_textlayer_v2 ../ocr_markup/data/stroyinf_textlayer_v2_part* \
        --val stroyinf_textlayer_v2_part1 --drop-vertical --drop-no-alnum --drop-filler \
        --min-cpu 0.8 --max-cpu 4 --out-prefix data/stroyinf/stroyinf_all

Lines must have 4 columns (path, text, w, h: append_crop_size in the generators).
Validation is made of whole folders: crops inside a folder go document by document,
so a tail split would cut a document in two.

Filters (all off by default) drop lines whose label most likely does not match the
crop -- typical for text layers of foreign OCR:
  --drop-vertical  rotated text on page margins (h > w, more than one character);
  --drop-no-alnum  no letters and no digits (stray "+", "—", ")", line fragments);
  --drop-filler    5+ equal punctuation marks in a row (dot leaders "....."): their
                   number in the label does not follow the picture;
  --min-cpu/--max-cpu  characters per unit of w/h outside the range: the crop is much
                   wider than the text (table rows with the number cut off or missing
                   in the crop) or much narrower (layer misaligned). Typical: ~1.9.
"""

import argparse
import os
import re
import sys
from pathlib import Path

_ALNUM = re.compile(r"[^\W_]")  # a letter or a digit of any script
_FILLER = re.compile(r"([^\w\s])\1{4,}")


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


def chars_per_unit(text, w, h):
    """Label characters per unit of the crop w/h ratio (text density of the crop)."""
    return len(text) * h / max(w, 1)


def drop_reason(text, w, h, filters):
    """Name of the first filter that rejects the line, or None."""
    if filters.get("vertical") and is_vertical(text, w, h):
        return "vertical"
    if filters.get("no_alnum") and not _ALNUM.search(text):
        return "no_alnum"
    if filters.get("filler") and _FILLER.search(text):
        return "filler"
    cpu = chars_per_unit(text, w, h)
    min_cpu, max_cpu = filters.get("min_cpu"), filters.get("max_cpu")
    if min_cpu is not None and cpu < min_cpu and len(text) > 3:
        return "too_wide"
    if max_cpu is not None and cpu > max_cpu:
        return "too_narrow"
    return None


def merge(root, inputs, val_names, label_name="good.txt", filters=None):
    """Returns (train_lines, val_lines, stats); lines are 'path\\ttext\\tw\\th'."""
    filters = filters or {}
    root = Path(root).resolve()
    train, val = [], []
    stats = {"folders": 0, "lines": 0, "bad": 0, "dropped": {}}
    for folder in inputs:
        folder = Path(folder).resolve()
        rel = folder.relative_to(root).as_posix()
        rows, bad = read_rows(folder / label_name)
        stats["folders"] += 1
        stats["bad"] += bad
        target = val if folder.name in val_names else train
        for path, text, w, h in rows:
            reason = drop_reason(text, w, h, filters)
            if reason:
                stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
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
    parser.add_argument("--drop-no-alnum", action="store_true")
    parser.add_argument("--drop-filler", action="store_true")
    parser.add_argument("--min-cpu", type=float, default=None)
    parser.add_argument("--max-cpu", type=float, default=None)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args(argv)

    names = {Path(p).name for p in args.inputs}
    missing = sorted(set(args.val) - names)
    if missing:
        parser.error("--val folders not among --inputs: {}".format(missing))
    for folder in args.inputs:
        if not (Path(folder) / args.label).is_file():
            parser.error("{} has no {}".format(folder, args.label))

    filters = {
        "vertical": args.drop_vertical,
        "no_alnum": args.drop_no_alnum,
        "filler": args.drop_filler,
        "min_cpu": args.min_cpu,
        "max_cpu": args.max_cpu,
    }
    train, val, stats = merge(
        args.root, args.inputs, set(args.val), args.label, filters
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    write_lines(args.out_prefix + "_train_w_h.txt", train)
    write_lines(args.out_prefix + "_val_w_h.txt", val)
    dropped = ", ".join(
        "{} {}".format(k, v) for k, v in sorted(stats["dropped"].items())
    )
    print(
        "folders {}: train {}, val {}; dropped: {}; bad lines {}".format(
            stats["folders"], len(train), len(val), dropped or "none", stats["bad"]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
