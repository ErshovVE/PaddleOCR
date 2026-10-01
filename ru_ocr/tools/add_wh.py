"""Add image width/height to recognition label files and split train/val.

MultiScaleDataSet with ds_width: true needs 'path<TAB>text<TAB>w<TAB>h' lines; crop
generators usually write 'path<TAB>text'. Sizes are read from the image headers.

    python ru_ocr/tools/add_wh.py --label-file data/crops/good.txt --data-dir data/crops \
        --out-prefix data/crops/stroyinf --val-ratio 0.1

writes <prefix>_train_w_h.txt and <prefix>_val_w_h.txt. The validation part is the
tail of the file: crops are written document by document, so the tail holds whole
documents that the model never sees in training.
"""

import argparse
import os
import sys

from PIL import Image


def read_pairs(label_file, delimiter="\t"):
    pairs = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(delimiter)
            if len(parts) >= 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


def with_sizes(pairs, data_dir):
    """[(path, text)] -> [(path, text, w, h)], skipping unreadable images."""
    rows, missing = [], []
    for path, text in pairs:
        try:
            with Image.open(os.path.join(data_dir, path)) as img:
                w, h = img.size
        except (OSError, ValueError):
            missing.append(path)
            continue
        rows.append((path, text, w, h))
    return rows, missing


def split_tail(rows, val_ratio):
    n_val = int(round(len(rows) * val_ratio))
    if n_val == 0:
        return rows, []
    return rows[:-n_val], rows[-n_val:]


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p, text, w, h in rows:
            f.write("{}\t{}\t{}\t{}\n".format(p, text, w, h))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label-file", required=True)
    parser.add_argument(
        "--data-dir", required=True, help="image paths are relative to it"
    )
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)
    assert 0.0 <= args.val_ratio < 1.0, "--val-ratio must be in [0, 1)"

    rows, missing = with_sizes(read_pairs(args.label_file), args.data_dir)
    train, val = split_tail(rows, args.val_ratio)
    write_rows(args.out_prefix + "_train_w_h.txt", train)
    write_rows(args.out_prefix + "_val_w_h.txt", val)
    print(
        "train {} / val {} lines, unreadable images skipped: {}".format(
            len(train), len(val), len(missing)
        )
    )
    for path in missing[:10]:
        print("  missing: " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
