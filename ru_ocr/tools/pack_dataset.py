"""Pack the training data root into a few uncompressed archives for upload (Kaggle, Hugging Face).

The data root is the folder training reads as data_dir (e.g. data/ru_root with links
synth -> doc-generator/out/synth_2m, stroyinf -> ocr_markup/data). Only the crops that the
label files reference are packed, under the same relative paths, so the archives can be
unpacked side by side and used with the same label files.

    python ru_ocr/tools/pack_dataset.py --root data/ru_root \
        --labels data/ru_labels/synth_2m_train_w_h.txt data/ru_labels/stroyinf_all_train_w_h.txt \
                 data/ru_labels/stroyinf_all_val_w_h.txt \
        --group-depth 2 --format tar --out data/kaggle_full

--group-depth N puts every first-N-path-component group into its own archive
(synth/part1, synth/part2, ..., stroyinf/<folder>); groups smaller than --min-group-mb are
merged into one archive of their top folder (all stroyinf folders -> stroyinf.tarpack).
The label files go into labels.zip. WebP crops are already compressed: nothing is compressed again.

--format tar (recommended) writes plain uncompressed tar files named *.tarpack: Kaggle unpacks an uploaded
.zip back into single files, and a notebook then waits hours while a million crops are fetched one by one;
a .tarpack is not recognised as an archive, stays one file and is unpacked by the notebook (tar -xf) in a
minute. --format zip writes ZIP_STORED zips.

Archives packed earlier as zip are repacked without unpacking them to disk:

    python ru_ocr/tools/pack_dataset.py --from-zip data/kaggle_full/synth_part1.zip --out data/kaggle_full_tar
"""

import argparse
import os
import sys
import tarfile
import time
import zipfile
from collections import defaultdict


def read_paths(label_files):
    """Unique image paths (first column) in the order of the label files."""
    paths = {}
    for label_file in label_files:
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                path = line.split("\t", 1)[0].strip()
                if path:
                    paths.setdefault(path, None)
    return list(paths)


def group_paths(paths, depth):
    groups = defaultdict(list)
    for path in paths:
        parts = path.split("/")
        groups["/".join(parts[:depth])].append(path)
    return groups


def plan_archives(root, groups, min_group_bytes):
    """{archive name: [paths]}; small groups are merged by their top folder."""
    archives, small = {}, defaultdict(list)
    for key, paths in sorted(groups.items()):
        size = sum(os.path.getsize(os.path.join(root, p)) for p in paths)
        if size >= min_group_bytes:
            archives[key.replace("/", "_")] = paths
        else:
            small[key.split("/")[0]].extend(paths)
    for top, paths in small.items():
        archives[top] = paths
    return archives


def write_zip(target, root, paths):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in paths:
            z.write(os.path.join(root, path), path)


def write_tar(target, root, paths):
    with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as t:
        for path in paths:
            t.add(os.path.join(root, path), path, recursive=False)


WRITERS = {"zip": (".zip", write_zip), "tar": (".tarpack", write_tar)}


def zip_to_tarpack(source, target):
    """Copy the files of a zip into a plain tar entry by entry (names kept); returns the file count."""
    count = 0
    with zipfile.ZipFile(source) as z, tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as t:
        for info in z.infolist():
            if info.is_dir():
                continue
            member = tarfile.TarInfo(info.filename)
            member.size = info.file_size
            member.mtime = time.mktime(info.date_time + (0, 0, -1))
            member.mode = 0o644
            with z.open(info) as f:
                t.addfile(member, f)
            count += 1
    return count


def repack_zips(sources, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for source in sources:
        target = os.path.join(out_dir, os.path.splitext(os.path.basename(source))[0] + ".tarpack")
        count = zip_to_tarpack(source, target)
        print("{}: {} files, {:.2f} GB".format(target, count, os.path.getsize(target) / 1e9), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="data_dir the label paths are relative to")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--group-depth", type=int, default=2)
    parser.add_argument("--min-group-mb", type=float, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", choices=sorted(WRITERS), default="zip", help="zip, or tar written as *.tarpack")
    parser.add_argument("--from-zip", nargs="+", metavar="ZIP", help="repack these zips into *.tarpack instead")
    args = parser.parse_args(argv)

    if args.from_zip:
        repack_zips(args.from_zip, args.out)
        return 0
    if not (args.root and args.labels):
        parser.error("--root and --labels are required (or --from-zip)")
    paths = read_paths(args.labels)
    missing = [p for p in paths if not os.path.isfile(os.path.join(args.root, p))]
    if missing:
        parser.error("{} label paths do not exist under {}, e.g. {}".format(len(missing), args.root, missing[0]))
    archives = plan_archives(args.root, group_paths(paths, args.group_depth), args.min_group_mb * 1e6)
    os.makedirs(args.out, exist_ok=True)
    ext, write = WRITERS[args.format]
    for name, files in archives.items():
        target = os.path.join(args.out, name + ext)
        write(target, args.root, files)
        print("{}: {} files, {:.2f} GB".format(target, len(files), os.path.getsize(target) / 1e9), flush=True)
    with zipfile.ZipFile(os.path.join(args.out, "labels.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for label_file in args.labels:
            z.write(label_file, "labels/" + os.path.basename(label_file))
    print("labels.zip: {} files".format(len(args.labels)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
