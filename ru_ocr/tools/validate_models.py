"""Validate exported recognition models on a labelled set, like real inference.

Every image is read at its own aspect ratio (batch size 1 by default), predictions go
to result_<model>.txt and metrics to metrics_<model>.csv: overall accuracy and
normalized edit distance, per label-length bucket and per group (first folder of the
image path).

    python ru_ocr/tools/validate_models.py --models output/inference/v6_small data/rec_150525_infer \
        --label-file ./train_big/all_real_clear_valid_w_h.txt --data-dir ./train_big --out results/

Needs the paddleocr package with its inference dependencies (paddlex).
"""

import argparse
import csv
import os
import sys

from rapidfuzz.distance import Levenshtein

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ppocr.metrics.rec_metric import (  # noqa: E402
    length_bucket_index,
    length_bucket_names,
)

DEFAULT_BUCKETS = [15, 40, 60]


def read_labels(label_file, delimiter="\t"):
    """Lines 'path<TAB>text[<TAB>w<TAB>h]' -> [(path, text)]."""
    records = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(delimiter)
            if len(parts) >= 2 and parts[0]:
                records.append((parts[0], parts[1]))
    return records


def group_of(path):
    """First folder of a relative image path, '.' for files at the top level."""
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "."


def _row(scope, name, stats):
    n = stats["n"]
    return {
        "scope": scope,
        "name": name,
        "n": n,
        "acc": stats["correct"] / n,
        "norm_edit_dis": 1.0 - stats["dist"] / n,
    }


def compute_metrics(samples, length_buckets=DEFAULT_BUCKETS):
    """samples: [(path, label, pred)] -> rows for overall, length buckets and groups.

    Exact match, no space or case normalisation (as RecMetric with ignore_space=False).
    Empty buckets and groups are omitted.
    """
    names = length_bucket_names(length_buckets)
    overall = {"n": 0, "correct": 0, "dist": 0.0}
    buckets = [{"n": 0, "correct": 0, "dist": 0.0} for _ in names]
    groups = {}
    for path, label, pred in samples:
        dist = Levenshtein.normalized_distance(pred, label)
        correct = int(pred == label)
        bucket = buckets[length_bucket_index(len(label), length_buckets)]
        group = groups.setdefault(group_of(path), {"n": 0, "correct": 0, "dist": 0.0})
        for stats in (overall, bucket, group):
            stats["n"] += 1
            stats["correct"] += correct
            stats["dist"] += dist
    if overall["n"] == 0:
        return []
    rows = [_row("all", "all", overall)]
    rows += [_row("length", n, s) for n, s in zip(names, buckets) if s["n"]]
    rows += [_row("group", g, groups[g]) for g in sorted(groups)]
    return rows


def write_outputs(out_dir, model_tag, samples, rows):
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result_{}.txt".format(model_tag))
    with open(result_path, "w", encoding="utf-8", newline="\n") as f:
        for path, _, pred in samples:
            f.write("{}\t{}\n".format(path, pred))
    metrics_path = os.path.join(out_dir, "metrics_{}.csv".format(model_tag))
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scope", "name", "n", "acc", "norm_edit_dis"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return result_path, metrics_path


def _rec_text(res):
    try:
        return res["rec_text"]
    except (KeyError, TypeError):
        return res.json["res"]["rec_text"]


def predict(model_dir, model_name, image_paths, batch_size=1, chunk=1000):
    from paddleocr import TextRecognition

    model = TextRecognition(model_name=model_name, model_dir=model_dir)
    preds = []
    for start in range(0, len(image_paths), chunk):
        part = image_paths[start : start + chunk]
        preds += [
            _rec_text(r) for r in model.predict(input=part, batch_size=batch_size)
        ]
        print("  {} / {}".format(len(preds), len(image_paths)))
    return preds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", required=True, help="inference dirs")
    parser.add_argument("--model-name", default="PP-OCRv6_small_rec")
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--buckets", type=int, nargs="+", default=DEFAULT_BUCKETS)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    records = read_labels(args.label_file)
    image_paths = [os.path.join(args.data_dir, path) for path, _ in records]
    for model_dir in args.models:
        tag = os.path.basename(os.path.normpath(model_dir))
        print("model {}: {} images".format(tag, len(records)))
        preds = predict(model_dir, args.model_name, image_paths, args.batch_size)
        if len(preds) != len(records):
            raise RuntimeError(
                "{}: {} predictions for {} images".format(tag, len(preds), len(records))
            )
        samples = [(p, label, pred) for (p, label), pred in zip(records, preds)]
        rows = compute_metrics(samples, args.buckets)
        _, metrics_path = write_outputs(args.out, tag, samples, rows)
        for row in rows:
            print(
                "  {scope:6} {name:12} n={n:<7} acc={acc:.4f} ned={norm_edit_dis:.4f}".format(
                    **row
                )
            )
        print("  -> {}".format(metrics_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
