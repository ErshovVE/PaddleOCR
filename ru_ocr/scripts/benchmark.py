"""Speed of an exported ru_ocr recognizer: Paddle (GPU, CPU), ONNX Runtime and OpenVINO on CPU.

Reads the lines of a label file through PaddleX `create_model`, as a user would run the model, and writes
benchmark.md (hardware header + table), benchmark.tsv and hardware.json to --out:

    python ru_ocr/scripts/benchmark.py --labels data/ru_labels/all_val5k_clean_w_h.txt --root data/ru_root \
        --paddle-model output/model/inference --onnx-model output/model/onnx --out output/benchmark

- Backends: `paddle gpu` (one card, only when nvidia-smi sees one), `paddle cpu`, `onnxruntime cpu`,
  `openvino cpu` (ONNX Runtime with the OpenVINO execution provider: pip package onnxruntime-openvino). The ONNX
  ones need --onnx-model.
- Each backend x batch size runs in its own process. Images are decoded into memory first, so disk and decoding
  are not timed; the first --warmup predict calls are not timed either. Thread counts are the libraries' defaults;
  the hardware header says how many cores the process may use.
- Batches above 1 also run with the lines sorted by width/height (each batch is padded to its widest line), unless
  --no-sort. Texts are compared in file order: accuracy (exact, spaces count) and agreement with the first run.

Models: `tools/export_model.py` gives the Paddle inference model; `paddlex --paddle2onnx` converts an export made
with CPU paddle (an export made with paddlepaddle-gpu holds ops paddle2onnx does not support).
"""

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version

BACKENDS = ["paddle gpu", "paddle cpu", "onnxruntime cpu", "openvino cpu"]
VERSIONS = ["paddlepaddle-gpu", "paddlepaddle", "paddlex", "onnxruntime-openvino", "onnxruntime", "openvino", "opencv-python"]


def out(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except OSError:
        return ""
    return res.stdout.strip() if res.returncode == 0 else ""


def has_gpu():
    return shutil.which("nvidia-smi") is not None and bool(out("nvidia-smi -L"))


def cgroup_cpus():
    """CPU quota of the container in cores (cgroup v2 cpu.max or v1 cfs quota), None when unlimited or unknown."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def ram_gb():
    """(total, available) GB from /proc/meminfo; (None, None) elsewhere."""
    try:
        with open("/proc/meminfo") as f:
            info = dict(line.split(":", 1) for line in f.read().splitlines())
        return tuple(round(int(info[k].split()[0]) / 2**20, 1) for k in ("MemTotal", "MemAvailable"))
    except (OSError, KeyError, ValueError):
        return None, None


def hardware():
    gpu = has_gpu()
    lscpu = {}
    for line in out("lscpu").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            lscpu[key.strip()] = value.strip()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    quota = cgroup_cpus()
    total, available = ram_gb()
    return {
        "gpu": out("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").splitlines() if gpu else [],
        "gpu_driver": out("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split("\n")[0] if gpu else "",
        "cuda_driver": (re.findall(r"CUDA Version: ([\d.]+)", out("nvidia-smi")) or [""])[0] if gpu else "",
        "cpu_model": lscpu.get("Model name") or platform.processor(),
        "cpu_sockets": lscpu.get("Socket(s)", ""),
        "cpu_cores_per_socket": lscpu.get("Core(s) per socket", ""),
        "cpu_threads_per_core": lscpu.get("Thread(s) per core", ""),
        "cpu_logical_host": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_quota": quota,
        "cpu_available": min(affinity, quota) if quota else affinity,  # what this process may actually use
        "ram_total_gb": total,
        "ram_available_gb": available,
        "python": platform.python_version(),
        "versions": {name: v for name, v in ((n, _version(n)) for n in VERSIONS) if v},
    }


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def model_kwargs(backend, paddle_model, onnx_model):
    """(create_model kwargs, environment of the worker) of a backend."""
    no_gpu = {"CUDA_VISIBLE_DEVICES": ""}
    if backend == "paddle gpu":
        return dict(model_dir=paddle_model, device="gpu:0"), {"CUDA_VISIBLE_DEVICES": "0"}
    if backend == "paddle cpu":
        return dict(model_dir=paddle_model, device="cpu"), no_gpu
    providers = {"onnxruntime cpu": {"providers": ["CPUExecutionProvider"]},
                 "openvino cpu": {"providers": ["OpenVINOExecutionProvider"], "provider_options": {"device_type": "CPU"}}}
    return dict(model_dir=onnx_model, device="cpu", engine="onnxruntime", engine_config=providers[backend]), no_gpu


def plan(backends, batches, sort):
    """[(backend, batch, sorted)]: every batch size above 1 also sorted, unless sort is off."""
    return [(b, n, s) for b in backends for n in batches for s in ([False, True] if sort and n > 1 else [False])]


def batch_order(sizes, sort):
    """Order of the lines: file order, or by width/height when sort. sizes: [(height, width)]."""
    order = list(range(len(sizes)))
    if sort:
        order.sort(key=lambda i: sizes[i][1] / sizes[i][0])
    return order


def summarize(calls, texts, labels, load_s):
    """Result row of one run. calls: seconds per predict call; texts and labels in file order."""
    n = len(labels)
    total = sum(calls)
    ms = sorted(c * 1000 for c in calls)
    return {
        "lines": n,
        "load_s": round(load_s, 2),
        "total_s": round(total, 2),
        "lines_per_s": round(n / total, 1),
        "ms_per_line": round(1000 * total / n, 2),
        "ms_per_call_mean": round(statistics.mean(ms), 2),
        "ms_per_call_p50": round(ms[len(ms) // 2], 2),
        "ms_per_call_p95": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 2),
        "acc": round(sum(t == l for t, l in zip(texts, labels)) / n, 4),
        "acc_rstrip": round(sum(t.rstrip() == l for t, l in zip(texts, labels)) / n, 4),  # padding can add a trailing space
    }


def read_labels(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t")[:2] for line in f if line.strip()]
    return rows[:limit] if limit else rows


def run_one(spec):
    """One backend with one batch size (runs in its own process): {"result": row, "texts": [...] in file order}."""
    import cv2
    from paddlex import create_model

    rows = read_labels(spec["labels"], spec["limit"])
    images = [cv2.imread(os.path.join(spec["root"], path)) for path, _ in rows]  # decoded up front: not timed
    missing = [path for (path, _), image in zip(rows, images) if image is None]
    if missing:
        raise SystemExit(f"{len(missing)} crops not found under {spec['root']}, e.g. {missing[:3]}")
    start = time.perf_counter()
    model = create_model(model_name=spec["model_name"], **spec["kwargs"])
    load_s = time.perf_counter() - start
    batch = spec["batch"]
    order = batch_order([image.shape[:2] for image in images], spec["sorted"])
    chunks = [[images[i] for i in order[k : k + batch]] for k in range(0, len(order), batch)]
    for chunk in chunks[: spec["warmup"]]:
        list(model.predict(chunk, batch_size=batch))
    calls, texts = [], []
    for chunk in chunks:
        t = time.perf_counter()
        res = list(model.predict(chunk, batch_size=batch))
        calls.append(time.perf_counter() - t)
        texts += [r["rec_text"] for r in res]
    texts = [text for _, text in sorted(zip(order, texts))]  # back to file order
    return {"result": summarize(calls, texts, [label for _, label in rows], load_s), "texts": texts}


def header(hw, args, n_lines):
    gpus = "; ".join(hw["gpu"]) or "none"
    lines = [
        "# RU OCR speed benchmark",
        "",
        f"- GPU: {gpus}" + (f" (driver {hw['gpu_driver']}, CUDA {hw['cuda_driver']}); Paddle uses one card" if hw["gpu"] else ""),
        f"- CPU: {hw['cpu_model']}, {hw['cpu_sockets'] or '?'} socket(s) x {hw['cpu_cores_per_socket'] or '?'} cores x "
        f"{hw['cpu_threads_per_core'] or '?'} threads; {hw['cpu_logical_host']} logical on the host, "
        f"**{hw['cpu_available']} available to the process** (affinity {hw['cpu_affinity']}, quota {hw['cpu_quota']})",
        f"- RAM: {hw['ram_total_gb']} GB ({hw['ram_available_gb']} GB available)",
        f"- Python {hw['python']}; " + ", ".join(f"{k} {v}" for k, v in hw["versions"].items()),
        f"- Lines: {os.path.basename(args.labels)}, {n_lines}; images decoded in memory; {args.warmup} warm-up calls "
        "not timed; library default threads",
    ]
    if args.note:
        lines.append(f"- {args.note}")
    return lines + [""]


def table(rows):
    cols = list(rows[0])
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join(str(row[c]) for c in cols) + " |" for row in rows]


def run_all(args):
    gpu = has_gpu()
    backends = [b for b in args.backends if (gpu or b != "paddle gpu") and (args.onnx_model or "paddle" in b)]
    skipped = sorted(set(args.backends) - set(backends))
    if skipped:
        print("skipped:", ", ".join(skipped), "(no GPU / no --onnx-model)", flush=True)
    hw = hardware()
    for key, value in hw.items():
        print(f"{key:22} {value}", flush=True)
    n_lines = len(read_labels(args.labels, args.limit))
    os.makedirs(args.out, exist_ok=True)
    rows, texts = [], {}
    runs = plan(backends, args.batches, args.sort)
    for k, (backend, batch, sort) in enumerate(runs, 1):
        kwargs, env = model_kwargs(backend, args.paddle_model, args.onnx_model)
        spec = dict(labels=args.labels, root=args.root, limit=args.limit, model_name=args.model_name, kwargs=kwargs,
                    batch=batch, sorted=sort, warmup=args.warmup)
        tag = f"{backend.replace(' ', '_')}_b{batch}" + ("_sorted" if sort else "")
        out_json = os.path.join(args.out, f"run_{tag}.json")
        res = subprocess.run([sys.executable, os.path.abspath(__file__), "--run-one", json.dumps(spec), "--out", out_json],
                             capture_output=True, text=True, env=dict(os.environ, **env))
        if res.returncode != 0:
            raise SystemExit(f"{backend} batch {batch} failed:\n{(res.stdout + res.stderr)[-3000:]}")
        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)
        texts[(backend, batch, sort)] = data["texts"]
        rows.append({"backend": backend, "batch": batch, "sorted": sort, **data["result"]})
        r = data["result"]
        print(f"[{k}/{len(runs)}] {backend:16} batch {batch:3}{' sorted' if sort else '       '}: "
              f"{r['lines_per_s']:8.1f} lines/s, {r['ms_per_line']:7.2f} ms/line, acc {r['acc']:.4f}", flush=True)
    ref_key = runs[0]
    ref = texts[ref_key]
    same = "same_as_" + ref_key[0].replace(" ", "_") + f"_b{ref_key[1]}"
    for row in rows:
        own = texts[(row["backend"], row["batch"], row["sorted"])]
        row[same] = round(sum(a == b for a, b in zip(own, ref)) / len(ref), 4)
    with open(os.path.join(args.out, "hardware.json"), "w", encoding="utf-8") as f:
        json.dump(hw, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "benchmark.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(rows[0]) + "\n")
        f.writelines("\t".join(str(v) for v in row.values()) + "\n" for row in rows)
    report = "\n".join(header(hw, args, n_lines) + table(rows)) + "\n"
    with open(os.path.join(args.out, "benchmark.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print("\n" + report, flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", help="label file: path<TAB>text[<TAB>w<TAB>h]")
    parser.add_argument("--root", help="folder the label paths are relative to")
    parser.add_argument("--paddle-model", help="Paddle inference model dir (tools/export_model.py)")
    parser.add_argument("--onnx-model", help="ONNX model dir (paddlex --paddle2onnx); without it no ONNX backends")
    parser.add_argument("--model-name", default="PP-OCRv6_small_rec", help="PaddleX model name of the architecture")
    parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS, metavar="BACKEND",
                        help="any of: " + ", ".join(f'"{b}"' for b in BACKENDS))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--no-sort", dest="sort", action="store_false", help="no width-sorted runs for batches above 1")
    parser.add_argument("--warmup", type=int, default=3, help="predict calls before timing starts")
    parser.add_argument("--limit", type=int, default=0, help="only the first N lines (0 = all)")
    parser.add_argument("--note", default="", help="a line added to the header, e.g. which checkpoint")
    parser.add_argument("--out", default="output/benchmark", help="result folder (with --run-one: the JSON file)")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)  # internal: one run in its own process
    args = parser.parse_args(argv)
    if not args.run_one and not (args.labels and args.root and args.paddle_model):
        parser.error("--labels, --root and --paddle-model are required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_one:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(run_one(json.loads(args.run_one)), f, ensure_ascii=False)
        return 0
    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
