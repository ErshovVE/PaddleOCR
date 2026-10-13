"""Writes ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb (run from the repo root)."""
import json
import sys

cells = []


def md(src):
    cells.append(("markdown", src))


def code(src):
    cells.append(("code", src))


md(r"""# RU OCR: third opinion on the stroyinf disagreements with PaddleOCR-VL

`review.tsv` holds the stroyinf v2 lines where our model (`pred`) and the text-layer label (`label`) differ in
characters (111 980 lines). Here a VLM, **PaddleOCR-VL** (0.9B, `OCR:` prompt), reads every crop on its own, and
each line gets a verdict: whom the VLM agrees with.

**Input**: `ru-ocr-stroyinf` (crops), `ru-ocr-stroyinf-review` (`review.tsv`). To resume: also `OUT_DATASET`.
**Settings**: Accelerator **GPU T4 x2**, Internet on, Add-ons -> Secrets: `KAGGLE_API_TOKEN`. Save & Run All.

- One worker process per GPU (`vlm_worker.py`, vLLM; `ENGINE = "transformers"` as a fallback), each takes every
  second line. Lines go in a fixed random order (`SEED`), so the first `PILOT` lines are a random sample.
- Each worker appends to `vlm_part{k}.tsv` every `SAVE_EVERY` lines; all parts are uploaded to `OUT_DATASET`
  every 10 min (a growing dataset = the run is alive). Parts found under `/kaggle/input` are skipped.
- The worker logs are printed here every minute.

**Output**: `review_vlm.tsv` (`review.tsv` + `vlm`, `vlm_score`, `verdict`), the verdict counts.
Verdicts (texts compared without spaces, Latin look-alikes folded to Cyrillic): `label_ok` (VLM = label),
`pred_ok` (VLM = our model), `label_closer` / `pred_closer` (VLM nearer by >= 2 edits), `unclear`.""")

code(r"""PILOT = 1000             # lines checked (a random sample); None = all of review.tsv
ENGINE = "vllm"          # "vllm" or "transformers" (slower fallback, batch 1)
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
GPUS = 2
SAVE_EVERY = 500         # lines per worker between appends to vlm_part{k}.tsv
MAX_TOKENS = 256         # generation limit per line
SEED = 0
OUT_DATASET = "vladislavershov/ru-ocr-stroyinf-vlm"  # parts uploaded here every 10 min; None: no upload
WORK = "/kaggle/working"
import os, subprocess, sys, time
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)""")

md("## 0. Packages, code")

code(r"""import shlex, signal, threading


def sh(cmd, timeout=1800, attempts=3):
    # a shell command with its output in the cell; a run that hangs is killed with its process group
    # after `timeout` seconds and retried (a hung `!pip` used to stall the cell)
    for attempt in range(1, attempts + 1):
        print(time.strftime("%H:%M:%S"), f"[{attempt}/{attempts}]", cmd, flush=True)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        timer = threading.Timer(timeout, os.killpg, (proc.pid, signal.SIGKILL))
        timer.start()
        for line in proc.stdout:
            print(line, end="", flush=True)
        hung = not timer.is_alive()
        timer.cancel()
        if proc.wait() == 0:
            return
        print(time.strftime("%H:%M:%S"), f"hung for {timeout} s, killed" if hung else f"exit code {proc.returncode}", flush=True)
    raise RuntimeError(f"failed {attempts} times: {cmd}")


PIP = f"{shlex.quote(sys.executable)} -m pip install --progress-bar off --timeout 120 --retries 10"
sh(f"rm -rf {WORK}/PaddleOCR && git clone -q --depth 1 --branch feat/ru-ppocrv6-small-training "
   f"https://github.com/ErshovVE/PaddleOCR.git {WORK}/PaddleOCR")
sys.path.insert(0, f"{WORK}/PaddleOCR/ru_ocr/tools")
# The Kaggle image has torch 2.11.0+cu128 and transformers 5.x: the transformers engine needs no install.
# vllm: the CUDA 12 wheel of 0.26.0 (PyPI builds are CUDA 13) with --no-deps, then its dependencies from
# ru_ocr/kaggle/requirements_vlm.txt under constraints = the image's torch family, so pip cannot replace them
# (vllm 0.27+ swaps torch for 2.13/CUDA 13; torchcodec>=0.14 is a CUDA 13 build that fails on import).
VLLM_WHEEL = ("https://github.com/vllm-project/vllm/releases/download/v0.26.0/"
              "vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl")
from importlib.metadata import version
IMAGE = {p: version(p) for p in ("torch", "torchvision", "torchaudio", "torchcodec")}
with open("/tmp/image_constraints.txt", "w") as f:
    f.writelines(f"{p}=={v}\n" for p, v in IMAGE.items())
print("image:", IMAGE)
if ENGINE == "vllm":
    sh(f"{PIP} --no-deps 'vllm @ {VLLM_WHEEL}'")
    sh(f"{PIP} -r {WORK}/PaddleOCR/ru_ocr/kaggle/requirements_vlm.txt -c /tmp/image_constraints.txt")
sh(f"{PIP} rapidfuzz --upgrade kaggle")


def check(code):
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(res.stdout.strip(), res.stderr.strip()[-1500:] if res.returncode else "", flush=True)
    return res.returncode == 0


assert check("import torch, torchcodec; print('torch', torch.__version__, 'torchcodec', torchcodec.__version__)")
assert version("torch") == TORCH or TORCH.startswith(version("torch")), "the install replaced torch"
if ENGINE == "vllm" and not check("from vllm import LLM; print('vllm imports')"):
    print("vllm does not import: the transformers engine is used", flush=True)
    ENGINE = "transformers"
sh(f"rm -rf {WORK}/PaddleOCR && git clone -q --depth 1 --branch feat/ru-ppocrv6-small-training "
   f"https://github.com/ErshovVE/PaddleOCR.git {WORK}/PaddleOCR")
sys.path.insert(0, f"{WORK}/PaddleOCR/ru_ocr/tools")""")

md("## 1. Data\nThe `review.tsv` paths start with `stroyinf/<folder>/crops/...`: the folders with `crops/` are linked into `/tmp/ru_root/stroyinf`.")

code(r"""import csv, random

ROOT = "/tmp/ru_root"
os.makedirs(os.path.join(ROOT, "stroyinf"), exist_ok=True)
review_path, parts = None, []
for top, dirs, files in os.walk("/kaggle/input"):
    if "crops" in dirs:
        dirs.remove("crops")
        if os.path.basename(os.path.dirname(top)) == "stroyinf":
            link = os.path.join(ROOT, "stroyinf", os.path.basename(top))
            if not os.path.exists(link):
                os.symlink(top, link)
    if "review.tsv" in files:
        review_path = os.path.join(top, "review.tsv")
    parts += [os.path.join(top, f) for f in files if f.startswith("vlm_part") and f.endswith(".tsv")]
assert review_path, "attach ru-ocr-stroyinf-review"
with open(review_path, encoding="utf-8") as f:
    review = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
missing = [r["path"] for r in review[:200] if not os.path.exists(os.path.join(ROOT, r["path"]))]
assert not missing, f"crops not found, e.g. {missing[:3]}"
print("stroyinf folders:", len(os.listdir(os.path.join(ROOT, "stroyinf"))), "review lines:", len(review))
print("earlier parts:", parts)

order = list(range(len(review)))
random.Random(SEED).shuffle(order)
if PILOT:
    order = order[:PILOT]
vlm = {}
for path in parts:
    with open(path, encoding="utf-8") as f:
        for line in f:
            i, text, score = line.rstrip("\n").split("\t")
            vlm[int(i)] = (text, float(score))
todo = [i for i in order if i not in vlm]
print(len(order), "lines to check,", len(order) - len(todo), "done earlier,", len(todo), "to do")
with open(f"{WORK}/vlm_earlier.tsv", "w", encoding="utf-8") as f:  # keeps the earlier lines in this output
    f.writelines(f"{i}\t{t}\t{s}\n" for i, (t, s) in vlm.items())
with open(f"{WORK}/vlm_todo.tsv", "w", encoding="utf-8") as f:
    f.writelines(f"{i}\t{os.path.join(ROOT, review[i]['path'])}\n" for i in todo)""")

md("## 2. Worker\nOne process per GPU: reads `vlm_todo.tsv`, takes every `nshards`-th line from `shard`, appends `index, text, score` to its part. `score` = exp(mean token log-prob).")

code(r'''%%writefile /kaggle/working/vlm_worker.py
import math
import os
import sys
import time

from PIL import Image

todo_file, shard, nshards, out_file, engine, model_id, save_every, max_tokens = sys.argv[1:9]
shard, nshards, save_every, max_tokens = int(shard), int(nshards), int(save_every), int(max_tokens)
with open(todo_file, encoding="utf-8") as f:
    todo = [line.rstrip("\n").split("\t") for line in f][shard::nshards]
if os.path.exists(out_file):  # a restart (another engine) goes on after the lines already written
    with open(out_file, encoding="utf-8") as f:
        done = {line.split("\t", 1)[0] for line in f}
    todo = [t for t in todo if t[0] not in done]
print(f"shard {shard}: {len(todo)} lines, engine {engine}, model {model_id}", flush=True)


def clean(text):
    return " ".join(text.replace("\t", " ").split())


if engine == "vllm":
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, trust_remote_code=True, dtype="half", max_model_len=4096,
              max_num_batched_tokens=16384, enable_prefix_caching=False, mm_processor_cache_gb=0,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1})
    params = SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=0)

    def recognize(paths):
        chats = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": Image.open(p).convert("RGB")},
                                               {"type": "text", "text": "OCR:"}]}] for p in paths]
        res = []
        for out in llm.chat(chats, params, use_tqdm=False):
            o = out.outputs[0]
            n = len(o.token_ids)
            score = math.exp(o.cumulative_logprob / n) if n and o.cumulative_logprob is not None else float("nan")
            res.append((clean(o.text), score))
        return res
else:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def recognize(paths):
        res = []
        for p in paths:
            messages = [{"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")},
                                                     {"type": "text", "text": "OCR:"}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
            new = out.sequences[0][inputs["input_ids"].shape[-1]:]
            logp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            score = math.exp(logp.float().mean().item()) if len(logp) else float("nan")
            res.append((clean(processor.decode(new, skip_special_tokens=True)), score))
        return res

start = time.time()
for k in range(0, len(todo), save_every):
    chunk = todo[k : k + save_every]
    res = recognize([p for _, p in chunk])
    with open(out_file, "a", encoding="utf-8") as f:
        f.writelines(f"{i}\t{t}\t{s:.4f}\n" for (i, _), (t, s) in zip(chunk, res))
    done = k + len(chunk)
    print(f"{time.strftime('%H:%M:%S')} shard {shard}: {done} / {len(todo)}, {done / (time.time() - start):.1f} lines/s", flush=True)
print(f"shard {shard} done", flush=True)''')

md("## 3. Run the workers")

code(r"""uploader = None
if OUT_DATASET:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
        from ckpt_uploader import CheckpointUploader
        uploader = CheckpointUploader(WORK, "/tmp/vlm_upload", OUT_DATASET, poll_s=600,
                                      patterns=("vlm_part*.tsv", "vlm_earlier.tsv"), required="vlm_part").start()
        print("parts are uploaded to", OUT_DATASET)
    except Exception as exc:
        print("parts are NOT uploaded (no KAGGLE_API_TOKEN secret?):", exc)

def run_workers(engine):
    # one worker per GPU; their new log lines are printed every minute; True when all of them finish
    workers, logs = [], [f"{WORK}/vlm_worker{k}.log" for k in range(GPUS)]
    for k in range(GPUS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        workers.append(subprocess.Popen(
            [sys.executable, f"{WORK}/vlm_worker.py", f"{WORK}/vlm_todo.tsv", str(k), str(GPUS),
             f"{WORK}/vlm_part{k}.tsv", engine, MODEL_ID, str(SAVE_EVERY), str(MAX_TOKENS)],
            stdout=open(logs[k], "a"), stderr=subprocess.STDOUT, env=env))
    seen = [0] * GPUS
    try:
        while True:
            time.sleep(60)
            for k, path in enumerate(logs):  # new lines, without the vLLM progress noise
                with open(path, errors="replace") as f:
                    lines = f.read().splitlines()
                for line in lines[seen[k]:]:
                    if any(key in line for key in ("shard", "Error", "error", "Traceback")):
                        print(f"[w{k}] {line[:300]}", flush=True)
                seen[k] = len(lines)
            codes = [w.poll() for w in workers]
            if all(c == 0 for c in codes):
                return True
            if any(c not in (None, 0) for c in codes):
                for k, path in enumerate(logs):
                    print(f"===== worker {k} ({engine}) log tail\n", open(path, errors="replace").read()[-3000:])
                print("worker exit codes", codes, flush=True)
                return False
    finally:
        for w in workers:
            if w.poll() is None:
                w.kill()


try:
    ok = run_workers(ENGINE)
    if not ok and ENGINE == "vllm":
        print("vLLM failed: the transformers engine (batch 1, slower) goes on with the lines left", flush=True)
        ok = run_workers("transformers")
finally:
    if uploader is not None:
        uploader.stop()
        print("uploads to", OUT_DATASET, ":", uploader.uploads)
assert ok, ("the workers failed, see the log tails above")""")

md("## 4. Verdicts")

code(r"""import pandas as pd
from rapidfuzz.distance import Levenshtein

for k in range(GPUS):
    path = f"{WORK}/vlm_part{k}.tsv"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                i, text, score = line.rstrip("\n").split("\t")
                vlm[int(i)] = (text, float(score))
missing = [i for i in order if i not in vlm]
assert not missing, f"{len(missing)} lines without a VLM text"

# Latin look-alikes -> Cyrillic, spaces dropped: the VLM and the text layer mix the two alphabets
FOLD = str.maketrans("ABCEHKMOPTXYaceopxy", "АВСЕНКМОРТХУасеорху")


def norm(s):
    return s.replace(" ", "").translate(FOLD)


def verdict(label, pred, text):
    lab, prd, txt = norm(label), norm(pred), norm(text)
    if txt == lab:
        return "label_ok"
    if txt == prd:
        return "pred_ok"
    dl, dp = Levenshtein.distance(txt, lab), Levenshtein.distance(txt, prd)
    if dl + 2 <= dp:
        return "label_closer"
    if dp + 2 <= dl:
        return "pred_closer"
    return "unclear"


df = pd.DataFrame([review[i] for i in order])
df["vlm"] = [vlm[i][0] for i in order]
df["vlm_score"] = [vlm[i][1] for i in order]
df["verdict"] = [verdict(l, p, t) for l, p, t in zip(df.label, df.pred, df.vlm)]
df.to_csv(f"{WORK}/review_vlm.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
print(df.verdict.value_counts().to_string())
print()
print(pd.crosstab(pd.cut(df.score.astype(float), [0, 0.5, 0.8, 0.9, 0.95, 1.0]), df.verdict))
pd.set_option("display.max_colwidth", 60)
df.sample(min(30, len(df)), random_state=0)[["label", "pred", "vlm", "verdict"]]""")

nb = {
    "cells": [],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
for kind, src in cells:
    lines = src.split("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    nb["cells"].append(cell)
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
