"""Writes ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb (run from the repo root)."""
import json
import sys

cells = []


def md(src):
    cells.append(("markdown", src))


def code(src):
    cells.append(("code", src))


md(r"""# RU OCR: third opinion on the stroyinf disagreements with PaddleOCR-VL

`review.tsv` (lines that differ in characters) and `review_spaces.tsv` (lines that differ only in spaces) hold the
stroyinf lines where our model (`pred`) and the text-layer label (`label`) differ; both are checked in one run. Here a VLM, **PaddleOCR-VL** (0.9B, `OCR:` prompt), reads every crop on its own, and
each line gets a verdict: whom the VLM agrees with.

**Input**: the crops (`ru-ocr-stroyinf` / `ru-ocr-stroyinf-modern`) and the output of `ru_export_eval_predict`
(`review.tsv`, `review_spaces.tsv`; Add Input -> Notebook Output). To resume: also `OUT_DATASET`.
**Settings**: Accelerator **GPU T4 x2**, Internet on, Add-ons -> Secrets: `KAGGLE_API_TOKEN`. Save & Run All.

- One worker process per GPU (`vlm_worker.py`, vLLM; `ENGINE = "transformers"` as a fallback), each takes every
  second line. Lines go in a fixed random order (`SEED`), so the first `PILOT` lines are a random sample.
- Each worker appends to `vlm_part{k}.tsv` every `SAVE_EVERY` lines; all parts are uploaded to `OUT_DATASET`
  every 10 min (a growing dataset = the run is alive). Parts found under `/kaggle/input` are skipped.
- The worker logs are printed here every minute.

**Output**: `review_vlm.tsv` (both review tables + `vlm`, `vlm_score`, `verdict`), the verdict counts.
Verdicts: `label_ok` (VLM = label), `pred_ok` (VLM = our model), `disputed`. Only the VLM text is normalized
(whitespace runs -> one space, ends stripped); label and pred are compared as they are, spaces included
(the same rule as `ru_ocr/tools/clean_labels_vlm.py`).""")

code(r"""KAGGLE_USER = "vladislavershov"  # owner of the private Kaggle datasets the notebook writes to
REPO = "https://github.com/ErshovVE/PaddleOCR.git"  # cloned for the configs and ru_ocr/tools
BRANCH = "feat/ru-ppocrv6-small-training"
REVIEWS = ["review.tsv", "review_spaces.tsv"]  # tables under /kaggle/input, checked together
PILOT = None             # lines checked (a random sample); None = all lines of REVIEWS
ENGINE = "vllm"          # "vllm" or "transformers" (slower fallback, batch 1)
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
GPUS = 2
SAVE_EVERY = 500         # lines per worker between appends to vlm_part{k}.tsv
MAX_TOKENS = 256         # generation limit per line
SEED = 0
OUT_DATASET = f"{KAGGLE_USER}/ru-ocr-stroyinf-modern-vlm"  # parts uploaded here every 10 min; None: no upload
WORK = "/kaggle/working"
VLLM_PY = "/tmp/vv/bin/python"  # the venv with vllm, made by the install cell
import os, subprocess, sys, time
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
if OUT_DATASET:  # fail at once: without the secret nothing is uploaded and a long run leaves no trace until it ends
    from kaggle_secrets import UserSecretsClient
    try:
        UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    except Exception as exc:
        raise RuntimeError("attach the KAGGLE_API_TOKEN secret (Add-ons -> Secrets; Kaggle drops it when a "
                           "notebook file is imported) or set OUT_DATASET = None") from exc""")

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
sh(f"{PIP} rapidfuzz --upgrade kaggle uv")
# The Kaggle image (Python 3.13, torch 2.11.0+cu128, transformers 5.x) runs the transformers engine as it is.
# vllm goes into its own venv, VLLM_PY, and only the vllm workers run there, so the image is never touched.
# The image sets UV_SYSTEM_PYTHON=true (uv would install into the image): it is unset for every uv call.
# vllm 0.26.0+cu129 (PyPI builds are CUDA 13) brings torch 2.11.0+cu129 and torchcodec 0.17, a CUDA 13 build
# that fails on import (libnvrtc.so.13); vllm imports it on start, so 0.11.0+cu129 (the torch 2.11 pair) replaces it.
VLLM_WHEEL = ("https://github.com/vllm-project/vllm/releases/download/v0.26.0/"
              "vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl")
UV = "env -u UV_SYSTEM_PYTHON -u VIRTUAL_ENV UV_NO_CONFIG=1 uv"
CU129 = "https://download.pytorch.org/whl/cu129"


def check(code, py=sys.executable):
    res = subprocess.run([py, "-c", code], capture_output=True, text=True)
    print(res.stdout.strip(), res.stderr.strip()[-1500:] if res.returncode else "", flush=True)
    return res.returncode == 0


if ENGINE == "vllm":
    sh(f"rm -rf /tmp/vv && {UV} venv /tmp/vv --python 3.12 -q")
    sh(f"{UV} pip install --python {VLLM_PY} 'vllm @ {VLLM_WHEEL}' --extra-index-url {CU129} "
       "--index-strategy unsafe-best-match")
    sh(f"{UV} pip install --python {VLLM_PY} --no-deps --reinstall torchcodec==0.11.0 --index-url {CU129}")
    if not check("import torch, vllm; from vllm import LLM; print('venv: torch', torch.__version__, "
                 "'vllm', vllm.__version__, 'cuda', torch.cuda.is_available())", VLLM_PY):
        print("vllm does not import: the transformers engine is used", flush=True)
        ENGINE = "transformers"
assert check("import torch, torchcodec; print('image: torch', torch.__version__, 'torchcodec', torchcodec.__version__)")
sh(f"rm -rf {WORK}/PaddleOCR && git clone -q --depth 1 --branch {BRANCH} {REPO} {WORK}/PaddleOCR")
sys.path.insert(0, f"{WORK}/PaddleOCR/ru_ocr/tools")""")

md("## 1. Data\nThe review paths start with `stroyinf/<folder>/crops/...`: the folders with `crops/` are linked into `/tmp/ru_root/stroyinf`.")

code(r"""import csv, random

ROOT = "/tmp/ru_root"
os.makedirs(os.path.join(ROOT, "stroyinf"), exist_ok=True)
# crops packed as *.tarpack (plain tar, see ru_ocr/tools/pack_dataset.py --format tar) are unpacked into ROOT,
# their paths already start with stroyinf/<folder>/crops/
for top, dirs, files in os.walk("/kaggle/input"):
    for name in sorted(f for f in files if f.endswith(".tarpack")):
        start = time.time()
        subprocess.run(["tar", "-xf", os.path.join(top, name), "-C", ROOT], check=True)
        print(f"unpacked {name} in {time.time() - start:.0f} s", flush=True)
review_paths, parts = {}, []
for top, dirs, files in os.walk("/kaggle/input"):
    if "crops" in dirs:
        dirs.remove("crops")
        if os.path.basename(os.path.dirname(top)) == "stroyinf":
            link = os.path.join(ROOT, "stroyinf", os.path.basename(top))
            if not os.path.exists(link):
                os.symlink(top, link)
    for name in REVIEWS:
        if name in files:
            review_paths.setdefault(name, []).append(os.path.join(top, name))
    parts += [os.path.join(top, f) for f in files if f.startswith("vlm_part") and f.endswith(".tsv")]
assert all(len(review_paths.get(n, [])) == 1 for n in REVIEWS), f"need exactly one of each {REVIEWS}: {review_paths}"
review = []  # line indices run over the tables in REVIEWS order
for name in REVIEWS:
    with open(review_paths[name][0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    print(name, len(rows), "lines")
    review += [dict(r, review=name) for r in rows]
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
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 3:  # skips a line cut by an upload taken mid-append
                vlm[int(fields[0])] = (fields[1], float(fields[2]))
todo = [i for i in order if i not in vlm]
for name in os.listdir(WORK):  # parts of an earlier Run All in this session: the workers would skip their lines
    if name.startswith(("vlm_part", "vlm_worker")) and name.endswith((".tsv", ".log")):
        os.remove(os.path.join(WORK, name))
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
        uploader = CheckpointUploader(WORK, "/tmp/vlm_upload", OUT_DATASET, poll_s=600, settle_s=5,
                                      patterns=("vlm_part*.tsv", "vlm_earlier.tsv"), required="vlm_part").start()
        print("parts are uploaded to", OUT_DATASET)
    except Exception as exc:
        print("parts are NOT uploaded (no KAGGLE_API_TOKEN secret?):", exc)

def run_workers(engine):
    # one worker per GPU (vllm ones in the venv); their new log lines are printed every minute; True when all finish
    workers, logs = [], [f"{WORK}/vlm_worker{k}.log" for k in range(GPUS)]
    py = VLLM_PY if engine == "vllm" else sys.executable
    assert os.path.exists(py), f"{py} is missing: run the install cell (section 0) first"
    for k in range(GPUS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        workers.append(subprocess.Popen(
            [py, f"{WORK}/vlm_worker.py", f"{WORK}/vlm_todo.tsv", str(k), str(GPUS),
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

for k in range(GPUS):  # every line has a reading (the workers finished)
    path = f"{WORK}/vlm_part{k}.tsv"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) == 3:
                    vlm[int(fields[0])] = (fields[1], float(fields[2]))
missing = [i for i in order if i not in vlm]
assert not missing, f"{len(missing)} lines without a VLM text"
# review_vlm.tsv = the review tables + review, vlm, vlm_score, verdict: ru_ocr/tools/vlm_review.py, the same rule
# (clean_labels_vlm.vlm_verdict) as the label cleaning; PILOT lines only get a verdict when PILOT is None
if PILOT is None:
    parts = [f"{WORK}/{name}" for name in os.listdir(WORK) if name.startswith("vlm_") and name.endswith(".tsv")
             and (name.startswith("vlm_part") or name == "vlm_earlier.tsv")]
    sh(f"{shlex.quote(sys.executable)} {WORK}/PaddleOCR/ru_ocr/tools/vlm_review.py --reviews "
       + " ".join(review_paths[name][0] for name in REVIEWS) + " --parts " + " ".join(parts)
       + f" --out {WORK}/review_vlm.tsv")
    df = pd.read_csv(f"{WORK}/review_vlm.tsv", sep="\t", quoting=3, keep_default_na=False)
    print(pd.crosstab(df.verdict, df.review, margins=True).to_string())
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
