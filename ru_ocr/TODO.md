# TODO

[Русская версия](docs/ru/TODO.md)

## 1. Speed benchmark: Paddle, ONNX, OpenVINO on GPU and CPU

Done: `scripts/benchmark.py` (any machine) and `kaggle/ru_benchmark.ipynb` (export + the script on Kaggle, input
`ru-ocr-val5k-tar`); the full val set on a 4-core Kaggle CPU, results in `benchmarks/kaggle-cpu-4cores/`, README and
EXPERIMENT (section 9). Left: the same on a GPU session (`paddle gpu`).

- **Hardware first.** The output starts with what the numbers were measured on:
  - GPU: model, number of cards, memory, driver and CUDA version;
  - CPU: model, physical and logical cores, cores actually available to the process (affinity and cgroup quota:
    Kaggle gives fewer than the host has), RAM;
  - versions of paddlepaddle, paddlex, onnxruntime, openvino.
- **Data:** the val set (`all_val5k_clean_w_h.txt`, 4 241 lines), the ft4 model.
- **Backends:**
  - the native engine (Paddle inference) on GPU;
  - the native engine on CPU;
  - ONNX Runtime;
  - OpenVINO.
- **Per backend:**
  - batch 1 and a larger batch;
  - warm-up excluded;
  - lines/s, ms per line (mean, p50, p95), total time;
  - accuracy on the same lines, as a check that the backend reads the same.
- **Output:** a table file (`benchmark.tsv` + markdown) with the hardware header; the result goes into README / EXPERIMENT.

## 2. Triton Inference Server on ONNX and OpenVINO

- Model repository for the ONNX Runtime backend and for the OpenVINO backend:
  - `config.pbtxt` with dynamic batch and width (`[-1, 3, 48, -1]`);
  - dynamic batching.
- Pre-processing: resize to height 48 and normalize. Post-processing: CTC decode with `dict/ru_dict.txt`. Done in an
  ensemble with a python backend, or on the client — choose one and document it.
- A client example and a `docker run` command.
- Throughput with `perf_analyzer`, next to the numbers from item 1.

## 3. Pull requests to PaddlePaddle/PaddleOCR

The core changes of this branch (everything outside `ru_ocr/`), as separate PRs. Checked by cherry-picking each
group onto `main` (upstream `dab3fe3`): the independent ones apply cleanly, the rest apply in the order below.

| # | PR | Commits | Depends on |
|---|---|---|---|
| 1 | **CTC length in training** (`Backbone.train_seq_len`): rec backbones pool to 40 time steps in training, and Paddle silently zeroes the CTC loss of longer labels; width-proportional pooling fixes long lines (PPLCNetV3/V4, PPHGNetV2, PPHGNet_small) | `8ad7cba`, `c79bc9b` | — |
| 2 | `RecMetric.length_buckets`: accuracy and edit distance by label length | `216f302` | — |
| 3 | perf: vectorised `WarpMLS` (TIA distort / stretch / perspective), the bottleneck of data loading | `3ad4051`, `04ff9ca` | — |
| 4 | `MultiScaleSampler`: mixed sorted / shuffled batches (`sorted_batch_ratio`), `pad_to_longest`, sampler for Eval; fix: `ratio_list` sampling in `MultiScaleDataSet`; fix: `RecResizeImg` in `inference.yml` when Eval uses a sampler | `c31cc69`, `cf29bd4`, `12a40b6` | — |
| 5 | Augmentations: `RecConAug` with spaces and batch-width fit, capped TIA vertical shift, extra samples read only when they fit; document-scan augmentations in `RecAug` (rotate, downscale, JPEG, stroke); table rules next to the text | `303eefb`, `6bfc567`, `252e063`, `999f90f` | 4 |
| 6 | Training: `Global.save_batch_step` and resume inside an epoch; `Global.save_on_signal` saves `latest` when stopped | `71ebae1`, `74b40fe` (without their `ru_ocr/` files) | 4 |
| 7 | Docs: training on long text lines (`text_recognition.md` / `.en.md`) | `dd1d7db`, `c58988b` | 1, 4, 5 (rewrite for what is merged) |

- **PR 1 is ready:** branch `feat/rec-train-seq-len` in the fork, from upstream `main`; tests and black/flake8 pass;
  the title, text and checks are in [`upstream_prs/01-rec-train-seq-len.md`](upstream_prs/01-rec-train-seq-len.md).
  Opened as PaddlePaddle/PaddleOCR#18420.
- **PR 2 and PR 3** (branches `feat/rec-metric-length-buckets`, `perf/warp-mls-vectorise` + a black commit `d2fa563`;
  texts and checks in [`upstream_prs/`](upstream_prs/)) opened as PaddlePaddle/PaddleOCR#18421 and #18422.
  CLA signed. All three wait for a maintainer to approve the CI run and for a review.
- `fea334d` (`se_gate` for RepSVTR) is only for porting our old PyTorch weights: not proposed.
- `71ebae1` and `74b40fe` also change `ru_ocr/` (smoke test, notebook): split those parts off.
- Before opening:
  - rebase onto the current upstream `main` (this clone is from 2026-10-01);
  - run their pre-commit and tests;
  - sign the CLA;
  - write the PR texts in English with the numbers from EXPERIMENT (long lines, data loading speed).

## 4. A separate repository for ru_ocr

The fork holds the whole PaddleOCR library, and the training work drowns in it. Move `ru_ocr/` (configs,
dictionaries, scripts, tools, notebooks, tests, docs) to its own repository without the library, and remove
`feat/ru-ppocrv6-small-training` from the public fork.

- **PaddleOCR from a pinned release plus a patch.**
  - The repository keeps `patches/paddleocr-v3.7.0.patch`: our core changes, `git diff` of everything outside
    `ru_ocr/`.
  - `scripts/setup_env.sh` clones the `v3.7.0` tag and runs `git apply`.
  - A patch, not copies of whole files. If a new release changes the same places, `git apply` fails loudly instead
    of silently overwriting upstream fixes.
  - When a PR of item 3 is merged and released, its part leaves the patch.
- **Checked on 2026-10-26:**
  - the 16 core files we change are identical in `v3.7.0` and in `main` (`dab3fe3`, the base of the branch); only
    the two `text_recognition` docs differ;
  - the whole core diff (25 files with tests and docs, ~4 000 lines) applies to `v3.7.0` cleanly.
- **What depends on the PaddleOCR tree and has to change:**
  - `scripts/run_train.py` and `benchmark.py` and the notebooks call `tools/train.py`, `tools/eval.py`,
    `tools/export_model.py`;
  - `tools/convert_torchocr_rec.py` and `validate_models.py` import `ppocr`;
  - `tests/conftest.py` puts the repository root on `sys.path`;
  - the notebooks clone `REPO` / `BRANCH` and use `ru_ocr/...` paths.

  Probably: the new repository next to a patched PaddleOCR checkout, its path in one variable.
- **History:** keep it with `git filter-repo --subdirectory-filter ru_ocr`, or start clean. To decide.
- **Before removing the branch from the fork:** none of #18420–#18422 links to it (checked); the `upstream_prs/`
  texts do (the EXPERIMENT link of PR 1): update them.
