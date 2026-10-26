# ru_ocr — Russian + English text recognition on PaddleOCR 3.7

<p align="center">
  <strong>Language:</strong>
  <b>English</b> |
  <a href="docs/ru/README.md">🇷🇺 Русский</a>
</p>

Configs, dictionaries, tools and Kaggle notebooks for training a text line recognizer for Russian and English
**documents**, including long lines (60–95+ characters), on PP-OCRv6 small. It sits on top of the core changes of
this branch (all of them config keys, off by default) and is not meant for an upstream PR.

The experiment itself — task, data, label cleaning with a VLM, training runs and results — is described in
[EXPERIMENT.md](EXPERIMENT.md). Planned work: [TODO.md](TODO.md).

| Model | Validation, cleaned labels | Validation, raw labels |
|---|---|---|
| PP-OCRv6 small, ft4 | **0.960** line accuracy | 0.784 |

## How the current model was trained

### Steps

1. **Core fix for long lines.** PP-LCNet backbones trained CTC on 40 time steps only, and Paddle's CTC loss silently
   gives zero loss and gradient to longer labels. `Backbone.train_seq_len: -1` makes the train-time CTC length
   width / 8, as at inference; batches go up to 2 400 px wide; validation runs like inference (each line at its own
   width). Other core changes: concatenation with a space, TIA limited for long lines, scan augmentations, metrics by
   line length ([EXPERIMENT.md, section 3](EXPERIMENT.md#3-changes-to-paddleocr-core)).
2. **Data.** Synthetic document lines (`doc-generator`, clean and degraded pages) + real lines cut from PDF text
   layers of Russian standards (`ocr_markup`: stroyinf v2, Soviet scans; stroyinf modern, ≥ 1992).
3. **First run (v6 small)** from the official PP-OCRv6 small weights, new output layers for our dictionary.
4. **ft3**: fine-tuning from its weights with the modern standards and degraded synthetic lines added.
5. **Label cleaning**: ft3 reads every real line; lines where it disagrees with the text-layer label are read by
   PaddleOCR-VL; labels confirmed by the VLM are kept, labels the VLM reads as the model did are replaced, the rest are
   dropped ([EXPERIMENT.md, section 7](EXPERIMENT.md#7-label-cleaning)).
6. **ft4**: fine-tuning from ft3 `best_accuracy` on the cleaned labels — the current model.
7. **Export** to a Paddle inference model and ONNX, checked through Paddle / ONNX Runtime / OpenVINO.

### Training stages

All on Kaggle, 2× T4, fp16, 12 h sessions chained through checkpoints (`kaggle/ru_train_2xT4.ipynb`).

| Stage | Start from | Training data (lines) | LR schedule | Trained | Validation |
|---|---|---|---|---|---|
| v6 small | official PP-OCRv6 small pretrained | synthetic 1 985 902 + stroyinf v2 580 800 | cosine 0.0005, 1 warmup epoch | 88 000 steps (~1.6 epochs) | stroyinf v2 val5k: 0.706 |
| ft3 | weights of v6 small step 88 000 | + stroyinf modern 1 060 968 + degraded synthetic 497 805 = 4 125 475 | cosine 0.0005, no warmup, 3 epochs planned | 103 355 steps (~1.15 epochs), best at 100 000 | `all_val5k`: 0.774 |
| **ft4** | weights of ft3 `best_accuracy` (step 100 000) | cleaned: 3 995 865 | cosine 0.0005, no warmup, 2 epochs | ~173 600 steps (2 epochs, ~18 h) | `all_val5k_clean`: **0.960**, `all_val5k`: 0.784 |

Each stage starts from the previous weights only (`Global.pretrained_model`): the optimizer and the schedule start
anew. `best_accuracy` is chosen on the validation every 10 000 steps.

### Parameters (`configs/ru_PP-OCRv6_small_rec.yml` + the training notebook)

| | |
|---|---|
| Model | PP-OCRv6 small: PPLCNetV4 backbone, `train_seq_len: -1`; CTC head with LightSVTR neck (dims 120, depth 2); NRTR head (dim 384, training only) |
| Loss | CTC + NRTR (`MultiLoss`) |
| Dictionary | `dict/ru_dict_ext011026.txt`, 164 characters + space; `max_text_length` 250 |
| Optimizer | Adam (β 0.9 / 0.999), L2 3e-5, cosine LR |
| Batch | 32 per card, 2 cards; fp16 AMP (`scale_loss` 1024, dynamic) |
| Input | height 32 / 48 / 64 (multi-scale) in training, 48 in validation; width by the line's aspect ratio, batch up to 2 400 px |
| Sampler | `MultiScaleSampler`: 80 % of batches sorted by aspect ratio, 20 % shuffled; batch width = its widest line |
| Concatenation | `RecConAug` p 0.7: another line glued after a 16–80 px background gap with a space in the label, never wider than the batch or longer than its CTC steps (~6 % of samples end up concatenated) |
| Augmentations | TIA with vertical shift ≤ 0.25 of the line height; crop 0.1, noise 0.1, color jitter 0.1, blur 0.1, HSV 0.1, reverse 0.01; scan: rotation 0.2 (up to 3° for words, 0.6° at w/h 30), low DPI 0.15, JPEG 0.15, stroke width 0.1, table rules at the edges 0.1 |
| Validation | `all_val5k_clean`, every line at its own width, exact match, spaces count, accuracy by length 0–15 / 16–40 / 41–60 / 61+ |

## Contents

| Path | What |
|---|---|
| `configs/ru_PP-OCRv6_small_rec.yml` | main model: PP-OCRv6 small, CTC length during training = width / 8, batches up to 2 400 px, concatenation with a space, scan augmentations, validation like inference, metrics by length |
| `configs/ru_RepSVTR_rec_150525.yml` | baseline: RepSVTR with weights ported from a PyTorch model (`se_gate: none`), same data and metrics |
| `dict/ru_dict_ext011026.txt` | dictionary: 164 characters (the space is added by the code) |
| `dict/ru_dict_ext100124.txt` | dictionary of the old PyTorch model (162 characters), used by the converter |
| `tools/add_wh.py` | `path\ttext` → `path\ttext\tw\th` (needed by `MultiScaleDataSet` with `ds_width: true`) + train/val split |
| `tools/merge_labels.py` | merges the label files of several crop folders under one root, filters text-layer labels, val by whole folders |
| `tools/pack_dataset.py` | packs the crops the labels reference into a few archives (`--format tar` → `*.tarpack`), repacks zips (`--from-zip`) |
| `tools/build_review.py` | predictions + labels → `preds_all.tsv.gz`, `review.tsv`, `review_spaces.tsv` (disagreements) |
| `tools/vlm_review.py` | review tables + VLM readings → `review_vlm.tsv` with a verdict per line |
| `tools/clean_labels_vlm.py` | writes `*_clean` label files: keeps or replaces disputed labels by the VLM verdict, drops the rest |
| `tools/ckpt_uploader.py` | uploads checkpoints / result files to a Kaggle dataset while a run is going |
| `tools/validate_models.py` | checks exported models through `TextRecognition`: accuracy and edit distance by length and by group |
| `tools/convert_torchocr_rec.py` | ports torchocr weights to `.pdparams`, checks against ONNX, renumbers output layers for a new dictionary |
| `scripts/setup_env.sh` | training environment on a Linux GPU machine: paddlepaddle-gpu, requirements, pretrained weights |
| `scripts/run_train.py` | training without notebooks: native `tools/train.py` under `paddle.distributed.launch` with the run's settings, auto-resume |
| `kaggle/ru_train_2xT4.ipynb` | training on Kaggle 2× T4 in 12 h sessions with checkpoint upload and resume |
| `kaggle/ru_export_eval_predict.ipynb` | export, ONNX, backend check, test of checkpoints, predictions on real lines |
| `kaggle/ru_benchmark.ipynb` | speed of Paddle GPU / CPU, ONNX Runtime and OpenVINO on the val lines, with a hardware header |
| `kaggle/ru_vlm_check_stroyinf.ipynb` | VLM (PaddleOCR-VL via vLLM) reading of the disputed lines; generated by `kaggle/make_vlm_nb.py` |
| `tests/` | tests of the tools and configs, an end-to-end smoke training run (`resource_intensive`) |

## Installation

```bash
pip install paddlepaddle==3.3.1 -r requirements.txt          # or paddlepaddle-gpu for training
pip install "paddlex[ocr-core]>=3.7,<3.8"                      # export check, validate_models.py, inference
```

Pretrained PP-OCRv6 small weights go to `./pretrain/PP-OCRv6_small_rec_pretrained.pdparams`
(link in `docs/version3.x/module_usage/text_recognition.en.md`).

For training on a Linux machine with NVIDIA GPUs, one script installs paddlepaddle-gpu 3.3.1 (cu126, install retried),
the requirements and the pretrained weights, then checks Paddle on the GPUs (`PADDLE_PKG`, `PADDLE_INDEX`, `PYTHON`
override it):

```bash
bash ru_ocr/scripts/setup_env.sh
```

## Data

`MultiScaleDataSet` with `ds_width: true` reads 4 columns: `path<TAB>text<TAB>width<TAB>height`. The configs read
crops from `./data/ru_root/` and label files from `./data/ru_labels/`. How the data were made and why is described
in [EXPERIMENT.md, section 5](EXPERIMENT.md#5-data). The steps below rebuild the label files from scratch; the
repositories `doc-generator` and `ocr_markup` sit next to this one.

**1. Synthetic lines** (doc-generator, Docker: LibreOffice and the fonts are in the image). There are four parts of
500 k lines, each in its own folder with its own seed, plus part 5 with page degradation:

```bash
cd ../doc-generator
docker compose --profile balancer run -d balancer --config config.yaml --n-lines 500000 --append-crop-size --parallel 4 \
    --seed 1 --out out/synth_2m/part1            # part2..4: --seed 2000000 / 3000000 / 4000000
docker compose --profile balancer run -d balancer --config out/synth_2m/config_part5.yaml --n-lines 500000 \
    --append-crop-size --parallel 4 --seed 5000000 --out out/synth_2m/part5   # balancer.degrade.page_share: 1.0
```

- An interrupted run continues where it stopped.
- The exit code is 0 when the part is balanced; `balance_report.html` in the part folder shows it.
- Fonts are in `doc-generator/data/fonts`, with their licenses in `data/fonts/licenses/`.

**2. Real lines** (ocr_markup):

```bash
cd ../ocr_markup
python scripts/scrape/stroyinf.py --out data/stroyinf --limit 2250                       # v2: up to 1991
python scripts/scrape/stroyinf.py --out data/stroyinf_modern --min-year 1992 --max-year 2100 --limit 1000 \
    --group ГОСТ --group "ГОСТ Р" --group СП --group СНиП --group РД
python -m uvicorn backend.main:app --port 8000
```

Then post one job per folder of PDFs (v2 was processed in 15 portions, modern in 2), and follow it with
`GET /status/<job_id>`:

```bash
curl -X POST localhost:8000/run -H "Content-Type: application/json" -d '{
  "input_dir": "data/stroyinf_modern/pdf_part1", "output_dir": "data/stroyinf_modern_textlayer_part1",
  "extract_pdf_text_layer": true, "pdf_ocr_fallback": false, "append_crop_size": true,
  "normalize_labels": true, "alphabet_file": "../PaddleOCR/ru_ocr/dict/ru_dict_ext011026.txt"}'
```

Each output folder holds `crops/` and `good.txt`. Lines with characters outside the dictionary go to
`needs_review` and are not used.

**3. Data root and label files** (PaddleOCR root; on Windows use `mklink /J` instead of `ln -s`):

```bash
mkdir -p data/ru_root data/ru_labels
ln -s ../../../doc-generator/out/synth_2m data/ru_root/synth
ln -s ../../../ocr_markup/data data/ru_root/stroyinf
S=../doc-generator/out/synth_2m M=../ocr_markup/data
FILTERS="--drop-vertical --drop-no-alnum --drop-filler --min-cpu 0.8 --max-cpu 4"
python ru_ocr/tools/merge_labels.py --root $S --inputs $S/part1 $S/part2 $S/part3 $S/part4 --label dataset.tsv \
    --prefix synth --out-prefix data/ru_labels/synth_2m
python ru_ocr/tools/merge_labels.py --root $S --inputs $S/part5 --label dataset.tsv \
    --prefix synth --out-prefix data/ru_labels/synth_part5
python ru_ocr/tools/merge_labels.py --root $M --inputs $M/stroyinf_textlayer_v2 $M/stroyinf_textlayer_v2_part* \
    --val stroyinf_textlayer_v2_part1 $FILTERS --prefix stroyinf --out-prefix data/ru_labels/stroyinf_all
python ru_ocr/tools/merge_labels.py --root $M --inputs $M/stroyinf_modern_textlayer_part1 \
    $M/stroyinf_modern_textlayer_part2 $FILTERS --prefix stroyinf --out-prefix data/ru_labels/stroyinf_modern_all
cd data/ru_labels
n=$(wc -l < stroyinf_modern_all_train_w_h.txt); v=$((n / 50))          # modern: val = the last 2 %
head -n $((n - v)) stroyinf_modern_all_train_w_h.txt > stroyinf_modern_train_w_h.txt
tail -n $v stroyinf_modern_all_train_w_h.txt > stroyinf_modern_val_w_h.txt
cat synth_2m_train_w_h.txt stroyinf_all_train_w_h.txt stroyinf_modern_train_w_h.txt synth_part5_train_w_h.txt \
    > all_train_w_h.txt
cat stroyinf_all_val_w_h.txt stroyinf_modern_val_w_h.txt > all_val_w_h.txt
python -c "import random; v = open('all_val_w_h.txt', encoding='utf-8').read().splitlines(); \
open('all_val5k_w_h.txt', 'w', encoding='utf-8', newline='\n').write('\n'.join(random.Random(0).sample(v, 5000)) + '\n')"
```

Expected sizes:

| File | Lines |
|---|---|
| `synth_2m` train | 1 985 902 |
| `synth_part5` train | 497 805 |
| `stroyinf_all` train / val | 580 800 / 28 913 |
| `stroyinf_modern` train / val | 1 060 968 / 21 652 |
| `all_train` | 4 125 475 |
| `all_val` | 50 565 |

The `*_clean` files come from the label cleaning (see below).

- The filters (`--drop-*`, `--min-cpu/--max-cpu`) are for text layers of someone else's OCR. Do not apply them to
  synthetic lines, which are balanced already.
- When a new source is added, split only that source and append it to the existing `all_train` / `all_val`.
- If a crop generator writes only 2 columns, `add_wh.py` adds the width and height and splits off the tail as val
  (crops are written document by document, so val holds whole documents):

```bash
python ru_ocr/tools/add_wh.py --label-file <crops>/good.txt --data-dir <crops> --out-prefix data/mydata --val-ratio 0.1
```

**4. Archives for Kaggle.** `.tarpack` archives of ~1 GB are grouped by folder. Upload them as they are: Kaggle
unpacks a `.zip` into single files.

```bash
python ru_ocr/tools/pack_dataset.py --root data/ru_root --labels data/ru_labels/stroyinf_modern_train_w_h.txt \
    data/ru_labels/stroyinf_modern_val_w_h.txt --format tar --out data/kaggle_full/ru-ocr-stroyinf-modern-tar
python ru_ocr/tools/pack_dataset.py --from-zip <old>.zip --out <folder>     # repack an existing zip
```

## Training

```bash
python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c ru_ocr/configs/ru_PP-OCRv6_small_rec.yml \
    -o Train.dataset.data_dir=<crops> "Train.dataset.label_file_list=[data/mydata_train_w_h.txt]" \
       Eval.dataset.data_dir=<crops> "Eval.dataset.label_file_list=[data/mydata_val_w_h.txt]"
```

`scripts/run_train.py` starts the same `tools/train.py` with the settings of the published run (fp16 AMP, batch 32 per
card, `sorted_batch_ratio` 0.8, `latest` every 2000 steps, validation every 10000, save on SIGINT/SIGTERM) and picks the
start itself:

```bash
# a new stage from the weights of the previous one (new optimizer and LR schedule), as ft3 -> ft4
python ru_ocr/scripts/run_train.py --gpus 0,1 --out output/ft4 --init output/ft3/best_accuracy     --data-dir data/ru_root --train-labels data/ru_labels/all_train_clean_w_h.txt     --val-labels data/ru_labels/all_val5k_clean_w_h.txt
```

- Run the same command again after a stop (Ctrl+C saves `latest` with its place in the epoch): it resumes from
  `<out>/latest`. `--resume <prefix>` resumes another checkpoint, `--no-auto-resume` ignores `latest`.
- A checkpoint that has finished all epochs is refused instead of "resuming" with no step: use another `--out` with
  `--init`, or raise `--epochs`.
- Without `--init` a run starts from `Global.pretrained_model` of the config (the official weights in `./pretrain/`).
- `--epochs`, `--lr`, `--warmup-epochs`, `--batch`, `--workers`, `--save-steps`, `--eval-steps`, `--no-amp` change
  the usual settings; anything else goes through `-o` as in `tools/train.py`. `--dry-run` prints the command;
  logs of each card are in `<out>/dist_log/`.

Notes for both ways:

- The CTC and NRTR output layers are trained from scratch: the official v6 dictionary has no Cyrillic.
  From random weights: `-o Global.pretrained_model=`; from a checkpoint's weights only: `Global.pretrained_model=<prefix>`.
- Wide batches need memory: lower `Train.sampler.first_bs` if a card runs out.
- In the log: `acc_len_0_15 … acc_len_61_inf` — accuracy by label length;
  `RecConAug: …% of … samples concatenated` — share of concatenated samples.

### On Kaggle (2× T4)

`kaggle/ru_train_2xT4.ipynb` trains in 12 h sessions. Set `KAGGLE_USER`, `REPO`, `BRANCH` and the run parameters
in its first cell, attach the crop archives (`*.tarpack`, from `pack_dataset.py --format tar`; they are unpacked
into `/tmp/ru_root` in parallel), the label files and the initial weights, add the `KAGGLE_API_TOKEN` secret and
Save & Run All. Checkpoints are uploaded to `CKPT_DATASET` every `SAVE_STEPS`; the next session resumes from the
newest one when that dataset is attached. Upload crops as `.tarpack`, not `.zip`: Kaggle unpacks a zip into single
files and a session then fetches a million crops one by one for hours.

### Smoke test

Each notebook has a commented smoke setting in its first cell for a small dataset (a few hundred crops in one
`.tarpack` with label files and review tables of the same names): training — `RUN_NAME, CKPT_DATASET, INIT_DATASET =
"smoke", None, <weights dataset>` and a short `TIME_LIMIT`; export/eval — defaults, `RUN_PREDICTIONS = True` to cover
the predictions too; VLM — `OUT_DATASET = None`. With the upload datasets set to `None` no secret is needed and nothing
is written to the real datasets. Do not attach the full label datasets together with the small one: the notebooks find
their files by name.

## Cleaning the labels of real lines

The pipeline of [EXPERIMENT.md, section 7](EXPERIMENT.md#7-label-cleaning): a trained model reads every real
crop, the disagreements go to a VLM, and the verdicts decide which labels are kept, replaced or dropped.

```bash
# 1. predictions: kaggle/ru_export_eval_predict.ipynb with RUN_PREDICTIONS = True writes preds_partial.tsv,
#    then (also done by the notebook):
python ru_ocr/tools/build_review.py --preds preds_partial.tsv \
    --labels data/ru_labels/stroyinf_modern_train_w_h.txt data/ru_labels/stroyinf_modern_val_w_h.txt --out preds_modern
# 2. VLM: kaggle/ru_vlm_check_stroyinf.ipynb on review.tsv + review_spaces.tsv writes vlm_part*.tsv, then:
python ru_ocr/tools/vlm_review.py --reviews preds_modern/review.tsv preds_modern/review_spaces.tsv \
    --parts vlm_part0.tsv vlm_part1.tsv --out preds_modern/review_vlm.tsv
# 3. cleaned label files next to the originals (*_clean_w_h.txt, never overwritten)
python ru_ocr/tools/clean_labels_vlm.py --preds preds_v2/preds_all.tsv.gz preds_modern/preds_all.tsv.gz \
    --vlm preds_v2/review_vlm.tsv preds_modern/review_vlm.tsv \
    --labels data/ru_labels/all_train_w_h.txt data/ru_labels/all_val_w_h.txt
```

The VLM notebook installs vLLM into its own virtual environment, so the Kaggle image (its torch) stays untouched;
it is generated by `python ru_ocr/kaggle/make_vlm_nb.py ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb` — edit the
generator, not the notebook.

## Export and inference

```bash
python tools/export_model.py -c ru_ocr/configs/ru_PP-OCRv6_small_rec.yml \
    -o Global.pretrained_model=./output/ru_PP-OCRv6_small_rec/best_accuracy \
       Global.save_inference_dir=./output/ru_PP-OCRv6_small_rec/inference
paddlex --paddle2onnx --paddle_model_dir ./output/ru_PP-OCRv6_small_rec/inference \
    --onnx_model_dir ./output/ru_PP-OCRv6_small_rec/onnx              # export on CPU paddle
python ru_ocr/tools/validate_models.py --models ./output/ru_PP-OCRv6_small_rec/inference \
    --label-file data/mydata_val_w_h.txt --data-dir <crops> --out results/
```

```python
from paddlex import create_model

model = create_model(model_name="PP-OCRv6_small_rec", model_dir="onnx", device="cpu", engine="onnxruntime",
                     engine_config={"providers": ["CPUExecutionProvider"]})
for res in model.predict(["line.png"], batch_size=1):
    print(res["rec_text"].strip(), res["rec_score"])
```

- Convert to ONNX from a model exported with **CPU** Paddle: an export made with `paddlepaddle-gpu` contains ops
  `paddle2onnx` does not support.
- Strip the predictions: in batches padded to the widest line a few percent of them get a space at an end.

## Porting the old PyTorch model

torch and paddle do not work in one process on Windows, so the two steps run in different environments:

```bash
python ru_ocr/tools/convert_torchocr_rec.py extract --src model_20.pth --out weights.npz          # python with torch
python ru_ocr/tools/convert_torchocr_rec.py convert -c ru_ocr/configs/ru_RepSVTR_rec_150525.yml \
    --src-dict ru_ocr/dict/ru_dict_ext100124.txt --src weights.npz --out rec.pdparams --onnx rec.onnx  # python with paddle
```

`--src-dict` is the dictionary the torch model was trained with: the model is checked against ONNX on it, then the
output layers are renumbered for the config's dictionary (known characters keep their weights, new ones are learned
in fine-tuning). `extract` uses `torch.load(weights_only=False)` — only for trusted checkpoints.

## Tests

```bash
python -m pytest tests/ppocr ru_ocr/tests -q
TORCH_PYTHON=<python with torch> python -m pytest -m resource_intensive ru_ocr/tests -q   # ONNX check, smoke training
```
