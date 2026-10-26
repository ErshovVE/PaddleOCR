# Russian + English document text recognition with PP-OCRv6 small

<p align="center">
  <strong>Language:</strong>
  <b>English</b> |
  <a href="docs/ru/EXPERIMENT.md">🇷🇺 Русский</a>
</p>

This document describes the experiment behind `ru_ocr/`: what problem we solved, which data and model we used,
how the training labels were cleaned with a second model and a VLM, how training ran, and what came out of it.
How to run the code is in [README.md](README.md).

## 1. Task

Text line recognition for Russian and English **documents**: standards, codes of practice, contracts, forms,
scans and born-digital PDFs. A line image comes from a detector (or a PDF text layer) and the model returns its
text. Two properties of this domain drive every decision below:

- **Long lines.** About a third of the lines are whole lines of a page, 60–95 characters, with a tail up to ~200.
  The rest are single words and short fragments (1–15 characters).
- **Noisy real labels.** Large amounts of real lines are only available with labels from someone else's OCR
  (the text layer of scanned PDFs). These labels contain recognition errors, split words (`п р и я т и я`) and
  Latin look-alike letters inside Russian words.

The model must also be small enough for CPU inference (ONNX / OpenVINO).

## 2. Starting point and the main finding

Our previous production model was a RepSVTR (OpenOCR architecture) trained in a PyTorch port of the PaddleOCR
pipeline. Before starting new training we ported it to PaddleOCR 3.7 (`tools/convert_torchocr_rec.py`, the
outputs match the original ONNX to ~1e-5) and looked at why its training metrics (~0.5 line accuracy) were so far
from its test accuracy (~0.9).

**CTC was limited to 40 time steps during training.** The PP-LCNet backbones (PP-OCRv4/v5/v6) pool the feature
map to 40 positions while training, and to width / 8 at inference. CTC needs at least as many steps as the label
has characters (plus one per repeated pair). Paddle's CTC loss **silently returns zero loss and zero gradient**
for a label that does not fit: lines longer than ~40 characters never trained the CTC head, they diluted the
batch loss, and `best_accuracy` was in effect chosen on short lines only. Validation at a fixed 48×320 had a
ceiling of ~0.63 line accuracy on our data.

| Label | CTC steps needed | CTC loss at T = 40 | Gradient |
|---|---|---|---|
| 10 characters | 10 | 180.1 | yes |
| 40 characters | 40 | 223.6 | yes |
| 38 characters with doubled letters | 44 | **0** | **0** |
| 85 characters | 85 | **0** | **0** |

Models still read long lines at inference because CTC generalises from short contexts, but they never learned
from them.

## 3. Changes to PaddleOCR (core)

All changes are config keys, **off by default**; upstream configs behave as before. They are documented in
`docs/version3.x/module_usage/text_recognition.en.md`, section 4.2.1.

| Key | Effect |
|---|---|
| `Backbone.train_seq_len: -1` | PPLCNetV3/V4, PPHGNetV2: during training T = width / 8, as at inference (default 40) |
| `MultiScaleSampler.sorted_batch_ratio`, `pad_to_longest` | a share of batches sorted by aspect ratio (little padding) mixed with shuffled ones; batch width follows its widest image |
| `Eval.sampler` | validation like inference: every line resized by its own aspect ratio, no width cap at 320 |
| `RecMetric.length_buckets: [15, 40, 60]` | accuracy and normalized edit distance by label length, in training logs and eval |
| `RecConAug.add_space`, `fit_batch_width`, `ctc_stride`, `skip_unfit`, `ext_aug` | concatenation augmentation inserts a background gap and a space into the label, never makes a sample wider than its batch or longer than its CTC steps |
| `RecAug.tia_max_vertical_ratio` | TIA distortion no longer moves points vertically by more than a share of the line height (it destroyed long lines) |
| `RecAug.rotate_prob`, `downscale_prob`, `jpeg_prob`, `stroke_prob`, `edge_lines_prob` | scan augmentations: rotation limited by line length, low DPI, JPEG artifacts, stroke width, table rules at the edges (a detector box on a table cell) |
| `Global.save_on_signal` | on SIGTERM / SIGINT finish the step and save `latest` with its position in the epoch (12 h Kaggle sessions) |
| fixes | `MultiScaleDataSet.ratio_list` picked the narrowest lines instead of a random subset; export of a model validated through a sampler lacked `RecResizeImg`; TIA vectorised (6–11× faster); RecConAug candidates filtered before reading (data loading 42 → ~120 lines/s per core) |

## 4. Model

**PP-OCRv6 small** (PPLCNetV4 backbone, 2.7 M parameters, LightSVTR neck, CTC + NRTR heads) with
`train_seq_len: -1`. It has the same layout as RepSVTR, is newer, and its pretrained weights load except for the
output layers: the official v6 dictionary has no Cyrillic, so the CTC and NRTR output layers are trained from
scratch on our dictionary (`dict/ru_dict_ext011026.txt`: 164 characters + space). Batches are up to 2 400 px wide
at height 48, so lines up to ~250 characters keep their resolution.

The ported RepSVTR (`configs/ru_RepSVTR_rec_150525.yml`) is kept as a baseline with the same data and metrics.

## 5. Data

| Source | Lines | Labels |
|---|---|---|
| **Synthetic** documents (`doc-generator`): clean pages | 1 985 902 (4 parts) | exact, generated |
| **Synthetic** with page degradation (Augraphy) | 497 805 | exact, generated |
| **stroyinf v2**: 1 700 Soviet-era standards (typewriter scans) | 609 713 (train 580 800 / val 28 913) | PDF text layer, filtered |
| **stroyinf modern**: 1 000 standards and codes ≥ 1992 (ГОСТ, ГОСТ Р, СП, СНиП, РД) | 1 082 620 (train 1 060 968 / val 21 652) | PDF text layer, filtered |

- The synthetic lines use 40 free-licensed fonts (caps-only fonts and fonts with missing glyphs removed) and are
  balanced by length, font and character frequency.
- Real lines are crops of PDF pages cut along the text layer (`ocr_markup`), pages below a text-layer quality
  threshold are skipped, no OCR is run. Filters drop vertical text, lines without letters or digits, filler lines
  and lines with an implausible characters-per-width ratio.
- Validation: the val folder of stroyinf v2 (whole documents never seen in training) plus the last 2 % of
  stroyinf modern; `all_val5k` is a fixed random sample of 5 000 of these lines.

Total training set before cleaning: **4 125 475** lines.

## 6. First training runs

All runs: Kaggle, 2× T4, fp16, batch 32 per card, 12 h sessions chained through checkpoints
(`kaggle/ru_train_2xT4.ipynb`). Throughput ~95–105 lines/s.

| Run | Start | Data | Steps | Validation |
|---|---|---|---|---|
| v6 small | official pretrained v6 small weights | synthetic + stroyinf v2 | 88 000 (~1.6 epochs) | stroyinf v2 val5k: acc 0.706 (ported RepSVTR: 0.534) |
| ft3 | weights of step 88k | + stroyinf modern + degraded synthetic | 103 355 (best at 100 000) | `all_val5k`: acc 0.774, ned 0.952 |

Looking at the "errors" of the step-85k model showed that most of them were **label errors** of the text layer
("накатил" for "нажатия", "утверащен", letters split by spaces) while the model read the crop correctly. The
measured accuracy was capped by the labels, so the next step was to clean them.

## 7. Label cleaning

Cleaning uses two independent readers of every real crop: our model (ft3) and a VLM. A label is changed only when
the two agree against it.

1. **Predictions.** ft3 reads all 1 692 333 real lines (v2 on CPU with OpenVINO, modern on 2× T4 with Paddle;
   `kaggle/ru_export_eval_predict.ipynb`). `tools/build_review.py` splits out the disagreements:

   | | Lines | Model = label | Differ in characters | Differ only in spaces |
   |---|---|---|---|---|
   | stroyinf v2 | 609 713 | 74.4 % | 111 980 | 44 028 |
   | stroyinf modern | 1 082 620 | 86.5 % | 71 205 | 75 294 |

2. **VLM third opinion.** Every disagreement is read by **PaddleOCR-VL-1.6** (0.9 B, prompt `OCR:`) served by
   vLLM 0.26 on 2× T4 in fp16, ~32 lines/s (`kaggle/ru_vlm_check_stroyinf.ipynb`). On its own the VLM is not a
   reliable Russian reader of line crops: it misreads letters, writes formulas in LaTeX, and in ~13 % of the
   hard v2 lines answers in Latin or other scripts. It is used only as a tie-breaker.

3. **Verdict** (`tools/clean_labels_vlm.py`, `vlm_verdict`). Only the VLM text is normalized (whitespace runs
   collapsed); label and prediction are compared as they are:
   - VLM == label: the label is kept; VLM == prediction: the label becomes the prediction.
   - **Spaces only.** When label and prediction differ only in spaces, their characters are confirmed by both;
     the VLM decides by *where it puts spaces* (same character count, same gap positions), its misread letters
     do not matter. A prediction with a space at a line end (an artifact of batch inference) is never chosen.
   - **Look-alike letters.** When they differ only in Cyrillic/Latin twins (`О/O`, `С/C`, `Р/P`…), neither the
     model nor the VLM can tell them apart; the alphabet of the line's other letters decides (`OK` → `ОК` in
     Russian text), and a lone letter keeps the label (the text layer holds the real code point).
   - Every kept stroyinf label has words in one alphabet (twins follow the word's majority).
   - Everything else is **dropped**.

| Label file | Before | After | Label confirmed | Label := prediction | Dropped |
|---|---|---|---|---|---|
| `all_train` | 4 125 475 | 3 995 865 | 95 092 | 63 325 | 129 610 (3.1 %) |
| `all_val` | 50 565 | 42 510 | 3 321 | 3 104 | 8 055 |
| `all_val5k` | 5 000 | 4 241 | 331 | 315 | 759 |

Synthetic lines have exact labels and are not touched.

## 8. Fine-tuning on the cleaned labels (ft4)

| | |
|---|---|
| Start | weights of ft3 `best_accuracy` (step 100 000); new optimizer and schedule |
| Data | `all_train_clean` (3 995 865 lines) |
| Schedule | 2 epochs (~173 600 steps), cosine from 0.0005, no warmup |
| Hardware | 2× T4, two 12 h sessions, ~9 700 steps/h |

## 9. Results

`tools/eval.py` (the training eval: every line at its own width, batch of similar widths, exact match, spaces
count), PaddleOCR 3.3.1, same checkpoints on both sets:

| Model | `all_val5k_clean` (4 241) | `all_val5k` (5 000, raw labels) |
|---|---|---|
| ft3 | 0.9505 | 0.7738 |
| **ft4** | **0.9599** | **0.7844** |

Line accuracy by label length on `all_val5k_clean`:

| Length | ft3 | ft4 | Lines |
|---|---|---|---|
| 0–15 | 0.923 | **0.936** | 1 246 |
| 16–40 | 0.972 | 0.970 | 1 024 |
| 41–60 | 0.964 | **0.974** | 1 257 |
| 61+ | 0.944 | **0.964** | 714 |

- ft4 is better on both sets; the gain on the **raw** labels (+1.06 points) is not smaller than on the cleaned
  ones (+0.94), so it is not an artifact of cleaning the validation with the same rule.
- The cleaned validation is easier than the raw one by construction: lines where model and label disagreed and
  the VLM confirmed neither were removed. The true accuracy lies between the two columns; the raw column is a
  lower bound, since many of its "errors" are label errors.
- Long lines gained the most (+2 points), confirming that long-line training works.
- Eval with batch 1 (no padding at all) gives the same accuracy as with batches (ft3 on an earlier version of
  the cleaned set: 0.904 vs 0.905): the model does not depend on padding.

## 10. Known limitations

- **Spaces at line ends in batch inference.** Through PaddleX with batches padded to the widest line, ~5–7 % of
  predictions get a leading or trailing space (none of the labels has one). The CTC metric counts it as an
  error; strip predictions at inference.
- **Single characters**: case of lone letters (`Т`/`т`, `К`/`к`) and digit/letter twins (`3`/`з`, `6`/`б`) are
  ambiguous without context.
- **Punctuation**: em dash `—` vs hyphen `-`, decimal comma vs point, a period added after headings.
- **Formulas, indices, superscripts** (`H₁`, `γk`) are out of scope of the dictionary and the labels.
- **Validation** is 2 % of the modern documents and one folder of v2; it is document-level text of standards,
  not handwriting or photos.
- Cleaning drops lines the VLM could not resolve; they are the hardest real lines and the training set lost them.

## 11. Reproducing

| Step | Where |
|---|---|
| Synthetic lines | `doc-generator` (separate repository) |
| Real crops from PDF text layers | `ocr_markup` (separate repository) |
| Label files with width / height, train/val split, merge | `tools/add_wh.py`, `tools/merge_labels.py` |
| Archives for Kaggle | `tools/pack_dataset.py --format tar` (or `--from-zip`) |
| Training | `kaggle/ru_train_2xT4.ipynb` + `configs/ru_PP-OCRv6_small_rec.yml` |
| Predictions on the real lines, review tables | `kaggle/ru_export_eval_predict.ipynb` (`RUN_PREDICTIONS = True`), `tools/build_review.py` |
| VLM check | `kaggle/ru_vlm_check_stroyinf.ipynb` (built by `kaggle/make_vlm_nb.py`), `tools/vlm_review.py` |
| Cleaned label files | `tools/clean_labels_vlm.py` |
| Export, ONNX, test of checkpoints | `kaggle/ru_export_eval_predict.ipynb` (sections 1–3b) |

Details of each step and the commands are in [README.md](README.md).
