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

## 2. The main finding

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
| `Backbone.train_seq_len: -1` | PPLCNetV3/V4, PPHGNetV2, PPHGNet_small: during training T = width / 8, as at inference (default 40) |
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
`train_seq_len: -1`. Its pretrained weights load except for the output layers: the official v6 dictionary has no
Cyrillic, so the CTC and NRTR output layers are trained from scratch on our dictionary (`dict/ru_dict.txt`:
164 characters + space). Batches are up to 2 400 px wide at height 48, so lines up to ~250 characters keep their
resolution.

## 5. Data

| Source | Lines | Labels |
|---|---|---|
| **Synthetic** documents (`doc-generator`): clean pages | 1 985 902 (4 parts) | exact, generated |
| **Synthetic** with page degradation (Augraphy) | 497 805 | exact, generated |
| **stroyinf v2**: 1 700 Soviet-era standards (typewriter scans) | 609 713 (train 580 800 / val 28 913) | PDF text layer, filtered |
| **stroyinf modern**: 1 000 standards and codes ≥ 1992 (ГОСТ, ГОСТ Р, СП, СНиП, РД) | 1 082 620 (train 1 060 968 / val 21 652) | PDF text layer, filtered |

Total training set before cleaning: **4 125 475** lines, 40 % of them real.

### 5.1 Synthetic lines

The lines come from the balancer of [doc-generator](https://github.com/ErshovVE/doc-generator) (`docgen.balance`,
described in its `docs/BALANCER.md`):

1. A batch of 50 pages is generated as `.docx` and converted to PDF by LibreOffice.
2. The lines are cut out of the PDF with their boxes, font, size and style (`ocr_markup/backend/pdf_extract.py`).
3. A line is kept only if it fills a deficit of the target distribution on every axis. The other lines are never
   written.

One line is one crop, and its label is the exact text that was drawn.

- **Text.**
  - Phase 1: ordinary 10-page documents with Faker text in Russian and English (headings, paragraphs, lists,
    tables).
  - When fewer than 10 % of a batch's lines are accepted, phase 2 generates lines to order. Each missing cell
    (font, style, size, length, language, numbers) gets a sentence cut to length, with numbers and punctuation
    placed where real text has them. A page is a stack of such lines.
- **Targets.**
  - Font: 40 families, equal shares.
  - Size: 8–24 pt, flat on 8–14 pt and falling to 20 % at 24 pt.
  - Style: regular / italic / bold / bold italic as 40 / 25 / 25 / 10 %; 2 % underlined.
  - Length: 1–250 characters, flat on 1–125, at least 20 % of that level on 126–250.
  - Language: Russian / English / no letters as 47.5 / 47.5 / 5 %.
  - Numbers: 10 % of lines with a number of 1–25 digits.
  - Punctuation and rare forms:
    - 75 % of lines have punctuation;
    - Roman numerals appear in 2 % of lines, CAPS in 3 %, dot leaders in 2 %;
    - every dictionary character occurs at least 5 % as often as the mean of its group, so `ё ъ Ъ Ы Ь` are
      present.
  - Tolerance: a part counts as balanced when every bin is within 1 p.p. of its target.
- **Fonts.** The dataset is public, so all 40 families have free licenses: SIL OFL, Apache 2.0, CC BY, GPL with the
  font exception and others. Commercial fonts and the Microsoft core fonts are out. Each family was checked on a
  glyph sheet:
  - caps-only fonts were removed;
  - styles a family does not really have are not ordered, so the balance stays honest;
  - for each font, lines with characters it lacks or draws blank are rejected.

  This check removed fonts with broken Cyrillic, with `й` lacking its breve, and with `т` drawn as `п`.
- **Labels follow the dictionary.** Typographic dashes and quotes are drawn as such, but the label holds the
  dictionary character: `–` → `—`, `“ ” „` → `"`, `‘ ’` → `'`. Real lines use the same table. A line with a
  character outside the dictionary is rejected.
- **Parts 1–4.** About 500 k lines each, each balanced on its own:
  - seeds 1, 2 000 000, 3 000 000 and 4 000 000; they must be far apart, because the phase-1 document seed is
    `seed + 10⁶ + batch·1000 + document`;
  - worst deviation 0.56–0.72 p.p. (punctuation);
  - 90–91 % unique texts;
  - 1.5–2 h per part on 4 cores.
- **Part 5** (seed 5 000 000) degrades every page before the lines are cut.
  - The effects are ink, paper and post-processing effects of [Augraphy](https://github.com/sparkfish/augraphy).
    They do not move pixels, so the boxes stay exact, and they are consistent across a page (stains, shadows,
    bleed-through).
  - Each line is compared to its clean copy; a line that became unreadable is taken from the clean page instead.
  - 85 % of part 5 lines are degraded.
  - Augraphy costs ~3 s per page: 5 h 14 min for the part.

### 5.2 Real lines (stroyinf)

The real lines are scans of Russian building standards from files.stroyinf.ru, downloaded by
`ocr_markup/scripts/scrape/stroyinf.py`:

- v2: documents up to 1991, typewritten; ~2 250 downloaded, ~1 700 gave lines;
- modern: 1 000 documents from 1992 on.

Every PDF has an invisible text layer made by the site's own OCR. [ocr_markup](https://github.com/ErshovVE/ocr_markup)
cuts the crops along that layer (`POST /run` with `extract_pdf_text_layer`, `pdf_ocr_fallback: false`); no OCR is
run.

- A page is taken only when the quality of its layer is ≥ 0.98 (`text_layer_quality.PAGE_MIN_QUALITY`). At this
  threshold ~3 % of lines are bad, against ~12 % at 0.95–0.98.
- Labels are normalized by the same table as the synthetic ones. Lines with characters outside the dictionary go
  to `needs_review` and are not used: `• © £ | ■ ®` are garbage of the foreign OCR.
- The `merge_labels.py` filters drop 4.7 % of v2:
  - lines without letters or digits (lone `+ — )`, often mislabelled);
  - vertical text on the margins;
  - crops much wider than their text (< 0.8 characters per unit of w/h: table rows with the number cut off);
  - crops much narrower than their text (> 4 characters per unit of w/h);
  - dot leaders `.....`, whose dot count does not follow the picture.
- These rules do not catch foreign OCR errors made of dictionary characters (`ЭВМ` → `Э Ш`). The label cleaning of
  section 7 removes them.

### 5.3 Assembly and split

- **Data root.** One folder of links: `ru_root/synth` → the doc-generator output, `ru_root/stroyinf` → the
  ocr_markup data. Label paths are relative to it. Label files have 4 columns (`path, text, width, height`); both
  generators write them (`append_crop_size`).
- **Split.**
  - All synthetic lines go to train.
  - v2: validation is one whole extraction folder, 160 documents that training never sees.
  - Modern: validation is the last 2 % of its merged file. Crops are written document by document, so these are
    the last documents; one document may be cut at the border.
  - A new source is split on its own and appended to the existing train and val; the old splits are not touched.
- **Label files.**
  - `all_train` = synthetic parts 1–4 + v2 train + modern train + synthetic part 5.
  - `all_val` = v2 val + modern val (50 565).
  - `all_val5k` = `random.Random(0).sample(all_val, 5000)`.
  - After cleaning (section 7) the same files exist with `_clean`: `all_train_clean` 3 995 865, `all_val_clean`
    42 510, `all_val5k_clean` 4 241.
- **Kaggle.** `pack_dataset.py --format tar` packs the crops into ~1 GB `.tarpack` archives grouped by folder. The
  notebooks read these Kaggle datasets: `ru-ocr-synth-part1…5-tar`, `ru-ocr-stroyinf-tar`,
  `ru-ocr-stroyinf-modern-tar` and `ru-ocr-labels`.

The commands are in [README.md, Data](README.md#data).

## 6. First training runs

All runs: Kaggle, 2× T4, fp16, batch 32 per card, 12 h sessions chained through checkpoints
(`kaggle/ru_train_2xT4.ipynb`). Throughput ~95–105 lines/s.

| Run | Start | Data | Steps | Validation |
|---|---|---|---|---|
| v6 small | official pretrained v6 small weights | synthetic + stroyinf v2 | 88 000 (~1.6 epochs) | stroyinf v2 val5k: acc 0.706 |
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

### Inference speed on CPU

`scripts/benchmark.py` on the 4 241 lines of `all_val5k_clean`. Kaggle CPU session: AMD EPYC 7B12, 2 cores ×
2 threads (4 available to the process), 31 GB RAM. PaddleX `create_model`, images decoded in memory, library default
threads. Full table and hardware header: [`benchmarks/kaggle-cpu-4cores/`](benchmarks/kaggle-cpu-4cores/).

| Backend | batch 1 | batch 4 | batch 4, sorted | batch 8 | batch 8, sorted |
|---|---|---|---|---|---|
| Paddle | 6.9 | 8.0 | 12.3 | 7.9 | 13.6 |
| ONNX Runtime | 18.4 | 13.0 | 20.5 | 11.3 | 18.1 |
| OpenVINO | 18.5 | 13.7 | 21.3 | 12.3 | **21.4** |

Lines per second; "sorted": lines ordered by width/height before batching.

- **The backends read the same.** With the same batching, Paddle, ONNX Runtime and OpenVINO give identical texts on
  all 4 241 lines: the ONNX export loses nothing, the backend changes only the speed.
- **ONNX Runtime and OpenVINO are ~2.7× faster than Paddle** at batch 1 (54 ms per line against 146 ms).
- **On CPU a batch helps only with lines of similar width.** Every line of a batch is padded to the widest one:
  - without sorting, batches 4 and 8 are slower than batch 1;
  - sorted, OpenVINO gains 15 % (21.3–21.4 lines/s, ~47 ms per line), ONNX Runtime 11 % at batch 4 and nothing at
    batch 8, Paddle doubles but stays the slowest.
- **Accuracy depends on the batching, not on the backend:** 0.9611–0.9637 (±11 lines).
  - Sorted batches give 99.7–99.9 % of the batch-1 texts.
  - Unsorted ones change 2 % of the lines, both ways (`ППР` → `ПР`, but `Таблица2` → `Таблица 2`).

## 10. Known limitations

This is a research project, not a production model. ft4 is two short fine-tuning stages; it was not polished by a
long training tail at a small learning rate. In our experience such a tail removes the rare systematic slips like a
trailing space or `б`/`6`, `з`/`3` confusions.

- **A trailing space in batch inference.** In batches padded to the widest line a prediction can end with a space
  (none of the labels does). ft3 had it on ~5 % of the lines. In ft4 it has almost gone: 8 of 4 241 lines with
  unsorted batches of 8, 1 with batch 1 or sorted batches. It can still appear, so strip predictions at inference.
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
