# ru_ocr — распознавание русского и английского текста на PaddleOCR 3.7

<p align="center">
  <strong>Язык:</strong>
  <a href="../../README.md">English</a> |
  <b>🇷🇺 Русский</b>
</p>

Конфиги, словари, инструменты и ноутбуки Kaggle для обучения распознавания строк русского и английского
**документного** текста, включая длинные строки (60–95+ символов), на PP-OCRv6 small. Слой работает поверх изменений
ядра в этой ветке (все — ключи конфига, по умолчанию выключены) и в апстрим-PR не предназначен.

Сам эксперимент — задача, данные, чистка разметки с VLM, прогоны и результаты — описан в [EXPERIMENT.md](EXPERIMENT.md).

| Модель | Валидация, очищенная разметка | Валидация, исходная разметка |
|---|---|---|
| PP-OCRv6 small, ft4 | **0.960** точности по строке | 0.784 |

## Как обучена текущая модель

### Шаги

1. **Исправление ядра для длинных строк.** Бэкбоны PP-LCNet учили CTC только на 40 шагах, а CTC loss в Paddle для
   более длинных меток молча даёт нулевой loss и градиент. `Backbone.train_seq_len: -1` делает длину CTC при обучении
   равной ширине / 8, как на инференсе; батчи — до 2 400 px в ширину; валидация как инференс (каждая строка своей
   шириной). Остальные изменения ядра: склейки с пробелом, ограниченный TIA для длинных строк, аугментации сканов,
   метрики по длине строки ([EXPERIMENT.md, раздел 3](EXPERIMENT.md#3-изменения-в-paddleocr-ядро)).
2. **Данные.** Синтетические строки документов (`doc-generator`, чистые и деградированные страницы) + реальные строки,
   нарезанные по текстовому слою PDF российских стандартов (`ocr_markup`: stroyinf v2 — советские сканы, stroyinf
   modern — с 1992 года).
3. **Первый прогон (v6 small)** с официальных весов PP-OCRv6 small, выходные слои — новые, под наш словарь.
4. **ft3**: дообучение с его весов, добавлены современные стандарты и деградированная синтетика.
5. **Чистка разметки**: ft3 читает каждую реальную строку; строки, где он расходится с меткой текстового слоя, читает
   PaddleOCR-VL; подтверждённые VLM метки остаются, метки, которые VLM прочитала как модель, заменяются, остальное
   выкидывается ([EXPERIMENT.md, раздел 7](EXPERIMENT.md#7-чистка-разметки)).
6. **ft4**: дообучение с `best_accuracy` ft3 на очищенной разметке — текущая модель.
7. **Экспорт** в Paddle inference-модель и ONNX, сверка через Paddle / ONNX Runtime / OpenVINO.

### Этапы обучения

Всё на Kaggle, 2× T4, fp16, 12-часовые сессии, связанные чекпоинтами (`kaggle/ru_train_2xT4.ipynb`).

| Этап | Старт | Обучающие данные (строк) | Расписание LR | Обучено | Валидация |
|---|---|---|---|---|---|
| v6 small | официальные предобученные PP-OCRv6 small | синтетика 1 985 902 + stroyinf v2 580 800 | cosine 0.0005, 1 эпоха warmup | 88 000 шагов (~1.6 эпохи) | val5k stroyinf v2: 0.706 |
| ft3 | веса v6 small шага 88 000 | + stroyinf modern 1 060 968 + деградированная синтетика 497 805 = 4 125 475 | cosine 0.0005, без warmup, план 3 эпохи | 103 355 шагов (~1.15 эпохи), лучший — 100 000 | `all_val5k`: 0.774 |
| **ft4** | веса `best_accuracy` ft3 (шаг 100 000) | очищенная: 3 995 865 | cosine 0.0005, без warmup, 2 эпохи | ~173 600 шагов (2 эпохи, ~18 ч) | `all_val5k_clean`: **0.960**, `all_val5k`: 0.784 |

Каждый этап стартует только с весов предыдущего (`Global.pretrained_model`): оптимизатор и расписание начинаются
заново. `best_accuracy` выбирается по валидации каждые 10 000 шагов.

### Параметры (`configs/ru_PP-OCRv6_small_rec.yml` + ноутбук обучения)

| | |
|---|---|
| Модель | PP-OCRv6 small: бэкбон PPLCNetV4, `train_seq_len: -1`; CTC-голова с neck LightSVTR (dims 120, depth 2); NRTR-голова (dim 384, только при обучении) |
| Loss | CTC + NRTR (`MultiLoss`) |
| Словарь | `dict/ru_dict_ext011026.txt`, 164 символа + пробел; `max_text_length` 250 |
| Оптимизатор | Adam (β 0.9 / 0.999), L2 3e-5, cosine LR |
| Батч | 32 на карту, 2 карты; fp16 AMP (`scale_loss` 1024, динамический) |
| Вход | высота 32 / 48 / 64 (multi-scale) при обучении, 48 при валидации; ширина по пропорциям строки, батч до 2 400 px |
| Сэмплер | `MultiScaleSampler`: 80 % батчей отсортированы по пропорциям, 20 % перемешаны; ширина батча — по самой широкой строке |
| Склейки | `RecConAug` p 0.7: к строке подклеивается другая через промежуток фона 16–80 px с пробелом в метке, не шире батча и не длиннее его шагов CTC (склеенными оказываются ~6 % примеров) |
| Аугментации | TIA с вертикальным сдвигом ≤ 0.25 высоты строки; crop 0.1, шум 0.1, цветовой jitter 0.1, blur 0.1, HSV 0.1, reverse 0.01; сканы: поворот 0.2 (до 3° у слов, 0.6° при w/h 30), низкое разрешение 0.15, JPEG 0.15, толщина штрихов 0.1, линии таблицы по краям 0.1 |
| Валидация | `all_val5k_clean`, каждая строка своей шириной, точное совпадение, пробелы учитываются, точность по длине 0–15 / 16–40 / 41–60 / 61+ |

## Состав

Пути ниже — от папки `ru_ocr/`.

| Путь | Что |
|---|---|
| `configs/ru_PP-OCRv6_small_rec.yml` | основная модель: PP-OCRv6 small, длина CTC при обучении = ширина / 8, батчи до 2 400 px, склейки с пробелом, аугментации сканов, валидация как инференс, метрики по длине |
| `configs/ru_RepSVTR_rec_150525.yml` | базовая линия: RepSVTR с весами, перенесёнными из PyTorch-модели (`se_gate: none`), те же данные и метрики |
| `dict/ru_dict_ext011026.txt` | словарь: 164 символа (пробел добавляется кодом) |
| `dict/ru_dict_ext100124.txt` | словарь старой PyTorch-модели (162 символа), нужен конвертеру |
| `tools/add_wh.py` | `путь\tтекст` → `путь\tтекст\tw\th` (нужно `MultiScaleDataSet` с `ds_width: true`) + деление train/val |
| `tools/merge_labels.py` | сводит разметку нескольких папок кропов к одному корню, фильтрует разметку текстового слоя, val — целыми папками |
| `tools/pack_dataset.py` | упаковывает кропы, на которые ссылается разметка, в несколько архивов (`--format tar` → `*.tarpack`), перепаковывает zip (`--from-zip`) |
| `tools/build_review.py` | предсказания + разметка → `preds_all.tsv.gz`, `review.tsv`, `review_spaces.tsv` (расхождения) |
| `tools/vlm_review.py` | таблицы расхождений + ответы VLM → `review_vlm.tsv` с вердиктом по каждой строке |
| `tools/clean_labels_vlm.py` | пишет файлы `*_clean`: спорные метки оставляет или заменяет по вердикту VLM, остальные выкидывает |
| `tools/ckpt_uploader.py` | выгружает чекпоинты и файлы результатов в датасет Kaggle по ходу прогона |
| `tools/validate_models.py` | проверка экспортированных моделей через `TextRecognition`: точность и редакционное расстояние по длине и группам |
| `tools/convert_torchocr_rec.py` | перенос весов torchocr в `.pdparams`, сверка с ONNX, перенумерация выходных слоёв под новый словарь |
| `kaggle/ru_train_2xT4.ipynb` | обучение на Kaggle 2× T4 в 12-часовых сессиях с выгрузкой чекпоинтов и продолжением |
| `kaggle/ru_export_eval_predict.ipynb` | экспорт, ONNX, сверка бэкендов, тест чекпоинтов, предсказания на реальных строках |
| `kaggle/ru_vlm_check_stroyinf.ipynb` | чтение спорных строк VLM (PaddleOCR-VL через vLLM); собирается `kaggle/make_vlm_nb.py` |
| `tests/` | тесты инструментов и конфигов, сквозное smoke-обучение (`resource_intensive`) |

## Установка

```bash
pip install paddlepaddle==3.3.1 -r requirements.txt          # или paddlepaddle-gpu для обучения
pip install "paddlex[ocr-core]>=3.7,<3.8"                      # проверка экспорта, validate_models.py, инференс
```

Предобученные веса PP-OCRv6 small кладутся в `./pretrain/PP-OCRv6_small_rec_pretrained.pdparams`
(ссылка в `docs/version3.x/module_usage/text_recognition.md`).

## Данные

`MultiScaleDataSet` с `ds_width: true` читает 4 колонки: `путь<TAB>текст<TAB>ширина<TAB>высота`; конфиги читают
кропы из `./data/ru_root/`, разметку — из `./data/ru_labels/`.

Если генератор кропов пишет только 2 колонки:

```bash
python ru_ocr/tools/add_wh.py --label-file <кропы>/good.txt --data-dir <кропы> --out-prefix data/mydata --val-ratio 0.1
```

Валидационная часть — хвост файла: кропы пишутся по документам, поэтому в val попадают целые документы.

Несколько папок генераторов (у каждой свои `crops/…`) сводит `merge_labels.py`: пути — от общего корня, val —
целые папки. Источники с разными корнями (синтетика `doc-generator`, кропы `ocr_markup`) собираются в одну
папку-корень из ссылок, а `--prefix` дописывает имя ссылки к путям:

```bash
# data/ru_root/synth -> doc-generator/out/synth_2m, data/ru_root/stroyinf -> ocr_markup/data
python ru_ocr/tools/merge_labels.py --root <synth_2m> --inputs <synth_2m>/part1 <synth_2m>/part2 \
    --label dataset.tsv --prefix synth --out-prefix data/ru_labels/synth_2m
python ru_ocr/tools/merge_labels.py --root <ocr_markup/data> --inputs <папки stroyinf> --val <папка для val> \
    --drop-vertical --drop-no-alnum --drop-filler --min-cpu 0.8 --max-cpu 4 --prefix stroyinf \
    --out-prefix data/ru_labels/stroyinf_all
```

Фильтры (`--drop-*`, `--min-cpu/--max-cpu`) — для текстового слоя чужого OCR; синтетику ими не трогать: она уже
сбалансирована.

## Обучение

```bash
python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c ru_ocr/configs/ru_PP-OCRv6_small_rec.yml \
    -o Train.dataset.data_dir=<кропы> "Train.dataset.label_file_list=[data/mydata_train_w_h.txt]" \
       Eval.dataset.data_dir=<кропы> "Eval.dataset.label_file_list=[data/mydata_val_w_h.txt]"
```

- Выходные слои CTC и NRTR учатся с нуля: в официальном словаре v6 нет кириллицы. Со случайных весов —
  `-o Global.pretrained_model=`; только с весов чекпоинта — `Global.pretrained_model=<префикс>`.
- Широкие батчи требуют памяти: при нехватке уменьшить `Train.sampler.first_bs`.
- В логе: `acc_len_0_15 … acc_len_61_inf` — точность по длине метки; `RecConAug: …% of … samples concatenated` —
  доля склеек.

### На Kaggle (2× T4)

`kaggle/ru_train_2xT4.ipynb` обучает в 12-часовых сессиях. В первой ячейке задаются `KAGGLE_USER`, `REPO`, `BRANCH`
и параметры прогона; к ноутбуку подключаются архивы кропов (`*.tarpack` из `pack_dataset.py --format tar`, они
распаковываются в `/tmp/ru_root` параллельно), разметка и начальные веса, добавляется секрет `KAGGLE_API_TOKEN`,
дальше Save & Run All. Чекпоинты выгружаются в `CKPT_DATASET` каждые `SAVE_STEPS`; следующая сессия продолжает с
самого нового, если этот датасет подключён. Кропы загружайте как `.tarpack`, а не `.zip`: zip Kaggle распаковывает в
отдельные файлы, и сессия потом часами качает миллион кропов по одному.

### Smoke-проверка

В первой ячейке каждого ноутбука есть закомментированная настройка для маленького набора (несколько сотен кропов в
одном `.tarpack` плюс файлы разметки и таблицы расхождений с теми же именами): обучение — `RUN_NAME, CKPT_DATASET,
INIT_DATASET = "smoke", None, <датасет с весами>` и короткий `TIME_LIMIT`; экспорт и тест — по умолчанию,
`RUN_PREDICTIONS = True`, чтобы проверить и предсказания; VLM — `OUT_DATASET = None`. Когда датасеты выгрузки равны
`None`, секрет не нужен и в настоящие датасеты ничего не пишется. Не подключайте полные датасеты разметки вместе с
маленьким: ноутбуки находят файлы по именам.

## Чистка разметки реальных строк

Конвейер из [EXPERIMENT.md, раздел 7](EXPERIMENT.md#7-чистка-разметки): обученная модель читает каждый реальный кроп,
расхождения уходят в VLM, а вердикты решают, какие метки оставить, заменить или выкинуть.

```bash
# 1. предсказания: kaggle/ru_export_eval_predict.ipynb с RUN_PREDICTIONS = True пишет preds_partial.tsv,
#    затем (ноутбук делает это сам):
python ru_ocr/tools/build_review.py --preds preds_partial.tsv \
    --labels data/ru_labels/stroyinf_modern_train_w_h.txt data/ru_labels/stroyinf_modern_val_w_h.txt --out preds_modern
# 2. VLM: kaggle/ru_vlm_check_stroyinf.ipynb на review.tsv + review_spaces.tsv пишет vlm_part*.tsv, затем:
python ru_ocr/tools/vlm_review.py --reviews preds_modern/review.tsv preds_modern/review_spaces.tsv \
    --parts vlm_part0.tsv vlm_part1.tsv --out preds_modern/review_vlm.tsv
# 3. очищенная разметка рядом с исходной (*_clean_w_h.txt, никогда не перезаписывается)
python ru_ocr/tools/clean_labels_vlm.py --preds preds_v2/preds_all.tsv.gz preds_modern/preds_all.tsv.gz \
    --vlm preds_v2/review_vlm.tsv preds_modern/review_vlm.tsv \
    --labels data/ru_labels/all_train_w_h.txt data/ru_labels/all_val_w_h.txt
```

VLM-ноутбук ставит vLLM в отдельное виртуальное окружение, поэтому образ Kaggle (его torch) остаётся нетронутым.
Ноутбук собирается командой `python ru_ocr/kaggle/make_vlm_nb.py ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb` —
правьте генератор, а не ноутбук.

## Экспорт и инференс

```bash
python tools/export_model.py -c ru_ocr/configs/ru_PP-OCRv6_small_rec.yml \
    -o Global.pretrained_model=./output/ru_PP-OCRv6_small_rec/best_accuracy \
       Global.save_inference_dir=./output/ru_PP-OCRv6_small_rec/inference
paddlex --paddle2onnx --paddle_model_dir ./output/ru_PP-OCRv6_small_rec/inference \
    --onnx_model_dir ./output/ru_PP-OCRv6_small_rec/onnx              # экспорт на CPU-версии paddle
python ru_ocr/tools/validate_models.py --models ./output/ru_PP-OCRv6_small_rec/inference \
    --label-file data/mydata_val_w_h.txt --data-dir <кропы> --out results/
```

```python
from paddlex import create_model

model = create_model(model_name="PP-OCRv6_small_rec", model_dir="onnx", device="cpu", engine="onnxruntime",
                     engine_config={"providers": ["CPUExecutionProvider"]})
for res in model.predict(["line.png"], batch_size=1):
    print(res["rec_text"].strip(), res["rec_score"])
```

- В ONNX конвертируйте модель, экспортированную **CPU**-версией Paddle: в экспорте `paddlepaddle-gpu` есть операторы,
  которых `paddle2onnx` не знает.
- Обрезайте предсказания: в батчах, дополненных до самой широкой строки, несколько процентов из них получают пробел
  на краю.

## Перенос старой PyTorch-модели

На Windows torch и paddle в одном процессе не работают, поэтому два шага — в разных окружениях:

```bash
python ru_ocr/tools/convert_torchocr_rec.py extract --src model_20.pth --out weights.npz          # python с torch
python ru_ocr/tools/convert_torchocr_rec.py convert -c ru_ocr/configs/ru_RepSVTR_rec_150525.yml \
    --src-dict ru_ocr/dict/ru_dict_ext100124.txt --src weights.npz --out rec.pdparams --onnx rec.onnx  # python с paddle
```

`--src-dict` — словарь, с которым обучалась torch-модель: модель сверяется с ONNX на нём, затем выходные слои
перенумеровываются под словарь конфига (известные символы сохраняют обученные веса, новые учатся при дообучении).
`extract` использует `torch.load(weights_only=False)` — только для доверенных чекпоинтов.

## Тесты

```bash
python -m pytest tests/ppocr ru_ocr/tests -q
TORCH_PYTHON=<python с torch> python -m pytest -m resource_intensive ru_ocr/tests -q   # ONNX-сверка, smoke-обучение
```
