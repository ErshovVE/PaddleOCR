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
