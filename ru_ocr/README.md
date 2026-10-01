# ru_ocr — обучение распознавания RU+EN на PaddleOCR 3.7

Локальный слой поверх изменений ядра в этой ветке: конфиги, словари и утилиты для обучения распознавания русского и
английского документного текста, включая длинные строки (60–95+ символов). В апстрим-PR не предназначен.

Ключи ядра, на которых всё построено (все выключены по умолчанию), описаны в
`docs/version3.x/module_usage/text_recognition.md`, раздел 4.2.1:
`train_seq_len`, `MultiScaleSampler.sorted_batch_ratio` / `pad_to_longest`, Eval через sampler,
`RecMetric.length_buckets`, `RecConAug.add_space` / `fit_batch_width` / `ctc_stride`, аугментации сканов в `RecAug`.

## Состав

| Путь | Что |
|---|---|
| `configs/ru_PP-OCRv6_small_rec.yml` | основная модель: PP-OCRv6 small, CTC-длина при обучении = ширина / 8, батчи до 1600 px, склейки с пробелом, аугментации сканов, валидация «как инференс», метрики по длине |
| `configs/ru_RepSVTR_rec_150525.yml` | базовая линия: RepSVTR с весами, перенесёнными из torchocr (`se_gate: none`); те же данные и метрики |
| `dict/ru_dict_ext011026.txt` | рабочий словарь: 164 символа (+ пробел добавляется кодом) |
| `dict/ru_dict_ext100124.txt` | словарь старой torch-модели (162 символа), нужен конвертеру |
| `tools/add_wh.py` | разметка `путь\tтекст` → `путь\tтекст\tw\th` (нужна `MultiScaleDataSet` с `ds_width: true`) + деление train/val |
| `tools/convert_torchocr_rec.py` | перенос весов torchocr → `.pdparams`, сверка с ONNX, перенумерация выходных слоёв под новый словарь |
| `tools/validate_models.py` | проверка экспортированных моделей через `TextRecognition`: точность и норм. редакционное расстояние по корзинам длины и группам |
| `tests/` | тесты утилит и конфигов, сквозное smoke-обучение (`resource_intensive`) |

## Данные

`MultiScaleDataSet` с `ds_width: true` читает 4 колонки: `путь<TAB>текст<TAB>ширина<TAB>высота`.
Если генератор кропов пишет только 2 колонки:

```bash
python ru_ocr/tools/add_wh.py --label-file <кропы>/good.txt --data-dir <кропы> --out-prefix data/mydata --val-ratio 0.1
```

Валидационная часть — хвост файла: кропы пишутся по документам, поэтому в val попадают целые документы.

## Обучение

Пути в конфигах относительные (`./train_big/…`); свои данные подставляются через `-o`:

```bash
python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c ru_ocr/configs/ru_PP-OCRv6_small_rec.yml \
    -o Train.dataset.data_dir=<кропы> "Train.dataset.label_file_list=[data/mydata_train_w_h.txt]" \
       Eval.dataset.data_dir=<кропы> "Eval.dataset.label_file_list=[data/mydata_val_w_h.txt]"
```

- Предобученные веса v6 small — `./pretrain/PP-OCRv6_small_rec_pretrained.pdparams` (ссылка в
  `docs/version3.x/module_usage/text_recognition.md`); с нуля — `-o Global.pretrained_model=`.
- Выходные слои CTC/NRTR учатся заново: в словаре v6 нет кириллицы.
- Широкие батчи требуют много памяти: при нехватке уменьшить `Train.sampler.first_bs`.
- В логе: `acc_len_0_15 … acc_len_61_inf` — точность по длине; `RecConAug: …% of … samples concatenated` — доля склеек.

## Экспорт и проверка

```bash
python tools/export_model.py -c ru_ocr/configs/ru_PP-OCRv6_small_rec.yml \
    -o Global.pretrained_model=./output/ru_PP-OCRv6_small_rec/best_accuracy \
       Global.save_inference_dir=./output/ru_PP-OCRv6_small_rec/inference
python ru_ocr/tools/validate_models.py --models ./output/ru_PP-OCRv6_small_rec/inference \
    --label-file data/mydata_val_w_h.txt --data-dir <кропы> --out results/
```

`validate_models.py` пишет `result_<модель>.txt` (предсказания) и `metrics_<модель>.csv`.

## Перенос старой torch-модели

torch и paddle на Windows не работают в одном процессе, поэтому два шага в разных окружениях:

```bash
python ru_ocr/tools/convert_torchocr_rec.py extract --src model_20.pth --out weights.npz          # python с torch
python ru_ocr/tools/convert_torchocr_rec.py convert -c ru_ocr/configs/ru_RepSVTR_rec_150525.yml \
    --src-dict ru_ocr/dict/ru_dict_ext100124.txt --src weights.npz --out rec.pdparams --onnx rec.onnx  # python с paddle
```

`--src-dict` — словарь, с которым обучалась torch-модель: модель сверяется с ONNX на нём, затем выходные слои
перенумеровываются под словарь конфига (известные символы сохраняют обученные веса, новые учатся при дообучении).
`extract` использует `torch.load(weights_only=False)` — только для доверенных чекпойнтов.

## Тесты

```bash
python -m pytest tests/ppocr ru_ocr/tests -q
TORCH_PYTHON=<python с torch> python -m pytest -m resource_intensive ru_ocr/tests -q   # ONNX-сверка, smoke-обучение
```
