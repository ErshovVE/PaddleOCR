# Задача: третье мнение VLM по расхождениям stroyinf (Kaggle)

## Цель

Для строк stroyinf v2, где метка текстового слоя (`label`) и наша модель (`pred`, PP-OCRv6 small ft3,
best_accuracy step 100k) расходятся по символам, получить независимое чтение кропа VLM
(**PaddleOCR-VL-1.6**, запрос `OCR:`) и вердикт: с кем согласна VLM. Результат нужен для чистки меток (этап 7).

Сначала отладка на 10 строках, потом пилот на 1000 случайных строк (скорость + раскладка вердиктов),
потом, если имеет смысл, все 111 980 строк.

## Что уже есть

**Предсказания** нашей модели на всех 609 713 строках stroyinf v2 (`stroyinf_all_train_w_h.txt` +
`stroyinf_all_val_w_h.txt`) собраны локально в `output/stroyinf_preds/` (не в git):
`preds_all.tsv.gz`, `review.tsv` (111 980 строк, расходятся по символам), `review_spaces.tsv` (44 028, только пробелы).
Колонки `review.tsv`: `path label pred score dist dist_ns`.

**Датасеты Kaggle** (все приватные, аккаунт `vladislavershov`):

| датасет | что внутри |
|---|---|
| `ru-ocr-stroyinf` | кропы, `stroyinf/<папка>/crops/<n>/image_*.webp` |
| `ru-ocr-stroyinf-review` | `review.tsv` (111 980 строк) |
| `ru-ocr-vlm-debug` | 10 строк `review.tsv` + их 10 кропов в той же структуре — для быстрой отладки |
| `ru-ocr-stroyinf-vlm` | сюда ноутбук грузит результаты (`vlm_part*.tsv`), пока не создан |

**Ноутбук** `ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb`, собирается генератором
`ru_ocr/kaggle/make_vlm_nb.py` (`python ru_ocr/kaggle/make_vlm_nb.py ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb`
из корня репо). Правки делать в генераторе, ноутбук пересобирать.

Устройство ноутбука:
- параметры: `PILOT` (None = всё), `ENGINE` (`vllm` / `transformers`), `MODEL_ID`, `GPUS=2`, `SAVE_EVERY`,
  `OUT_DATASET` (None = не грузить; для отладки ставить None, иначе индексы 0–9 отладочного review попадут
  в боевой датасет);
- `sh()` — запуск shell-команд с выводом в ячейку и таймаутом (раньше `!pip` вис намертво);
- раздел 1: находит `review.tsv` и папки `crops` во всех `/kaggle/input`, линкует в `/tmp/ru_root/stroyinf`,
  строки в фиксированном случайном порядке (`SEED`), продолжение по найденным `vlm_part*.tsv`;
- раздел 2: `vlm_worker.py` — по процессу на GPU (`CUDA_VISIBLE_DEVICES`), пишет `index \t text \t score`
  каждые `SAVE_EVERY` строк, при перезапуске пропускает записанное;
- раздел 3: `run_workers(engine)` — запускает воркеров, раз в минуту печатает новые строки их логов,
  при падении печатает хвосты логов; если `vllm` упал — автоматически перезапуск на `transformers`;
  загрузка частей в `OUT_DATASET` раз в 10 мин через `ru_ocr/tools/ckpt_uploader.py` (параметр `required`
  уже в ветке, коммит dd46306);
- раздел 4: `review_vlm.tsv` = review + `vlm`, `vlm_score`, `verdict` (`label_ok`, `pred_ok`, `label_closer`,
  `pred_closer`, `unclear`; сравнение без пробелов, латинские двойники приводятся к кириллице).

## Что выяснилось при отладке (Kaggle, 2x T4, Python 3.13, torch 2.11.0+cu128, transformers 5.16)

1. **vllm 0.27–0.31 с PyPI** требуют `torch==2.13.0` (CUDA 13): pip сносит torch образа и ставит весь CUDA 13
   стек (~3 ГБ), после чего `transformers` падает на `torchaudio` (cu128 против cu13). Не использовать.
2. **vllm 0.26.0 с PyPI** собран под PyPI-torch 2.11 = CUDA 13. Для образа нужна сборка
   `vllm-0.26.0+cu129` из GitHub-релиза (пинит `torch==2.11.0`, `humming-kernels[cu12]`).
3. Даже с ней vllm требует `torchcodec>=0.14`, pip ставит свежий (CUDA 13) → `libnvrtc.so.13` не найден,
   а vllm импортирует torchcodec при старте (`vllm/multimodal/video.py`). В образе был `torchcodec 0.11.0+cu128`
   (колесо есть на `https://download.pytorch.org/whl/cu128`).
4. **Движок `transformers`** (уже в образе, ничего не ставит) **работает**: отладка на 10 строках прошла,
   обе GPU, `PaddlePaddle/PaddleOCR-VL-1.6`, fp32, batch 1. Скорость 0.1 стр/с — с учётом загрузки модели,
   не показательна. Таблицу вердиктов этого прогона пользователь ещё не прислал.

## Незакоммиченное состояние (локально, не в GitHub)

- `ru_ocr/kaggle/ru_vlm_check_stroyinf.ipynb` — версия с переустановкой torchcodec 0.11 после vllm и проверкой
  `from vllm import LLM` (если не импортируется — сразу `transformers`).
- `ru_ocr/kaggle/make_vlm_nb.py` — генератор, в нём уже следующая (НЕ пересобранная в ноутбук) правка: vllm
  ставится `--no-deps`, затем `pip install -r ru_ocr/kaggle/requirements_vlm.txt -c /tmp/image_constraints.txt`,
  где constraints = версии torch/torchvision/torchaudio/torchcodec образа. Для этого клон репо перенесён
  в начало ячейки установки.
- `ru_ocr/kaggle/requirements_vlm.txt` — зависимости vllm 0.26.0+cu129 (из METADATA колеса) без torch-семейства.

**Открытый вопрос, ждёт решения пользователя:** клонировать ли репо в Kaggle (тогда `requirements_vlm.txt`
надо запушить — только с разрешения) или сделать ноутбук самодостаточным (зависимости и загрузчик
через `%%writefile` прямо в ноутбуке, без клона). Рекомендация — самодостаточный.

## Следующие шаги

1. Решить вопрос с клоном (см. выше), довести установку vllm до `vllm imports` на отладочном датасете.
   Если vllm на T4 не взлетает за разумное время — оставаться на `transformers` и ускорить его:
   батч > 1 (паддинг в processor), fp16 (проверить, что не ломает качество на 10 строках).
2. Пилот: Input `ru-ocr-stroyinf` + `ru-ocr-stroyinf-review`, `PILOT = 1000`, `OUT_DATASET` боевой.
   Оценить строк/с и раскладку вердиктов, показать пользователю 20–30 примеров по каждому вердикту.
3. Полный прогон — только после «да» пользователя, с оценкой времени.

## Правила работы с пользователем

- Тяжёлые/долгие прогоны и любой запуск на Kaggle GPU — сначала оценка и «да» пользователя.
- Коммит и push — только по явной просьбе. Ничего не клонировать/не скачивать на локальную машину пользователя
  без нужды (пользователь против повторных скачиваний разных версий пакетов).
- Не вызывать внешние LLM API без подтверждения.
- Живого лога у commit-прогонов Kaggle в UI пользователя нет: признак жизни — обновление `OUT_DATASET`.
- Kaggle CLI: `kaggle kernels output ... --file-pattern '<regex>'` (без фильтра качает весь клон репо → 429);
  `kaggle datasets create -p .` запускать изнутри папки (с путём на Windows падает).
