# PR 2: recognition accuracy by label length (`RecMetric.length_buckets`)

- **Branch:** `ErshovVE:feat/rec-metric-length-buckets`, from `PaddlePaddle:main` (`dab3fe3`); 1 commit
  (`060856b`): 2 files, +158 / −3.
- **Checked on the branch** (Python 3.12, paddlepaddle 3.3.1 CPU, `.venv-test`):
  - `pytest tests/ppocr`: 51 passed, 15 of them in the new test file;
  - `black 24.10.0 --check` and `flake8 7.1.1` with the hook's `--select=E9,F63,F7,F82,E721`: clean.
- **Open:** <https://github.com/ErshovVE/PaddleOCR/pull/new/feat/rec-metric-length-buckets>, base
  `PaddlePaddle/PaddleOCR:main`. Copy the title and the body from the raw text of this file (not from a preview),
  into the Write tab.

---

## Title

[Feature] RecMetric: accuracy and edit distance by label length (`length_buckets`)

## Body

### Motivation

`RecMetric` reports one accuracy and one normalized edit distance over all lines. A model that reads short lines
well and long ones badly looks fine in that number. This is how a training problem on long text lines (CTC length,
see #18420) stayed hidden in our runs. The training log and `tools/eval.py` give no way to see accuracy by line
length.

### Change

New optional `RecMetric` argument `length_buckets`: increasing upper bounds of the label length.

```yaml
Metric:
  name: RecMetric
  main_indicator: acc
  length_buckets: [15, 40, 60]   # buckets 0-15, 16-40, 41-60, 61+
```

For every non-empty bucket the metric dict gets three more keys:

- `acc_len_<lo>_<hi>`: line accuracy;
- `ned_len_<lo>_<hi>`: 1 − normalized edit distance;
- `n_len_<lo>_<hi>`: number of lines;

for example `acc_len_0_15`, `acc_len_61_inf`. They appear in the training log and in `tools/eval.py` like the
existing keys.

- Without `length_buckets` the metric dict is exactly as before; `main_indicator` and best-model selection do not
  change.
- Upper bounds are inclusive. The length is that of the label as compared, so with `ignore_space: True` (the default)
  spaces are not counted, and with `is_filter: True` the filtered label is used.
- Empty buckets are not reported.
- A non-increasing or empty list fails at construction.
- `length_buckets` is passed through `**kwargs` like the other `RecMetric` options, so the metric wrappers that build
  `RecMetric` from config get it too.

Example from `tools/eval.py` (PP-OCRv6 small, 4 241 real Russian/English document lines, `ignore_space: False`):

```
acc: 0.9599, norm_edit_dis: 0.9895,
acc_len_0_15: 0.9358, ned_len_0_15: 0.9668, n_len_0_15: 1246.0,
acc_len_16_40: 0.9697, ned_len_16_40: 0.9981, n_len_16_40: 1024.0,
acc_len_41_60: 0.9737, ned_len_41_60: 0.9994, n_len_41_60: 1257.0,
acc_len_61_inf: 0.9636, ned_len_61_inf: 0.9994, n_len_61_inf: 714.0
```

### Tests

`tests/ppocr/test_rec_metric_buckets.py` (15 tests):

- bucket names and the bucket of a given length, bounds inclusive;
- invalid lists rejected;
- per-bucket accuracy, edit distance and counts against hand-computed values;
- empty buckets left out;
- the dict without `length_buckets` unchanged;
- the length counted after `ignore_space`;
- `reset()` between evaluations.
