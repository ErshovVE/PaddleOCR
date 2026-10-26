# PR 1: width-proportional CTC length in training (`train_seq_len`)

- **Branch:** `ErshovVE:feat/rec-train-seq-len`, from `PaddlePaddle:main` (`dab3fe3`); 2 commits
  (`63cac37`, `d2efafc`): 6 files, +171 / −18.
- **Checked on the branch** (Python 3.12, paddlepaddle 3.3.1 CPU, `.venv-test`):
  - `pytest tests/ppocr -m ""`: 79 passed, including the `resource_intensive` cases;
  - `black 24.10.0 --check` and `flake8 7.1.1` with the hook's `--select=E9,F63,F7,F82,E721`: clean;
  - the CTC repro below.
- **Not checked:** the rest of `tests/` (it downloads models), GPU, `Global.to_static`. With `train_seq_len: -1`
  under `to_static` the backbone asserts with a clear message instead of failing later.
- **Before opening:**
  - sign the CLA (the bot links it on the PR);
  - open from the fork: <https://github.com/ErshovVE/PaddleOCR/pull/new/feat/rec-train-seq-len>, base
    `PaddlePaddle/PaddleOCR:main`.

---

## Title

[Feature] Rec: width-proportional CTC length in training (`train_seq_len`) for PPLCNetV3/V4, PPHGNetV2, PPHGNet_small

## Body

### Problem

The PP-LCNet / PP-HGNet recognition backbones pool the feature map to a **fixed 40 time steps in training**
(`adaptive_avg_pool2d(x, [1, 40])`), but to `W / 8` steps at inference (`avg_pool2d(x, [3, 2])`). The backbones
are PPLCNetV3 (PP-OCRv4/v5 mobile rec), PPLCNetV4 (PP-OCRv6), PPHGNetV2 (PP-OCRv5 server rec) and PPHGNet_small
(PP-OCRv4 server rec).

CTC needs at least as many steps as the label has characters, plus one per repeated pair. For a label that does not
fit, `paddle.nn.functional.ctc_loss` **silently returns zero loss and zero gradient**. As a result:

- text lines longer than ~35–40 characters never train the CTC head;
- they still count in the batch, so they dilute the loss;
- the training accuracy looks far worse than the test accuracy.

On our Russian/English document lines (many 60–95 characters long), the training line accuracy was ~0.5 while the
test accuracy was ~0.9, and the CTC head had never seen a long line.

Repro (on this branch, CPU):

```python
import numpy as np, paddle
from ppocr.losses.rec_ctc_loss import CTCLoss
from ppocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3

for train_seq_len in (40, -1):
    backbone = PPLCNetV3(scale=0.95, train_seq_len=train_seq_len)
    backbone.train()
    T = backbone(paddle.rand([1, 3, 48, 960])).shape[3]   # a 960 px wide line
    logits = paddle.randn([1, T, 100])
    label = paddle.randint(1, 100, [1, 60], dtype="int32")  # 60 characters
    loss = CTCLoss()(logits, [None, label, paddle.to_tensor([60], dtype="int64")])["loss"]
    print(train_seq_len, T, float(loss))
# 40  40   0.0       <- the line is silently ignored
# -1  120  ~460      (random logits)
```

### Change

- New backbone option `train_seq_len` (default `40`: unchanged behaviour, existing configs and weights are not
  affected). With `-1`, the training sequence length is `W / 8` (`adaptive_avg_pool2d(x, [1, W_feat // 2])`), the
  same as at inference.
- The shared logic is in `ppocr/modeling/backbones/rec_seq_pool.py` (`rec_seq_pool`, `check_train_seq_len`). The
  four backbones call it; the eval and export paths are unchanged, so an exported model is identical.
- Invalid values (`0`, `< -1`) fail at construction. `-1` needs a known feature width at train time, so with
  `Global.to_static` it asserts with an explicit message.

Usage:

```yaml
Architecture:
  Backbone:
    name: PPLCNetV3          # or PPLCNetV4 / PPHGNetV2_B4 / PPHGNet_small
    train_seq_len: -1        # CTC length W / 8 at training time, the same as inference
```

With `-1`, batches of different widths give different sequence lengths. That is fine for CTC, and it works with
`MultiScaleSampler`, whose batches already have different widths.

### Tests

`tests/ppocr/test_rec_seq_pool.py`:

- `rec_seq_pool` itself: fixed and proportional lengths, input heights 32 / 48 / 64, a narrow input, the eval path,
  the height check;
- every backbone: the default is still 40 in training; `-1` gives `W / 8`; eval is unchanged with both settings;
  invalid values are rejected.

PPHGNetV2_B4 and PPHGNet_small are marked `resource_intensive`.

```
pytest tests/ppocr/test_rec_seq_pool.py -m ""      # 43 passed
```

### Results

We trained PP-OCRv6 small (PPLCNetV4) on 4.1 M Russian/English lines of 1–250 characters with `train_seq_len: -1`.
Line accuracy on our validation (4 241 real document lines, exact match, spaces count), by label length:

| Length | 0–15 | 16–40 | 41–60 | 61+ |
|---|---|---|---|---|
| Line accuracy | 0.936 | 0.970 | 0.974 | 0.964 |
| Lines | 1 246 | 1 024 | 1 257 | 714 |

Lines over 40 characters read as well as the short ones.

This is the result of the whole recipe; we did not run an ablation that changes only `train_seq_len`. The direct
evidence for the option is the zero loss above: with 40 steps those lines give no gradient at all. Details:
<https://github.com/ErshovVE/PaddleOCR/blob/feat/ru-ppocrv6-small-training/ru_ocr/EXPERIMENT.md>.
