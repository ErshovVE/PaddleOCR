"""End-to-end smoke run of tools/train.py with ru_PP-OCRv6_small_rec.yml on synthetic lines.

Checks that the whole pipeline works together: proportional pooling at train time,
mixed sorted / shuffled batches, RecConAug with spaces, Eval through the sampler and
length-bucket metrics in the log. Takes a few minutes on CPU.
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]
WORDS = "привет мир документ строка номер договор Москва 2026 invoice total amount OCR".split()


def _font_path():
    return next((f for f in FONTS if os.path.exists(f)), None)


def _make_dataset(root, n, font_path, seed):
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    font = ImageFont.truetype(font_path, 28)
    lines = []
    for i in range(n):
        # short words and long lines (up to ~90 chars), as in the real data
        n_words = rng.choice([1, 1, 2, 3, 8, 12])
        text = " ".join(rng.choice(WORDS) for _ in range(n_words))[:95]
        width = int(font.getlength(text)) + 16
        img = Image.new("RGB", (width, 40), "white")
        ImageDraw.Draw(img).text((8, 4), text, fill="black", font=font)
        name = f"img_{i}.png"
        img.save(root / name)
        lines.append(f"{name}\t{text}\t{width}\t40")
    label_file = root / "labels_w_h.txt"
    label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label_file


@pytest.mark.resource_intensive
def test_smoke_train(repo_root, tmp_path):
    font_path = _font_path()
    if font_path is None:
        pytest.skip("no TrueType font with Cyrillic found")
    train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    train_labels = _make_dataset(train_dir, 64, font_path, seed=0)
    eval_labels = _make_dataset(eval_dir, 16, font_path, seed=1)
    out = tmp_path / "output"

    def p(path):
        return Path(path).as_posix()

    opts = [
        "Global.epoch_num=1",
        "Global.use_gpu=false",
        "Global.distributed=false",
        "Global.pretrained_model=",
        "Global.print_batch_step=1",
        "Global.eval_batch_step=[0,5]",
        f"Global.save_model_dir={p(out)}",
        f"Global.save_res_path={p(out / 'predicts.txt')}",
        f"Train.dataset.data_dir={p(train_dir)}",
        f"Train.dataset.label_file_list=[{p(train_labels)}]",
        "Train.sampler.first_bs=4",
        "Train.loader.num_workers=0",
        f"Eval.dataset.data_dir={p(eval_dir)}",
        f"Eval.dataset.label_file_list=[{p(eval_labels)}]",
        "Eval.sampler.first_bs=4",
        "Eval.loader.num_workers=0",
    ]
    cmd = [
        sys.executable,
        "tools/train.py",
        "-c",
        "ru_ocr/configs/ru_PP-OCRv6_small_rec.yml",
        "-o",
    ] + opts
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-4000:]
    assert "acc_len_" in log, log[-4000:]
    assert (out / "latest.pdparams").exists()
