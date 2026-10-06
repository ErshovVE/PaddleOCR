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


@pytest.mark.resource_intensive
def test_save_every_n_steps_and_resume_inside_epoch(repo_root, tmp_path):
    """Global.save_batch_step saves "latest" inside the epoch; resuming from such a
    checkpoint continues the same epoch and skips the batches already trained."""
    import pickle

    font_path = _font_path()
    if font_path is None:
        pytest.skip("no TrueType font with Cyrillic found")
    data = tmp_path / "data"
    data.mkdir()
    labels = _make_dataset(data, 6, font_path, seed=2)
    out = tmp_path / "output"

    def p(path):
        return Path(path).as_posix()

    def run(*extra):
        opts = [
            "Global.epoch_num=1", "Global.use_gpu=false", "Global.distributed=false",
            "Global.pretrained_model=", "Global.print_batch_step=1", "Global.eval_batch_step=[0,100000]",
            "Global.save_batch_step=2", f"Global.save_model_dir={p(out)}",
            f"Global.save_res_path={p(out / 'predicts.txt')}",
            f"Train.dataset.data_dir={p(data)}", f"Train.dataset.label_file_list=[{p(labels)}]",
            # one image per batch, one height: each step is a fraction of a second on CPU
            "Train.sampler.first_bs=1", "Train.loader.batch_size_per_card=1", "Train.sampler.scales=[[320,32]]",
            "Train.loader.num_workers=0",
            f"Eval.dataset.data_dir={p(data)}", f"Eval.dataset.label_file_list=[{p(labels)}]",
            "Eval.sampler.first_bs=1", "Eval.loader.num_workers=0", *extra,
        ]
        cmd = [sys.executable, "tools/train.py", "-c", "ru_ocr/configs/ru_PP-OCRv6_small_rec.yml", "-o"] + opts
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=1800)
        log = proc.stdout + proc.stderr
        assert proc.returncode == 0, log[-4000:]
        return log

    log = run()
    steps = log.count("global_step:")
    assert steps >= 4, log[-3000:]
    # saves inside the epoch (every 2 steps) plus the end-of-epoch one
    assert log.count("save model in") >= 1 + steps // 2, log[-3000:]
    with open(out / "latest.states", "rb") as f:
        states = pickle.load(f)
    assert "step_in_epoch" not in states  # the end-of-epoch save resumes at the next epoch

    # pretend the session stopped after 2 batches of epoch 1
    states.update(epoch=1, step_in_epoch=2, global_step=2)
    with open(out / "latest.states", "wb") as f:
        pickle.dump(states, f, protocol=2)
    log = run(f"Global.checkpoints={p(out / 'latest')}")
    assert "resume inside epoch 1: skipping the 2 batches already trained" in log, log[-3000:]
    # tools/program.py drops the last batch of an epoch on Windows (max_iter = len - 1);
    # the resumed epoch is shorter than len, so it trains up to the true last batch
    batches = steps + (1 if sys.platform == "win32" else 0)
    assert log.count("global_step:") == batches - 2, log[-3000:]


@pytest.mark.resource_intensive
def test_stop_signal_saves_latest_inside_epoch(repo_root, tmp_path):
    """Global.save_on_signal: a stop signal (SIGTERM from paddle.distributed.launch on
    Linux, Ctrl+Break here on Windows) finishes the step, saves "latest" with the
    position inside the epoch and exits 0."""
    import pickle
    import signal
    import time

    font_path = _font_path()
    if font_path is None:
        pytest.skip("no TrueType font with Cyrillic found")
    data = tmp_path / "data"
    data.mkdir()
    labels = _make_dataset(data, 6, font_path, seed=3)
    out = tmp_path / "output"

    def p(path):
        return Path(path).as_posix()

    opts = [
        "Global.epoch_num=200", "Global.use_gpu=false", "Global.distributed=false",
        "Global.pretrained_model=", "Global.print_batch_step=1", "Global.eval_batch_step=[0,100000]",
        "Global.save_batch_step=0", "Global.save_on_signal=true", f"Global.save_model_dir={p(out)}",
        f"Train.dataset.data_dir={p(data)}", f"Train.dataset.label_file_list=[{p(labels)}]",
        "Train.sampler.first_bs=1", "Train.loader.batch_size_per_card=1", "Train.sampler.scales=[[320,32]]",
        "Train.loader.num_workers=0",
        f"Eval.dataset.data_dir={p(data)}", f"Eval.dataset.label_file_list=[{p(labels)}]",
        "Eval.sampler.first_bs=1", "Eval.loader.num_workers=0",
    ]
    cmd = [sys.executable, "tools/train.py", "-c", "ru_ocr/configs/ru_PP-OCRv6_small_rec.yml", "-o"] + opts
    log_path = tmp_path / "train_stdout.txt"
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=repo_root, stdout=log_file, stderr=subprocess.STDOUT, **kwargs)
        try:
            deadline = time.time() + 600
            # stop in epoch 2, after a few of its steps (6 lines -> 5 or 6 steps per epoch)
            while log_path.read_text(encoding="utf-8", errors="replace").count("global_step:") < 8:
                assert proc.poll() is None and time.time() < deadline, log_path.read_text(errors="replace")[-3000:]
                time.sleep(0.5)
            proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
            proc.wait(timeout=300)
        finally:
            if proc.poll() is None:
                proc.kill()
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert proc.returncode == 0, log[-4000:]
    assert "stopped by signal" in log, log[-3000:]
    with open(out / "latest.states", "rb") as f:
        states = pickle.load(f)
    steps = log.count("global_step:")
    assert states["global_step"] == steps
    assert states["epoch"] >= 2 and states.get("step_in_epoch", 0) >= 1, states
