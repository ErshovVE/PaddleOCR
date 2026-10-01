import numpy as np
from PIL import Image

from conftest import load_tool

aw = load_tool("add_wh")


def _make(tmp_path, n=10):
    (tmp_path / "crops").mkdir()
    lines = []
    for i in range(n):
        name = "crops/img_{}.webp".format(i)
        Image.fromarray(np.full((20 + i, 100 + 10 * i, 3), 255, np.uint8)).save(
            tmp_path / name
        )
        lines.append("{}\tтекст {}".format(name, i))
    lines += ["crops/missing.webp\tнет", "broken line", "crops/img_0.webp\t"]
    label = tmp_path / "good.txt"
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label


def test_sizes_and_split(tmp_path):
    label = _make(tmp_path)
    code = aw.main(
        ["--label-file", str(label), "--data-dir", str(tmp_path)]
        + ["--out-prefix", str(tmp_path / "ds"), "--val-ratio", "0.2"]
    )
    assert code == 0
    train = (tmp_path / "ds_train_w_h.txt").read_text(encoding="utf-8").splitlines()
    val = (tmp_path / "ds_val_w_h.txt").read_text(encoding="utf-8").splitlines()
    assert len(train) == 8 and len(val) == 2
    assert train[0] == "crops/img_0.webp\tтекст 0\t100\t20"
    assert val[-1] == "crops/img_9.webp\tтекст 9\t190\t29"  # tail goes to val


def test_skips_empty_text_and_missing_images(tmp_path):
    label = _make(tmp_path, n=2)
    rows, missing = aw.with_sizes(aw.read_pairs(str(label)), str(tmp_path))
    assert [r[0] for r in rows] == ["crops/img_0.webp", "crops/img_1.webp"]
    assert missing == ["crops/missing.webp"]


def test_zero_val_ratio():
    rows = [("a", "t", 1, 1)] * 3
    assert aw.split_tail(rows, 0.0) == (rows, [])
