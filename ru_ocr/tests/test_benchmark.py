import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, load_tool

bm = load_tool("benchmark", folder="scripts")

# stand-ins for cv2 and paddlex: a crop img_<w>.webp is an image <w> pixels wide, the model reads it as "<w>"
CV2_STUB = """import numpy as np
def imread(path):
    if "missing" in path:
        return None
    return np.zeros((10, int(path.rsplit("_", 1)[1].split(".")[0]), 3), np.uint8)
"""
PADDLEX_STUB = """class Model:
    def predict(self, images, batch_size):
        return [{"rec_text": str(image.shape[1])} for image in images]
def create_model(**kwargs):
    return Model()
"""


def test_plan_sorts_only_batches_above_one():
    assert bm.plan(["paddle cpu"], [1, 4], sort=True) == [("paddle cpu", 1, False), ("paddle cpu", 4, False),
                                                        ("paddle cpu", 4, True)]
    assert bm.plan(["a", "b"], [1, 8], sort=False) == [("a", 1, False), ("a", 8, False), ("b", 1, False), ("b", 8, False)]


def test_batch_order_by_width_over_height():
    sizes = [(10, 300), (20, 40), (10, 50), (48, 48)]  # (h, w): ratios 30, 2, 5, 1
    assert bm.batch_order(sizes, sort=False) == [0, 1, 2, 3]
    assert bm.batch_order(sizes, sort=True) == [3, 1, 2, 0]


def test_summarize_counts_spaces_and_trailing_space():
    row = bm.summarize([0.1, 0.3], ["ab", "c d ", "x"], ["ab", "c d", "y"], load_s=1.234)

    assert row["lines"] == 3 and row["total_s"] == 0.4 and row["lines_per_s"] == 7.5
    assert row["ms_per_line"] == pytest.approx(133.33) and row["ms_per_call_p95"] == 300.0
    assert row["acc"] == pytest.approx(0.3333) and row["acc_rstrip"] == pytest.approx(0.6667)
    assert row["load_s"] == 1.23


def test_model_kwargs_per_backend():
    kw, env = bm.model_kwargs("paddle gpu", "inf", "onnx")
    assert kw == {"model_dir": "inf", "device": "gpu:0"} and env == {"CUDA_VISIBLE_DEVICES": "0"}
    kw, env = bm.model_kwargs("openvino cpu", "inf", "onnx")
    assert kw["model_dir"] == "onnx" and kw["engine"] == "onnxruntime"
    assert kw["engine_config"]["providers"] == ["OpenVINOExecutionProvider"] and env == {"CUDA_VISIBLE_DEVICES": ""}
    assert bm.model_kwargs("onnxruntime cpu", "inf", "onnx")[0]["engine_config"]["providers"] == ["CPUExecutionProvider"]


def test_required_arguments():
    with pytest.raises(SystemExit):
        bm.parse_args(["--labels", "l.txt"])


def _stub_env(tmp_path):
    stubs = tmp_path / "stubs"
    (stubs / "paddlex").mkdir(parents=True)
    (stubs / "cv2.py").write_text(CV2_STUB)
    (stubs / "paddlex" / "__init__.py").write_text(PADDLEX_STUB)
    return dict(os.environ, PYTHONPATH=str(stubs))


def test_end_to_end_with_stub_model(tmp_path):
    widths = [120, 30, 400, 75, 260, 18, 90]
    labels = tmp_path / "val.txt"
    labels.write_text("".join(f"img_{w}.webp\t{w if k != 2 else 'wrong'}\t{w}\t10\n" for k, w in enumerate(widths)),
                      encoding="utf-8")
    out = tmp_path / "bench"
    cmd = [sys.executable, str(REPO_ROOT / "ru_ocr" / "scripts" / "benchmark.py"), "--labels", str(labels),
           "--root", str(tmp_path), "--paddle-model", "inf", "--onnx-model", "onnx",
           "--backends", "paddle cpu", "openvino cpu", "--batches", "1", "3", "--warmup", "1", "--out", str(out)]

    res = subprocess.run(cmd, capture_output=True, text=True, env=_stub_env(tmp_path))

    assert res.returncode == 0, res.stderr
    rows = [line.split("\t") for line in (out / "benchmark.tsv").read_text(encoding="utf-8").splitlines()]
    head, body = rows[0], rows[1:]
    assert head[:3] == ["backend", "batch", "sorted"] and head[-1] == "same_as_paddle_cpu_b1"
    assert [r[:3] for r in body] == [["paddle cpu", "1", "False"], ["paddle cpu", "3", "False"],
                                     ["paddle cpu", "3", "True"], ["openvino cpu", "1", "False"],
                                     ["openvino cpu", "3", "False"], ["openvino cpu", "3", "True"]]
    acc = head.index("acc")
    assert {r[acc] for r in body} == {str(round(6 / 7, 4))}  # sorted runs give the texts back in file order
    assert {r[-1] for r in body} == {"1.0"}
    run = json.loads((out / "run_openvino_cpu_b3_sorted.json").read_text(encoding="utf-8"))
    assert run["texts"] == [str(w) for w in widths]
    report = (out / "benchmark.md").read_text(encoding="utf-8")
    assert "available to the process" in report and "| openvino cpu | 3 | True |" in report
    assert json.loads((out / "hardware.json").read_text(encoding="utf-8"))["cpu_logical_host"] == os.cpu_count()


def test_missing_crop_fails_with_its_name(tmp_path):
    labels = tmp_path / "val.txt"
    labels.write_text("img_10.webp\ta\t10\t10\nmissing_20.webp\tb\t20\t10\n", encoding="utf-8")
    cmd = [sys.executable, str(REPO_ROOT / "ru_ocr" / "scripts" / "benchmark.py"), "--labels", str(labels),
           "--root", str(tmp_path), "--paddle-model", "inf", "--backends", "paddle cpu", "--batches", "1",
           "--out", str(tmp_path / "bench")]

    res = subprocess.run(cmd, capture_output=True, text=True, env=_stub_env(tmp_path))

    assert res.returncode != 0 and "missing_20.webp" in res.stdout + res.stderr
