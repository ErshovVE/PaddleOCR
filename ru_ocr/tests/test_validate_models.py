import csv

import pytest

from conftest import load_tool

vm = load_tool("validate_models")


@pytest.fixture
def samples():
    return [
        ("print/a.png", "a" * 5, "a" * 5),
        ("print/b.png", "b" * 30, "b" * 30),
        ("hand/c.png", "c" * 50, "c" * 49),
        ("hand/d.png", "d" * 80, "d" * 80),
        ("e.png", "e" * 80, "x" * 80),
    ]


def _by(rows, scope):
    return {r["name"]: r for r in rows if r["scope"] == scope}


class TestComputeMetrics:
    def test_overall(self, samples):
        row = _by(vm.compute_metrics(samples), "all")["all"]
        assert row["n"] == 5
        assert row["acc"] == pytest.approx(0.6)

    def test_length_buckets(self, samples):
        rows = _by(vm.compute_metrics(samples), "length")
        assert list(rows) == ["0_15", "16_40", "41_60", "61_inf"]
        assert rows["41_60"]["acc"] == 0.0
        assert rows["41_60"]["norm_edit_dis"] == pytest.approx(1 - 1 / 50)
        assert rows["61_inf"]["n"] == 2
        assert rows["61_inf"]["acc"] == 0.5

    def test_groups(self, samples):
        rows = _by(vm.compute_metrics(samples), "group")
        assert rows["print"]["acc"] == 1.0
        assert rows["hand"]["acc"] == 0.5
        assert rows["."]["n"] == 1

    def test_empty_buckets_omitted(self):
        rows = vm.compute_metrics([("x.png", "abc", "abc")])
        assert list(_by(rows, "length")) == ["0_15"]

    def test_no_samples(self):
        assert vm.compute_metrics([]) == []

    def test_custom_buckets(self, samples):
        rows = _by(vm.compute_metrics(samples, [40]), "length")
        assert list(rows) == ["0_40", "41_inf"]


@pytest.mark.parametrize(
    "path,group",
    [
        ("print/a.png", "print"),
        ("./hand/x/y.png", "hand"),
        ("a.png", "."),
        ("a\\b.png", "a"),
    ],
)
def test_group_of(path, group):
    assert vm.group_of(path) == group


def test_read_labels_with_and_without_size(tmp_path):
    label_file = tmp_path / "labels.txt"
    label_file.write_text(
        "a.png\tпривет мир\t100\t48\r\nb.png\thello\n\nbroken\n", encoding="utf-8"
    )
    assert vm.read_labels(str(label_file)) == [
        ("a.png", "привет мир"),
        ("b.png", "hello"),
    ]


def test_write_outputs(tmp_path, samples):
    rows = vm.compute_metrics(samples)
    result, metrics = vm.write_outputs(str(tmp_path), "m1", samples, rows)
    assert open(result, encoding="utf-8").read().splitlines()[0] == "print/a.png\taaaaa"
    with open(metrics, encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == len(rows)
    assert written[0]["scope"] == "all"


def test_main_with_fake_predictor(tmp_path, monkeypatch):
    label_file = tmp_path / "labels.txt"
    label_file.write_text(
        "g/a.png\tabc\t60\t48\ng/b.png\tdef\t60\t48\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        vm, "predict", lambda model_dir, name, paths, bs: ["abc", "xyz"]
    )
    out = tmp_path / "out"
    code = vm.main(
        ["--models", str(tmp_path / "model_a"), "--label-file", str(label_file)]
        + ["--out", str(out)]
    )
    assert code == 0
    rows = list(csv.DictReader(open(out / "metrics_model_a.csv", encoding="utf-8")))
    assert rows[0]["acc"] == "0.5"


def test_main_fails_on_missing_predictions(tmp_path, monkeypatch):
    label_file = tmp_path / "labels.txt"
    label_file.write_text("a.png\tabc\nb.png\tdef\n", encoding="utf-8")
    monkeypatch.setattr(vm, "predict", lambda model_dir, name, paths, bs: ["abc"])
    with pytest.raises(RuntimeError, match="1 predictions for 2 images"):
        vm.main(["--models", str(tmp_path / "m"), "--label-file", str(label_file)])
