import os
import pickle
import subprocess
import sys
import types

import numpy as np
import pytest

from conftest import load_tool

conv = load_tool("convert_torchocr_rec")

DATA = "data/rec_150525"


class FakeTensor:
    """Duck-typed torch tensor: detach().cpu().float().numpy()."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype="float32")

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.arr


class TestMapStateDict:
    @pytest.fixture
    def torch_state(self):
        rng = np.random.default_rng(0)
        return {
            "fc_square.weight": rng.random((4, 4)),
            "fc_rect.weight": rng.random((3, 5)),
            "fc_rect.bias": rng.random(3),
            "conv.weight": rng.random((8, 3, 3, 3)),
            "emb.weight": rng.random((10, 4)),
            "bn.running_mean": rng.random(8),
            "bn.running_var": rng.random(8),
            "bn.num_batches_tracked": np.array(5),
            "head.weight": rng.random((7, 4)),
            "extra.weight": rng.random(2),
        }

    @pytest.fixture
    def target(self):
        return {
            "fc_square.weight": (4, 4),
            "fc_rect.weight": (5, 3),
            "fc_rect.bias": (3,),
            "conv.weight": (8, 3, 3, 3),
            "emb.weight": (10, 4),
            "bn._mean": (8,),
            "bn._variance": (8,),
            "head.weight": (4, 6),  # other vocabulary size
            "new.weight": (2, 2),
        }

    @pytest.fixture
    def result(self, torch_state, target):
        linear = {"fc_square.weight", "fc_rect.weight", "head.weight"}
        return conv.map_state_dict(torch_state, target, linear)

    def test_linear_transposed_including_square(self, torch_state, result):
        mapped, _, _ = result
        np.testing.assert_allclose(
            mapped["fc_square.weight"], torch_state["fc_square.weight"].T
        )
        np.testing.assert_allclose(
            mapped["fc_rect.weight"], torch_state["fc_rect.weight"].T
        )

    def test_conv_and_embedding_untouched(self, torch_state, result):
        mapped, _, _ = result
        np.testing.assert_allclose(mapped["conv.weight"], torch_state["conv.weight"])
        np.testing.assert_allclose(mapped["emb.weight"], torch_state["emb.weight"])

    def test_batchnorm_renamed(self, torch_state, result):
        mapped, _, _ = result
        np.testing.assert_allclose(mapped["bn._mean"], torch_state["bn.running_mean"])
        np.testing.assert_allclose(
            mapped["bn._variance"], torch_state["bn.running_var"]
        )
        assert not any("num_batches_tracked" in k for k in mapped)

    def test_mismatches_listed(self, result):
        mapped, skipped, missing = result
        assert "head.weight" not in mapped
        assert any(s.startswith("head.weight shape") for s in skipped)
        assert any(s.startswith("extra.weight") for s in skipped)
        assert sorted(missing) == ["head.weight", "new.weight"]
        assert all(v.dtype == np.float32 for v in mapped.values())


class TestCollectState:
    def test_plain_state_dict(self):
        state = conv.collect_state({"a.weight": FakeTensor([1.0]), "meta": "x"})
        assert list(state) == ["a.weight"]

    def test_checkpoint_dict(self):
        state = conv.collect_state(
            {"state_dict": {"a.weight": FakeTensor([1.0])}, "epoch": 3}
        )
        assert list(state) == ["a.weight"]

    def test_pickled_module_tree(self):
        child = conv._Stub()
        child.__dict__.update(
            _parameters={"weight": FakeTensor([[1.0]]), "bias": None},
            _buffers={"running_mean": FakeTensor([0.0])},
            _modules={},
        )
        root = conv._Stub()
        root.__dict__.update(
            _parameters={}, _buffers={}, _modules={"block": child, "empty": None}
        )
        assert sorted(conv.collect_state(root)) == [
            "block.running_mean",
            "block.weight",
        ]


class TestTolerantUnpickler:
    def test_missing_module_becomes_stub(self):
        module = types.ModuleType("torchocr_fake_mod")

        class Net:
            pass

        Net.__module__ = "torchocr_fake_mod"
        Net.__qualname__ = "Net"
        module.Net = Net
        sys.modules["torchocr_fake_mod"] = module
        try:
            obj = Net()
            obj.layers = {"x": 1}
            payload = pickle.dumps(obj)
        finally:
            del sys.modules["torchocr_fake_mod"]

        import io

        restored = conv.TolerantUnpickler(io.BytesIO(payload)).load()
        assert isinstance(restored, conv._Stub)
        assert restored.layers == {"x": 1}


class TestCtcDecode:
    def test_merges_repeats_and_blanks(self):
        chars = ["blank", "a", "b"]
        probs = np.eye(3)[[1, 1, 0, 1, 2, 2, 0]]
        assert conv.ctc_greedy_decode(probs, chars) == "aab"


class TestHeadOutChannels:
    def test_multihead_nrtr(self):
        config = {
            "Architecture": {"Head": {"name": "MultiHead"}},
            "Loss": {"loss_config_list": [{"CTCLoss": None}, {"NRTRLoss": None}]},
        }
        head = conv.head_out_channels(config, 164)["Architecture"]["Head"]
        assert head["out_channels_list"] == {
            "CTCLabelDecode": 164,
            "NRTRLabelDecode": 167,
        }

    def test_single_head(self):
        config = {"Architecture": {"Head": {"name": "CTCHead"}}}
        head = conv.head_out_channels(config, 10)["Architecture"]["Head"]
        assert head["out_channels"] == 10


def test_linear_weight_keys_from_paddle_model(repo_root):
    model, character = conv.build_paddle_model(
        str(repo_root / "ru_ocr/configs/ru_RepSVTR_rec_150525.yml")
    )
    keys = conv.linear_weight_keys(model)
    shapes = {k: v.shape for k, v in model.state_dict().items()}
    assert keys and all(len(shapes[k]) == 2 for k in keys)
    assert not any("embedding" in k for k in keys)
    assert len(character) == 166  # blank + 164 chars + space
    assert character[-3:] == ["°", "—", " "]


def _torch_python():
    python = os.environ.get("TORCH_PYTHON", sys.executable)
    ok = subprocess.run([python, "-c", "import torch"], capture_output=True)
    return python if ok.returncode == 0 else None


@pytest.mark.resource_intensive
def test_reproduces_onnx_model(repo_root, tmp_path):
    data = repo_root / DATA
    if not (data / "model_20.pth").exists() or not (data / "rec_150525.onnx").exists():
        pytest.skip("data/rec_150525 is not available")
    python = _torch_python()
    if python is None:
        pytest.skip("no python with torch (set TORCH_PYTHON)")
    tool = repo_root / "ru_ocr" / "tools" / "convert_torchocr_rec.py"
    npz = tmp_path / "w.npz"
    subprocess.run(
        [python, str(tool), "extract", "--src", str(data / "model_20.pth")]
        + ["--out", str(npz)],
        check=True,
        cwd=repo_root,
    )
    code = conv.convert(
        str(repo_root / "ru_ocr/configs/ru_RepSVTR_rec_150525.yml"),
        str(npz),
        str(tmp_path / "rec.pdparams"),
        onnx_path=str(data / "rec_150525.onnx"),
        widths=[320, 960, 1600],
        src_dict=str(repo_root / "ru_ocr/dict/ru_dict_ext100124.txt"),
    )
    assert code == 0


class TestRemapVocab:
    OLD = ["a", "b", " "]
    NEW = ["a", "b", "°", "—", " "]

    def test_index_ctc(self):
        # blank, a, b, °(new), —(new), space
        assert conv.vocab_index(self.OLD, self.NEW, 1, 6, 4) == [0, 1, 2, -1, -1, 3]

    def test_index_nrtr_keeps_specials_and_spare_slot(self):
        # 4 specials + chars + 1 spare slot
        index = conv.vocab_index(self.OLD, self.NEW, 4, 10, 8)
        assert index == [0, 1, 2, 3, 4, 5, -1, -1, 6, 7]

    def test_index_size_mismatch(self):
        with pytest.raises(AssertionError, match="vocabulary axis"):
            conv.vocab_index(self.OLD, self.NEW, 1, 7, 4)

    def test_remap_keeps_trained_rows(self):
        src = {
            "head.ctc_head.fc.weight": np.arange(8, dtype="float32").reshape(2, 4),
            "head.ctc_head.fc.bias": np.array([10, 11, 12, 13], dtype="float32"),
            "backbone.conv.weight": np.ones((3, 3), dtype="float32"),
        }
        dst = {
            "head.ctc_head.fc.weight": np.full((2, 6), -1, dtype="float32"),
            "head.ctc_head.fc.bias": np.full(6, -1, dtype="float32"),
            "backbone.conv.weight": np.zeros((3, 3), dtype="float32"),
        }
        state, added = conv.remap_vocab(src, dst, self.OLD, self.NEW)
        assert added == ["°", "—"]
        np.testing.assert_array_equal(
            state["head.ctc_head.fc.bias"], [10, 11, 12, -1, -1, 13]
        )
        np.testing.assert_array_equal(
            state["head.ctc_head.fc.weight"][:, [0, 1, 2, 5]],
            src["head.ctc_head.fc.weight"],
        )
        np.testing.assert_array_equal(state["head.ctc_head.fc.weight"][:, 3:5], -1)
        np.testing.assert_array_equal(state["backbone.conv.weight"], 1)

    def test_remap_embedding_rows(self):
        src = {"head.gtc_head.embedding.embedding.weight": np.arange(8).reshape(8, 1)}
        dst = {"head.gtc_head.embedding.embedding.weight": np.full((10, 1), -1)}
        state, _ = conv.remap_vocab(src, dst, self.OLD, self.NEW)
        assert state["head.gtc_head.embedding.embedding.weight"][:, 0].tolist() == [
            0,
            1,
            2,
            3,
            4,
            5,
            -1,
            -1,
            6,
            7,
        ]


def test_new_dict_extends_old(repo_root):
    old = (repo_root / "ru_ocr/dict/ru_dict_ext100124.txt").read_text(encoding="utf-8")
    new = (repo_root / "ru_ocr/dict/ru_dict_ext011026.txt").read_text(encoding="utf-8")
    assert new.split("\n") == old.split("\n") + ["°", "—"]
