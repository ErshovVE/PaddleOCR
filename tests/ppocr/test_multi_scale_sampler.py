# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random

import cv2
import numpy as np
import pytest

from ppocr.data import build_dataloader
from ppocr.data.multi_scale_sampler import MultiScaleSampler
from ppocr.data.simple_dataset import MultiScaleDataSet, SimpleDataSet

np.random.seed(42)
random.seed(42)

LOGGER = logging.getLogger("test_multi_scale_sampler")


class FakeSource:
    """Minimal stand-in for MultiScaleDataSet as seen by the sampler."""

    def __init__(self, ratios, ds_width=True, seed=None):
        self.wh_ratio = np.asarray(ratios, dtype="float64")
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = list(range(len(ratios)))
        self.ds_width = ds_width
        self.seed = seed

    def __len__(self):
        return len(self.wh_ratio)


def _sampler(source, **kwargs):
    params = dict(scales=[[320, 48]], first_bs=4, fix_bs=True, max_w=1600)
    params.update(kwargs)
    return MultiScaleSampler(source, **params)


def _file_ids(source, batch):
    ids = []
    for item in batch:
        is_sorted = item[4] if len(item) > 4 else item[3] is not None
        ids.append(
            source.wh_ratio_sort[item[2]]
            if is_sorted
            else source.data_idx_order_list[item[2]]
        )
    return np.asarray(ids)


@pytest.fixture
def source():
    return FakeSource(np.random.uniform(0.5, 40, size=8000))


class TestSortedBatchRatio:
    def test_default_keeps_legacy_tuples(self, source):
        sampler = _sampler(source)
        for batch in sampler.batchs_in_one_epoch:
            assert all(len(item) == 4 for item in batch)
            ratios = source.wh_ratio[_file_ids(source, batch)]
            expected = min(ratios.mean(), 1600 / 48)
            assert batch[0][3] == pytest.approx(expected)

    def test_default_batches_are_sorted(self, source):
        batch = _sampler(source).batchs_in_one_epoch[0]
        ratios = source.wh_ratio[_file_ids(source, batch)]
        assert np.all(np.diff(ratios) >= 0)

    def test_share_of_sorted_batches(self, source):
        sampler = _sampler(source, sorted_batch_ratio=0.5)
        flags = [batch[0][4] for batch in sampler.batchs_in_one_epoch]
        assert len(flags) == 2000
        assert 0.45 <= np.mean(flags) <= 0.55

    @pytest.mark.parametrize("ratio", [0.0, 0.5])
    def test_each_image_once_per_epoch(self, source, ratio):
        sampler = _sampler(source, sorted_batch_ratio=ratio)
        ids = np.concatenate(
            [_file_ids(source, b) for b in sampler.batchs_in_one_epoch]
        )
        assert sorted(ids.tolist()) == list(range(len(source)))

    def test_shuffled_batches_use_fixed_width(self, source):
        sampler = _sampler(source, sorted_batch_ratio=0.0)
        for batch in sampler.batchs_in_one_epoch:
            assert all(item[3] is None and item[4] is False for item in batch)

    def test_sorted_batches_have_similar_ratio(self, source):
        sampler = _sampler(source, sorted_batch_ratio=0.5)
        sorted_spread, shuffled_spread = [], []
        for batch in sampler.batchs_in_one_epoch:
            ratios = source.wh_ratio[_file_ids(source, batch)]
            spread = sorted_spread if batch[0][4] else shuffled_spread
            spread.append(np.ptp(ratios))
        assert np.mean(sorted_spread) < 0.1 * np.mean(shuffled_spread)

    def test_redrawn_every_epoch(self, source):
        sampler = _sampler(source, sorted_batch_ratio=0.5)
        first = [b[0][4] for b in sampler.batchs_in_one_epoch]
        list(iter(sampler))
        second = [b[0][4] for b in sampler.batchs_in_one_epoch]
        assert first != second

    def test_two_ranks_cover_dataset_once(self, source, monkeypatch):
        import paddle.distributed as dist

        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        ids, sorted_flags = [], []
        for rank in (0, 1):
            monkeypatch.setattr(dist, "get_rank", lambda r=rank: r)
            random.seed(100 + rank)  # batch_list order differs between ranks
            sampler = _sampler(source, sorted_batch_ratio=0.5)
            batches = sampler.batchs_in_one_epoch
            ids += [_file_ids(source, b) for b in batches]
            sorted_flags.append(sorted(b[0][4] for b in batches))
        ids = np.concatenate(ids)
        assert sorted(ids.tolist()) == list(range(len(source)))
        assert sorted_flags[0] == sorted_flags[1]

    @pytest.mark.parametrize("ratio", [0.01, 0.99])
    def test_tiny_dataset_keeps_batch_count(self, ratio):
        source = FakeSource([1.0, 5.0, 2.0, 9.0, 3.0])
        sampler = _sampler(source, first_bs=4, sorted_batch_ratio=ratio)
        for _ in range(5):
            list(iter(sampler))
            batches = sampler.batchs_in_one_epoch
            assert len(batches) == len(sampler)
            assert all(len(b) == 4 for b in batches)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_out_of_range(self, source, bad):
        with pytest.raises(AssertionError, match="sorted_batch_ratio"):
            _sampler(source, sorted_batch_ratio=bad)

    def test_requires_ds_width(self):
        with pytest.raises(AssertionError, match="ds_width"):
            _sampler(FakeSource([1.0] * 10, ds_width=False), sorted_batch_ratio=0.5)


class TestPadToLongest:
    def test_sorted_batch_uses_max_ratio(self, source):
        sampler = _sampler(source, pad_to_longest=True)
        for batch in sampler.batchs_in_one_epoch:
            ratios = source.wh_ratio[_file_ids(source, batch)]
            assert batch[0][3] == pytest.approx(min(ratios.max(), 1600 / 48))

    def test_capped_by_max_w(self):
        source = FakeSource([2.0, 3.0, 30.0, 30.0])
        batch = _sampler(source, pad_to_longest=True, max_w=960).batchs_in_one_epoch[0]
        assert batch[0][3] == pytest.approx(960 / 48)

    def test_equal_ratios(self):
        source = FakeSource([5.0] * 4)
        batch = _sampler(source, pad_to_longest=True).batchs_in_one_epoch[0]
        assert batch[0][3] == pytest.approx(5.0)

    def test_shuffled_batches_get_own_width(self, source):
        sampler = _sampler(source, sorted_batch_ratio=0.0, pad_to_longest=True)
        for batch in sampler.batchs_in_one_epoch:
            ratios = source.wh_ratio[_file_ids(source, batch)]
            assert batch[0][4] is False
            assert batch[0][3] == pytest.approx(min(ratios.max(), 1600 / 48))


class TestEvalMode:
    def test_eval_is_sorted_and_not_split(self, source, monkeypatch):
        import paddle.distributed as dist

        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        sampler = _sampler(source, is_training=False, pad_to_longest=True)
        ids = np.concatenate(
            [_file_ids(source, b) for b in sampler.batchs_in_one_epoch]
        )
        assert sorted(ids.tolist()) == list(range(len(source)))
        assert np.all(np.diff(source.wh_ratio[ids]) >= 0)

    def test_eval_last_batch_not_padded(self):
        source = FakeSource(np.random.uniform(0.5, 40, size=10))
        sampler = _sampler(source, is_training=False)
        sizes = [len(b) for b in sampler.batchs_in_one_epoch]
        assert sizes == [4, 4, 2]
        ids = np.concatenate([_file_ids(source, b) for b in iter(sampler)])
        assert sorted(ids.tolist()) == list(range(10))

    def test_eval_ignores_sorted_batch_ratio(self, source):
        sampler = _sampler(source, is_training=False, sorted_batch_ratio=0.3)
        assert all(len(item) == 4 for b in sampler.batchs_in_one_epoch for item in b)

    def test_eval_order_is_fixed(self, source):
        sampler = _sampler(source, is_training=False)
        first = [b[0][2] for b in iter(sampler)]
        second = [b[0][2] for b in iter(sampler)]
        assert first == second
        assert first == sorted(first)


# ---------------------------------------------------------------------------
# MultiScaleDataSet / build_dataloader on synthetic images
# ---------------------------------------------------------------------------

SIZES = [(48, 40), (48, 100), (48, 200), (48, 480), (48, 960), (48, 1500)]


@pytest.fixture
def label_file(tmp_path):
    lines = []
    for i, (h, w) in enumerate(SIZES):
        name = f"img_{i}.png"
        cv2.imwrite(str(tmp_path / name), np.full((h, w, 3), 200, dtype=np.uint8))
        lines.append(f"{name}\ttext{i}\t{w}\t{h}")
    path = tmp_path / "label_w_h.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _config(label_file, mode="Eval", sampler=None):
    section = {
        "dataset": {
            "name": "MultiScaleDataSet",
            "ds_width": True,
            "data_dir": str(label_file.parent),
            "label_file_list": [str(label_file)],
            "transforms": [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"KeepKeys": {"keep_keys": ["image", "valid_ratio"]}},
            ],
        },
        "loader": {
            "shuffle": False,
            "drop_last": False,
            "batch_size_per_card": 2,
            "num_workers": 0,
        },
    }
    if sampler is not None:
        section["sampler"] = sampler
    return {"Global": {}, mode: section}


class TestMultiScaleDataSet:
    @pytest.fixture
    def dataset(self, label_file):
        return MultiScaleDataSet(_config(label_file), "Eval", LOGGER)

    def test_batch_shape_passed_to_transforms(self, dataset, monkeypatch):
        import ppocr.data.simple_dataset as sd

        seen = []
        real_transform = sd.transform

        def spy(data, ops=None):
            seen.append(data.get("batch_shape"))
            return real_transform(data, ops)

        monkeypatch.setattr(sd, "transform", spy)
        dataset[(320, 48, 1, 10.0, False)]
        dataset[(320, 32, 0, None)]
        assert seen[0] == (48, 480)
        assert (32, 320) in seen

    def test_legacy_four_tuple_uses_sorted_index(self, dataset):
        # position 5 in ratio order is the widest image (1500 / 48 -> 31)
        image, _ = dataset[(320, 48, 5, 31.25)]
        assert image.shape == (3, 48, 48 * 31)

    def test_legacy_tuple_without_ratio(self, dataset):
        image, _ = dataset[(320, 48, 0, None)]
        assert image.shape == (3, 48, 320)

    def test_five_tuple_shuffled_index(self, dataset):
        # index 1 in data order is the 100 px image, padded to ratio 10
        image, valid_ratio = dataset[(320, 48, 1, 10.0, False)]
        assert image.shape == (3, 48, 480)
        assert valid_ratio == pytest.approx(100 / 480, abs=0.01)

    def test_five_tuple_sorted_index(self, dataset):
        image, valid_ratio = dataset[(320, 48, 5, 31.25, True)]
        assert image.shape == (3, 48, 48 * 31)
        assert valid_ratio == pytest.approx(1.0)


def test_build_dataloader_eval_with_sampler(label_file):
    sampler = {
        "name": "MultiScaleSampler",
        "scales": [[320, 48]],
        "first_bs": 2,
        "fix_bs": True,
        "is_training": False,
        "max_w": 2400,
        "pad_to_longest": True,
    }
    loader = build_dataloader(
        _config(label_file, sampler=sampler), "Eval", "cpu", LOGGER
    )
    widths = [batch[0].shape[3] for batch in loader]
    assert len(widths) == 3
    # ratio-sorted batches in a fixed order, each as wide as its longest image
    assert widths == [48 * 2, 48 * 10, 48 * 31]


# ---------------------------------------------------------------------------
# ratio_list (per-epoch subsampling of label files) with MultiScaleDataSet
# ---------------------------------------------------------------------------


@pytest.fixture
def two_files(tmp_path):
    """File A: 40 narrow lines (w/h 1-10), file B: 40 wide lines (w/h 20-40)."""
    rng = random.Random(0)
    paths = {}
    for name, (lo, hi) in {"A": (1, 10), "B": (20, 40)}.items():
        lines = []
        for i in range(40):
            w, h = int(rng.uniform(lo, hi) * 16), 16
            fn = "{}_{}.png".format(name, i)
            cv2.imwrite(str(tmp_path / fn), np.full((h, w, 3), 255, np.uint8))
            lines.append("{}\t{}_{}\t{}\t{}".format(fn, name, i, w, h))
        path = tmp_path / "{}.txt".format(name)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[name] = str(path)
    return tmp_path, paths


def _ratio_config(two_files, ratio_list, ds_width, name="MultiScaleDataSet"):
    data_dir, paths = two_files
    return {
        "Global": {},
        "Train": {
            "dataset": {
                "name": name,
                "ds_width": ds_width,
                "data_dir": str(data_dir),
                "label_file_list": [paths["A"], paths["B"]],
                "ratio_list": ratio_list,
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"KeepKeys": {"keep_keys": ["label"]}},
                ],
            },
            "loader": {
                "shuffle": True,
                "drop_last": False,
                "batch_size_per_card": 8,
                "num_workers": 0,
            },
        },
    }


def _epoch_labels(ds, sampler):
    return [ds[item][0] for batch in sampler for item in batch]


def _by_file(labels):
    return {f: sum(l.startswith(f) for l in labels) for f in ("A", "B")}


class TestRatioList:
    @pytest.mark.parametrize("ds_width", [False, True])
    @pytest.mark.parametrize(
        "ratio_list,expected", [([1.0, 0.5], (40, 20)), ([0.5, 1.0], (20, 40))]
    )
    def test_same_sample_as_simple_dataset(
        self, two_files, ds_width, ratio_list, expected
    ):
        ms = MultiScaleDataSet(
            _ratio_config(two_files, ratio_list, ds_width), "Train", LOGGER, seed=0
        )
        simple = SimpleDataSet(
            _ratio_config(two_files, ratio_list, False, "SimpleDataSet"),
            "Train",
            LOGGER,
            seed=0,
        )
        sampler = MultiScaleSampler(
            ms, scales=[[320, 16]], first_bs=4, fix_bs=True, max_w=2000, seed=0
        )
        labels = _epoch_labels(ms, sampler)
        assert _by_file(labels) == {"A": expected[0], "B": expected[1]}
        reference = {simple[i][0] for i in range(len(simple))}
        assert set(labels) == reference

    @pytest.mark.parametrize("ds_width", [False, True])
    def test_sample_changes_every_epoch(self, two_files, ds_width):
        ds = MultiScaleDataSet(
            _ratio_config(two_files, [1.0, 0.5], ds_width), "Train", LOGGER, seed=0
        )
        sampler = MultiScaleSampler(
            ds, scales=[[320, 16]], first_bs=4, fix_bs=True, max_w=2000, seed=0
        )
        first = {l for l in _epoch_labels(ds, sampler) if l.startswith("B")}
        ds.reset_data_lines(seed=1, epoch=1)  # what tools/program.py does
        second = {l for l in _epoch_labels(ds, sampler) if l.startswith("B")}
        assert len(first) == len(second) == 20
        assert first != second

    def test_sorted_batches_follow_epoch_sample(self, two_files):
        ds = MultiScaleDataSet(
            _ratio_config(two_files, [0.5, 0.5], True), "Train", LOGGER, seed=0
        )
        sampler = MultiScaleSampler(
            ds, scales=[[320, 16]], first_bs=4, fix_bs=True, max_w=2000, seed=0
        )
        ds.reset_data_lines(seed=3, epoch=3)
        sampled = {
            ds._all_lines[g].decode("utf-8").split("\t")[1] for g in ds._index_map
        }
        seen = []
        for batch in sampler:
            assert all(item[4] if len(item) > 4 else True for item in batch)
            ratios = [ds.wh_ratio[ds.wh_ratio_sort[item[2]]] for item in batch]
            assert ratios == sorted(ratios)
            seen += [ds[item][0] for item in batch]
        assert set(seen) == sampled

    def test_worker_copy_serves_same_lines(self, two_files):
        main = MultiScaleDataSet(
            _ratio_config(two_files, [1.0, 0.5], True), "Train", LOGGER, seed=0
        )
        worker = MultiScaleDataSet(
            _ratio_config(two_files, [1.0, 0.5], True), "Train", LOGGER, seed=0
        )
        main.reset_data_lines(seed=2, epoch=2)
        worker._shared_epoch.value = 2  # a worker only sees the shared epoch
        for pos in range(len(main)):
            item = (320, 16, pos, 10.0, True)
            assert worker[item][0] == main[item][0]

    def test_without_ratio_list_unchanged(self, two_files):
        ds = MultiScaleDataSet(
            _ratio_config(two_files, [1.0, 1.0], True), "Train", LOGGER, seed=0
        )
        assert ds._index_map is None and len(ds) == 80
        assert ds.wh_version == 1
        np.testing.assert_array_equal(ds.wh_ratio_sort, np.argsort(ds.wh_ratio))
