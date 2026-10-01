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

import random

import numpy as np
import pytest

from ppocr.data.imaug.rec_img_aug import RecAug, RecConAug
from ppocr.data.imaug.text_image_aug import tia_distort

np.random.seed(42)
random.seed(42)


def _text_image(w, h=48, bg=255, fg=0):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    img[h // 4 : -h // 4, 4:-4] = fg  # "ink" away from the edges
    return img


def _data(label, w, ext=(), bg=255, fg=0):
    return {
        "image": _text_image(w, bg=bg, fg=fg),
        "label": label,
        "ext_data": [
            {"image": _text_image(ew, bg=bg, fg=fg), "label": el} for el, ew in ext
        ],
    }


def _aug(**kwargs):
    params = dict(prob=1.0, image_shape=(48, 2000, 3), max_text_length=50)
    params.update(kwargs)
    return RecConAug(**params)


class TestRecConAugDefault:
    def test_glues_without_space(self):
        out = _aug()(_data("a", 100, [("b", 60)]))
        assert out["label"] == "ab"
        assert out["image"].shape == (48, 160, 3)
        assert "ext_data" not in out

    def test_prob_zero_returns_input(self):
        out = _aug(prob=0.0)(_data("a", 100, [("b", 60)]))
        assert out["label"] == "a"
        assert out["image"].shape == (48, 100, 3)

    def test_length_limit(self):
        out = _aug(max_text_length=4)(_data("ab", 100, [("abc", 60)]))
        assert out["label"] == "ab"

    def test_ratio_limit(self):
        out = _aug(image_shape=(48, 320, 3))(_data("a", 200, [("b", 200)]))
        assert out["label"] == "a"


class TestRecConAugSpace:
    def test_space_and_gap(self):
        out = _aug(add_space=True, space_width_range=(16, 80))(
            _data("a", 100, [("b", 60)])
        )
        assert out["label"] == "a b"
        gap = out["image"].shape[1] - 160
        assert 16 <= gap <= 80

    def test_several_parts(self):
        out = _aug(add_space=True)(_data("a", 100, [("b", 60), ("c", 60)]))
        assert out["label"] == "a b c"

    def test_gap_matches_light_background(self):
        out = _aug(add_space=True, space_width_range=(30, 30))(
            _data("a", 100, [("b", 60)])
        )
        gap = out["image"][:, 100:130]
        assert gap.dtype == np.uint8
        assert np.all(gap == 255)

    def test_gap_matches_dark_background(self):
        out = _aug(add_space=True, space_width_range=(30, 30))(
            _data("a", 100, [("b", 60)], bg=10, fg=240)
        )
        gap = out["image"][:, 100:130]
        assert np.all(gap == 10)

    def test_length_limit_counts_space(self):
        out = _aug(add_space=True, max_text_length=5)(_data("ab", 100, [("abc", 60)]))
        assert out["label"] == "ab"

    def test_gap_counts_in_ratio(self):
        # 100 + 60 px fit 170 px, but not with a 16+ px gap
        out = _aug(add_space=True, image_shape=(48, 170, 3))(
            _data("a", 100, [("b", 60)])
        )
        assert out["label"] == "a"

    def test_ratio_jitter_varies_length(self):
        aug = _aug(max_ratio_jitter=True, image_shape=(48, 960, 3))
        widths = {
            aug(_data("a", 48, [("b", 48)] * 19))["image"].shape[1] for _ in range(30)
        }
        assert len(widths) > 3
        assert max(widths) <= 960

    def test_ext_aug_applied(self):
        aug = _aug(
            ext_aug=dict(
                tia_prob=0,
                crop_prob=0,
                reverse_prob=1.0,
                noise_prob=0,
                jitter_prob=0,
                blur_prob=0,
                hsv_aug_prob=0,
            )
        )
        out = aug(_data("a", 100, [("b", 60)]))
        assert out["image"][0, 0, 0] == 255  # base image untouched
        assert out["image"][0, 110, 0] == 0  # attached image inverted

    def test_bad_space_range(self):
        with pytest.raises(AssertionError, match="space_width_range"):
            _aug(space_width_range=(80, 16))


class TestTiaVerticalCap:
    def test_long_line_keeps_shape(self):
        img = _text_image(2000)
        out = tia_distort(img, 4, max_vertical_shift=12)
        assert out.shape == img.shape

    def test_zero_cap_does_not_fail(self):
        img = _text_image(2000)
        assert tia_distort(img, 4, max_vertical_shift=0).shape == img.shape

    def test_cap_bounds_vertical_shift(self, monkeypatch):
        captured = {}
        import ppocr.data.imaug.text_image_aug.augment as augment

        class FakeWarp:
            def __init__(self, src, src_pts, dst_pts, w, h):
                captured["shift"] = np.array(dst_pts) - np.array(src_pts)
                self.src = src

            def generate(self):
                return self.src

        monkeypatch.setattr(augment, "WarpMLS", FakeWarp)
        tia_distort(_text_image(2000), 4, max_vertical_shift=12)
        assert np.abs(captured["shift"][:, 1]).max() <= 12
        assert np.abs(captured["shift"][:, 0]).max() > 12  # horizontal untouched

    def test_rec_aug_with_cap(self):
        aug = RecAug(
            tia_prob=1.0,
            crop_prob=0,
            reverse_prob=0,
            noise_prob=0,
            jitter_prob=0,
            blur_prob=0,
            hsv_aug_prob=0,
            tia_max_vertical_ratio=0.25,
        )
        out = aug({"image": _text_image(2000)})
        assert out["image"].shape == (48, 2000, 3)


def _batch_data(label, w, ext, batch_w, batch_h=48):
    data = _data(label, w, ext)
    data["batch_shape"] = (batch_h, batch_w)
    return data


class TestRecConAugFitBatch:
    def test_without_flag_ignores_batch_shape(self):
        out = _aug()(_batch_data("a", 96, [("b", 480)], batch_w=96))
        assert out["label"] == "ab"
        assert out["image"].shape[1] == 576

    def test_narrow_batch_blocks_wide_glue(self):
        out = _aug(fit_batch_width=True)(_batch_data("a", 90, [("b", 480)], batch_w=96))
        assert out["label"] == "a"
        assert out["image"].shape[1] == 90

    def test_wide_batch_allows_glue(self):
        out = _aug(fit_batch_width=True)(
            _batch_data("a", 96, [("b", 480)], batch_w=1440)
        )
        assert out["label"] == "ab"
        assert out["image"].shape[1] <= 1440

    def test_result_never_wider_than_batch(self):
        random.seed(0)
        aug = _aug(fit_batch_width=True, add_space=True, skip_unfit=True)
        for batch_w in (96, 192, 480, 1440):
            for _ in range(20):
                ext = [("w", random.choice([40, 96, 300, 900])) for _ in range(10)]
                out = aug(_batch_data("a", 60, ext, batch_w=batch_w))
                assert out["image"].shape[1] <= batch_w

    def test_ratio_scaled_by_batch_height(self):
        # batch 32 x 320 -> w/h 10 -> 480 px at the aug height 48
        out = _aug(fit_batch_width=True)(
            _batch_data("a", 200, [("b", 200), ("c", 200)], batch_w=320, batch_h=32)
        )
        assert out["label"] == "ab"

    def test_ctc_steps_limit(self):
        # 400 px batch -> 50 CTC steps; 30 + 1 + 30 distinct-neighbour chars need 61
        aug = _aug(fit_batch_width=True, ctc_stride=8, add_space=True)
        out = aug(_batch_data("ab" * 15, 100, [("cd" * 15, 100)], batch_w=400))
        assert out["label"] == "ab" * 15
        out = aug(_batch_data("ab" * 10, 100, [("cd" * 10, 100)], batch_w=400))
        assert out["label"] == "ab" * 10 + " " + "cd" * 10

    def test_ctc_min_steps_counts_repeats(self):
        assert RecConAug.ctc_min_steps("abc") == 3
        assert RecConAug.ctc_min_steps("aab") == 4
        assert RecConAug.ctc_min_steps("") == 0

    def test_skip_unfit_tries_next(self):
        ext = [("long", 900), ("b", 40), ("c", 40)]
        stop = _aug(fit_batch_width=True)(_batch_data("a", 40, ext, batch_w=192))
        assert stop["label"] == "a"
        skip = _aug(fit_batch_width=True, skip_unfit=True)(
            _batch_data("a", 40, ext, batch_w=192)
        )
        assert skip["label"] == "abc"

    def test_skip_unfit_on_text_length(self):
        out = _aug(max_text_length=4, skip_unfit=True)(
            _data("ab", 40, [("abcdef", 40), ("cd", 40)])
        )
        assert out["label"] == "abcd"

    def test_stats_logged(self, monkeypatch):
        import ppocr.utils.logging as plog

        messages = []

        class FakeLogger:
            def info(self, msg):
                messages.append(msg)

        monkeypatch.setattr(plog, "get_logger", lambda *a, **k: FakeLogger())
        aug = _aug(log_every=3, fit_batch_width=True, skip_unfit=True)
        for batch_w in (1440, 96, 1440):
            aug(_batch_data("a", 40, [("b", 300)], batch_w=batch_w))
        aug(_batch_data("a", 40, [("b", 300)], batch_w=1440))
        assert len(messages) == 1
        assert "66.7% of 3 samples" in messages[0]
        assert "width 1" in messages[0]
        assert aug.stats["calls"] == 1
