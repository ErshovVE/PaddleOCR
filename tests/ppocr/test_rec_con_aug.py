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

import math
import os
import random

import numpy as np
import pytest

from ppocr.data.imaug.rec_img_aug import (
    RecAug,
    RecConAug,
    _ink_mask,
    _ink_threshold,
    _n_components,
    change_stroke,
    downscale_upscale,
    jpeg_compress,
    max_rotation_deg,
    rotate_text,
)
from ppocr.data.imaug.text_image_aug import tia_distort

np.random.seed(42)
random.seed(42)

NO_BDA = dict(
    crop_prob=0,
    reverse_prob=0,
    noise_prob=0,
    jitter_prob=0,
    blur_prob=0,
    hsv_aug_prob=0,
)


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



class FakeLazy:
    """Lazy extra sample like SimpleDataSet.get_ext_data hands to RecConAug."""

    def __init__(self, label, w, wh_ratio="image", readable=True):
        self.label = label
        self.w = w
        self.wh_ratio = w / 48 if wh_ratio == "image" else wh_ratio
        self.readable = readable
        self.loads = 0

    def load(self):
        self.loads += 1
        if not self.readable:
            return None
        return {"image": _text_image(self.w), "label": self.label}


class TestRecConAugLazy:
    def test_glues_lazy_like_eager(self):
        lazy = _data("a", 100)
        lazy["ext_data"] = [FakeLazy("b", 60)]
        out = _aug()(lazy)
        assert out["label"] == "ab"
        assert out["image"].shape == (48, 160, 3)
        assert "ext_data" not in out

    def test_rejected_by_label_never_loaded(self):
        too_long, too_many_steps = FakeLazy("x" * 60, 40), FakeLazy("aaaa", 40)
        data = _batch_data("a", 40, [], batch_w=48)
        data["ext_data"] = [too_long, too_many_steps]
        aug = _aug(fit_batch_width=True, ctc_stride=8, skip_unfit=True)
        out = aug(data)
        assert out["label"] == "a"
        assert too_long.loads == 0 and too_many_steps.loads == 0
        assert aug.stats["reject_length"] == 1 and aug.stats["reject_ctc"] == 1

    def test_too_wide_by_label_never_loaded(self):
        wide, fits = FakeLazy("b", 900), FakeLazy("c", 40)
        data = _batch_data("a", 40, [], batch_w=192)
        data["ext_data"] = [wide, fits]
        aug = _aug(fit_batch_width=True, skip_unfit=True)
        out = aug(data)
        assert out["label"] == "ac"
        assert wide.loads == 0 and fits.loads == 1
        assert aug.stats["reject_width"] == 1

    def test_unknown_ratio_checked_on_image(self):
        wide = FakeLazy("b", 900, wh_ratio=None)
        data = _batch_data("a", 40, [], batch_w=192)
        data["ext_data"] = [wide]
        aug = _aug(fit_batch_width=True, skip_unfit=True)
        assert aug(data)["label"] == "a"
        assert wide.loads == 1 and aug.stats["reject_width"] == 1

    def test_unreadable_is_skipped(self):
        data = _data("a", 100)
        data["ext_data"] = [FakeLazy("b", 60, readable=False), FakeLazy("c", 60)]
        assert _aug(skip_unfit=True)(data)["label"] == "ac"

    def test_prob_zero_loads_nothing(self):
        ext = FakeLazy("b", 60)
        data = _data("a", 100)
        data["ext_data"] = [ext]
        assert _aug(prob=0.0)(data)["label"] == "a"
        assert ext.loads == 0


class TestRotation:
    @pytest.mark.parametrize(
        "ratio,expected", [(1, 3.0), (3, 3.0), (30, 0.6), (29, 0.6144)]
    )
    def test_anchor_points(self, ratio, expected):
        assert max_rotation_deg(ratio) == pytest.approx(expected, abs=1e-3)

    def test_smooth_and_monotone(self):
        ratios = np.linspace(1, 100, 5000)
        limits = np.array([max_rotation_deg(r) for r in ratios])
        assert np.all(np.diff(limits) <= 0)
        # continuous at the first anchor and never steeper than 1 degree per w/h unit
        assert max_rotation_deg(3 - 1e-9) == pytest.approx(max_rotation_deg(3 + 1e-9))
        assert np.max(-np.diff(limits) / np.diff(ratios)) < 1.0

    def test_custom_anchors(self):
        assert max_rotation_deg(10, (2, 20), (4, 1)) == pytest.approx(
            4 * (2 / 10) ** (np.log(4) / np.log(10))
        )

    def test_grows_height_only(self):
        img = _text_image(1500)
        out = rotate_text(img, 0.6)
        expected_h = math.ceil(
            48 * math.cos(math.radians(0.6)) + 1500 * math.sin(math.radians(0.6))
        )
        assert out.shape == (expected_h, 1500, 3)
        assert rotate_text(img, 0.0).shape == img.shape

    def test_tight_crop_nothing_cut(self):
        img = np.full((30, 900, 3), 255, np.uint8)
        for x in range(20, 880, 8):
            img[:, x : x + 3] = 0  # letter strokes touch the top and bottom edges
        for angle in (0.6, -0.6):
            out = rotate_text(img, angle)
            assert (out < 128).sum() >= 0.97 * (img < 128).sum()
            # new margins are background, not smeared ink
            assert (out[0, :, 0] > 128).mean() > 0.4

    @pytest.mark.parametrize("width", [96, 480, 1440])
    def test_rec_aug_angle_within_limit(self, width, monkeypatch):
        import ppocr.data.imaug.rec_img_aug as rec_aug

        angles = []
        real = rec_aug.rotate_text
        monkeypatch.setattr(
            rec_aug,
            "rotate_text",
            lambda img, angle: angles.append(angle) or real(img, angle),
        )
        aug = RecAug(**NO_BDA, tia_prob=0, rotate_prob=1.0)
        for _ in range(200):
            out = aug({"image": _text_image(width)})
            assert out["image"].shape[1:] == (width, 3)
        limit = max_rotation_deg(width / 48)
        assert max(abs(a) for a in angles) <= limit
        assert max(abs(a) for a in angles) > 0.8 * limit  # the range is used

    def test_disabled_by_default(self, monkeypatch):
        import ppocr.data.imaug.rec_img_aug as rec_aug

        monkeypatch.setattr(rec_aug, "rotate_text", lambda *a: pytest.fail("rotated"))
        random.seed(5)
        RecAug(**NO_BDA, tia_prob=0)({"image": _text_image(100)})
        after_default = random.random()
        random.seed(5)
        RecAug(**NO_BDA, tia_prob=0, rotate_prob=0.0)({"image": _text_image(100)})
        assert random.random() == after_default

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rotate_ratios": (30, 3)},
            {"rotate_degs": (0.6, 3)},
            {"rotate_degs": (3, 0)},
        ],
    )
    def test_invalid_params(self, kwargs):
        with pytest.raises(AssertionError, match="rotate_"):
            RecAug(**kwargs)


def _font_line(size, font="arial.ttf", bold=False):
    from PIL import Image, ImageDraw, ImageFont

    candidates = [
        r"C:\Windows\Fonts\%s" % ("arialbd.ttf" if bold else font),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        pytest.skip("no TrueType font")
    f = ImageFont.truetype(path, size)
    text = "Документ ГОСТ 1.13-85 тонкий"
    im = Image.new("RGB", (int(f.getlength(text)) + 10, int(size * 1.4)), "white")
    ImageDraw.Draw(im).text((5, 2), text, fill="black", font=f)
    return np.array(im)[:, :, ::-1].copy()


def _ink(img):
    t, light = _ink_threshold(img)
    return _ink_mask(img, t, light)


class TestStroke:
    def test_thicken_adds_ink(self):
        img = _font_line(30)
        out = change_stroke(img, thicken=True, alpha=0.6)
        assert out.shape == img.shape and out.dtype == img.dtype
        assert _ink(out).sum() > _ink(img).sum()

    def test_thin_large_bold_text_keeps_strokes(self):
        img = _font_line(30, bold=True)
        out = change_stroke(img, thicken=False, alpha=0.6)
        assert out is not img
        t, light = _ink_threshold(img)
        before, after = _ink_mask(img, t, light), _ink_mask(out, t, light)
        assert 0.6 * before.sum() <= after.sum() < before.sum()
        assert _n_components(after) <= _n_components(before)

    def test_thin_small_text_left_alone(self):
        img = _font_line(12)
        assert change_stroke(img, thicken=False, alpha=0.6) is img

    def test_one_pixel_strokes_never_thinned(self):
        img = np.full((40, 200, 3), 255, np.uint8)
        img[20, 10:190] = 0  # 1 px line
        img[5:35, 100] = 0
        assert change_stroke(img, thicken=False, alpha=1.0) is img

    def test_thicken_that_glues_letters_is_undone(self):
        img = np.full((30, 200, 3), 255, np.uint8)
        for x in range(10, 190, 3):  # 2 px bars, 1 px gaps
            img[5:25, x : x + 2] = 0
        assert change_stroke(img, thicken=True, alpha=1.0) is img

    def test_light_text_on_dark_background(self):
        img = 255 - _font_line(30)
        out = change_stroke(img, thicken=True, alpha=0.6)
        assert (out > 128).sum() > (img > 128).sum()  # light strokes grew

    def test_blank_image(self):
        img = np.full((30, 100, 3), 200, np.uint8)
        assert change_stroke(img, thicken=False, alpha=0.5) is img


class TestScanArtifacts:
    def test_downscale_keeps_shape(self):
        img = _text_image(500)
        out = downscale_upscale(img, 0.5)
        assert out.shape == img.shape and out.dtype == img.dtype

    def test_jpeg_keeps_shape_and_changes_pixels(self):
        img = _font_line(20)
        out = jpeg_compress(img, 20)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - img).mean() > 0

    def test_downscale_min_height(self, monkeypatch):
        import ppocr.data.imaug.rec_img_aug as rec_aug

        scales = []
        real = rec_aug.downscale_upscale
        monkeypatch.setattr(
            rec_aug,
            "downscale_upscale",
            lambda img, s: scales.append(s) or real(img, s),
        )
        aug = RecAug(**NO_BDA, tia_prob=0, downscale_prob=1.0)
        for _ in range(100):
            aug({"image": _text_image(300, h=28)})
        assert min(scales) * 28 >= 14 - 1e-6
        aug({"image": _text_image(300, h=12)})  # too small to shrink
        assert len(scales) == 100

    def test_jpeg_quality_range(self, monkeypatch):
        import ppocr.data.imaug.rec_img_aug as rec_aug

        qualities = []
        monkeypatch.setattr(
            rec_aug, "jpeg_compress", lambda img, q: qualities.append(q) or img
        )
        aug = RecAug(**NO_BDA, tia_prob=0, jpeg_prob=1.0, jpeg_quality=(30, 40))
        for _ in range(100):
            aug({"image": _text_image(100)})
        assert min(qualities) >= 30 and max(qualities) <= 40

    def test_all_new_options_keep_shape(self):
        aug = RecAug(
            **NO_BDA,
            tia_prob=0,
            rotate_prob=1.0,
            downscale_prob=1.0,
            jpeg_prob=1.0,
            stroke_prob=1.0,
        )
        img = _font_line(30)
        for _ in range(20):
            out = aug({"image": img.copy()})["image"]
            assert out.shape[1:] == img.shape[1:] and out.shape[0] >= img.shape[0]

    def test_defaults_draw_no_random_numbers(self):
        random.seed(3)
        RecAug(**NO_BDA, tia_prob=0)({"image": _text_image(100)})
        expected = random.random()
        random.seed(3)
        RecAug(**NO_BDA, tia_prob=0, downscale_prob=0, jpeg_prob=0, stroke_prob=0)(
            {"image": _text_image(100)}
        )
        assert random.random() == expected

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"downscale_range": (0, 0.5)},
            {"jpeg_quality": (80, 20)},
            {"stroke_alpha": (0.5, 1.5)},
            {"stroke_thin_share": 2},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(AssertionError):
            RecAug(**kwargs)
