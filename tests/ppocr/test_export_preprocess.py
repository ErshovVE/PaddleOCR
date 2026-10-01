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

import pytest

from ppocr.utils.export_model import infer_preprocess_ops

DECODE = {"DecodeImage": {"img_mode": "BGR", "channel_first": False}}
ENCODE = {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}}
KEEP = {"KeepKeys": {"keep_keys": ["image", "label_ctc"]}}


def _config(transforms, sampler=None, model_type="rec"):
    eval_cfg = {"dataset": {"transforms": transforms}}
    if sampler is not None:
        eval_cfg["sampler"] = sampler
    return {"Architecture": {"model_type": model_type}, "Eval": eval_cfg}


def test_sampler_eval_gets_resize_before_keep_keys():
    config = _config(
        [DECODE, ENCODE, KEEP],
        sampler={"name": "MultiScaleSampler", "scales": [[320, 48]]},
    )
    ops = infer_preprocess_ops(config)
    assert ops == [
        DECODE,
        ENCODE,
        {"RecResizeImg": {"image_shape": [3, 48, 320]}},
        KEEP,
    ]
    # the training config itself is not modified
    assert len(config["Eval"]["dataset"]["transforms"]) == 3


def test_existing_resize_kept():
    resize = {"RecResizeImg": {"image_shape": [3, 48, 320]}}
    config = _config(
        [DECODE, ENCODE, resize, KEEP],
        sampler={"name": "MultiScaleSampler", "scales": [[320, 32]]},
    )
    assert infer_preprocess_ops(config) == [DECODE, ENCODE, resize, KEEP]


@pytest.mark.parametrize(
    "config",
    [
        _config([DECODE, ENCODE, KEEP]),  # no sampler
        _config([DECODE, KEEP], sampler={"scales": [[640, 640]]}, model_type="det"),
    ],
)
def test_unchanged_without_rec_sampler(config):
    assert infer_preprocess_ops(config) == config["Eval"]["dataset"]["transforms"]


def test_int_scale():
    config = _config([DECODE, KEEP], sampler={"scales": [48]})
    assert infer_preprocess_ops(config)[1] == {
        "RecResizeImg": {"image_shape": [3, 48, 48]}
    }
