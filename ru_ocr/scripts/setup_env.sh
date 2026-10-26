#!/usr/bin/env bash
# Training environment for ru_ocr on a Linux machine with NVIDIA GPUs (driver for CUDA 12.6 or newer).
# Run from the repository root, in the Python environment training will use:
#
#   bash ru_ocr/scripts/setup_env.sh
#
# Installs paddlepaddle-gpu (retried: the Paddle CDN cuts large downloads now and then), the repository
# requirements, and the pretrained PP-OCRv6 small weights into ./pretrain/, then checks Paddle on the GPUs.
# Override with environment variables: PADDLE_PKG, PADDLE_INDEX, PYTHON.
set -euo pipefail

PYTHON=${PYTHON:-python}
PADDLE_PKG=${PADDLE_PKG:-paddlepaddle-gpu==3.3.1}
PADDLE_INDEX=${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}
PRETRAINED_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams

if [ ! -f tools/train.py ] || [ ! -d ru_ocr ]; then
    echo "run from the repository root" >&2
    exit 1
fi

for attempt in 1 2 3 4 5; do
    if "$PYTHON" -m pip install --progress-bar off --timeout 120 --retries 10 "$PADDLE_PKG" -i "$PADDLE_INDEX"; then
        break
    fi
    if [ "$attempt" = 5 ]; then
        echo "$PADDLE_PKG install failed" >&2
        exit 1
    fi
    echo "paddle install attempt $attempt failed, retrying" >&2
done

"$PYTHON" -m pip install --progress-bar off -r requirements.txt

mkdir -p pretrain
if [ ! -s pretrain/PP-OCRv6_small_rec_pretrained.pdparams ]; then
    wget -q -O pretrain/PP-OCRv6_small_rec_pretrained.pdparams "$PRETRAINED_URL"
fi

"$PYTHON" -c "import paddle; paddle.utils.run_check()"
