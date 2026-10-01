"""Convert a torchocr (PyTorch) text recognition model into PaddleOCR 3.x weights.

torch and paddle clash when loaded in one process on Windows (WinError 127), so the
conversion runs in two steps, each in its own process:

    # 1. needs torch only: checkpoint -> plain numpy arrays
    python ru_ocr/tools/convert_torchocr_rec.py extract --src model_20.pth --out weights.npz

    # 2. needs paddle only: arrays -> .pdparams for a PaddleOCR config, optional ONNX check
    python ru_ocr/tools/convert_torchocr_rec.py convert -c ru_ocr/configs/ru_RepSVTR_rec_150525.yml \
        --src weights.npz --out rec.pdparams --onnx rec.onnx --widths 320 960 1600

Mapping rules: BatchNorm running_mean/var -> _mean/_variance, weights of nn.Linear are
transposed (torch [out, in] -> paddle [in, out], square ones too, decided by the layer
type of the Paddle model, not by shape), num_batches_tracked is dropped, tensors whose
shape does not match are skipped and listed.

New dictionary: with --src-dict (the dictionary the torch model was trained with) the
model is checked against ONNX with that dictionary, then the output layers are
re-indexed to the config dictionary: known characters keep their trained rows, new
ones keep the initial values and are learned during fine-tuning.
"""

import argparse
import os
import pickle
import sys
import types

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ----------------------------------------------------------------------------
# extract (torch)
# ----------------------------------------------------------------------------


class _Stub:
    """Placeholder for classes whose code is not importable (e.g. torchocr modules)."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return type(name, (_Stub,), {"__module__": module})


def _tolerant_pickle_module():
    module = types.ModuleType("tolerant_pickle")
    module.Unpickler = TolerantUnpickler
    module.load = lambda f, **kw: TolerantUnpickler(f, **kw).load()
    module.__name__ = "tolerant_pickle"
    return module


def _is_tensor(value):
    return hasattr(value, "detach") and hasattr(value, "cpu")


def _to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def collect_state(obj, prefix=""):
    """Flatten a state_dict, a checkpoint dict or a pickled nn.Module into {name: array}."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net"):
            if key in obj and not _is_tensor(obj[key]):
                return collect_state(obj[key], prefix)
        state = {}
        for key, value in obj.items():
            if _is_tensor(value):
                state[prefix + str(key)] = _to_numpy(value)
        return state
    attrs = getattr(obj, "__dict__", {})
    if "_state" in attrs and isinstance(attrs["_state"], dict):
        attrs = attrs["_state"]
    state = {}
    for group in ("_parameters", "_buffers"):
        for key, value in (attrs.get(group) or {}).items():
            if _is_tensor(value):
                state[prefix + key] = _to_numpy(value)
    for key, child in (attrs.get("_modules") or {}).items():
        if child is not None:
            state.update(collect_state(child, prefix + key + "."))
    return state


def extract(src, out):
    import torch

    # weights_only=False runs pickled code: only use on trusted checkpoints
    obj = torch.load(
        src,
        map_location="cpu",
        weights_only=False,
        pickle_module=_tolerant_pickle_module(),
    )
    state = {
        k: v
        for k, v in collect_state(obj).items()
        if not k.endswith("num_batches_tracked")
    }
    if not state:
        raise SystemExit("no tensors found in {}".format(src))
    np.savez(out, **state)
    print("extracted {} tensors -> {}".format(len(state), out))
    return state


# ----------------------------------------------------------------------------
# convert (paddle)
# ----------------------------------------------------------------------------


def torch_key_to_paddle(key):
    return key.replace("running_mean", "_mean").replace("running_var", "_variance")


def map_state_dict(torch_state, target_shapes, linear_keys):
    """Map torch arrays onto Paddle parameter names.

    Args:
        torch_state: {torch name: np.ndarray}
        target_shapes: {paddle name: shape} of the Paddle model
        linear_keys: Paddle names of nn.Linear weights (transposed)
    Returns:
        mapped {paddle name: array}, skipped [description], missing [paddle name]
    """
    mapped, skipped = {}, []
    for key, value in torch_state.items():
        if key.endswith("num_batches_tracked"):
            continue
        pkey = torch_key_to_paddle(key)
        if pkey not in target_shapes:
            skipped.append("{} (no such paddle parameter)".format(key))
            continue
        arr = np.asarray(value, dtype="float32")
        if pkey in linear_keys and arr.ndim == 2:
            arr = arr.T
        target = tuple(target_shapes[pkey])
        if arr.shape != target:
            skipped.append("{} shape {} vs paddle {}".format(key, arr.shape, target))
            continue
        mapped[pkey] = arr
    missing = [k for k in target_shapes if k not in mapped]
    return mapped, skipped, missing


def head_out_channels(config, char_num):
    """Set head output sizes the same way tools/train.py does for rec models."""
    head = config["Architecture"]["Head"]
    if head["name"] != "MultiHead":
        head["out_channels"] = char_num
        return config
    out_channels_list = {"CTCLabelDecode": char_num}
    second_loss = list(config["Loss"]["loss_config_list"][1].keys())[0]
    if second_loss == "NRTRLoss":
        out_channels_list["NRTRLabelDecode"] = char_num + 3
    elif second_loss == "SARLoss":
        out_channels_list["SARLabelDecode"] = char_num + 2
    head["out_channels_list"] = out_channels_list
    return config


# vocabulary-sized parameters: name suffix -> (axis, special tokens before the chars)
# CTC: [blank] + chars; NRTR: [blank, <unk>, <s>, </s>] + chars + one spare slot
VOCAB_PARAMS = {
    "ctc_head.fc.weight": (1, 1),
    "ctc_head.fc.bias": (0, 1),
    "gtc_head.embedding.embedding.weight": (0, 4),
    "gtc_head.tgt_word_prj.weight": (1, 4),
}


def vocab_index(old_chars, new_chars, n_special, size_new, size_old):
    """For every row of the new vocabulary axis: its row in the old one or -1.

    Special tokens map to themselves, characters by value, trailing slots after the
    characters (e.g. the spare NRTR slot) by their offset from the end.
    """
    old_pos = {c: i for i, c in enumerate(old_chars)}
    tail = size_new - n_special - len(new_chars)
    assert (
        tail == size_old - n_special - len(old_chars) and tail >= 0
    ), "vocabulary axis {} / {} does not match {} / {} characters".format(
        size_new, size_old, len(new_chars), len(old_chars)
    )
    index = list(range(n_special))
    index += [n_special + old_pos[c] if c in old_pos else -1 for c in new_chars]
    index += list(range(size_old - tail, size_old))
    return index


def remap_vocab(src_state, dst_state, old_chars, new_chars):
    """Copy src_state into dst_state, re-indexing vocabulary-sized parameters.

    Args:
        src_state / dst_state: {name: np.ndarray}; dst_state holds the initial
            values of the model built with the new dictionary.
        old_chars / new_chars: dictionary characters (with the space, without blank).
    Returns:
        new state dict and the list of characters that got no trained weights.
    """
    out = {}
    for key, dst in dst_state.items():
        src = src_state[key]
        spec = next((v for k, v in VOCAB_PARAMS.items() if key.endswith(k)), None)
        if spec is None or src.shape == dst.shape and old_chars == new_chars:
            out[key] = src
            continue
        axis, n_special = spec
        index = vocab_index(
            old_chars, new_chars, n_special, dst.shape[axis], src.shape[axis]
        )
        result = np.array(dst, copy=True)
        for new_i, old_i in enumerate(index):
            if old_i >= 0:
                sl_new = [slice(None)] * dst.ndim
                sl_old = [slice(None)] * src.ndim
                sl_new[axis], sl_old[axis] = new_i, old_i
                result[tuple(sl_new)] = src[tuple(sl_old)]
        out[key] = result
    added = [c for c in new_chars if c not in set(old_chars)]
    return out, added


def build_paddle_model(config_path, dict_path=None):
    import yaml

    sys.path.insert(0, REPO_ROOT)
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)
    if dict_path is not None:
        config["Global"]["character_dict_path"] = dict_path
    post_process = build_post_process(config["PostProcess"], config["Global"])
    config = head_out_channels(config, len(post_process.character))
    model = build_model(config["Architecture"])
    model.eval()
    return model, post_process.character


def linear_weight_keys(model):
    import paddle.nn as nn

    return {
        "{}.weight".format(name)
        for name, layer in model.named_sublayers()
        if isinstance(layer, nn.Linear)
    }


def ctc_greedy_decode(probs, character):
    """Greedy CTC decoding of [T, C] scores; character[0] is the blank."""
    idx = probs.argmax(-1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(character[i])
        prev = i
    return "".join(out)


def compare_with_onnx(model, onnx_path, widths, character, height=48, seed=0):
    import onnxruntime as ort
    import paddle

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    worst = 0.0
    for width in widths:
        x = rng.uniform(-1, 1, size=(1, 3, height, width)).astype("float32")
        with paddle.no_grad():
            pred = model(paddle.to_tensor(x))
        pred = (pred["ctc"] if isinstance(pred, dict) else pred).numpy()
        ref = sess.run(None, {input_name: x})[0]
        diff = float(np.abs(pred - ref).max())
        agree = float(np.mean(pred.argmax(-1) == ref.argmax(-1)))
        same = ctc_greedy_decode(pred[0], character) == ctc_greedy_decode(
            ref[0], character
        )
        print(
            "width={}: steps={} max|diff|={:.2e} argmax agree={:.1%} decoded equal={}".format(
                width, pred.shape[1], diff, agree, same
            )
        )
        worst = max(worst, diff)
    return worst


def convert(
    config_path,
    src,
    out,
    onnx_path=None,
    widths=(320,),
    tol=1e-4,
    allow_missing=False,
    src_dict=None,
):
    import paddle

    model, character = build_paddle_model(config_path, src_dict)
    target = {k: v.shape for k, v in model.state_dict().items()}
    mapped, skipped, missing = map_state_dict(
        dict(np.load(src)), target, linear_weight_keys(model)
    )
    print("paddle params={} loaded={}".format(len(target), len(mapped)))
    for line in skipped:
        print("  skipped: " + line)
    for key in missing:
        print("  not loaded (keeps init): " + key)
    if missing and not allow_missing:
        print(
            "FAIL: {} paddle parameters not loaded (use --allow-missing when "
            "the vocabulary or head differs on purpose)".format(len(missing))
        )
        return 1
    model.set_state_dict(mapped)
    if onnx_path:
        worst = compare_with_onnx(model, onnx_path, widths, character)
        if worst > tol:
            print("FAIL: max|diff| {:.2e} > {:.0e}".format(worst, tol))
            return 1
        print("OK: max|diff| {:.2e} <= {:.0e}".format(worst, tol))
    if src_dict is not None:
        target, new_character = build_paddle_model(config_path)
        state, added = remap_vocab(
            {k: v.numpy() for k, v in model.state_dict().items()},
            {k: v.numpy() for k, v in target.state_dict().items()},
            character[1:],
            new_character[1:],
        )
        removed = [c for c in character[1:] if c not in set(new_character[1:])]
        print(
            "dictionary {} -> {} chars, new (trained from scratch): {}".format(
                len(character) - 1, len(new_character) - 1, added
            )
        )
        if removed:
            print("  dropped from the dictionary: {}".format(removed))
        target.set_state_dict(state)
        model = target
    paddle.save(model.state_dict(), out)
    print("saved -> {}".format(out))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ext = sub.add_parser("extract", help="torch checkpoint -> .npz (needs torch)")
    p_ext.add_argument("--src", required=True)
    p_ext.add_argument("--out", required=True)
    p_conv = sub.add_parser("convert", help=".npz -> .pdparams (needs paddle)")
    p_conv.add_argument("-c", "--config", required=True)
    p_conv.add_argument("--src", required=True)
    p_conv.add_argument("--out", required=True)
    p_conv.add_argument("--onnx", default=None, help="reference ONNX export to compare")
    p_conv.add_argument("--widths", type=int, nargs="+", default=[320, 960, 1600])
    p_conv.add_argument("--tol", type=float, default=1e-4)
    p_conv.add_argument(
        "--src-dict",
        default=None,
        help="dictionary of the torch model if it differs from the config one",
    )
    p_conv.add_argument(
        "--allow-missing",
        action="store_true",
        help="save even if some paddle parameters keep their initial values",
    )
    args = parser.parse_args(argv)
    if args.cmd == "extract":
        extract(args.src, args.out)
        return 0
    return convert(
        args.config,
        args.src,
        args.out,
        args.onnx,
        args.widths,
        args.tol,
        args.allow_missing,
        args.src_dict,
    )


if __name__ == "__main__":
    sys.exit(main())
