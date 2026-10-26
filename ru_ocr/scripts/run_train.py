"""Train a ru_ocr recognizer with the native tools/train.py on one or more GPUs.

Builds the tools/train.py overrides used for the published model and starts it under
paddle.distributed.launch. Run from the repository root after ru_ocr/scripts/setup_env.sh:

    # first run of a stage: weights of an earlier checkpoint only (new optimizer and LR schedule)
    python ru_ocr/scripts/run_train.py --gpus 0,1 --out output/ft4 --init output/ft3/best_accuracy \
        --data-dir data/ru_root --train-labels data/ru_labels/all_train_clean_w_h.txt \
        --val-labels data/ru_labels/all_val5k_clean_w_h.txt
    # the same command again resumes from output/ft4/latest (optimizer, epoch, position inside the epoch)

Without --init and --resume a run starts from Global.pretrained_model of the config (the official
PP-OCRv6 small weights in ./pretrain/). A checkpoint that has finished all epochs of the run is refused:
"resuming" it would stop at once without a step. Options left out keep the config's values; anything else
goes through -o, e.g. `-o Train.dataset.ratio_list=[0.5]`. --dry-run prints the command.
"""

import argparse
import os
import pickle
import shlex
import subprocess
import sys

import yaml

CONFIG = "ru_ocr/configs/ru_PP-OCRv6_small_rec.yml"


def resume_epoch(states_path):
    """Epoch a checkpoint resumes at, as ppocr.utils.save_load.resume_position: an end-of-epoch save goes on with
    the next epoch, a save inside an epoch (step_in_epoch) with the same one."""
    with open(states_path, "rb") as f:
        states = pickle.load(f, encoding="latin1")
    return states.get("epoch", 0) + (0 if states.get("step_in_epoch") else 1)


def find_resume(args):
    """The checkpoint prefix to resume from: --resume, else <out>/latest when it exists, else None."""
    if args.resume:
        return args.resume
    latest = os.path.join(args.out, "latest")
    return latest if os.path.exists(latest + ".pdparams") and not args.no_auto_resume else None


def epoch_num(args):
    if args.epochs:
        return args.epochs
    with open(args.config, encoding="utf-8") as f:
        return yaml.safe_load(f)["Global"]["epoch_num"]


def list_opt(paths):
    return "[" + ",".join(paths) + "]"


def build_opts(args, resume):
    """The -o overrides of tools/train.py."""
    opts = [
        f"Global.save_model_dir={args.out}",
        "Global.save_epoch_step=1",
        "Global.save_on_signal=true",  # SIGINT/SIGTERM: finish the step, save latest with its place in the epoch
        f"Global.print_batch_step={args.print_steps}",
    ]
    if args.save_steps:
        opts.append(f"Global.save_batch_step={args.save_steps}")
    if args.eval_steps:
        opts.append(f"Global.eval_batch_step=[0,{args.eval_steps}]")
    if args.epochs:
        opts.append(f"Global.epoch_num={args.epochs}")
    if args.lr is not None:
        opts.append(f"Optimizer.lr.learning_rate={args.lr}")
    if args.warmup_epochs is not None:
        opts.append(f"Optimizer.lr.warmup_epoch={args.warmup_epochs}")
    if args.amp:
        opts += ["Global.use_amp=true", "Global.amp_dtype=float16", "Global.scale_loss=1024",
                 "Global.use_dynamic_loss_scaling=true"]
    if args.data_dir:
        opts += [f"Train.dataset.data_dir={args.data_dir}", f"Eval.dataset.data_dir={args.data_dir}"]
    if args.train_labels:
        opts.append(f"Train.dataset.label_file_list={list_opt(args.train_labels)}")
    if args.val_labels:
        opts.append(f"Eval.dataset.label_file_list={list_opt(args.val_labels)}")
    if args.batch:
        opts += [f"Train.sampler.first_bs={args.batch}", f"Train.loader.batch_size_per_card={args.batch}"]
    if args.sorted_ratio is not None:
        opts.append(f"Train.sampler.sorted_batch_ratio={args.sorted_ratio}")
    if args.workers is not None:
        opts.append(f"Train.loader.num_workers={args.workers}")
    if args.eval_workers is not None:
        opts.append(f"Eval.loader.num_workers={args.eval_workers}")
    if resume:
        opts.append(f"Global.checkpoints={resume}")
    elif args.init:
        opts.append(f"Global.pretrained_model={args.init}")  # weights only: new optimizer, epochs and LR schedule
    return opts + list(args.opt)


def build_command(args, resume):
    return [sys.executable, "-m", "paddle.distributed.launch", "--gpus", args.gpus,
            "--log_dir", os.path.join(args.out, "dist_log"),
            "tools/train.py", "-c", args.config, "-o", *build_opts(args, resume)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="GPU ids for paddle.distributed.launch, e.g. 0,1")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--out", required=True, help="Global.save_model_dir: checkpoints, logs (dist_log/)")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--init", help="checkpoint prefix whose weights start a new run (e.g. .../best_accuracy)")
    start.add_argument("--resume", help="checkpoint prefix to resume (default: <out>/latest when it exists)")
    parser.add_argument("--no-auto-resume", action="store_true", help="do not resume from <out>/latest")
    parser.add_argument("--data-dir", help="root the label paths are relative to (train and eval)")
    parser.add_argument("--train-labels", nargs="+")
    parser.add_argument("--val-labels", nargs="+")
    parser.add_argument("--epochs", type=int, help="Global.epoch_num")
    parser.add_argument("--lr", type=float, help="Optimizer.lr.learning_rate")
    parser.add_argument("--warmup-epochs", type=float, help="Optimizer.lr.warmup_epoch")
    parser.add_argument("--batch", type=int, default=32, help="batch per card (first_bs = batch_size_per_card)")
    parser.add_argument("--sorted-ratio", type=float, default=0.8, help="share of batches sorted by aspect ratio")
    parser.add_argument("--workers", type=int, default=2, help="data workers per GPU process")
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=2000, help="save latest every N steps")
    parser.add_argument("--eval-steps", type=int, default=10000, help="validate every N steps")
    parser.add_argument("--print-steps", type=int, default=100)
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="fp32 instead of fp16 AMP")
    parser.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    parser.add_argument("-o", "--opt", nargs="*", default=[], help="more tools/train.py overrides")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists("tools/train.py"):
        sys.exit("run from the repository root")
    resume = find_resume(args)
    if resume:
        epochs = epoch_num(args)
        start_epoch = resume_epoch(resume + ".states")
        if start_epoch > epochs:
            sys.exit(f"{resume} has finished all {epochs} epochs: for a new run use another --out "
                     f"and --init with its weights, or raise --epochs")
        print(f"resume from {resume} at epoch {start_epoch} of {epochs}", flush=True)
    elif args.init:
        print(f"start from the weights of {args.init}", flush=True)
    cmd = build_command(args, resume)
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0
    os.makedirs(args.out, exist_ok=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
