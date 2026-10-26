import pickle
import sys

import pytest

from conftest import load_tool

rt = load_tool("run_train", folder="scripts")


def _ckpt(prefix, **states):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + ".pdparams").write_bytes(b"")
    with open(str(prefix) + ".states", "wb") as f:
        pickle.dump(states, f, protocol=2)
    return str(prefix)


def _opts(cmd):
    return cmd[cmd.index("-o") + 1:]


@pytest.fixture
def in_repo(repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)


def test_command_runs_native_train_under_distributed_launch(tmp_path):
    args = rt.parse_args(["--gpus", "0,1", "--out", str(tmp_path / "ft4")])
    cmd = rt.build_command(args, None)

    assert cmd[:5] == [sys.executable, "-m", "paddle.distributed.launch", "--gpus", "0,1"]
    assert cmd[cmd.index("--log_dir") + 1] == str(tmp_path / "ft4" / "dist_log")
    assert cmd[cmd.index("tools/train.py") + 1:cmd.index("-o")] == ["-c", rt.CONFIG]


def test_defaults_match_the_published_run(tmp_path):
    out = str(tmp_path / "ft4")
    opts = _opts(rt.build_command(rt.parse_args(["--out", out]), None))

    for o in [f"Global.save_model_dir={out}", "Global.save_on_signal=true", "Global.save_batch_step=2000",
              "Global.eval_batch_step=[0,10000]", "Global.use_amp=true", "Global.scale_loss=1024",
              "Train.sampler.first_bs=32", "Train.loader.batch_size_per_card=32",
              "Train.sampler.sorted_batch_ratio=0.8", "Train.loader.num_workers=2"]:
        assert o in opts
    assert not any(o.startswith(("Global.epoch_num", "Optimizer.", "Global.checkpoints", "Global.pretrained_model",
                                 "Train.dataset.", "Eval.dataset.")) for o in opts)


def test_data_lr_and_passthrough_overrides(tmp_path):
    args = rt.parse_args(["--out", str(tmp_path), "--data-dir", "root", "--train-labels", "a.txt", "b.txt",
                          "--val-labels", "v.txt", "--epochs", "3", "--lr", "0.0005", "--warmup-epochs", "0",
                          "--no-amp", "-o", "Train.dataset.ratio_list=[0.5,1.0]"])
    opts = _opts(rt.build_command(args, None))

    assert "Train.dataset.data_dir=root" in opts and "Eval.dataset.data_dir=root" in opts
    assert "Train.dataset.label_file_list=[a.txt,b.txt]" in opts
    assert "Eval.dataset.label_file_list=[v.txt]" in opts
    assert {"Global.epoch_num=3", "Optimizer.lr.learning_rate=0.0005", "Optimizer.lr.warmup_epoch=0.0"} <= set(opts)
    assert not any(o.startswith("Global.use_amp") for o in opts)
    assert opts[-1] == "Train.dataset.ratio_list=[0.5,1.0]"


def test_init_starts_from_weights_and_resume_wins_over_it(tmp_path):
    args = rt.parse_args(["--out", str(tmp_path / "ft4"), "--init", "ft3/best_accuracy"])

    assert "Global.pretrained_model=ft3/best_accuracy" in _opts(rt.build_command(args, None))
    resumed = _opts(rt.build_command(args, "ft4/latest"))
    assert "Global.checkpoints=ft4/latest" in resumed
    assert not any(o.startswith("Global.pretrained_model") for o in resumed)


def test_init_and_resume_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        rt.parse_args(["--out", str(tmp_path), "--init", "a", "--resume", "b"])


def test_find_resume_picks_latest_of_out(tmp_path):
    out = tmp_path / "ft4"
    assert rt.find_resume(rt.parse_args(["--out", str(out)])) is None
    latest = _ckpt(out / "latest", epoch=1, step_in_epoch=500)

    assert rt.find_resume(rt.parse_args(["--out", str(out)])) == latest
    assert rt.find_resume(rt.parse_args(["--out", str(out), "--no-auto-resume"])) is None
    assert rt.find_resume(rt.parse_args(["--out", str(out), "--resume", "x/iter"])) == "x/iter"


@pytest.mark.parametrize("states, epoch", [
    ({"epoch": 1, "step_in_epoch": 500}, 1),  # inside epoch 1: goes on with it
    ({"epoch": 1}, 2),  # end of epoch 1
    ({"epoch": 2, "step_in_epoch": 0}, 3),
])
def test_resume_epoch_as_resume_position(tmp_path, states, epoch):
    assert rt.resume_epoch(_ckpt(tmp_path / "latest", **states) + ".states") == epoch


def test_finished_run_is_refused(tmp_path, in_repo):
    out = tmp_path / "ft4"
    _ckpt(out / "latest", epoch=2)

    with pytest.raises(SystemExit, match="finished all 2 epochs"):
        rt.main(["--out", str(out), "--dry-run"])
    assert rt.main(["--out", str(out), "--epochs", "3", "--dry-run"]) == 0


def test_dry_run_prints_and_does_not_create_out(tmp_path, in_repo, capsys):
    out = tmp_path / "ft4"

    assert rt.main(["--out", str(out), "--init", "ft3/best_accuracy", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "start from the weights of ft3/best_accuracy" in printed
    assert "paddle.distributed.launch" in printed
    assert not out.exists()
