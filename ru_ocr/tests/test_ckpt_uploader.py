import json
import os
import sys
import time

from conftest import load_tool

cu = load_tool("ckpt_uploader")

FAKE_KAGGLE = """
import json, os, sys
args = sys.argv[1:]
log = os.environ["FAKE_KAGGLE_LOG"]
with open(log, "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[:2] == ["datasets", "status"]:
    sys.exit(0 if os.path.exists(log + ".created") else 1)
if args[:2] == ["datasets", "create"]:
    open(log + ".created", "w").close()
if os.environ.get("FAKE_KAGGLE_FAIL") and args[:2] == ["datasets", "version"]:
    print(os.environ["FAKE_KAGGLE_FAIL"])
    sys.exit(2)
"""


def _setup(tmp_path, monkeypatch):
    fake = tmp_path / "fake_kaggle.py"
    fake.write_text(FAKE_KAGGLE)
    log = tmp_path / "kaggle_calls.txt"
    monkeypatch.setenv("FAKE_KAGGLE_LOG", str(log))
    src = tmp_path / "output"
    src.mkdir()
    messages = []
    up = cu.CheckpointUploader(
        str(src),
        str(tmp_path / "stage"),
        "someone/ru-ocr-checkpoints",
        kaggle_cmd=(sys.executable, str(fake)),
        log=messages.append,
    )
    return up, src, log, messages


def _save(src, prefix="latest", age_s=60):
    t = time.time() - age_s
    for ext in (".pdparams", ".pdopt", ".states"):
        path = src / (prefix + ext)
        path.write_bytes(b"x")
        os.utime(path, (t, t))


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_first_upload_creates_then_versions(tmp_path, monkeypatch):
    up, src, log, messages = _setup(tmp_path, monkeypatch)
    _save(src)
    for name in ("train.log", "config.yml"):
        (src / name).write_text(name)
        os.utime(src / name, (time.time() - 60,) * 2)
    (src / "best_accuracy").mkdir()  # folders are not uploaded
    assert up.poll_once()
    staged = sorted(os.listdir(tmp_path / "stage"))
    assert staged == [
        "config.yml", "dataset-metadata.json", "latest.pdopt", "latest.pdparams", "latest.states",
        "train.log",
    ]
    meta = json.loads((tmp_path / "stage" / "dataset-metadata.json").read_text())
    assert meta["id"] == "someone/ru-ocr-checkpoints"
    assert [c[:2] for c in _calls(log)] == [["datasets", "status"], ["datasets", "create"]]

    assert not up.poll_once()  # nothing new
    _save(src, age_s=45)
    assert up.poll_once()
    version = _calls(log)[-1]
    assert version[:2] == ["datasets", "version"] and "-d" in version
    assert up.uploads == 2 and len(messages) == 2


def test_waits_until_files_settle(tmp_path, monkeypatch):
    up, src, log, _ = _setup(tmp_path, monkeypatch)
    _save(src, age_s=5)  # still being written
    assert not up.poll_once()
    assert not log.exists()
    assert up.poll_once(now=time.time() + 60)


def test_no_latest_no_upload(tmp_path, monkeypatch):
    up, src, log, _ = _setup(tmp_path, monkeypatch)
    _save(src, prefix="best_accuracy")
    assert not up.poll_once()
    assert not log.exists()


def test_failed_upload_is_retried(tmp_path, monkeypatch):
    up, src, log, messages = _setup(tmp_path, monkeypatch)
    _save(src)
    assert up.poll_once()  # create
    _save(src, age_s=40)
    monkeypatch.setenv("FAKE_KAGGLE_FAIL", "1")
    assert not up.poll_once()
    assert "failed" in messages[-1]
    monkeypatch.delenv("FAKE_KAGGLE_FAIL")
    assert up.poll_once()


def test_thread_stop_uploads_last_checkpoint(tmp_path, monkeypatch):
    up, src, log, _ = _setup(tmp_path, monkeypatch)
    up.poll_s = 0.05
    up.start()
    _save(src, age_s=1)  # too fresh for the thread
    time.sleep(0.2)
    up.stop()  # final upload does not wait for settle_s
    assert up.uploads == 1


def test_version_of_missing_dataset_creates_it_next_time(tmp_path, monkeypatch):
    up, src, log, _ = _setup(tmp_path, monkeypatch)
    up._exists = True  # status said it exists, but it does not
    _save(src)
    monkeypatch.setenv("FAKE_KAGGLE_FAIL", "404 Client Error: Not Found")
    assert not up.poll_once()
    monkeypatch.delenv("FAKE_KAGGLE_FAIL")
    assert up.poll_once()
    assert _calls(log)[-1][:2] == ["datasets", "create"]
