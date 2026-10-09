"""Upload training checkpoints to a private Kaggle dataset while training runs.

A Kaggle commit run saves /kaggle/working only when the notebook ends; a crash or a
cancelled run loses every checkpoint. This watcher copies the newest checkpoint files
(latest.*, best_accuracy.*, the train log and config.yml) of the save_model_dir to a staging folder
once they have stopped changing, and uploads them as a new version of a dataset with the
Kaggle CLI (old versions are deleted). Attach that dataset as input to resume from it.

    uploader = CheckpointUploader(out_dir, "/tmp/ckpt_upload", "user/ru-ocr-checkpoints")
    uploader.start()          # background thread, polls every poll_s seconds
    ...                       # training
    uploader.stop()           # waits for the thread and uploads the last checkpoint

Other files: patterns=("preds_partial.tsv",), required="preds_partial.tsv".

The CLI reads the credentials from the environment (KAGGLE_API_TOKEN).
"""

import glob
import json
import os
import shutil
import subprocess
import threading
import time

PATTERNS = ("latest.*", "best_accuracy.*", "train.log", "config.yml")


class CheckpointUploader(object):
    def __init__(
        self,
        src_dir,
        stage_dir,
        dataset_id,
        patterns=PATTERNS,
        required="latest.",
        settle_s=30,
        poll_s=60,
        kaggle_cmd=("kaggle",),
        log=print,
    ):
        self.src_dir = src_dir
        self.stage_dir = stage_dir
        self.dataset_id = dataset_id
        self.patterns = tuple(patterns)
        self.required = required  # nothing is uploaded until a file with this name prefix exists
        self.settle_s = settle_s
        self.poll_s = poll_s
        self.kaggle_cmd = tuple(kaggle_cmd)
        self.log = log
        self.uploaded = None  # signature of the last uploaded checkpoint
        self.uploads = 0
        self._exists = None
        self._stop = threading.Event()
        self._thread = None

    def files(self):
        """Checkpoint files (not folders) in src_dir matching the patterns."""
        found = set()
        for pattern in self.patterns:
            found.update(glob.glob(os.path.join(self.src_dir, pattern)))
        return sorted(p for p in found if os.path.isfile(p))

    def signature(self, files):
        return tuple((os.path.basename(p), os.path.getmtime(p)) for p in files)

    def ready(self, now=None):
        """Files of a new checkpoint that has not changed for settle_s, else None."""
        files = self.files()
        if not any(os.path.basename(p).startswith(self.required) for p in files):
            return None
        newest = max(os.path.getmtime(p) for p in files)
        now = time.time() if now is None else now
        if now - newest < self.settle_s:
            return None  # still being written
        if self.signature(files) == self.uploaded:
            return None
        return files

    def poll_once(self, now=None):
        """Upload the checkpoint if a new one is ready; True when uploaded."""
        files = self.ready(now)
        if files is None:
            return False
        signature = self.signature(files)
        self._stage(files)
        if not self._upload():
            return False  # retried at the next poll
        self.uploaded = signature
        self.uploads += 1
        return True

    def _stage(self, files):
        if os.path.isdir(self.stage_dir):
            shutil.rmtree(self.stage_dir)
        os.makedirs(self.stage_dir)
        for path in files:
            shutil.copy2(path, self.stage_dir)
        meta = {
            "title": self.dataset_id.split("/")[-1],
            "id": self.dataset_id,
            "licenses": [{"name": "unknown"}],
        }
        with open(os.path.join(self.stage_dir, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)

    def _run(self, *args):
        proc = subprocess.run(
            list(self.kaggle_cmd) + list(args),
            capture_output=True,
            text=True,
            errors="replace",
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()

    def _upload(self):
        if self._exists is None:
            code, _ = self._run("datasets", "status", self.dataset_id)
            self._exists = code == 0
        message = "checkpoint {}".format(time.strftime("%Y-%m-%d %H:%M"))
        if self._exists:
            code, out = self._run(
                "datasets", "version", "-p", self.stage_dir, "-m", message, "-d", "-q"
            )
        else:
            code, out = self._run("datasets", "create", "-p", self.stage_dir, "-q")
        if code != 0:
            self.log("checkpoint upload failed: {}".format(out[-500:]))
            if self._exists and ("404" in out or "not found" in out.lower()):
                self._exists = False  # create it at the next attempt
            return False
        self._exists = True
        self.log("checkpoint uploaded to {} ({})".format(self.dataset_id, message))
        return True

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            try:
                self.poll_once()
            except Exception as exc:  # never break training because of an upload
                self.log("checkpoint upload error: {}".format(exc))

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self, final=True):
        """Stop the thread; with final, upload the last checkpoint right away."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if final:
            try:
                self.poll_once(now=float("inf"))
            except Exception as exc:
                self.log("checkpoint upload error: {}".format(exc))
