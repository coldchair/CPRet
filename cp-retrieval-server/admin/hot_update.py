from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

from admin.local_admin import private_directory
from retrieval.retrieval_state import ReadWriteLock, load_bundle, validate_stable_prefix


class HotUpdateController:
    def __init__(self, data, encoder, activate, model_fingerprint: str, update_root: str | Path,
                 state_path: str | Path, query_tokens: int, embedding_tokens: int = 2048) -> None:
        self.lock = ReadWriteLock()
        self._update_lock = threading.Lock()
        self._data = data
        self._encoder = encoder
        self._activate = activate
        self._fingerprint = model_fingerprint
        self._root = Path(update_root).resolve()
        private_directory(self._root)
        self._state_path = Path(state_path)
        private_directory(self._state_path.parent)
        self._length = query_tokens
        self._embedding_length = embedding_tokens
        self._bundle = None
        self._previous = None
        self._generation = 0
        self._job_lock = threading.Lock()
        self._job = None
        self._cancel_job = threading.Event()

    def _bundle_path(self, value) -> Path:
        if not isinstance(value, str):
            raise ValueError("Bundle must be a local directory path")
        path = Path(value).resolve(strict=True)
        if not path.is_relative_to(self._root) or path == self._root:
            raise ValueError("Bundle must be inside DATA_UPDATE_ROOT")
        return path

    def _validate_length(self, value) -> int:
        if type(value) is not int or value not in (2048, 8192) or value > self._encoder.capacity:
            raise ValueError("Length must be 2048 or 8192 and fit the running engine capacity")
        return value

    def _persist(self, bundle, length: int) -> None:
        payload = {"schema_version": 1, "model_fingerprint": self._fingerprint,
                   "bundle": str(bundle) if bundle else None, "query_tokens": length,
                   "previous": {"bundle": str(self._bundle) if self._bundle else None,
                                "query_tokens": self._length}}
        descriptor, temporary = tempfile.mkstemp(dir=self._state_path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def restore(self) -> None:
        if not self._state_path.exists():
            return
        if self._state_path.is_symlink():
            raise ValueError("Active state cannot be a symlink")
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("model_fingerprint") != self._fingerprint:
            raise ValueError("Persisted update state belongs to a different model")
        length = self._validate_length(payload.get("query_tokens"))
        bundle = self._bundle_path(payload["bundle"]) if payload.get("bundle") else None
        data = load_bundle(bundle, self._fingerprint, self._data.embeddings.shape[1]) if bundle else self._data
        if bundle:
            validate_stable_prefix(self._data.probs, data.probs)
        previous = None
        pointer = payload.get("previous")
        if isinstance(pointer, dict):
            previous_length = self._validate_length(pointer.get("query_tokens"))
            previous_bundle = self._bundle_path(pointer["bundle"]) if pointer.get("bundle") else None
            previous_data = load_bundle(previous_bundle, self._fingerprint, self._data.embeddings.shape[1]) if previous_bundle else self._data
            previous = (previous_data, previous_length, previous_bundle,
                        previous_data.manifest["max_seq_length"] if previous_data.manifest else 2048)
        self._commit(data, length, bundle, persist=False)
        self._previous = previous

    def _commit(self, data, length: int, bundle, persist: bool = True) -> None:
        with self.lock.write():
            old = (self._data, self._length, self._bundle, self._embedding_length)
            try:
                self._encoder.set_max_seq_length(length)
                self._activate(data, length)
                if persist:
                    self._persist(bundle, length)
            except BaseException:
                self._encoder.set_max_seq_length(old[1])
                self._activate(old[0], old[1])
                raise
            self._previous = old
            self._data = data
            self._length = length
            self._bundle = bundle
            self._embedding_length = data.manifest["max_seq_length"] if data.manifest else 2048
            self._generation += 1

    def _start_build(self, message: dict) -> dict:
        input_path = self._bundle_path(message.get("input"))
        if not input_path.is_file():
            raise ValueError("Build input must be a JSONL file inside DATA_UPDATE_ROOT")
        name = message.get("version")
        if not isinstance(name, str) or not name or name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in name):
            raise ValueError("Version must be a simple local directory name")
        output = self._root / name
        length = self._validate_length(message.get("tokens", 8192))
        if output.exists():
            raise ValueError("Version directory already exists")
        with self._job_lock:
            if self._job and self._job["state"] == "running":
                raise ValueError("A build is already running")
            with self.lock.read():
                base = self._data
            self._cancel_job = threading.Event()
            self._job = {"id": uuid.uuid4().hex[:12], "state": "running", "bundle": str(output)}
            thread = threading.Thread(target=self._build, args=(input_path, output, base, length),
                                      daemon=True, name="incremental-problem-build")
            thread.start()
            return dict(self._job)

    def _build(self, input_path, output, base, length) -> None:
        from admin.prepare_update import prepare_bundle
        from encoder.async_pooler import EncoderBusyError, EncoderTimeoutError

        def progress(values):
            with self._job_lock:
                self._job.update(values)

        def encode(text, limit):
            timeouts = 0
            while not self._cancel_job.is_set():
                try:
                    result = self._encoder.encode_at_length(text, limit)
                    time.sleep(0.05)
                    return result
                except EncoderBusyError:
                    self._cancel_job.wait(0.1)
                except EncoderTimeoutError:
                    timeouts += 1
                    if timeouts >= 3:
                        raise
                    self._cancel_job.wait(1)
            raise RuntimeError("Bundle build cancelled")

        try:
            manifest = prepare_bundle(input_path, output, base, self._fingerprint, length,
                                      self._encoder.full_token_count, encode, progress=progress,
                                      cancelled=self._cancel_job)
            with self._job_lock:
                self._job.update(state="ready", stats=manifest["stats"])
        except Exception as error:
            with self._job_lock:
                self._job.update(state="failed", error=str(error))

    def status(self) -> dict:
        with self.lock.read():
            result = {"generation": self._generation, "problems": len(self._data.probs),
                      "query_tokens": self._length, "embedding_tokens": self._embedding_length,
                      "engine_capacity": self._encoder.capacity,
                      "bundle": str(self._bundle) if self._bundle else None,
                      "rollback_available": self._previous is not None}
        with self._job_lock:
            result["build"] = dict(self._job) if self._job else None
        if hasattr(self._encoder, "status"):
            result["encoder"] = self._encoder.status()
        return result

    def handle(self, message: dict) -> dict:
        command = message.get("command")
        if command == "status":
            return self.status()
        if command == "build":
            return self._start_build(message)
        if command == "cancel-build":
            self._cancel_job.set()
            return self.status()
        if command not in ("set-query-length", "reload", "rollback"):
            raise ValueError("Unsupported local command")
        with self._update_lock:
            if command == "set-query-length":
                length = self._validate_length(message.get("tokens"))
                self._commit(self._data, length, self._bundle)
            elif command == "reload":
                bundle = self._bundle_path(message.get("bundle"))
                data = load_bundle(bundle, self._fingerprint, self._data.embeddings.shape[1])
                validate_stable_prefix(self._data.probs, data.probs)
                length = self._validate_length(message.get("tokens", self._length))
                self._commit(data, length, bundle)
            else:
                if self._previous is None:
                    raise ValueError("No previous snapshot to roll back to")
                data, length, bundle, _ = self._previous
                self._commit(data, length, bundle)
        return self.status()
