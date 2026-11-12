from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import warnings

import numpy as np


class ReadWriteLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._writer and not self._waiting_writers)
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            self._condition.notify_all()

    @contextmanager
    def read(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self, timeout: float = 30):
        with self._condition:
            self._waiting_writers += 1
            try:
                if not self._condition.wait_for(lambda: not self._writer and self._readers == 0, timeout=timeout):
                    raise TimeoutError("Active requests did not finish; update aborted, old snapshot retained")
                self._writer = True
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def problem_key(problem: dict) -> str:
    url = problem.get("url")
    identity = [problem.get("source", ""), url] if isinstance(url, str) and url.startswith(("https://", "http://")) else [problem.get("source", ""), text_key(problem["text"])]
    return json.dumps(identity, ensure_ascii=False)


def model_fingerprint(model_path: str) -> str:
    directory = Path(model_path)
    if not directory.is_dir():
        raise ValueError("Hot updates require a local model directory with fixed weights")
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "modules.json",
             "1_Pooling/config.json", "config_sentence_transformers.json"]
    names.extend(str(path.relative_to(directory)) for path in sorted(directory.glob("*.safetensors")))
    if not list(directory.glob("*.safetensors")):
        raise ValueError("Model weights are missing")
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProblemData:
    probs: list[dict]
    embeddings: np.ndarray
    norms: np.ndarray | None
    available_ojs: list[str]
    oj_code: dict[str, int]
    source_codes: np.ndarray
    manifest: dict | None = None


def load_problem_data(prob_path: str | Path, emb_path: str | Path, mmap: bool = True,
                      expected_dimension: int | None = None,
                      allow_empty_text: bool = False) -> ProblemData:
    problems = []
    empty_lines = []
    with Path(prob_path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            problem = json.loads(line)
            if not isinstance(problem, dict) or not isinstance(problem.get("text"), str):
                raise ValueError(f"Invalid problem on line {number}")
            if not problem["text"].strip():
                if not allow_empty_text:
                    raise ValueError(f"Empty problem text on line {number}")
                empty_lines.append(number)
            for field in ("source", "title", "url"):
                if field in problem and problem[field] is not None and not isinstance(problem[field], str):
                    raise ValueError(f"Invalid {field} on line {number}")
            problems.append(problem)
    embeddings = np.load(emb_path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(problems) or not len(problems):
        raise ValueError("Embedding rows and problem count do not match")
    if embeddings.dtype != np.float32 or (expected_dimension is not None and embeddings.shape[1] != expected_dimension):
        raise ValueError("Embeddings must be float32 with matching model dimension")
    norms = np.empty(len(problems), dtype=np.float32)
    for start in range(0, len(problems), 2048):
        block = embeddings[start:start + 2048]
        values = np.linalg.norm(block, axis=1)
        if not np.all(np.isfinite(block)) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid embedding values near row {start}")
        norms[start:start + len(block)] = values
    if not mmap:
        embeddings /= norms[:, None]
        norms = None
    embeddings.setflags(write=False)
    if norms is not None:
        norms.setflags(write=False)
    ojs = sorted({p.get("source") for p in problems if p.get("source")})
    codes = {name: i for i, name in enumerate(ojs)}
    source_codes = np.array([codes.get(p.get("source"), -1) for p in problems], dtype=np.int32)
    source_codes.setflags(write=False)
    if empty_lines:
        warnings.warn(
            f"Preserving {len(empty_lines)} legacy empty-text rows and their embeddings; "
            f"no rows removed (first JSONL lines: {empty_lines[:5]})",
            RuntimeWarning,
            stacklevel=2,
        )
    return ProblemData(problems, embeddings, norms, ojs, codes, source_codes)


def load_bundle(directory: str | Path, expected_fingerprint: str, expected_dimension: int) -> ProblemData:
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_fingerprint") != expected_fingerprint:
        raise ValueError("Bundle schema/model fingerprint does not match")
    if manifest.get("max_seq_length") not in (2048, 8192):
        raise ValueError("Unsupported bundle embedding length")
    for name in ("problems.jsonl", "embeddings.npy"):
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != manifest.get("sha256", {}).get(name):
            raise ValueError(f"Bundle checksum failed: {name}")
    empty_rows = manifest.get("preserved_empty_rows", [])
    if not isinstance(empty_rows, list) or any(type(i) is not int or i < 0 for i in empty_rows) or empty_rows != sorted(set(empty_rows)):
        raise ValueError("Invalid preserved empty-row manifest")
    data = load_problem_data(directory / "problems.jsonl", directory / "embeddings.npy",
                             mmap=True, expected_dimension=expected_dimension,
                             allow_empty_text=bool(empty_rows))
    if empty_rows != [i for i, p in enumerate(data.probs) if not p["text"].strip()]:
        raise ValueError("Preserved empty rows do not match bundle data")
    if manifest.get("count") != len(data.probs) or manifest.get("dimension") != data.embeddings.shape[1]:
        raise ValueError("Bundle manifest shape does not match its data")
    return ProblemData(data.probs, data.embeddings, data.norms, data.available_ojs,
                       data.oj_code, data.source_codes, manifest)


def validate_stable_prefix(old: list[dict], new: list[dict]) -> None:
    if len(new) < len(old):
        raise ValueError("Hot updates cannot delete old rows; stable /p/<pid> links must be preserved")
    for index, problem in enumerate(old):
        if problem_key(problem) != problem_key(new[index]):
            raise ValueError(f"Problem identity changed at row {index}; reorder bundle to preserve old IDs")
    for index, problem in enumerate(new):
        if not problem["text"].strip() and (index >= len(old) or old[index]["text"].strip()):
            raise ValueError(f"Empty text at row {index} is not a preserved legacy row")
