"""Model-free query encoder for local debugging (MOCK_MODEL=1).

It never loads a neural model. A query vector is derived from existing problem
embeddings:

1. ``#<pid>`` -> embedding of that problem;
2. case-insensitive substring match on titles;
3. substring match on the first ``text_chars`` characters of statements;
4. otherwise a problem picked deterministically from sha256(query).

A small deterministic noise is added so that the query is not exactly equal
to a stored problem vector. Token count is approximated.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

_PID_RE = re.compile(r"^\s*#(\d+)\s*$")


class MockEncoder:
    def __init__(
        self,
        probs: Sequence[Dict[str, Any]],
        get_vecs: Callable[[np.ndarray], np.ndarray],
        noise: float = 0.004,
        text_chars: int = 2000,
    ) -> None:
        self._probs = probs
        self._get_vecs = get_vecs
        self._noise = float(noise)
        self._text_chars = int(text_chars)
        self._titles_lower: List[str] = [str(p.get("title") or "").lower() for p in probs]

    @staticmethod
    def _seed(q: str) -> int:
        return int.from_bytes(hashlib.sha256(q.encode("utf-8")).digest()[:8], "little")

    def pick_problem(self, q: str) -> int:
        n = len(self._probs)
        m = _PID_RE.match(q)
        if m:
            pid = int(m.group(1))
            if 0 <= pid < n:
                return pid
        ql = q.strip().lower()
        if ql:
            for i, t in enumerate(self._titles_lower):
                if ql in t:
                    return i
            if len(ql) <= 200:
                for i, p in enumerate(self._probs):
                    text = p.get("text") or ""
                    if ql in text[: self._text_chars].lower():
                        return i
        return self._seed(q) % n

    def encode(self, q: str) -> np.ndarray:
        pid = self.pick_problem(q)
        v = np.array(self._get_vecs(np.array([pid]))[0], dtype=np.float32)
        rng = np.random.default_rng(self._seed(q))
        v = v + self._noise * rng.standard_normal(v.shape[0]).astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    @staticmethod
    def count_tokens(q: str, max_tokens: Optional[int] = None) -> int:
        n = max(1, len(q) // 3)
        return min(n, max_tokens) if max_tokens else n
