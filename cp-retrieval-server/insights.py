"""Result insights for a retrieval query (pure numpy, no Flask dependency).

Given the top-K retrieved problems (already filtered and sorted by score), this
module provides:

* cross-OJ duplicate folding (greedy, by cosine similarity threshold)
* direction clustering (spherical k-means + silhouette-based k selection)
* MMR re-ranking for diversity
* 2D PCA projection for a scatter plot

All public outputs are plain Python types so they can be JSON-serialised.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "dedup_greedy",
    "spherical_kmeans",
    "silhouette_from_sim",
    "choose_clusters",
    "mmr_rerank",
    "pca_2d",
    "analyze",
]

_EPS = 1e-12


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, _EPS)


# --------------------------------------------------------------------------- #
# Duplicate folding
# --------------------------------------------------------------------------- #
_TITLE_STRIP = re.compile(r"[\W_]+", re.UNICODE)
_TITLE_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]|「[^」]*」|【[^】]*】)\s*")


def title_key(title: Optional[str]) -> str:
    """Normalise a title for duplicate matching.

    Drops a leading contest tag such as ``[NOI2021]`` / ``「NOI2021」`` and all
    punctuation/whitespace; lower-cases the rest.
    """
    if not title:
        return ""
    t = _TITLE_PREFIX.sub("", str(title))
    t = _TITLE_STRIP.sub("", t).lower()
    return t


def dedup_greedy(
    sim: np.ndarray,
    threshold: float,
    title_keys: Optional[Sequence[str]] = None,
    title_threshold: float = 0.85,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Fold near-duplicates.

    ``sim`` is the (n, n) cosine similarity matrix of items sorted by score
    descending. Each item is attached to the most similar already-kept primary
    if that similarity is >= ``threshold`` (or >= ``title_threshold`` when the
    normalised titles are identical and non-empty); otherwise it becomes a
    primary.

    Returns (primary local indices in order, {primary: [duplicate locals]}).
    """
    n = sim.shape[0]
    primaries: List[int] = []
    dups: Dict[int, List[int]] = {}
    if n == 0:
        return primaries, dups
    for i in range(n):
        if primaries:
            row = sim[i, primaries].copy()
            if title_keys is not None and title_keys[i]:
                same = np.fromiter(
                    (title_keys[p] == title_keys[i] for p in primaries), dtype=bool, count=len(primaries)
                )
                # Boost title matches so they pass the relaxed threshold.
                row = np.where(same & (row >= title_threshold), np.maximum(row, threshold), row)
            j = int(np.argmax(row))
            if row[j] >= threshold:
                dups[primaries[j]].append(i)
                continue
        primaries.append(i)
        dups[i] = []
    return primaries, dups


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _kmeanspp_init(sim: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    n = sim.shape[0]
    first = int(rng.integers(n))
    centers = [first]
    dist = np.clip(1.0 - sim[first], 0.0, None)
    for _ in range(1, k):
        w = dist ** 2
        total = float(w.sum())
        if total <= _EPS:
            remaining = [i for i in range(n) if i not in centers]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(rng.choice(n, p=w / total))
        centers.append(nxt)
        dist = np.minimum(dist, np.clip(1.0 - sim[nxt], 0.0, None))
    return centers


def spherical_kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 50,
    sim: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Spherical k-means on L2-normalised rows.

    Returns (labels (n,), centroids (k, d) normalised, objective = sum of
    cosine similarity to assigned centroid; higher is better).
    """
    n = x.shape[0]
    if k <= 1 or n <= k:
        labels = np.zeros(n, dtype=np.int64) if k <= 1 else np.arange(n, dtype=np.int64)
        kk = 1 if k <= 1 else n
        cents = np.zeros((kk, x.shape[1]), dtype=x.dtype)
        for c in range(kk):
            cents[c] = x[labels == c].sum(axis=0)
        cents = _normalize_rows(cents)
        obj = float((x * cents[labels]).sum())
        return labels, cents, obj

    if sim is None:
        sim = x @ x.T
    rng = np.random.default_rng(seed)
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    for _ in range(max(1, n_init)):
        init = _kmeanspp_init(sim, k, rng)
        cents = x[init].copy()
        labels = np.full(n, -1, dtype=np.int64)
        for _it in range(max_iter):
            cs = x @ cents.T
            new_labels = np.argmax(cs, axis=1)
            # Re-seed empty clusters with the worst-fitting point.
            counts = np.bincount(new_labels, minlength=k)
            if (counts == 0).any():
                fit = cs[np.arange(n), new_labels]
                order = np.argsort(fit)
                ptr = 0
                for c in np.where(counts == 0)[0]:
                    while ptr < n and counts[new_labels[order[ptr]]] <= 1:
                        ptr += 1
                    if ptr >= n:
                        break
                    p = order[ptr]
                    counts[new_labels[p]] -= 1
                    new_labels[p] = c
                    counts[c] += 1
                    ptr += 1
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            onehot = np.zeros((k, n), dtype=x.dtype)
            onehot[labels, np.arange(n)] = 1.0
            cents = _normalize_rows(onehot @ x)
        obj = float((x * cents[labels]).sum())
        if best is None or obj > best[2]:
            best = (labels.copy(), cents.copy(), obj)
    assert best is not None
    return best


def silhouette_from_sim(sim: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette using cosine distance (1 - sim)."""
    n = sim.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2 or n < 3:
        return 0.0
    dist = 1.0 - sim
    onehot = (labels[:, None] == uniq[None, :]).astype(np.float64)
    sizes = onehot.sum(axis=0)
    sums = dist @ onehot  # (n, c): total distance to each cluster
    own = np.searchsorted(uniq, labels)
    own_size = sizes[own]
    a = np.where(own_size > 1, sums[np.arange(n), own] / np.maximum(own_size - 1, 1), 0.0)
    mean_other = sums / np.maximum(sizes[None, :], 1)
    mean_other[np.arange(n), own] = np.inf
    b = mean_other.min(axis=1)
    denom = np.maximum(np.maximum(a, b), _EPS)
    s = np.where(own_size > 1, (b - a) / denom, 0.0)
    return float(s.mean())


def choose_clusters(
    x: np.ndarray,
    sim: np.ndarray,
    max_k: int = 6,
    min_size_ratio: int = 4,
    min_silhouette: float = 0.03,
    tolerance: float = 0.85,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Pick k in [2, min(max_k, n // min_size_ratio)] by silhouette.

    Embedding neighbourhoods usually have weak structure, where silhouette
    favours k=2 which is too coarse to be useful. We therefore take the
    largest k whose silhouette is within ``tolerance`` of the best one.
    Falls back to a single cluster when n is small or structure is weak.
    Returns (labels, centroids, k, silhouette).
    """
    n = x.shape[0]
    upper = min(max_k, n // max(1, min_size_ratio))
    if n < 8 or upper < 2:
        labels, cents, _ = spherical_kmeans(x, 1)
        return labels, cents, 1, 0.0

    cands: List[Tuple[np.ndarray, np.ndarray, int, float]] = []
    for k in range(2, upper + 1):
        labels, cents, _ = spherical_kmeans(x, k, seed=seed, sim=sim, n_init=3)
        cands.append((labels, cents, k, silhouette_from_sim(sim, labels)))
    best_s = max(c[3] for c in cands)
    if best_s < min_silhouette:
        labels, cents, _ = spherical_kmeans(x, 1)
        return labels, cents, 1, best_s
    ok = [c for c in cands if c[3] >= tolerance * best_s]
    return max(ok, key=lambda c: c[2])


# --------------------------------------------------------------------------- #
# MMR
# --------------------------------------------------------------------------- #
def mmr_rerank(sim: np.ndarray, scores: np.ndarray, lam: float = 0.7) -> List[int]:
    """Greedy Maximal Marginal Relevance ordering of all items (local idx)."""
    n = sim.shape[0]
    if n == 0:
        return []
    selected: List[int] = []
    remaining = np.ones(n, dtype=bool)
    max_sim = np.full(n, -np.inf)
    for step in range(n):
        if step == 0:
            gain = scores.astype(np.float64).copy()
        else:
            gain = lam * scores - (1.0 - lam) * max_sim
        gain = np.where(remaining, gain, -np.inf)
        i = int(np.argmax(gain))
        selected.append(i)
        remaining[i] = False
        max_sim = np.maximum(max_sim, sim[i])
    return selected


# --------------------------------------------------------------------------- #
# 2D projection
# --------------------------------------------------------------------------- #
def pca_2d(x: np.ndarray, extra: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Project rows of x (and optional extra point) onto the top-2 PCs.

    Coordinates are jointly scaled into [-1, 1].
    """
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 2)), (np.zeros(2) if extra is not None else None)
    mean = x.mean(axis=0, keepdims=True)
    xc = (x - mean).astype(np.float64)
    if n == 1:
        pts = np.zeros((1, 2))
        ex = np.zeros(2) if extra is not None else None
        return pts, ex
    # Economy SVD on the small Gram side for speed (n << d).
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:2]
    if comps.shape[0] < 2:
        comps = np.vstack([comps, np.zeros_like(comps[:1])])
    # Deterministic sign: make the largest-|loading| positive.
    for r in range(2):
        j = int(np.argmax(np.abs(comps[r])))
        if comps[r, j] < 0:
            comps[r] = -comps[r]
    pts = xc @ comps.T
    ex = None
    if extra is not None:
        ex = (extra.astype(np.float64) - mean[0]) @ comps.T
    allp = pts if ex is None else np.vstack([pts, ex[None, :]])
    scale = float(np.abs(allp).max())
    if scale > _EPS:
        pts = pts / scale
        if ex is not None:
            ex = ex / scale
    return pts, ex


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
def analyze(
    top_idx: Sequence[int],
    vecs: np.ndarray,
    scores: np.ndarray,
    titles: Sequence[str],
    q_vec: Optional[np.ndarray] = None,
    dedup_threshold: float = 0.92,
    mmr_lambda: float = 0.7,
    max_k: int = 6,
    dedup: bool = True,
    seed: int = 0,
    title_threshold: float = 0.88,
) -> Dict[str, Any]:
    """Compute insights for the top-K results.

    Args:
        top_idx: global problem ids, sorted by score descending.
        vecs: (K, D) L2-normalised embeddings aligned with top_idx.
        scores: (K,) similarity to the query.
        titles: K titles.
        q_vec: (D,) normalised query embedding (for the scatter plot).
    """
    top_idx = [int(i) for i in top_idx]
    n = len(top_idx)
    empty = {
        "k": 0,
        "primary": [],
        "duplicates": {},
        "mmr_order": [],
        "clusters": [],
        "assign": {},
        "points": [],
        "query_point": None,
        "silhouette": 0.0,
    }
    if n == 0:
        return empty

    x = _normalize_rows(np.asarray(vecs, dtype=np.float32))
    sc = np.asarray(scores, dtype=np.float64)
    sim = (x @ x.T).astype(np.float64)
    np.fill_diagonal(sim, 1.0)

    if dedup:
        keys = [title_key(t) for t in titles]
        prim_local, dup_local = dedup_greedy(sim, dedup_threshold, keys, title_threshold)
    else:
        prim_local, dup_local = list(range(n)), {i: [] for i in range(n)}

    px = x[prim_local]
    psim = sim[np.ix_(prim_local, prim_local)]
    psc = sc[prim_local]

    labels, cents, k, sil = choose_clusters(px, psim, max_k=max_k, seed=seed)

    # Build cluster descriptors and order them by max score.
    raw_clusters = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        csim = px[members] @ cents[c]
        label_local = int(members[int(np.argmax(csim))])
        by_score = members[np.argsort(-psc[members], kind="stable")]
        raw_clusters.append(
            {
                "_raw": c,
                "label_local": label_local,
                "size": int(members.size),
                "mean_score": float(psc[members].mean()),
                "max_score": float(psc[members].max()),
                "top_locals": [int(m) for m in by_score[:3]],
            }
        )
    raw_clusters.sort(key=lambda d: -d["max_score"])
    remap = {d["_raw"]: new_id for new_id, d in enumerate(raw_clusters)}

    def g(local_in_prim: int) -> int:
        return top_idx[prim_local[local_in_prim]]

    clusters = []
    for new_id, d in enumerate(raw_clusters):
        ll = d["label_local"]
        clusters.append(
            {
                "id": new_id,
                "label": titles[prim_local[ll]],
                "label_pid": g(ll),
                "size": d["size"],
                "mean_score": round(d["mean_score"], 4),
                "max_score": round(d["max_score"], 4),
                "top_pids": [g(m) for m in d["top_locals"]],
                "top_titles": [titles[prim_local[m]] for m in d["top_locals"]],
            }
        )

    assign = {g(i): remap[int(labels[i])] for i in range(len(prim_local))}

    mmr_local = mmr_rerank(psim, psc, mmr_lambda)
    mmr_order = [g(i) for i in mmr_local]

    pts, qp = pca_2d(px, None if q_vec is None else np.asarray(q_vec, dtype=np.float32))
    points = [
        {
            "pid": g(i),
            "x": round(float(pts[i, 0]), 4),
            "y": round(float(pts[i, 1]), 4),
            "cluster": assign[g(i)],
            "score": round(float(psc[i]), 4),
            "title": titles[prim_local[i]],
            "dups": len(dup_local.get(prim_local[i], [])),
        }
        for i in range(len(prim_local))
    ]

    duplicates = {
        top_idx[p]: [top_idx[d] for d in ds] for p, ds in dup_local.items() if ds
    }

    return {
        "k": n,
        "primary": [top_idx[p] for p in prim_local],
        "duplicates": duplicates,
        "mmr_order": mmr_order,
        "clusters": clusters if len(clusters) > 1 else [],
        "assign": assign if len(clusters) > 1 else {},
        "points": points,
        "query_point": (
            None if qp is None else {"x": round(float(qp[0]), 4), "y": round(float(qp[1]), 4)}
        ),
        "silhouette": round(float(sil), 4),
    }
