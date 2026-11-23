"""Vector search indexes: exact brute force and IVF (inverted file).

Both indexes use cosine similarity on unit-normalised vectors, the same
similarity a RAG retriever uses over embeddings. The IVF index mirrors the
design in FAISS (IVFFlat) and pgvector (IVFFlat): k-means training partitions
the space into `nlist` Voronoi cells; at search time only the `nprobe`
nearest cells are scanned, trading a little recall for a large latency win.

This is a from-scratch NumPy implementation for learning and benchmarking —
not a production vector database. No FAISS, no pgvector, no hosted service.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores per row, best first. scores: (nq, n)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


class BruteForceIndex:
    """Exact search: score every document. The ground truth for recall."""

    def __init__(self, docs: np.ndarray):
        self.docs = _unit(np.asarray(docs, dtype=np.float32))
        self.n, self.dim = self.docs.shape

    def search(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        q = _unit(np.asarray(queries, dtype=np.float32))
        if q.shape[0] == 0 or self.n == 0:
            return np.empty((q.shape[0], 0), dtype=np.int64)
        scores = q @ self.docs.T
        return _top_k(scores, k)


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    """Lloyd's k-means with k-means++ seeding. Distances via matmul
    (||x-c||^2 = ||x||^2 + ||c||^2 - 2x.c) so 100k docs stay memory-light.
    Returns (k, dim) centroids."""
    n, dim = x.shape
    x_sq = np.sum(x * x, axis=1)
    # k-means++ seeding.
    centroids = np.empty((k, dim), dtype=np.float32)
    centroids[0] = x[rng.integers(n)]
    closest = np.maximum(x_sq + np.sum(centroids[0] ** 2) - 2 * (x @ centroids[0]), 0)
    for j in range(1, k):
        probs = closest / closest.sum()
        centroids[j] = x[rng.choice(n, p=probs)]
        d = np.maximum(x_sq + np.sum(centroids[j] ** 2) - 2 * (x @ centroids[j]), 0)
        closest = np.minimum(closest, d)
    for _ in range(iters):
        # (n, k) squared distances, computed in chunks to bound memory.
        labels = np.empty(n, dtype=np.int64)
        for s in range(0, n, 20_000):
            chunk = x[s:s + 20_000]
            d = (x_sq[s:s + 20_000, None] + np.sum(centroids ** 2, axis=1)[None, :]
                 - 2 * (chunk @ centroids.T))
            labels[s:s + 20_000] = np.argmin(d, axis=1)
        new = centroids.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members):
                new[j] = members.mean(axis=0)
        shift = np.linalg.norm(new - centroids)
        centroids = new
        if shift < 1e-4:
            break
    return _unit(centroids)


class IVFIndex:
    """Inverted-file index: k-means cells, probe the nearest `nprobe` cells."""

    def __init__(self, docs: np.ndarray, nlist: int = 64, seed: int = 42):
        self.docs = _unit(np.asarray(docs, dtype=np.float32))
        self.n, self.dim = self.docs.shape
        self.nlist = nlist
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.lists: list[np.ndarray] | None = None
        self.train_time_s: float = 0.0
        self.build_time_s: float = 0.0

    def build(self) -> "IVFIndex":
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        self.centroids = _kmeans(self.docs, self.nlist, rng)
        self.train_time_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        # Assign every doc to its nearest centroid (cosine = dot on unit vecs).
        sims = self.docs @ self.centroids.T
        labels = np.argmax(sims, axis=1)
        self.lists = [np.where(labels == j)[0].astype(np.int64) for j in range(self.nlist)]
        self.build_time_s = time.perf_counter() - t1
        return self

    def search(self, queries: np.ndarray, k: int = 10, nprobe: int = 8) -> np.ndarray:
        if self.centroids is None or self.lists is None:
            raise RuntimeError("call build() before search()")
        q = _unit(np.asarray(queries, dtype=np.float32))
        if q.shape[0] == 0 or self.n == 0:
            return np.empty((q.shape[0], 0), dtype=np.int64)
        nprobe = min(max(nprobe, 1), self.nlist)
        # Nearest cells per query.
        cell_scores = q @ self.centroids.T
        probe_cells = _top_k(cell_scores, nprobe)
        # Batched per-cell scoring: for each cell, score all queries probing
        # it in one matmul, then merge each query's candidates across cells.
        cand_scores: list[list[np.ndarray]] = [[] for _ in range(q.shape[0])]
        cand_ids: list[list[np.ndarray]] = [[] for _ in range(q.shape[0])]
        for cell in range(self.nlist):
            q_idx = np.where((probe_cells == cell).any(axis=1))[0]
            if len(q_idx) == 0 or len(self.lists[cell]) == 0:
                continue
            scores = self.docs[self.lists[cell]] @ q[q_idx].T  # (cell_n, nq_cell)
            for col, qi in enumerate(q_idx):
                cand_scores[qi].append(scores[:, col])
                cand_ids[qi].append(self.lists[cell])
        out = np.full((q.shape[0], min(k, self.n)), -1, dtype=np.int64)
        for i in range(q.shape[0]):
            if not cand_scores[i]:
                continue
            s = np.concatenate(cand_scores[i])
            ids = np.concatenate(cand_ids[i])
            take = min(k, len(s))
            top = np.argpartition(-s, kth=take - 1)[:take]
            top = top[np.argsort(-s[top])]
            out[i, :take] = ids[top]
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path,
            centroids=self.centroids,
            lists=np.array([l for l in self.lists], dtype=object),
            nlist=self.nlist,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, docs: np.ndarray) -> "IVFIndex":
        z = np.load(path, allow_pickle=True)
        idx = cls(docs, nlist=int(z["nlist"]))
        idx.centroids = z["centroids"]
        idx.lists = [np.asarray(l, dtype=np.int64) for l in z["lists"]]
        return idx


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """Mean fraction of the ground-truth top-k present in the retrieved top-k."""
    hits = 0
    for r, g in zip(retrieved, ground_truth):
        r_set = set(int(x) for x in r[:k] if x >= 0)
        hits += len(r_set & set(int(x) for x in g[:k]))
    return hits / (len(ground_truth) * k)


def topic_precision_at_k(
    retrieved: np.ndarray, query_topics: np.ndarray, doc_topics: np.ndarray, k: int = 5
) -> float:
    """Mean share of retrieved docs whose topic matches the query's topic."""
    hits = 0
    total = 0
    for r, t in zip(retrieved, query_topics):
        valid = [int(x) for x in r[:k] if x >= 0]
        hits += sum(1 for x in valid if doc_topics[x] == t)
        total += len(valid)
    return hits / total if total else 0.0
