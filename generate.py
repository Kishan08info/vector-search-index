"""Generate a synthetic embedding corpus for the vector-search demo.

Corpus: 100,000 documents, 128-dim unit embeddings, clustered around 50 topic
centroids so nearest neighbours are meaningful (a flat random cloud would make
every query's neighbours arbitrary). Queries: 200 vectors, each a noisy copy of
a random corpus document (so the true nearest neighbour is known) or a noisy
copy of a topic centroid. Everything is seeded (seed 42) and re-runnable.

Outputs (data/):
  docs.npy      float32 (10000, 128), unit-normalised
  doc_topics.npy int64 (10000,), topic id 0..49 per document
  queries.npy   float32 (200, 128), unit-normalised
  query_doc.npy int64 (200,), source doc id for doc-derived queries (-1 for
                centroid-derived queries)
  query_topic.npy int64 (200,), true topic id per query
"""

import json
from pathlib import Path

import numpy as np

SEED = 42
N_DOCS = 100_000
N_TOPICS = 50
DIM = 128
N_QUERIES = 200
DOC_QUERY_FRAC = 0.75  # share of queries derived from a real document
DOC_NOISE = 0.15       # per-dim noise on a document (norm ~1.7 vs centroid 1.0:
                       # clusters overlap a little, so recall is a real trade-off)
NOISE_DOC = 0.08       # noise added to a doc-derived query
NOISE_CENTROID = 0.15  # noise added to a centroid-derived query

DATA_DIR = Path(__file__).parent / "data"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def generate(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    # Topic centroids: random directions, unit-normalised.
    centroids = _unit(rng.normal(size=(N_TOPICS, DIM)).astype(np.float32))
    # Documents: each doc belongs to one topic; doc = centroid + gaussian noise.
    doc_topics = rng.integers(0, N_TOPICS, size=N_DOCS)
    docs = centroids[doc_topics] + DOC_NOISE * rng.normal(size=(N_DOCS, DIM)).astype(np.float32)
    docs = _unit(docs).astype(np.float32)

    # Queries.
    n_doc_q = int(N_QUERIES * DOC_QUERY_FRAC)
    query_doc = np.full(N_QUERIES, -1, dtype=np.int64)
    query_topic = np.empty(N_QUERIES, dtype=np.int64)
    queries = np.empty((N_QUERIES, DIM), dtype=np.float32)
    src_docs = rng.choice(N_DOCS, size=n_doc_q, replace=False)
    for i, d in enumerate(src_docs):
        queries[i] = _unit(
            (docs[d] + NOISE_DOC * rng.normal(size=DIM)).astype(np.float32)[None, :]
        )[0]
        query_doc[i] = d
        query_topic[i] = doc_topics[d]
    for j in range(n_doc_q, N_QUERIES):
        t = int(rng.integers(0, N_TOPICS))
        queries[j] = _unit(
            (centroids[t] + NOISE_CENTROID * rng.normal(size=DIM)).astype(np.float32)[None, :]
        )[0]
        query_topic[j] = t
    return {
        "docs": docs,
        "doc_topics": doc_topics,
        "queries": queries,
        "query_doc": query_doc,
        "query_topic": query_topic,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = generate()
    np.save(DATA_DIR / "docs.npy", out["docs"])
    np.save(DATA_DIR / "doc_topics.npy", out["doc_topics"])
    np.save(DATA_DIR / "queries.npy", out["queries"])
    np.save(DATA_DIR / "query_doc.npy", out["query_doc"])
    np.save(DATA_DIR / "query_topic.npy", out["query_topic"])
    meta = {
        "seed": SEED,
        "n_docs": N_DOCS,
        "n_topics": N_TOPICS,
        "dim": DIM,
        "n_queries": N_QUERIES,
        "doc_query_frac": DOC_QUERY_FRAC,
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {N_DOCS} docs x {DIM} dims, {N_QUERIES} queries -> {DATA_DIR}")


if __name__ == "__main__":
    main()
