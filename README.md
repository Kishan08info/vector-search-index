# Vector Search Index — IVF Retrieval Layer for RAG

**SYNTHETIC DATA** — every number below comes from a seeded, re-runnable demo
run over synthetic embeddings (seed 42). This is a portfolio skill demo,
**not** work experience.

A from-scratch **inverted-file (IVF) vector index** in NumPy — the retrieval
layer a RAG system actually runs on — benchmarked against exact brute-force
search on the same corpus. The design mirrors FAISS `IVFFlat` and pgvector
`IVFFlat`: k-means partitions the embedding space into cells; at query time
only the `nprobe` nearest cells are scanned.

## Why this exists

AI Engineer postings keep asking for **vector search** as a core skill —
Clara ("semantic search and RAG systems, vector similarity search"), FedEx
AI Engineer I ("high-performance RAG architectures — Vertex AI Vector Search,
pgvector"), Northern Trust ("RAG with PostgreSQL + pgvector"). Projects 1 and
3 built RAG systems on TF-IDF retrieval; no project covered the vector-index
layer itself — how an embedding index trades recall for latency, and where
that trade-off actually pays off.

## Business question

> A RAG service holds 100,000 document embeddings. Can an IVF index answer
> top-10 retrieval questions much faster than scanning everything, without
> losing the documents that matter — and where does the recall/latency
> trade-off stop being worth it?

## What it does

- `generate.py` — synthetic corpus: 100,000 docs, 128-dim unit embeddings in
  50 topic clusters; 200 queries (150 noisy copies of real docs, 50 noisy
  topic centroids). Seeded, re-runnable.
- `index.py` — `BruteForceIndex` (exact cosine search, the ground truth) and
  `IVFIndex` (k-means++ training, inverted lists, batched per-cell scoring,
  `save`/`load` to `.npz`), plus `recall_at_k` and `topic_precision_at_k`.
- `bench.py` — recall@10 vs brute force, per-query latency (mean/median/p95),
  speedup, and docs scanned, for nprobe 1/4/8/16/32.
- `demo.py` — RAG-style demo: 5 example queries with retrieved docs and
  their topics.
- `run.py` — end-to-end: generate → build → benchmark → demo.
- `tests/test_index.py` — 12 pytest checks, incl. determinism, save/load
  round-trip, and recall growing with nprobe.

## Results (real run, seed 42)

Corpus: 100,000 docs × 128 dims, 50 topics, 200 queries. IVF: nlist=64,
train ~11s, build ~0.2s. Brute force: 2.66 ms/query (median), topic
precision@5 0.928, source doc in top-10 for 150/150 doc-derived queries.
Latency is the median across 3 full re-runs (full table in
`outputs/benchmark-3runs.md`); recall, topic precision and docs scanned are
identical in every run.

| nprobe | recall@10 | topic P@5 | ms/query | speedup | docs scanned |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8300 | 0.984 | 1.01 | 2.82x | 1,862 (1.9%) |
| 4 | 0.9650 | 0.953 | 1.05 | 2.48x | 6,503 (6.5%) |
| 8 | 0.9845 | 0.939 | 1.52 | 1.72x | 12,512 (12.5%) |
| 16 | 0.9945 | 0.932 | 1.97 | 1.36x | 24,930 (24.9%) |
| 32 | 0.9995 | 0.928 | 2.60 | 1.01x | 49,757 (49.8%) |

Headline finding: **nprobe=4 is the sweet spot — 0.965 recall at ~2.5x speedup
while scanning only 6.5% of the corpus.** Probing more cells buys almost no
recall (0.965 → 0.9995) while the speedup evaporates (2.5x → 1.0x), because
at 100k docs a single BLAS brute-force matmul is already fast — the index
only wins when it scans a small slice. That is the honest engineering point:
IVF pays off when the corpus is large enough that a full scan hurts, and the
right nprobe is the one where the recall curve flattens, not the maximum.
(Timings were measured on a shared-CPU VM and vary run to run — one run's
nprobe=8 median was 4.38 ms vs 1.49–1.52 ms in the other two — so treat
speedups as approximate; recall is exact.)

12/12 pytest checks pass.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py            # generate + benchmark + demo (~1 min)
python -m pytest tests/ -q
```

## Honest labelling

- Synthetic embeddings everywhere — no real documents, no real embedding
  model, no FAISS/pgvector. The IVF behaviour is a real implementation of
  the algorithm, not a simulation of it, but it is NumPy on one machine:
  no production vector-database experience is claimed.
- Topic precision@5 uses the synthetic topic labels as ground truth; a real
  RAG eval would use human relevance judgements (see project 17).
- Latency is single-machine NumPy on 100k docs; at 10M+ docs the brute-force
  baseline would be far slower and the IVF speedup far larger.

## 60-second pitch

"I built the retrieval layer RAG runs on from scratch: a k-means IVF index
over 100,000 embeddings, benchmarked against exact search. At nprobe=4 it
keeps 96.5% recall at 2.5x the speed while scanning 6.5% of the corpus, and
the benchmark shows exactly where probing more stops paying — recall flattens
while the speedup disappears. That is the trade-off I would tune first on any
real vector store."
