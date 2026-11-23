"""Benchmark brute-force vs IVF search on the generated corpus.

For each nprobe value, measures recall@10 against the brute-force ground
truth, per-query latency (mean / p50 / p95), and the search speedup. Writes
outputs/benchmark.json and outputs/benchmark.md.
"""

import json
import time
from pathlib import Path

import numpy as np

from generate import DATA_DIR
from index import IVFIndex, BruteForceIndex, recall_at_k, topic_precision_at_k

K = 10
NPROBES = [1, 4, 8, 16, 32]
NLIST = 64
OUT_DIR = Path(__file__).parent / "outputs"


def _latency(fn, queries, reps: int = 7) -> dict:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(queries)
        times.append((time.perf_counter() - t0) / len(queries) * 1000)  # ms/query
    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "p50_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}


def main() -> None:
    docs = np.load(DATA_DIR / "docs.npy")
    doc_topics = np.load(DATA_DIR / "doc_topics.npy")
    queries = np.load(DATA_DIR / "queries.npy")
    query_doc = np.load(DATA_DIR / "query_doc.npy")
    query_topic = np.load(DATA_DIR / "query_topic.npy")

    bf = BruteForceIndex(docs)
    t0 = time.perf_counter()
    gt = bf.search(queries, k=K)
    bf_total = time.perf_counter() - t0
    bf_lat = _latency(lambda q: bf.search(q, k=K), queries)
    bf_topic_p5 = topic_precision_at_k(gt, query_topic, doc_topics, k=5)
    # Sanity: doc-derived queries should retrieve their own source doc.
    src_hits = sum(
        1 for i, d in enumerate(query_doc) if d >= 0 and d in set(gt[i].tolist())
    )
    src_total = int((query_doc >= 0).sum())

    ivf = IVFIndex(docs, nlist=NLIST).build()
    rows = []
    for nprobe in NPROBES:
        res = ivf.search(queries, k=K, nprobe=nprobe)
        lat = _latency(lambda q, np_=nprobe: ivf.search(q, k=K, nprobe=np_), queries)
        scanned = float(np.mean([sum(len(ivf.lists[c]) for c in
                                    np.argsort(-(queries @ ivf.centroids.T), axis=1)[i, :nprobe])
                                 for i in range(len(queries))]))
        rows.append({
            "nprobe": nprobe,
            "recall_at_10": recall_at_k(res, gt, k=K),
            "topic_precision_at_5": topic_precision_at_k(res, query_topic, doc_topics, k=5),
            "mean_ms_per_query": lat["mean_ms"],
            "p50_ms_per_query": lat["p50_ms"],
            "p95_ms_per_query": lat["p95_ms"],
            "speedup_vs_brute_force": bf_lat["median_ms"] / lat["median_ms"],
            "avg_docs_scanned": scanned,
            "scan_fraction": scanned / len(docs),
        })

    summary = {
        "corpus": {"n_docs": int(len(docs)), "dim": int(docs.shape[1]),
                   "n_topics": int(doc_topics.max()) + 1, "n_queries": int(len(queries))},
        "brute_force": {
            "total_s": bf_total,
            "mean_ms_per_query": bf_lat["mean_ms"],
            "p50_ms_per_query": bf_lat["p50_ms"],
            "p95_ms_per_query": bf_lat["p95_ms"],
            "topic_precision_at_5": bf_topic_p5,
            "source_doc_in_top10": f"{src_hits}/{src_total}",
        },
        "ivf": {"nlist": NLIST, "train_time_s": ivf.train_time_s,
                "build_time_s": ivf.build_time_s},
        "results": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Vector search benchmark (synthetic data, seed 42)", "",
             f"Corpus: {len(docs)} docs, {docs.shape[1]} dims, "
             f"{int(doc_topics.max()) + 1} topics, {len(queries)} queries.",
             f"Brute force: {bf_lat['p50_ms']:.3f} ms/query median, "
             f"topic precision@5 {bf_topic_p5:.3f}, source doc in top-10 "
             f"{src_hits}/{src_total}.",
             f"IVF: nlist={NLIST}, train {ivf.train_time_s:.2f}s, "
             f"build {ivf.build_time_s:.3f}s.", "",
             "| nprobe | recall@10 | topic P@5 | ms/query (median) | speedup | docs scanned |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['nprobe']} | {r['recall_at_10']:.4f} | "
                     f"{r['topic_precision_at_5']:.4f} | {r['p50_ms_per_query']:.3f} | "
                     f"{r['speedup_vs_brute_force']:.1f}x | {r['avg_docs_scanned']:.0f} "
                     f"({r['scan_fraction']:.1%}) |")
    (OUT_DIR / "benchmark.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
