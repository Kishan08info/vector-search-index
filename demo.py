"""Small RAG-style demo: retrieve with IVF and show what comes back.

Prints 5 example queries with their top-5 retrieved docs (doc id + topic),
and the topic precision@5 — i.e. how often the retrieved docs are actually
about the query's topic. This is the retrieval step a RAG pipeline (project 1)
would feed to its answerer.
"""

import numpy as np

from generate import DATA_DIR
from index import IVFIndex, BruteForceIndex, topic_precision_at_k

N_EXAMPLES = 5


def main() -> None:
    docs = np.load(DATA_DIR / "docs.npy")
    doc_topics = np.load(DATA_DIR / "doc_topics.npy")
    queries = np.load(DATA_DIR / "queries.npy")
    query_topic = np.load(DATA_DIR / "query_topic.npy")

    bf = BruteForceIndex(docs)
    ivf = IVFIndex(docs, nlist=64).build()
    gt = bf.search(queries, k=10)
    res = ivf.search(queries, k=5, nprobe=16)
    print(f"IVF topic precision@5 (nprobe=16): "
          f"{topic_precision_at_k(res, query_topic, doc_topics, k=5):.3f}")
    print(f"Brute-force topic precision@5: "
          f"{topic_precision_at_k(gt, query_topic, doc_topics, k=5):.3f}\n")
    for i in range(N_EXAMPLES):
        topics = [int(doc_topics[d]) for d in res[i] if d >= 0]
        print(f"query {i}: true topic {int(query_topic[i])} -> "
              f"retrieved docs {res[i].tolist()} topics {topics}")


if __name__ == "__main__":
    main()
