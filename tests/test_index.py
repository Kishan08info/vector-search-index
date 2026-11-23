"""Tests for the vector-search indexes. All assertions use tiny hand-built
cases or the seeded generator, so the suite is deterministic.
"""

import numpy as np
import pytest

from generate import generate
from index import IVFIndex, BruteForceIndex, recall_at_k, topic_precision_at_k


def _tiny():
    # 6 docs in 3 clear clusters; queries sit on top of a doc each.
    docs = np.array([
        [1.0, 0.0], [0.99, 0.14], [0.0, 1.0],
        [0.14, 0.99], [-1.0, 0.0], [-0.99, 0.14],
    ], dtype=np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    return docs


def test_brute_force_exact_top1():
    docs = _tiny()
    bf = BruteForceIndex(docs)
    res = bf.search(docs[:3], k=1)
    assert res[:, 0].tolist() == [0, 1, 2]


def test_brute_force_k_larger_than_n():
    docs = _tiny()
    bf = BruteForceIndex(docs)
    res = bf.search(docs[:1], k=20)
    assert res.shape == (1, 6)  # capped at n docs


def test_brute_force_empty_queries():
    docs = _tiny()
    bf = BruteForceIndex(docs)
    res = bf.search(np.empty((0, 2), dtype=np.float32), k=3)
    assert res.shape == (0, 0)


def test_ivf_recall_high_on_seeded_corpus():
    data = generate()
    bf = BruteForceIndex(data["docs"])
    gt = bf.search(data["queries"], k=10)
    ivf = IVFIndex(data["docs"], nlist=64).build()
    res = ivf.search(data["queries"], k=10, nprobe=16)
    assert recall_at_k(res, gt, k=10) >= 0.90


def test_ivf_recall_grows_with_nprobe():
    data = generate()
    bf = BruteForceIndex(data["docs"])
    gt = bf.search(data["queries"], k=10)
    ivf = IVFIndex(data["docs"], nlist=64).build()
    r1 = recall_at_k(ivf.search(data["queries"], k=10, nprobe=1), gt, k=10)
    r16 = recall_at_k(ivf.search(data["queries"], k=10, nprobe=16), gt, k=10)
    assert r16 > r1


def test_ivf_full_probe_matches_brute_force():
    data = generate()
    bf = BruteForceIndex(data["docs"])
    gt = bf.search(data["queries"][:20], k=10)
    ivf = IVFIndex(data["docs"], nlist=64).build()
    res = ivf.search(data["queries"][:20], k=10, nprobe=64)
    assert recall_at_k(res, gt, k=10) == 1.0


def test_ivf_deterministic_build():
    data = generate()
    a = IVFIndex(data["docs"], nlist=64).build()
    b = IVFIndex(data["docs"], nlist=64).build()
    assert np.allclose(a.centroids, b.centroids)
    assert [l.tolist() for l in a.lists] == [l.tolist() for l in b.lists]


def test_ivf_save_load_roundtrip(tmp_path):
    data = generate()
    ivf = IVFIndex(data["docs"], nlist=64).build()
    path = ivf.save(tmp_path / "ivf.npz")
    loaded = IVFIndex.load(path, data["docs"])
    r1 = ivf.search(data["queries"][:10], k=5, nprobe=8)
    r2 = loaded.search(data["queries"][:10], k=5, nprobe=8)
    assert np.array_equal(r1, r2)


def test_ivf_search_before_build_raises():
    docs = _tiny()
    ivf = IVFIndex(docs, nlist=2)
    with pytest.raises(RuntimeError):
        ivf.search(docs[:1], k=1)


def test_recall_at_k_hand_case():
    retrieved = np.array([[0, 1, 2], [3, 4, 5]])
    gt = np.array([[0, 1, 9], [3, 8, 9]])
    assert recall_at_k(retrieved, gt, k=3) == pytest.approx(3 / 6)


def test_topic_precision_hand_case():
    retrieved = np.array([[0, 1], [2, 3]])
    query_topics = np.array([7, 8])
    doc_topics = np.array([7, 7, 8, 9])
    assert topic_precision_at_k(retrieved, query_topics, doc_topics, k=2) == pytest.approx(0.75)


def test_generator_is_seeded():
    a = generate()
    b = generate()
    assert np.array_equal(a["docs"], b["docs"])
    assert np.array_equal(a["queries"], b["queries"])
