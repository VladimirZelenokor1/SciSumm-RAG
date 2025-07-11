# tests/test_index.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import faiss
from src.retriever.index import normalize_embeddings, build_flat_index, search


def test_normalize_embeddings():
    vecs = np.array([[3.0, 4.0]], dtype=np.float32)
    normed = normalize_embeddings(vecs)
    # Norm should be 1
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_build_flat_index_and_search_ip():
    # Two simple orthogonal vectors
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # Normalize for inner product
    normed = normalize_embeddings(vecs)
    idx = build_flat_index(dim=normed.shape[1], metric="IP")
    idx.add(normed)
    ids = [("a", "b", "c"), ("d", "e", "f")]
    # Query exactly matches first vector
    results = search(idx, ids, np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)
    assert results[0][0][0] == ("a", "b", "c")
    assert pytest.approx(results[0][0][1], rel=1e-6) == pytest.approx(1.0)


def test_build_flat_index_and_search_l2():
    # Vectors at distance
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    idx = build_flat_index(dim=vecs.shape[1], metric="L2")
    idx.add(vecs)
    ids = [("x", "y", "z"), ("u", "v", "w")]
    # Query closer to second vector
    results = search(idx, ids, np.array([[2.5, 3.5]], dtype=np.float32), top_k=1)
    assert results[0][0][0] == ("u", "v", "w")
    # L2 distance squared between [2.5,3.5] and [3,4] is (0.5^2+0.5^2)=0.5
    assert pytest.approx(results[0][0][1], rel=1e-6) == pytest.approx(0.5)


def test_invalid_metric_fallback_to_l2():
    # Unknown metric should produce an L2 index (fallback)
    idx = build_flat_index(dim=3, metric="UNKNOWN_METRIC")
    assert isinstance(idx, faiss.IndexFlatL2)
