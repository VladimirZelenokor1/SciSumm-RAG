# tests/test_index.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.retriever.index import (
    normalize_embeddings, build_flat_index, search
)

def test_normalize_embeddings():
    vecs = np.array([[3.0, 4.0]])
    normed = normalize_embeddings(vecs)
    # Check that the norm is 1
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)

def test_build_flat_index_and_search():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Normalize for IP metrics
    normed = normalize_embeddings(vecs)
    idx = build_flat_index(normed, metric="IP")
    ids = [("a","b","c"), ("d","e","f")]
    # Request for “unit on first coordinate”
    res = search(idx, ids, np.array([[1.0, 0.0]]), top_k=1)
    assert res[0][0][0] == ("a","b","c")
    assert pytest.approx(res[0][0][1], rel=1e-3) == 1.0

def test_invalid_metric():
    vecs = np.array([[1.0]])
    with pytest.raises(ValueError):
        build_flat_index(vecs, metric="unknown")
