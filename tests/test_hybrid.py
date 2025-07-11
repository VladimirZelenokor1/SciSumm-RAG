import pytest
import numpy as np
import json

from src.retriever.index import build_flat_index, hybrid_search

def test_hybrid_search_rerank(tmp_path):
    # prepare temporary chunks file
    chunks_file = tmp_path / "chunks.jsonl"
    data = [
        ["id0", "sec", "0", "apple red"],
        ["id1", "sec", "1", "banana yellow"]
    ]
    with open(chunks_file, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

    # build simple FAISS index with two 2D vectors
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    idx = build_flat_index(dim=2, metric="L2")
    idx.add(vecs)
    ids = [(d[0], d[1], d[2]) for d in data]

    # dummy reranker: score=1 if "apple" in text else 0
    class DummyReranker:
        def predict(self, pairs):
            return [1.0 if "apple" in txt else 0.0 for _, txt in pairs]
    reranker = DummyReranker()

    query_texts = ["apple"]
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    results = hybrid_search(
        coarse_idx=idx,
        ids=ids,
        queries=queries,
        query_texts=query_texts,
        chunks_path=chunks_file,
        rerank_model=reranker,
        top_k_coarse=2,
        top_k=2
    )
    # Expect the "apple" chunk first
    assert results[0][0][0] == ("id0", "sec", "0")
    assert results[0][1][0] == ("id1", "sec", "1")