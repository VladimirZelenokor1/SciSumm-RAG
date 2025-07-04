import argparse
import json
import random
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_embeddings(emb_path: Path, ids_path: Path) -> Tuple[List[Tuple[str,str,str]], np.ndarray]:
    vecs = np.load(emb_path)
    with open(ids_path, 'r', encoding='utf-8') as f:
        ids = json.load(f)
    if len(ids) != vecs.shape[0]:
        raise ValueError(f"Length mismatch: {len(ids)} ids vs {vecs.shape[0]} embeddings")
    return ids, vecs


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


def load_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def search(index: faiss.Index, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(queries, top_k)


def hybrid_search(
    index: faiss.Index,
    ids: List[Tuple[str,str,str]],
    queries: np.ndarray,
    chunks_file: Path,
    rerank_model: str,
    top_k_coarse: int,
    top_k: int
) -> List[List[Tuple[Tuple[str,str,str], float]]]:
    # load chunk texts
    chunk_texts = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            pid, sec, cid, txt = json.loads(line)
            chunk_texts[(pid, sec, cid)] = txt
    reranker = CrossEncoder(rerank_model)
    D_coarse, I_coarse = index.search(queries, top_k_coarse)
    results = []
    for scores, idxs in zip(D_coarse, I_coarse):
        cands = [ids[i] for i in idxs]
        texts = [chunk_texts[c] for c in cands]
        pairs = [("", t) for t in texts]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(cands, rerank_scores), key=lambda x: x[1], reverse=True)
        results.append(ranked[:top_k])
    return results


def evaluate_recall(
    ids: List[Tuple[str,str,str]],
    vecs: np.ndarray,
    index: faiss.Index,
    sample_size: int,
    top_k: int
) -> Tuple[float, float]:
    N = vecs.shape[0]
    idxs = random.sample(range(N), min(sample_size, N))
    exact_hits = 0
    paper_hits = 0
    for i in idxs:
        q = vecs[i:i+1]
        D, I = index.search(q, top_k)
        returned = I[0]
        true_id = ids[i]
        true_paper = true_id[0]
        if any(ids[j] == true_id for j in returned):
            exact_hits += 1
        if any(ids[j][0] == true_paper for j in returned):
            paper_hits += 1
    return exact_hits / len(idxs), paper_hits / len(idxs)


def main():
    parser = argparse.ArgumentParser(description="Test FAISS search quality")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--mode", choices=["flat","hnsw","ivf","hybrid"], default="flat")
    parser.add_argument("--chunks", type=Path, help="JSONL file for hybrid mode")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/stsb-roberta-large")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--topk-coarse", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    print("Loading embeddings and IDs...")
    ids, vecs = load_embeddings(args.embeddings, args.ids)
    print(f"Loaded {len(ids)} vectors of dim {vecs.shape[1]}")

    print("Normalizing embeddings...")
    vecs = normalize(vecs.astype('float32'))

    print(f"Loading index from {args.index}...")
    index = load_index(args.index)

    print(f"Loading OPQ matrix from...")
    opq_mat = faiss.read_VectorTransform("data/index/faiss_ivfopq.opq")

    vecs_opq = opq_mat.apply_py(vecs)

    print(f"Evaluating recall@{args.topk} on sample of {args.sample_size}... ")
    ex, pap = evaluate_recall(ids, vecs_opq, index, args.sample_size, args.topk)
    print(f"Exact chunk recall@{args.topk}: {ex:.3f}")
    print(f"Same-paper recall@{args.topk}: {pap:.3f}")

    if args.mode == "hybrid":
        if not args.chunks:
            parser.error("--chunks is required for hybrid mode")
        print("Running hybrid search on first query...")
        hybrid = hybrid_search(
            index, ids, vecs_opq[:1], args.chunks, args.rerank_model,
            args.topk_coarse, args.topk
        )
        print(hybrid)
    else:
        print("Running simple search on first query...")
        D, I = index.search(vecs_opq[:1], args.topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((ids[idx], float(score)))
        print(results)

if __name__ == "__main__":
    main()