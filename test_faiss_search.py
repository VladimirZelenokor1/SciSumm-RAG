import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
import time

from src.retriever.index import load_index, search, hybrid_search, load_embeddings, normalize_embeddings

IndexID = Tuple[str, str, str]


def recall_at_k(true_ids: List[IndexID], pred_ids: List[IndexID], k: int):
    return int(true_ids[0] in pred_ids[:k])

def paper_recall_at_k(true_ids: List[IndexID], pred_ids: List[IndexID], k: int):
    true_paper = true_ids[0][0]
    return int(any(pid[0] == true_paper for pid in pred_ids[:k]))

def precision_at_k(true_ids: List[IndexID], pred_ids: List[IndexID], k: int):
    # fraction of top-k that match the true paper
    true_paper = true_ids[0][0]
    hits = sum(1 for pid in pred_ids[:k] if pid[0] == true_paper)
    return hits / k

def reciprocal_rank(true_ids: List[IndexID], pred_ids: List[IndexID]):
    true_paper = true_ids[0][0]
    for rank, pid in enumerate(pred_ids, start=1):
        if pid[0] == true_paper:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(
    true_ids: List[IndexID],
    pred_ids: List[IndexID],
    k: int
) -> float:
    true_paper = true_ids[0][0]

    # 1) Collecting relevance in predictions
    rels = [1.0 if pid[0] == true_paper else 0.0
            for pid in pred_ids[:k]]

    # 2) DCG
    dcg = 0.0
    for i, rel in enumerate(rels, start=1):
        dcg += rel / math.log2(i + 1)

    # 3) IDCG: equal to the sum of the first R positions, where R = sum(rels)
    R = int(sum(rels))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, R + 1))

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_flat(
    ids: List[IndexID],
    vecs: np.ndarray,
    index: faiss.Index,
    sample_size: int,
    top_k: int
):
    N = vecs.shape[0]
    idxs = random.sample(range(N), min(sample_size, N))
    rec_exact, rec_paper, prec, mrr, ndcg = 0, 0, 0.0, 0.0, 0.0
    t_search = 0.0

    for i in idxs:
        q = vecs[i : i + 1]
        true = [ids[i]]
        t0 = time.time()
        results = search(index, ids, q, top_k)
        t_search += time.time() - t0
        preds = [pid for pid, _ in results[0]]

        rec_exact += recall_at_k(true, preds, top_k)
        rec_paper += paper_recall_at_k(true, preds, top_k)
        prec    += precision_at_k(true, preds, top_k)
        mrr     += reciprocal_rank(true, preds)
        ndcg    += ndcg_at_k(true, preds, top_k)

    n = len(idxs)
    return {
        "recall@{}".format(top_k): rec_exact / n,
        "paper_recall@{}".format(top_k): rec_paper / n,
        "precision@{}".format(top_k): prec / n,
        "MRR@{}".format(top_k): mrr / n,
        "nDCG@{}".format(top_k): ndcg / n,
        "avg_search_time": t_search / n
    }


def evaluate_hybrid(
    ids: List[IndexID],
    vecs: np.ndarray,
    index: faiss.Index,
    chunks_path: Path,
    rerank_model: str,
    sample_size: int,
    top_k_coarse: int,
    top_k: int
):
    N = vecs.shape[0]
    idxs = random.sample(range(N), min(sample_size, N))
    rec_exact, rec_paper, prec, mrr, ndcg = 0, 0, 0.0, 0.0, 0.0
    t_coarse, t_rerank = 0.0, 0.0

    for i in idxs:
        qvec = vecs[i : i + 1]
        true = [ids[i]]

        # coarse
        t0 = time.time()
        D_r, _ = index.search(qvec, top_k_coarse)
        t_coarse += time.time() - t0

        # hybrid (using our function)
        t0 = time.time()
        results = hybrid_search(
            coarse_idx=index,
            ids=ids,
            queries=qvec,
            query_texts=[true[0][0]],
            chunks_path=chunks_path,
            rerank_model=rerank_model,
            top_k_coarse=top_k_coarse,
            top_k=top_k
        )
        t_rerank += time.time() - t0

        preds = [pid for pid, _ in results[0]]

        rec_exact += recall_at_k(true, preds, top_k)
        rec_paper += paper_recall_at_k(true, preds, top_k)
        prec    += precision_at_k(true, preds, top_k)
        mrr     += reciprocal_rank(true, preds)
        ndcg    += ndcg_at_k(true, preds, top_k)

    n = len(idxs)
    return {
        "recall@{}".format(top_k): rec_exact / n,
        "paper_recall@{}".format(top_k): rec_paper / n,
        "precision@{}".format(top_k): prec / n,
        "MRR@{}".format(top_k): mrr / n,
        "nDCG@{}".format(top_k): ndcg / n,
        "avg_coarse_time": t_coarse / n,
        "avg_rerank_time": t_rerank / n
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--ids",        type=Path, required=True)
    parser.add_argument("--index",      type=Path, required=True)
    parser.add_argument("--chunks",     type=Path, help="JSONL for hybrid")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--mode", choices=["flat","hybrid"], default="flat")
    parser.add_argument("--topk",       type=int, default=5)
    parser.add_argument("--topk-coarse",type=int, default=100)
    parser.add_argument("--sample-size",type=int, default=1000)
    args = parser.parse_args()

    print("Load embeddings/ids …")
    ids, vecs = load_embeddings(args.embeddings, args.ids)
    vecs = normalize_embeddings(vecs.astype("float32"))

    print("Load index …")
    dim = vecs.shape[1]
    # If you're using an OPQ+IVF index, you may also need to pass use_opq=True
    index, _ = load_index(str(args.index).split('.', 1)[0], dim)

    if args.mode == "flat":
        print(f"Evaluating flat retrieval @ topk={args.topk}")
        metrics = evaluate_flat(ids, vecs, index, args.sample_size, args.topk)
    else:
        if not args.chunks:
            parser.error("--chunks required for hybrid")
        print(f"Evaluating hybrid retrieval @ topk_coarse={args.topk_coarse}, topk={args.topk}")
        metrics = evaluate_hybrid(
            ids, vecs, index, args.chunks, args.rerank_model,
            args.sample_size, args.topk_coarse, args.topk
        )

    print("\n=== Metrics ===")
    for k,v in metrics.items():
        print(f"{k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()