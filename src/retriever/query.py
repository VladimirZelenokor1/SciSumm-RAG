import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retriever.index import (
    load_index,
    retrieve_candidates,
    rerank_candidates,
)


def embed_queries(
    query_texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """
    Embed a list of query strings to a numpy array of shape (len(query_texts), D).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    # normalize for IP search
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def load_queries(query_file: Path) -> List[str]:
    with open(query_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Perform retrieval and optional rerank for input queries."
    )
    parser.add_argument("--index", type=Path, required=True,
                        help="Path to FAISS index file")
    parser.add_argument("--ids", type=Path, required=True,
                        help="Path to JSON list of IDs corresponding to index")
    parser.add_argument("--chunks", type=Path,
                        help="Path to JSONL of chunk texts (for hybrid rerank)")
    parser.add_argument("--type", choices=["FlatIP","FlatL2","HNSW","IVFPQ","IVFOPQ"],
                        default="FlatIP",
                        help="Index type used to build the index, determines retrieval behavior")
    parser.add_argument("--query-file", type=Path, required=True,
                        help="File with one query per line")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for query embedding")
    parser.add_argument("--topk-coarse", type=int, default=100,
                        help="Number of coarse candidates to retrieve from FAISS")
    parser.add_argument("--topk", type=int, default=5,
                        help="Final number of top candidates to output")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/stsb-roberta-large",
                        help="Cross-encoder model for reranking")
    args = parser.parse_args()

    # Load index and IDs
    print(f"Loading index from {args.index} and IDs from {args.ids}...")
    index, ids = load_index(args.index, args.ids)

    # Load queries
    print(f"Loading queries from {args.query_file}...")
    queries = load_queries(args.query_file)

    # Embed queries
    print(f"Embedding {len(queries)} queries with model {args.embed_model}...")
    q_vecs = embed_queries(queries, model_name=args.embed_model)

    # Retrieve coarse candidates
    print(f"Retrieving top-{args.topk_coarse} coarse candidates...")
    coarse = retrieve_candidates(index, ids, q_vecs, args.topk_coarse)

    # If hybrid, rerank
    if args.chunks:
        print("Performing cross-encoder rerank...")
        final = rerank_candidates(
            coarse,
            args.chunks,
            args.rerank_model,
            args.topk,
            queries
        )
        # Output final
        for qi, reranked in enumerate(final):
            print(f"\nQuery {qi+1}: {queries[qi]}")
            for rank, (cid, score) in enumerate(reranked, start=1):
                pid, section, chunk_id = cid
                print(f"{rank}. {pid} | {section} | {chunk_id} → {score:.4f}")
    else:
        # No rerank, just show coarse topk
        print("Showing coarse top-k results (no rerank):")
        for qi, cand in enumerate(coarse):
            print(f"\nQuery {qi+1}: {queries[qi]}")
            for rank, (cid, score) in enumerate(cand[:args.topk], start=1):
                pid, section, chunk_id = cid
                print(f"{rank}. {pid} | {section} | {chunk_id} → {score:.4f}")


if __name__ == "__main__":
    main()
