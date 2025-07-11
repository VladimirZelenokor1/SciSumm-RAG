import os

import numpy as np
import logging
import json
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import argparse
from sentence_transformers import CrossEncoder

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# CPU-only FAISS indexer for RAG pipeline
IndexID = Tuple[str, str, str]  # paper_id, section, chunk_id


def load_embeddings(path_vec: Path, path_ids: Path) -> Tuple[List[Tuple[str,str,str]], np.ndarray]:
    vectors = np.load(path_vec)  # shape (N, D)
    with open(path_ids, "r", encoding="utf-8") as f:
        raw_ids = json.load(f)
    ids = [tuple(x) for x in raw_ids]
    if len(ids) != vectors.shape[0]:
        raise ValueError("Mismatch between IDs and embedding vectors: {} vs {}".format(
            len(ids), vectors.shape[0]
        ))
    return ids, vectors


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norm, 1e-12, None)


def build_flat_index(dim: int, metric: str = "IP") -> faiss.Index:
    """
    Flat (brute-force) index.
    metric: “IP” or “L2”
    """
    if metric.upper() == "IP":
        return faiss.IndexFlatIP(dim)
    else:
        return faiss.IndexFlatL2(dim)


def build_hnsw_index(dim: int,
                     M: int = 64,
                     ef_construction: int = 200,
                     ef_search: int = 200,
                     metric: str = "IP") -> faiss.Index:
    """
    HNSW index. The parameters M, efConstruction, efSearch can be fine-tuned.
    Remember: memory ~ O(N * M).
    """
    space = faiss.METRIC_INNER_PRODUCT if metric.upper() == "IP" else faiss.METRIC_L2
    idx = faiss.IndexHNSWFlat(dim, M, space)
    idx.hnsw.efConstruction = ef_construction
    idx.hnsw.efSearch = ef_search
    return idx


def apply_opq(vectors: np.ndarray, M: int = 64) -> Tuple[np.ndarray, faiss.OPQMatrix]:
    D = vectors.shape[1]
    opq = faiss.OPQMatrix(D, M)
    opq.train(vectors)
    return opq.apply_py(vectors), opq


def build_ivfopq_index(dim: int,
                       nlist: int = 1024,
                       m_pq: int = 64,
                       nbits: int = 8,
                       use_opq: bool = True,
                       metric: str = "IP"
                       ) -> Tuple[faiss.Index, Optional[faiss.OPQMatrix]]:
    """
    IVF-PQ (+ OPQ) index.
 nlist: number of clusters, m_pq: number of sub-vectors, nbits: bits per sub-vector.
    If use_opq=True, return the OPQ matrix for training/saving.
    """
    space = faiss.METRIC_INNER_PRODUCT if metric.upper() == "IP" else faiss.METRIC_L2
    quantizer = faiss.IndexFlatIP(dim) if metric.upper() == "IP" else faiss.IndexFlatL2(dim)

    opq_matrix: Optional[faiss.OPQMatrix] = None

    if use_opq:
        # OPQ is trained on the original data
        opq_matrix = faiss.OPQMatrix(dim, m_pq)
        # create the IVF-PQ itself wrapped in an OPQ transform
        ivfpq = faiss.IndexPreTransform(
            opq_matrix,
            faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, space)
        )
    else:
        ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, space)

    return ivfpq, opq_matrix


def save_index(index: faiss.Index,
               ids: List[IndexID],
               path_prefix: str,
               opq_matrix: Optional[faiss.OPQMatrix] = None
               ) -> None:
    """
    Save the index and metadata.
    If there is an OPQ matrix, save it separately.
    """
    out_dir = os.path.dirname(path_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # The FAISS file itself
    faiss.write_index(index, f"{path_prefix}.index")
    # Metadata
    with open(f"{path_prefix}_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    # OPQ matrix is also separate
    if opq_matrix is not None:
        faiss.write_VectorTransform(opq_matrix, f"{path_prefix}.opq")


def load_index(path_prefix: str,
               dim: int,
               metric: str = "IP",
               use_opq: bool = False
               ) -> Tuple[faiss.Index, List[IndexID]]:
    """
    Load the index, OPQ matrix and metadata.
    If use_opq=True, wait for the .opq file and wrap the index in OPQ.
    """
    # Metadata
    with open(f"{path_prefix}_ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)

    # Main index
    idx = faiss.read_index(f"{path_prefix}.index")

    if use_opq:
        # wrap the loaded IVF-PQ in OPQ preprocessing
        opq = faiss.read_VectorTransform(f"{path_prefix}.opq")
        idx = faiss.IndexPreTransform(opq, idx)

    # For HNSW you can reconfigure efSearch after loading
    if hasattr(idx, "hnsw") and hasattr(idx.hnsw, "efSearch"):
        idx.hnsw.efSearch = getattr(idx.hnsw, "efSearch", 200)

    return idx, ids


def retrieve_candidates(
    index: faiss.Index,
    ids: List[Tuple[str,str,str]],
    queries: np.ndarray,
    top_k_coarse: int
) -> List[List[Tuple[Tuple[str,str,str], float]]]:
    D, I = index.search(queries, top_k_coarse)
    candidates = []
    for scores, idxs in zip(D, I):
        row = [(ids[i], float(scores[j])) for j,i in enumerate(idxs)]
        candidates.append(row)
    return candidates


def search(
    index: faiss.Index,
    ids: List[IndexID],
    queries: np.ndarray,
    top_k: int = 5
) -> List[List[Tuple[IndexID, float]]]:
    """
    Exact k-NN: returns for each query top-k (id, score).
    """
    # FAISS expects float32
    q = queries.astype(np.float32)
    distances, indices = index.search(q, top_k)

    results: List[List[Tuple[IndexID, float]]] = []
    for dist_row, idx_row in zip(distances, indices):
        row: List[Tuple[IndexID, float]] = []
        for rank, idx in enumerate(idx_row):
            if idx < 0:
                continue
            row.append((ids[idx], float(dist_row[rank])))
        results.append(row)
    return results


def hybrid_search(
    coarse_idx: faiss.Index,
    ids: List[IndexID],
    queries: np.ndarray,
    query_texts: List[str],
    chunks_path: Path,
    rerank_model: Union[str, CrossEncoder],
    top_k_coarse: int = 100,
    top_k: int = 5
) -> List[List[Tuple[IndexID, float]]]:
    """
    Mixed search: first FAISS, then re-ranking by CrossEncoder.
    """
    # load chunk texts
    chunk_texts: Dict[IndexID, str] = {}
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            pid, sec, cid, txt = json.loads(line)
            chunk_texts[(pid, sec, cid)] = txt

    if isinstance(rerank_model, str):
        reranker = CrossEncoder(rerank_model)
    else:
        reranker = rerank_model

    # coarse FAISS
    Dc, Ic = coarse_idx.search(queries.astype(np.float32), top_k_coarse)

    results: List[List[Tuple[IndexID, float]]] = []
    for i, (dist_row, idx_row) in enumerate(zip(Dc, Ic)):
        qtext = query_texts[i]

        # collect candidates and their texts
        cands = [tuple(ids[j]) for j in idx_row if j >= 0]
        texts = [chunk_texts[c] for c in cands]

        # form pairs (query, chunk) and make predictions
        pairs = [(qtext, chunk) for chunk in texts]
        rerank_scores = reranker.predict(pairs)

        # sort by score desc and give top_k
        ranked = sorted(zip(cands, rerank_scores),
                        key=lambda x: x[1], reverse=True)[:top_k]
        results.append(ranked)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--index-prefix", type=Path, default=Path("data/index/faiss"))
    p.add_argument(
        "--type",
        choices=["FlatIP", "FlatL2", "HNSW", "IVFPQ", "OPQIVFPQ"],
        default="FlatIP"
    )
    p.add_argument("--chunks", type=Path, help="для HYBRID reranking")
    p.add_argument("--rerank-model", type=str, default="cross-encoder/stsb-roberta-large")
    p.add_argument("--topk-coarse", type=int, default=100)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nlist", type=int, default=256, help="IVFPQ: число кластеров")
    p.add_argument("--m", type=int, default=16, help="IVFPQ: число под-векторов")
    p.add_argument("--nbits", type=int, default=4, help="IVFPQ: бит на под-вектор")
    p.add_argument("--M", type=int, default=64, help="HNSW: M")
    p.add_argument("--efC", type=int, default=200, help="HNSW: efConstruction")
    p.add_argument("--efS", type=int, default=200, help="HNSW: efSearch")

    args = p.parse_args()

    # load embeddings + ids
    ids_list, vecs = load_embeddings(args.embeddings, args.ids)
    # if IP metric - normalize
    vecs = normalize_embeddings(vecs)

    dim = vecs.shape[1]
    metric = "IP" if args.type.endswith("IP") or args.type.endswith("PQ") else "L2"
    index = None
    opq_matrix = None

    if args.type in ("FlatIP", "FlatL2"):
        index = build_flat_index(dim, metric=metric)
    elif args.type == "HNSW":
        index = build_hnsw_index(
            dim,
            M=args.M,
            ef_construction=args.efC,
            ef_search=args.efS,
            metric=metric
        )
    elif args.type == "IVFPQ":
        n_vectors = vecs.shape[0]
        requested_nlist = args.nlist
        actual_nlist = min(requested_nlist, n_vectors)
        if actual_nlist < requested_nlist:
            logger.warning(
                "Requested nlist=%d but only %d vectors available; reducing to %d",
                requested_nlist, n_vectors, actual_nlist
            )
        ivf, _ = build_ivfopq_index(
            dim,
            nlist=actual_nlist,
            m_pq=args.m,
            nbits=args.nbits,
            use_opq=False,
            metric=metric
        )
        ivf.train(vecs.astype(np.float32))
        index = ivf
    elif args.type == "OPQIVFPQ":
        n_vectors = vecs.shape[0]
        requested_nlist = args.nlist
        actual_nlist = min(requested_nlist, n_vectors)
        if actual_nlist < requested_nlist:
            logger.warning(
                "Requested nlist=%d but only %d vectors available; reducing to %d",
                requested_nlist, n_vectors, actual_nlist
            )
        ivf, opq_matrix = build_ivfopq_index(
            dim,
            nlist=actual_nlist,
            m_pq=args.m,
            nbits=args.nbits,
            use_opq=True,
            metric=metric
        )

        opq_matrix.train(vecs.astype(np.float32))

        vecs_opq = opq_matrix.apply(vecs.astype(np.float32))
        ivf.train(vecs_opq)

        index = faiss.IndexPreTransform(opq_matrix, ivf)

    # Add all the vectors
    index.add(vecs.astype(np.float32))

    # Save the index and metadata (including .opq, if any)
    save_index(index, ids_list, str(args.index_prefix), opq_matrix)