import numpy as np
import json
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from sentence_transformers import CrossEncoder

# CPU-only FAISS indexer for RAG pipeline


def load_embeddings(path_vec: Path, path_ids: Path) -> Tuple[List[Tuple[str,str,str]], np.ndarray]:
    vectors = np.load(path_vec)  # shape (N, D)
    with open(path_ids, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if len(ids) != vectors.shape[0]:
        raise ValueError("Mismatch between IDs and embedding vectors: {} vs {}".format(
            len(ids), vectors.shape[0]
        ))
    return ids, vectors


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norm, 1e-12, None)


def build_flat_index(vectors: np.ndarray, metric: str = "IP") -> faiss.Index:
    D = vectors.shape[1]
    if metric == "IP":
        index = faiss.IndexFlatIP(D)
    elif metric == "L2":
        index = faiss.IndexFlatL2(D)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    index.add(vectors)
    return index


def build_hnsw_index(vectors: np.ndarray, M: int = 64, efC: int = 200, efS: int = 200) -> faiss.Index:
    D = vectors.shape[1]
    index = faiss.IndexHNSWFlat(D, M)
    index.hnsw.efConstruction = efC
    index.add(vectors)
    index.hnsw.efSearch = efS
    return index


def apply_opq(vectors: np.ndarray, M: int = 64) -> Tuple[np.ndarray, faiss.OPQMatrix]:
    D = vectors.shape[1]
    opq = faiss.OPQMatrix(D, M)
    opq.train(vectors)
    return opq.apply_py(vectors), opq


def build_ivfopq_index(
    vectors: np.ndarray,
    index_out: Path,
    nlist: int = 4096,
    m: int = 64,
    nbits: int = 8
) -> faiss.Index:
    """
    Построить OPQ + IVF-PQ индекс и сохранить OPQ-матрицу рядом с индексом.

    :param vectors: корпус эмбеддингов, shape (N, D)
    :param index_out: полный путь до .index файла (например data/index/faiss_ivfopq.index)
    :param nlist: число кластеров для IVF
    :param m: число суб-векторов для PQ и OPQ
    :param nbits: число бит на суб-вектор в PQ
    :returns: построенный faiss.IndexIVFPQ
    """
    # 1) Обучаем OPQ и трансформируем корпус
    v_opq, opq_mat = apply_opq(vectors, M=m)

    # 2) Подготавливаем директорию и сохраняем OPQ-матрицу
    index_out = Path(index_out)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    opq_path = index_out.with_suffix(".opq")  # e.g. data/index/faiss_ivfopq.opq
    faiss.write_VectorTransform(opq_mat, str(opq_path))

    # 3) Строим IVF-PQ на OPQ-векторах
    D = v_opq.shape[1]
    quantizer = faiss.IndexFlatIP(D)
    ivfpq_idx = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)
    ivfpq_idx.train(v_opq)
    ivfpq_idx.add(v_opq)

    return ivfpq_idx


def save_index(index: faiss.Index, ids: List[Tuple[str, str, str]], index_path: Path, ids_path: Path):
    # FAISS CPU-only, direct save
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def load_index(index_path: Path, ids_path: Path) -> Tuple[faiss.Index, List[Tuple[str, str, str]]]:
    index = faiss.read_index(str(index_path))
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return index, ids


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


def rerank_candidates(
    candidates: List[List[Tuple[Tuple[str,str,str], float]]],
    chunks_path: Path,
    rerank_model: str,
    top_k: int,
    query_texts: List[str]
) -> List[List[Tuple[Tuple[str,str,str], float]]]:
    # load chunk_texts from JSONL: [paper_id, section, chunk_id, text]
    chunk_texts: Dict[Tuple[str,str,str], str] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            pid, sec, cid, txt = json.loads(line)
            chunk_texts[(pid, sec, cid)] = txt
    reranker = CrossEncoder(rerank_model)
    final_ranks = []
    for qtext, cand_list in zip(query_texts, candidates):
        pairs = [(qtext, chunk_texts[cid]) for cid,_ in cand_list]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip((cid for cid,_ in cand_list), scores),
            key=lambda x: x[1], reverse=True
        )[:top_k]
        final_ranks.append(ranked)
    return final_ranks


def search(index: faiss.Index, ids: List[Tuple[str,str,str]], queries: np.ndarray, top_k: int = 5) -> List[List[Tuple[Tuple[str,str,str], float]]]:
    D, I = index.search(queries, top_k)
    results = []
    for scores, idxs in zip(D, I):
        res = [(ids[i], float(scores[j])) for j,i in enumerate(idxs)]
        results.append(res)
    return results


def hybrid_search(
    coarse_idx: faiss.Index,
    ids: List[Tuple[str,str,str]],
    queries: np.ndarray,
    chunks_path: Path,
    rerank_model: str,
    top_k_coarse: int = 100,
    top_k: int = 5
) -> List[List[Tuple[Tuple[str,str,str], float]]]:
    # Load chunk texts: JSONL with [paper_id, section, chunk_id, text]
    chunk_texts: Dict[Tuple[str,str,str], str] = {}
    for line in open(chunks_path, encoding="utf-8"):
        pid, sec, cid, txt = json.loads(line)
        chunk_texts[(pid, sec, cid)] = txt
    reranker = CrossEncoder(rerank_model)
    D_coarse, I_coarse = coarse_idx.search(queries, top_k_coarse)
    final_res = []
    for scores, idxs in zip(D_coarse, I_coarse):
        cands = [ids[i] for i in idxs]
        texts = [chunk_texts[c] for c in cands]
        pairs = [("", t) for t in texts]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(cands, rerank_scores), key=lambda x: x[1], reverse=True)
        final_res.append(ranked[:top_k])
    return final_res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--index-out", type=Path, default=Path("data/index/faiss.index"))
    p.add_argument("--ids-out", type=Path, default=Path("data/index/ids.json"))
    p.add_argument(
        "--type",
        choices=["FlatIP","FlatL2","HNSW","IVFPQ","OPQIVFPQ","HYBRID"],
        default="FlatIP"
    )
    p.add_argument("--chunks", type=Path, help="Path to chunks JSONL for HYBRID")
    p.add_argument("--rerank-model", type=str, default="cross-encoder/stsb-roberta-large")
    p.add_argument("--topk-coarse", type=int, default=100)
    p.add_argument("--topk", type=int, default=5)

    # OPQ+IVF-PQ
    p.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of IVF clusters for IVFPQ (nlist)"
    )
    p.add_argument(
        "--m",
        type=int,
        default=64,
        help="Number of PQ subquantizers (m)"
    )
    p.add_argument(
        "--nbits",
        type=int,
        default=8,
        help="Bits per PQ codebook entry (nbits)"
    )

    args = p.parse_args()

    ids, vecs = load_embeddings(args.embeddings, args.ids)
    # cosine normalization for IP indices
    vecs = normalize_embeddings(vecs)

    if args.type == "HNSW":
        index = build_hnsw_index(vecs)
    elif args.type == "IVFPQ":
        index = build_ivfopq_index(
            vecs,
            index_out=args.index_out,
            nlist=args.nlist,  # если вы добавили эти флаги
            m=args.m,
            nbits=args.nbits
        )
    elif args.type == "OPQIVFPQ":
        # alias to IVFPQ with OPQ applied
        index = build_ivfopq_index(
            vecs,
            index_out=args.index_out,
            nlist=args.nlist,  # если вы добавили эти флаги
            m=args.m,
            nbits=args.nbits
        )
    else:
        # FlatIP/FlatL2
        metric = "IP" if args.type == "FlatIP" else "L2"
        index = build_flat_index(vecs, metric=metric)

    save_index(index, ids, args.index_out, args.ids_out)