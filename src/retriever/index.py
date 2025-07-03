import numpy as np
import json
import faiss
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import torch
from sentence_transformers import CrossEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_embeddings(path_vec: Path, path_ids: Path) -> Tuple[List[Tuple[str,str,str]], np.ndarray]:
    vectors = np.load(path_vec)  # shape (N, D)
    with open(path_ids, "r", encoding="utf-8") as f:
        ids = json.load(f)
    assert len(ids) == vectors.shape[0], "IDs and vectors count mismatch"
    return ids, vectors


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norm, 1e-12, None)


def build_flat_index(
    vectors: np.ndarray, metric: str = "IP", use_gpu: bool = False
) -> faiss.Index:
    D = vectors.shape[1]
    if metric == "IP":
        cpu_index = faiss.IndexFlatIP(D)
    else:
        cpu_index = faiss.IndexFlatL2(D)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index
    index.add(vectors)
    return index


def build_hnsw_index(
    vectors: np.ndarray,
    M: int = 32,
    efC: int = 200,
    efS: int = 50,
    use_gpu: bool = False
) -> faiss.Index:
    D = vectors.shape[1]
    cpu_index = faiss.IndexHNSWFlat(D, M)
    cpu_index.hnsw.efConstruction = efC
    cpu_index.add(vectors)
    cpu_index.hnsw.efSearch = efS
    if use_gpu:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cpu_index


def apply_opq(vectors: np.ndarray, M: int = 64) -> Tuple[np.ndarray, faiss.OPQMatrix]:
    D = vectors.shape[1]
    opq = faiss.OPQMatrix(D, M)
    opq.train(vectors)
    return opq.apply_py(vectors), opq


def build_ivfopq_index(
    vectors: np.ndarray,
    nlist: int = 1024,
    m: int = 64,
    nbits: int = 8,
    use_gpu: bool = False
) -> faiss.Index:
    # OPQ + IVF-PQ
    v_opq, opq_mat = apply_opq(vectors, M=m)
    D = v_opq.shape[1]
    quantizer = faiss.IndexFlatIP(D)
    cpu_index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)
    cpu_index.train(v_opq)
    cpu_index.add(v_opq)
    if use_gpu:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cpu_index


def save_index(
    index: faiss.Index,
    ids: List[Tuple[str,str,str]],
    index_path: Path,
    ids_path: Path
):
    # convert to CPU before saving
    if isinstance(index, faiss.GpuIndex):
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, str(index_path))
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def load_index(
    index_path: Path,
    ids_path: Path,
    use_gpu: bool = False
) -> Tuple[faiss.Index, List[Tuple[str,str,str]]]:
    cpu_index = faiss.read_index(str(index_path))
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return index, ids


def search(
    index: faiss.Index,
    ids: List[Tuple[str,str,str]],
    queries: np.ndarray,
    top_k: int = 5
) -> List[List[Tuple[Tuple[str,str,str], float]]]:
    D, I = index.search(queries, top_k)
    results = []
    for scores, idxs in zip(D, I):
        row = [(ids[i], float(scores[j])) for j,i in enumerate(idxs)]
        results.append(row)
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
    # load chunk texts mapping
    chunk_texts = {tuple(item[:3]): item[3] for item in map(json.loads, open(chunks_path))}
    # cross-encoder
    reranker = CrossEncoder(rerank_model, device=DEVICE)
    D_coarse, I_coarse = coarse_idx.search(queries, top_k_coarse)
    final = []
    for qi, (scores, idxs) in enumerate(zip(D_coarse, I_coarse)):
        cands = [ids[i] for i in idxs]
        texts = [chunk_texts[c] for c in cands]
        pairs = [("", t) for t in texts]  # or supply actual query text
        rerank_scores = reranker.predict(pairs)
        zipped = list(zip(cands, rerank_scores))
        zipped.sort(key=lambda x: x[1], reverse=True)
        final.append(zipped[:top_k])
    return final

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
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--chunks", type=Path, help="Path to chunks JSONL for HYBRID")
    p.add_argument("--rerank-model", type=str, default="cross-encoder/stsb-roberta-large")
    p.add_argument("--topk-coarse", type=int, default=100)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    ids, vecs = load_embeddings(args.embeddings, args.ids)
    # cosine normalization for IP indices
    vecs = normalize_embeddings(vecs)

    if args.type == "HNSW":
        index = build_hnsw_index(vecs, use_gpu=args.use_gpu)
    elif args.type == "IVFPQ":
        index = build_ivfopq_index(vecs, use_gpu=args.use_gpu)
    elif args.type == "OPQIVFPQ":
        # alias to IVFPQ with OPQ applied
        index = build_ivfopq_index(vecs, use_gpu=args.use_gpu)
    else:
        # FlatIP/FlatL2
        metric = "IP" if args.type == "FlatIP" else "L2"
        index = build_flat_index(vecs, metric=metric, use_gpu=args.use_gpu)

    save_index(index, ids, args.index_out, args.ids_out)

    # test search
    q = vecs[:1]
    if args.type == "HYBRID":
        assert args.chunks, "--chunks required for HYBRID"
        res = hybrid_search(
            index, ids, q,
            chunks_path=args.chunks,
            rerank_model=args.rerank_model,
            top_k_coarse=args.topk_coarse,
            top_k=args.topk
        )
    else:
        res = search(index, ids, q, top_k=args.topk)
    print(res)