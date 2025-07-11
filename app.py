import streamlit as st
import faiss
import json
import numpy as np
from pathlib import Path
from src.retriever.embed import embed_texts
from src.retriever.index import normalize_embeddings, hybrid_search
from src.generator.hf_summarizer import HFSummarizer

# Paths configuration
DATA_DIR = Path("D:/SciSumm-RAG/data")
INDEX_DIR = DATA_DIR / "index/faiss"
CLEAN_DIR = DATA_DIR / "clean"

# Available indexes mapping
INDEX_OPTIONS = {
    "FlatIP": INDEX_DIR / "flat_index.index",
    "FlatL2": INDEX_DIR / "flatl2_index.index",
    "HNSW": INDEX_DIR / "hnsw_index.index",
    "IVFPQ": INDEX_DIR / "ivfpq_index.index",
    "OPQIVFPQ": INDEX_DIR / "opqivfpq_index.index",
}

# Sidebar controls
st.sidebar.title("Settings")
index_type = st.sidebar.selectbox("Select FAISS Index:", list(INDEX_OPTIONS.keys()))
use_hybrid = st.sidebar.checkbox("Use Hybrid Rerank (Cross-Encoder)", value=False)
if use_hybrid:
    top_k_coarse = st.sidebar.slider("Top-K Coarse FAISS:", 10, 200, 100)
    rerank_model = st.sidebar.text_input("Cross-Encoder Model:", value="cross-encoder/ms-marco-MiniLM-L-6-v2")
else:
    top_k_coarse = None
    rerank_model = None

# Common slider for final Top-K
top_k = st.sidebar.slider("Top-K Final Chunks:", 1, 20, 5)

@st.cache_resource
def load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device="cuda")

@st.cache_resource
def load_index(index_file: Path):
    return faiss.read_index(str(index_file))

@st.cache_resource
def load_chunk_data():
    chunk_keys = []
    chunk_texts = {}
    with open(CLEAN_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            pid, section, cid, txt = json.loads(line)
            key = (pid, section, cid)
            chunk_keys.append(key)
            chunk_texts[key] = txt
    return chunk_keys, chunk_texts

@st.cache_resource
def load_summarizer():
    return HFSummarizer(model_name=None, device=None)

st.title("🔍 SciSumm-RAG Quick Demo")

index_path = INDEX_OPTIONS[index_type]
idx = load_index(index_path)
chunk_keys, chunk_texts = load_chunk_data()

query = st.text_area("Enter a query or the text of an article:", height=200)

if st.button("Search and Summarize"):
    # 1) Embed and normalize the query
    q_emb = embed_texts([query])
    q_emb = normalize_embeddings(q_emb.astype(np.float32))

    # 2) Retrieve
    if use_hybrid:
        reranker = load_reranker(rerank_model)
        results = hybrid_search(
            coarse_idx=idx,
            ids=chunk_keys,
            queries=q_emb,
            query_texts=[query],
            chunks_path=CLEAN_DIR / "chunks.jsonl",
            rerank_model=reranker,
            top_k_coarse=top_k_coarse,
            top_k=top_k
        )
    else:
        distances, indices = idx.search(q_emb, top_k)
        results = [[(chunk_keys[i], float(score)) for i, score in zip(indices[0], distances[0])]]

    # 3) Display retrieved chunks
    st.subheader("🔖 Retrieved Chunks")
    for rank, (key, score) in enumerate(results[0], start=1):
        pid, section, cid = key
        txt_snippet = chunk_texts[key][:300]
        st.markdown(f"**{rank}. {pid} | {section} | chunk {cid}**  (score={score:.3f})")
        st.write(txt_snippet + "…")

    # 4) Summary from top chunk
    st.subheader("📝 Summary (all chunks of top article)")
    top_key = results[0][0][0]
    top_pid = top_key[0]
    passages = [
        chunk_texts[key]
        for key, _ in results[0]
        if key[0] == top_pid
    ]
    combined_text = "\n\n".join(passages)
    summarizer = load_summarizer()
    summary = summarizer.summarize(
        combined_text,
        max_length=150,
        min_length=30
    )
    st.write(summary)
