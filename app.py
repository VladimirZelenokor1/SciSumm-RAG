# app.py

import streamlit as st
import faiss
import json
import numpy as np
import pickle
from src.retriever.index import normalize_embeddings
from src.retriever.embed import embed_texts
from src.generator.hf_summarizer import generate_summary_hf

@st.cache_resource
def load_index_and_data():
    # 1) FAISS
    idx = faiss.read_index(r"D:\SciSumm-RAG\data\index\faiss_flat_ip.index")
    # 2) Precalculated Embeddings (normalized)
    embs = np.load(r"D:\SciSumm-RAG\data\clean\embeddings.npy")

    chunk_keys = []
    chunk_texts = {}
    with open(r"D:\SciSumm-RAG\data\clean\chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            pid, section, cid, txt = json.loads(line)
            key = (pid, section, cid)
            chunk_keys.append(key)
            chunk_texts[key] = txt
    return idx, embs, chunk_keys, chunk_texts

st.title("🔍 SciSumm-RAG Quick Demo")

idx, embs, chunk_keys, chunk_texts = load_index_and_data()

query = st.text_area("Enter a query or the text of an article:", height=200)
top_k = st.slider("Top-K chunks:", 1, 20, 5)

if st.button("Search and summarization"):
    # 1) Embed the query and normalize
    q_emb = embed_texts([query])
    q_emb = normalize_embeddings(q_emb)

    # 2) Search the index
    D, I = idx.search(q_emb, top_k)
    scores = D[0].tolist()
    indices = I[0].tolist()

    st.subheader("🔖 Retrieved chunks")
    for rank, (ind, score) in enumerate(zip(indices, scores), start=1):
        chunk_id = chunk_keys[ind]
        txt = chunk_texts[chunk_id]
        st.markdown(f"**{rank}. {chunk_id}**  (score={score:.3f})")
        st.write(txt[:300] + "…")

    # 3) Generate a summary on all introductory text
    st.subheader("📝 Summary")
    summary = generate_summary_hf(
        txt, max_length=150, min_length=30
    )
    st.write(summary)
