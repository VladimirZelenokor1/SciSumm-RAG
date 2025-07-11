# tests/test_embed.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
import src.retriever.embed as embed_mod
from src.retriever.embed import (
    preprocess_text, split_into_sections,
    chunk_section_by_tokens, extract_keywords,
    summarize_one_sentence, build_contextual_chunk, embed_texts
)

# Dummy tokenizer and model for chunking and embedding
class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # one token per word
        return list(range(len(text.split())))
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(f"w{t}" for t in tokens)

class DummyModel:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

# override the global MODEL used by chunk_section_by_tokens
embed_mod.MODEL = DummyModel()

def test_preprocess_text_removes_latex_and_whitespace():
    text = "$$E=mc^2$$ Some \\alpha{test}   text"
    processed = preprocess_text(text)
    assert "$$" not in processed
    assert "alpha" not in processed
    assert "  " not in processed

def test_extract_keywords_counts_frequency():
    text = "apple banana apple orange apple banana"
    keywords = extract_keywords(text, top_k=2)
    assert keywords == ["apple", "banana"]

def test_summarize_one_sentence_picks_first():
    text = "Hello world. Second sentence!"
    assert summarize_one_sentence(text) == "Hello world."

def test_split_sections_numeric_headings():
    # Embed.py splits on \n1. Header\n …\n2. …\n (SECTION_RE = r"\n(\d+\.\s+[^\n]+)\n")
    text = "\n1. H1\nLine1\n\n2. H2\nLine2"
    sections = split_into_sections(text)
    assert sections == [("1. H1", "Line1"), ("2. H2", "Line2")]

def test_split_sections_no_headings_fallback_to_paragraphs():
    text = "ParaA\n\nParaB\n\n\nParaC"
    sections = split_into_sections(text)
    # should split into three paras
    assert sections == [
        ("para_0", "ParaA"),
        ("para_1", "ParaB"),
        ("para_2", "ParaC"),
    ]

def test_build_contextual_chunk_includes_metadata_and_summary():
    import pandas as pd
    meta = pd.Series({
        "title": "TitleX",
        "authors": "AuthY",
        "year": 2021,
        "categories": "catA catB",
    })
    chunk = "This is first. This is second."
    ctx = build_contextual_chunk(meta, "Head", "0", chunk, summary_flag=True)
    assert "Title: TitleX" in ctx
    assert "Authors: AuthY" in ctx
    assert "Section: Head" in ctx
    assert "Summary: This is first." in ctx

def test_embed_texts_returns_array_and_shape():
    # just check that signature matches and returns numpy array
    texts = ["hello world", "foo bar"]
    embs = embed_texts(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape[0] == 2
    # dimensionality >0
    assert embs.shape[1] > 0
    assert np.issubdtype(embs.dtype, np.floating)