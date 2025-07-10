# tests/test_embed.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.retriever.embed import (
    preprocess_text, split_into_sections,
    chunk_section_by_tokens, extract_keywords,
    summarize_one_sentence, build_contextual_chunk
)

def test_preprocess_text():
    text = "$$E=mc^2$$ Some \\\\alpha{test} text"
    processed = preprocess_text(text)
    assert "$$" not in processed
    assert "alpha" not in processed

def test_extract_keywords():
    text = "apple banana apple orange apple banana"
    keywords = extract_keywords(text, top_k=2)
    assert keywords == ["apple", "banana"]

def test_summarize_one_sentence():
    text = "Hello world. Second sentence."
    assert summarize_one_sentence(text) == "Hello world."

def test_build_contextual_chunk():
    meta = pd.Series({
        "title": "T", "authors": "A", "year": 2020,
        "categories": "cat"
    })
    chunk = "This is a sentence. Another one."
    ctx = build_contextual_chunk(meta, "Sec", "0", chunk, True)
    assert "Title: T" in ctx
    assert "Summary: This is a sentence." in ctx