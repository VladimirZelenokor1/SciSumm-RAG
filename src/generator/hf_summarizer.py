# src/generator/hf_summarizer.py

import os
import torch
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

class HFSummarizer:
    def __init__(
        self,
        model_name: str = None,
        device: int = None,
        max_input_len: int = 1024,
        chunk_overlap: int = 200,
    ):
        """
        model_name: HF model identifier (e.g. "facebook/bart-large-cnn")
        OR local path to the directory with the saved model.
        device: GPU index (0,1,...) or -1 for CPU.
        max_input_len: maximum length of input (tokens).
        chunk_overlap: number of tokens-overlaps between chunks.
        """
        self.model_name = (
            model_name
            or os.getenv("HF_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
        )

        if os.path.isdir(self.model_name):
            local_dir = self.model_name
        else:
            local_dir = None

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        # pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir or self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(local_dir or self.model_name)
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # sliding-window
        self.max_input_len = max_input_len
        self.chunk_overlap = chunk_overlap

    @lru_cache(maxsize=256)
    def _summarize_chunk(self, chunk: str, **kwargs) -> str:
        """Internal method with caching by chunk text."""
        out = self.summarizer(
            chunk,
            max_length=kwargs.get("max_length", 150),
            min_length=kwargs.get("min_length", 30),
            do_sample=kwargs.get("do_sample", False),
            truncation=True,
        )
        return out[0]["summary_text"].strip()

    def summarize(self, text: str, **kwargs) -> str:
        """
        Universal method: if the text is short, one call;
        otherwise - split into windows, summarize each, then merge.
        """
        # tokenize for length calculation
        tokens = self.tokenizer.encode(text, return_tensors="pt")[0]
        total_len = tokens.size(0)

        if total_len <= self.max_input_len:
            return self._summarize_chunk(text, **kwargs)

        # break down into overlapping windows
        summaries: List[str] = []
        step = self.max_input_len - self.chunk_overlap
        for start in range(0, total_len, step):
            end = min(start + self.max_input_len, total_len)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            summaries.append(self._summarize_chunk(chunk_text, **kwargs))
            if end == total_len:
                break

        # aggregate: concatenation + final summarize
        combined = "\n".join(summaries)
        return self._summarize_chunk(combined, **kwargs)