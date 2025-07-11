#!/usr/bin/env python3
import re, os, sys, json, unicodedata, logging, time
from pathlib import Path
from typing import Iterator, Tuple, List
import argparse

import numpy as np
import pandas as pd
import multiprocessing
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
import nltk
from collections import Counter
from tqdm import tqdm

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Download NLTK data once
def _init_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
_init_nltk()

# global worshippers
_worker_model: SentenceTransformer = None
_skip_pids: set = None
MODEL: SentenceTransformer = None

# Configuration defaults and environment overrides
SECTION_RE       = re.compile(r"\n(\d+\.\s+[^\n]+)\n")
MODEL_NAME       = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # lighter model
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS       = int(os.getenv("MAX_TOKENS",        "512"))
OVERLAP_TOKENS   = int(os.getenv("OVERLAP_TOKENS",    "50"))
STREAM_BATCH     = int(os.getenv("STREAM_BATCH",     "5000"))  # increased batch size
MAX_PAPERS       = int(os.getenv("MAX_PAPERS",    "100000"))
MIN_FULLTEXT_LEN = int(os.getenv("MIN_FULLTEXT_LENGTH", "1000"))
SUMMARY_FIRST_N  = int(os.getenv("SUMMARY_FIRST_N",    "1"))  # summary only for first chunk per paper

# Simple keyword extraction using Counter
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def _init_worker(model_name, device, skip_pids):
    """Initializes the model and skip_pids in each process."""
    global _worker_model, _skip_pids
    _worker_model = SentenceTransformer(model_name, device=device)
    _skip_pids = skip_pids


def _process_row(row) -> List[Tuple[Tuple[str,str,str], str]]:
    """
    Handles a single df string: preprocessing → sections → chunks → build_contextual_chunk.
    Returns a list (identifier, context) for this article.
    """
    results: List[Tuple[Tuple[str,str,str], str]] = []
    pid = row["paper_id"]
    if pid in _skip_pids:
        return results

    text = preprocess_text(row["full_text"])
    summary_done = 0

    for heading, body in split_into_sections(text):
        for cid, chunk in chunk_section_by_tokens(heading, body):
            summary_flag = (summary_done < SUMMARY_FIRST_N)
            ctx = build_contextual_chunk(
                pd.Series(row),
                heading,
                str(cid),
                chunk,
                summary_flag
            )
            results.append(((pid, heading, str(cid)), ctx))
            if summary_flag:
                summary_done += 1

    return results


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encodes a list of texts into a numpy array of embeddings.
    """
    _embedder = SentenceTransformer(MODEL_NAME, device=DEVICE)

    return _embedder.encode(
        texts,
        batch_size=STREAM_BATCH,
        convert_to_numpy=True
    )


def preprocess_text(text: str) -> str:
    """
    Remove LaTeX/math, normalize unicode and whitespace.
    """
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def load_metadata(
    path: Path,
    max_papers: int = 50000,
    min_year: int = 2020,
    min_length: int = 2000,
    allowed_categories: list[str] = None
) -> pd.DataFrame:
    """
    Loads and filters article metadata.
    Default:
      - articles from 2020 and newer,
      - texts ≥2000 characters,
    """
    df = pd.read_csv(path, low_memory=False)
    required = {"full_text", "title", "authors", "year", "categories", "paper_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing metadata columns: %s", missing)
        raise ValueError(f"Missing columns: {missing}")

    # 1) Leave only with full_text
    df = df.dropna(subset=["full_text"]).copy()
    df["text_len"] = df["full_text"].str.len()

    # 2) Filter by length and year
    df = df[df["text_len"] >= min_length]
    df = df[df["year"] >= min_year]

    # 3) If allowed_categories are set - filter, otherwise skip everything
    if allowed_categories:
        df = df[df["categories"].apply(
            lambda cats: any(cat in cats.split() for cat in allowed_categories)
        )]

    # 4) only 'en'
    def is_english(text: str) -> bool:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words: return False
        eng_cnt = sum(1 for w in words if w in ENGLISH_STOP_WORDS)
        return eng_cnt / len(words) > 0.05

    df = df[df["full_text"].apply(is_english)]

    # 5) If max_papers is exceeded, a random sample will be taken
    n = len(df)
    if n > max_papers:
        logger.info("Sampling down from %d to %d papers", n, max_papers)
        df = df.sample(n=max_papers, random_state=42)
    else:
        logger.info("Using all %d papers after filtering", n)

    return df.reset_index(drop=True)


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    If there are obvious headings in the text, split by them.
    Otherwise - by paragraphs (double line feed).
    """
    parts = SECTION_RE.split(text)
    sections: List[Tuple[str, str]] = []

    if len(parts) > 1:
        # parts[0] - text before the first header
        intro = parts[0].strip()
        if intro:
            sections.append(("Introduction", intro))
        # next: [header1, content1, header2, content2, ...].
        for i in range(1, len(parts), 2):
            heading = parts[i].strip() or "Unknown"
            content = parts[i+1].strip()
            if content:
                sections.append((heading, content))
    else:
        # no headings - break it down into paragraphs
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paras):
            sections.append((f"para_{idx}", p))

    return sections


def chunk_section_by_tokens(
    heading: str, text: str
) -> Iterator[Tuple[str, str]]:
    """
    Chunk the tokenized section.
    """
    model = _worker_model or MODEL
    tokenizer = model.tokenizer
    model_max = getattr(tokenizer, "model_max_length", MAX_TOKENS)

    tokens = tokenizer.encode(text, add_special_tokens=False)
    effective_max = model_max - 2
    step = effective_max - OVERLAP_TOKENS

    for cid, start in enumerate(range(0, len(tokens), step)):
        end = min(len(tokens), start + effective_max)
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
        yield f"{heading}__{cid}", chunk
        if end == len(tokens):
            break


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    # simple frequency-based keywords
    words = [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in STOPWORDS]
    freq = Counter(words)
    return [w for w,_ in freq.most_common(top_k)]


"""
def summarize_one_sentence(text):
    sents = sent_tokenize(text)
    if not sents:
        return ""
    # for example, the shortest or the first
    return sents[0] if sents else ""
"""


def summarize_one_sentence(text: str) -> str:
    # first sentence by regex
    m = re.match(r"(.+?[\.\!?])\s", text)
    return m.group(1) if m else text.split(". ")[0]


def build_contextual_chunk(
    meta: pd.Series, heading: str, cid: str, chunk: str, summary_flag: bool
) -> str:
    """
    Collect context for chunk:
    Title, Authors, Year, Categories, Section, Keywords, chunk text.
    """
    kws = extract_keywords(chunk)
    summ = summarize_one_sentence(chunk) if summary_flag else ""
    parts = [
        f"Title: {meta['title']}",
        f"Authors: {meta['authors']}",
        f"Year: {meta['year']}",
        f"Categories: {meta['categories']}",
        f"Section: {heading}",
        f"Keywords: {', '.join(kws)}",
    ]
    if summ:
        parts.append(f"Summary: {summ}")
    parts.append("")
    parts.append(chunk)
    return "\n".join(parts)


def iter_contextual_chunks(
    df: pd.DataFrame, skip_pids: set
) -> Iterator[Tuple[Tuple[str,str,str], str]]:
    for row in df.itertuples(index=False):
        pid = row.paper_id
        if pid in skip_pids:
            continue
        text = preprocess_text(row.full_text)
        summary_done = 0
        for heading, body in split_into_sections(text):
            for cid, chunk in chunk_section_by_tokens(heading, body):
                summary_flag = summary_done < SUMMARY_FIRST_N
                ctx = build_contextual_chunk(
                    pd.Series(row._asdict()), heading, str(cid), chunk, summary_flag
                )
                yield (pid, heading, str(cid)), ctx
                if summary_flag:
                    summary_done += 1


def load_existing_pids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)[0] for line in path.open(encoding='utf-8')}


def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--out-dir",       type=Path, default=Path("data/clean/"))
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metadata(args.metadata_path)
    ids_file = args.out_dir / "ids.jsonl"
    skip = load_existing_pids(ids_file) if args.resume else set()

    rows = df.to_dict(orient="records")

    MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info("Using device: %s", DEVICE)

    buf_ids, buf_texts = [], []
    part = 0
    t_chunk = t_encode = 0.0
    start_iter = time.time()

    chunks_file = args.out_dir / "chunks.jsonl"

    with open(ids_file, "a", encoding="utf-8") as sink_ids, \
         open(chunks_file, "a", encoding="utf-8") as sink_chunks:

        with multiprocessing.Pool(
            processes=os.cpu_count(),
            initializer=_init_worker,
            initargs=(MODEL_NAME, DEVICE, skip)
        ) as pool:
            for batch in tqdm(pool.imap(_process_row, rows), total=len(rows), desc="Chunk→Context"):
                for idt, ctx in batch:
                    buf_ids.append(idt)
                    buf_texts.append(ctx)

                # as soon as STREAM_BATCH contexts are accumulated - encode and reset
                while len(buf_texts) >= STREAM_BATCH:
                    t0 = time.time()
                    vecs = MODEL.encode(buf_texts[:STREAM_BATCH],
                                        batch_size=STREAM_BATCH,
                                        convert_to_numpy=True)
                    t_encode += time.time() - t0

                    np.save(args.out_dir / f"embeddings_part{part}.npy", vecs)
                    for tid, full_ctx in zip(buf_ids[:STREAM_BATCH], buf_texts[:STREAM_BATCH]):
                        sink_ids.write(json.dumps(tid, ensure_ascii=False) + "\n")
                        raw_chunk = full_ctx.split("\n\n")[-1]
                        pid, section, cid = tid
                        sink_chunks.write(
                            json.dumps([pid, section, cid, raw_chunk], ensure_ascii=False) + "\n"
                        )

                    t_chunk += STREAM_BATCH
                    part += 1
                    # delete already saved ones from the buffer
                    buf_ids = buf_ids[STREAM_BATCH:]
                    buf_texts = buf_texts[STREAM_BATCH:]
                    logger.info("Processed %d chunks, encode time %.1f s", t_chunk, t_encode)

        # flush remainder
        if buf_texts:
            t0 = time.time()
            vecs = MODEL.encode(buf_texts,
                                batch_size=STREAM_BATCH,
                                convert_to_numpy=True)
            t_encode += time.time() - t0
            np.save(args.out_dir / f"embeddings_part{part}.npy", vecs)
            for tid, full_ctx in zip(buf_ids, buf_texts):
                sink_ids.write(json.dumps(tid, ensure_ascii=False) + "\n")
                raw_chunk = full_ctx.split("\n\n")[-1]
                pid, section, cid = tid
                sink_chunks.write(
                    json.dumps([pid, section, cid, raw_chunk], ensure_ascii=False) + "\n"
                )
            t_chunk += len(buf_texts)

    total_time = time.time() - start_iter
    logger.info(
        "Total chunks: %d, total encode time: %.1f s, total time: %.1f s",
        t_chunk,
        t_encode,
        total_time
    )


if __name__ == "__main__":
    main()