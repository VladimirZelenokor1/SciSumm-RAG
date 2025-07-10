#!/usr/bin/env python3
import re, os, sys, json, unicodedata, logging, time
from pathlib import Path
from typing import Iterator, Tuple, List
import argparse

import numpy as np
import pandas as pd
import tiktoken
from sentence_transformers import SentenceTransformer
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

# Configuration defaults and environment overrides
SECTION_RE       = re.compile(r"\n(\d+\.\s+[^\n]+)\n")
MODEL_NAME       = os.getenv("EMBED_MODEL", "paraphrase-MiniLM-L3-v2")  # lighter model
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER        = tiktoken.get_encoding("gpt2")
MAX_TOKENS       = int(os.getenv("MAX_TOKENS",        "400"))
OVERLAP_TOKENS   = int(os.getenv("OVERLAP_TOKENS",    "50"))
STREAM_BATCH     = int(os.getenv("STREAM_BATCH",     "5000"))  # increased batch size
MAX_PAPERS       = int(os.getenv("MAX_PAPERS",    "100000"))
MIN_FULLTEXT_LEN = int(os.getenv("MIN_FULLTEXT_LENGTH", "1000"))
SUMMARY_FIRST_N  = int(os.getenv("SUMMARY_FIRST_N",    "1"))  # summary only for first chunk per paper

# Simple keyword extraction using Counter
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


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
    allowed_categories: list[str] = ["cs.AI", "cs.CL", "stat.ML"]
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"full_text", "title", "authors", "year", "categories", "paper_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing metadata columns: %s", missing)
        raise ValueError(f"Missing columns: {missing}")

    # 1) Discard short texts and old articles
    df = df.dropna(subset=["full_text"]).copy()
    df["text_len"] = df["full_text"].str.len()
    df = df[df["text_len"] >= min_length]
    df = df[df["year"] >= min_year]

    # 2) Filter by category
    # If the article belongs to at least one of the allowed_categories
    df = df[df["categories"].apply(
        lambda cats: any(cat in cats.split() for cat in allowed_categories)
    )]

    # 3) Sampling, if it's still too much
    n = len(df)
    if n > max_papers:
        logger.info("Sampling down from %d to %d papers", n, max_papers)
        df = df.sample(n=max_papers, random_state=42)
    else:
        logger.info("Using all %d papers after filtering", n)

    return df.reset_index(drop=True)


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    parts = SECTION_RE.split(text)
    if len(parts) <= 1:
        return [("Unknown", text)]
    sections = []
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i+1].strip()))
    return sections


def chunk_section_by_tokens(
    heading: str, text: str
) -> Iterator[Tuple[str, str]]:
    """
    Chunk the tokenized section of the GPT-2 tokenizer.
    """
    tokens = TOKENIZER.encode(text)
    start, cid = 0, 0
    while start < len(tokens):
        end = min(len(tokens), start + MAX_TOKENS)
        chunk = TOKENIZER.decode(tokens[start:end])
        yield f"{heading}__{cid}", chunk
        cid += 1
        start = end - OVERLAP_TOKENS


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--out-dir",       type=Path, default=Path("data/clean/"))
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metadata(args.metadata_path)
    ids_file = args.out_dir / "ids.jsonl"
    skip = load_existing_pids(ids_file) if args.resume else set()

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info("Using device: %s", DEVICE)

    buf_ids, buf_texts = [], []
    part = 0
    t_chunk = t_encode = 0.0
    start_iter = time.time()

    chunks_file = args.out_dir / "chunks.jsonl"

    # Открываем оба файла сразу, в режиме «добавить»
    with open(ids_file, "a", encoding="utf-8") as sink_ids, \
            open(chunks_file, "a", encoding="utf-8") as sink_chunks:

        for idt, ctx in tqdm(iter_contextual_chunks(df, skip), desc="Chunk→Context"):
            buf_ids.append(idt)
            buf_texts.append(ctx)

            if len(buf_texts) >= STREAM_BATCH:
                # кодируем батч
                t0 = time.time()
                vecs = model.encode(buf_texts, batch_size=STREAM_BATCH, convert_to_numpy=True)
                t_encode += time.time() - t0

                # сохраняем эмбеддинги
                np.save(args.out_dir / f"embeddings_part{part}.npy", vecs)

                # записываем ids и параллельно текст чанков
                for tid, full_ctx in zip(buf_ids, buf_texts):
                    sink_ids.write(json.dumps(tid, ensure_ascii=False) + "\n")
                    # из full_ctx берём часть после пустой строки — собственно текст чанка
                    raw_chunk = full_ctx.split("\n\n")[-1]
                    pid, section, cid = tid
                    sink_chunks.write(
                        json.dumps([pid, section, cid, raw_chunk], ensure_ascii=False) + "\n"
                    )

                t_chunk += STREAM_BATCH
                part += 1
                buf_ids, buf_texts = [], []
                logger.info("Processed %d chunks, encode time %.1f s", t_chunk, t_encode)

        # flush remainder
        if buf_texts:
            t0 = time.time()
            vecs = model.encode(buf_texts, batch_size=STREAM_BATCH, convert_to_numpy=True)
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