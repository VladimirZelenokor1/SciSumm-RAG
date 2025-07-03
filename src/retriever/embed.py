import re
import os
import json
import sys
import unicodedata
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import argparse
import tiktoken
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from nltk.tokenize import sent_tokenize
import nltk

# Initialize logging
txt_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=txt_format, level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK prerequisites for RAKE
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Constants and configuration
SECTION_RE = re.compile(r"\n(\d+\.\s+[^\n]+)\n")
MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOKENIZER = tiktoken.get_encoding("gpt2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 400))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
RAKE = Rake()


def preprocess_text(text: str) -> str:
    """
    Remove LaTeX/math, normalize unicode and whitespace.
    """
    # Remove inline and display math
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    # Normalize whitespace and unicode
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"full_text", "title", "authors", "year", "categories", "paper_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing metadata columns: %s", missing)
        sys.exit(1)
    return df


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    parts = SECTION_RE.split(text)
    if len(parts) <= 1:
        return [("Unknown", text)]
    sections = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i+1].strip()
        sections.append((heading, body))
    return sections


def chunk_section_by_tokens(
    sec_heading: str,
    sec_text: str,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS
) -> List[Tuple[str, str]]:
    """
    Chunk the tokenized section of the GPT-2 tokenizer.
    """
    tokens = TOKENIZER.encode(sec_text)
    chunks = []
    start = 0
    cid = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_tokens)
        chunks.append((f"{sec_heading}__{cid}", chunk_text))
        cid += 1
        start = end - overlap
    return chunks


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    RAKE.extract_keywords_from_text(text)
    phrases = RAKE.get_ranked_phrases()[:top_k]
    return phrases


def summarize_one_sentence(text):
    sents = sent_tokenize(text)
    if not sents:
        return ""
    # for example, the shortest or the first
    return sents[0] if sents else ""


def build_contextual_chunk(
    meta: pd.Series,
    sec_heading: str,
    chunk_id: str,
    chunk_text: str
) -> str:
    """
    Collect context for chunk:
    Title, Authors, Year, Categories, Section, Keywords, chunk text.
    """
    keywords = extract_keywords(chunk_text)
    summary = summarize_one_sentence(chunk_text)
    parts = [
        f"Title: {meta['title']}",
        f"Authors: {meta['authors']}",
        f"Year: {meta['year']}",
        f"Categories: {meta['categories']}",
        f"Section: {sec_heading}",
        f"Keywords: {', '.join(keywords)}",
        f"Summary: {summary}",
        "",
        chunk_text
    ]
    return "\n".join(parts)


def compute_embeddings(
    texts: List[str],
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def save_embeddings(
    ids: List[Tuple[str, str, str]],
    vectors: np.ndarray,
    out_dir: str = "data/clean/"
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), vectors)
    with open(os.path.join(out_dir, "ids.json"), "w") as f:
        json.dump(ids, f)
    logger.info("Saved %d embeddings to %s", len(ids), out_dir)


def load_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, "r") as f:
        items = json.load(f)
    return {tuple(item) for item in items}


def process_row(
    row: pd.Series
) -> List[Tuple[Tuple[str, str, str], str]]:
    """
    Preprocess one paper and return list of (id_tuple, contextual_text).
    """
    paper_id = row['paper_id']
    text = preprocess_text(row['full_text'])
    results = []
    for sec_heading, sec_body in split_into_sections(text):
        for chunk_id, chunk_text in chunk_section_by_tokens(sec_heading, sec_body):
            id_tuple = (paper_id, sec_heading, chunk_id)
            ctx = build_contextual_chunk(row, sec_heading, chunk_id, chunk_text)
            results.append((id_tuple, ctx))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate contextual embeddings from metadata CSV"
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/clean/"),
        help="Output directory for embeddings and ids"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers already processed based on ids.json"
    )
    args = parser.parse_args()

    df = load_metadata(args.metadata_path)
    ids_path = args.out_dir / "ids.json"
    existing = load_existing_ids(ids_path) if args.resume else set()

    tasks = []
    for _, row in df.iterrows():
        pid = row['paper_id']
        if args.resume and any(pid == eid[0] for eid in existing):
            continue
        tasks.append(row)

    all_ids = []
    all_texts = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_row = {executor.submit(process_row, row): row for row in tasks}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                results = future.result()
                for id_tuple, text in results:
                    all_ids.append(id_tuple)
                    all_texts.append(text)
            except Exception as e:
                logger.error("Error processing paper %s: %s", row['paper_id'], e)

    if not all_texts:
        logger.info("No new papers to process. Exiting.")
        sys.exit(0)

    vectors = compute_embeddings(all_texts)
    if args.resume and (args.out_dir / "embeddings.npy").exists():
        old_vectors = np.load(args.out_dir / "embeddings.npy")
        old_ids = list(load_existing_ids(ids_path))
        vectors = np.vstack([old_vectors, vectors])
        all_ids = old_ids + all_ids

    save_embeddings(all_ids, vectors, args.out_dir)


if __name__ == "__main__":
    main()