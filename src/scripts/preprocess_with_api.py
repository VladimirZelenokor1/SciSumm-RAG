#!/usr/bin/env python3
# src/scripts/preprocess_pipeline.py

import argparse
import json
import re
import time
from pathlib import Path

import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# 1) One-time downloads of NLTK models
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def fetch_abstract(paper_id: str) -> str:
    """
    Queries the arXiv API for an abstract for a given paper_id (without version),
 returns a plain text summary or "".
    """
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    summ = root.find(".//atom:entry/atom:summary", ns)
    return summ.text.strip() if summ is not None else ""


def clean_text(text: str) -> str:
    """
    Clears from:
     - HTML tags
     - LaTeX math $...$
     - LaTeX commands \cmd{...}, curly brackets { }
     - common ligatures
     - any non-alphanumeric characters (leave a space)
     - normalizes spaces
    """
    # 1. strip HTML
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        pass

    # 2. remove inline math $...$
    text = re.sub(r"\$[^$]+\$", " ", text)
    # 3. remove LaTeX commands and braces
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    # 4. ligatures
    for lig, rep in [("ﬁ","fi"),("ﬂ","fl"),("ﬀ","ff"),("ﬃ","ffi"),("ﬄ","ffl")]:
        text = text.replace(lig, rep)
    # 5. remove non-alphanumeric (except spaces)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    # 6. normalize spaces
    return " ".join(text.split())


def tokenize_normalize(text: str) -> list[list[str]]:
    """
    1. sent_tokenize
    2. word_tokenize
    3. lowercase, alphabetic only, drop stop words
    """
    out = []
    for sent in sent_tokenize(text):
        words = word_tokenize(sent)
        norm = [
            w.lower()
            for w in words
            if w.isalpha() and w.lower() not in STOP_WORDS
        ]
        if norm:
            out.append(norm)
    return out


def select_latest_versions(texts_dir: Path) -> dict[str, Path]:
    """
    Scans texts_dir for *.txt, looks for files like baseidvN.txt,
    and returns the dictionary baseid -> path to the file with the largest N.
    """
    candidates: dict[str, tuple[int, Path]] = {}
    for p in texts_dir.rglob("*.txt"):
        stem = p.stem  # e.g. "2102.00007v2" or "2102.00008"
        m = re.match(r"^(.+?)v(\d+)$", stem)
        if m:
            base, ver = m.group(1), int(m.group(2))
        else:
            base, ver = stem, 0
        prev = candidates.get(base)
        if prev is None or ver > prev[0]:
            candidates[base] = (ver, p)
    return {base: path for base, (ver, path) in candidates.items()}


def main(texts_dir: Path, out_dir: Path, delay: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = select_latest_versions(texts_dir)
    print(f"Found {len(latest)} unique papers → processing…")

    for idx, (paper_id, txt_path) in enumerate(latest.items(), 1):
        print(f"[{idx}/{len(latest)}] {paper_id}")

        # load and clean body
        raw_body = txt_path.read_text(encoding="utf-8", errors="ignore")
        clean_body = clean_text(raw_body)

        # fetch, clean, tokenize abstract
        try:
            raw_abs = fetch_abstract(paper_id)
        except Exception as e:
            print(f"  → failed to fetch abstract: {e}, skipping")
            continue
        clean_abs = clean_text(raw_abs)

        # tokenize & normalize
        body_tok = tokenize_normalize(clean_body)
        abs_tok  = tokenize_normalize(clean_abs)

        # save JSON
        rec = {"paper_id": paper_id, "body": body_tok, "abstract": abs_tok}
        dest = out_dir / f"{paper_id}.json"
        dest.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

        time.sleep(delay)

    print("Done. Processed files are in", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocess texts + fetch abstracts (latest versions only)"
    )
    p.add_argument(
        "texts_dir", type=Path,
        help="Root folder with .txt files (e.g. data/raw/texts)"
    )
    p.add_argument(
        "--out-dir", "-o", type=Path, default=Path("data/processed"),
        help="Where to save processed JSONs"
    )
    p.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API calls"
    )
    args = p.parse_args()
    main(args.texts_dir, args.out_dir, args.delay)
