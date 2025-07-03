#!/usr/bin/env python3
import json, logging
from pathlib import Path
import argparse
import numpy as np

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--out-dir",       type=Path, default=Path("data/clean/"))
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids_file = args.out_dir / "ids.jsonl"

    # merge parts
    parts = sorted(args.out_dir.glob("embeddings_part*.npy"))
    arrays = [np.load(p) for p in parts]
    all_vecs = np.vstack(arrays)
    np.save(args.out_dir / "embeddings.npy", all_vecs)
    ids = [json.loads(line) for line in ids_file.open(encoding='utf-8')]
    with open(args.out_dir / "ids.json", "w", encoding='utf-8') as f:
        json.dump(ids, f, ensure_ascii=False)
    logger.info("Final matrix: %d×%d, ids: %d", all_vecs.shape[0], all_vecs.shape[1], len(ids))


if __name__ == "__main__":
    main()