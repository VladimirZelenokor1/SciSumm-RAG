#!/usr/bin/env python3
# scripts/jsons_to_csv.py

import json
import csv
from pathlib import Path

def jsons_to_csv(input_dir: Path, output_csv: Path):
    """
    Проходит по всем .json в input_dir, пропускает те, у которых
    body или abstract пусты, и пишет в CSV:
      paper_id, body, abstract
    где body и abstract — это объединённый в строки токенизированный текст.
    """
    rows = []
    for fn in sorted(input_dir.glob("*.json")):
        data = json.loads(fn.read_text(encoding="utf-8"))
        body_tokens = data.get("body", [])
        abs_tokens  = data.get("abstract", [])

        if not body_tokens or not abs_tokens:
            # пропускаем неполные
            continue

        # Восстанавливаем строки: соединяем токены в предложения, предложения в текст
        body_text = " ".join(" ".join(sent) for sent in body_tokens)
        abs_text  = " ".join(" ".join(sent) for sent in abs_tokens)

        rows.append({
            "paper_id": data.get("paper_id", fn.stem),
            "body":      body_text,
            "abstract":  abs_text
        })

    # Записываем CSV
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id", "body", "abstract"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✔ Wrote {len(rows)} records to {output_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert processed JSONs to single CSV")
    p.add_argument("-i", "--input-dir",  default="data/processed",
                   help="Папка с JSON-файлами")
    p.add_argument("-o", "--output-csv", default="data/final_corpus.csv",
                   help="Итоговый CSV с paper_id, body, abstract")
    args = p.parse_args()

    jsons_to_csv(Path(args.input_dir), Path(args.output_csv))
