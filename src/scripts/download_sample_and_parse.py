#!/usr/bin/env python3
"""
scripts/download_sample_and_parse.py

Скрипт выборочного скачивания и парсинга PDF-статей arXiv из GCS-бакета.
Обрабатывает месяцы 2020–2025, берёт первые N PDF за месяц, скачивает их,
парсит в текст и сохраняет локально, удаляя временные PDF.
"""
import argparse
from pathlib import Path
from google.cloud import storage
from pdfminer.high_level import extract_text
from tqdm import tqdm


def download_and_parse(months, per_month, out_root, tmp_root):
    # Создаём анонимный клиент для публичного бакета
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("arxiv-dataset")

    out_root = Path(out_root)
    tmp_root = Path(tmp_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    for ym in months:
        print(f"\n>>> Processing {ym}, sampling {per_month} PDFs <<<")
        prefix = f"arxiv/arxiv/pdf/{ym}/"
        # Получаем список объектов (blobs)
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            print(f"No PDFs found for {ym}")
            continue
        sampled = blobs[:per_month]

        month_out = out_root / ym
        month_out.mkdir(parents=True, exist_ok=True)

        for blob in tqdm(sampled, desc=f"{ym} downloading", unit="file"):
            pid = Path(blob.name).stem  # e.g. '2301.12345v1'
            tmp_pdf = tmp_root / f"{pid}.pdf"
            txt_path = month_out / f"{pid}.txt"

            # Пропустить, если текст уже есть
            if txt_path.exists():
                continue

            # Скачиваем PDF во временную папку
            blob.download_to_filename(str(tmp_pdf))

            # Парсим PDF в текст
            try:
                text = extract_text(str(tmp_pdf))
                if text.strip():
                    txt_path.write_text(text, encoding="utf-8")
                else:
                    print(f"Warning: empty text for {pid}")
            except Exception as e:
                print(f"Failed to parse {pid}: {e}")
            finally:
                # Удаляем временный PDF
                try:
                    tmp_pdf.unlink()
                except OSError:
                    pass

        print(f"Done {ym}: text files in {month_out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample download and parse arXiv PDFs by month"
    )
    parser.add_argument(
        "--per-month", "-n", type=int, default=10,
        help="Number of PDFs to download per month"
    )
    parser.add_argument(
        "--out-dir", "-o", default="data/raw/texts",
        help="Directory to save parsed text files"
    )
    parser.add_argument(
        "--tmp-dir", "-t", default="tmp",
        help="Temporary directory for downloading PDFs"
    )
    return parser.parse_args()


def build_month_list(start_year=2020, end_year=2025, end_month=7):
    months = []
    for year in range(start_year, end_year + 1):
        yy = str(year)[2:]
        last = 12 if year < end_year else end_month
        for m in range(1, last + 1):
            months.append(f"{yy}months{m:02d}")
    return months


if __name__ == "__main__":
    args = parse_args()
    months = build_month_list(2020, 2025, end_month=7)
    download_and_parse(months, args.per_month, args.out_dir, args.tmp_dir)
