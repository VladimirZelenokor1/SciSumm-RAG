import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def download_arxiv(dataset: str = "Cornell-University/arxiv"):
    # 1) Find the root of the project
    project_root = Path(__file__).resolve().parents[2]

    # 2) Build the absolute path data/raw/arxiv inside the project
    dest_dir = project_root / "data" / "raw" / "arxiv"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 3) Start the download
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)
    print(f"The data has been uploaded to {dest_dir}")

if __name__ == "__main__":
    download_arxiv()
