# SciSumm-RAG

**SciSumm-RAG** — Retrieval-Augmented Generation (RAG) pipeline with fine-tuned BART for automated abstract-style summarization of arXiv papers. Given a user query, the system retrieves the most relevant passages from a curated subset of arXiv articles and then generates a coherent, concise summary using either zero-shot "sshleifer/distilbart-cnn-12-6" or a custom fine-tuned BART model .

## Project Structure
```graphql
.
├── app.py                     # entry point for the web service demo
├── data/                      # data folder / git empty
├── experiments/               # bart-tuning results & configs
├── notebooks/                 
│   ├── "EDA and clean.ipynb"              # EDA, preliminary data cleaning for indexing
│   ├── "Eda_new.ipynb"                    # EDA for BART data 
│   ├── "finetune_evaluation.ipynb"        # baseline & finetuned BART comparison with ROUGE
│   ├── "retrieval_baseline.ipynb"         # baseline retrieval pipeline & test summary with baseline model
├── src/
│   ├── data/                  # dataset preparation and loading (for index)
│   ├── retriever/             # embeddings and FAISS search
│   │   ├── embed.py
│   │   ├── embed_combine.py
│   │   └── index.py
│   ├── generator/             # wrappers for "sshleifer/distilbart-cnn-12-6" and BART summary
│   │   └── hf_summarizer.py
│   ├── models/             # BART train & evaluate
│   │   ├── split_train_val.py
│   │   ├── train_bart.py
│   │   └── collect_metrics.py
│   └── scripts/               # creating a data corpus for BART training
├── tests/                     # unit tests (pytest)
├── requirements.txt
├── test_faiss_search.py       # testing index & hybrid search and metrics
└── README.md
```

## Getting Started

### 0. Clone the repository

```bash
git clone https://github.com/VladimirZelenokor1/SciSumm-RAG
cd SciSumm-RAG
```

### 1. Create a virtual environment and activate it

MacOS / Linux:
    
```bash
python -m venv venv
source venv/bin/activate
```
Windows:

```bash
python -m venv venv
venv\scripts\activate
```

### 2. Install dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118 
pip install -r requirements.txt
```

### 3. Put your kaggle.json in the project root & Run
```bash
python src/data/setup_kaggle.py
```

## Usage
### 0. Download data for index creation

```bash
python src/data/download_arxiv.py
```

### 1. Embedding generation
#### Create parts & combine
```bash
python src/retriever/embed.py --metadata-path data/clean/metadata_clean.csv --out-dir data/clean/ 
python src/retriever/embed_combine.py --metadata-path data/clean/metadata_clean.csv --out-dir data/clean/ 
```

### 2. Indexing
#### You have 5 options for indexing via FAISS
```bash
python src/retriever/index.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index-prefix data/index/faiss/flat_index --type FlatIP
python src/retriever/index.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index-prefix data/index/faiss/flatl2_index --type FlatL2 
python src/retriever/index.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index-prefix data/index/faiss/hnsw_index --type HNSW  
python src/retriever/index.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index-prefix data/index/faiss/ivfpq_index --type IVFPQ  
python src/retriever/index.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index-prefix data/index/faiss/opqivfpq_index --type OPQIVFPQ  
```

### 3. Test Indexing Search
#### You can test index search via direct index or hybrid mode (Mixed search: first FAISS, then re-ranking by CrossEncoder.)
```bash
python test_faiss_search.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index data/index/faiss/flat_index.index --mode flat --topk 5 --sample-size 1000
python test_faiss_search.py --embeddings data/clean/embeddings.npy --ids data/clean/ids.json --index data/index/faiss/ivfpq_index.index --chunks data/clean/chunks.jsonl --mode hybrid --topk-coarse 50 --topk 5 --sample-size 500 --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2
```

## BART Training
### 1. Download, clean data (for BART train)
```bash
python src/scripts/download_sample_and_parse.py -n 10 -o data/raw/texts -t tmp
python src/scripts/preprocess_with_api.py data/raw/texts -o data/processed --delay 1.0
python src/scripts/jsons_to_csv.py -i data/processed -o data/training/train_pairs.csv
```

### 2. Split data, train & evaluate 
```bash
 python src/models/split_train_val.py
 python src/models/train_bart.py 
 python src/models/collect_metrics.py 
```

## Demo (Streamlit)
```bash
streamlit run app.py
```

## Testing
#### Run all the tests
```bash
pytest tests/ 
```

### Notes

### Important

```bash
Python == 3.12.5
pip
```

## Further development
* Additional training of other models (T5, PEGASUS)
* Support for extended set of metrics (BERTScore)
* Deployment in Docker/Kubernetes