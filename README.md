# StyleGraph

StyleGraph is a personal-scale, single-GPU pipeline for building shoppable fashion collections from images and metadata. It covers:

- Data ingestion (DeepFashion2 + optional extras)
- VLM attribute extraction (BLIP-2)
- Attribute curation
- Dual encoder training (OpenCLIP)
- FAISS indexing and retrieval
- Gradio demo + Hugging Face Spaces deployment

## Quick start (demo after you build an index)
1. Create a virtual environment.
2. Install demo dependencies: pip install -r requirements.txt
3. Build an index (see "Build the index").
4. Run the app: python app.py

## Setup
Python 3.10+ recommended.

Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Linux/macOS:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

For training and BLIP-2 extraction, also install:
pip install -r requirements-train.txt

Optional LLM judge:
pip install -r requirements-llm.txt

## Stage 1 - Data (DeepFashion2)
1. Download DeepFashion2 and place it in data/raw/deepfashion2.
2. Build a catalog:
python -m stylegraph.data.build_catalog --input-dir data/raw/deepfashion2 --output data/catalog/products.jsonl --splits train,validation --limit 50000

DeepFashion2 does not include titles or prices, so this script generates a simple title and a synthetic price per item.

## Stage 2 - Attribute extraction (BLIP-2)
Run on GPU (Colab or Kaggle recommended):
python -m stylegraph.attributes.extract_blip2 --catalog data/catalog/products.jsonl --output data/attrs/attrs.jsonl --device cuda --limit 20000

You can change the model with --model if VRAM is limited.

## Stage 2b - Attribute curation (optional but recommended)
python -m stylegraph.attributes.curate_vocab --input data/attrs/attrs.jsonl --output-dir data/attrs --cluster

This produces:
- data/attrs/vocab.json
- data/attrs/attrs_canon.jsonl

## Stage 2c - LLM-as-judge (optional)
python -m stylegraph.attributes.judge_pairs --catalog data/catalog/products.jsonl --attrs data/attrs/attrs_canon.jsonl --output data/attrs/pairs.jsonl

## Stage 3 - Hard negatives (optional)
python -m stylegraph.model.mine_hard_negatives --pairs data/attrs/pairs.jsonl --output data/attrs/pairs_hard.jsonl --negatives 5

## Stage 3 - Train dual encoder
python -m stylegraph.model.train_dual_encoder --pairs data/attrs/pairs_hard.jsonl --output models/clip --device cuda --batch-size 64 --epochs 3 --neg-per-sample 2

## Stage 3b - Build the index
python -m stylegraph.model.build_faiss --catalog data/catalog/products.jsonl --attrs data/attrs/attrs_canon.jsonl --model-dir models/clip --output data/index --device cuda

## Stage 4 - Run the demo
python app.py

You can override the index path:
STYLEGRAPH_INDEX_DIR=data/index python app.py

## Deploy to Hugging Face Spaces
1. Create a new Space (Gradio).
2. Push this repo to the Space.
3. Build the FAISS index locally and upload data/index to the Space (or build inside the Space if you have storage).
4. Add optional secrets (OPENAI_API_KEY) if you use the LLM judge.
5. The Space will run app.py by default.

Notes:
- faiss-cpu is not available on native Windows pip. Use WSL, conda, or build the index on Linux (Kaggle/Colab) and copy data/index back.
- DeepFashion2 requires manual download due to license terms.
