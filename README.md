---
title: StyleGraph - Multimodal Fashion Search
emoji: 👗
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: false
---

# StyleGraph

**Contextual multimodal fashion search** powered by Google's Gemini Embedding 2. Upload a photo or describe what you want — StyleGraph understands style, occasion, price, and more.

## Features

- 🔍 **Text-to-Image Search**: "red summer dress under $80" → contextually relevant results
- 🖼️ **Image-to-Image Search**: Upload a photo → find visually similar items
- 🧠 **3-Layer Contextual Pipeline**:
  1. **Query Understanding** (Gemini Flash) — parses intent, filters, exclusions
  2. **Vector Retrieval** (Gemini Embedding 2 + FAISS) — fast nearest-neighbor search
  3. **Contextual Re-ranking** (Gemini Flash) — scores relevance beyond raw similarity
- 💰 All free — no paid APIs, no GPU needed for inference

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your free API key
Get one at [aistudio.google.com](https://aistudio.google.com) — no credit card needed.
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY = "your_key_here"

# Linux/Mac:
export GEMINI_API_KEY="your_key_here"
```

### 3. Prepare demo images (300 from DeepFashion2)
```bash
python -m scripts.prepare_demo_subset --limit 300
```

### 4. Build the FAISS index
```bash
python -m stylegraph.model.build_gemini_index
```

### 5. Run the app
```bash
python app.py
```

## Architecture

```
User Query (text or image)
    │
    ├── [Layer 1] Gemini Flash — Parse intent, extract filters
    │
    ├── [Layer 2] Gemini Embedding 2 — Vector similarity via FAISS
    │
    └── [Layer 3] Gemini Flash — Contextual re-ranking
    │
    └── Final Results (filtered, re-ranked, contextually relevant)
```

### Why contextual, not just semantic?

| Feature | Semantic Search | StyleGraph (Contextual) |
|---------|----------------|------------------------|
| "red dress under $50" | Ignores price | Filters by price ≤ $50 |
| "office outfit, not sporty" | Can't handle negation | Excludes sporty items |
| "beach wedding guest" | Matches keywords | Understands occasion context |
| "bohemian summer vibes" | Surface similarity | Deep style understanding |

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (Gradio SDK).
2. Push this repo to the Space.
3. Add `GEMINI_API_KEY` as a secret in Space Settings.
4. Upload the `data/index/` directory (faiss.index + metadata.jsonl + model.json).
5. The Space runs `app.py` by default.

## Pipeline Stages (for training / custom datasets)

| Stage | Script | Description |
|-------|--------|-------------|
| Data prep | `scripts/prepare_demo_subset.py` | Extract demo images from DeepFashion2 |
| Catalog | `stylegraph/data/build_catalog.py` | Build product catalog JSONL |
| Attributes | `stylegraph/attributes/extract_blip2.py` | BLIP-2 attribute extraction (GPU) |
| Vocab | `stylegraph/attributes/curate_vocab.py` | LLM-powered vocab curation |
| Index | `stylegraph/model/build_gemini_index.py` | **Gemini Embedding 2 → FAISS** |
| Search | `stylegraph/retrieval/search.py` | 3-layer contextual search engine |
| Demo | `stylegraph/demo/app.py` | Gradio web UI |

## Tech Stack (all free)

| Component | Technology | Cost |
|-----------|------------|------|
| Embeddings | Gemini Embedding 2 (Google AI Studio) | Free |
| Query Understanding | Gemini 2.0 Flash | Free |
| Re-ranking | Gemini 2.0 Flash | Free |
| Vector Search | FAISS (CPU) | Free |
| UI | Gradio | Free |
| Hosting | Hugging Face Spaces | Free |
| CI/CD | GitHub Actions | Free |

## Notes

- `faiss-cpu` is not available on native Windows pip. Use WSL, conda, or build the index on Linux (Kaggle/Colab).
- DeepFashion2 requires manual download due to license terms.
- The old OpenCLIP pipeline is preserved for backward compatibility.
