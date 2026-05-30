import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
import open_clip

from stylegraph.utils.io import read_jsonl


def embed_texts(model, tokenizer, texts, device, batch_size=256):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
    return np.vstack(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negative texts.")
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=5)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.pairs)))
    texts = [row["text"] for row in rows]

    device = torch.device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)

    embeddings = embed_texts(model, tokenizer, texts, device)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    scores, indices = index.search(embeddings, args.top_k + 1)

    with open(Path(args.output), "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(rows, desc="mine")):
            candidates = []
            for idx in indices[i]:
                if idx == i:
                    continue
                candidates.append(texts[idx])
                if len(candidates) >= args.negatives:
                    break

            output_row = dict(row)
            output_row["hard_negs"] = candidates
            f.write(json.dumps(output_row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
