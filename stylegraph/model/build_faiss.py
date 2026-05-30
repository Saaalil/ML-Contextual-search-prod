import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
import open_clip

from stylegraph.config import ROOT_DIR
from stylegraph.utils.io import ensure_dir, load_image, read_jsonl


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT_DIR / path
    if candidate.exists():
        return candidate
    return path


def load_model_from_dir(model_dir: Path, device: torch.device):
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "checkpoint.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, _, preprocess = open_clip.create_model_and_transforms(
        config["model_name"], pretrained=config["pretrained"]
    )
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    return model, preprocess, config, checkpoint_path


def embed_images(model, preprocess, image_paths, device, batch_size=128):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [preprocess(load_image(resolve_image_path(p))) for p in batch_paths]
            images = torch.stack(images).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
    return np.vstack(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from product images.")
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--attrs", type=str, default="")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    model, preprocess, config, checkpoint_path = load_model_from_dir(model_dir, device)

    catalog_rows = list(read_jsonl(Path(args.catalog)))
    attrs = {}
    if args.attrs:
        for row in read_jsonl(Path(args.attrs)):
            attrs[row["id"]] = row.get("attrs_canon") or row.get("attrs") or {}

    image_paths = [row["image_path"] for row in catalog_rows]
    embeddings = embed_images(model, preprocess, image_paths, device, batch_size=args.batch_size)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "faiss.index"))

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in catalog_rows:
            meta = dict(row)
            if row.get("id") in attrs:
                meta["attrs"] = attrs[row["id"]]
            f.write(json.dumps(meta, ensure_ascii=True) + "\n")

    checkpoint_value = ""
    if checkpoint_path.exists():
        try:
            checkpoint_value = str(checkpoint_path.relative_to(ROOT_DIR))
        except ValueError:
            checkpoint_value = str(checkpoint_path)

    model_meta = {
        "model_name": config["model_name"],
        "pretrained": config["pretrained"],
        "checkpoint": checkpoint_value,
    }
    with open(output_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
