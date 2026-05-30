import json
from pathlib import Path

import faiss
import numpy as np
import torch
import open_clip

from stylegraph.utils.io import load_image, read_jsonl


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    return Path(path_str)


class SearchEngine:
    def __init__(self, model, preprocess, tokenizer, index, metadata, device):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.index = index
        self.metadata = metadata
        self.device = device

    @classmethod
    def from_dir(cls, index_dir: Path, device: str = "cpu"):
        index_dir = Path(index_dir)
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.jsonl"
        model_path = index_dir / "model.json"

        if not index_path.exists() or not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError("Index files missing in index_dir")

        with open(model_path, "r", encoding="utf-8") as f:
            model_meta = json.load(f)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_meta["model_name"], pretrained=model_meta["pretrained"]
        )
        checkpoint = model_meta.get("checkpoint") or ""
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.is_absolute():
                if (index_dir / ckpt_path).exists():
                    ckpt_path = index_dir / ckpt_path
                elif (Path.cwd() / ckpt_path).exists():
                    ckpt_path = Path.cwd() / ckpt_path
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state, strict=False)

        device_t = torch.device(device)
        model.to(device_t)
        model.eval()

        tokenizer = open_clip.get_tokenizer(model_meta["model_name"])
        index = faiss.read_index(str(index_path))
        metadata = list(read_jsonl(meta_path))

        return cls(model, preprocess, tokenizer, index, metadata, device_t)

    def _search(self, query_vec: np.ndarray, top_k: int):
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def search_text(self, text: str, top_k: int = 20):
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
        return self._search(features.cpu().numpy(), top_k)

    def search_image(self, image, top_k: int = 20):
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                features = self.model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
        return self._search(features.cpu().numpy(), top_k)

    def load_image(self, path_str: str):
        return load_image(resolve_image_path(path_str))
