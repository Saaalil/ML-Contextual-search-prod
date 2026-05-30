import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import open_clip

from stylegraph.config import ROOT_DIR
from stylegraph.utils.io import ensure_dir, load_image


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT_DIR / path
    if candidate.exists():
        return candidate
    return path


def clip_loss(image_features, text_features, logit_scale, extra_text_features=None):
    if extra_text_features is not None and extra_text_features.numel() > 0:
        text_all = torch.cat([text_features, extra_text_features], dim=0)
    else:
        text_all = text_features

    logits_per_image = logit_scale * image_features @ text_all.t()
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)

    logits_per_text = logit_scale * text_features @ image_features.t()
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


class PairDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = resolve_image_path(row["image_path"])
        image = load_image(image_path)
        image = self.preprocess(image)
        return image, row["text"], row.get("hard_negs", [])


def collate_batch(batch):
    images, texts, neg_lists = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(texts), list(neg_lists)


def sample_negatives(neg_lists, per_sample):
    if per_sample <= 0:
        return []

    sampled = []
    for negs in neg_lists:
        if not negs:
            continue
        for _ in range(per_sample):
            sampled.append(random.choice(negs))

    seen = set()
    unique = []
    for text in sampled:
        if text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def save_checkpoint(output_dir: Path, model_name: str, pretrained: str, model) -> None:
    ensure_dir(output_dir)
    torch.save(model.state_dict(), output_dir / "checkpoint.pt")
    config = {"model_name": model_name, "pretrained": pretrained}
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CLIP-style dual encoder.")
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neg-per-sample", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    rows = []
    with open(Path(args.pairs), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.train()

    dataset = PairDataset(rows, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_dir = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, (images, texts, neg_lists) in enumerate(tqdm(loader, desc=f"epoch {epoch}"), 1):
            images = images.to(device)
            text_tokens = tokenizer(texts).to(device)
            neg_texts = sample_negatives(neg_lists, args.neg_per_sample)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                extra_text_features = None
                if neg_texts:
                    neg_tokens = tokenizer(neg_texts).to(device)
                    extra_text_features = model.encode_text(neg_tokens)
                    extra_text_features = extra_text_features / extra_text_features.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                loss = clip_loss(image_features, text_features, logit_scale, extra_text_features)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

            running_loss += loss.item()
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(f"epoch {epoch} step {step} loss {avg_loss:.4f}")
                running_loss = 0.0

        save_checkpoint(output_dir, args.model, args.pretrained, model)


if __name__ == "__main__":
    main()
