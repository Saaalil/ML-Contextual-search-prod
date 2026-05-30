import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from stylegraph.utils.io import ensure_dir, load_image, read_jsonl

KEYS = ["color", "occasion", "style", "fit", "season", "material"]


def build_prompt(title: str, category: str) -> str:
    return (
        "You are a fashion attribute extractor. "
        "Return JSON with keys: color, occasion, style, fit, season, material. "
        "Use short lower-case values. If unknown, use 'unknown'. "
        f"Title: {title}. Category: {category}."
    )


def normalize_value(value) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9\- ]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value if value else "unknown"


def extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def parse_attrs(text: str) -> dict:
    data = {}
    block = extract_json_block(text)
    if block:
        cleaned = block.replace("\n", " ")
        cleaned = cleaned.replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            data = json.loads(cleaned)
        except Exception:
            data = {}

    if not data:
        pairs = re.findall(
            r"(color|occasion|style|fit|season|material)\s*[:=]\s*([a-zA-Z0-9\- ]+)",
            text,
            flags=re.IGNORECASE,
        )
        for key, value in pairs:
            data[key.lower()] = value.strip()

    parsed = {}
    for key in KEYS:
        value = data.get(key) or data.get(key.lower())
        parsed[key] = normalize_value(value)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract attributes with BLIP-2.")
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    existing_ids = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            existing_ids.add(row.get("id"))

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    limit = args.limit if args.limit and args.limit > 0 else None
    seen = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for item in tqdm(read_jsonl(Path(args.catalog)), desc="extract"):
            if limit and seen >= limit:
                break
            if item.get("id") in existing_ids:
                continue

            image = load_image(Path(item["image_path"]))
            prompt = build_prompt(item.get("title", ""), item.get("category", ""))
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                autocast = torch.cuda.amp.autocast(enabled=device.type == "cuda")
                with autocast:
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

            text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
            attrs = parse_attrs(text)
            row = {"id": item.get("id"), "attrs": attrs}
            out_f.write(json.dumps(row, ensure_ascii=True) + "\n")
            seen += 1


if __name__ == "__main__":
    main()
