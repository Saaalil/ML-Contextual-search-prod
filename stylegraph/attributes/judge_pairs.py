import argparse
import json
import re
from pathlib import Path

from stylegraph.utils.io import read_jsonl

KEYS = ["color", "occasion", "style", "fit", "season", "material"]


def build_attr_text(attrs: dict) -> str:
    parts = [f"{key}: {attrs.get(key, 'unknown')}" for key in KEYS if attrs.get(key) != "unknown"]
    return "; ".join(parts) if parts else "unknown"


def score_heuristic(attrs: dict, title: str) -> float:
    known = [v for v in attrs.values() if v and v != "unknown"]
    coverage = len(known) / len(KEYS)
    score = 0.7 * coverage
    if len(known) >= 3:
        score += 0.2
    if title:
        score += 0.1
    return min(score, 1.0)


def parse_score(text: str) -> float:
    match = re.search(r"(0\.\d+|1\.0|1)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def score_openai(client, model: str, title: str, attrs: dict) -> float:
    prompt = (
        "Score how searchable these attributes are for shopping (0 to 1). "
        "Return only a number.\n"
        f"Title: {title}\n"
        f"Attributes: {json.dumps(attrs)}"
    )
    response = client.responses.create(model=model, input=prompt)
    return parse_score(response.output_text or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter product-attribute pairs.")
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--attrs", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--provider", type=str, default="heuristic", choices=["heuristic", "openai"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()

    catalog = {row["id"]: row for row in read_jsonl(Path(args.catalog))}

    client = None
    if args.provider == "openai":
        from openai import OpenAI

        client = OpenAI()

    written = 0
    with open(Path(args.output), "w", encoding="utf-8") as f:
        for row in read_jsonl(Path(args.attrs)):
            if args.max_items and written >= args.max_items:
                break

            product = catalog.get(row.get("id"))
            if not product:
                continue

            attrs = row.get("attrs_canon") or row.get("attrs") or {}
            attr_text = row.get("attr_text") or build_attr_text(attrs)

            if args.provider == "openai":
                score = score_openai(client, args.model, product.get("title", ""), attrs)
            else:
                score = score_heuristic(attrs, product.get("title", ""))

            if score < args.min_score:
                continue

            output_row = {
                "id": product.get("id"),
                "image_path": product.get("image_path"),
                "text": attr_text,
                "score": score,
                "title": product.get("title"),
                "category": product.get("category"),
                "price": product.get("price"),
            }
            f.write(json.dumps(output_row, ensure_ascii=True) + "\n")
            written += 1


if __name__ == "__main__":
    main()
