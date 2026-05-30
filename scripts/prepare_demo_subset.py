"""
Prepare a demo subset of 300 images from DeepFashion2 for the prototype.

Usage:
    python -m scripts.prepare_demo_subset
    python -m scripts.prepare_demo_subset --limit 300 --input-dir DeepFashion2/deepfashion2_original_images
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────

CATEGORY_MAP = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ROOT_DIR = Path(__file__).resolve().parents[1]


def synthetic_price(seed: str) -> float:
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    value = int(h[:8], 16)
    return float(10 + (value % 190))


def find_images(input_dir: Path) -> list[Path]:
    """Recursively find all image files."""
    images = []
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images.append(p)
    return images


def try_parse_category(image_path: Path, input_dir: Path) -> str:
    """Try to find an annotation file for category info."""
    # Look for annotations in parallel directory structures
    for annos_name in ["annos", "annotations"]:
        parts = image_path.relative_to(input_dir).parts
        for i in range(len(parts)):
            anno_dir = input_dir / Path(*parts[:i]) / annos_name
            anno_file = anno_dir / f"{image_path.stem}.json"
            if anno_file.exists():
                try:
                    with open(anno_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for value in data.values():
                        if isinstance(value, dict) and "category_id" in value:
                            return CATEGORY_MAP.get(
                                int(value["category_id"]), "unknown"
                            )
                except Exception:
                    pass
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare demo image subset.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT_DIR / "DeepFashion2" / "deepfashion2_original_images"),
        help="Root directory containing DeepFashion2 images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "demo_images"),
        help="Output directory for demo images.",
    )
    parser.add_argument(
        "--catalog-output",
        type=str,
        default=str(ROOT_DIR / "data" / "catalog" / "demo_products.jsonl"),
        help="Output JSONL catalog path.",
    )
    parser.add_argument("--limit", type=int, default=300, help="Number of images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    catalog_path = Path(args.catalog_output)

    if not input_dir.exists():
        print(f"❌  Input directory not found: {input_dir}")
        print("    Make sure DeepFashion2 images are available.")
        return

    print(f"🔍  Scanning images in {input_dir}...")
    all_images = find_images(input_dir)
    print(f"    Found {len(all_images)} images total.")

    if len(all_images) == 0:
        print("❌  No images found!")
        return

    # Sample subset
    random.seed(args.seed)
    limit = min(args.limit, len(all_images))
    selected = random.sample(all_images, limit)
    print(f"📸  Selected {limit} images for demo subset.")

    # Copy images and build catalog
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    with open(catalog_path, "w", encoding="utf-8") as f:
        for i, img_path in enumerate(tqdm(selected, desc="Copying")):
            # Use a clean filename
            new_name = f"demo_{i:04d}{img_path.suffix.lower()}"
            dest = output_dir / new_name
            shutil.copy2(img_path, dest)

            category = try_parse_category(img_path, input_dir)
            product_id = f"demo_{i:04d}"
            title = f"{category} item".replace("_", " ")
            price = synthetic_price(product_id)

            row = {
                "id": product_id,
                "image_path": dest.as_posix(),
                "title": title,
                "category": category,
                "price": price,
                "source": "deepfashion2_demo",
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"✅  {limit} images copied → {output_dir}")
    print(f"✅  Catalog written → {catalog_path}")


if __name__ == "__main__":
    main()
