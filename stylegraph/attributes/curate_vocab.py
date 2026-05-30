import argparse
from collections import Counter
from pathlib import Path

from stylegraph.utils.io import read_jsonl, write_json, write_jsonl

KEYS = ["color", "occasion", "style", "fit", "season", "material"]

SYNONYMS = {
    "boho": "bohemian",
    "navy blue": "navy",
    "blk": "black",
    "off white": "ivory",
}


def normalize_value(value: str) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip().lower()
    if not value or value in {"unknown", "n/a", "na", "none"}:
        return "unknown"
    return SYNONYMS.get(value, value)


def cluster_values(values, model_name: str, min_cluster_size: int):
    from sentence_transformers import SentenceTransformer
    import hdbscan

    model = SentenceTransformer(model_name)
    embeddings = model.encode(values, normalize_embeddings=True, show_progress_bar=True)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate attribute vocabulary.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.input)))
    values_by_key = {key: [] for key in KEYS}

    for row in rows:
        attrs = row.get("attrs") or row.get("attrs_canon") or {}
        for key in KEYS:
            value = normalize_value(attrs.get(key))
            if value != "unknown":
                values_by_key[key].append(value)

    mapping = {key: {} for key in KEYS}
    vocab = {key: [] for key in KEYS}

    for key in KEYS:
        counts = Counter(values_by_key[key])
        filtered = [v for v, c in counts.items() if c >= args.min_count]
        filtered.sort()

        if args.cluster and filtered:
            try:
                labels = cluster_values(filtered, args.model, args.min_cluster_size)
            except Exception:
                labels = [-1 for _ in filtered]
        else:
            labels = [-1 for _ in filtered]

        label_to_values = {}
        for value, label in zip(filtered, labels):
            label_to_values.setdefault(label, []).append(value)

        for label, values in label_to_values.items():
            if label == -1:
                for value in values:
                    mapping[key][value] = value
                    vocab[key].append(value)
                continue

            ranked = sorted(values, key=lambda v: counts[v], reverse=True)
            canonical = ranked[0]
            for value in values:
                mapping[key][value] = canonical
            vocab[key].append(canonical)

        vocab[key] = sorted(set(vocab[key]))

    curated_rows = []
    for row in rows:
        attrs = row.get("attrs") or row.get("attrs_canon") or {}
        canon_attrs = {}
        for key in KEYS:
            raw = normalize_value(attrs.get(key))
            canon_attrs[key] = mapping[key].get(raw, "unknown")

        parts = [f"{key}: {canon_attrs[key]}" for key in KEYS if canon_attrs[key] != "unknown"]
        attr_text = "; ".join(parts) if parts else "unknown"

        curated_rows.append(
            {
                "id": row.get("id"),
                "attrs_canon": canon_attrs,
                "attr_text": attr_text,
            }
        )

    output_dir = Path(args.output_dir)
    write_json(output_dir / "vocab.json", {"keys": KEYS, "values": vocab, "mapping": mapping})
    write_jsonl(output_dir / "attrs_canon.jsonl", curated_rows)


if __name__ == "__main__":
    main()
