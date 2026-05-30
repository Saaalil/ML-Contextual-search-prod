import argparse
import os
import re
from pathlib import Path

import gradio as gr

from stylegraph.config import INDEX_DIR
from stylegraph.retrieval.search import SearchEngine


def extract_price_limit(text: str):
    match = re.search(r"under\s*\$?(\d+)", text.lower())
    if not match:
        return None
    return float(match.group(1))


def results_to_gallery(engine: SearchEngine, results):
    gallery = []
    for item in results:
        image = engine.load_image(item["image_path"])
        caption = f"{item.get('title', '')} | ${item.get('price', 0):.2f}"
        gallery.append((image, caption))
    return gallery


def build_no_index_app(index_dir: Path):
    with gr.Blocks() as demo:
        gr.Markdown(
            "# StyleGraph\n"
            "Index not found. Build one and set STYLEGRAPH_INDEX_DIR or pass --index-dir.\n"
            f"Expected: {index_dir / 'faiss.index'}"
        )
    return demo


def build_app(index_dir: str | None = None, device: str | None = None):
    index_dir = index_dir or os.environ.get("STYLEGRAPH_INDEX_DIR")
    index_dir = Path(index_dir) if index_dir else INDEX_DIR
    device = device or os.environ.get("STYLEGRAPH_DEVICE", "cpu")

    if not (index_dir / "faiss.index").exists():
        return build_no_index_app(index_dir)

    engine = SearchEngine.from_dir(index_dir, device=device)

    def search_by_image(image, top_k):
        if image is None:
            return []
        results = engine.search_image(image, top_k=int(top_k))
        return results_to_gallery(engine, results)

    def search_by_text(text, top_k):
        if not text:
            return []
        price_limit = extract_price_limit(text)
        results = engine.search_text(text, top_k=int(top_k) * 3)
        if price_limit is not None:
            results = [r for r in results if float(r.get("price", 1e9)) <= price_limit]
        return results_to_gallery(engine, results[: int(top_k)])

    with gr.Blocks() as demo:
        gr.Markdown("# StyleGraph - Multimodal Fashion Collection Builder")

        with gr.Tab("Image to collection"):
            image_in = gr.Image(type="pil")
            top_k_img = gr.Slider(5, 50, value=20, step=1, label="Top K")
            search_btn = gr.Button("Search")
            gallery_img = gr.Gallery(columns=4, height=600)
            search_btn.click(search_by_image, inputs=[image_in, top_k_img], outputs=[gallery_img])

        with gr.Tab("Text to collection"):
            text_in = gr.Textbox(placeholder="office outfits under $100, minimal aesthetic")
            top_k_txt = gr.Slider(5, 50, value=20, step=1, label="Top K")
            search_btn_txt = gr.Button("Search")
            gallery_txt = gr.Gallery(columns=4, height=600)
            search_btn_txt.click(search_by_text, inputs=[text_in, top_k_txt], outputs=[gallery_txt])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gradio demo.")
    parser.add_argument("--index-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    app = build_app(index_dir=args.index_dir or None, device=args.device)
    app.launch()


if __name__ == "__main__":
    main()
