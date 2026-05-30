"""
Gradio demo — premium multimodal fashion search powered by Gemini.

Features:
- Image-to-image search (upload a photo → find similar)
- Text-to-image contextual search (describe what you want → find matches)
- Price filtering, occasion awareness, style understanding
- Modern glassmorphism UI with dark theme
"""

import argparse
import logging
import os
import re
from pathlib import Path

import gradio as gr
from PIL import Image

from stylegraph.config import INDEX_DIR, TOP_K

logger = logging.getLogger(__name__)

# ── Custom CSS for premium look ──────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@8..144,400..700&family=JetBrains+Mono:wght@400;500;700&display=swap');

* {
    font-family: 'Google Sans Flex', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background-color: #0b0f19 !important;
    background-image: radial-gradient(circle at 50% 0%, rgba(66, 133, 244, 0.15), transparent 40%),
                      radial-gradient(circle at 100% 100%, rgba(234, 67, 53, 0.1), transparent 40%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-top: 2px solid #4285F4;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.main-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #e8eaed !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.05em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

.main-header h1 span.highlight {
    background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header p {
    color: #9aa0a6 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 400;
}

.search-tab {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
}

.result-gallery {
    background: transparent !important;
    border: none !important;
}

.search-btn {
    background: #4285F4 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    font-weight: 500 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}

.search-btn:hover {
    background: #3b78e7 !important;
    transform: translateY(-1px) !important;
}

.context-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}

.badge-filter {
    background: rgba(66, 133, 244, 0.1) !important;
    color: #8ab4f8 !important;
    border: 1px solid rgba(66, 133, 244, 0.3) !important;
}

.badge-intent {
    background: rgba(52, 168, 83, 0.1) !important;
    color: #81c995 !important;
    border: 1px solid rgba(52, 168, 83, 0.3) !important;
}

footer { display: none !important; }

.tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #9aa0a6 !important;
}

.tab-nav button.selected {
    color: #e8eaed !important;
    border-bottom: 2px solid #4285F4 !important;
}

.example-queries {
    color: #5f6368 !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin-top: 0.5rem;
}
"""

# ── App Logic ────────────────────────────────────────────────────────


def build_no_index_app(index_dir: Path):
    """Show a helpful message when no index is found."""
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(
            """
            <div class="main-header">
                <h1>> <span class="highlight">Antigravity</span> StyleGraph</h1>
                <p>AI-Powered Visual Search Prototype</p>
            </div>
            """
        )
        gr.Markdown(
            f"""
            ### 🔧 Index Not Found

            The FAISS index has not been built yet. Follow these steps:

            1. **Prepare demo images** (300 from DeepFashion2):
               ```
               python -m scripts.prepare_demo_subset --limit 300
               ```

            2. **Build the Gemini FAISS index**:
               ```
               set GEMINI_API_KEY=your_key_here
               python -m stylegraph.model.build_gemini_index
               ```

            3. **Run the app**:
               ```
               python app.py
               ```

            Expected index path: `{index_dir / 'faiss.index'}`
            """
        )
    return demo


def build_app(index_dir: str | None = None, device: str | None = None):
    """Build the main Gradio app."""
    index_dir = index_dir or os.environ.get("STYLEGRAPH_INDEX_DIR")
    index_dir = Path(index_dir) if index_dir else INDEX_DIR

    if not (index_dir / "faiss.index").exists():
        return build_no_index_app(index_dir)

    # Lazy-load engine
    from stylegraph.retrieval.search import SearchEngine

    engine = SearchEngine.from_dir(index_dir, device=device or "cpu")

    def results_to_gallery(results, query=""):
        """Convert search results to Gradio gallery format."""
        gallery = []
        
        # Make a nice dynamic title out of the query if the DB doesn't have one
        fallback_title = "Fashion Item"
        if query and len(query) < 20:
            fallback_title = f"{query.title()} Match"
            
        for item in results:
            try:
                image = engine.load_image(item["image_path"])
                score = item.get("rerank_score", item.get("score", 0))
                title = item.get("title", "Unknown")
                
                # Hide the ugly 'unknown item' label
                if title.lower() == "unknown item" or title.lower() == "unknown":
                    title = fallback_title
                    
                price = item.get("price", 0)
                reason = item.get("rerank_reason", "")

                caption = f"{title} · ${price:.0f} · Score: {score:.2f}"
                if reason:
                    caption += f"\n{reason}"

                gallery.append((image, caption))
            except Exception as e:
                logger.warning(f"Could not load image: {e}")
        return gallery

    def format_context_info(parsed: dict) -> str:
        """Format the parsed query context as a readable string."""
        if not parsed:
            return ""

        parts = []
        filters = parsed.get("filters", {})
        if filters:
            filter_badges = []
            for k, v in filters.items():
                if v is not None:
                    if k.startswith("price"):
                        filter_badges.append(f"💰 {k}: ${v}")
                    else:
                        filter_badges.append(f"🏷️ {k}: {v}")
            if filter_badges:
                parts.append("**Filters:** " + " · ".join(filter_badges))

        exclude = parsed.get("exclude", [])
        if exclude:
            parts.append("**Excluding:** " + ", ".join(f"~~{e}~~" for e in exclude))

        intent = parsed.get("intent", "")
        if intent:
            intent_labels = {
                "find_items": "🔍 Finding matching items",
                "find_similar": "🔄 Finding similar items",
                "browse_category": "📂 Browsing category",
            }
            parts.append(intent_labels.get(intent, f"Intent: {intent}"))

        semantic = parsed.get("semantic_query", "")
        if semantic:
            parts.append(f"**Searching for:** _{semantic}_")

        return "\n\n".join(parts) if parts else ""

    def search_by_text(text, top_k):
        """Text-to-image contextual search."""
        if not text:
            return [], ""
        try:
            results = engine.search_text(text, top_k=int(top_k))
            # Try to get parsed context info
            context_info = ""
            if engine.query_parser:
                try:
                    parsed = engine.query_parser.parse(text)
                    context_info = format_context_info(parsed)
                except Exception:
                    pass
            return results_to_gallery(results, text), context_info
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [], f"⚠️ Error: {str(e)}"

    def search_by_image(image, top_k):
        """Image-to-image visual search."""
        if image is None:
            return []
        try:
            results = engine.search_image(image, top_k=int(top_k))
            return results_to_gallery(results, "Similar Items")
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return []

    # ── Build the UI ─────────────────────────────────────────────────

    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
        title="StyleGraph — Multimodal Fashion Search",
    ) as demo:
        # Header
        gr.HTML(
            """
            <div class="main-header">
                <h1>> <span class="highlight">Antigravity</span> StyleGraph</h1>
                <p>AI-Powered Visual Search Prototype</p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            # ── Text Search Tab ──────────────────────────────────────
            with gr.Tab("✍️ Text Search", id="text-tab"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            placeholder="Try: 'casual summer dress under $80' or 'office outfit, minimal aesthetic'",
                            label="Describe what you're looking for",
                            lines=2,
                            elem_id="text-search-input",
                        )
                        gr.HTML(
                            '<p class="example-queries">💡 Try: '
                            '"red floral dress for beach" · '
                            '"winter jacket under $100" · '
                            '"bohemian skirt, not too formal" · '
                            '"streetwear shorts"</p>'
                        )
                    with gr.Column(scale=1):
                        top_k_text = gr.Slider(
                            5, 50, value=20, step=1, label="Results"
                        )
                        text_btn = gr.Button(
                            "🔍 Search",
                            variant="primary",
                            elem_classes=["search-btn"],
                        )

                context_display = gr.Markdown(
                    label="Context Understanding",
                    visible=True,
                    elem_id="context-info",
                )
                gallery_text = gr.Gallery(
                    label="Search Results",
                    columns=4,
                    height=700,
                    object_fit="cover",
                    elem_classes=["result-gallery"],
                )

                text_btn.click(
                    search_by_text,
                    inputs=[text_input, top_k_text],
                    outputs=[gallery_text, context_display],
                )
                text_input.submit(
                    search_by_text,
                    inputs=[text_input, top_k_text],
                    outputs=[gallery_text, context_display],
                )

            # ── Image Search Tab ─────────────────────────────────────
            with gr.Tab("🖼️ Image Search", id="image-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload a fashion image",
                            height=350,
                        )
                    with gr.Column(scale=1):
                        top_k_img = gr.Slider(
                            5, 50, value=20, step=1, label="Results"
                        )
                        image_btn = gr.Button(
                            "🔍 Find Similar",
                            variant="primary",
                            elem_classes=["search-btn"],
                        )
                        gr.Markdown(
                            "Upload any fashion photo and we'll find "
                            "visually similar items in our catalog."
                        )

                gallery_img = gr.Gallery(
                    label="Similar Items",
                    columns=4,
                    height=700,
                    object_fit="cover",
                    elem_classes=["result-gallery"],
                )

                image_btn.click(
                    search_by_image,
                    inputs=[image_input, top_k_img],
                    outputs=[gallery_img],
                )

            # ── Hidden API Endpoints ─────────────────────────────────
            count_btn = gr.Button("Get Count", visible=False)
            count_text = gr.Textbox(visible=False)
            
            def get_db_count():
                return str(len(engine.metadata)) if engine else "0"
                
            count_btn.click(get_db_count, inputs=[], outputs=[count_text], api_name="get_count")

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; padding: 1.5rem; color: rgba(255,255,255,0.3); font-size: 0.8rem;">
                Built with Gemini Embedding 2 · FAISS · Gradio
                <br>
                Contextual search: understands style, occasion, price, and exclusions
            </div>
            """
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StyleGraph demo.")
    parser.add_argument("--index-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(index_dir=args.index_dir or None, device=args.device)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
