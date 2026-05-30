"""
Gemini Embedding 2 wrapper — embeds images and text into a shared vector space.

Usage:
    from stylegraph.model.gemini_embed import GeminiEmbedder
    embedder = GeminiEmbedder()
    vec = embedder.embed_text("red floral dress")
    vec = embedder.embed_image(Path("image.jpg"))
"""

import io
import os
import time
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from google import genai
from google.genai import types

from stylegraph.config import (
    GEMINI_EMBED_MODEL,
    GEMINI_EMBED_DIM,
    GEMINI_EMBED_TASK_DOC,
    GEMINI_EMBED_TASK_QUERY,
    GEMINI_BATCH_DELAY,
    EMBED_CACHE_DIR,
)

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Wraps the Gemini Embedding API with rate limiting and caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_EMBED_MODEL,
        dim: int = GEMINI_EMBED_DIM,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com"
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.dim = dim
        self.use_cache = use_cache
        self.cache_dir = cache_dir or EMBED_CACHE_DIR
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0

    # ── Rate Limiting ────────────────────────────────────────────────

    def _throttle(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_call
        if elapsed < GEMINI_BATCH_DELAY:
            time.sleep(GEMINI_BATCH_DELAY - elapsed)
        self._last_call = time.time()

    # ── Caching ──────────────────────────────────────────────────────

    def _cache_key(self, content_id: str) -> str:
        return hashlib.sha256(content_id.encode()).hexdigest()

    def _load_cached(self, content_id: str) -> Optional[np.ndarray]:
        if not self.use_cache:
            return None
        key = self._cache_key(content_id)
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cache(self, content_id: str, vec: np.ndarray):
        if not self.use_cache:
            return
        key = self._cache_key(content_id)
        cache_path = self.cache_dir / f"{key}.npy"
        np.save(cache_path, vec)

    # ── Core Embedding Methods ───────────────────────────────────────

    def embed_text(
        self, text: str, task_type: str = GEMINI_EMBED_TASK_QUERY
    ) -> np.ndarray:
        """Embed a text string → 768-d float32 vector."""
        cached = self._load_cached(f"text:{text}")
        if cached is not None:
            return cached

        self._throttle()
        result = self._call_api_with_retry(
            contents=text,
            task_type=task_type,
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        self._save_cache(f"text:{text}", vec)
        return vec

    def embed_image(
        self, image_path: Path, task_type: str = GEMINI_EMBED_TASK_DOC
    ) -> np.ndarray:
        """Embed an image file → 768-d float32 vector."""
        image_path = Path(image_path)
        cached = self._load_cached(f"img:{image_path}")
        if cached is not None:
            return cached

        self._throttle()
        image_bytes = self._prepare_image(image_path)
        mime = self._guess_mime(image_path)

        result = self._call_api_with_retry(
            contents=types.Content(
                parts=[types.Part.from_bytes(data=image_bytes, mime_type=mime)]
            ),
            task_type=task_type,
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        self._save_cache(f"img:{image_path}", vec)
        return vec

    def embed_image_pil(
        self, image: Image.Image, task_type: str = GEMINI_EMBED_TASK_DOC
    ) -> np.ndarray:
        """Embed a PIL Image → 768-d float32 vector (no caching)."""
        self._throttle()
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()

        result = self._call_api_with_retry(
            contents=types.Content(
                parts=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
            ),
            task_type=task_type,
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    # ── Batch Methods ────────────────────────────────────────────────

    def embed_images_batch(
        self,
        image_paths: list[Path],
        task_type: str = GEMINI_EMBED_TASK_DOC,
        progress_callback=None,
    ) -> np.ndarray:
        """Embed a list of images with rate limiting. Returns (N, dim) array."""
        vectors = []
        for i, path in enumerate(image_paths):
            vec = self.embed_image(path, task_type=task_type)
            vectors.append(vec)
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        return np.vstack(vectors)

    # ── Internal Helpers ─────────────────────────────────────────────

    def _call_api_with_retry(self, contents, task_type: str, max_retries: int = 5):
        """Call the embedding API with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                return self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dim,
                    ),
                )
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}), "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Embedding API error: {e}")
                    raise
        raise RuntimeError(f"Failed after {max_retries} retries")

    @staticmethod
    def _prepare_image(image_path: Path, max_size: int = 1024) -> bytes:
        """Load and resize image to reduce API payload size."""
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @staticmethod
    def _guess_mime(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
