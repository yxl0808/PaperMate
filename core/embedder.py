import time
from sentence_transformers import SentenceTransformer
from config import MAX_RETRIES, RETRY_BASE_DELAY

# Small multilingual model, works offline after first download
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts with retry. Returns list of embedding vectors."""
    embedder = _get_embedder()
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            result = embedder.encode(texts, normalize_embeddings=True)
            return result.tolist()
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {last_exception}")


def embed_query(text: str) -> list[float]:
    """Embed a single query text with retry."""
    embedder = _get_embedder()
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            result = embedder.encode(text, normalize_embeddings=True)
            return result.tolist()
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {last_exception}")
