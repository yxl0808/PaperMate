import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import CHROMA_PATH, CHROMA_COLLECTION, RAG_TOP_K, RAG_DISTANCE_THRESHOLD
from core.embedder import embed_texts, embed_query
from core.error_handler import DatabaseError


_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def check_chroma_integrity() -> bool:
    """Check if ChromaDB is accessible on startup."""
    try:
        _get_collection()
        return True
    except Exception:
        return False


def add_chunks(
    texts: list[str],
    paper_id: str,
    metadatas: list[dict] = None,
    ids: list[str] = None
) -> list[str]:
    """Embed texts and add to ChromaDB. Returns list of chunk IDs."""
    if not texts:
        return []
    if ids is None:
        ids = [f"{paper_id}_chunk_{i}" for i in range(len(texts))]
    if metadatas is None:
        metadatas = [{"paper_id": paper_id} for _ in range(len(texts))]
    else:
        for m in metadatas:
            m["paper_id"] = paper_id

    embeddings = embed_texts(texts)
    collection = _get_collection()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return ids


def search(
    query: str,
    paper_id: str = None,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """Search ChromaDB. If paper_id is given, filter to that paper.
    Returns list of dicts: {id, document, metadata, distance}
    """
    query_embedding = embed_query(query)
    collection = _get_collection()
    where = {"paper_id": paper_id} if paper_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return chunks


def search_with_quality_check(query: str, paper_id: str = None, top_k: int = RAG_TOP_K) -> tuple[list[dict], bool]:
    """Search and check quality. Returns (chunks, is_low_quality)."""
    chunks = search(query, paper_id=paper_id, top_k=top_k)
    is_low_quality = False
    if not chunks or chunks[0]["distance"] > RAG_DISTANCE_THRESHOLD:
        is_low_quality = True
    return chunks, is_low_quality


def delete_paper_chunks(paper_id: str):
    """Delete all chunks for a given paper from ChromaDB."""
    collection = _get_collection()
    try:
        collection.delete(where={"paper_id": paper_id})
    except Exception:
        pass


def is_chroma_empty() -> bool:
    """Check if ChromaDB has any data."""
    collection = _get_collection()
    return collection.count() == 0
