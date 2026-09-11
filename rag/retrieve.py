"""Vector search over the indexed PMC chunks."""

from pathlib import Path

import chromadb
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "lifting_lit"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    return _collection


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks for `query`, each with a similarity `score`
    (1.0 = identical direction, 0.0 = unrelated, per the collection's cosine space)."""
    collection = _get_collection()
    query_embedding = ollama.embed(model=EMBED_MODEL, input=query).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": text, "score": 1 - distance, **metadata})
    return hits


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is the effect of training volume on hypertrophy?"
    for i, hit in enumerate(search(query, k=5), 1):
        print(f"[{i}] score={hit['score']:.3f} {hit['pmcid']} ({hit['section']}) - {hit['title'][:80]}")
        print(f"    {hit['text'][:200]}...")
        print()
