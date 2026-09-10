"""Embed chunks with Ollama and load them into a persistent ChromaDB collection."""

import json
from pathlib import Path

import chromadb
import ollama
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "lifting_lit"
EMBED_BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Explicit cosine space: keeps the distance->score conversion in retrieve.py
    # (Phase 3) consistent regardless of Chroma's own default metric.
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="Embedding + indexing"):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = ollama.embed(model=EMBED_MODEL, input=texts).embeddings

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "pmcid": c["pmcid"],
                    "title": c["title"],
                    "section": c["section"],
                    "position": c["position"],
                    "year": c["year"] or "",
                    "journal": c["journal"],
                    "license": c["license"],
                    "url": c["url"],
                }
                for c in batch
            ],
        )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
