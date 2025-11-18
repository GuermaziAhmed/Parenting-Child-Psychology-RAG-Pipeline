"""Build or load the Chroma vector store using saved chunks.jsonl instead of reprocessing files."""
from __future__ import annotations
from pathlib import Path
import json
import chromadb

from config import CHROMA_DIR, CHROMA_COLLECTION
from embeddings import EmbeddingsWrapper

CHUNKS_FILE = Path("chunks.jsonl")  # Adjust path if needed


def ensure_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return client.create_collection(CHROMA_COLLECTION)


def load_chunks_from_jsonl() -> list[dict]:
    """Load preprocessed chunks from JSONL file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_FILE}")

    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def build_if_needed(verbose: bool = True):
    CHROMA_DIR.mkdir(exist_ok=True, parents=True)
    collection = ensure_chroma_collection()

    if collection.count() > 0:
        if verbose:
            print(f"✓ Existing Chroma collection '{CHROMA_COLLECTION}' found with {collection.count()} entries.")
        return collection

    if verbose:
        print("\nBuilding Chroma collection from chunks.jsonl ...")

    chunks = load_chunks_from_jsonl()
    if verbose:
        print(f"  - Loaded {len(chunks)} saved chunks")

    texts = [c["content"] for c in chunks]
    ids = [f"{c['source']}-chunk-{c['chunk_id']}" for c in chunks]
    metadatas = [{"source": c["source"], "type": c["type"], "chunk_id": c["chunk_id"]} for c in chunks]

    if verbose:
        print(f"  - Generating embeddings...")

    embedder = EmbeddingsWrapper()
    embeddings = embedder.embed_documents(texts)

    BATCH_SIZE = 5000
    for i in range(0, len(texts), BATCH_SIZE):
        collection.add(
            documents=texts[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE],
        )
        if verbose:
            print(f"    ✓ Inserted batch {i // BATCH_SIZE + 1}")

    if verbose:
        print(f"\n✅ Finished building collection '{CHROMA_COLLECTION}'")
        print(f"Total chunks: {len(ids)}")

    return collection


if __name__ == "__main__":
    build_if_needed()
