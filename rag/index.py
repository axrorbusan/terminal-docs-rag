"""Build and load the search indexes.

- Dense: chunk embeddings stored in a persistent Chroma collection (cosine distance).
- Sparse: BM25 over the same chunks, rebuilt in memory at load time (it takes < 1 s).

Every chunk is indexed as "<document title>\n<chunk text>", so a chunk taken from the
middle of a table still says which message it belongs to.

Run:  python -m rag.index        (after python -m rag.ingest)
"""
import json
import re

import chromadb
from rank_bm25 import BM25Okapi

from . import config
from .embed import get_embedder

COLLECTION = "smdg_chunks"


def indexed_text(chunk):
    return f"{chunk['title']}\n{chunk['text']}"


def tokenize(text):
    # Lower-case words and codes: keeps EDIFACT tags ("eqd", "c401", "an") and numbers.
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks():
    with open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def chroma_client():
    return chromadb.PersistentClient(
        path=str(config.INDEX_DIR / "chroma"),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def build():
    chunks = load_chunks()
    vecs = get_embedder().encode([indexed_text(c) for c in chunks])
    client = chroma_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        part = chunks[i : i + 500]
        col.add(
            ids=[c["id"] for c in part],
            embeddings=vecs[i : i + 500].tolist(),
            documents=[c["text"] for c in part],
            metadatas=[{"doc": c["doc"], "title": c["title"], "page": c["page"]} for c in part],
        )
    print(f"indexed {col.count()} chunks into Chroma at {config.INDEX_DIR / 'chroma'}")


class Index:
    """Everything retrieval needs, loaded once at start-up."""

    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.col = chroma_client().get_collection(COLLECTION)
        self.bm25 = BM25Okapi([tokenize(indexed_text(c)) for c in self.chunks])
        self.embedder = get_embedder()


if __name__ == "__main__":
    build()
