"""Paths and settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = Path(os.getenv("RAG_RAW_DIR", ROOT / "data" / "raw"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "data" / "index"))
MODEL_DIR = Path(os.getenv("RAG_MODEL_DIR", ROOT / "models" / "all-MiniLM-L6-v2"))

# Chunking: the embedding model reads at most 256 word-pieces, and EDIFACT tables
# (codes, qualifiers) split into many word-pieces, so chunks are kept short.
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", 900))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 150))

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", 6))
RRF_K = 60  # standard reciprocal-rank-fusion constant

# LLM
ANSWER_MODEL = os.getenv("RAG_ANSWER_MODEL", "claude-haiku-5-5")
REWRITE_MODEL = os.getenv("RAG_REWRITE_MODEL", "claude-haiku-5-5")
MAX_ANSWER_TOKENS = int(os.getenv("RAG_MAX_ANSWER_TOKENS", 700))
