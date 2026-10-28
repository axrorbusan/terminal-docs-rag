"""Retrieval: dense (vector), sparse (BM25) and hybrid (reciprocal rank fusion)."""
import numpy as np

from . import config
from .index import Index, tokenize


def dense(index: Index, query, k):
    q = index.embedder.encode([query])[0]
    res = index.col.query(query_embeddings=[q.tolist()], n_results=k)
    return res["ids"][0]


def sparse(index: Index, query, k):
    scores = index.bm25.get_scores(tokenize(query))
    top = np.argsort(-scores)[:k]
    return [index.chunks[i]["id"] for i in top if scores[i] > 0]


def rrf(rankings, k=config.RRF_K):
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank)."""
    score = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
    return [cid for cid, _ in sorted(score.items(), key=lambda x: -x[1])]


def search(index: Index, query, k=config.TOP_K, mode="hybrid", keywords=""):
    """Return the top-k chunks for a query.

    `keywords` (from the query rewriter) is added to the BM25 query only: exact EDIFACT
    tags such as "EQD" or "VGM" are what keyword search is good at.
    """
    pool = max(k * 5, 30)
    if mode == "dense":
        ids = dense(index, query, k)
    elif mode == "sparse":
        ids = sparse(index, f"{query} {keywords}", k)
    elif mode == "hybrid":
        ids = rrf([dense(index, query, pool), sparse(index, f"{query} {keywords}", pool)])[:k]
    else:
        raise ValueError(f"unknown mode {mode}")
    return [index.by_id[i] for i in ids]
