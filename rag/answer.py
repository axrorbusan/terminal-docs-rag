"""Question -> (rewrite) -> retrieve -> Claude answer with numbered citations."""
import json
import re
import time

import anthropic

from . import config
from .index import Index
from .llm import create, text_of
from .retrieve import rrf, search

REWRITE_PROMPT = """You prepare a search query for a document search over SMDG / UN/EDIFACT \
container-terminal message guides. The documents are in English.

Question (any language): {question}

Rules:
- query_en: translate the question into English, keeping its meaning and wording. Do not add \
topics, message names or words that are not in the question.
- keywords: only EDIFACT tags, codes or technical terms that the question itself names or \
clearly refers to (e.g. "VGM" for "verified gross mass"). Do not guess other message names, \
and do not expand abbreviations you are not sure of. Leave empty if unsure.

Return only JSON: {{"query_en": "...", "keywords": "..."}}"""

ANSWER_SYSTEM = """You answer questions about container-terminal EDI messages using ONLY the \
numbered excerpts from SMDG message implementation guides given to you.

Rules:
- Every factual sentence must end with the excerpt number(s) it comes from, like [2] or [1][3].
- If the excerpts do not contain the answer, say so plainly (in the user's language) and do \
not guess. A partial answer is fine if you say what is missing.
- Answer in the same language as the question (Korean question -> Korean answer). Keep EDIFACT \
tags, codes and message names as written (e.g. EQD, MEA, VGM).
- Be concise: a short paragraph or a few bullet points."""

def is_english(text):
    """True when the question is (almost) all ASCII letters, i.e. no Hangul etc."""
    letters = [ch for ch in text if ch.isalpha()]
    return not letters or sum(ch.isascii() for ch in letters) / len(letters) > 0.9


NOT_FOUND_MARKERS = ("not contain", "do not contain", "doesn't contain", "no information",
                     "찾을 수 없", "포함되어 있지 않", "나와 있지 않", "없습니다")


class RAG:
    def __init__(self, index: Index | None = None, client=None):
        self.index = index or Index()
        self.client = client or anthropic.Anthropic()

    # -- query rewriting ---------------------------------------------------------
    def rewrite(self, question):
        msg = create(
            self.client, config.REWRITE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
        )
        text = text_of(msg)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        return (
            data.get("query_en") or question,
            data.get("keywords") or "",
            msg.usage,
        )

    def retrieve(self, question, k=config.TOP_K, mode="hybrid", rewrite="auto"):
        """Retrieve chunks; returns (chunks, search_query, keywords, usage list).

        rewrite policy:
          "none"   search with the question as typed
          "always" search with the LLM-rewritten English query
          "auto"   rewrite only questions that are not in English (the documents are English)
          "fuse"   search with both and merge the two rankings (reciprocal rank fusion)
        """
        usage = []
        query, keywords = question, ""
        if rewrite == "always" or (rewrite == "auto" and not is_english(question)) or rewrite == "fuse":
            query, keywords, u = self.rewrite(question)
            usage.append(u)
        if rewrite == "fuse":
            pool = max(k * 5, 30)
            a = [c["id"] for c in search(self.index, question, k=pool, mode=mode)]
            b = [c["id"] for c in search(self.index, query, k=pool, mode=mode, keywords=keywords)]
            chunks = [self.index.by_id[i] for i in rrf([a, b])[:k]]
        else:
            chunks = search(self.index, query, k=k, mode=mode, keywords=keywords)
        return chunks, query, keywords, usage

    # -- full pipeline -------------------------------------------------------------
    def ask(self, question, k=config.TOP_K, mode="hybrid", rewrite="auto",
            model=config.ANSWER_MODEL):
        t0 = time.time()
        chunks, query, keywords, usage = self.retrieve(question, k=k, mode=mode, rewrite=rewrite)

        context = "\n\n".join(
            f"[{n}] {c['title']}, page {c['page']}\n{c['text']}" for n, c in enumerate(chunks, 1)
        )
        msg = create(
            self.client, model,
            max_tokens=config.MAX_ANSWER_TOKENS,
            system=ANSWER_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}",
            }],
        )
        usage.append(msg.usage)
        answer = text_of(msg).strip()

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(chunks)})
        return {
            "question": question,
            "search_query": query,
            "keywords": keywords,
            "answer": answer,
            "found": not any(s in answer.lower() for s in NOT_FOUND_MARKERS) or bool(cited),
            "sources": [
                {"n": n, "doc": c["doc"], "title": c["title"], "page": c["page"],
                 "cited": n in cited, "text": c["text"]}
                for n, c in enumerate(chunks, 1)
            ],
            "model": model,
            "input_tokens": sum(u.input_tokens for u in usage),
            "output_tokens": sum(u.output_tokens for u in usage),
            "seconds": round(time.time() - t0, 2),
        }
