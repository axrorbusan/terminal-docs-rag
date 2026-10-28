"""Unit tests that need no API key (the LLM is replaced by a fake client).

Run:  pytest -q
"""
from types import SimpleNamespace

import pytest

from rag.ingest import chunk_text
from rag.index import tokenize
from rag.retrieve import rrf


def test_chunk_text_respects_size_and_overlap():
    text = "\n".join(f"Line {i}: the EQD segment identifies the container." for i in range(60))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # overlap: the end of one chunk reappears at the start of the next
    assert chunks[0][-20:] in chunks[1]


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("MEA+AAE+VGM+KGM:21700' is the VGM example.") == [
        "MEA+AAE+VGM+KGM:21700' is the VGM example."
    ]


def test_tokenize_keeps_edifact_codes():
    assert tokenize("EQD+CN+ABCU1234567+22G1 an..3 C401") == [
        "eqd", "cn", "abcu1234567", "22g1", "an", "3", "c401"
    ]


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"           # ranked high in both lists
    assert set(fused) == {"a", "b", "c", "d"}


# --------------------------------------------------------------------------- API
class FakeClient:
    """Mimics anthropic.Anthropic().messages.create for the rewrite and answer calls."""

    def __init__(self):
        self.messages = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if "system" not in kw:  # rewrite call
            text = '{"query_en": "verified gross mass segment VERMAS", "keywords": "VERMAS MEA VGM"}'
        else:
            text = "The VGM is sent in the MEA segment, e.g. MEA+AAE+VGM+KGM:21700 [1]."
        return SimpleNamespace(content=[SimpleNamespace(text=text)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as main
    from rag.answer import RAG

    with TestClient(main.app) as c:          # runs the lifespan (loads the real index)
        main.state["rag"] = RAG(index=main.state["rag"].index, client=FakeClient())
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["chunks"] > 1000


def test_search_without_llm(client):
    r = client.get("/search", params={"q": "verified gross mass VERMAS MEA", "k": 5})
    assert r.status_code == 200
    assert any("VERMAS" in s["title"] for s in r.json())


def test_ask_returns_cited_answer(client):
    r = client.post("/ask", json={"question": "VERMAS에서 VGM은 어느 세그먼트에 들어가나요?"})
    body = r.json()
    assert r.status_code == 200
    assert body["search_query"] == "verified gross mass segment VERMAS"   # rewrite was used
    assert body["sources"][0]["cited"] is True
    assert body["found"] is True


def test_is_english():
    from rag.answer import is_english
    assert is_english("What does COARRI report?")
    assert not is_english("VERMAS에서 VGM은 어느 세그먼트에 들어가나요?")


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
