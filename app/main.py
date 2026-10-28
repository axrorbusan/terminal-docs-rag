"""REST API + a one-page web UI.

Run:  uvicorn app.main:app --port 8000      then open http://localhost:8000
API docs (Swagger) are generated at http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import config
from rag.answer import RAG
from rag.retrieve import search

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = RAG()  # loads the embedding model, Chroma and BM25 once
    yield
    state.clear()


app = FastAPI(
    title="Terminal Docs RAG",
    description="Ask questions about SMDG container-terminal EDI message guides; "
                "answers cite the document and page.",
    version="1.0.0",
    lifespan=lifespan,
)

Mode = Literal["hybrid", "dense", "sparse"]
Rewrite = Literal["auto", "always", "none", "fuse"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["VERMAS에서 VGM은 어느 세그먼트에 들어가나요?"])
    k: int = Field(config.TOP_K, ge=1, le=20)
    mode: Mode = "hybrid"
    rewrite: Rewrite = "auto"


class Source(BaseModel):
    n: int
    doc: str
    title: str
    page: int
    cited: bool
    text: str


class AskResponse(BaseModel):
    question: str
    search_query: str
    keywords: str
    answer: str
    found: bool
    sources: list[Source]
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


@app.get("/health")
def health():
    rag = state.get("rag")
    return {"status": "ok" if rag else "loading",
            "chunks": len(rag.index.chunks) if rag else 0,
            "answer_model": config.ANSWER_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return state["rag"].ask(req.question, k=req.k, mode=req.mode, rewrite=req.rewrite)
    except Exception as e:  # e.g. missing API key, rate limit
        raise HTTPException(status_code=502, detail=f"LLM call failed: {type(e).__name__}: {e}")


@app.get("/search", response_model=list[Source])
def search_only(q: str, k: int = config.TOP_K, mode: Mode = "hybrid"):
    """Retrieval without the LLM (no API key needed): useful for debugging."""
    chunks = search(state["rag"].index, q, k=k, mode=mode)
    return [{"n": n, "doc": c["doc"], "title": c["title"], "page": c["page"],
             "cited": False, "text": c["text"]} for n, c in enumerate(chunks, 1)]


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
