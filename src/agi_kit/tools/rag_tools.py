"""RAG tools exposed to agents."""
from __future__ import annotations

from functools import lru_cache

from agi_kit.config import get_settings
from agi_kit.tools.base import tool


@lru_cache(maxsize=1)
def _get_store():
    from agi_kit.rag.store import RagStore
    s = get_settings()
    return RagStore(
        model_name=s.rag.embedder,
        index_path=s.rag.index_path,
        chunk_size=s.rag.chunk_size,
        chunk_overlap=s.rag.chunk_overlap,
    )


@tool(
    "rag_add",
    "Index a local file (txt/md/pdf) into the RAG store",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    tags=["rag"],
)
def rag_add(path: str) -> str:
    try:
        n = _get_store().add_file(path)
        return f"indexed {n} chunks"
    except Exception as e:
        return f"error: {e}"


@tool(
    "rag_search",
    "Semantic search over indexed documents",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    },
    tags=["rag"],
)
def rag_search(query: str, top_k: int = 3) -> str:
    try:
        results = _get_store().search(query, top_k=top_k)
        if not results:
            return "(no matches)"
        return "\n\n---\n\n".join(
            r["text"][:800] + f"\n[src: {r['source']}, score: {r['score']:.2f}]" for r in results
        )
    except Exception as e:
        return f"error: {e}"


@tool(
    "rag_clear",
    "Clear the entire RAG index",
    parameters={"type": "object", "properties": {}},
    tags=["rag"],
)
def rag_clear() -> str:
    try:
        _get_store().clear()
        return "cleared"
    except Exception as e:
        return f"error: {e}"