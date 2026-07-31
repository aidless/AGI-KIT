"""Local RAG store: embedder + FAISS index."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agi_kit.exceptions import RAGError
from agi_kit.observability import get_logger, metrics

log = get_logger(__name__)


@dataclass
class RagHit:
    """Single search hit."""

    text: str
    source: str
    chunk_id: int
    score: float


class RagStore:
    """FAISS-backed RAG with pluggable embedder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        index_path: Path | str = "./data/rag.index",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._dim: int | None = None
        self._embedder: Any = None
        self._index: Any = None
        self._docs: list[dict[str, Any]] = []
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RAGError("sentence-transformers required. pip install sentence-transformers") from e
            log.info("loading_embedder", model=self.model_name)
            self._embedder = SentenceTransformer(self.model_name)
            self._dim = self._embedder.get_sentence_embedding_dimension()
        return self._embedder

    @property
    def index(self) -> Any:
        if self._index is None:
            try:
                import faiss
                import numpy as np
            except ImportError as e:
                raise RAGError("faiss-cpu required. pip install faiss-cpu") from e
            self._dim = self._dim or self.embedder.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(self._dim)
        return self._index

    def _chunk(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        out = []
        i = 0
        while i < len(text):
            out.append(text[i:i + self.chunk_size])
            i += self.chunk_size - self.chunk_overlap
        return [c for c in out if c.strip()]

    def add_text(self, text: str, source: str = "text") -> int:
        chunks = self._chunk(text)
        if not chunks:
            return 0
        import numpy as np
        vecs = self.embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        self.index.add(np.asarray(vecs, dtype="float32"))
        for c in chunks:
            self._docs.append({"text": c, "source": source, "chunk_id": len(self._docs)})
        self._save()
        metrics.inc(metrics.rag_chunks_indexed_total, {"source": source[:32]} if False else {})
        return len(chunks)

    def add_file(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            raise RAGError(f"file not found: {path}")
        if p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() for page in PdfReader(str(p)).pages)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        return self.add_text(text, source=p.name)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            metrics.observe(metrics.rag_search_hits, 0)
            return []
        import numpy as np
        q = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.asarray(q, dtype="float32"), min(top_k, self.index.ntotal))
        out = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self._docs):
                hit = dict(self._docs[idx])
                hit["score"] = float(score)
                out.append(hit)
        metrics.inc(metrics.rag_search_total)
        metrics.observe(metrics.rag_search_hits, len(out))
        return out

    def clear(self) -> None:
        self._index = None
        self._docs = []
        if self.index_path.exists():
            os.remove(self.index_path)
        meta = self.index_path.with_suffix(".meta.jsonl")
        if meta.exists():
            os.remove(meta)

    def _save(self) -> None:
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        import json
        with open(self.index_path.with_suffix(".meta.jsonl"), "w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        import json
        if self.index_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self.index_path))
                meta = self.index_path.with_suffix(".meta.jsonl")
                if meta.exists():
                    with open(meta, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                self._docs.append(json.loads(line))
                            except Exception:
                                pass
            except Exception as e:
                log.warning("rag_load_failed", error=str(e))