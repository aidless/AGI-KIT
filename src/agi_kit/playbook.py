"""L2 绛栫暐搴?- persistent strategy memory with semantic retrieval.

Each strategy is a small chunk of tactical knowledge learned from past tasks.
The Playbook uses BGE embeddings + FAISS for semantic search, JSONL for persistence.

Usage:
    pb = Playbook()
    pb.add(pattern="arithmetic question", strategy="use calculator tool",
           success_rate=0.95)
    matches = pb.search("compute 2**10", k=3)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agi_kit.observability import get_logger

log = get_logger(__name__)


# ============================================================
# Data structures
# ============================================================
@dataclass
class Strategy:
    sid: str
    pattern: str
    strategy: str
    success_rate: float = 0.5
    uses: int = 0
    examples: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self):
        return asdict(self)


# ============================================================
# Playbook
# ============================================================
class Playbook:
    """Strategy library with BGE + FAISS semantic retrieval.

    Persistence: data/playbook.jsonl
    Backed by FAISS index in data/playbook.faiss
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DIM = 384

    def __init__(self, path="data/playbook.jsonl",
                 index_path="data/playbook.faiss",
                 embedder=None,
                 model_name=None,
                 auto_load=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.strategies = []
        self._embedder = embedder
        self._model_name = model_name or self.DEFAULT_MODEL
        self._index = None
        self._dirty = False
        if auto_load:
            self._load()

    # ---- embedder (lazy) ----
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            log.info("playbook_load_embedder", model=self._model_name)
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    # ---- persistence ----
    def _load(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.strategies.append(Strategy(**json.loads(line)))
                except Exception:
                    pass

    def save(self):
        with self.path.open("w", encoding="utf-8") as f:
            for s in self.strategies:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        self._dirty = False
        log.info("playbook_saved", count=len(self.strategies))

    # ---- core ops ----
    def add(self, pattern, strategy, success_rate=0.5, examples=None):
        sid = uuid.uuid4().hex[:10]
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        s = Strategy(
            sid=sid,
            pattern=pattern[:500],
            strategy=strategy[:500],
            success_rate=float(success_rate),
            uses=0,
            examples=list(examples or [])[:5],
            created_at=ts,
            updated_at=ts,
        )
        self.strategies.append(s)
        self._dirty = True
        return s

    def update_success(self, sid, success_delta=0.1):
        for s in self.strategies:
            if s.sid == sid:
                s.success_rate = max(0.0, min(1.0, s.success_rate + success_delta))
                s.uses += 1
                s.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._dirty = True
                return s
        return None

    def warmup(self):
        _ = self.embedder
        if self.strategies:
            self._rebuild_index()

    def search(self, query, k=3, min_score=0.0):
        if not self.strategies or not query:
            return []
        # Lazy build index
        if self._index is None or self._dirty:
            self._rebuild_index()
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        import numpy as np
        scores, idxs = self._index.search(q_emb.astype("float32"), min(k, len(self.strategies)))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.strategies):
                continue
            if score < min_score:
                continue
            s = self.strategies[idx]
            out.append((s, float(score)))
        return out

    def _rebuild_index(self):
        try:
            import faiss
            import numpy as np
        except ImportError:
            log.warning("faiss_missing")
            return
        if not self.strategies:
            return
        texts = [s.pattern + " || " + s.strategy for s in self.strategies]
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        self._index = faiss.IndexFlatIP(self.DIM)
        self._index.add(embs.astype("float32"))
        self._dirty = False

    # ---- context rendering ----
    def as_system_context(self, query, k=3):
        hits = self.search(query, k=k)
        if not hits:
            return ""
        lines = ["Relevant strategies from past experience:"]
        for s, score in hits:
            rate_pct = int(s.success_rate * 100)
            ex = ""
            if s.examples:
                ex = " Example: " + str(s.examples[0])[:80]
            lines.append("- [match=" + str(round(score, 2)) +
                         ", success=" + str(rate_pct) + "%] " +
                         s.strategy + ex)
        return "\n".join(lines)

    def size(self):
        return len(self.strategies)