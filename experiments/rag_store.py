"""rag_store.py - Lightweight local RAG: sentence-transformers + FAISS"""
import os, hashlib
from pathlib import Path

class RagStore:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.dim = 384  # bge-small-en
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs = []  # list of {"text", "source", "chunk_id"}
        self.embedder = SentenceTransformer(model_name)
        self.index_path = Path(__file__).parent.parent / "data" / "rag.index"
        self.meta_path = Path(__file__).parent.parent / "data" / "rag.meta.jsonl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        import faiss, json
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.docs.append(json.loads(line))
                        except Exception:
                            pass

    def _save(self):
        import faiss, json
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _chunk(self, text, size=400, overlap=50):
        text = re.sub(r"\s+", " ", text).strip()
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i+size])
            i += size - overlap
        return [c for c in out if c.strip()]

    def add_text(self, text, source="text"):
        import json
        chunks = self._chunk(text)
        if not chunks:
            return 0
        vecs = self.embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        import numpy as np
        self.index.add(np.array(vecs, dtype="float32"))
        for i, c in enumerate(chunks):
            self.docs.append({"text": c, "source": source, "chunk_id": len(self.docs)})
        self._save()
        return len(chunks)

    def add_file(self, path):
        path = str(path)
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            r = PdfReader(path)
            text = "\n".join(p.extract_text() for p in r.pages)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        return self.add_text(text, source=os.path.basename(path))

    def search(self, query, top_k=3):
        if self.index.ntotal == 0:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)
        import numpy as np
        D, I = self.index.search(np.array(q, dtype="float32"), min(top_k, self.index.ntotal))
        out = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            d = dict(self.docs[idx])
            d["score"] = float(score)
            out.append(d)
        return out

    def clear(self):
        import json
        self.index.reset()
        self.docs = []
        if self.index_path.exists():
            os.remove(self.index_path)
        if self.meta_path.exists():
            os.remove(self.meta_path)

import re