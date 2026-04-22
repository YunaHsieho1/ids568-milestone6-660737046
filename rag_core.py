"""
Shared RAG index for Part 1 notebook and Part 2 agent (same chunking / FAISS / embedder).
Run from repository root so paths resolve.
"""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_corpus_texts(corpus_dir: Path) -> list[dict]:
    paths = sorted(set(corpus_dir.glob("*.txt")) | set(corpus_dir.glob("*.md")))
    docs: list[dict] = []
    for p in paths:
        # If reads hang for minutes: Desktop + iCloud "Optimize Storage" often evicts files to cloud.
        text = p.read_text(encoding="utf-8")
        docs.append({"source": p.name, "text": text.strip() + "\n"})
    if not docs:
        raise FileNotFoundError(f"No .txt or .md files found under: {corpus_dir}")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(raw_docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    rows: list[dict] = []
    for doc in raw_docs:
        parts = chunk_text(doc["text"], chunk_size, overlap)
        for i, c in enumerate(parts):
            c = c.strip()
            if not c:
                continue
            rows.append(
                {
                    "chunk_id": f"{doc['source']}::{i}",
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": c,
                }
            )
    return rows


class RAGPipeline:
    """Builds FAISS index over corpus_dir (same defaults as rag_pipeline.ipynb)."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        corpus_subdir: str = "data/corpus",
        chunk_size: int = 450,
        chunk_overlap: int = 90,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.root = Path(root).resolve() if root else Path(".").resolve()
        self.corpus_dir = self.root / corpus_subdir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model_name = embed_model_name

        raw_docs = load_corpus_texts(self.corpus_dir)
        self.chunks = build_chunks(raw_docs, chunk_size, chunk_overlap)
        self.embedder = SentenceTransformer(embed_model_name)
        texts = [c["text"] for c in self.chunks]
        # show_progress_bar=True so Jupyter does not look "frozen" for many seconds
        emb = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    def retrieve(self, query: str, k: int) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        q = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q, k)
        dt = time.perf_counter() - t0
        hits: list[dict] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            c = self.chunks[i]
            hits.append(
                {
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "score": float(score),
                    "text": c["text"],
                }
            )
        return hits, dt

    def hits_to_context_block(self, hits: list[dict], max_chars: int = 12000) -> str:
        parts: list[str] = []
        n = 0
        for j, c in enumerate(hits, start=1):
            block = f"[Passage {j} | source: {c['source']}]\n{c['text']}\n"
            if n + len(block) > max_chars:
                break
            parts.append(block)
            n += len(block)
        return "\n".join(parts).strip()
