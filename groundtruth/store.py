"""Qdrant vector store + in-memory BM25 index kept in sync with it.

Qdrant is the source of truth for chunks. The BM25 index is rebuilt from Qdrant payloads
on startup and after every ingest/delete, which is cheap at the corpus sizes this project targets.
"""
import threading

from qdrant_client import QdrantClient, models as qm
from rank_bm25 import BM25Okapi

from .config import get_settings
from .ingest import Chunk
from .models import embed_passages, embed_query, embedding_dim
from .text import tokenize


class Store:
    def __init__(self):
        s = get_settings()
        if s.qdrant_url:
            self.client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
        else:
            self.client = QdrantClient(path=s.qdrant_path)
        self.collection = s.collection
        self._lock = threading.Lock()
        self._ensure_collection()
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_bm25()

    # ---------- collection management ----------
    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=embedding_dim(), distance=qm.Distance.COSINE),
            )
            if get_settings().qdrant_url:  # payload indexes are a no-op in embedded mode
                self.client.create_payload_index(self.collection, "doc_id", qm.PayloadSchemaType.KEYWORD)

    def _all_chunks(self) -> list[Chunk]:
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            out.extend(Chunk(**p.payload) for p in points)
            if offset is None:
                break
        out.sort(key=lambda c: (c.doc_name, c.chunk_index))
        return out

    def _rebuild_bm25(self):
        self.chunks = self._all_chunks()
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in self.chunks]) if self.chunks else None

    # ---------- writes ----------
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        with self._lock:
            self.delete_doc(chunks[0].doc_id, rebuild=False)  # re-uploading a file replaces it
            vectors = embed_passages([c.text for c in chunks])
            self.client.upsert(
                self.collection,
                points=[
                    qm.PointStruct(id=c.id, vector=v.tolist(), payload=c.to_payload())
                    for c, v in zip(chunks, vectors)
                ],
            )
            self._rebuild_bm25()
        return len(chunks)

    def delete_doc(self, doc_id: str, rebuild: bool = True):
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
        )
        if rebuild:
            self._rebuild_bm25()

    def close(self):
        self.client.close()

    def reset(self):
        with self._lock:
            self.client.delete_collection(self.collection)
            self._ensure_collection()
            self._rebuild_bm25()

    # ---------- reads ----------
    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "doc_name": c.doc_name, "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())

    def dense_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        res = self.client.query_points(
            self.collection, query=embed_query(query).tolist(), limit=k, with_payload=True
        )
        return [(Chunk(**p.payload), float(p.score)) for p in res.points]

    def bm25_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
