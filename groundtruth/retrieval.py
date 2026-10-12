"""Retrieval strategies: BM25, dense, hybrid (reciprocal rank fusion), hybrid + cross-encoder rerank."""
from dataclasses import dataclass
from enum import Enum

from .config import get_settings
from .ingest import Chunk
from .models import rerank_scores
from .store import Store


class Mode(str, Enum):
    bm25 = "bm25"
    dense = "dense"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rerank_score: float | None = None


def rrf(rankings: list[list[Chunk]], k: int) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[chunk.id] = chunk
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: x[1], reverse=True)


def retrieve(store: Store, query: str, mode: Mode = Mode.hybrid_rerank, k: int | None = None) -> list[Hit]:
    s = get_settings()
    k = k or s.final_k
    n = max(s.candidate_k, k)
    if mode == Mode.bm25:
        return [Hit(c, sc) for c, sc in store.bm25_search(query, k)]
    if mode == Mode.dense:
        return [Hit(c, sc) for c, sc in store.dense_search(query, k)]

    fused = rrf(
        [[c for c, _ in store.bm25_search(query, n)], [c for c, _ in store.dense_search(query, n)]], s.rrf_k
    )
    if mode == Mode.hybrid:
        return [Hit(c, sc) for c, sc in fused[:k]]

    candidates = fused[:n]
    scores = rerank_scores(query, [c.text for c, _ in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
    return [Hit(c, float(fs), float(rs)) for (c, fs), rs in ranked]
