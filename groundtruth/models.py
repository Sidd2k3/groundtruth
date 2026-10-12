"""Lazy, process-wide singletons for the three local models (embedder, reranker, NLI)."""
from functools import lru_cache

import numpy as np

from .config import get_settings

# bge-*-v1.5 models recommend this instruction prefix on queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model, device="cpu")


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().rerank_model, device="cpu")


@lru_cache
def nli():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(get_settings().nli_model, device="cpu")
    labels = {v.lower(): k for k, v in model.model.config.id2label.items()}
    return model, labels["entailment"], labels["contradiction"]


def embed_passages(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    prefix = BGE_QUERY_PREFIX if "bge" in get_settings().embed_model.lower() else ""
    return embedder().encode(prefix + text, normalize_embeddings=True)


def embedding_dim() -> int:
    model = embedder()
    # renamed in sentence-transformers 6; support both
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return fn()


def rerank_scores(query: str, passages: list[str]) -> np.ndarray:
    if not passages:
        return np.array([])
    return np.asarray(reranker().predict([(query, p) for p in passages], show_progress_bar=False))


def nli_probs(pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """For (premise, hypothesis) pairs return (P(entailment), P(contradiction))."""
    if not pairs:
        return np.array([]), np.array([])
    model, ent, con = nli()
    logits = np.asarray(model.predict(pairs, batch_size=32, show_progress_bar=False))
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs[:, ent], probs[:, con]
