"""Central configuration. Every value can be overridden with an env var (or .env file)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    # Groq retired llama-3.3-70b-versatile on its free tier (Aug 2026); gpt-oss-120b is its recommended replacement.
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    reasoning_effort: str = "low"   # for reasoning models (gpt-oss, qwen3): low keeps latency and token use down
    judge_model: str = "openai/gpt-oss-120b"   # eval only; a different model family reduces self-preference bias

    # --- Vector store (Qdrant) ---
    # Leave qdrant_url empty to use embedded local mode (data stored in qdrant_path).
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "./qdrant_data"
    collection: str = "groundtruth"
    seed_dir: str = ""          # if set and the store is empty at startup, index every file in this folder

    # --- Models ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    # --- Chunking ---
    chunk_chars: int = 900
    chunk_overlap_sentences: int = 1

    # --- Retrieval ---
    candidate_k: int = 20       # candidates pulled from each retriever before fusion / reranking
    final_k: int = 5            # passages given to the LLM
    rrf_k: int = 60             # reciprocal-rank-fusion constant

    # --- Verification / refusal policy ---
    entail_threshold: float = 0.5        # P(entailment) needed for a sentence to count as supported
    contradict_threshold: float = 0.5    # P(contradiction) needed to mark a sentence contradicted
    groundedness_show: float = 0.8       # >= this: answer shown as grounded
    groundedness_refuse: float = 0.5     # <  this: answer refused; in between: shown with a warning
    min_rerank_score: float = -10.0    # retrieval gate: refuse before generating if the best reranker logit is below this


@lru_cache
def get_settings() -> Settings:
    return Settings()
