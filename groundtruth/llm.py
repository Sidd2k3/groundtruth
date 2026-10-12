"""Groq LLM client and prompts."""
import re
from functools import lru_cache

from .config import get_settings

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered source passages provided.

Rules:
- Every sentence you write must be directly supported by the passages. Do not add outside knowledge, assumptions, or advice.
- End every sentence with the citation(s) of the passage(s) that support it, e.g. "Employees get 25 days of leave [2]."
- Prefer the passages' own wording for numbers, dates, and conditions. Keep the answer concise (1-5 sentences).
- Write plain prose sentences. Do not start with filler like "According to the passages".
- If the passages do not contain the information needed to answer, reply with exactly: {REFUSAL_TOKEN}
"""


class LLMNotConfigured(RuntimeError):
    pass


@lru_cache
def _client():
    from groq import Groq

    key = get_settings().groq_api_key
    if not key:
        raise LLMNotConfigured("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
    return Groq(api_key=key, max_retries=6)


def format_passages(passages: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def _is_reasoning_model(model: str) -> bool:
    return any(tag in model.lower() for tag in ("gpt-oss", "qwen3", "deepseek-r1"))


def chat(system: str, user: str, model: str | None = None, max_tokens: int = 512) -> str:
    s = get_settings()
    model = model or s.groq_model
    kwargs = {}
    if _is_reasoning_model(model):
        # Reasoning tokens count against the completion budget, so leave room for them,
        # and keep the reasoning out of the answer text.
        max_tokens += 2048
        if "gpt-oss" in model.lower():
            kwargs["reasoning_effort"] = s.reasoning_effort
            kwargs["include_reasoning"] = False
        else:
            kwargs["reasoning_format"] = "hidden"
    resp = _client().chat.completions.create(
        model=model,
        temperature=s.llm_temperature,
        max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def generate_answer(question: str, passages: list[str]) -> str:
    user = f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}"
    return chat(SYSTEM_PROMPT, user)
