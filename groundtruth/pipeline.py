"""End-to-end question answering: retrieve -> gate -> generate -> verify -> decide."""
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .config import get_settings
from .llm import REFUSAL_TOKEN, generate_answer
from .retrieval import Hit, Mode, retrieve
from .store import Store
from .verify import SentenceVerdict, groundedness, verify_answer

REFUSAL_MESSAGE = "I can't answer that from the uploaded documents."


@dataclass
class Source:
    n: int
    doc_name: str
    section: str
    text: str
    rerank_score: float | None


@dataclass
class Answer:
    question: str
    status: str                    # grounded | warning | refused
    answer: str                    # what the user is shown
    raw_answer: str                # the LLM's unfiltered output ("" if never generated)
    groundedness: float
    refusal_reason: str | None = None   # retrieval_gate | model_declined | low_groundedness
    sentences: list[SentenceVerdict] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _sources(hits: list[Hit]) -> list[Source]:
    return [
        Source(i + 1, h.chunk.doc_name, h.chunk.section, h.chunk.text, h.rerank_score) for i, h in enumerate(hits)
    ]


def _cite(sentence: str, n: int | None) -> str:
    body, end = (sentence[:-1], sentence[-1]) if sentence[-1:] in ".!?" else (sentence, ".")
    return f"{body} [{n}]{end}"


def decide(question: str, raw: str, verdicts: list[SentenceVerdict], sources: list[Source], verify: bool) -> Answer:
    """Apply the groundedness policy to a generated answer."""
    s = get_settings()
    g = groundedness(verdicts)
    if REFUSAL_TOKEN in raw:
        return Answer(question, "refused", REFUSAL_MESSAGE, raw, 0.0, "model_declined", [], sources)
    if not verify:
        return Answer(question, "grounded", raw, raw, g, None, verdicts, sources)
    if g < s.groundedness_refuse:
        return Answer(question, "refused", REFUSAL_MESSAGE, raw, g, "low_groundedness", verdicts, sources)
    # Show only sentences the verifier could back up; drop unsupported / contradicted ones.
    kept = [v for v in verdicts if v.label == "supported"]
    shown = " ".join(_cite(v.text, v.evidence_passage) for v in kept)
    status = "grounded" if g >= s.groundedness_show else "warning"
    return Answer(question, status, shown, raw, g, None, verdicts, sources)


def answer_question(
    store: Store,
    question: str,
    verify: bool = True,
    mode: Mode = Mode.hybrid_rerank,
    generate: Callable[[str, list[str]], str] = generate_answer,
) -> Answer:
    s = get_settings()
    t = {}
    t0 = time.perf_counter()
    hits = retrieve(store, question, mode)
    t["retrieve_s"] = time.perf_counter() - t0
    sources = _sources(hits)

    top = hits[0].rerank_score if hits else None
    if not hits or (verify and top is not None and top < s.min_rerank_score):
        return Answer(question, "refused", REFUSAL_MESSAGE, "", 0.0, "retrieval_gate", [], sources, t)

    t0 = time.perf_counter()
    passages = [h.chunk.text for h in hits]
    raw = generate(question, passages)
    t["generate_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verdicts = verify_answer(raw, passages) if REFUSAL_TOKEN not in raw else []
    t["verify_s"] = time.perf_counter() - t0

    ans = decide(question, raw, verdicts, sources, verify)
    ans.timings = {k: round(v, 3) for k, v in t.items()}
    return ans
