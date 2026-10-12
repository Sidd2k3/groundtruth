"""Sentence-level verification with an NLI cross-encoder.

For each answer sentence (hypothesis) we score it against short windows of the retrieved
passages (premises): every single sentence and every pair of adjacent sentences. NLI models are
trained on sentence-sized premises, so these give much sharper entailment scores than whole
900-char chunks (extra clauses in a long premise push the model towards "neutral"), while the
pairs still cover claims that span a sentence boundary.

  supported     max P(entail)  >= entail_threshold
  contradicted  not supported and P(contradict) >= contradict_threshold on a cited/relevant window
  unsupported   everything else
"""
from dataclasses import asdict, dataclass, field

from .config import get_settings
from .models import nli_probs
from .text import extract_citations, split_answer_sentences, split_sentences

WINDOW_SIZES = (1, 2)


@dataclass
class SentenceVerdict:
    text: str
    label: str                     # supported | unsupported | contradicted | skipped
    entailment: float
    contradiction: float
    cited: list[int] = field(default_factory=list)       # passage numbers the LLM cited (1-based)
    evidence_passage: int | None = None                 # passage number the verifier found support in
    evidence: str | None = None                         # the supporting / contradicting window
    citation_ok: bool | None = None                     # did the LLM cite the passage that actually supports it?

    def to_dict(self):
        return asdict(self)


def _windows(passage: str) -> list[str]:
    sents = split_sentences(passage)
    out = []
    for size in WINDOW_SIZES:
        out += [" ".join(sents[i : i + size]) for i in range(0, len(sents) - size + 1)]
    return out or [passage]


def _is_connective(sentence: str) -> bool:
    """Non-claim sentences like 'The policy states the following:'."""
    return sentence.rstrip().endswith(":") or len(sentence.split()) < 3


def verify_sentences(sentences: list[str], passages: list[str]) -> list[SentenceVerdict]:
    s = get_settings()
    windows = [(pi, w) for pi, p in enumerate(passages) for w in _windows(p)]
    parsed = [extract_citations(x) for x in sentences]
    claims = [(i, clean) for i, (clean, _) in enumerate(parsed) if clean and not _is_connective(clean)]

    pairs = [(w, clean) for _, clean in claims for _, w in windows]
    ent, con = nli_probs(pairs)

    verdicts: list[SentenceVerdict] = []
    claim_rows = {i: row for row, (i, _) in enumerate(claims)}
    nw = len(windows)
    for i, (clean, cited) in enumerate(parsed):
        if i not in claim_rows:
            verdicts.append(SentenceVerdict(clean or sentences[i], "skipped", 0.0, 0.0, cited))
            continue
        row = claim_rows[i]
        e = ent[row * nw : (row + 1) * nw]
        c = con[row * nw : (row + 1) * nw]
        best = int(e.argmax()) if nw else None

        # Only trust contradiction signals from passages the model cited (or all, if it cited none):
        # NLI models over-predict contradiction on unrelated text.
        cited_idx = {n - 1 for n in cited if 0 < n <= len(passages)}
        relevant = [j for j, (pi, _) in enumerate(windows) if not cited_idx or pi in cited_idx]
        worst = max(relevant, key=lambda j: c[j]) if relevant else None

        max_e = float(e[best]) if best is not None else 0.0
        max_c = float(c[worst]) if worst is not None else 0.0
        if max_e >= s.entail_threshold:
            label, ev = "supported", best
        elif max_c >= s.contradict_threshold:
            label, ev = "contradicted", worst
        else:
            label, ev = "unsupported", best

        ev_passage = windows[ev][0] + 1 if ev is not None else None
        verdicts.append(
            SentenceVerdict(
                text=clean,
                label=label,
                entailment=round(max_e, 4),
                contradiction=round(max_c, 4),
                cited=cited,
                evidence_passage=ev_passage,
                evidence=windows[ev][1] if ev is not None else None,
                citation_ok=(ev_passage in cited) if label == "supported" and cited else None,
            )
        )
    return verdicts


def verify_answer(answer: str, passages: list[str]) -> list[SentenceVerdict]:
    return verify_sentences(split_answer_sentences(answer), passages)


def groundedness(verdicts: list[SentenceVerdict]) -> float:
    scored = [v for v in verdicts if v.label != "skipped"]
    if not scored:
        return 0.0
    return sum(v.label == "supported" for v in scored) / len(scored)
