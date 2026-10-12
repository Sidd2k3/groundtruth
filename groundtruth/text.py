"""Small text utilities: sentence splitting, tokenisation for BM25, normalisation."""
import re

_ABBREV = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp",
    "no", "fig", "sec", "art", "para", "approx", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "st", "u.s", "u.k",
}
_CITATION = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that avoids breaking on common abbreviations and decimals."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces, start = [], 0
    for m in _SPLIT.finditer(text):
        candidate = text[start:m.start()].rstrip()
        last_word = candidate.rsplit(" ", 1)[-1].rstrip(".").lower()
        if last_word in _ABBREV or re.fullmatch(r"[a-z]", last_word):
            continue  # "e.g. Foo", "Section A. Bar" style false splits
        pieces.append(text[start:m.end()].strip())
        start = m.end()
    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def split_answer_sentences(answer: str) -> list[str]:
    """Split an LLM answer into sentences, treating bullet/numbered list items as sentences too."""
    out = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if line:
            out.extend(split_sentences(line))
    return out


def extract_citations(sentence: str) -> tuple[str, list[int]]:
    """Return (sentence without [n] markers, cited passage numbers)."""
    cited = [int(n) for m in _CITATION.finditer(sentence) for n in m.group(1).split(",")]
    clean = _CITATION.sub("", sentence).strip()
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    return clean, cited


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for with by at from is are was were be been being it its this that "
    "these those as if then than so such can could may might must shall should will would do does did "
    "not no nor i you he she we they them their our your what which who whom how when where why".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
