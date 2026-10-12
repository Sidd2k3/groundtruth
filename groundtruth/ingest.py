"""Document loading and chunking.

Supported formats: .pdf, .docx, .md, .txt. Each document is split into "sections"
(PDF pages, or markdown/docx heading sections) and each section is chunked on sentence
boundaries so a chunk never cuts a sentence in half — important because the verifier
works at sentence level.
"""
import io
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import get_settings
from .text import split_sentences

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    section: str
    chunk_index: int
    text: str

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass
class Section:
    title: str
    text: str


def _load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [Section(f"page {i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _load_docx(data: bytes) -> list[Section]:
    import docx

    document = docx.Document(io.BytesIO(data))
    sections, title, buf = [], "", []
    for p in document.paragraphs:
        if p.style is not None and p.style.name.lower().startswith("heading"):
            if buf:
                sections.append(Section(title, "\n".join(buf)))
            title, buf = p.text.strip(), []
        elif p.text.strip():
            buf.append(p.text)
    if buf:
        sections.append(Section(title, "\n".join(buf)))
    return sections


def _load_markdown(text: str) -> list[Section]:
    sections, path, buf = [], [], []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                sections.append(Section(" > ".join(path), "\n".join(buf)))
                buf = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        elif line.strip():
            buf.append(line)
    if buf:
        sections.append(Section(" > ".join(path), "\n".join(buf)))
    return sections


def load_sections(filename: str, data: bytes) -> list[Section]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(data)
    if suffix == ".docx":
        return _load_docx(data)
    if suffix in {".md", ".txt"}:
        text = data.decode("utf-8", errors="replace")
        return _load_markdown(text) if suffix == ".md" else [Section("", text)]
    raise ValueError(f"Unsupported file type {suffix!r}; supported: {sorted(SUPPORTED)}")


def _paragraph_sentences(text: str) -> list[str]:
    """Sentences, keeping list items / short lines as their own units."""
    out = []
    for para in re.split(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)", text):
        para = re.sub(r"^\s*(?:[-*•])\s+", "", para.strip())
        out.extend(split_sentences(para))
    return out


def chunk_sections(doc_id: str, doc_name: str, sections: list[Section]) -> list[Chunk]:
    s = get_settings()
    chunks: list[Chunk] = []
    for section in sections:
        sents = _paragraph_sentences(section.text)
        i = 0
        while i < len(sents):
            buf, size, j = [], 0, i
            while j < len(sents) and (not buf or size + len(sents[j]) <= s.chunk_chars):
                buf.append(sents[j])
                size += len(sents[j]) + 1
                j += 1
            idx = len(chunks)
            body = " ".join(buf)
            # Prefix the section title so both BM25 and the embedder see the context.
            text = f"{section.title}: {body}" if section.title else body
            chunks.append(
                Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}")),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    section=section.title,
                    chunk_index=idx,
                    text=text,
                )
            )
            if j >= len(sents):
                break
            i = max(j - s.chunk_overlap_sentences, i + 1)
    return chunks


def doc_id_for(filename: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, Path(filename).name))


def chunk_file(filename: str, data: bytes) -> list[Chunk]:
    name = Path(filename).name
    return chunk_sections(doc_id_for(name), name, load_sections(name, data))
