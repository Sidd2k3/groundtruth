from groundtruth.ingest import chunk_file
from groundtruth.text import extract_citations, split_answer_sentences, split_sentences


def test_split_sentences_keeps_abbreviations_and_decimals():
    text = "Mileage is paid at 0.30 euros per km, e.g. for site visits. Taxis are fine after 22:00. Ask Dr. Smith."
    assert split_sentences(text) == [
        "Mileage is paid at 0.30 euros per km, e.g. for site visits.",
        "Taxis are fine after 22:00.",
        "Ask Dr. Smith.",
    ]


def test_answer_split_handles_bullets():
    ans = "You get:\n- 22 days of leave [1].\n- 10 sick days [2]."
    assert split_answer_sentences(ans) == ["You get:", "22 days of leave [1].", "10 sick days [2]."]


def test_extract_citations():
    assert extract_citations("Leave is 22 days [1, 3].") == ("Leave is 22 days.", [1, 3])
    assert extract_citations("No citation.") == ("No citation.", [])


def test_markdown_chunking_respects_sections_and_sentences():
    md = b"# Policy\n\n## Leave\n\nYou get 22 days. Carry over 5.\n\n## Sick\n\nTen sick days."
    chunks = chunk_file("p.md", md)
    assert [c.section for c in chunks] == ["Policy > Leave", "Policy > Sick"]
    assert chunks[0].text == "Policy > Leave: You get 22 days. Carry over 5."
    assert len({c.id for c in chunks}) == 2


def test_long_section_is_split_with_overlap(monkeypatch):
    from groundtruth import config

    monkeypatch.setattr(config.get_settings(), "chunk_chars", 60)
    body = " ".join(f"Sentence number {i} is here." for i in range(10))
    chunks = chunk_file("long.txt", body.encode())
    assert len(chunks) > 2
    # one sentence of overlap between consecutive chunks
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text
