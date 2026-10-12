"""End-to-end pipeline tests with a fake LLM (no API key needed). Downloads the local models on first run."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def store(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("QDRANT_PATH", str(tmp_path_factory.mktemp("qdrant")))
    monkeypatch_module.setenv("QDRANT_URL", "")
    monkeypatch_module.setenv("COLLECTION", "test")
    from groundtruth.config import get_settings

    get_settings.cache_clear()
    from groundtruth.ingest import chunk_file
    from groundtruth.store import Store

    s = Store()
    for f in (ROOT / "data" / "corpus").glob("*.md"):
        s.add(chunk_file(f.name, f.read_bytes()))
    return s


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def fake_llm(answer):
    return lambda question, passages: answer


def test_retrieval_modes_find_the_right_passage(store):
    from groundtruth.retrieval import Mode, retrieve

    for mode in Mode:
        hits = retrieve(store, "What is the minimum password length?", mode, k=5)
        assert any("14 characters" in h.chunk.text for h in hits), mode


def test_grounded_answer_is_shown(store):
    from groundtruth.pipeline import answer_question

    a = answer_question(store, "What is the minimum password length?",
                        generate=fake_llm("Passwords must be at least 14 characters long [1]."))
    assert a.status == "grounded"
    assert a.sentences[0].label == "supported"
    assert "14 characters" in a.answer


def test_hallucinated_sentence_is_removed(store):
    from groundtruth.pipeline import answer_question

    a = answer_question(
        store, "How many days of annual leave do full-time employees get?",
        generate=fake_llm("Full-time employees accrue 22 days of paid annual leave per calendar year [1]. "
                          "Unused annual leave is paid out in cash every December [1]."),
    )
    labels = [s.label for s in a.sentences]
    assert labels[0] == "supported" and labels[1] != "supported"
    assert "December" not in a.answer
    assert a.status in {"warning", "refused"}


def test_model_refusal_is_respected(store):
    from groundtruth.pipeline import answer_question

    a = answer_question(store, "Who is the CEO?", generate=fake_llm("INSUFFICIENT_CONTEXT"))
    assert a.status == "refused"
