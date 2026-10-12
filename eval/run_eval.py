"""GroundTruth evaluation harness.

    python -m eval.run_eval retrieval      # Recall@k / MRR: bm25 vs dense vs hybrid vs hybrid+rerank   (no API key needed)
    python -m eval.run_eval verifier       # accuracy of the NLI verifier on hand-labelled claims       (no API key needed)
    python -m eval.run_eval answers        # faithfulness + refusal accuracy, with vs without verification (needs GROQ_API_KEY)
    python -m eval.run_eval all

Uses its own Qdrant collection on disk (./qdrant_eval) so it never touches the app's data.
Results are written to eval/results/ as JSON plus a Markdown report.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("QDRANT_PATH", str(ROOT / "qdrant_eval"))
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("COLLECTION", "groundtruth_eval")
sys.path.insert(0, str(ROOT))

from groundtruth.config import get_settings  # noqa: E402
from groundtruth.ingest import chunk_file  # noqa: E402
from groundtruth.llm import REFUSAL_TOKEN, chat, format_passages, generate_answer  # noqa: E402
from groundtruth.pipeline import REFUSAL_MESSAGE, Answer, _sources, decide  # noqa: E402
from groundtruth.retrieval import Mode, retrieve  # noqa: E402
from groundtruth.store import get_store  # noqa: E402
from groundtruth.text import normalize, split_answer_sentences, extract_citations  # noqa: E402
from groundtruth.verify import verify_answer, verify_sentences  # noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS = EVAL_DIR / "results"
CORPUS = ROOT / "data" / "corpus"
KS = (1, 3, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_index():
    store = get_store()
    store.reset()
    n = 0
    for f in sorted(CORPUS.iterdir()):
        if f.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
            n += store.add(chunk_file(f.name, f.read_bytes()))
    print(f"Indexed {n} chunks from {CORPUS}")
    return store


def contains(chunk_text: str, snippet: str) -> bool:
    return normalize(snippet) in normalize(chunk_text)


# ----------------------------------------------------------------- retrieval
def eval_retrieval(store, dataset):
    rows = [d for d in dataset if d["answerable"]]
    results = {}
    for mode in Mode:
        recalls = {k: [] for k in KS}
        rr = []
        t0 = time.perf_counter()
        for d in rows:
            hits = retrieve(store, d["question"], mode, k=max(KS))
            texts = [h.chunk.text for h in hits]
            for k in KS:
                found = sum(any(contains(t, ev) for t in texts[:k]) for ev in d["evidence"])
                recalls[k].append(found / len(d["evidence"]))
            first = next((i for i, t in enumerate(texts) if any(contains(t, ev) for ev in d["evidence"])), None)
            rr.append(0.0 if first is None else 1.0 / (first + 1))
        results[mode.value] = {
            **{f"recall@{k}": sum(v) / len(v) for k, v in recalls.items()},
            "mrr": sum(rr) / len(rr),
            "ms_per_query": 1000 * (time.perf_counter() - t0) / len(rows),
        }
        print(f"  {mode.value:14s} " + "  ".join(f"{m}={v:.3f}" for m, v in results[mode.value].items()))
    return results


# ------------------------------------------------------------------ verifier
def eval_verifier(path: Path):
    """Score the NLI verifier on hand-labelled (passage, claim, label) triples."""
    items = load_jsonl(path)
    labels = ["supported", "unsupported", "contradicted"]
    confusion = {g: {p: 0 for p in labels} for g in labels}
    errors = []
    for it in items:
        v = verify_sentences([it["claim"]], [it["passage"]])[0]
        confusion[it["label"]][v.label] += 1
        if v.label != it["label"]:
            errors.append({"claim": it["claim"], "gold": it["label"], "pred": v.label,
                           "entail": v.entailment, "contradict": v.contradiction})
    per_label = {}
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in labels if g != lab)
        fn = sum(confusion[lab][p] for p in labels if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    acc = sum(confusion[l][l] for l in labels) / len(items)
    # The number that matters most for safety: how often does a NOT-supported claim slip through as supported?
    bad = [it for it in items if it["label"] != "supported"]
    leak = sum(confusion[g]["supported"] for g in ("unsupported", "contradicted")) / len(bad) if bad else 0.0
    print(f"  accuracy={acc:.3f}  unsupported-claims-accepted={leak:.3f}  n={len(items)}")
    return {"n": len(items), "accuracy": acc, "false_accept_rate": leak, "per_label": per_label,
            "confusion": confusion, "errors": errors}


# ------------------------------------------------------- answers (needs Groq)
class GenCache:
    """Disk cache so re-running the eval doesn't burn Groq free-tier tokens."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key, fn):
        if key not in self.data:
            self.data[key] = fn()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1))
        return self.data[key]


JUDGE_SYSTEM = """You are a strict fact-checking judge. You receive numbered source passages, a question, a reference answer (may be null) and a list of numbered answer sentences.

For EACH answer sentence decide, using ONLY the passages:
- "supported": every factual claim in the sentence is stated in or directly implied by the passages
- "contradicted": the passages state something incompatible with the sentence
- "unsupported": anything else (not mentioned, over-generalised, extra details, outside knowledge)

Also decide "correct": true if the answer sentences together convey the reference answer's key facts (false if the reference is null).

Return ONLY JSON: {"labels": ["supported" | "unsupported" | "contradicted", ...], "correct": true | false}"""


def judge(cache, question, reference, sentences, passages):
    if not sentences:
        return {"labels": [], "correct": False}
    payload = (
        f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}\nReference answer: {json.dumps(reference)}\n\n"
        "Answer sentences:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    )
    s = get_settings()
    key = "judge:" + hashlib.sha256((s.judge_model + payload).encode()).hexdigest()

    def call():
        raw = chat(JUDGE_SYSTEM, payload, model=s.judge_model, max_tokens=300)
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
        out = json.loads(raw)
        labs = [str(x).lower() for x in out.get("labels", [])]
        labs = (labs + ["unsupported"] * len(sentences))[: len(sentences)]
        return {"labels": labs, "correct": bool(out.get("correct", False))}

    return cache.get(key, call)


def eval_answers(store, dataset, judge_kind: str):
    s = get_settings()
    cache = GenCache(RESULTS / "cache.json")
    per_q = []
    for i, d in enumerate(dataset):
        print(f"  [{i + 1}/{len(dataset)}] {d['id']} {d['question'][:60]}")
        hits = retrieve(store, d["question"], Mode.hybrid_rerank)
        passages = [h.chunk.text for h in hits]
        sources = _sources(hits)
        gkey = "gen:" + hashlib.sha256((s.groq_model + d["question"] + "\x00".join(passages)).encode()).hexdigest()
        raw = cache.get(gkey, lambda: generate_answer(d["question"], passages))

        # Condition A: plain RAG (no retrieval gate, raw model output shown).
        baseline_refused = REFUSAL_TOKEN in raw
        baseline_sents = [] if baseline_refused else [extract_citations(x)[0] for x in split_answer_sentences(raw)]

        # Condition B: GroundTruth (retrieval gate + sentence verification + groundedness policy).
        top = hits[0].rerank_score if hits else None
        if top is None or top < s.min_rerank_score:
            verified = Answer(d["question"], "refused", REFUSAL_MESSAGE, raw, 0.0, "retrieval_gate", [], sources)
        else:
            verdicts = verify_answer(raw, passages) if not baseline_refused else []
            verified = decide(d["question"], raw, verdicts, sources, verify=True)
        verified_sents = [] if verified.status == "refused" else [
            v.text for v in verified.sentences if v.label == "supported"
        ]

        conds = {}
        for name, sents, refused in (
            ("baseline", baseline_sents, baseline_refused),
            ("verified", verified_sents, verified.status == "refused"),
        ):
            if judge_kind == "llm":
                j = judge(cache, d["question"], d["reference"], sents, passages)
            else:  # nli: cheap but circular for the verified condition — reported as such
                labs = [v.label for v in verify_sentences(sents, passages)]
                j = {"labels": labs, "correct": None}
            conds[name] = {"refused": refused, "sentences": sents, "labels": j["labels"], "correct": j["correct"]}
        per_q.append({"id": d["id"], "question": d["question"], "answerable": d["answerable"], "raw": raw,
                      "status": verified.status, "groundedness": verified.groundedness,
                      "refusal_reason": verified.refusal_reason, "top_rerank": top, **conds})
    return summarize_answers(per_q), per_q


def summarize_answers(per_q):
    out = {}
    for cond in ("baseline", "verified"):
        labels = [l for q in per_q if not q[cond]["refused"] for l in q[cond]["labels"]]
        answered = [q for q in per_q if not q[cond]["refused"]]
        ans = [q for q in per_q if q["answerable"]]
        unans = [q for q in per_q if not q["answerable"]]
        correct_refusals = sum(q[cond]["refused"] for q in unans)
        false_refusals = sum(q[cond]["refused"] for q in ans)
        out[cond] = {
            "faithfulness": labels.count("supported") / len(labels) if labels else None,
            "unsupported_rate": labels.count("unsupported") / len(labels) if labels else None,
            "contradicted_rate": labels.count("contradicted") / len(labels) if labels else None,
            "answers_with_any_unfaithful_sentence": (
                sum(any(l != "supported" for l in q[cond]["labels"]) for q in answered) / len(answered) if answered else None
            ),
            "n_sentences": len(labels),
            "refusal_recall": correct_refusals / len(unans) if unans else None,       # refused when it should
            "false_refusal_rate": false_refusals / len(ans) if ans else None,          # refused when it shouldn't
            "refusal_accuracy": (correct_refusals + len(ans) - false_refusals) / len(per_q),
            "answer_correctness": (
                sum(bool(q[cond]["correct"]) for q in ans if not q[cond]["refused"]) / len(ans)
                if ans and ans[0][cond]["correct"] is not None else None
            ),
        }
    reasons = {}
    for q in per_q:
        if q["status"] == "refused":
            reasons[q["refusal_reason"]] = reasons.get(q["refusal_reason"], 0) + 1
    out["verified"]["refusal_reasons"] = reasons
    for cond in ("baseline", "verified"):
        print(f"  {cond:9s} " + "  ".join(
            f"{k}={v:.3f}" for k, v in out[cond].items() if isinstance(v, float)))
    return out


# -------------------------------------------------------------------- report
def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def write_report(report: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))
    lines = ["# GroundTruth evaluation report", "", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    cfg = report.get("config", {})
    lines += [f"Corpus: {cfg.get('chunks')} chunks · dataset: {cfg.get('n_answerable')} answerable + "
              f"{cfg.get('n_unanswerable')} unanswerable questions · LLM: `{cfg.get('llm')}` · judge: `{cfg.get('judge')}`", ""]
    if "retrieval" in report:
        lines += ["## Retrieval quality (answerable questions)", "",
                  "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | ms/query |",
                  "|---|---|---|---|---|---|---|"]
        for mode, m in report["retrieval"].items():
            lines.append(f"| {mode} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | "
                         f"{m['recall@10']:.3f} | {m['mrr']:.3f} | {m['ms_per_query']:.0f} |")
        lines.append("")
    if "verifier" in report:
        v = report["verifier"]
        lines += ["## Verifier accuracy (hand-labelled claims)", "",
                  f"Accuracy **{pct(v['accuracy'])}** on {v['n']} claims. Unsupported/contradicted claims wrongly "
                  f"accepted as supported: **{pct(v['false_accept_rate'])}**.", "",
                  "| Label | Precision | Recall | F1 |", "|---|---|---|---|"]
        for lab, m in v["per_label"].items():
            lines.append(f"| {lab} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
        lines.append("")
    if "answers" in report:
        a = report["answers"]
        lines += ["## Faithfulness and refusal: plain RAG vs GroundTruth", "",
                  "| Metric | Plain RAG | + Verification layer |", "|---|---|---|"]
        rows = [("Faithfulness (supported sentences)", "faithfulness"),
                ("Unsupported sentence rate", "unsupported_rate"),
                ("Contradicted sentence rate", "contradicted_rate"),
                ("Answers containing ≥1 unfaithful sentence", "answers_with_any_unfaithful_sentence"),
                ("Refusal recall (unanswerable refused)", "refusal_recall"),
                ("False refusal rate (answerable refused)", "false_refusal_rate"),
                ("Refusal accuracy (overall)", "refusal_accuracy"),
                ("Answer correctness (answerable)", "answer_correctness")]
        for label, key in rows:
            lines.append(f"| {label} | {pct(a['baseline'][key])} | {pct(a['verified'][key])} |")
        lines += ["", f"Refusal reasons (verified): {a['verified'].get('refusal_reasons')}", ""]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print(f"\nReport written to {RESULTS / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "verifier", "answers", "all"])
    ap.add_argument("--dataset", default=str(EVAL_DIR / "dataset.jsonl"))
    ap.add_argument("--verifier-set", default=str(EVAL_DIR / "verifier_set.jsonl"))
    ap.add_argument("--judge", choices=["llm", "nli"], default="llm",
                    help="who labels answer sentences for faithfulness (nli is free but circular for the verified condition)")
    args = ap.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    prev = RESULTS / "results.json"
    report = json.loads(prev.read_text()) if prev.exists() else {}
    s = get_settings()

    if args.what in ("answers", "all") and not s.groq_api_key:
        sys.exit("GROQ_API_KEY not set — needed for the answers eval (retrieval and verifier evals work without it).")
    if args.what in ("verifier", "all"):
        print("== Verifier ==")
        report["verifier"] = eval_verifier(Path(args.verifier_set))
    if args.what in ("retrieval", "answers", "all"):
        store = build_index()
        report["config"] = {"chunks": len(store.chunks), "llm": s.groq_model,
                            "judge": s.judge_model if args.judge == "llm" else "nli (circular)",
                            "n_answerable": sum(d["answerable"] for d in dataset),
                            "n_unanswerable": sum(not d["answerable"] for d in dataset)}
        if args.what in ("retrieval", "all"):
            print("== Retrieval ==")
            report["retrieval"] = eval_retrieval(store, dataset)
        if args.what in ("answers", "all"):
            print("== Answers ==")
            report["answers"], per_q = eval_answers(store, dataset, args.judge)
            (RESULTS / "per_question.json").write_text(json.dumps(per_q, indent=2))
        store.close()
    write_report(report)


if __name__ == "__main__":
    main()
