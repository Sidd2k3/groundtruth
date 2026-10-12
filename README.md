# GroundTruth — a RAG system that verifies its own answers

Upload documents (policies, regulations, course notes) and ask questions. Every sentence of the
answer is checked against the source passages with an NLI model and labelled **supported**,
**unsupported** or **contradicted**, with the exact evidence it was checked against. Answers whose
groundedness falls below a threshold are trimmed, flagged, or refused instead of being shown as fact.

Everything is free: Groq (gpt-oss-120b), Qdrant, open-source local models, FastAPI, Streamlit.

```
question
   │
   ▼
┌──────────────┐   BM25 (rank-bm25)  ┐
│  Retrieval   │                      ├─► Reciprocal Rank Fusion ─► cross-encoder rerank ─► top 5 passages
│              │   dense (bge-small + Qdrant) ┘
└──────────────┘
   │  retrieval gate: best rerank score too low? ──► refuse (no LLM call)
   ▼
┌──────────────┐
│  Generation  │  Groq · gpt-oss-120b · "cite [n] after every sentence, or say INSUFFICIENT_CONTEXT"
└──────────────┘
   │
   ▼
┌──────────────┐  each answer sentence × each 1- and 2-sentence window of the passages
│ Verification │  NLI cross-encoder (DeBERTa-v3) → P(entail), P(contradict)
└──────────────┘  → supported / unsupported / contradicted + evidence + citation check
   │
   ▼
 groundedness = supported claims / all claims
   ≥ 0.8  → grounded: show supported sentences
   ≥ 0.5  → warning:  show supported sentences, flag the rest
   < 0.5  → refused
```

## Results

See [`eval/results/report.md`](eval/results/report.md) for the latest run. Summary on the bundled corpus
(4 fictional policy documents, 32 answerable + 12 unanswerable hand-labelled questions, 40 hand-labelled verifier claims):

**Retrieval** (32 answerable questions, 29 chunks)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| BM25 | 0.781 | 0.875 | 0.875 | 0.822 |
| Dense (bge-small) | 0.844 | 0.938 | 0.969 | 0.902 |
| Hybrid (RRF) | 0.781 | 0.906 | 1.000 | 0.866 |
| **Hybrid + reranker** | **0.875** | **1.000** | **1.000** | **0.938** |

Hybrid fixes recall at k=5 (BM25 misses paraphrases, dense misses exact terms), and the reranker fixes the
ordering: the right passage is first 87.5% of the time and always in the top 3.

**Verifier** (40 hand-labelled claims; how the NLI model was chosen)

| NLI model | Premise | Accuracy | Bad claims accepted |
|---|---|---|---|
| nli-deberta-v3-small | whole chunk / 3-sentence windows | 70.0% | 4.2% |
| nli-deberta-v3-small | 1- and 2-sentence windows | 82.5% | 4.2% |
| nli-deberta-v3-base | 1- and 2-sentence windows | 90.0% | 0.0% |
| **DeBERTa-v3-base-mnli-fever-anli** (default) | 1- and 2-sentence windows | **92.5%** | **0.0%** |

All errors left are the safe kind: unsupported claims labelled *contradicted*. No unsupported or
contradicted claim was accepted as supported. (The same set was used to pick the model, so treat these
as dev-set numbers.)

**Faithfulness and refusal** (plain RAG vs + verification layer): run `python -m eval.run_eval answers`
with a Groq key. It fills in this table in `eval/results/report.md`: faithfulness, unsupported and
contradicted sentence rates, refusal recall, false-refusal rate and answer correctness.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # paste your free Groq key into GROQ_API_KEY
```

Run the app (the UI runs the backend in-process):

```bash
streamlit run streamlit_app.py
```

Or run the API separately, with its own docs at http://localhost:8000/docs, and point the UI at it:

```bash
uvicorn api.main:app --reload
```

```bash
API_URL=http://localhost:8000 streamlit run ui/app.py
```

Open http://localhost:8501, upload the files in `data/corpus/` (or your own PDFs/DOCX/MD/TXT), and ask
something like *"Can I fly business class to a conference?"* or *"What is the office dress code?"* (not covered → refused).

No Docker needed: Qdrant runs embedded and stores data in `./qdrant_data`. To use a Qdrant server
instead, set `QDRANT_URL` (or run the full stack with `docker compose up --build`).

## API

| Method | Path | Body | Purpose |
|---|---|---|---|
| `GET` | `/health` | — | status, model, thresholds |
| `GET` | `/documents` | — | indexed documents |
| `POST` | `/documents` | multipart `files` | upload + index (re-uploading a file replaces it) |
| `DELETE` | `/documents/{doc_id}` | — | remove a document |
| `POST` | `/search` | `{query, mode, k}` | retrieval only (`bm25`, `dense`, `hybrid`, `hybrid_rerank`) |
| `POST` | `/ask` | `{question, verify, mode}` | answer + per-sentence verdicts + sources |

`/ask` returns `status` (`grounded` / `warning` / `refused`), the shown `answer`, the model's `raw_answer`,
`groundedness`, `refusal_reason` (`retrieval_gate` / `model_declined` / `low_groundedness`), a `sentences`
list with label, entailment/contradiction probabilities, evidence text and whether the model's own citation
was right, plus `sources` and per-stage `timings`. Interactive docs at http://localhost:8000/docs.

## Evaluation harness

```bash
python -m eval.run_eval retrieval   # Recall@k + MRR for bm25 / dense / hybrid / hybrid+rerank   (no key needed)
python -m eval.run_eval verifier    # NLI verifier vs 40 hand-labelled claims                    (no key needed)
python -m eval.run_eval answers     # faithfulness + refusal, plain RAG vs verified              (needs GROQ_API_KEY)
python -m eval.run_eval all
```

Writes `eval/results/report.md`, `results.json` and `per_question.json`. It indexes `data/corpus/` into a
separate Qdrant store (`./qdrant_eval`), so it never touches your app data. Groq calls are cached in
`eval/results/cache.json`, so re-runs are free and stay within the free-tier limits.

**What each part measures**

- **Retrieval** — for each answerable question, `evidence` lists the exact sentence(s) that answer it.
  A retrieved chunk counts as relevant if it contains one (after normalisation), so labels don't depend
  on chunk size or IDs: you can change the chunker and re-run. Recall@k = fraction of evidence found in
  the top k; MRR = 1 / rank of the first relevant chunk.
- **Verifier** — `eval/verifier_set.jsonl` has 40 (passage, claim, label) triples written to be hard:
  paraphrases that should pass, wrong numbers and negations that should be contradicted, and plausible
  extra details that should be unsupported. The key safety number is the **false-accept rate**:
  how often a bad claim is let through as supported.
- **Faithfulness** — one Groq generation per question (shared by both conditions). *Plain RAG* shows the
  raw answer; *GroundTruth* applies the retrieval gate, verification and the groundedness policy. Every
  shown sentence is then labelled by an **independent LLM judge** (not the NLI model — that would be
  grading its own homework). `--judge nli` is available as a free, offline but circular alternative.
  Set `JUDGE_MODEL` to a different model family than `GROQ_MODEL` to reduce self-preference bias.
- **Refusal** — on the 12 unanswerable questions (several deliberately close to the corpus, e.g. per diem
  for *day trips*, hotel limit for *Singapore*) we want a refusal; on the 32 answerable ones we don't.
  Reported as refusal recall, false-refusal rate and overall accuracy, plus which stage refused.

To evaluate on your own documents: drop them in `data/corpus/`, write questions in `eval/dataset.jsonl`
(`answerable`, `reference`, and verbatim `evidence` snippets), and re-run.

## Design decisions

- **Hybrid retrieval with RRF.** BM25 nails exact terms ("SMS", "401(k)", section names); dense retrieval
  handles paraphrase ("doctor's note" → "medical certificate"). Reciprocal Rank Fusion merges them without
  having to calibrate their very different score scales.
- **Rerank, then gate.** The cross-encoder sees query and passage together, so its score is a much better
  "is this actually relevant?" signal than cosine similarity — good enough to refuse before spending an LLM
  call when nothing relevant was retrieved.
- **Sentence-sized NLI premises.** NLI models are trained on single-sentence premises. Feeding a whole
  900-character chunk makes them answer "neutral" even for near-verbatim claims, so each claim is checked
  against every sentence and every adjacent pair of sentences in the retrieved passages, taking the max.
- **Contradiction only from cited passages.** NLI models label unrelated text as "contradiction" with high
  confidence (try "A man eats pizza" against a password policy). A sentence is only marked contradicted
  based on the passages the LLM cited for it.
- **Trim, don't just score.** In the verified answer, unsupported and contradicted sentences are removed
  and citations are rewritten to the passage the verifier actually found support in. The UI still shows
  the raw answer with each sentence highlighted, so nothing is hidden.
- **Chunks never split a sentence**, and carry their section path ("Leave Policy > Sick Leave: …") so both
  BM25 and embeddings see the context.

## Configuration

All settings live in [`groundtruth/config.py`](groundtruth/config.py) and can be overridden with env vars
or `.env` (see [`.env.example`](.env.example)). The ones you'll most likely tune:

| Variable | Default | Meaning |
|---|---|---|
| `NLI_MODEL` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | verifier model |
| `ENTAIL_THRESHOLD` | 0.5 | P(entailment) for "supported" |
| `GROUNDEDNESS_SHOW` / `GROUNDEDNESS_REFUSE` | 0.8 / 0.5 | answer policy |
| `MIN_RERANK_SCORE` | -10.0 | retrieval gate (cross-encoder logit) |
| `CHUNK_CHARS` | 900 | target chunk size |
| `CANDIDATE_K` / `FINAL_K` | 20 / 5 | candidates before rerank / passages sent to the LLM |

## Deployment (free)

**Streamlit Community Cloud** runs the whole app as one process: with no `API_URL` set, the UI runs the
FastAPI app in-process (same code path as the HTTP API), and [`streamlit_app.py`](streamlit_app.py)
indexes the sample corpus on first start. Peak memory is about 0.9 GB, well inside the free tier.

1. Push this repo to GitHub.
2. On https://share.streamlit.io, click **Create app**, pick the repo and branch, and set the main file to `streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.12 and add the secret `GROQ_API_KEY = "gsk_..."`.
4. Deploy. The first build takes a few minutes (models download on first start).

Notes:
- Storage is not persistent: uploaded documents disappear when the app restarts (the sample corpus is
  re-indexed automatically). To keep uploads, add `QDRANT_URL` / `QDRANT_API_KEY` secrets pointing at a
  free Qdrant Cloud cluster.
- Apps sleep after a period without visitors and take a minute to wake up.
- Every visitor's question uses your Groq quota.

**Docker** (API + UI in one container, models baked in): the root `Dockerfile` works on any Docker host,
for example a Hugging Face Docker Space (needs HF PRO, see [`deploy/push_space.py`](deploy/push_space.py))
or Render (Starter plan or above, see [`render.yaml`](render.yaml)).

## Tests

```bash
pytest -q
```

`tests/test_pipeline.py` runs the full pipeline over the sample corpus with a fake LLM (no key needed),
including a check that an injected hallucination is caught and removed.

## Project layout

```
groundtruth/      core library
  config.py       settings (env-overridable)
  ingest.py       PDF/DOCX/MD/TXT loading, section-aware sentence chunking
  models.py       embedder, reranker, NLI (lazy singletons)
  store.py        Qdrant + BM25 index kept in sync
  retrieval.py    bm25 / dense / hybrid (RRF) / hybrid + rerank
  llm.py          Groq client + grounded-answer prompt
  verify.py       sentence-level NLI verification
  pipeline.py     retrieve → gate → generate → verify → decide
api/main.py       FastAPI backend
ui/app.py         Streamlit frontend
eval/             dataset, verifier set, harness, results
data/corpus/      sample corpus (fictional company policies)
tests/
```

## Limitations

- The bundled corpus is small and synthetic, so absolute numbers are optimistic; the *relative*
  comparisons (retrievers, with vs without verification) are the point. Re-run on your own documents.
- NLI checks each sentence against one or two source sentences at a time; claims that need combining
  facts from distant parts of a document (multi-hop) can be marked unsupported even when true.
- The LLM judge is itself a model; spot-check `per_question.json` before quoting numbers.
- Scanned PDFs need OCR first (no text layer → rejected at upload).
