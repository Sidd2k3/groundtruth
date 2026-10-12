# GroundTruth evaluation report

_Generated 2026-10-12 12:36_

Corpus: 29 chunks · dataset: 32 answerable + 12 unanswerable questions · LLM: `llama-3.3-70b-versatile` · judge: `llama-3.3-70b-versatile`

## Retrieval quality (answerable questions)

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 0.781 | 0.875 | 0.875 | 0.938 | 0.822 | 0 |
| dense | 0.844 | 0.938 | 0.969 | 1.000 | 0.902 | 6 |
| hybrid | 0.781 | 0.906 | 1.000 | 1.000 | 0.866 | 6 |
| hybrid_rerank | 0.875 | 1.000 | 1.000 | 1.000 | 0.938 | 350 |

## Verifier accuracy (hand-labelled claims)

Accuracy **92.5%** on 40 claims. Unsupported/contradicted claims wrongly accepted as supported: **0.0%**.

| Label | Precision | Recall | F1 |
|---|---|---|---|
| supported | 1.000 | 1.000 | 1.000 |
| unsupported | 1.000 | 0.700 | 0.824 |
| contradicted | 0.824 | 1.000 | 0.903 |
