"""Streamlit frontend.

Two modes, same code path:
  - API_URL set   -> talks to a separately running FastAPI backend over HTTP
  - API_URL unset -> runs the FastAPI app in-process (one process; used on Streamlit Community Cloud)

Run:  streamlit run ui/app.py
"""
import html
import os
import sys
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "").rstrip("/")

st.set_page_config(page_title="GroundTruth", page_icon="✅", layout="wide")

LABEL_STYLE = {
    "supported": ("#1a7f37", "rgba(46,160,67,0.15)", "Supported"),
    "unsupported": ("#9a6700", "rgba(212,167,44,0.20)", "Unsupported"),
    "contradicted": ("#cf222e", "rgba(248,81,73,0.18)", "Contradicted"),
    "skipped": ("#6e7781", "transparent", "Not a claim"),
}
STATUS_BANNER = {
    "grounded": ("success", "Grounded — every claim shown is backed by the sources."),
    "warning": ("warning", "Partially grounded — unsupported sentences were removed. Check the breakdown."),
    "refused": ("error", "Refused — the documents don't support an answer to this question."),
}


@st.cache_resource(show_spinner="Loading models and indexing the sample documents (first start takes a few minutes)…")
def local_client():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    client.__enter__()  # runs the app's startup: warm models, seed corpus
    return client


def api(method: str, path: str, **kw):
    if not API_URL:
        r = local_client().request(method, path, **kw)
    else:
        try:
            r = requests.request(method, f"{API_URL}{path}", timeout=180, **kw)
        except requests.ConnectionError:
            st.error(f"Can't reach the API at {API_URL}. Is `uvicorn api.main:app` running?")
            st.stop()
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        st.error(f"API error {r.status_code}: {detail}")
        st.stop()
    return r.json()


# ---------------- Sidebar: documents & settings ----------------
with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDF, DOCX, MD or TXT", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index documents", type="primary"):
        with st.spinner("Chunking and embedding…"):
            res = api("POST", "/documents", files=[("files", (u.name, u.getvalue())) for u in uploads])
        st.success(", ".join(f"{d['doc_name']} ({d['chunks']} chunks)" for d in res))

    docs = api("GET", "/documents")
    if not docs:
        st.caption("No documents indexed yet.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d['doc_name']}**  \n<small>{d['chunks']} chunks</small>", unsafe_allow_html=True)
        if c2.button("🗑", key=f"del-{d['doc_id']}", help="Remove document"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()

    st.divider()
    st.header("⚙️ Options")
    verify = st.toggle("Verification layer", value=True, help="Off = plain RAG, shown for comparison.")
    mode = st.selectbox("Retrieval", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    health = api("GET", "/health")
    if not health["llm_configured"]:
        st.warning("GROQ_API_KEY is not set on the server — answering is disabled.")
    th = health["thresholds"]
    st.caption(f"Model: `{health['model']}` · show ≥ {th['show']:.0%} · refuse < {th['refuse']:.0%}")


# ---------------- Main: Q&A ----------------
st.title("GroundTruth")
st.caption("Answers that check themselves: every sentence is verified against the source passages.")

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.setdefault("history", []).insert(0, None)
    with st.spinner("Retrieving, answering, verifying…"):
        result = api("POST", "/ask", json={"question": question, "verify": verify, "mode": mode})
    result["_verify"] = verify
    st.session_state["history"][0] = result

for i, res in enumerate(st.session_state.get("history", [])):
    if res is None:
        continue
    with st.container(border=True):
        st.markdown(f"#### ❓ {res['question']}")
        kind, msg = STATUS_BANNER[res["status"]]
        if not res.get("_verify", True) and res["status"] != "refused":
            kind, msg = "info", "Verification off — raw model output shown. Highlighting below is for comparison."
        getattr(st, kind)(msg)

        m1, m2, m3 = st.columns(3)
        m1.metric("Groundedness", f"{res['groundedness']:.0%}")
        counts = {k: sum(s["label"] == k for s in res["sentences"]) for k in LABEL_STYLE}
        m2.metric("Supported / claims", f"{counts['supported']} / {len(res['sentences']) - counts['skipped']}")
        m3.metric("Latency", f"{sum(res['timings'].values()):.1f}s" if res["timings"] else "—")

        if res["status"] != "refused" or res["raw_answer"]:
            st.markdown("**Answer**")
            st.write(res["answer"])

        if res["sentences"]:
            st.markdown("**Sentence-by-sentence verification** (of the model's raw answer)")
            spans = []
            for s in res["sentences"]:
                fg, bg, name = LABEL_STYLE[s["label"]]
                cite = f" → [{s['evidence_passage']}]" if s["evidence_passage"] else ""
                tip = f"{name}{cite} · entail {s['entailment']:.2f} · contradict {s['contradiction']:.2f}"
                spans.append(
                    f'<span title="{html.escape(tip)}" style="background:{bg};border-bottom:2px solid {fg};'
                    f'padding:2px 3px;border-radius:3px;line-height:2">{html.escape(s["text"])}</span>'
                )
            st.markdown(" ".join(spans), unsafe_allow_html=True)
            legend = " ".join(
                f'<span style="color:{fg}">■</span> {name}' for fg, _, name in LABEL_STYLE.values()
            )
            st.markdown(f"<small>{legend} · hover a sentence for scores</small>", unsafe_allow_html=True)

            with st.expander("Evidence for each sentence"):
                for s in res["sentences"]:
                    if s["label"] == "skipped":
                        continue
                    fg, _, name = LABEL_STYLE[s["label"]]
                    st.markdown(f'<span style="color:{fg};font-weight:600">{name}</span> — {html.escape(s["text"])}', unsafe_allow_html=True)
                    if s["evidence"]:
                        st.caption(f"Best evidence (passage [{s['evidence_passage']}]): {s['evidence']}")
                    if s["citation_ok"] is False:
                        st.caption("⚠️ The model cited a different passage than the one that actually supports this.")

        if res.get("refusal_reason"):
            st.caption(f"Refusal reason: `{res['refusal_reason']}`")

        with st.expander(f"Sources ({len(res['sources'])})"):
            for src in res["sources"]:
                score = f" · rerank {src['rerank_score']:.2f}" if src["rerank_score"] is not None else ""
                st.markdown(f"**[{src['n']}] {src['doc_name']}** {('— ' + src['section']) if src['section'] else ''}{score}")
                st.caption(src["text"])
