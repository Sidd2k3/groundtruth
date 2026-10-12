"""FastAPI backend.

Run:  uvicorn api.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from groundtruth.config import get_settings
from groundtruth.ingest import SUPPORTED, chunk_file
from groq import APIStatusError

from groundtruth.llm import LLMNotConfigured
from groundtruth.models import embedder, nli, reranker
from groundtruth.pipeline import answer_question
from groundtruth.retrieval import Mode, retrieve
from groundtruth.store import get_store

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the models and open the store once, so the first request isn't slow.
    embedder(), reranker(), nli()
    store = get_store()
    seed = get_settings().seed_dir
    if seed and not store.chunks:  # demo deployments start with the sample corpus loaded
        for f in sorted(Path(seed).iterdir()):
            if f.suffix.lower() in SUPPORTED:
                store.add(chunk_file(f.name, f.read_bytes()))
    yield


app = FastAPI(title="GroundTruth", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    verify: bool = True
    mode: Mode = Mode.hybrid_rerank


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Mode = Mode.hybrid_rerank
    k: int = Field(5, ge=1, le=50)


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "llm_configured": bool(s.groq_api_key),
        "model": s.groq_model,
        "documents": len(get_store().documents()),
        "thresholds": {
            "show": s.groundedness_show,
            "refuse": s.groundedness_refuse,
            "entail": s.entail_threshold,
        },
    }


@app.get("/documents")
def list_documents():
    return get_store().documents()


@app.post("/documents")
async def upload(files: list[UploadFile] = File(...)):
    store = get_store()
    results = []
    for f in files:
        suffix = "." + (f.filename or "").rsplit(".", 1)[-1].lower()
        if suffix not in SUPPORTED:
            raise HTTPException(400, f"{f.filename}: unsupported type. Supported: {sorted(SUPPORTED)}")
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{f.filename}: larger than 20 MB")
        try:
            chunks = await run_in_threadpool(chunk_file, f.filename, data)
        except Exception as e:  # malformed PDF/DOCX etc.
            raise HTTPException(422, f"{f.filename}: could not parse ({e})")
        if not chunks:
            raise HTTPException(422, f"{f.filename}: no extractable text (scanned PDF?)")
        results.append({"doc_name": f.filename, "doc_id": chunks[0].doc_id, "chunks": await run_in_threadpool(store.add, chunks)})
    return results


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    get_store().delete_doc(doc_id)
    return {"deleted": doc_id}


@app.post("/search")
def search(req: SearchRequest):
    hits = retrieve(get_store(), req.query, req.mode, req.k)
    return [
        {"doc_name": h.chunk.doc_name, "section": h.chunk.section, "text": h.chunk.text,
         "score": h.score, "rerank_score": h.rerank_score}
        for h in hits
    ]


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer_question(get_store(), req.question, verify=req.verify, mode=req.mode).to_dict()
    except LLMNotConfigured as e:
        raise HTTPException(503, str(e))
    except APIStatusError as e:  # bad key, retired model, rate limit...
        msg = e.body.get("error", {}).get("message") if isinstance(e.body, dict) else None
        raise HTTPException(502, f"Groq API error {e.status_code} for model '{get_settings().groq_model}': {msg or e}")
