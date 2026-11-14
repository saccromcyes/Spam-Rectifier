"""Web app: classify a message with all three models side by side, and show the benchmark."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .benchmark import ARTIFACTS, RESULTS, build
from .dataset import ROOT
from .embedding_model import EmbeddingClassifier, build_embedder
from .llm_classifier import LLMClassifier
from .model import NaiveBayesModel

log = logging.getLogger("spamrectifier")
WEB = ROOT / "web"


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class Models:
    def __init__(self):
        embedder = build_embedder(os.getenv("EMBEDDER", "fastembed"), os.getenv("MODEL_CACHE_DIR"))
        try:
            self.nb = NaiveBayesModel.load(ARTIFACTS / "naive_bayes.json")
            self.emb = EmbeddingClassifier.load(ARTIFACTS / "embedding_model.npz", embedder)
        except (FileNotFoundError, ValueError) as exc:
            log.info("Training models (%s)", exc)
            _, self.nb, self.emb = build(os.getenv("EMBEDDER", "fastembed"), os.getenv("MODEL_CACHE_DIR"))
        self.llm = LLMClassifier.from_env()


def create_app() -> FastAPI:
    hits: dict[str, deque] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = Models()
        yield

    app = FastAPI(title="SpamRectifier", version="2.0.0", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)

    def rate_limit(request: Request, limit: int = 60, window: int = 600) -> None:
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
        now, q = time.monotonic(), hits[ip]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, "Rate limit reached. Try again in a few minutes.")
        q.append(now)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config(request: Request):
        m: Models = request.app.state.models
        return {"embedder": m.emb.embedder.name, "llm_model": m.llm.model if m.llm else None, "train_size": m.nb.dataset_size}

    @app.post("/api/classify")
    async def classify(body: ClassifyRequest, request: Request):
        rate_limit(request)
        m: Models = request.app.state.models
        text = body.text.strip()

        def local():
            t0 = time.perf_counter()
            nb_proba = m.nb.predict_proba(text)
            nb_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            emb_proba = m.emb.predict_proba(text)
            emb_ms = (time.perf_counter() - t0) * 1000
            return [
                {
                    "key": "naive_bayes", "name": "Naive Bayes", "label": max(nb_proba, key=nb_proba.get),
                    "spam_probability": round(nb_proba.get("spam", 0.0), 4), "latency_ms": round(nb_ms, 2),
                    "evidence": m.nb.token_evidence(text),
                },
                {
                    "key": "embeddings", "name": "Embeddings", "label": "spam" if emb_proba["spam"] >= 0.5 else "ham",
                    "spam_probability": round(emb_proba["spam"], 4), "latency_ms": round(emb_ms, 2),
                    "neighbours": m.emb.neighbours(text),
                },
            ]

        def llm():
            try:
                out = m.llm.classify(text)
                return {"key": "llm", "name": "LLM", **out}
            except Exception as exc:
                log.warning("LLM failed: %s", exc)
                return {"key": "llm", "name": "LLM", "error": "The language model didn't respond. The other two verdicts still stand."}

        with ThreadPoolExecutor(max_workers=2) as pool:  # the LLM call overlaps the local models
            llm_future = pool.submit(llm) if m.llm else None
            results = await run_in_threadpool(local)
            if llm_future:
                results.append(await run_in_threadpool(llm_future.result))
        votes = [r["label"] for r in results if "label" in r]
        spam_votes = votes.count("spam")
        return {
            "results": results,
            "consensus": "spam" if spam_votes > len(votes) / 2 else "ham",
            "agreement": f"{max(spam_votes, len(votes) - spam_votes)} of {len(votes)}",
        }

    @app.get("/api/benchmark")
    def benchmark():
        if not RESULTS.exists():
            return JSONResponse({"available": False})
        return {"available": True, **json.loads(RESULTS.read_text())}

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB / "index.html")

    return app


app = create_app()
