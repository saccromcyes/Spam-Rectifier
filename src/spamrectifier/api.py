"""FastAPI service for SpamRectifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .model import NaiveBayesModel
from .preview import load_preview_html
from .reporting import Metrics, build_model_card


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class ExplainResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    top_tokens: list[dict[str, Any]]


def create_app(model_path: str | Path) -> FastAPI:
    model = NaiveBayesModel.load(model_path)
    app = FastAPI(title="SpamRectifier API", version="1.0")
    ui_html = load_preview_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def preview() -> HTMLResponse:
        return HTMLResponse(ui_html)

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        probabilities = model.predict_proba(request.text)
        return PredictResponse(
            prediction=model.predict(request.text), probabilities=probabilities
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(request: PredictRequest) -> ExplainResponse:
        explanation = model.explain(request.text)
        return ExplainResponse(**explanation)

    @app.get("/model-card")
    def model_card() -> dict[str, str]:
        if not model.dataset_size:
            raise HTTPException(status_code=400, detail="Model metadata missing dataset size.")
        top_tokens = {label: model.top_tokens(label) for label in model.labels}
        card = build_model_card(
            model_name="SpamRectifier",
            version="1.0",
            labels=model.labels,
            metrics=Metrics(precision=0.0, recall=0.0, f1=0.0, accuracy=0.0),
            dataset_size=model.dataset_size,
            trained_at=model.trained_at,
            positive_label="spam",
            top_tokens=top_tokens,
        )
        return {"card": card}

    return app
