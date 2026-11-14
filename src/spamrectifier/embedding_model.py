"""Sentence embeddings + logistic regression, explained by nearest labelled neighbours.

Embeddings capture meaning that word counts miss ("you've been selected for a cash reward" and
"claim your prize" share no tokens), and the nearest training messages give a concrete,
human-checkable reason for each verdict.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class HashEmbedder:
    """Dependency-free fallback used in tests and offline development."""

    name = "hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = re.findall(r"[a-z0-9£$]+", text.lower())
            for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class FastEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        threads = int(os.environ["ONNX_THREADS"]) if os.environ.get("ONNX_THREADS") else None  # 1 on small shared CPUs
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.stack(list(self._model.embed(texts, batch_size=64))).astype(np.float32)
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def build_embedder(kind: str = "fastembed", cache_dir: str | None = None):
    if kind == "fastembed":
        try:
            return FastEmbedder(cache_dir=cache_dir)
        except Exception as exc:
            log.warning("FastEmbed unavailable (%s); using hash embeddings", exc)
    return HashEmbedder()


class EmbeddingClassifier:
    """Binary logistic regression (spam = 1) trained with full-batch gradient descent, class-balanced."""

    def __init__(self, embedder, weights: np.ndarray | None = None, bias: float = 0.0,
                 train_vectors: np.ndarray | None = None, train_texts: list[str] | None = None,
                 train_labels: list[str] | None = None):
        self.embedder = embedder
        self.w = weights
        self.b = bias
        self.train_vectors = train_vectors
        self.train_texts = train_texts or []
        self.train_labels = train_labels or []

    @classmethod
    def train(cls, embedder, texts: list[str], labels: list[str], epochs: int = 400, lr: float = 0.5, l2: float = 1e-3):
        X = embedder.embed(texts)
        y = np.array([1.0 if lbl == "spam" else 0.0 for lbl in labels], dtype=np.float32)
        pos = y.mean()
        sample_w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos)).astype(np.float32)  # balance classes
        w = np.zeros(X.shape[1], dtype=np.float32)
        b = 0.0
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
            grad = (p - y) * sample_w
            w -= lr * (X.T @ grad / len(y) + l2 * w)
            b -= lr * float(grad.mean())
        return cls(embedder, w, b, X.astype(np.float16), list(texts), list(labels))

    def spam_probability(self, vectors: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(vectors @ self.w + self.b)))

    def predict_proba(self, text: str) -> dict[str, float]:
        p = float(self.spam_probability(self.embedder.embed([text]))[0])
        return {"spam": p, "ham": 1 - p}

    def predict_many(self, texts: list[str]) -> list[str]:
        probs = self.spam_probability(self.embedder.embed(texts))
        return ["spam" if p >= 0.5 else "ham" for p in probs]

    def neighbours(self, text: str, k: int = 3) -> list[dict]:
        v = self.embedder.embed([text])[0]
        sims = self.train_vectors.astype(np.float32) @ v
        top = np.argsort(sims)[::-1][:k]
        return [{"text": self.train_texts[i], "label": self.train_labels[i], "similarity": round(float(sims[i]), 3)} for i in top]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, w=self.w, b=np.array([self.b]), vectors=self.train_vectors,
            texts=np.array(self.train_texts, dtype=object), labels=np.array(self.train_labels, dtype=object),
            embedder=np.array([self.embedder.name]),
        )

    @classmethod
    def load(cls, path: str | Path, embedder) -> EmbeddingClassifier:
        data = np.load(path, allow_pickle=True)
        if str(data["embedder"][0]) != embedder.name:
            raise ValueError(f"Model was trained with {data['embedder'][0]}, not {embedder.name}")
        return cls(embedder, data["w"], float(data["b"][0]), data["vectors"], list(data["texts"]), list(data["labels"]))
