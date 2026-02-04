"""Naive Bayes spam classifier with persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .features import FeatureConfig, featurize

Label = str


@dataclass
class NaiveBayesModel:
    """Multinomial Naive Bayes for text classification."""

    label_counts: dict[Label, int]
    token_counts: dict[Label, dict[str, int]]
    total_tokens: dict[Label, int]
    vocabulary: set[str]
    config: FeatureConfig
    trained_at: str
    dataset_size: int

    @classmethod
    def train(
        cls, texts: Iterable[str], labels: Iterable[Label], config: FeatureConfig
    ) -> "NaiveBayesModel":
        label_counts: dict[Label, int] = {}
        token_counts: dict[Label, dict[str, int]] = {}
        total_tokens: dict[Label, int] = {}
        vocabulary: set[str] = set()

        for features, label in zip(featurize(texts, config), labels, strict=True):
            label_counts[label] = label_counts.get(label, 0) + 1
            token_bucket = token_counts.setdefault(label, {})
            for token, count in features.items():
                token_bucket[token] = token_bucket.get(token, 0) + count
                vocabulary.add(token)
            total_tokens[label] = sum(token_bucket.values())

        return cls(
            label_counts=label_counts,
            token_counts=token_counts,
            total_tokens=total_tokens,
            vocabulary=vocabulary,
            config=config,
            trained_at=datetime.now(timezone.utc).isoformat(),
            dataset_size=sum(label_counts.values()),
        )

    @property
    def labels(self) -> list[Label]:
        return sorted(self.label_counts.keys())

    def predict_proba(self, text: str) -> dict[Label, float]:
        features = featurize([text], self.config)[0]
        vocab_size = max(len(self.vocabulary), 1)
        total_docs = sum(self.label_counts.values())
        log_probs: dict[Label, float] = {}
        for label in self.labels:
            prior = math.log(self.label_counts[label] / total_docs)
            log_likelihood = 0.0
            label_total = self.total_tokens[label]
            token_bucket = self.token_counts[label]
            for token, count in features.items():
                token_count = token_bucket.get(token, 0)
                log_likelihood += count * math.log(
                    (token_count + 1) / (label_total + vocab_size)
                )
            log_probs[label] = prior + log_likelihood

        max_log = max(log_probs.values())
        exp_scores = {label: math.exp(score - max_log) for label, score in log_probs.items()}
        total = sum(exp_scores.values())
        return {label: score / total for label, score in exp_scores.items()}

    def predict(self, text: str) -> Label:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)

    def explain(self, text: str, top_n: int = 8) -> dict[str, object]:
        """Explain prediction by returning top contributing tokens."""
        features = featurize([text], self.config)[0]
        vocab_size = max(len(self.vocabulary), 1)
        label_total = self.total_tokens
        token_counts = self.token_counts

        label_scores: dict[Label, float] = {}
        token_contribs: dict[Label, dict[str, float]] = {}
        for label in self.labels:
            label_log = 0.0
            contribs: dict[str, float] = {}
            for token, count in features.items():
                token_count = token_counts[label].get(token, 0)
                contribution = count * math.log(
                    (token_count + 1) / (label_total[label] + vocab_size)
                )
                contribs[token] = contribution
                label_log += contribution
            label_scores[label] = label_log
            token_contribs[label] = contribs

        prediction = max(label_scores, key=label_scores.get)
        sorted_tokens = sorted(
            token_contribs[prediction].items(), key=lambda item: item[1], reverse=True
        )
        return {
            "prediction": prediction,
            "probabilities": self.predict_proba(text),
            "top_tokens": [
                {"token": token, "contribution": score}
                for token, score in sorted_tokens[:top_n]
            ],
        }

    def top_tokens(self, label: Label, top_n: int = 12) -> list[tuple[str, float]]:
        vocab_size = max(len(self.vocabulary), 1)
        label_total = self.total_tokens[label]
        scored = []
        for token, count in self.token_counts[label].items():
            score = math.log((count + 1) / (label_total + vocab_size))
            scored.append((token, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]

    def save(self, path: str | Path) -> None:
        payload = {
            "label_counts": self.label_counts,
            "token_counts": self.token_counts,
            "total_tokens": self.total_tokens,
            "vocabulary": sorted(self.vocabulary),
            "config": {
                "use_bigrams": self.config.use_bigrams,
                "min_token_length": self.config.min_token_length,
                "redact_emails": self.config.redact_emails,
                "redact_urls": self.config.redact_urls,
                "redact_numbers": self.config.redact_numbers,
            },
            "metadata": {
                "trained_at": self.trained_at,
                "dataset_size": self.dataset_size,
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesModel":
        payload = json.loads(Path(path).read_text())
        config = FeatureConfig(**payload["config"])
        metadata = payload.get("metadata", {})
        return cls(
            label_counts={k: int(v) for k, v in payload["label_counts"].items()},
            token_counts={
                label: {token: int(count) for token, count in tokens.items()}
                for label, tokens in payload["token_counts"].items()
            },
            total_tokens={k: int(v) for k, v in payload["total_tokens"].items()},
            vocabulary=set(payload["vocabulary"]),
            config=config,
            trained_at=metadata.get(
                "trained_at", datetime.now(timezone.utc).isoformat()
            ),
            dataset_size=int(metadata.get("dataset_size", 0)),
        )
