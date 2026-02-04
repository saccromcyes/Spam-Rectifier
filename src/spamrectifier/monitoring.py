"""Drift monitoring utilities."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .features import FeatureConfig, featurize
from .model import NaiveBayesModel


def _normalize(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {token: count / total for token, count in counter.items()}


def _jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}
    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    score = 0.0
    for key, value in p.items():
        if value == 0:
            continue
        score += value * math.log2(value / max(q.get(key, 1e-12), 1e-12))
    return score


def token_distribution(texts: Iterable[str], config: FeatureConfig) -> dict[str, float]:
    counter: Counter[str] = Counter()
    for features in featurize(texts, config):
        counter.update(features)
    return _normalize(counter)


def model_token_distribution(model: NaiveBayesModel) -> dict[str, float]:
    counter: Counter[str] = Counter()
    for token_bucket in model.token_counts.values():
        counter.update(token_bucket)
    return _normalize(counter)


def drift_report(
    model: NaiveBayesModel, texts: Iterable[str], top_n: int = 10
) -> dict[str, object]:
    texts_list = list(texts)
    model_dist = model_token_distribution(model)
    data_dist = token_distribution(texts_list, model.config)
    js_divergence = _jensen_shannon_divergence(model_dist, data_dist)

    all_tokens = set(model_dist) | set(data_dist)
    shifts = []
    for token in all_tokens:
        model_prob = model_dist.get(token, 0.0)
        data_prob = data_dist.get(token, 0.0)
        shifts.append(
            {
                "token": token,
                "model_prob": model_prob,
                "data_prob": data_prob,
                "delta": data_prob - model_prob,
            }
        )
    shifts.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "js_divergence": js_divergence,
        "top_shifted_tokens": shifts[:top_n],
        "data_size": len(texts_list),
    }
