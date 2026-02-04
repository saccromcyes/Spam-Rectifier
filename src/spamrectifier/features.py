"""Feature extraction utilities."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d{2,}\b")


@dataclass(frozen=True)
class FeatureConfig:
    use_bigrams: bool = True
    min_token_length: int = 2
    redact_emails: bool = True
    redact_urls: bool = True
    redact_numbers: bool = False


def _redact(text: str, config: FeatureConfig) -> str:
    redacted = text
    if config.redact_emails:
        redacted = EMAIL_RE.sub(" <email> ", redacted)
    if config.redact_urls:
        redacted = URL_RE.sub(" <url> ", redacted)
    if config.redact_numbers:
        redacted = NUMBER_RE.sub(" <number> ", redacted)
    return redacted


def normalize(text: str, config: FeatureConfig) -> str:
    return _redact(text, config).lower()


def tokenize(text: str, config: FeatureConfig) -> list[str]:
    normalized = normalize(text, config)
    tokens = [
        token
        for token in TOKEN_RE.findall(normalized)
        if len(token) >= config.min_token_length
    ]
    if not config.use_bigrams:
        return tokens
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def featurize(texts: Iterable[str], config: FeatureConfig) -> list[Counter[str]]:
    return [Counter(tokenize(text, config)) for text in texts]
