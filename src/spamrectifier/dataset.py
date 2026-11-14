"""The SMS Spam Collection and a fixed, stratified train/test split.

Almeida, T.A. and Gómez Hidalgo, J.M. (2011). SMS Spam Collection v.1. UCI Machine Learning
Repository, CC BY 4.0. 5,574 messages, 747 of them spam.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "sms_spam_collection.tsv"


@dataclass(frozen=True)
class Split:
    train_texts: list[str]
    train_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]


def load_sms(path: str | Path = DEFAULT_PATH) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        label, _, text = line.partition("\t")
        if label in ("ham", "spam") and text.strip():
            labels.append(label)
            texts.append(text.strip())
    return texts, labels


def stratified_split(texts: list[str], labels: list[str], test_size: float = 0.2, seed: int = 13) -> Split:
    """Same split every run, same spam ratio in train and test, and no message in both."""
    rng = random.Random(seed)
    seen: set[str] = set()
    by_label: dict[str, list[str]] = {}
    for text, label in zip(texts, labels, strict=True):
        key = text.lower()
        if key in seen:  # the corpus has exact duplicates; keep one so test never leaks into train
            continue
        seen.add(key)
        by_label.setdefault(label, []).append(text)
    train, test = [], []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        cut = round(len(items) * test_size)
        test += [(t, label) for t in items[:cut]]
        train += [(t, label) for t in items[cut:]]
    rng.shuffle(train)
    rng.shuffle(test)
    return Split([t for t, _ in train], [lbl for _, lbl in train], [t for t, _ in test], [lbl for _, lbl in test])
