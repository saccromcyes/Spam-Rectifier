"""Dataset loading utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabeledDataset:
    texts: list[str]
    labels: list[str]


def load_csv(path: str | Path) -> LabeledDataset:
    """Load CSV with columns text,label."""
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)
    return LabeledDataset(texts=texts, labels=labels)
