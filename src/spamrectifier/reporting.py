"""Evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    accuracy: float


def classification_report(
    labels: Iterable[str], predictions: Iterable[str], positive_label: str
) -> Metrics:
    labels_list = list(labels)
    preds_list = list(predictions)
    if len(labels_list) != len(preds_list):
        raise ValueError("labels and predictions must be same length")

    tp = sum(1 for y, y_hat in zip(labels_list, preds_list, strict=True) if y == positive_label and y_hat == positive_label)
    fp = sum(1 for y, y_hat in zip(labels_list, preds_list, strict=True) if y != positive_label and y_hat == positive_label)
    fn = sum(1 for y, y_hat in zip(labels_list, preds_list, strict=True) if y == positive_label and y_hat != positive_label)
    tn = len(labels_list) - tp - fp - fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels_list) if labels_list else 0.0

    return Metrics(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def build_model_card(
    *,
    model_name: str,
    version: str,
    labels: Iterable[str],
    metrics: Metrics,
    dataset_size: int,
    trained_at: str,
    positive_label: str,
    top_tokens: dict[str, list[tuple[str, float]]],
) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    label_list = ", ".join(sorted(labels))
    token_blocks = []
    for label, tokens in top_tokens.items():
        token_lines = "\n".join(f"- `{token}` ({score:.4f})" for token, score in tokens)
        token_blocks.append(f"### Top tokens for `{label}`\n{token_lines}")
    tokens_section = "\n\n".join(token_blocks)

    return f"""# Model Card: {model_name}

**Version**: {version}  
**Trained At**: {trained_at}  
**Report Generated**: {timestamp}

## Overview
- **Labels**: {label_list}
- **Positive Label**: {positive_label}
- **Dataset Size**: {dataset_size}

## Metrics
- **Precision**: {metrics.precision:.3f}
- **Recall**: {metrics.recall:.3f}
- **F1 Score**: {metrics.f1:.3f}
- **Accuracy**: {metrics.accuracy:.3f}

## Feature Highlights
{tokens_section}

## Intended Use
- High-volume SMS/email filtering.
- Human-in-the-loop moderation workflows.

## Limitations
- Uses token frequency only; sarcasm and context can be missed.
- Drift monitoring is recommended for new domains.
"""
