"""Train both local models and benchmark all three on the same held-out test set.

    python -m spamrectifier.benchmark                 # NB + embeddings (+ LLM if LLM_API_KEY is set)
    python -m spamrectifier.benchmark --build-only    # train and save artifacts, no evaluation
    python -m spamrectifier.benchmark --llm-sample 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone

from .dataset import ROOT, load_sms, stratified_split
from .embedding_model import EmbeddingClassifier, build_embedder
from .features import FeatureConfig
from .model import NaiveBayesModel

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results" / "benchmark.json"
README = ROOT / "README.md"


def metrics(truth: list[str], pred: list[str]) -> dict:
    tp = sum(t == "spam" and p == "spam" for t, p in zip(truth, pred))
    fp = sum(t == "ham" and p == "spam" for t, p in zip(truth, pred))
    fn = sum(t == "spam" and p == "ham" for t, p in zip(truth, pred))
    tn = sum(t == "ham" and p == "ham" for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": round((tp + tn) / len(truth), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": len(truth),
    }


def build(embedder_kind: str = "fastembed", cache_dir: str | None = None):
    texts, labels = load_sms()
    split = stratified_split(texts, labels)
    ARTIFACTS.mkdir(exist_ok=True)
    nb = NaiveBayesModel.train(split.train_texts, split.train_labels, FeatureConfig(use_bigrams=True, redact_numbers=False))
    nb.save(ARTIFACTS / "naive_bayes.json")
    embedder = build_embedder(embedder_kind, cache_dir)
    emb = EmbeddingClassifier.train(embedder, split.train_texts, split.train_labels)
    emb.save(ARTIFACTS / "embedding_model.npz")
    return split, nb, emb


def timed(fn, texts: list[str]) -> tuple[list[str], float]:
    start = time.perf_counter()
    out = fn(texts)
    return out, (time.perf_counter() - start) * 1000 / max(1, len(texts))


def render_markdown(report: dict) -> str:
    rows = report["models"]
    lines = [
        f"_Held-out test set of {report['test_size']} messages ({report['test_spam']} spam), run {report['run_at'][:10]}._",
        "",
        "| Model | F1 (spam) | Precision | Recall | False positives | Latency / message |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        sample = f" (sample of {m['n']})" if m["n"] != report["test_size"] else ""
        lines.append(
            f"| {r['label']}{sample} | **{m['f1']:.3f}** | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['false_positive_rate']:.1%} of ham | {r['latency_ms']:.2f} ms |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--embedder", default=os.getenv("EMBEDDER", "fastembed"))
    parser.add_argument("--llm-sample", type=int, default=300)
    parser.add_argument("--pause", type=float, default=0.3)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    split, nb, emb = build(args.embedder, os.getenv("MODEL_CACHE_DIR"))
    print(f"Trained on {len(split.train_texts)} messages; artifacts in {ARTIFACTS}")
    if args.build_only:
        return 0

    truth = split.test_labels
    nb_pred, nb_ms = timed(lambda xs: [nb.predict(x) for x in xs], split.test_texts)
    emb_pred, emb_ms = timed(emb.predict_many, split.test_texts)
    models = [
        {"key": "naive_bayes", "label": "Naive Bayes (from scratch)", "metrics": metrics(truth, nb_pred), "latency_ms": round(nb_ms, 3)},
        {"key": "embeddings", "label": f"Embeddings + logistic regression ({emb.embedder.name})", "metrics": metrics(truth, emb_pred), "latency_ms": round(emb_ms, 3)},
    ]
    predictions = {"naive_bayes": nb_pred, "embeddings": emb_pred}

    from .llm_classifier import LLMClassifier

    llm = LLMClassifier.from_env()
    if llm:
        n = min(args.llm_sample, len(split.test_texts))
        llm_pred, latencies, tokens = [], [], []
        for text in split.test_texts[:n]:
            try:
                out = llm.classify(text)
                llm_pred.append(out["label"])
                latencies.append(out["latency_ms"])
                tokens.append(out["tokens"] or 0)
            except Exception as exc:  # count provider failures as misses rather than crashing
                print(f"LLM error: {exc}")
                llm_pred.append("ham")
            time.sleep(args.pause)
        models.append({
            "key": "llm", "label": f"LLM zero-shot ({llm.model})", "metrics": metrics(truth[:n], llm_pred),
            "latency_ms": round(statistics.median(latencies), 1) if latencies else None,
            "tokens_per_message": round(statistics.mean(tokens)) if tokens else None,
        })
        predictions["llm"] = llm_pred + [None] * (len(truth) - n)

    for row in models:
        print(f"{row['label']:<60} F1={row['metrics']['f1']:.3f}  FPR={row['metrics']['false_positive_rate']:.3f}  {row['latency_ms']} ms")

    disagreements = []
    for i, text in enumerate(split.test_texts):
        votes = {k: v[i] for k, v in predictions.items() if v[i] is not None}
        if len(set(votes.values())) > 1:
            disagreements.append({"text": text, "truth": truth[i], "votes": votes})
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "SMS Spam Collection v.1 (UCI, CC BY 4.0), de-duplicated",
        "train_size": len(split.train_texts),
        "test_size": len(truth),
        "test_spam": truth.count("spam"),
        "models": models,
        "disagreements": disagreements[:12],
    }
    if not args.no_write:
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(report, indent=2))
        if README.exists():
            pattern = re.compile(r"(<!-- BENCHMARK:START -->\n).*?(<!-- BENCHMARK:END -->)", flags=re.DOTALL)
            README.write_text(pattern.sub(lambda m: m.group(1) + render_markdown(report) + m.group(2), README.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
