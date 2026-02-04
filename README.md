# SpamRectifier

SpamRectifier is a compact, production-minded spam detection toolkit that focuses on clarity and deployability. It includes:

- A fast Naive Bayes classifier with configurable tokenization and bigram features.
- A CLI for training, evaluation, and prediction.
- JSON model persistence so you can ship a model artifact easily.
- Built-in explainability for top token contributions.
- Drift monitoring with Jensen-Shannon divergence on new data.
- A FastAPI service wrapper for production inference.
- Optional PII redaction (emails, URLs, numbers) during training.
- A lightweight data contract (CSV with `text` + `label`).
- Example data and tests.

## Why this stands out

- **Explicit feature pipeline**: Tokenization and bigrams are configurable, making decisions transparent.
- **No black boxes**: The model is fully inspectable, which helps in compliance-heavy environments.
- **Production-friendly**: Model artifacts are pure JSON; no pickling or opaque binaries.
- **Metrics included**: Precision, recall, F1, and accuracy are computed in one command.
- **Explainability & drift**: Token-level contributions and dataset drift reports are first-class.

## Project structure

```
.
├── data/
│   └── sample.csv
├── src/
│   └── spamrectifier/
│       ├── api.py
│       ├── cli.py
│       ├── data.py
│       ├── features.py
│       ├── monitoring.py
│       ├── model.py
│       └── reporting.py
└── tests/
    └── test_model.py
```

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Train a model

```bash
spamrectifier train --data data/sample.csv --output model.json
```

### 3) Generate a model card

```bash
spamrectifier train --data data/sample.csv --output model.json --model-card model-card.md
```

### 4) Evaluate

```bash
spamrectifier evaluate --data data/sample.csv --model model.json --positive-label spam
```

### 5) Predict

```bash
spamrectifier predict --model model.json --text "Free prizes for winners"
```

### 6) Explain a prediction

```bash
spamrectifier explain --model model.json --text "Claim your reward now"
```

### 7) Drift monitoring

```bash
spamrectifier drift --model model.json --data data/sample.csv
```

### 8) Run the API + preview UI

```bash
pip install -e ".[api]"
spamrectifier serve --model model.json --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/` to use the live preview UI.

### 9) Preview-only UI (no API dependencies)

```bash
spamrectifier preview --port 8001
```

Then open `http://localhost:8001/` for the static preview experience (served from Python).

## Data contract

Input CSV must include:

| Column | Description |
|--------|-------------|
| text   | The raw message or email content |
| label  | The target class (e.g., `spam`, `ham`) |

## Model details

- **Algorithm**: Multinomial Naive Bayes
- **Smoothing**: Laplace (+1)
- **Features**: Unigrams + optional bigrams
- **Tokenization**: Lowercased alphanumeric tokens with apostrophes/hyphens retained
- **PII Redaction**: Optional email/URL/number redaction for privacy-safe training

## Roadmap ideas

- Plug-in feature extractors (sender reputation, advanced URL parsing).
- Incremental training for high-volume pipelines.
- Vector embeddings to combine semantics with Naive Bayes interpretability.
