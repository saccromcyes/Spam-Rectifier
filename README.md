# SpamRectifier

[![CI](https://github.com/saccromcyes/Spam-Rectifier/actions/workflows/ci.yml/badge.svg)](https://github.com/saccromcyes/Spam-Rectifier/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-Render-5B3CC4)](https://saccromcyes-spam.onrender.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Three ways to catch SMS spam, judged on the same messages:

| Model | How it decides | How it explains itself |
|---|---|---|
| **Naive Bayes**, written from scratch | Word and bigram counts | Highlights each word by how strongly it points to spam or to a normal message |
| **Embeddings + logistic regression** | Sentence meaning (bge-small, ONNX) | Shows the most similar labelled training messages |
| **LLM, zero-shot** | Reads the message like a person | Writes a one-sentence reason |

**[Try it →](https://saccromcyes-spam.onrender.com)** Paste a message and see all three verdicts side by side, or open the Benchmark page. It runs on a free tier, so if nobody has used it for a while the first visit takes about a minute to wake up.

<!-- Screenshot added after the first deploy: docs/screenshot.png -->

## Benchmark

All models are trained on the same 80% of the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (de-duplicated, so no test message appears in training) and scored on the remaining 20%. The LLM is scored on a fixed 150-message sample to stay within the free tier's daily token limit; calls that fail are reported and left out of its metrics rather than counted as verdicts. The table is refreshed by GitHub Actions.

<!-- BENCHMARK:START -->
_Held-out test set of 1031 messages (128 spam), run 2026-11-14._

| Model | F1 (spam) | Precision | Recall | False positives | Latency / message |
|---|---|---|---|---|---|
| Naive Bayes (from scratch) | **0.908** | 0.860 | 0.961 | 2.2% of ham | 0.02 ms |
| Embeddings + logistic regression (BAAI/bge-small-en-v1.5) | **0.844** | 0.758 | 0.953 | 4.3% of ham | 16.40 ms |
<!-- BENCHMARK:END -->

**What to look at.** In spam filtering a false positive (a real message hidden as spam) costs more than a missed spam, so the false-positive column matters as much as F1. Latency and tokens per message decide what each model costs at a million messages a day: Naive Bayes runs in microseconds on a CPU, while the LLM needs a network call per message. The "Where the models disagree" list on the live Benchmark page shows each model's blind spots.

## How it's built

- **Data.** `dataset.py` loads the corpus, removes exact duplicates and makes a seeded, stratified split, so every run and every model sees the same test set.
- **Naive Bayes.** `model.py` is a dependency-free multinomial Naive Bayes with Laplace smoothing, optional bigrams and PII redaction. `token_evidence()` returns the log-odds each word contributes, which drives the word highlighting.
- **Embeddings classifier.** `embedding_model.py` embeds messages with bge-small through FastEmbed (ONNX, CPU) and trains a class-balanced logistic regression in NumPy. The training vectors are kept for nearest-neighbour explanations.
- **LLM classifier.** `llm_classifier.py` calls any OpenAI-compatible API (Groq by default) in JSON mode and treats the message as untrusted input.
- **Serving.** `web.py` runs the two local models while the LLM call is in flight, so the page waits only for the slowest one. If the LLM fails, the other two verdicts still come back.
- **Ops.** Both local models are trained during the Docker build. CI runs lint, tests and a free-tier fit check on every push; the Benchmark workflow retrains with the real embedding model and re-scores.

## Run locally

```bash
pip install -r requirements-dev.txt
export PYTHONPATH=src
python -m spamrectifier.benchmark          # train + benchmark (set LLM_API_KEY to include the LLM)
uvicorn spamrectifier.web:app --port 7860  # http://localhost:7860
EMBEDDER=hash pytest                       # offline tests
```

The original command-line tool is still here:

```bash
pip install -e .
spamrectifier train --data data/sample.csv --output model.json
spamrectifier explain --model model.json --text "Claim your reward now"
spamrectifier drift --model model.json --data data/sample.csv   # Jensen-Shannon drift report
```

## Deployment

**Deployment.** The app runs on Render's free tier (512 MB of memory, a tenth of a CPU core), defined in the [Blueprint](https://github.com/saccromcyes/saccromcyes.github.io/blob/main/render.yaml) that deploys all of my demos; every push to `main` that passes CI redeploys it. The CI job *Fits Render free tier* starts the production image under exactly those limits, sends a real request, and fails if the app runs out of memory or doesn't become healthy, so a change can't silently break the live demo.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/classify` | `{"text": "..."}` → each model's label, spam probability, explanation and latency, plus the majority verdict |
| `GET /api/benchmark` | The latest benchmark |
| `GET /health` | Liveness |

## Data

SMS Spam Collection v.1 by Tiago A. Almeida and José María Gómez Hidalgo, UCI Machine Learning Repository, licensed CC BY 4.0. 5,574 English SMS messages, 747 of them spam.
