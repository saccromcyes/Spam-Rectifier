"""Command line interface for SpamRectifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socketserver

from .data import load_csv
from .features import FeatureConfig
from .monitoring import drift_report
from .model import NaiveBayesModel
from .preview import PreviewRequestHandler
from .reporting import build_model_card, classification_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spamrectifier",
        description="Train, evaluate, and run a spam classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from CSV data")
    train_parser.add_argument("--data", required=True, help="Path to CSV with text,label")
    train_parser.add_argument("--output", required=True, help="Path to write model JSON")
    train_parser.add_argument("--no-bigrams", action="store_true", help="Disable bigram features")
    train_parser.add_argument(
        "--min-token-length", type=int, default=2, help="Minimum token length"
    )
    train_parser.add_argument(
        "--no-email-redact", action="store_true", help="Disable email redaction"
    )
    train_parser.add_argument(
        "--no-url-redact", action="store_true", help="Disable URL redaction"
    )
    train_parser.add_argument(
        "--redact-numbers", action="store_true", help="Replace numbers with <number>"
    )
    train_parser.add_argument(
        "--model-card", help="Optional path to write a model card markdown file"
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    eval_parser.add_argument("--data", required=True, help="Path to CSV with text,label")
    eval_parser.add_argument("--model", required=True, help="Path to model JSON")
    eval_parser.add_argument(
        "--positive-label", default="spam", help="Label treated as positive"
    )

    predict_parser = subparsers.add_parser("predict", help="Predict from a saved model")
    predict_parser.add_argument("--model", required=True, help="Path to model JSON")
    predict_parser.add_argument("--text", required=True, help="Text to classify")

    explain_parser = subparsers.add_parser("explain", help="Explain a prediction")
    explain_parser.add_argument("--model", required=True, help="Path to model JSON")
    explain_parser.add_argument("--text", required=True, help="Text to explain")
    explain_parser.add_argument(
        "--top-n", type=int, default=8, help="Number of tokens to highlight"
    )

    drift_parser = subparsers.add_parser("drift", help="Analyze drift on new data")
    drift_parser.add_argument("--model", required=True, help="Path to model JSON")
    drift_parser.add_argument("--data", required=True, help="Path to CSV with text,label")
    drift_parser.add_argument(
        "--top-n", type=int, default=10, help="Number of shifted tokens to show"
    )

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI inference service")
    serve_parser.add_argument("--model", required=True, help="Path to model JSON")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")

    preview_parser = subparsers.add_parser("preview", help="Serve the preview UI only")
    preview_parser.add_argument("--port", type=int, default=8001, help="Bind port")

    return parser.parse_args()


def _handle_train(args: argparse.Namespace) -> None:
    dataset = load_csv(args.data)
    config = FeatureConfig(
        use_bigrams=not args.no_bigrams,
        min_token_length=args.min_token_length,
        redact_emails=not args.no_email_redact,
        redact_urls=not args.no_url_redact,
        redact_numbers=args.redact_numbers,
    )
    model = NaiveBayesModel.train(dataset.texts, dataset.labels, config)
    model.save(args.output)
    print(f"Model saved to {args.output}")
    if args.model_card:
        predictions = [model.predict(text) for text in dataset.texts]
        metrics = classification_report(dataset.labels, predictions, "spam")
        card = build_model_card(
            model_name="SpamRectifier",
            version="1.0",
            labels=model.labels,
            metrics=metrics,
            dataset_size=model.dataset_size,
            trained_at=model.trained_at,
            positive_label="spam",
            top_tokens={label: model.top_tokens(label) for label in model.labels},
        )
        Path(args.model_card).write_text(card)
        print(f"Model card written to {args.model_card}")


def _handle_evaluate(args: argparse.Namespace) -> None:
    dataset = load_csv(args.data)
    model = NaiveBayesModel.load(args.model)
    predictions = [model.predict(text) for text in dataset.texts]
    metrics = classification_report(dataset.labels, predictions, args.positive_label)
    print(json.dumps(metrics.__dict__, indent=2))


def _handle_predict(args: argparse.Namespace) -> None:
    model = NaiveBayesModel.load(args.model)
    probabilities = model.predict_proba(args.text)
    result = {
        "prediction": model.predict(args.text),
        "probabilities": probabilities,
    }
    print(json.dumps(result, indent=2))


def _handle_explain(args: argparse.Namespace) -> None:
    model = NaiveBayesModel.load(args.model)
    explanation = model.explain(args.text, top_n=args.top_n)
    print(json.dumps(explanation, indent=2))


def _handle_drift(args: argparse.Namespace) -> None:
    dataset = load_csv(args.data)
    model = NaiveBayesModel.load(args.model)
    report = drift_report(model, dataset.texts, top_n=args.top_n)
    print(json.dumps(report, indent=2))


def _handle_serve(args: argparse.Namespace) -> None:
    import importlib.util

    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
        raise SystemExit(
            "FastAPI dependencies not installed. "
            "Run `pip install .[api]` to enable the service."
        )
    from .api import create_app
    import uvicorn
    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)

def _handle_preview(args: argparse.Namespace) -> None:
    with socketserver.TCPServer(("", args.port), PreviewRequestHandler) as httpd:
        print(f"Preview UI running at http://localhost:{args.port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Preview server stopped.")


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        _handle_train(args)
    elif args.command == "evaluate":
        _handle_evaluate(args)
    elif args.command == "predict":
        _handle_predict(args)
    elif args.command == "explain":
        _handle_explain(args)
    elif args.command == "drift":
        _handle_drift(args)
    elif args.command == "serve":
        _handle_serve(args)
    elif args.command == "preview":
        _handle_preview(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
