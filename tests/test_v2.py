import os

import pytest

os.environ.setdefault("EMBEDDER", "hash")

from spamrectifier.benchmark import metrics  # noqa: E402
from spamrectifier.dataset import load_sms, stratified_split  # noqa: E402
from spamrectifier.embedding_model import EmbeddingClassifier, HashEmbedder  # noqa: E402
from spamrectifier.features import FeatureConfig  # noqa: E402
from spamrectifier.model import NaiveBayesModel  # noqa: E402


@pytest.fixture(scope="module")
def split():
    return stratified_split(*load_sms())


def test_split_is_stratified_deterministic_and_leak_free(split):
    again = stratified_split(*load_sms())
    assert split.test_texts == again.test_texts
    assert not {t.lower() for t in split.train_texts} & {t.lower() for t in split.test_texts}
    train_ratio = split.train_labels.count("spam") / len(split.train_labels)
    test_ratio = split.test_labels.count("spam") / len(split.test_labels)
    assert abs(train_ratio - test_ratio) < 0.01


def test_naive_bayes_on_real_data(split):
    nb = NaiveBayesModel.train(split.train_texts, split.train_labels, FeatureConfig())
    preds = [nb.predict(t) for t in split.test_texts]
    assert metrics(split.test_labels, preds)["f1"] > 0.85


def test_token_evidence_is_discriminative(split):
    nb = NaiveBayesModel.train(split.train_texts, split.train_labels, FeatureConfig())
    evidence = {e["token"]: e["log_odds"] for e in nb.token_evidence("Claim your free prize now, see you at lunch")}
    assert evidence["prize"] > 1 and evidence["claim"] > 1
    assert evidence["lunch"] < 0


def test_embedding_classifier_roundtrip(tmp_path, split):
    emb = EmbeddingClassifier.train(HashEmbedder(), split.train_texts[:800], split.train_labels[:800], epochs=150)
    path = tmp_path / "m.npz"
    emb.save(path)
    loaded = EmbeddingClassifier.load(path, HashEmbedder())
    text = "WINNER!! You have won a guaranteed cash prize, call now"
    assert abs(loaded.predict_proba(text)["spam"] - emb.predict_proba(text)["spam"]) < 1e-3
    assert len(loaded.neighbours(text, k=3)) == 3


def test_metrics():
    m = metrics(["spam", "spam", "ham", "ham"], ["spam", "ham", "spam", "ham"])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["false_positive_rate"] == 0.5
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from spamrectifier.web import create_app

    with TestClient(create_app()) as c:
        yield c


def test_classify_without_llm(client):
    body = client.post("/api/classify", json={"text": "URGENT! You have won a £2000 prize. Call 09061701461 now"}).json()
    assert [r["key"] for r in body["results"]] == ["naive_bayes", "embeddings"]
    assert body["results"][0]["label"] == "spam"
    assert body["results"][0]["evidence"]
    assert len(body["results"][1]["neighbours"]) == 3


def test_classify_with_llm_and_with_llm_failure(client):
    class FakeLLM:
        model = "fake"

        def __init__(self, fail=False):
            self.fail = fail

        def classify(self, text):
            if self.fail:
                raise RuntimeError("down")
            return {"label": "spam", "spam_probability": 0.9, "reason": "prize claim", "latency_ms": 5.0, "tokens": 90}

    client.app.state.models.llm = FakeLLM()
    body = client.post("/api/classify", json={"text": "Free entry to win cash"}).json()
    assert body["results"][2]["reason"] == "prize claim"
    assert body["agreement"].endswith("of 3")

    client.app.state.models.llm = FakeLLM(fail=True)
    body = client.post("/api/classify", json={"text": "Free entry to win cash"}).json()
    assert "error" in body["results"][2]
    assert body["agreement"].endswith("of 2")
    client.app.state.models.llm = None


def test_validation_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/api/classify", json={"text": ""}).status_code == 422
