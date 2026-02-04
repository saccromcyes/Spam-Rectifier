from spamrectifier.features import FeatureConfig
from spamrectifier.model import NaiveBayesModel
from spamrectifier.monitoring import drift_report


def test_drift_report_shape():
    texts = ["Free prize now", "Let's meet tomorrow"]
    labels = ["spam", "ham"]
    model = NaiveBayesModel.train(texts, labels, FeatureConfig())

    report = drift_report(model, texts, top_n=5)

    assert report["data_size"] == 2
    assert report["js_divergence"] >= 0.0
    assert len(report["top_shifted_tokens"]) <= 5
