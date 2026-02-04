from spamrectifier.features import FeatureConfig
from spamrectifier.model import NaiveBayesModel


def test_explain_returns_contributions():
    texts = ["Win a free prize", "Schedule the meeting"]
    labels = ["spam", "ham"]
    model = NaiveBayesModel.train(texts, labels, FeatureConfig())

    explanation = model.explain("Free prize for you", top_n=3)

    assert explanation["prediction"] in {"spam", "ham"}
    assert "probabilities" in explanation
    assert len(explanation["top_tokens"]) <= 3
