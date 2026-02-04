from spamrectifier.features import FeatureConfig
from spamrectifier.model import NaiveBayesModel


def test_train_and_predict():
    texts = [
        "Win a free prize today",
        "Let's sync on the proposal",
        "Claim your exclusive reward",
        "Lunch at 1 pm?",
    ]
    labels = ["spam", "ham", "spam", "ham"]
    config = FeatureConfig(use_bigrams=True, min_token_length=2)
    model = NaiveBayesModel.train(texts, labels, config)

    assert model.predict("Free reward for you") == "spam"
    assert model.predict("Proposal review meeting") == "ham"
