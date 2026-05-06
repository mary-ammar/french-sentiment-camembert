import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.model import predict_tfidf, get_influential_words

_TRAIN = [
    "excellent magnifique superbe chef-oeuvre",
    "excellent magnifique bon film",
    "mauvais ennuyeux nul catastrophique",
    "ennuyeux mauvais navet décevant",
]
_LABELS = [1, 1, 0, 0]


@pytest.fixture
def tfidf_model():
    vec = TfidfVectorizer()
    X = vec.fit_transform(_TRAIN)
    clf = LogisticRegression(random_state=42).fit(X, _LABELS)
    return clf, vec


def test_predict_tfidf_positive(tfidf_model):
    clf, vec = tfidf_model
    label, confidence = predict_tfidf("excellent magnifique", clf, vec)
    assert label == 1
    assert 0.5 <= confidence <= 1.0


def test_predict_tfidf_negative(tfidf_model):
    clf, vec = tfidf_model
    label, confidence = predict_tfidf("mauvais ennuyeux", clf, vec)
    assert label == 0
    assert 0.5 <= confidence <= 1.0


def test_predict_tfidf_confidence_range(tfidf_model):
    clf, vec = tfidf_model
    _, confidence = predict_tfidf("bon film", clf, vec)
    assert 0.0 <= confidence <= 1.0


def test_get_influential_words_sorted_by_magnitude(tfidf_model):
    clf, vec = tfidf_model
    words = get_influential_words("excellent mauvais film", vec, clf)
    scores = [abs(s) for _, s in words]
    assert scores == sorted(scores, reverse=True)


def test_get_influential_words_respects_limit(tfidf_model):
    clf, vec = tfidf_model
    words = get_influential_words("excellent mauvais film bon", vec, clf, n=2)
    assert len(words) <= 2
