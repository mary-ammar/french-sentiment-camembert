import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

MAX_LENGTH = 256

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMEMBERT_PATH = os.path.join(BASE_DIR, 'results', 'models', 'camembert-sentiment')
BASELINE_CLF   = os.path.join(BASE_DIR, 'results', 'models', 'baseline_clf.pkl')
BASELINE_VEC   = os.path.join(BASE_DIR, 'results', 'models', 'baseline_vectorizer.pkl')


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model():
    if os.path.exists(os.path.join(CAMEMBERT_PATH, 'config.json')):
        try:
            from transformers import pipeline
            device = 0 if _cuda_available() else -1
            pipe = pipeline(
                'text-classification',
                model=CAMEMBERT_PATH,
                tokenizer=CAMEMBERT_PATH,
                device=device,
            )
            return 'camembert', pipe, None, None
        except Exception as e:
            logger.warning("CamemBERT failed to load: %s. Falling back to TF-IDF.", e)
    if os.path.exists(BASELINE_CLF) and os.path.exists(BASELINE_VEC):
        with open(BASELINE_CLF, 'rb') as f:
            clf = pickle.load(f)
        with open(BASELINE_VEC, 'rb') as f:
            vec = pickle.load(f)
        return 'tfidf', None, clf, vec
    return 'none', None, None, None


def predict_tfidf(text, clf, vec):
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    label = int(np.argmax(proba))
    return label, float(proba[label])


def predict_camembert(text, pipe):
    result = pipe(text, truncation=True, max_length=MAX_LENGTH)[0]
    label = 1 if result['label'] == 'Positive' else 0
    return label, float(result['score'])


def get_influential_words(text, vec, clf, n=10):
    coef = clf.coef_[0]
    present = []
    for token in text.lower().split():
        if token in vec.vocabulary_:
            idx = vec.vocabulary_[token]
            present.append((token, coef[idx]))
    present.sort(key=lambda x: abs(x[1]), reverse=True)
    return present[:n]
