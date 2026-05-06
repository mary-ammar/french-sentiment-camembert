import streamlit as st
import pickle
import numpy as np
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="French Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f0f0f; }

h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem !important;
    color: #f5f0e8;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    margin-top: -0.5rem;
    margin-bottom: 2rem;
}

.model-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    color: #888;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 2rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.result-box {
    padding: 2rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    border: 1px solid;
}

.result-positive {
    background: #0a1f0a;
    border-color: #1a4d1a;
}

.result-negative {
    background: #1f0a0a;
    border-color: #4d1a1a;
}

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.result-positive .result-label { color: #4ade80; }
.result-negative .result-label { color: #f87171; }

.confidence-bar-container {
    background: #1a1a1a;
    border-radius: 6px;
    height: 8px;
    margin: 0.5rem 0 0.25rem;
    overflow: hidden;
}

.confidence-bar-positive {
    height: 100%;
    background: linear-gradient(90deg, #16a34a, #4ade80);
    border-radius: 6px;
    transition: width 0.5s ease;
}

.confidence-bar-negative {
    height: 100%;
    background: linear-gradient(90deg, #dc2626, #f87171);
    border-radius: 6px;
    transition: width 0.5s ease;
}

.confidence-text {
    font-size: 0.85rem;
    color: #666;
    font-weight: 500;
}

.word-section {
    margin-top: 2rem;
}

.word-section h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #f5f0e8;
    margin-bottom: 1rem;
}

.word-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.word-chip {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid;
}

.word-positive {
    background: #0a1f0a;
    border-color: #1a4d1a;
    color: #4ade80;
}

.word-negative {
    background: #1f0a0a;
    border-color: #4d1a1a;
    color: #f87171;
}

.examples-section {
    margin-top: 2.5rem;
    border-top: 1px solid #1a1a1a;
    padding-top: 2rem;
}

.example-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

.example-card:hover { border-color: #444; }

.example-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.label-pos { color: #4ade80; }
.label-neg { color: #f87171; }

.example-text { color: #aaa; font-size: 0.9rem; line-height: 1.4; }

.stTextArea textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #f5f0e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}

.stTextArea textarea:focus {
    border-color: #444 !important;
    box-shadow: none !important;
}

.stButton > button {
    background: #f5f0e8;
    color: #0f0f0f;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.footer {
    text-align: center;
    color: #333;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'results', 'models')
USE_CAMEMBERT = False

@st.cache_resource
def load_model():
    global USE_CAMEMBERT

    # Try CamemBERT first
    camembert_path = os.path.join(MODEL_DIR, 'camembert-sentiment')
    if os.path.exists(camembert_path) and os.path.exists(
            os.path.join(camembert_path, 'config.json')):
        try:
            from transformers import pipeline
            pipe = pipeline(
                'text-classification',
                model=camembert_path,
                tokenizer=camembert_path,
                device=-1
            )
            USE_CAMEMBERT = True
            return pipe, None, None
        except Exception:
            pass

    # Fallback: TF-IDF
    clf_path = os.path.join(MODEL_DIR, 'baseline_clf.pkl')
    vec_path = os.path.join(MODEL_DIR, 'baseline_vectorizer.pkl')
    if os.path.exists(clf_path) and os.path.exists(vec_path):
        clf = pickle.load(open(clf_path, 'rb'))
        vec = pickle.load(open(vec_path, 'rb'))
        return None, clf, vec

    return None, None, None


def predict_tfidf(text, clf, vec):
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    label = int(np.argmax(proba))
    confidence = float(proba[label])
    return label, confidence


def predict_camembert(text, pipe):
    result = pipe(text[:512])[0]
    label = 1 if result['label'] == 'Positive' else 0
    confidence = float(result['score'])
    return label, confidence


def get_influential_words(text, vec, clf, n=10):
    tokens = text.lower().split()
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]

    present = []
    for token in tokens:
        if token in vec.vocabulary_:
            idx = vec.vocabulary_[token]
            present.append((token, coef[idx]))

    present.sort(key=lambda x: abs(x[1]), reverse=True)
    return present[:n]


# ── Load ──────────────────────────────────────────────────────────────────────
pipe, clf, vec = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>French Sentiment<br>Analysis</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse the sentiment of French movie reviews</p>',
            unsafe_allow_html=True)

if USE_CAMEMBERT:
    st.markdown('<span class="model-badge">⚡ CamemBERT fine-tuned · F1 97.18%</span>',
                unsafe_allow_html=True)
else:
    st.markdown('<span class="model-badge">📊 TF-IDF Baseline · F1 94.06%</span>',
                unsafe_allow_html=True)

if pipe is None and clf is None:
    st.error("No model found. Make sure results/models/ contains the model files.")
    st.stop()

# ── Example reviews ───────────────────────────────────────────────────────────
EXAMPLES = [
    ("positive", "Un film absolument magnifique, une mise en scène sublime et des acteurs époustouflants. Je recommande vivement !"),
    ("negative", "Décevant et ennuyeux, l'histoire ne tient pas la route et les personnages sont creux. Une vraie déception."),
    ("positive", "Chef-d'œuvre du cinéma français ! Émotions intenses, scénario brillant, une expérience inoubliable."),
    ("negative", "Navet sans intérêt, deux heures perdues. Les dialogues sont nuls et la réalisation catastrophique."),
]

st.markdown('<div class="examples-section"><h3 style="font-family:\'DM Serif Display\',serif;color:#f5f0e8;font-size:1.1rem;margin-bottom:1rem;">Try an example</h3>',
            unsafe_allow_html=True)

cols = st.columns(2)
selected_example = None
for i, (sentiment, text) in enumerate(EXAMPLES):
    with cols[i % 2]:
        label_class = "label-pos" if sentiment == "positive" else "label-neg"
        label_text = "✦ Positive" if sentiment == "positive" else "✦ Negative"
        if st.button(f"{label_text} — {text[:60]}...", key=f"ex_{i}"):
            selected_example = text

st.markdown('</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
default_text = selected_example if selected_example else ""
review = st.text_area(
    "Your review",
    value=default_text,
    placeholder="Écrivez votre critique de film en français...",
    height=140,
    label_visibility="collapsed"
)

analyze = st.button("Analyse sentiment →")

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze and review.strip():
    with st.spinner("Analysing..."):
        if USE_CAMEMBERT:
            label, confidence = predict_camembert(review, pipe)
        else:
            label, confidence = predict_tfidf(review, clf, vec)

    is_positive = label == 1
    sentiment_text = "Positive" if is_positive else "Negative"
    emoji = "✦" if is_positive else "✗"
    box_class = "result-positive" if is_positive else "result-negative"
    bar_class = "confidence-bar-positive" if is_positive else "confidence-bar-negative"
    bar_width = int(confidence * 100)

    st.markdown(f"""
    <div class="result-box {box_class}">
        <div class="result-label">{emoji} {sentiment_text}</div>
        <div class="confidence-bar-container">
            <div class="{bar_class}" style="width:{bar_width}%"></div>
        </div>
        <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Influential words (TF-IDF only)
    if not USE_CAMEMBERT and clf is not None:
        words = get_influential_words(review, vec, clf)
        if words:
            pos_words = [(w, s) for w, s in words if s > 0]
            neg_words = [(w, s) for w, s in words if s < 0]

            st.markdown('<div class="word-section"><h3>Key words detected</h3>',
                        unsafe_allow_html=True)

            if pos_words:
                chips = " ".join([
                    f'<span class="word-chip word-positive">+{w}</span>'
                    for w, _ in pos_words[:6]
                ])
                st.markdown(
                    f'<p style="color:#555;font-size:0.8rem;margin-bottom:4px">POSITIVE SIGNALS</p>'
                    f'<div class="word-grid">{chips}</div>',
                    unsafe_allow_html=True
                )

            if neg_words:
                chips = " ".join([
                    f'<span class="word-chip word-negative">−{w}</span>'
                    for w, _ in neg_words[:6]
                ])
                st.markdown(
                    f'<p style="color:#555;font-size:0.8rem;margin-bottom:4px;margin-top:12px">NEGATIVE SIGNALS</p>'
                    f'<div class="word-grid">{chips}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

elif analyze and not review.strip():
    st.warning("Please enter a review first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    french-sentiment-camembert · CamemBERT fine-tuned on Allociné · Mary Ammar
</div>
""", unsafe_allow_html=True)
