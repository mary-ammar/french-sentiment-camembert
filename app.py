import streamlit as st
from src.model import (
    load_model as _load_model,
    predict_tfidf,
    predict_camembert,
    get_influential_words,
)

st.set_page_config(page_title="French Sentiment Analysis", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1 { font-family: 'DM Serif Display', serif; font-size: 2.8rem !important; color: #f5f0e8; letter-spacing: -0.02em; line-height: 1.1; }
.subtitle { color: #888; font-size: 1rem; font-weight: 300; margin-top: -0.5rem; margin-bottom: 2rem; }
.model-badge { display: inline-block; background: #1a1a1a; border: 1px solid #333; color: #888; font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; margin-bottom: 2rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
.result-box { padding: 2rem; border-radius: 12px; margin: 1.5rem 0; border: 1px solid; }
.result-positive { background: #0a1f0a; border-color: #1a4d1a; }
.result-negative { background: #1f0a0a; border-color: #4d1a1a; }
.result-label { font-family: 'DM Serif Display', serif; font-size: 2rem; margin-bottom: 0.5rem; }
.result-positive .result-label { color: #4ade80; }
.result-negative .result-label { color: #f87171; }
.confidence-bar-container { background: #1a1a1a; border-radius: 6px; height: 8px; margin: 0.5rem 0 0.25rem; overflow: hidden; }
.confidence-bar-positive { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 6px; }
.confidence-bar-negative { height: 100%; background: linear-gradient(90deg, #dc2626, #f87171); border-radius: 6px; }
.confidence-text { font-size: 0.85rem; color: #666; font-weight: 500; }
.word-chip { padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; border: 1px solid; display: inline-block; margin: 3px; }
.word-positive { background: #0a1f0a; border-color: #1a4d1a; color: #4ade80; }
.word-negative { background: #1f0a0a; border-color: #4d1a1a; color: #f87171; }
.footer { text-align: center; color: #333; font-size: 0.8rem; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1a1a1a; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return _load_model()


model_type, pipe, clf, vec = load_model()

st.markdown("<h1>French Sentiment<br>Analysis</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse the sentiment of French movie reviews</p>', unsafe_allow_html=True)

if model_type == 'camembert':
    st.markdown('<span class="model-badge">⚡ CamemBERT fine-tuned · F1 97.18%</span>', unsafe_allow_html=True)
elif model_type == 'tfidf':
    st.markdown('<span class="model-badge">📊 TF-IDF Baseline · F1 94.06%</span>', unsafe_allow_html=True)
else:
    st.error("No model found. Run `make train` to generate model weights, then restart the app.")
    st.stop()

EXAMPLES = [
    ("positive", "Un film absolument magnifique, une mise en scène sublime et des acteurs époustouflants. Je recommande vivement !"),
    ("negative", "Décevant et ennuyeux, l'histoire ne tient pas la route et les personnages sont creux. Une vraie déception."),
    ("positive", "Chef-d'œuvre du cinéma français ! Émotions intenses, scénario brillant, une expérience inoubliable."),
    ("negative", "Navet sans intérêt, deux heures perdues. Les dialogues sont nuls et la réalisation catastrophique."),
]

st.markdown("### Try an example")
cols = st.columns(2)
selected_example = None
for i, (sentiment, text) in enumerate(EXAMPLES):
    with cols[i % 2]:
        icon = "✦" if sentiment == "positive" else "✗"
        if st.button(f"{icon} {text[:55]}...", key=f"ex_{i}"):
            selected_example = text

st.markdown("<br>", unsafe_allow_html=True)
review = st.text_area("Your review", value=selected_example or "", placeholder="Écrivez votre critique de film en français...", height=140, label_visibility="collapsed")
analyze = st.button("Analyse sentiment →")

if analyze:
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            with st.spinner("Analysing..."):
                if model_type == 'camembert':
                    label, confidence = predict_camembert(review, pipe)
                else:
                    label, confidence = predict_tfidf(review, clf, vec)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        is_positive = label == 1
        sentiment   = "Positive" if is_positive else "Negative"
        emoji       = "✦" if is_positive else "✗"
        box_class   = "result-positive" if is_positive else "result-negative"
        bar_class   = "confidence-bar-positive" if is_positive else "confidence-bar-negative"
        bar_width   = int(confidence * 100)

        st.markdown(f"""
        <div class="result-box {box_class}">
            <div class="result-label">{emoji} {sentiment}</div>
            <div class="confidence-bar-container">
                <div class="{bar_class}" style="width:{bar_width}%"></div>
            </div>
            <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if model_type == 'tfidf':
            words = get_influential_words(review, vec, clf)
            if words:
                pos_words = [(w, s) for w, s in words if s > 0]
                neg_words = [(w, s) for w, s in words if s < 0]
                st.markdown("**Key words detected:**")
                if pos_words:
                    chips = " ".join([f'<span class="word-chip word-positive">+{w}</span>' for w, _ in pos_words[:6]])
                    st.markdown(f'<p style="color:#555;font-size:0.8rem">POSITIVE SIGNALS</p>{chips}', unsafe_allow_html=True)
                if neg_words:
                    chips = " ".join([f'<span class="word-chip word-negative">−{w}</span>' for w, _ in neg_words[:6]])
                    st.markdown(f'<p style="color:#555;font-size:0.8rem;margin-top:12px">NEGATIVE SIGNALS</p>{chips}', unsafe_allow_html=True)

st.markdown('<div class="footer">french-sentiment-camembert · CamemBERT fine-tuned on Allociné · Mary Ammar</div>', unsafe_allow_html=True)
