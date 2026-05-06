# French Sentiment Analysis with CamemBERT

Fine-tuning CamemBERT for sentiment classification (positive / negative) on French movie reviews.

## Project Structure

```
french-sentiment-camembert/
├── app.py              → Streamlit dashboard
├── src/                → reusable model loading & prediction logic
├── scripts/            → standalone preprocess.py and train.py
├── notebooks/          → exploration, preprocessing, fine-tuning
├── data/               → raw dataset
├── results/            → models, metrics, figures
└── docs/               → project documentation
```

## Dataset

[Allociné](https://huggingface.co/datasets/allocine) — 160k French movie reviews (pos/neg) via HuggingFace.

## Model

- Base model: `camembert-base`
- Fine-tuned with HuggingFace `Trainer`
- Evaluated with F1 score vs TF-IDF baseline

## Results

| Model | Test Accuracy | Test F1 |
|-------|--------------|---------|
| TF-IDF + LogisticRegression | 94.06% | 94.06% |
| CamemBERT fine-tuned | 97.18% | 97.18% |

CamemBERT outperforms the TF-IDF baseline by **+3.12% F1**.

## Reproduce from scratch

> **Note:** The trained model weights (`model.safetensors`, ~442 MB) are not committed to git because of their size. You must generate them locally by running the steps below before the app will use CamemBERT. Without weights it falls back to the TF-IDF baseline automatically.

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. (Optional) Pre-tokenize the dataset

Caches the tokenized Allociné dataset to `results/data/tokenized_datasets.pkl` so training starts faster. Skip this step and `train.py` will tokenize on the fly.

```bash
make preprocess
# or: python scripts/preprocess.py
```

### 3. Train both models

Trains the TF-IDF baseline and fine-tunes CamemBERT. Saves weights to `results/models/`.

```bash
make train
# or: python scripts/train.py
```

> **GPU strongly recommended.** Training takes ~2 hours on a T4 GPU (Google Colab). On CPU it will work but take many hours. The script auto-detects your hardware.

### 4. Run the app

```bash
make app
# or: streamlit run app.py
```

### 5. Run with Docker

```bash
make docker-build
make docker-run
# then open http://localhost:8501
```

> Docker requires the model weights to already exist in `results/models/` — run step 3 first.

### 6. Run tests

```bash
make test
# or: pytest tests/ -v
```

## Development

```bash
pip install -e ".[dev]"
make lint   # ruff check
make test   # pytest
```

## Progress

### Step 1 — Project Setup
- Initialized project structure (`data/`, `notebooks/`, `scripts/`, `src/`, `results/`, `docs/`)
- Added `pyproject.toml` with dependencies
- Created GitHub issues to track progress

### Step 2 — Data Exploration
- Loaded Allociné dataset via HuggingFace (160k train, 20k test)
- Dataset is perfectly balanced (50.4% pos / 49.6% neg)
- Average review length: 91 words — well within CamemBERT's 512 token limit
- Saved figures in `results/figures/`

### Step 3 — Preprocessing & Tokenization
- Tokenized all reviews with `camembert-base` tokenizer (vocab: 32 005 tokens)
- Chose MAX_LENGTH=256 — covers 100% of reviews (max observed: 502 tokens)
- Split: 160 000 train / 20 000 val / 20 000 test
- Created DataLoaders with batch size 16
- Saved tokenized datasets in `results/data/tokenized_datasets.pkl`

### Step 4 — TF-IDF Baseline
- Trained TF-IDF + Logistic Regression on 160k reviews
- Val Accuracy: 93.92% — Test Accuracy: 94.06%
- Val F1: 93.92% — Test F1: 94.06%
- No over-fitting: val ≈ test scores
- Saved metrics in `results/metrics/baseline_metrics.pkl`
- **Target for CamemBERT: F1 > 94%**

### Step 5 — Fine-tuning CamemBERT
- Fine-tuned `camembert-base` on 160k French movie reviews
- 3 epochs on T4 GPU (Google Colab) — training time ~2 hours
- Test Accuracy: 97.18% — Test F1: 97.18%
- Improvement over TF-IDF baseline: **+3.12%**
- Model saved in `results/models/camembert-sentiment/`

### Step 6 — Evaluation vs TF-IDF baseline

- Compared TF-IDF baseline vs CamemBERT fine-tuned on test set
- Generated model comparison bar chart and training curves
- Error analysis on baseline misclassified reviews

| Model | Test Accuracy | Test F1 |
|-------|--------------|---------|
| TF-IDF + LogisticRegression | 94.06% | 94.06% |
| CamemBERT fine-tuned | 97.18% | 97.18% |
| **Improvement** | **+3.12%** | **+3.12%** |

- Saved figures in `results/figures/`
- Saved final summary in `results/metrics/final_summary.pkl`

### Step 7 — Dashboard & Docker
- Built Streamlit dashboard for French sentiment analysis
- Supports CamemBERT (F1 97.18%) with TF-IDF fallback
- Dockerized — run with `docker run -p 8501:8501 french-sentiment-camembert`
