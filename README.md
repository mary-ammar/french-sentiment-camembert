# French Sentiment Analysis with CamemBERT

Fine-tuning CamemBERT for sentiment classification (positive / negative) on French customer reviews.

## Project Structure

```
french-sentiment-camembert/
├── data/               → raw dataset
├── notebooks/          → exploration, preprocessing, fine-tuning
├── scripts/            → training and evaluation scripts
├── src/                → source code modules
├── results/            → metrics and figures
└── docs/               → project documentation
```
## Dataset
[Allociné](https://huggingface.co/datasets/allocine) — 160k French movie reviews (pos/neg) via HuggingFace.

## Model
- Base model: `camembert-base`
- Fine-tuned with HuggingFace `Trainer`
- Evaluated with F1 score vs TF-IDF baseline

## Setup
```bash
pip install -e .
```

## Results

| Model | Test Accuracy | Test F1 |
|-------|--------------|---------|
| TF-IDF + LogisticRegression | 94.06% | 94.06% |
| CamemBERT fine-tuned | 97.18% | 97.18% |

CamemBERT outperforms the TF-IDF baseline by **+3.12% F1**.

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
