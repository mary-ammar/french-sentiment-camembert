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
*To be updated after training.*

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

### Step 4 — Fine-tuning CamemBERT
### Step 5 — Evaluation vs TF-IDF baseline
### Step 6 — Documentation