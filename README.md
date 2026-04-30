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