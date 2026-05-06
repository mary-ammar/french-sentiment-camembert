#!/usr/bin/env python3
"""Download and tokenize the Allociné dataset. Run this before train.py."""
import pickle
from pathlib import Path

from datasets import load_dataset
from transformers import CamembertTokenizer

MODEL_NAME  = 'camembert-base'
MAX_LENGTH  = 256
OUTPUT_PATH = Path(__file__).parent.parent / 'results' / 'data' / 'tokenized_datasets.pkl'


def main():
    print("Loading Allociné dataset...")
    dataset = load_dataset('allocine')

    print("Loading tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch['review'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing (this may take a few minutes)...")
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(tokenized, f)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
