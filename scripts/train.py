#!/usr/bin/env python3
"""Train CamemBERT and TF-IDF baseline end-to-end on Allociné.

Usage:
    python scripts/preprocess.py   # optional — train.py will tokenize on the fly if skipped
    python scripts/train.py
"""
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    CamembertForSequenceClassification,
    CamembertTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate

MODEL_NAME    = 'camembert-base'
MAX_LENGTH    = 256
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS  = 500
WEIGHT_DECAY  = 0.01

ROOT          = Path(__file__).parent.parent
TOKENIZED_PKL = ROOT / 'results' / 'data' / 'tokenized_datasets.pkl'
MODEL_OUT     = ROOT / 'results' / 'models' / 'camembert-sentiment'
BASELINE_CLF  = ROOT / 'results' / 'models' / 'baseline_clf.pkl'
BASELINE_VEC  = ROOT / 'results' / 'models' / 'baseline_vectorizer.pkl'
CHECKPOINTS   = ROOT / 'results' / 'checkpoints'

accuracy_metric = evaluate.load('accuracy')
f1_metric       = evaluate.load('f1')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {'accuracy': acc['accuracy'], 'f1': f1['f1']}


def train_baseline(dataset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    print("\n--- Training TF-IDF baseline ---")
    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents='unicode',
        lowercase=True,
    )
    X_train = vec.fit_transform(dataset['train']['review'])
    X_test  = vec.transform(dataset['test']['review'])
    y_train = dataset['train']['label']
    y_test  = dataset['test']['label']

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='saga', random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    f1    = f1_score(y_test, preds, average='weighted')
    print(f"Baseline Test F1: {f1:.4f}")

    BASELINE_CLF.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_CLF, 'wb') as f:
        pickle.dump(clf, f)
    with open(BASELINE_VEC, 'wb') as f:
        pickle.dump(vec, f)
    print(f"Saved to {BASELINE_CLF.parent}")


def train_camembert():
    print("\n--- Training CamemBERT ---")
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)

    if TOKENIZED_PKL.exists():
        print(f"Loading tokenized data from {TOKENIZED_PKL}")
        with open(TOKENIZED_PKL, 'rb') as f:
            tokenized = pickle.load(f)
    else:
        print("Tokenized data not found — tokenizing now (run preprocess.py to cache this)...")
        dataset = load_dataset('allocine')

        def tokenize(batch):
            return tokenizer(
                batch['review'],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
            )

        tokenized = dataset.map(tokenize, batched=True)
        tokenized = tokenized.rename_column('label', 'labels')
        tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = CamembertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: 'Negative', 1: 'Positive'},
        label2id={'Negative': 0, 'Positive': 1},
    )

    args = TrainingArguments(
        output_dir=str(CHECKPOINTS),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics,
    )

    print("Training (~2 hours on a T4 GPU)...")
    trainer.train()

    results = trainer.evaluate(tokenized['test'])
    print(f"\nTest Accuracy : {results['eval_accuracy']:.4f}")
    print(f"Test F1       : {results['eval_f1']:.4f}")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"Model saved to {MODEL_OUT}")


def main():
    dataset = load_dataset('allocine')
    train_baseline(dataset)
    train_camembert()


if __name__ == '__main__':
    main()
