"""
Dataset Separability Baseline (PriBench Section 4)

Evaluates linear separability between privacy-seeking (label=1) and benign (label=0)
prompts using TF-IDF feature extraction and Logistic Regression.

Usage:
    python utils/separability_baseline.py
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "data/pribench_dataset.csv"


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset non trovato in {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["prompt"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print("=== SEPARABILITY BASELINE RESULTS (TF-IDF + LOGISTIC REGRESSION) ===")
    print(f"Test Accuracy: {acc * 100.0:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Benign (0)", "Privacy (1)"]))


if __name__ == "__main__":
    main()