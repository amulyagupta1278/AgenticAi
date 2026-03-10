"""
Generate keywords.json from the trained TF-IDF vectorizer.

Extracts the top N keywords per category based on mean TF-IDF score,
which helps the prevention agent understand what each category is about.

Usage:
    python scripts/generate_kb.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "training_dataset.csv"
KB_DIR = ROOT / "knowledge_base"


def main(top_n: int = 20):
    df = pd.read_csv(DATA_PATH)
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    feature_names = np.array(vectorizer.get_feature_names_out())
    categories = encoder.classes_.tolist()

    keywords_map = {}
    for cat in categories:
        cat_texts = df[df["label"] == cat]["text"]
        if cat_texts.empty:
            continue
        tfidf = vectorizer.transform(cat_texts)
        mean_scores = np.asarray(tfidf.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[::-1][:top_n]
        keywords_map[cat] = feature_names[top_idx].tolist()

    KB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = KB_DIR / "keywords.json"
    with open(out_path, "w") as f:
        json.dump(keywords_map, f, indent=2)

    print(f"Wrote keywords for {len(keywords_map)} categories to {out_path}")
    for cat, kws in keywords_map.items():
        print(f"  {cat}: {', '.join(kws[:5])} ...")


if __name__ == "__main__":
    main()
