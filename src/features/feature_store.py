"""
Feature store for the ticket classifier.

Saves and loads pre-computed TF-IDF feature matrices so training runs
don't need to re-vectorize the data from scratch each time.

Layout on disk
--------------
data/features/
    <version>/
        X_train.npz          sparse TF-IDF matrix (train)
        X_val.npz            sparse TF-IDF matrix (val)
        X_test.npz           sparse TF-IDF matrix (test)
        y_train.npy          label array (train)
        y_val.npy            label array (val)
        y_test.npy           label array (test)
        tfidf_vectorizer.pkl fitted TF-IDF vectorizer
        label_encoder.pkl    fitted LabelEncoder
        meta.json            provenance: source file, split sizes, timestamp

Usage
-----
    # Build and save features
    python -m src.features.feature_store

    # Load in training code
    from src.features.feature_store import load_features
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, encoder = load_features()
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES_DIR = ROOT / "data" / "features"
PROCESSED_CSV = ROOT / "data" / "processed" / "training_dataset.csv"
DEFAULT_VERSION = "v1"

from src.features.build_features import get_vectorizer, split_train_val_test


def build_and_save(
    data_path=None,
    version=DEFAULT_VERSION,
    val_size=0.1,
    test_size=0.1,
    seed=42,
):
    """Vectorize the processed CSV and persist features to disk.

    Parameters
    ----------
    data_path : path-like, optional
        Path to tickets_clean.csv. Defaults to data/processed/tickets_clean.csv.
    version : str
        Sub-directory name under data/features/ (e.g. "v1", "v2").
    val_size, test_size : float
        Fraction of total data for validation and test sets.
    seed : int
        Random seed for reproducibility.
    """
    data_path = Path(data_path) if data_path else PROCESSED_CSV
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {data_path}\n"
            "Run:  python src/data/preprocess.py\n"
            "Then: python src/data/build_training_dataset.py"
        )

    out_dir = FEATURES_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows, {df['label'].nunique()} classes")

    print("Splitting ...")
    X_train, X_val, X_test, y_train, y_val, y_test, le = split_train_val_test(
        df, val_size=val_size, test_size=test_size, seed=seed
    )

    print("Vectorizing (TF-IDF) ...")
    vectorizer = get_vectorizer()
    X_tr = vectorizer.fit_transform(X_train)
    X_v  = vectorizer.transform(X_val)
    X_te = vectorizer.transform(X_test)
    print(f"  Feature matrix shape: {X_tr.shape}")

    print(f"Saving to {out_dir} ...")
    sp.save_npz(out_dir / "X_train.npz", X_tr)
    sp.save_npz(out_dir / "X_val.npz",   X_v)
    sp.save_npz(out_dir / "X_test.npz",  X_te)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "y_test.npy",  y_test)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")
    joblib.dump(le,         out_dir / "label_encoder.pkl")

    meta = {
        "version": version,
        "source_file": str(data_path),
        "n_total": len(df),
        "n_train": int(X_tr.shape[0]),
        "n_val":   int(X_v.shape[0]),
        "n_test":  int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "n_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFeature store saved  →  {out_dir}")
    print(f"  Classes : {meta['classes']}")
    print(f"  Features: {meta['n_features']:,}")
    return X_tr, X_v, X_te, y_train, y_val, y_test, vectorizer, le


def load_features(version=DEFAULT_VERSION):
    """Load pre-computed features from the feature store.

    Parameters
    ----------
    version : str
        Version directory to load from (default: "v1").

    Returns
    -------
    X_train, X_val, X_test : scipy sparse matrices
    y_train, y_val, y_test : numpy arrays
    vectorizer : fitted TfidfVectorizer
    encoder : fitted LabelEncoder
    """
    store_dir = FEATURES_DIR / version
    if not store_dir.exists():
        raise FileNotFoundError(
            f"Feature store not found: {store_dir}\n"
            "Run:  python -m src.features.feature_store"
        )

    X_train = sp.load_npz(store_dir / "X_train.npz")
    X_val   = sp.load_npz(store_dir / "X_val.npz")
    X_test  = sp.load_npz(store_dir / "X_test.npz")
    y_train = np.load(store_dir / "y_train.npy")
    y_val   = np.load(store_dir / "y_val.npy")
    y_test  = np.load(store_dir / "y_test.npy")
    vectorizer = joblib.load(store_dir / "tfidf_vectorizer.pkl")
    encoder    = joblib.load(store_dir / "label_encoder.pkl")

    with open(store_dir / "meta.json") as f:
        meta = json.load(f)

    print(f"Loaded feature store '{version}'  ({meta['n_train']:,} train / "
          f"{meta['n_val']:,} val / {meta['n_test']:,} test, "
          f"{meta['n_features']:,} features, {meta['n_classes']} classes)")

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, encoder


if __name__ == "__main__":
    build_and_save()
