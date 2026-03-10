import joblib
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def get_vectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=5,    # ignore terms appearing in fewer than 5 documents (noise filter)
    max_df=0.8,  # ignore terms appearing in >80% of documents (stop-word effect)
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"\b[a-z][a-z]+\b",  # skip single chars and numbers
    )


def split(df: pd.DataFrame, test_size=0.2, seed=42):
    """Train / test split (80/20 by default, stratified)."""
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, le


def split_train_val_test(df: pd.DataFrame, val_size=0.1, test_size=0.1, seed=42):
    """Three-way stratified split: train / validation / test.

    Default proportions: 80% train, 10% val, 10% test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    """
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    # first cut off the test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        df["text"], y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # split the remainder into train + val
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_val,
        random_state=seed,
        stratify=y_tmp,
    )

    print(f"Split sizes — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def vectorize(X_train, X_test, vectorizer=None):
    if vectorizer is None:
        vectorizer = get_vectorizer()
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    return vectorizer, X_tr, X_te


def save_artifacts(vectorizer, label_encoder, prefix=""):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vec_path = MODELS_DIR / f"{prefix}tfidf_vectorizer.pkl"
    enc_path = MODELS_DIR / f"{prefix}label_encoder.pkl"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(label_encoder, enc_path)
    return vec_path, enc_path


def load_vectorizer(path=None) -> TfidfVectorizer:
    path = path or MODELS_DIR / "tfidf_vectorizer.pkl"
    return joblib.load(path)


def load_encoder(path=None) -> LabelEncoder:
    path = path or MODELS_DIR / "label_encoder.pkl"
    return joblib.load(path)
