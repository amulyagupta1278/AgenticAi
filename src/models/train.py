import argparse
import shutil
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.build_features import save_artifacts, split, vectorize
from src.models.evaluate import compute_metrics, print_confusion_matrix

PROCESSED_DIR    = ROOT / "data" / "processed"
TRAINING_DATASET = PROCESSED_DIR / "training_dataset.csv"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

EXPERIMENT = "ticket-classifier"

# LinearSVC doesn't natively support predict_proba, so we wrap it
_CATALOG = {
    "logistic_regression": lambda: LogisticRegression(
        C=5.0, max_iter=1000, solver="lbfgs", class_weight="balanced", n_jobs=-1
    ),
    "linear_svc": lambda: CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, dual="auto", class_weight="balanced")
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
        class_weight="balanced_subsample", min_samples_leaf=2
    ),
}


def _setup_mlflow():
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT)


def train_one(model_name: str, df: pd.DataFrame) -> tuple:
    if model_name not in _CATALOG:
        raise ValueError(f"Unknown model '{model_name}'. Pick from: {list(_CATALOG)}")

    X_train, X_test, y_train, y_test, le = split(df)
    vectorizer, X_tr, X_te = vectorize(X_train, X_test)

    model = _CATALOG[model_name]()

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model": model_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": len(le.classes_),
            "tfidf_max_features": vectorizer.max_features,
            "ngram_range": str(vectorizer.ngram_range),
        })

        model.fit(X_tr, y_train)

        metrics = compute_metrics(model, X_te, y_test, class_names=le.classes_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

    return model, vectorizer, le, metrics


def train(model_name="linear_svc", data_path=None) -> tuple:
    if data_path is None:
        data_path = TRAINING_DATASET

    df = pd.read_csv(data_path)
    print(f"Dataset: {len(df):,} rows | {df['label'].nunique()} classes")

    _setup_mlflow()
    model, vectorizer, le, metrics = train_one(model_name, df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    save_artifacts(vectorizer, le)

    print(f"\nSaved: {model_path}")
    return model, vectorizer, le, metrics


def train_all(data_path=None) -> dict:
    if data_path is None:
        data_path = TRAINING_DATASET

    df = pd.read_csv(data_path)
    _setup_mlflow()

    results = {}
    for name in _CATALOG:
        print(f"\n{'─' * 45}")
        print(f"  Training: {name}")
        print(f"{'─' * 45}")
        _, vectorizer, le, metrics = train_one(name, df)
        joblib.dump(_, MODELS_DIR / f"{name}.pkl")
        results[name] = metrics

    # promote best to classifier.pkl
    best = max(results, key=lambda k: results[k]["f1"])
    shutil.copy(MODELS_DIR / f"{best}.pkl", MODELS_DIR / "classifier.pkl")
    save_artifacts(vectorizer, le)

    print(f"\nBest model: {best}  (F1={results[best]['f1']:.4f})")
    print(f"Promoted -> models/classifier.pkl")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ticket classifier")
    parser.add_argument("--model", default="linear_svc", choices=list(_CATALOG))
    parser.add_argument("--all", action="store_true", help="Train all models, keep best")
    parser.add_argument("--data", default=None, help="Path to cleaned CSV")
    args = parser.parse_args()

    if args.all:
        train_all(args.data)
    else:
        train(args.model, args.data)
