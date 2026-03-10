"""
Data acquisition — downloads the customer support tickets dataset from
HuggingFace Hub and saves it as a CSV for the rest of the pipeline.

Dataset: Tobi-Bueck/customer-support-tickets (61.8K tickets)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

SUBJECT_COL = "subject"
BODY_COL = "body"
LABEL_COL = "queue"
LANGUAGE_COL = "language"

HF_DATASET = "Tobi-Bueck/customer-support-tickets"
FILENAME = "customer_support_tickets.csv"


def download_customer_support_data():
    """
    Pull the dataset from HuggingFace and save to data/raw/.

    Requires: pip install datasets
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / FILENAME

    if dest.exists():
        print(f"Dataset already present: {dest}")
        df = pd.read_csv(dest)
    else:
        print(f"Downloading from HuggingFace: {HF_DATASET} ...")

        try:
            from datasets import load_dataset
        except ImportError:
            print("Error: 'datasets' package not installed.")
            print("  pip install datasets")
            sys.exit(1)

        ds = load_dataset(HF_DATASET, split="train")
        df = ds.to_pandas()
        df.to_csv(dest, index=False)
        print(f"Saved {len(df):,} rows to {dest}")

    _print_stats(df)
    return df


def _print_stats(df):
    print(f"\nShape: {df.shape}")
    print(f"\nQueue distribution:")
    print(df[LABEL_COL].value_counts().to_string())
    if "type" in df.columns:
        print(f"\nType distribution:")
        print(df["type"].value_counts().to_string())
    print(f"\nLanguage distribution:")
    print(df[LANGUAGE_COL].value_counts().to_string())
    print(f"\nMissing values:")
    cols = [c for c in [SUBJECT_COL, BODY_COL, LABEL_COL, LANGUAGE_COL] if c in df.columns]
    print(df[cols].isnull().sum())


def load_raw(filename=FILENAME):
    """Load the raw CSV. Raises FileNotFoundError with help text if missing."""
    path = RAW_DIR / filename
    if not path.exists():
        existing = list(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
        hint = (
            f"\nFound in data/raw/: {[f.name for f in existing]}"
            if existing
            else "\ndata/raw/ is empty — run download_data.py first."
        )
        raise FileNotFoundError(
            f"Dataset not found: {path}{hint}\n"
            f"Run:  python src/data/download_data.py"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    download_customer_support_data()
