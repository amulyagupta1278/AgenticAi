"""
Data acquisition script for Multilingual Customer Support Tickets Dataset
Downloads the dataset from Kaggle using the Kaggle CLI
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Column names as they appear in the Kaggle CSV (all lowercase)
SUBJECT_COL = "subject"
BODY_COL = "body"
LABEL_COL = "queue"
LANGUAGE_COL = "language"

DATASET = "tobiasbueck/multilingual-customer-support-tickets"
FILENAME = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"


def download_customer_support_data():
    """
    Download Multilingual Customer Support Tickets dataset from Kaggle.

    Requires the Kaggle CLI to be installed and configured:
        pip install kaggle
        # Place kaggle.json at ~/.kaggle/kaggle.json (from kaggle.com/settings → API)

    If you prefer to download manually:
        1. Visit https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
        2. Click Download and unzip
        3. Place the CSV in data/raw/
        4. Update FILENAME above if the extracted name differs
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / FILENAME

    if dest.exists():
        print(f"Dataset already present: {dest}")
    else:
        print("Downloading Multilingual Customer Support Tickets dataset...")

        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("Error: Kaggle CLI not found. Install it with:  pip install kaggle")
            print("\nThen set up your API key:")
            print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
            print("  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
            print("  3. chmod 600 ~/.kaggle/kaggle.json")
            print("\nOr download manually:")
            print("  1. Visit: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets")
            print("  2. Click Download and unzip")
            print(f"  3. Place the CSV in: {RAW_DIR}/")
            sys.exit(1)

        if result.returncode != 0:
            print(f"Error downloading dataset: {result.stderr.strip()}")
            sys.exit(1)

        extracted = list(RAW_DIR.glob("*.csv"))
        if not dest.exists() and extracted:
            print(f"\nNote: expected '{FILENAME}' but found {[f.name for f in extracted]}.")
            print(f"Update FILENAME in download_data.py to match, then re-run preprocess.py.")
            dest = extracted[0]

        print(f"Dataset downloaded successfully!")

    # Load and inspect the data
    df = pd.read_csv(dest)

    print(f"Shape: {df.shape}")
    print(f"Saved to: {dest}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLabel distribution ({LABEL_COL}):")
    print(df[LABEL_COL].value_counts().to_string())
    print(f"\nLanguage distribution ({LANGUAGE_COL}):")
    print(df[LANGUAGE_COL].value_counts().to_string())
    print(f"\nMissing values:")
    print(df[[SUBJECT_COL, BODY_COL, LABEL_COL, LANGUAGE_COL]].isnull().sum())

    return df


def load_raw(filename=FILENAME):
    """Load the raw CSV. Raises FileNotFoundError with instructions if missing."""
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
            "Download from: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    download_customer_support_data()
