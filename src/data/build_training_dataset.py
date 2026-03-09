"""
Build the final training dataset from the cleaned CSV.

Applies EDA-driven rules on top of tickets_clean.csv to produce a
training-ready dataset with enriched text features.

Rules applied
-------------
1. Remove tickets with fewer than MIN_WORDS words (too little signal).
2. Drop label classes with fewer than MIN_CLASS_SAMPLES tickets.
3. Encode priority as a structured token appended to the text
   (e.g. "priority_high") so the model can use it without treating it
   as a separate numeric feature.

Output
------
data/processed/training_dataset.csv
  columns: text, label

Usage
-----
    python src/data/build_training_dataset.py
    python src/data/build_training_dataset.py --min-words 3 --min-class 30
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROCESSED_DIR   = ROOT / "data" / "processed"
INPUT_CSV       = PROCESSED_DIR / "tickets_clean.csv"
OUTPUT_CSV      = PROCESSED_DIR / "training_dataset.csv"

MIN_WORDS         = 5   # tickets shorter than this are noise
MIN_CLASS_SAMPLES = 50  # drop labels with too few examples to generalise

# Classes whose vocabulary overlaps so heavily with another class that the
# model cannot learn the distinction. Merge the source → target.
# Evidence from training results:
#   it_support recall=0.49 (half its tickets mis-routed to technical_support)
#   general_inquiry recall=0.60 with only ~235 samples (absorbed by customer_service)
CLASS_MERGES = {
    "it_support":      "technical_support",
    "general_inquiry": "customer_service",
}


def _add_priority_token(text: str, priority) -> str:
    """Append a structured priority token to the ticket text."""
    if pd.isna(priority) or str(priority).strip() == "":
        return text
    token = "priority_" + str(priority).strip().lower().replace(" ", "_")
    return f"{text} {token}"


def build(
    input_path=INPUT_CSV,
    output_path=OUTPUT_CSV,
    min_words: int = MIN_WORDS,
    min_class_samples: int = MIN_CLASS_SAMPLES,
    merge_classes: bool = True,
) -> pd.DataFrame:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {input_path}\n"
            "Run:  python src/data/preprocess.py"
        )

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path.name}")

    # ── 0. Merge overlapping classes ──────────────────────────────────────
    if merge_classes:
        for src, tgt in CLASS_MERGES.items():
            n = (df["label"] == src).sum()
            if n:
                df["label"] = df["label"].replace(src, tgt)
                print(f"Merged '{src}' → '{tgt}'  ({n:,} tickets)")

    # ── 1. Enrich text with priority token ────────────────────────────────
    if "priority" in df.columns:
        df["text"] = df.apply(
            lambda r: _add_priority_token(r["text"], r["priority"]), axis=1
        )
        print("Priority token appended to text.")
    else:
        print("No 'priority' column found — skipping priority enrichment.")

    # ── 2. Remove very short tickets ──────────────────────────────────────
    word_counts = df["text"].str.split().str.len()
    short_mask  = word_counts < min_words
    if short_mask.any():
        print(f"Removed {short_mask.sum():,} tickets with < {min_words} words.")
        df = df[~short_mask]

    # ── 3. Drop rare classes ──────────────────────────────────────────────
    class_counts = df["label"].value_counts()
    rare_classes = class_counts[class_counts < min_class_samples].index.tolist()
    if rare_classes:
        print(f"Dropping {len(rare_classes)} rare class(es) "
              f"(< {min_class_samples} samples): {rare_classes}")
        df = df[~df["label"].isin(rare_classes)]

    df = df[["text", "label"]].reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nFinal shape : {df.shape}")
    print(f"Classes     : {df['label'].nunique()}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset from cleaned tickets")
    parser.add_argument("--input",     default=str(INPUT_CSV),  help="Path to tickets_clean.csv")
    parser.add_argument("--output",    default=str(OUTPUT_CSV), help="Output CSV path")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS,
                        help=f"Min word count per ticket (default: {MIN_WORDS})")
    parser.add_argument("--min-class", type=int, default=MIN_CLASS_SAMPLES,
                        help=f"Min samples per class (default: {MIN_CLASS_SAMPLES})")
    parser.add_argument("--no-merge", action="store_true",
                        help="Disable class merging (keep all original classes)")
    args = parser.parse_args()
    build(args.input, args.output, args.min_words, args.min_class,
          merge_classes=not args.no_merge)
