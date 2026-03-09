"""
Clean and preprocess the multilingual customer support tickets CSV.

Reads from  data/raw/<filename>
Writes to   data/processed/tickets_clean.csv

By default only English tickets are kept so the TF-IDF model trains on a
consistent language. Pass --lang all to keep every language.

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --lang all          # keep all languages
    python src/data/preprocess.py --lang DE           # German only
    python src/data/preprocess.py myfile.csv --lang EN
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Greeting / noise phrases that appear in virtually every ticket
_NOISE = re.compile(
    r"\b(hello|hi|hey|dear|thanks|thank you|thank|regards|best regards|"
    r"sincerely|please|kindly|help|team|greetings|good morning|good afternoon|"
    r"good evening|to whom it may concern)\b"
)

# Normalize common variant phrasings to a single consistent token
_NORMALIZATIONS = [
    (re.compile(r"\blog[- ]?in\b"),          "login"),
    (re.compile(r"\bsign[- ]?in\b"),         "login"),
    (re.compile(r"\bpassword[- ]?reset\b"),  "reset_password"),
    (re.compile(r"\breset[- ]?password\b"),  "reset_password"),
    (re.compile(r"\bsign[- ]?up\b"),         "signup"),
]
sys.path.insert(0, str(ROOT))
PROCESSED_DIR = ROOT / "data" / "processed"

from src.data.download_data import (
    BODY_COL,
    LABEL_COL,
    LANGUAGE_COL,
    SUBJECT_COL,
    load_raw,
)


def _clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # strip non-ASCII (removes emojis if any appear in values)
    text = text.encode("ascii", errors="ignore").decode()
    # normalize common phrase variants before removing punctuation
    for pattern, replacement in _NORMALIZATIONS:
        text = pattern.sub(replacement, text)
    # remove greeting / noise words
    text = _NOISE.sub(" ", text)
    # collapse repeated punctuation (!!!!, ????, ----) then strip non-word chars
    text = re.sub(r"[!?.\-]{2,}", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    # strip non-ASCII so any emoji prefixes in Queue values don't survive
    label = label.encode("ascii", errors="ignore").decode().strip()
    return re.sub(r"\s+", "_", label)


def clean(df: pd.DataFrame, lang_filter: str = "EN") -> pd.DataFrame:
    before = len(df)

    # drop rows missing the text or label we need
    df = df.dropna(subset=[BODY_COL, LABEL_COL]).copy()

    # optional language filter
    if lang_filter.lower() != "all":
        lang_filter = lang_filter.upper()
        if LANGUAGE_COL in df.columns:
            df = df[df[LANGUAGE_COL].str.upper() == lang_filter]
        else:
            print(f"[WARNING] Column '{LANGUAGE_COL}' not found — skipping language filter")

    # combine subject + body into a single text field
    subject = df[SUBJECT_COL].fillna("") if SUBJECT_COL in df.columns else pd.Series([""] * len(df), index=df.index)
    df["text"] = (subject + " " + df[BODY_COL]).apply(_clean_text)
    df["label"] = df[LABEL_COL].apply(_normalize_label)

    # deduplicate on content (new dataset has no ticket ID column)
    df = df.drop_duplicates(subset=["text", "label"])

    # drop anything that came out empty after cleaning
    df = df[df["text"].str.len() > 0]

    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate / null / filtered-language rows")

    # keep useful extra columns if present (dataset uses lowercase column names)
    keep = ["text", "label"]
    for extra in ["priority", LANGUAGE_COL, "type", "business_type"]:
        if extra in df.columns:
            keep.append(extra)

    return df[keep].reset_index(drop=True)


def run(filename=None, lang_filter: str = "EN") -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(filename) if filename else load_raw()
    clean_df = clean(df, lang_filter=lang_filter)

    print(f"\nLanguage filter : {lang_filter}")
    print(f"Final shape     : {clean_df.shape}")
    print(f"Label distribution:\n{clean_df['label'].value_counts().to_string()}\n")

    out = PROCESSED_DIR / "tickets_clean.csv"
    clean_df.to_csv(out, index=False)
    print(f"Saved -> {out}")
    return clean_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess customer support tickets")
    parser.add_argument("filename", nargs="?", default=None, help="CSV filename inside data/raw/ (optional)")
    parser.add_argument(
        "--lang",
        default="EN",
        help="Language to keep: EN, DE, ES, FR, PT, or 'all' (default: EN)",
    )
    args = parser.parse_args()
    run(args.filename, lang_filter=args.lang)
