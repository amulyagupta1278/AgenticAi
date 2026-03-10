import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import _clean_text, clean
from src.features.build_features import get_vectorizer, split


def _make_df(n=120):
    rng = np.random.default_rng(0)
    labels = ["technical_issue", "billing_inquiry", "product_inquiry", "refund_request"]
    return pd.DataFrame({
        "subject": [f"issue report {i}" for i in range(n)],
        "body": [
            f"account problem number {i} experiencing connection error" for i in range(n)
        ],
        "queue": rng.choice(labels, n),
        "language": ["EN"] * n,
    })


# ── ingest ──────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases(self):
        # "HELLO" is a noise word and gets stripped; "WORLD" is lowercased
        assert _clean_text("HELLO WORLD") == "world"

    def test_strips_punctuation(self):
        result = _clean_text("Can't login!!! Help??")
        assert "!" not in result and "?" not in result

    def test_null_returns_empty(self):
        assert _clean_text(None) == ""
        assert _clean_text(float("nan")) == ""

    def test_extra_whitespace(self):
        assert _clean_text("  too   many   spaces  ") == "too many spaces"


class TestClean:
    def test_output_columns(self):
        raw = _make_df()
        out = clean(raw)
        assert "text" in out.columns
        assert "label" in out.columns

    def test_no_nulls_in_text(self):
        raw = _make_df()
        out = clean(raw)
        assert out["text"].isna().sum() == 0

    def test_label_snake_case(self):
        raw = _make_df()
        out = clean(raw)
        # no spaces in labels
        assert not out["label"].str.contains(" ").any()

    def test_dedup(self):
        raw = _make_df(60)
        # inject 10 duplicates
        raw = pd.concat([raw, raw.iloc[:10]]).reset_index(drop=True)
        out = clean(raw)
        assert len(out) == 60


# ── features ────────────────────────────────────────────────────────────────

class TestFeatures:
    @pytest.fixture
    def clean_df(self):
        raw = _make_df()
        return clean(raw)

    def test_split_shapes(self, clean_df):
        X_tr, X_te, y_tr, y_te, le = split(clean_df, test_size=0.2)
        total = len(X_tr) + len(X_te)
        assert total == len(clean_df)
        assert len(y_tr) == len(X_tr)

    def test_label_encoder_classes(self, clean_df):
        _, _, _, _, le = split(clean_df)
        assert len(le.classes_) == clean_df["label"].nunique()

    def test_tfidf_output_shape(self, clean_df):
        X_tr, X_te, y_tr, y_te, le = split(clean_df)
        # Use min_df=1 / max_df=1.0 for the small synthetic corpus; production
        # defaults (min_df=5, max_df=0.8) are tested implicitly by train.py.
        vec = get_vectorizer(max_features=200, min_df=1, max_df=1.0)
        X_tr_v = vec.fit_transform(X_tr)
        X_te_v = vec.transform(X_te)
        assert X_tr_v.shape[0] == len(X_tr)
        assert X_te_v.shape[1] == X_tr_v.shape[1]

    def test_stratified_split(self, clean_df):
        _, _, y_tr, y_te, le = split(clean_df, test_size=0.2)
        tr_dist = np.bincount(y_tr) / len(y_tr)
        te_dist = np.bincount(y_te) / len(y_te)
        # distributions should be close (within 10 pp)
        assert np.abs(tr_dist - te_dist).max() < 0.10
