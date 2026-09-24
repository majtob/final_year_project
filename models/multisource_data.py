"""
Multisource training frame: 21 features — four financial ratios, the full
12-column KAM/firm block, and five FinBERT news aggregates.

Used by the SHAP script, the Streamlit app, pipeline figures and LLM verdicts,
so the feature set is defined once here and imported everywhere else.

The frame is built by joining the three processed CSVs
(`financials_processed.csv`, `news_features_processed.csv`,
`kams_processed.csv`) rather than read from a pre-merged file: the full KAM/firm
block only exists in `kams_processed.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.xgboost_all_features import (
    ALL_FEATURE_COLS,
    FINANCIAL_COLS,
    FULL_KAM_COLS,
    SENTIMENT_COLS,
    load_and_merge,
    prepare,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FULL_FEATURE_COLS = ALL_FEATURE_COLS

__all__ = [
    "FULL_FEATURE_COLS",
    "FINANCIAL_COLS",
    "FULL_KAM_COLS",
    "SENTIMENT_COLS",
    "prepare_target",
    "raw_merged_frame",
    "load_or_build_merged_training",
]


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add `rating_category`, drop rows with missing ratios, fill KAM/news gaps."""
    return prepare(df)


def raw_merged_frame() -> pd.DataFrame:
    """Financials × news × full KAM block, with rating categories attached."""
    df = prepare_target(load_and_merge())
    if "rating_agency" not in df.columns:
        df["rating_agency"] = "Tassnief"
    if "sector" not in df.columns:
        df["sector"] = ""
    return df


def load_or_build_merged_training(save: bool = False) -> pd.DataFrame:
    """
    Build the 21-feature modelling frame from the processed CSVs.

    `save` is accepted for call-site compatibility and ignored: the frame is
    derived from files the rebuild script already owns, so writing a fourth copy
    would just be one more thing to go stale.
    """
    return raw_merged_frame()
