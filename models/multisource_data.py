"""
Multisource training frame (financials + KAM dummies + FinBERT news aggregates).

Used by SHAP script, Streamlit app, and LLM verdict generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.xgboost_full import (
    FINANCIAL_COLS,
    KAM_COLS,
    SENTIMENT_COLS,
    load_data,
    prepare_modeling_dataframe,
    prepare_target,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_TRAINING = PROJECT_ROOT / "data" / "processed" / "merged_multisource_training.csv"

FULL_FEATURE_COLS = FINANCIAL_COLS + KAM_COLS + SENTIMENT_COLS

_MERGED_REQUIRED = set(
    FULL_FEATURE_COLS
    + [
        "rating",
        "ticker",
        "fiscal_year",
        "rating_category",
        "company_name",
    ]
)


def raw_merged_frame() -> pd.DataFrame:
    """Inner-join financials, news, KAMs (same as xgboost_full.load_data) + rating categories."""
    df = load_data()
    df = prepare_target(df)
    if "rating_agency" not in df.columns:
        df["rating_agency"] = "Tassnief"
    if "sector" not in df.columns:
        df["sector"] = ""
    return df


def load_or_build_merged_training(save: bool = True) -> pd.DataFrame:
    """
    Load merged_multisource_training.csv if present and schema-complete; otherwise
    rebuild from processed CSVs and optionally save.
    """
    if MERGED_TRAINING.exists():
        df = pd.read_csv(MERGED_TRAINING)
        if _MERGED_REQUIRED.issubset(df.columns):
            return df
    df = raw_merged_frame()
    df_clean = prepare_modeling_dataframe(df)
    if save:
        MERGED_TRAINING.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(MERGED_TRAINING, index=False)
    return df_clean
