"""
XGBoost trained on kams_processed.csv only (no financial ratios or news).

Uses all integer/paper columns in that file as features and the same
rating → rating_category target as xgboost_with_kams.py for comparability.

Run from repo root:
  python models/xgboost_kams_only.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
KAMS_FILE = PROJECT_ROOT / "data" / "processed" / "kams_processed.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Every modeling column shipped in kams_processed (excludes ids + rating label; no KAM_COUNT).
FEATURE_COLS = [
    "AUSIZE",
    "AUOP",
    "EMP",
    "GCUP",
    "GCKAM",
    "REVKAM",
    "ASSETKAM",
    "LIABKAM",
    "OTHERKAM",
    "FIRMAGE",
    "FIRMSIZE",
    "INDUSTRY",
]


def rating_category(rating: str) -> str:
    r = (rating or "").strip()
    if r in ["AAA", "AA+", "AA", "AA-"]:
        return "AA"
    if r in ["A+", "A", "A-"]:
        return "A"
    if r in ["BBB+", "BBB", "BBB-"]:
        return "BBB"
    return "BB"


def load_kams_only() -> pd.DataFrame:
    if not KAMS_FILE.exists():
        raise FileNotFoundError(f"Missing {KAMS_FILE}; run scripts/rebuild_processed_datasets.py")
    df = pd.read_csv(KAMS_FILE)
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fiscal_year", "rating"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["rating_category"] = df["rating"].astype(str).str.strip().apply(rating_category)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  XGBoost — KAMs processed file ONLY")
    print(f"  Data: {KAMS_FILE.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    df = load_kams_only()
    print(f"Rows: {len(df)}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print("\nTarget (rating_category) counts:")
    print(df["rating_category"].value_counts().to_string())

    X = df[FEATURE_COLS].values
    y_cat = df["rating_category"].values
    le = LabelEncoder()
    y = le.fit_transform(y_cat)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    counts = np.bincount(y)
    n_splits = min(5, int(counts.min()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"\n{n_splits}-fold stratified CV accuracy: {cv_scores.mean():.2%} "
              f"(+/- {cv_scores.std() * 2:.2%})")
    else:
        cv_scores = np.array([0.0])
        print("\nWARNING: smallest class count < 2; skipping CV.")

    model.fit(X, y)
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)

    print(f"\nTrain accuracy (in-sample): {train_acc:.2%}")
    print("\nClassification report (in-sample):")
    print(classification_report(y, y_pred, target_names=le.classes_, digits=3))

    importance = sorted(
        zip(FEATURE_COLS, model.feature_importances_.tolist()),
        key=lambda x: -x[1],
    )
    print("Feature importance:")
    for name, imp in importance:
        print(f"  {name:<12} {imp:.4f}")

    out = {
        "model": "xgboost_kams_only",
        "data_file": str(KAMS_FILE.relative_to(PROJECT_ROOT)),
        "n_samples": int(len(df)),
        "features": FEATURE_COLS,
        "target": "rating_category (AA / A / BBB / BB)",
        "cv_folds": int(n_splits) if n_splits >= 2 else 0,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "train_accuracy": float(train_acc),
        "feature_importance": {k: float(v) for k, v in importance},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = RESULTS_DIR / "kams_only_model_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
