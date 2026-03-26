#!/usr/bin/env python3
"""
Compare multiple classifiers on the same multisource panel as xgboost_full.py
(14 features: financials + KAM dummies + FinBERT aggregates; 4-class rating_category).

Uses StratifiedKFold CV (same random_state as other scripts). Writes:
  results/multisource_model_comparison.json
  figures/multisource_model_comparison.png

Run from repo root:
  PYTHONPATH=. python models/evaluate_multisource_models.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multisource_data import FULL_FEATURE_COLS, load_or_build_merged_training  # noqa: E402
from models.xgboost_full import prepare_target  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def build_estimators() -> dict[str, object]:
    """Tree / kernel / linear models; XGBoost included as reference."""
    return {
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "sklearn_gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            max_iter=100,
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2500,
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        max_iter=8000,
                        random_state=42,
                        dual="auto",
                    ),
                ),
            ]
        ),
        "knn_k5": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=800,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_build_merged_training(save=True)
    if "rating_category" not in df.columns:
        df = prepare_target(df)

    X = df[FULL_FEATURE_COLS].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["rating_category"].values)
    counts = np.bincount(y)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = ["accuracy", "f1_macro", "f1_weighted"]
    estimators = build_estimators()
    summary: dict[str, dict] = {}

    print(f"Samples: {len(df)}, features: {len(FULL_FEATURE_COLS)}, CV folds: {n_splits}")
    print(f"Classes: {list(le.classes_)} counts: {counts.tolist()}\n")

    for name, est in estimators.items():
        out = cross_validate(
            est,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row = {}
        for m in scoring:
            key = f"test_{m}"
            row[f"{m}_mean"] = float(np.mean(out[key]))
            row[f"{m}_std"] = float(np.std(out[key]))
        summary[name] = row
        print(
            f"{name:28}  acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
            f"f1_macro={row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}"
        )

    ranking = sorted(
        summary.items(),
        key=lambda kv: kv[1]["accuracy_mean"],
        reverse=True,
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": int(len(df)),
        "n_features": len(FULL_FEATURE_COLS),
        "feature_names": FULL_FEATURE_COLS,
        "classes": list(le.classes_),
        "class_counts": {c: int(n) for c, n in zip(le.classes_, counts)},
        "cv_folds": n_splits,
        "cv_random_state": 42,
        "models": summary,
        "ranking_by_accuracy": [name for name, _ in ranking],
    }
    out_json = RESULTS_DIR / "multisource_model_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    # Bar chart: accuracy with error bars
    names = [k for k, _ in ranking]
    acc = [summary[k]["accuracy_mean"] for k in names]
    err = [summary[k]["accuracy_std"] for k in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, acc, xerr=err, capsize=3, color="steelblue", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Stratified CV accuracy (mean ± std)")
    ax.set_title("Multisource rating prediction — model comparison (non–XGBoost + XGBoost)")
    ax.set_xlim(0, 1.05)
    for i, (a, s) in enumerate(zip(acc, err)):
        ax.text(min(a + s + 0.02, 0.99), i, f"{a:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "multisource_model_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
