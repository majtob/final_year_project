"""
Regenerate figures in figures/ from the current merged multisource panel (14 features).

Writes / overwrites:
  rating_distribution.png      — counts of rating_category
  confusion_matrix_multisource.png — XGBoost 4-class, StratifiedKFold CV predictions
  pca_multisource.png          — first two PCA components (scaled), colored by category
  feature_distributions_multisource.png — four ratios, boxplot by rating_category
  confidence_distribution_multisource.png — max class probability from CV predict_proba
  error_scatter_multisource.png — PCA(2) colored correct vs wrong
  model_comparison.png         — copy of multisource benchmark bar chart (if benchmark exists)

Also writes results/error_analysis_multisource.json for the current XGBoost multiclass run.

Run: PYTHONPATH=. python models/generate_pipeline_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multisource_data import FULL_FEATURE_COLS, load_or_build_merged_training  # noqa: E402
from models.xgboost_full import prepare_target  # noqa: E402

FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_build_merged_training(save=True)
    if "rating_category" not in df.columns:
        df = prepare_target(df)
    df = df.dropna(subset=["liquid", "cumprof", "profitab", "leverage"], how="any").reset_index(
        drop=True
    )

    X = df[FULL_FEATURE_COLS].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["rating_category"].values)
    classes = list(le.classes_)
    counts = np.bincount(y)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=1,
    )
    y_pred = cross_val_predict(model, X, y, cv=cv)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    conf = np.max(y_proba, axis=1)
    acc = accuracy_score(y, y_pred)

    # --- rating distribution ---
    plt.figure(figsize=(7, 4))
    order = list(le.classes_)
    vc = df["rating_category"].value_counts().reindex(order).fillna(0).astype(int)
    plt.bar(vc.index.astype(str), vc.values, color="steelblue", edgecolor="white")
    plt.xlabel("Rating category")
    plt.ylabel("Count (rows)")
    plt.title("Rating category distribution — current multisource panel")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- confusion matrix ---
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"XGBoost multisource (CV) — acc={acc:.1%}  n={len(df)}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_multisource.png", dpi=150, bbox_inches="tight")
    plt.close()
    # also refresh legacy filename used by Streamlit caption
    shutil.copy(FIGURES_DIR / "confusion_matrix_multisource.png", FIGURES_DIR / "confusion_matrix_gb.png")

    # --- PCA ---
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    plt.figure(figsize=(7, 5))
    for c in classes:
        m = df["rating_category"].values == c
        plt.scatter(Z[m, 0], Z[m, 1], label=c, alpha=0.75, s=45)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    plt.legend(title="Category", fontsize=8)
    plt.title("PCA (14 features, standardized) — multisource panel")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_multisource.png", dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(FIGURES_DIR / "pca_multisource.png", FIGURES_DIR / "pca_scatter.png")

    # --- feature distributions (four ratios) ---
    ratio_cols = ["liquid", "cumprof", "profitab", "leverage"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for i, col in enumerate(ratio_cols):
        ax = axes[i]
        plot_df = df[["rating_category", col]].copy()
        sns.boxplot(data=plot_df, x="rating_category", y=col, ax=ax, order=classes)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle("Financial ratios by rating category (current panel)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_distributions_multisource.png", dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(
        FIGURES_DIR / "feature_distributions_multisource.png",
        FIGURES_DIR / "feature_distributions.png",
    )

    # --- confidence distribution ---
    plt.figure(figsize=(6, 4))
    plt.hist(conf, bins=12, color="teal", edgecolor="white", alpha=0.85)
    plt.xlabel("Max predicted class probability (CV)")
    plt.ylabel("Count")
    plt.title("Model confidence — multisource XGBoost (CV folds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_distribution_multisource.png", dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(
        FIGURES_DIR / "confidence_distribution_multisource.png",
        FIGURES_DIR / "confidence_dist.png",
    )

    # --- error scatter (correct vs wrong in PCA space) ---
    correct = y_pred == y
    plt.figure(figsize=(7, 5))
    plt.scatter(Z[~correct, 0], Z[~correct, 1], c="crimson", s=55, label="Misclassified", alpha=0.9)
    plt.scatter(Z[correct, 0], Z[correct, 1], c="0.75", s=35, label="Correct", alpha=0.6)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    plt.legend()
    plt.title("Misclassifications in PCA space (multisource XGBoost CV)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_scatter_multisource.png", dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(FIGURES_DIR / "error_scatter_multisource.png", FIGURES_DIR / "error_scatter.png")

    # --- simple error patterns: confusion pairs counts ---
    pairs: dict[str, int] = {}
    for a, p in zip(df["rating_category"].values, le.inverse_transform(y_pred)):
        if a != p:
            key = f"{a} → {p}"
            pairs[key] = pairs.get(key, 0) + 1
    plt.figure(figsize=(8, max(3, 0.35 * len(pairs) + 1)))
    if pairs:
        items = sorted(pairs.items(), key=lambda x: -x[1])
        labs = [k for k, _ in items]
        vals = [v for _, v in items]
        plt.barh(labs[::-1], vals[::-1], color="coral")
        plt.xlabel("Count")
    else:
        plt.text(0.5, 0.5, "No CV errors", ha="center", va="center")
    plt.title("Misclassification patterns (actual → predicted)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_patterns_multisource.png", dpi=150, bbox_inches="tight")
    plt.close()
    shutil.copy(FIGURES_DIR / "error_patterns_multisource.png", FIGURES_DIR / "error_patterns.png")

    # --- sync model_comparison.png with benchmark if present ---
    bench = FIGURES_DIR / "multisource_model_comparison.png"
    if bench.exists():
        shutil.copy(bench, FIGURES_DIR / "model_comparison.png")

    # --- error_analysis JSON (multiclass) ---
    mis_rows = []
    for i in range(len(df)):
        if y_pred[i] == y[i]:
            continue
        row = df.iloc[i]
        mis_rows.append(
            {
                "ticker": row["ticker"],
                "company_name": str(row.get("company_name", "")),
                "fiscal_year": int(row["fiscal_year"]),
                "actual_category": row["rating_category"],
                "predicted_category": le.inverse_transform([int(y_pred[i])])[0],
                "confidence": float(conf[i]),
            }
        )

    error_payload = {
        "generated_for": "multisource XGBoost 14 features, 4-class rating_category",
        "n_samples": int(len(df)),
        "cv_folds": n_splits,
        "cv_accuracy": float(acc),
        "classes": classes,
        "summary": {
            "total_samples": int(len(df)),
            "correct": int(np.sum(y_pred == y)),
            "incorrect": int(np.sum(y_pred != y)),
            "accuracy": float(acc),
        },
        "misclassified": mis_rows,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
        },
    }
    with open(RESULTS_DIR / "error_analysis_multisource.json", "w", encoding="utf-8") as f:
        json.dump(error_payload, f, indent=2)
    shutil.copy(RESULTS_DIR / "error_analysis_multisource.json", RESULTS_DIR / "error_analysis.json")

    meta = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "n_rows": len(df),
        "figures_written": [
            "rating_distribution.png",
            "confusion_matrix_multisource.png",
            "confusion_matrix_gb.png",
            "pca_multisource.png",
            "pca_scatter.png",
            "feature_distributions_multisource.png",
            "feature_distributions.png",
            "confidence_distribution_multisource.png",
            "confidence_dist.png",
            "error_scatter_multisource.png",
            "error_scatter.png",
            "error_patterns_multisource.png",
            "error_patterns.png",
            "model_comparison.png (from multisource benchmark if present)",
        ],
        "results_written": ["error_analysis_multisource.json", "error_analysis.json"],
    }
    with open(RESULTS_DIR / "pipeline_figures_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote figures under {FIGURES_DIR}")
    print(f"Wrote {RESULTS_DIR / 'error_analysis.json'} (multiclass CV summary)")


if __name__ == "__main__":
    main()
