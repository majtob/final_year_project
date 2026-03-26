#!/usr/bin/env python3
"""
SHAP explainability for the full multisource XGBoost model (14 features:
financials + paper KAM dummies + FinBERT news aggregates).

Writes:
  figures/shap_beeswarm.png
  figures/shap_bar_importance.png
  figures/shap_dependence_*.png (two interaction pairs)
  figures/shap_waterfall_<ticker>_<year>.png (sample of companies)
  results/shap_report.json

Run from repo root:
  PYTHONPATH=. python models/shap_explainability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multisource_data import (  # noqa: E402
    FULL_FEATURE_COLS,
    load_or_build_merged_training,
)
from models.xgboost_full import prepare_target  # noqa: E402

FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "liquid": "Liquidity (working capital / total assets)",
    "cumprof": "Cumulative profitability (retained earnings / total assets)",
    "profitab": "Profitability (EBIT / total assets)",
    "leverage": "Leverage (book equity / total liabilities)",
    "GCKAM": "KAM: going concern (0/1)",
    "REVKAM": "KAM: revenue recognition (0/1)",
    "ASSETKAM": "KAM: impairment / assets (0/1)",
    "LIABKAM": "KAM: liabilities (0/1)",
    "OTHERKAM": "KAM: other topics (0/1)",
    "sentiment_mean": "FinBERT mean article score (P(pos)−P(neg))",
    "sentiment_std": "FinBERT sentiment dispersion",
    "sentiment_pos_pct": "Share of articles with positive FinBERT score",
    "sentiment_neg_pct": "Share of articles with negative FinBERT score",
    "news_count": "Number of news articles in fiscal year",
}


def _train_full_xgb(X: np.ndarray, y: np.ndarray, le: LabelEncoder):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    return model


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_build_merged_training(save=True)
    if "rating_category" not in df.columns:
        df = prepare_target(df)
    feat_names = FULL_FEATURE_COLS
    X = df[feat_names].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["rating_category"].values)

    model = _train_full_xgb(X, y, le)
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X)

    if isinstance(shap_raw, list):
        global_mean_abs = np.mean(
            [np.abs(s).mean(axis=0) for s in shap_raw],
            axis=0,
        )
        global_mean_abs = np.asarray(global_mean_abs).ravel()
        majority_class = int(np.argmax(np.bincount(y)))
        sv_for_summary = np.asarray(shap_raw[majority_class])
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        global_mean_abs = np.abs(shap_raw).mean(axis=(0, 2))
        global_mean_abs = np.asarray(global_mean_abs).ravel()
        sv_for_summary = shap_raw[:, :, int(np.argmax(np.bincount(y)))]
    else:
        shap_arr = np.asarray(shap_raw)
        global_mean_abs = np.abs(shap_arr).mean(axis=0)
        global_mean_abs = np.asarray(global_mean_abs).ravel()
        sv_for_summary = shap_arr

    order = np.argsort(-global_mean_abs)
    ranked = [(feat_names[i], float(global_mean_abs[i])) for i in order]
    global_importance = {}
    for rank, (name, mabs) in enumerate(ranked, start=1):
        global_importance[name] = {
            "rank": rank,
            "mean_abs_shap": mabs,
            "description": FEATURE_DESCRIPTIONS.get(name, name),
        }

    total_mabs = sum(global_mean_abs) + 1e-12
    pct = {feat_names[i]: 100.0 * global_mean_abs[i] / total_mabs for i in range(len(feat_names))}

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_for_summary,
        X,
        feature_names=feat_names,
        show=False,
        plot_size=(10, 8),
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv_for_summary,
        X,
        feature_names=feat_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    def dependence_pair(x_col: str, color_col: str, fname: str) -> None:
        if x_col not in feat_names or color_col not in feat_names:
            return
        xi = feat_names.index(x_col)
        ci = feat_names.index(color_col)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            xi,
            sv_for_summary,
            X,
            feature_names=feat_names,
            interaction_index=ci,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()

    dependence_pair("leverage", "profitab", "shap_dependence_leverage_profitab.png")
    dependence_pair("liquid", "cumprof", "shap_dependence_liquid_cumprof.png")
    dependence_pair("news_count", "sentiment_mean", "shap_dependence_news_sentiment.png")

    def _per_class_shap_rows(shap_vals):
        if isinstance(shap_vals, list):
            return [np.asarray(s) for s in shap_vals]
        arr = np.asarray(shap_vals)
        if arr.ndim == 3:
            return [arr[:, :, c] for c in range(arr.shape[2])]
        return [arr]

    y_hat = model.predict(X)
    ev = explainer.expected_value
    class_rows = _per_class_shap_rows(shap_raw)

    for old in FIGURES_DIR.glob("shap_waterfall_*.png"):
        old.unlink(missing_ok=True)
    sample_idx = np.linspace(0, len(df) - 1, num=min(8, len(df)), dtype=int)
    for i in sample_idx:
        row = df.iloc[i]
        cls = int(y_hat[i])
        sv_row = np.asarray(class_rows[cls][i]).ravel()
        if hasattr(ev, "__len__") and not isinstance(ev, (str, bytes)):
            base = float(ev[cls])
        else:
            base = float(ev)
        exp = shap.Explanation(
            values=sv_row,
            base_values=base,
            data=X[i],
            feature_names=[FEATURE_DESCRIPTIONS.get(f, f) for f in feat_names],
        )
        fig, _ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        t = str(row["ticker"]).replace(".", "_")
        fy = int(row["fiscal_year"])
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{t}_{fy}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    per_company = []
    for i in range(len(df)):
        row = df.iloc[i]
        cls = int(y_hat[i])
        sv_row = np.asarray(class_rows[cls][i]).ravel()
        j = int(np.argmax(np.abs(sv_row)))
        per_company.append(
            {
                "ticker": row["ticker"],
                "company": str(row.get("company_name", ""))[:60],
                "fiscal_year": int(row["fiscal_year"]),
                "rating": row["rating"],
                "predicted_category": le.inverse_transform([cls])[0],
                "top_feature": feat_names[j],
                "shap_value": float(sv_row[j]),
                "feature_value": float(X[i, j]),
            }
        )

    report = {
        "model": "XGBClassifier full (financials + KAM + FinBERT aggregates)",
        "n_samples": int(len(df)),
        "features": feat_names,
        "classes": list(le.classes_),
        "global_importance": global_importance,
        "pct_total_abs_shap": {k: float(pct[k]) for k in feat_names},
        "top_feature_share_pct": float(max(pct.values())),
        "per_company_top_factors": per_company[:25],
    }
    with open(RESULTS_DIR / "shap_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote SHAP figures under {FIGURES_DIR}")
    print(f"Wrote {RESULTS_DIR / 'shap_report.json'}")


if __name__ == "__main__":
    main()
