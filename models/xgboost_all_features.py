"""
All-features model comparison: 4 financial + 12 full KAM + 5 news = 21 features.

Runs:
  1. XGBoost ablation across feature subsets
  2. Multi-classifier benchmark on the full 21-feature matrix

Writes:
  results/all_features_model_results.json
  figures/all_features_model_comparison.png

Run from repo root:
  PYTHONPATH=. python models/xgboost_all_features.py
"""

from __future__ import annotations

import json
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
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINANCIALS_FILE = PROJECT_ROOT / "data" / "processed" / "financials_processed.csv"
NEWS_FILE = PROJECT_ROOT / "data" / "processed" / "news_features_processed.csv"
KAMS_FILE = PROJECT_ROOT / "data" / "processed" / "kams_processed.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

FINANCIAL_COLS = ["liquid", "cumprof", "profitab", "leverage"]
FULL_KAM_COLS = [
    "AUSIZE", "AUOP", "EMP", "GCUP",
    "GCKAM", "REVKAM", "ASSETKAM", "LIABKAM", "OTHERKAM",
    "FIRMAGE", "FIRMSIZE", "INDUSTRY",
]
SENTIMENT_COLS = [
    "sentiment_mean", "sentiment_std", "sentiment_pos_pct",
    "sentiment_neg_pct", "news_count",
]
ALL_FEATURE_COLS = FINANCIAL_COLS + FULL_KAM_COLS + SENTIMENT_COLS


def load_and_merge() -> pd.DataFrame:
    """Inner-join financials, full KAMs, and news on (ticker, fiscal_year)."""
    fin = pd.read_csv(FINANCIALS_FILE)
    fin["fiscal_year"] = pd.to_numeric(fin["fiscal_year"], errors="coerce").astype("Int64")
    fin = fin.dropna(subset=["fiscal_year"])
    fin["fiscal_year"] = fin["fiscal_year"].astype(int)
    if "rating_agency" in fin.columns:
        agency = fin["rating_agency"].astype(str).str.strip().str.lower()
        fin = fin.assign(_pri=(agency == "tassnief").astype(int))
        fin = fin.sort_values("_pri", ascending=False)
        fin = fin.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first").drop(columns=["_pri"])
    print(f"Financials: {len(fin)} rows")

    news = pd.read_csv(NEWS_FILE)
    news["fiscal_year"] = news["fiscal_year"].astype(int)
    print(f"News:       {len(news)} rows")

    kams = pd.read_csv(KAMS_FILE)
    kams["fiscal_year"] = kams["fiscal_year"].astype(int)
    print(f"KAMs:       {len(kams)} rows")

    keep_fin = ["ticker", "fiscal_year", "rating"] + FINANCIAL_COLS
    extras = [c for c in ("company_name", "rating_agency") if c in fin.columns]
    fin_slim = fin[keep_fin + extras].copy()

    keep_news = ["ticker", "fiscal_year"] + SENTIMENT_COLS
    news_slim = news[[c for c in keep_news if c in news.columns]].copy()

    keep_kams = ["ticker", "fiscal_year"] + FULL_KAM_COLS
    kams_slim = kams[[c for c in keep_kams if c in kams.columns]].copy()

    merged = fin_slim.merge(news_slim, on=["ticker", "fiscal_year"], how="inner")
    merged = merged.merge(kams_slim, on=["ticker", "fiscal_year"], how="inner")
    print(f"Merged:     {len(merged)} rows (inner join across all three)")
    return merged


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add rating_category, drop NaN ratios, fill missing sentiment/KAM cols."""
    def _cat(r: str) -> str:
        if r in ("AAA", "AA+", "AA", "AA-"):
            return "AA"
        if r in ("A+", "A", "A-"):
            return "A"
        if r in ("BBB+", "BBB", "BBB-"):
            return "BBB"
        return "BB"

    df = df.dropna(subset=FINANCIAL_COLS).copy()
    df["rating_category"] = df["rating"].apply(_cat)
    for c in SENTIMENT_COLS:
        df[c] = df[c].fillna(0)
    for c in FULL_KAM_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
        else:
            df[c] = 0
    return df


def run_ablation(X, y, le, feat_names, label):
    """Train XGBoost with stratified 5-fold CV and print results."""
    print(f"\n{'=' * 64}")
    print(f"  {label}  ({len(feat_names)} features)")
    print(f"{'=' * 64}")
    print(f"  Features: {feat_names}")
    print(f"  Samples:  {len(y)}   Classes: {dict(zip(le.classes_, np.bincount(y)))}")

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="mlogloss",
    )
    n_splits = int(min(5, min(np.bincount(y))))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)

    print(f"\n  {n_splits}-fold CV accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")

    importance = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    print("  Feature importance:")
    for feat, imp in importance:
        bar = "#" * int(imp * 50)
        print(f"    {feat:<22} {imp:.4f}  {bar}")

    y_pred = model.predict(X)
    print(f"\n  Train accuracy: {accuracy_score(y, y_pred):.2%}")
    print(f"  Classification report:\n{classification_report(y, y_pred, target_names=le.classes_, digits=3)}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  Classes: {list(le.classes_)}")
    print(confusion_matrix(y, y_pred))

    return model, scores, importance


def build_estimators() -> dict[str, object]:
    """All candidate classifiers for the 21-feature benchmark."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="mlogloss", n_jobs=1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=2, random_state=42,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2500, random_state=42, solver="lbfgs")),
        ]),
        "Linear SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=8000, random_state=42, dual="auto")),
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=800,
                random_state=42, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20,
            )),
        ]),
    }


def run_benchmark(X, y, le, cv):
    """Run all classifiers on the full 21-feature matrix."""
    print("\n" + "=" * 64)
    print("  MULTI-CLASSIFIER BENCHMARK (21 features)")
    print("=" * 64)

    estimators = build_estimators()
    scoring = ["accuracy", "f1_macro", "f1_weighted"]
    summary = {}

    for name, est in estimators.items():
        out = cross_validate(
            est, X, y, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False,
        )
        row = {}
        for m in scoring:
            key = f"test_{m}"
            row[f"{m}_mean"] = float(np.mean(out[key]))
            row[f"{m}_std"] = float(np.std(out[key]))
        summary[name] = row
        print(
            f"  {name:28s}  acc={row['accuracy_mean']:.3f} +/-{row['accuracy_std']:.3f}  "
            f"f1_macro={row['f1_macro_mean']:.3f}"
        )

    return summary


def plot_benchmark(summary: dict, out_path: Path):
    """Horizontal bar chart of classifier accuracies with error bars."""
    ranking = sorted(summary.items(), key=lambda kv: kv[1]["accuracy_mean"], reverse=True)
    names = [k for k, _ in ranking]
    acc = [summary[k]["accuracy_mean"] for k in names]
    err = [summary[k]["accuracy_std"] for k in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, acc, xerr=err, capsize=3, color="steelblue", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Stratified CV accuracy (mean +/- std)")
    ax.set_title("21-feature rating prediction — model comparison (all sources)")
    ax.set_xlim(0, 1.05)
    for i, (a, s) in enumerate(zip(acc, err)):
        ax.text(min(a + s + 0.02, 0.99), i, f"{a:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Wrote {out_path}")


def main():
    print("=" * 64)
    print("  ALL-FEATURES MODEL (4 financial + 12 KAM + 5 news = 21)")
    print("=" * 64)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    df = load_and_merge()
    df = prepare(df)
    print(f"\nUsable rows: {len(df)}")
    print(f"Rating distribution:\n{df['rating_category'].value_counts().to_string()}\n")

    le = LabelEncoder()
    y = le.fit_transform(df["rating_category"].values)

    n_splits = int(min(5, min(np.bincount(y))))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- Part 1: XGBoost ablation across feature subsets ---
    ablations = {
        "financials_only": FINANCIAL_COLS,
        "financials_plus_full_kams": FINANCIAL_COLS + FULL_KAM_COLS,
        "financials_plus_news": FINANCIAL_COLS + SENTIMENT_COLS,
        "full_kams_plus_news": FULL_KAM_COLS + SENTIMENT_COLS,
        "all_features": ALL_FEATURE_COLS,
    }

    abl_results = {}
    for name, feats in ablations.items():
        X = df[feats].values.astype(np.float64)
        _, scores, imp = run_ablation(X, y, le, feats, name.replace("_", " ").upper())
        abl_results[name] = {
            "n_features": len(feats),
            "features": feats,
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "top_3_features": [(f, float(v)) for f, v in imp[:3]],
        }

    print("\n" + "=" * 64)
    print("  XGBOOST ABLATION SUMMARY")
    print("=" * 64)
    print(f"  {'Ablation':<35} {'Features':>8}  {'CV Accuracy':>12}")
    print(f"  {'-' * 58}")
    for name, r in abl_results.items():
        print(f"  {name:<35} {r['n_features']:>8}  {r['accuracy_mean']*100:>9.2f}%")

    best_abl = max(abl_results, key=lambda k: abl_results[k]["accuracy_mean"])
    print(f"\n  Best ablation: {best_abl} ({abl_results[best_abl]['accuracy_mean']*100:.2f}%)")

    # --- Part 2: Multi-classifier benchmark on ALL 21 features ---
    X_all = df[ALL_FEATURE_COLS].values.astype(np.float64)
    bench = run_benchmark(X_all, y, le, cv)

    ranking = sorted(bench.items(), key=lambda kv: kv[1]["accuracy_mean"], reverse=True)
    print("\n" + "=" * 64)
    print("  BENCHMARK RANKING (21 features)")
    print("=" * 64)
    print(f"  {'Model':<28} {'Accuracy':>10} {'F1 macro':>10} {'F1 weighted':>12}")
    print(f"  {'-' * 62}")
    for name, r in ranking:
        print(
            f"  {name:<28} {r['accuracy_mean']*100:>8.2f}%"
            f"  {r['f1_macro_mean']*100:>8.2f}%"
            f"  {r['f1_weighted_mean']*100:>9.2f}%"
        )

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_benchmark(bench, FIGURES_DIR / "all_features_model_comparison.png")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "total_features": len(ALL_FEATURE_COLS),
        "feature_groups": {
            "financial": FINANCIAL_COLS,
            "kam_full": FULL_KAM_COLS,
            "news": SENTIMENT_COLS,
        },
        "classes": list(le.classes_),
        "class_counts": {c: int(n) for c, n in zip(le.classes_, np.bincount(y))},
        "ablations": abl_results,
        "best_ablation": best_abl,
        "benchmark_21_features": bench,
        "benchmark_ranking": [name for name, _ in ranking],
    }
    out = RESULTS_DIR / "all_features_model_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
