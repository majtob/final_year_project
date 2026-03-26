"""
Streamlit Demo: Credit Rating Prediction System
Multisource XGBoost (financials + KAMs + FinBERT news) + SHAP + LLM-style verdicts
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.llm_verdict import generate_template_verdict  # noqa: E402
from models.multisource_data import FULL_FEATURE_COLS, load_or_build_merged_training  # noqa: E402
from models.xgboost_full import prepare_target  # noqa: E402

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "merged_multisource_training.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
VERDICTS_DIR = RESULTS_DIR / "verdicts"

FEATURE_LABELS = {
    "liquid": "Liquidity (WC / Total Assets)",
    "cumprof": "Cumulative profitability (RE / Total Assets)",
    "profitab": "Profitability (EBIT / Total Assets)",
    "leverage": "Leverage (Equity / Liabilities)",
    "GCKAM": "KAM: Going concern (0/1)",
    "REVKAM": "KAM: Revenue recognition (0/1)",
    "ASSETKAM": "KAM: Assets / impairment (0/1)",
    "LIABKAM": "KAM: Liabilities (0/1)",
    "OTHERKAM": "KAM: Other (0/1)",
    "sentiment_mean": "FinBERT mean score",
    "sentiment_std": "FinBERT sentiment std",
    "sentiment_pos_pct": "Share positive (FinBERT)",
    "sentiment_neg_pct": "Share negative (FinBERT)",
    "news_count": "News article count",
}


@st.cache_data
def load_data():
    df = load_or_build_merged_training(save=True)
    df = df.dropna(subset=["liquid", "cumprof", "profitab", "leverage"], how="any")
    if "rating_category" not in df.columns:
        df = prepare_target(df)
    return df.reset_index(drop=True)


@st.cache_resource
def train_model_bundle(df):
    X = df[FULL_FEATURE_COLS].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["rating_category"].values)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X)
    return model, explainer, shap_raw, le


def _class_shap_rows(shap_raw, n_classes: int):
    if isinstance(shap_raw, list):
        return [np.asarray(s) for s in shap_raw]
    arr = np.asarray(shap_raw)
    if arr.ndim == 3:
        return [arr[:, :, c] for c in range(arr.shape[2])]
    return [arr]


@st.cache_data
def load_verdicts():
    path = VERDICTS_DIR / "all_verdicts.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_verdict_for_row(row, pred_category: str, confidence: float):
    return generate_template_verdict(
        row.to_dict(), pred_category, confidence, row.get("rating")
    )


def main():
    st.set_page_config(
        page_title="Saudi Credit Rating Predictor",
        page_icon="📊",
        layout="wide",
    )

    st.title("Credit Rating Prediction System")
    st.markdown(
        "**Multisource XGBoost** (financial ratios + KAM dummies + FinBERT news) "
        "with **SHAP** and **template verdicts** — Saudi Tadawul panel"
    )
    st.markdown("---")

    df = load_data()
    model, explainer, shap_raw, le_ml = train_model_bundle(df)
    class_rows = _class_shap_rows(shap_raw, len(le_ml.classes_))
    X_all = df[FULL_FEATURE_COLS].values.astype(np.float64)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Company Predictor",
            "Model Performance",
            "SHAP Explainability",
            "Error Analysis",
            "Dataset Explorer",
        ]
    )

    with tab1:
        st.header("Company credit rating (multicategory)")

        col1, col2 = st.columns([1, 2])

        with col1:
            unique_companies = df.drop_duplicates(subset=["ticker"]).sort_values(
                "company_name"
            )
            company_options = [
                f"{row['company_name']} ({row['ticker']})"
                for _, row in unique_companies.iterrows()
            ]
            selected = st.selectbox("Select a company", company_options)

            if selected:
                ticker = selected.split("(")[-1].rstrip(")")
                company_rows = df[df["ticker"] == ticker].sort_values(
                    "fiscal_year", ascending=False
                )

                if len(company_rows) > 1:
                    years = sorted(company_rows["fiscal_year"].unique(), reverse=True)
                    selected_year = st.selectbox("Fiscal year", years)
                    company_row = company_rows[
                        company_rows["fiscal_year"] == selected_year
                    ].iloc[0]
                else:
                    company_row = company_rows.iloc[0]

                row_pos = int(company_row.name)

                st.markdown("#### Financial ratios")
                for feat in ["liquid", "cumprof", "profitab", "leverage"]:
                    val = company_row[feat]
                    label = FEATURE_LABELS[feat]
                    color = "green" if val > 0.1 else ("orange" if val > 0 else "red")
                    st.metric(label, f"{val:.4f}")

                st.markdown("#### KAM dummies (audit topics)")
                for feat in ["GCKAM", "REVKAM", "ASSETKAM", "LIABKAM", "OTHERKAM"]:
                    st.metric(FEATURE_LABELS[feat], int(company_row[feat]))

                st.markdown("#### News / FinBERT")
                st.metric(FEATURE_LABELS["sentiment_mean"], f"{company_row['sentiment_mean']:.4f}")
                st.metric(FEATURE_LABELS["news_count"], int(company_row["news_count"]))

        with col2:
            if selected:
                X_single = X_all[row_pos : row_pos + 1]
                pred_enc = int(model.predict(X_single)[0])
                proba = model.predict_proba(X_single)[0]
                confidence = float(np.max(proba))
                pred_cat = le_ml.inverse_transform([pred_enc])[0]
                labels_order = list(le_ml.classes_)
                actual_cat = company_row["rating_category"]

                st.markdown("#### ML prediction (XGBoost)")
                st.metric("Predicted category", pred_cat, delta=f"{confidence:.0%} confidence")
                st.caption("Class probabilities")
                st.json({labels_order[i]: float(proba[i]) for i in range(len(proba))})
                match = "Yes" if pred_cat == actual_cat else "No"
                st.metric("Actual category", actual_cat, delta=f"Match: {match}")

                st.markdown("#### SHAP (waterfall for predicted class)")
                sv_mats = class_rows
                sv_row = np.asarray(sv_mats[pred_enc][row_pos]).ravel()
                ev = explainer.expected_value
                base = float(ev[pred_enc]) if hasattr(ev, "__len__") else float(ev)
                fnames = [FEATURE_LABELS.get(f, f) for f in FULL_FEATURE_COLS]
                explanation = shap.Explanation(
                    values=sv_row,
                    base_values=base,
                    data=X_single[0],
                    feature_names=fnames,
                )
                fig_w, _ = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(explanation, show=False)
                plt.tight_layout()
                st.pyplot(fig_w)
                plt.close(fig_w)

                st.markdown("#### Credit verdict (template, same inputs as `llm_verdict.py`)")
                verdict = generate_verdict_for_row(company_row, pred_cat, confidence)
                st.markdown(f"**Overall:** {verdict['overall_assessment']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strengths**")
                    for s in verdict["strengths"]:
                        st.markdown(f"- {s}")
                with c2:
                    st.markdown("**Weaknesses**")
                    for w in verdict["weaknesses"]:
                        st.markdown(f"- {w}")
                st.markdown("**Key risks**")
                for r in verdict["key_risks"]:
                    st.markdown(f"- {r}")
                with st.expander("Prediction analysis"):
                    st.write(verdict["prediction_analysis"])

    with tab2:
        st.header("Model performance")

        st.markdown(
            "**Current multisource panel (14 features, 4-class `rating_category`):** "
            "Stratified 5-fold CV comparing XGBoost to Random Forest, Extra Trees, "
            "sklearn Gradient Boosting, HistGradientBoosting, trees, linear models, kNN, MLP."
        )
        bench_png = FIGURES_DIR / "multisource_model_comparison.png"
        if bench_png.exists():
            st.image(str(bench_png), caption="CV accuracy — multisource benchmark")
        bench_json = RESULTS_DIR / "multisource_model_comparison.json"
        if bench_json.exists():
            with open(bench_json, encoding="utf-8") as f:
                bench = json.load(f)
            st.markdown("#### Ranking by CV accuracy")
            rows = []
            for name in bench.get("ranking_by_accuracy", []):
                m = bench["models"].get(name, {})
                rows.append(
                    {
                        "Model": name,
                        "Accuracy (mean)": f"{m.get('accuracy_mean', 0):.1%}",
                        "Accuracy (std)": f"{m.get('accuracy_std', 0):.3f}",
                        "F1 macro (mean)": f"{m.get('f1_macro_mean', 0):.3f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("Full benchmark JSON"):
                st.json(bench)

        st.markdown("---")
        st.markdown(
            "**XGBoost feature ablation (same data pipeline):** "
            "`results/full_model_comparison.json` (financials only → +KAMs → +sentiment → full)."
        )
        full_path = RESULTS_DIR / "full_model_comparison.json"
        if full_path.exists():
            with open(full_path, encoding="utf-8") as f:
                st.json(json.load(f))

        st.markdown("---")
        st.markdown("**Legacy figure (older binary / larger-sample experiments):**")
        fig_path = FIGURES_DIR / "model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Historical multi-model comparison (GroupKFold)")

    with tab3:
        st.header("SHAP (full multisource XGBoost)")
        st.caption("Regenerate with: `PYTHONPATH=. python models/shap_explainability.py`")

        c1, c2 = st.columns(2)
        with c1:
            p = FIGURES_DIR / "shap_beeswarm.png"
            if p.exists():
                st.image(str(p), caption="SHAP beeswarm (majority-class projection)")
        with c2:
            p = FIGURES_DIR / "shap_bar_importance.png"
            if p.exists():
                st.image(str(p), caption="Mean |SHAP| (14 features)")

        st.markdown("#### Feature interactions")
        c3, c4, c5 = st.columns(3)
        with c3:
            p = FIGURES_DIR / "shap_dependence_leverage_profitab.png"
            if p.exists():
                st.image(str(p), caption="Leverage × profitability")
        with c4:
            p = FIGURES_DIR / "shap_dependence_liquid_cumprof.png"
            if p.exists():
                st.image(str(p), caption="Liquidity × cumulative profit")
        with c5:
            p = FIGURES_DIR / "shap_dependence_news_sentiment.png"
            if p.exists():
                st.image(str(p), caption="News count × FinBERT mean")

        st.markdown("#### Per-company waterfalls (sample)")
        wfs = sorted(FIGURES_DIR.glob("shap_waterfall_*.png"))
        if wfs:
            cols = st.columns(2)
            for i, wf in enumerate(wfs):
                with cols[i % 2]:
                    st.image(str(wf), caption=wf.stem.replace("_", " "))

        rp = RESULTS_DIR / "shap_report.json"
        if rp.exists():
            with open(rp, encoding="utf-8") as f:
                shap_report = json.load(f)
            st.markdown("#### Global ranking (`results/shap_report.json`)")
            ranking = pd.DataFrame(
                [
                    {
                        "Rank": info["rank"],
                        "Feature": feat,
                        "Mean |SHAP|": f"{info['mean_abs_shap']:.4f}",
                        "Description": info["description"],
                    }
                    for feat, info in sorted(
                        shap_report["global_importance"].items(),
                        key=lambda x: x[1]["rank"],
                    )
                ]
            )
            st.dataframe(ranking, use_container_width=True, hide_index=True)

    with tab4:
        st.header("Error analysis (multisource XGBoost CV)")
        for name, cap in [
            ("error_scatter.png", "Misclassified vs correct (PCA space, CV)"),
            ("error_patterns.png", "Actual → predicted confusion pairs"),
            ("confidence_dist.png", "Max class probability (CV)"),
        ]:
            p = FIGURES_DIR / name
            if p.exists():
                st.image(str(p), caption=cap)
        ep = RESULTS_DIR / "error_analysis.json"
        if ep.exists():
            with open(ep, encoding="utf-8") as f:
                st.json(json.load(f))

    with tab5:
        st.header("Dataset explorer")
        p = FIGURES_DIR / "pca_scatter.png"
        if p.exists():
            st.image(str(p), caption="PCA on 14 multisource features (standardized)")

        show_cols = (
            ["ticker", "company_name", "rating_agency", "rating", "fiscal_year", "rating_category"]
            + FULL_FEATURE_COLS
        )
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            df[show_cols].sort_values(["ticker", "fiscal_year"]),
            use_container_width=True,
            hide_index=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Companies", df["ticker"].nunique())
        c3.metric("Categories", df["rating_category"].nunique())

    st.markdown("---")
    st.markdown(
        "*FYP: multisource ML + FinBERT news + KAM features + SHAP + verdicts*"
    )


if __name__ == "__main__":
    main()
