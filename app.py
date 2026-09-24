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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
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


@st.cache_data
def compute_cv_error_bundle(df: pd.DataFrame):
    """
    Stratified CV predictions matching models/generate_pipeline_figures.py so the
    Error Analysis tab works without pre-generated PNG/JSON (e.g. minimal Docker image).
    """
    X = df[FULL_FEATURE_COLS].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["rating_category"].values)
    classes = list(le.classes_)
    counts = np.bincount(y)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        n_splits = 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=1,
    )
    y_pred = cross_val_predict(cv_model, X, y, cv=cv)
    y_proba = cross_val_predict(cv_model, X, y, cv=cv, method="predict_proba")
    max_proba = np.max(y_proba, axis=1)
    acc = accuracy_score(y, y_pred)
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=42).fit_transform(Xs)
    correct = y_pred == y
    pairs: dict[str, int] = {}
    for actual, pred_i in zip(df["rating_category"].values, y_pred):
        pred = le.inverse_transform([int(pred_i)])[0]
        if actual != pred:
            key = f"{actual} → {pred}"
            pairs[key] = pairs.get(key, 0) + 1
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
                "confidence": float(max_proba[i]),
            }
        )
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(classes)))
    payload = {
        "generated_for": "multisource XGBoost 14 features, 4-class rating_category (in-app CV)",
        "n_samples": int(len(df)),
        "cv_folds": n_splits,
        "cv_accuracy": float(acc),
        "classes": classes,
        "summary": {
            "total_samples": int(len(df)),
            "correct": int(np.sum(correct)),
            "incorrect": int(np.sum(~correct)),
            "accuracy": float(acc),
        },
        "misclassified": mis_rows,
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
    }
    return {
        "Z": Z,
        "correct": correct,
        "max_proba": max_proba,
        "pairs": pairs,
        "payload": payload,
    }


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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Company Predictor",
            "Model Performance",
            "SHAP Explainability",
            "Error Analysis",
            "Dataset Explorer",
            "Pipeline outputs",
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
            "**Superset benchmark (21 features):** the full 12-column KAM/firm block "
            "instead of the five KAM dummies, same 45 rows — `models/xgboost_all_features.py`. "
            "This is where the repository's highest accuracy comes from; see §15.5.1 of "
            "`docs/PROJECT_REPORT.md` for why the ranking should be read with its macro-F1 "
            "and its pinned library versions."
        )
        all_feat_path = RESULTS_DIR / "all_features_model_results.json"
        if all_feat_path.exists():
            with open(all_feat_path, encoding="utf-8") as f:
                all_feat = json.load(f)

            all_feat_png = FIGURES_DIR / "all_features_model_comparison.png"
            if all_feat_png.exists():
                st.image(str(all_feat_png), caption="CV accuracy — 21-feature benchmark")

            bench21 = all_feat.get("benchmark_21_features", {})
            ranking = all_feat.get("benchmark_ranking", list(bench21))
            rows21 = [
                {
                    "Model": name,
                    "Accuracy (mean)": f"{bench21[name].get('accuracy_mean', 0):.1%}",
                    "Accuracy (std)": f"{bench21[name].get('accuracy_std', 0):.3f}",
                    "F1 macro (mean)": f"{bench21[name].get('f1_macro_mean', 0):.3f}",
                }
                for name in ranking
                if name in bench21
            ]
            if rows21:
                st.dataframe(pd.DataFrame(rows21), use_container_width=True, hide_index=True)

            ablations = all_feat.get("ablations", {})
            if ablations:
                st.markdown("#### XGBoost ablation across feature blocks (21-feature superset)")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Feature set": name,
                                "n features": a.get("n_features"),
                                "Accuracy (mean)": f"{a.get('accuracy_mean', 0):.1%}",
                                "Accuracy (std)": f"{a.get('accuracy_std', 0):.3f}",
                            }
                            for name, a in ablations.items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with st.expander("Full 21-feature benchmark JSON"):
                st.json(all_feat)
        else:
            st.caption(
                "`results/all_features_model_results.json` not found — regenerate with: "
                "`PYTHONPATH=. python models/xgboost_all_features.py`."
            )

    with tab3:
        st.header("SHAP (full multisource XGBoost)")
        st.caption(
            "Live plots use the XGBoost trained in this session. "
            "Pre-rendered PNGs + `shap_report.json` come from "
            "`PYTHONPATH=. python models/shap_explainability.py`."
        )

        fnames = [FEATURE_LABELS.get(f, f) for f in FULL_FEATURE_COLS]
        imp = np.mean([np.abs(r).mean(axis=0) for r in class_rows], axis=0)
        ord_idx = np.argsort(imp)
        fig_imp, ax_imp = plt.subplots(figsize=(8, max(3.5, 0.32 * len(imp) + 1)))
        ax_imp.barh(
            np.arange(len(imp)),
            imp[ord_idx],
            color="darkslategray",
            edgecolor="white",
        )
        ax_imp.set_yticks(np.arange(len(imp)))
        ax_imp.set_yticklabels([fnames[i] for i in ord_idx], fontsize=9)
        ax_imp.set_xlabel("Mean |SHAP| (average over classes and samples)")
        ax_imp.set_title("Global importance (in-app model)")
        fig_imp.tight_layout()
        st.pyplot(fig_imp, clear_figure=True)
        plt.close(fig_imp)

        st.markdown("#### SHAP summary (first outcome class)")
        fig_sum = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            class_rows[0],
            X_all,
            feature_names=fnames,
            max_display=len(FULL_FEATURE_COLS),
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig_sum, clear_figure=True)
        plt.close(fig_sum)

        st.markdown("### Pre-rendered figures (repository)")
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
        st.caption(
            "Computed live with stratified CV (same setup as `generate_pipeline_figures.py`) "
            "so this tab works in Docker even when `figures/*.png` are not baked into the image."
        )
        eb = compute_cv_error_bundle(df)
        Z, correct, max_proba, pairs = eb["Z"], eb["correct"], eb["max_proba"], eb["pairs"]

        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.scatter(
            Z[~correct, 0],
            Z[~correct, 1],
            c="crimson",
            s=55,
            label="Misclassified",
            alpha=0.9,
        )
        ax1.scatter(
            Z[correct, 0],
            Z[correct, 1],
            c="0.75",
            s=35,
            label="Correct",
            alpha=0.6,
        )
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.legend()
        ax1.set_title("Misclassifications in PCA space (CV out-of-fold)")
        fig1.tight_layout()
        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.35 * max(len(pairs), 1) + 1)))
        if pairs:
            items = sorted(pairs.items(), key=lambda x: -x[1])
            labs = [k for k, _ in items]
            vals = [v for _, v in items]
            ax2.barh(labs[::-1], vals[::-1], color="coral")
            ax2.set_xlabel("Count")
        else:
            ax2.text(0.5, 0.5, "No CV errors", ha="center", va="center")
        ax2.set_title("Misclassification patterns (actual → predicted)")
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(max_proba, bins=12, color="teal", edgecolor="white", alpha=0.85)
        ax3.set_xlabel("Max predicted class probability (CV)")
        ax3.set_ylabel("Count")
        ax3.set_title("Model confidence (CV folds)")
        fig3.tight_layout()
        st.pyplot(fig3, clear_figure=True)
        plt.close(fig3)

        st.markdown("#### `error_analysis` summary (JSON)")
        st.json(eb["payload"])

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

    with tab6:
        st.header("Pipeline outputs (saved figures & JSON)")
        st.caption(
            "Files written by `scripts/regenerate_artifacts.py` and the underlying model scripts. "
            "Full descriptions: **`docs/FIGURES_AND_RESULTS.md`**."
        )

        st.subheader("Diagnostic figures (`figures/`)")
        st.markdown(
            "These come from `models/generate_pipeline_figures.py` (multisource XGBoost, stratified CV)."
        )
        _diag_pairs = [
            (
                FIGURES_DIR / "rating_distribution.png",
                "`figures/rating_distribution.png` — row counts per `rating_category`.",
            ),
            (
                FIGURES_DIR / "confusion_matrix_multisource.png",
                "`figures/confusion_matrix_multisource.png` — CV confusion matrix (same as `confusion_matrix_gb.png`).",
            ),
            (
                FIGURES_DIR / "pca_multisource.png",
                "`figures/pca_multisource.png` — PCA of standardized 14 features (same plot as `pca_scatter.png`).",
            ),
            (
                FIGURES_DIR / "feature_distributions_multisource.png",
                "`figures/feature_distributions_multisource.png` — four financial ratios by category.",
            ),
            (
                FIGURES_DIR / "confidence_distribution_multisource.png",
                "`figures/confidence_distribution_multisource.png` — histogram of max CV predicted probability.",
            ),
            (
                FIGURES_DIR / "error_scatter_multisource.png",
                "`figures/error_scatter_multisource.png` — correct vs misclassified rows in PCA space.",
            ),
            (
                FIGURES_DIR / "error_patterns_multisource.png",
                "`figures/error_patterns_multisource.png` — bar chart of actual→predicted error pairs.",
            ),
        ]
        for i in range(0, len(_diag_pairs), 2):
            dc1, dc2 = st.columns(2)
            with dc1:
                pp, cap = _diag_pairs[i]
                if pp.exists():
                    st.image(str(pp), caption=cap)
                else:
                    st.caption(f"Missing: `{pp.name}`")
            with dc2:
                if i + 1 < len(_diag_pairs):
                    pp2, cap2 = _diag_pairs[i + 1]
                    if pp2.exists():
                        st.image(str(pp2), caption=cap2)
                    else:
                        st.caption(f"Missing: `{pp2.name}`")

        st.subheader("Benchmark & SHAP PNGs")
        st.markdown(
            "Benchmark: `models/evaluate_multisource_models.py`. SHAP static plots: `models/shap_explainability.py`."
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            bp = FIGURES_DIR / "multisource_model_comparison.png"
            if bp.exists():
                st.image(str(bp), caption="`figures/multisource_model_comparison.png` (also copied to `model_comparison.png`).")
            else:
                st.caption("Missing: `multisource_model_comparison.png`")
        with bc2:
            st.markdown("**SHAP (pre-rendered)**")
            for stem, txt in [
                ("shap_beeswarm.png", "`figures/shap_beeswarm.png`"),
                ("shap_bar_importance.png", "`figures/shap_bar_importance.png`"),
            ]:
                sp = FIGURES_DIR / stem
                if sp.exists():
                    st.image(str(sp), caption=txt)
            st.caption("Dependence plots: `shap_dependence_*.png`. Waterfalls: `shap_waterfall_<ticker>_<year>.png` — see SHAP tab.")

        st.subheader("Result files (`results/`)")
        _json_files = [
            (
                "All-features benchmark (21 features)",
                RESULTS_DIR / "all_features_model_results.json",
                "Nine learners + five ablations on the 45×21 superset (`xgboost_all_features.py`).",
            ),
            (
                "Financials vs combined (extended KAM study)",
                RESULTS_DIR / "combined_model_results.json",
                "`xgboost_with_kams.py` style comparison on the merged financial sample.",
            ),
            (
                "KAM-only panel (full `kams_processed.csv`)",
                RESULTS_DIR / "kams_only_model_results.json",
                "`xgboost_kams_only.py` — 12 KAM / firm-structure features.",
            ),
            (
                "Pipeline CV error snapshot",
                RESULTS_DIR / "error_analysis_multisource.json",
                "Static snapshot from `generate_pipeline_figures.py` (Error Analysis tab is computed live).",
            ),
            (
                "Pipeline run metadata",
                RESULTS_DIR / "pipeline_figures_meta.json",
                "Timestamp, row count, list of figures/JSON written last run.",
            ),
            (
                "SHAP global ranking",
                RESULTS_DIR / "shap_report.json",
                "Mean |SHAP| ranks and feature descriptions (`shap_explainability.py`).",
            ),
            (
                "Verdict quality summary",
                RESULTS_DIR / "verdicts" / "verdict_summary.json",
                "Template verdict QA metrics vs ML labels (`llm_verdict.py`).",
            ),
        ]
        for title, jpath, blurb in _json_files:
            with st.expander(f"{title} — `{jpath.relative_to(PROJECT_ROOT)}`"):
                st.caption(blurb)
                if jpath.exists():
                    with open(jpath, encoding="utf-8") as jf:
                        st.json(json.load(jf))
                else:
                    st.warning("File not found — run `scripts/regenerate_artifacts.py`.")

    st.markdown("---")
    st.markdown(
        "*FYP: multisource ML + FinBERT news + KAM features + SHAP + verdicts*"
    )


if __name__ == "__main__":
    main()
