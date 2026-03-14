"""
Streamlit Demo: Credit Rating Prediction System
Classical ML + LLM Verdict Generation for Saudi Exchange Companies
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_training_data_v2.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
VERDICTS_DIR = RESULTS_DIR / "verdicts"

FEATURES = ['liquid', 'cumprof', 'profitab', 'leverage']
FEATURE_LABELS = {
    'liquid': 'Liquidity (WC / Total Assets)',
    'cumprof': 'Cumulative Profitability (RE / Total Assets)',
    'profitab': 'Profitability (EBIT / Total Assets)',
    'leverage': 'Leverage (Book Equity / Total Liabilities)',
}

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_binary(rating):
    return 0 if RATING_TO_NUMERIC.get(rating, 0) >= 15 else 1


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['target'] = df['rating'].apply(to_binary)
    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES].values
    y = df['target'].values
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return model, explainer, shap_values


@st.cache_data
def get_cv_predictions(df):
    X = df[FEATURES].values
    y = df['target'].values
    groups = df['ticker'].values
    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42
    )
    y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    y_proba = cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict_proba')
    return y_pred, y_proba


@st.cache_data
def load_verdicts():
    path = VERDICTS_DIR / "all_verdicts.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_verdict_for_row(row, pred_class, confidence):
    """Generate a template verdict for display."""
    from models.llm_verdict import generate_template_verdict
    pred_rating = "A" if pred_class == 0 else "BBB"
    return generate_template_verdict(
        row.to_dict(), pred_rating, confidence, row.get('rating')
    )


def main():
    st.set_page_config(
        page_title="Saudi Credit Rating Predictor",
        page_icon="📊",
        layout="wide",
    )

    st.title("Credit Rating Prediction System")
    st.markdown("**Classical ML + LLM Verdict Generation** for Saudi Exchange Companies (2021-2024)")
    st.markdown("---")

    df = load_data()
    model, explainer, shap_values_all = train_model(df)
    y_pred, y_proba = get_cv_predictions(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Company Predictor",
        "Model Performance",
        "SHAP Explainability",
        "Error Analysis",
        "Dataset Explorer",
    ])

    # ─── TAB 1: Company Predictor ───
    with tab1:
        st.header("Company Credit Rating Prediction")

        col1, col2 = st.columns([1, 2])

        with col1:
            unique_companies = df.drop_duplicates(subset=['ticker']).sort_values('company_name')
            company_options = [
                f"{row['company_name']} ({row['ticker']})"
                for _, row in unique_companies.iterrows()
            ]
            selected = st.selectbox("Select a Company", company_options)

            if selected:
                ticker = selected.split('(')[-1].rstrip(')')
                company_rows = df[df['ticker'] == ticker].sort_values('fiscal_year', ascending=False)

                if len(company_rows) > 1:
                    years = sorted(company_rows['fiscal_year'].unique(), reverse=True)
                    selected_year = st.selectbox("Fiscal Year", years)
                    company_row = company_rows[company_rows['fiscal_year'] == selected_year].iloc[0]
                else:
                    company_row = company_rows.iloc[0]

                st.markdown("#### Financial Ratios (Altman Z''-Score)")
                for feat in FEATURES:
                    val = company_row[feat]
                    label = FEATURE_LABELS[feat]
                    color = "green" if val > 0.1 else ("orange" if val > 0 else "red")
                    st.metric(label, f"{val:.4f}")

        with col2:
            if selected:
                row_idx = company_row.name
                X_single = df.loc[[row_idx], FEATURES].values

                pred = model.predict(X_single)[0]
                proba = model.predict_proba(X_single)[0]
                confidence = float(max(proba))
                pred_label = "Investment Grade (A- and above)" if pred == 0 else "Speculative Grade (BBB+ and below)"
                actual_label = "Investment Grade" if company_row['target'] == 0 else "Speculative Grade"

                st.markdown("#### ML Prediction")
                pred_col, actual_col = st.columns(2)
                with pred_col:
                    st.metric("Predicted Class", pred_label.split(' (')[0],
                              delta=f"Confidence: {confidence:.0%}")
                with actual_col:
                    match = "Correct" if pred == company_row['target'] else "Incorrect"
                    st.metric("Actual Rating",
                              f"{company_row['rating']} ({actual_label})",
                              delta=match,
                              delta_color="normal" if match == "Correct" else "inverse")

                st.markdown("#### SHAP Explanation")
                sv = shap_values_all[row_idx]
                ev = explainer.expected_value
                if hasattr(ev, '__len__'):
                    ev = float(ev[0]) if len(ev) == 1 else float(ev[1])

                feature_names = [FEATURE_LABELS[f] for f in FEATURES]
                explanation = shap.Explanation(
                    values=sv,
                    base_values=ev,
                    data=X_single[0],
                    feature_names=feature_names,
                )

                fig_w, ax_w = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig_w)
                plt.close(fig_w)

                st.markdown("#### LLM Credit Verdict")
                verdict = generate_verdict_for_row(company_row, pred, confidence)

                st.markdown(f"**Overall Assessment:** {verdict['overall_assessment']}")

                str_col, weak_col = st.columns(2)
                with str_col:
                    st.markdown("**Strengths:**")
                    for s in verdict['strengths']:
                        st.markdown(f"- {s}")
                with weak_col:
                    st.markdown("**Weaknesses:**")
                    for w in verdict['weaknesses']:
                        st.markdown(f"- {w}")

                st.markdown("**Key Risks:**")
                for r in verdict['key_risks']:
                    st.markdown(f"- {r}")

                with st.expander("Prediction Analysis"):
                    st.write(verdict['prediction_analysis'])

    # ─── TAB 2: Model Performance ───
    with tab2:
        st.header("Model Performance Comparison")

        fig_path = FIGURES_DIR / "model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="8-Model Comparison (GroupKFold CV)")

        col1, col2 = st.columns(2)
        with col1:
            fig_path = FIGURES_DIR / "confusion_matrix_gb.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Confusion Matrix (Gradient Boosting)")
        with col2:
            fig_path = FIGURES_DIR / "rating_distribution.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Rating Distribution")

        col3, col4 = st.columns(2)
        with col3:
            fig_path = FIGURES_DIR / "decision_tree.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Decision Tree Visualization")
        with col4:
            fig_path = FIGURES_DIR / "feature_distributions.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Feature Distributions by Class")

        results_path = RESULTS_DIR / "model_comparison_final.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            st.markdown("#### Detailed Results")
            results_df = pd.DataFrame([
                {"Model": k, "Accuracy": f"{v:.1%}"}
                for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ─── TAB 3: SHAP Explainability ───
    with tab3:
        st.header("SHAP Feature Importance Analysis")

        col1, col2 = st.columns(2)
        with col1:
            fig_path = FIGURES_DIR / "shap_beeswarm.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="SHAP Beeswarm (Global Feature Impact)")
        with col2:
            fig_path = FIGURES_DIR / "shap_bar_importance.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Mean |SHAP| Feature Importance")

        st.markdown("#### Feature Interactions")
        col3, col4 = st.columns(2)
        with col3:
            fig_path = FIGURES_DIR / "shap_dependence_leverage_profitab.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Leverage vs Profitability Interaction")
        with col4:
            fig_path = FIGURES_DIR / "shap_dependence_liquid_cumprof.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Liquidity vs Cumulative Profit Interaction")

        st.markdown("#### Per-Company SHAP Waterfall Plots")
        waterfall_files = sorted(FIGURES_DIR.glob("shap_waterfall_*.png"))
        if waterfall_files:
            cols = st.columns(min(3, len(waterfall_files)))
            for i, wf in enumerate(waterfall_files):
                with cols[i % len(cols)]:
                    st.image(str(wf), caption=wf.stem.replace('shap_waterfall_', '').replace('_', '.'))

        report_path = RESULTS_DIR / "shap_report.json"
        if report_path.exists():
            with open(report_path) as f:
                shap_report = json.load(f)
            st.markdown("#### Global Feature Ranking")
            ranking = pd.DataFrame([
                {
                    "Rank": info['rank'],
                    "Feature": feat,
                    "Mean |SHAP|": f"{info['mean_abs_shap']:.4f}",
                    "Description": info['description'],
                }
                for feat, info in sorted(
                    shap_report['global_importance'].items(),
                    key=lambda x: x[1]['rank']
                )
            ])
            st.dataframe(ranking, use_container_width=True, hide_index=True)

    # ─── TAB 4: Error Analysis ───
    with tab4:
        st.header("Error Analysis")

        col1, col2 = st.columns(2)
        with col1:
            fig_path = FIGURES_DIR / "error_scatter.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Misclassified Companies in Feature Space")
        with col2:
            fig_path = FIGURES_DIR / "error_patterns.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Error Pattern Categories")

        fig_path = FIGURES_DIR / "confidence_dist.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Model Confidence Distribution")

        error_path = RESULTS_DIR / "error_analysis.json"
        if error_path.exists():
            with open(error_path) as f:
                error_data = json.load(f)

            st.markdown("#### Summary")
            summary = error_data['summary']
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", summary['total_samples'])
            m2.metric("Accuracy", f"{summary['accuracy']:.1%}")
            m3.metric("False Positives", summary['false_positives'])
            m4.metric("False Negatives", summary['false_negatives'])

            st.markdown("#### Error Patterns")
            for pattern, info in error_data['error_patterns'].items():
                st.markdown(f"- **{pattern.replace('_', ' ').title()}**: "
                           f"{info['count']} errors ({info['pct']:.0%}) -- {info['explanation']}")

            st.markdown("#### Misclassified Companies")
            for m in error_data['misclassified_companies']:
                with st.expander(f"{m['ticker']} ({m['company'][:40]}) -- "
                                f"Actual: {m['actual_rating']}, Predicted: {m['predicted_class']}"):
                    st.write(f"**Error Type:** {m['error_type']}")
                    st.write(f"**Confidence:** {m['confidence']:.1%}")
                    st.write(f"**Boundary Rating:** {'Yes' if m['is_boundary_rating'] else 'No'}")
                    st.write(f"**Agency Disagreement:** {'Yes' if m['has_agency_disagreement'] else 'No'}")
                    st.write("**Likely Reasons:**")
                    for reason in m['likely_reasons']:
                        st.write(f"  - {reason}")

        fig_path = FIGURES_DIR / "agency_disagreement.png"
        if fig_path.exists():
            st.markdown("#### Inter-Agency Rating Disagreement")
            st.image(str(fig_path))

    # ─── TAB 5: Dataset Explorer ───
    with tab5:
        st.header("Dataset Explorer")

        fig_path = FIGURES_DIR / "pca_scatter.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Companies in Financial Ratio Space (PCA)")

        st.markdown("#### Full Dataset")
        st.dataframe(
            df[['ticker', 'company_name', 'rating_agency', 'rating',
                'fiscal_year'] + FEATURES + ['target']].sort_values(
                ['ticker', 'fiscal_year']),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Unique Companies", df['ticker'].nunique())
        col3.metric("Rating Agencies", df['rating_agency'].nunique())

        st.markdown("#### Feature Statistics")
        stats = df[FEATURES].describe().T
        stats.index = [FEATURE_LABELS.get(f, f) for f in stats.index]
        st.dataframe(stats.round(4), use_container_width=True)

    st.markdown("---")
    st.markdown(
        "*Built for FYP: Credit Rating Prediction for Saudi Exchange Companies "
        "using Classical ML + LLM Verdict Generation*"
    )


if __name__ == "__main__":
    main()
