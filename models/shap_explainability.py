"""
SHAP Explainability for Credit Rating Predictions.

Generates per-prediction waterfall plots, global feature importance,
and interaction effects using SHapley Additive exPlanations.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_training_data_v2.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES = ['liquid', 'cumprof', 'profitab', 'leverage']
FEATURE_LABELS = {
    'liquid': 'Liquidity\n(WC/TA)',
    'cumprof': 'Cumulative Profit\n(RE/TA)',
    'profitab': 'Profitability\n(EBIT/TA)',
    'leverage': 'Leverage\n(Equity/Liabilities)',
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


def load_and_train():
    """Load data and train the Gradient Boosting model on full dataset."""
    df = pd.read_csv(DATA_FILE)
    df['target'] = df['rating'].apply(to_binary)

    X = df[FEATURES]
    y = df['target'].values

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    return model, X, y, df


def compute_shap_values(model, X):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_global_importance(shap_values, X, save_path=None):
    """SHAP beeswarm plot showing global feature importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    X_display = X.rename(columns=FEATURE_LABELS)
    shap.summary_plot(shap_values, X_display, show=False, plot_size=None)

    plt.title("SHAP Feature Importance: Impact on Credit Rating Prediction",
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("SHAP Value (impact on model output)\n← Investment Grade | Speculative Grade →",
               fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_bar_importance(shap_values, X, save_path=None):
    """SHAP bar plot of mean absolute feature importance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURES]
    sorted_idx = np.argsort(mean_abs)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_idx)))

    ax.barh(range(len(sorted_idx)),
            mean_abs[sorted_idx],
            color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)",
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_waterfall(explainer, shap_values, X, idx, company_info, save_path=None):
    """SHAP waterfall plot for a single prediction."""
    fig, ax = plt.subplots(figsize=(10, 5))

    feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURES]

    ev = explainer.expected_value
    if hasattr(ev, '__len__'):
        ev = float(ev[0]) if len(ev) == 1 else float(ev[1])

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=ev,
        data=X.iloc[idx].values,
        feature_names=feature_names,
    )

    shap.plots.waterfall(explanation, show=False)

    company = company_info.get('company_name', 'Unknown')[:30]
    ticker = company_info.get('ticker', '')
    rating = company_info.get('rating', '')
    year = company_info.get('fiscal_year', '')
    plt.title(f"SHAP Waterfall: {company} ({ticker}) -- {rating}, FY{year}",
              fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_dependence(shap_values, X, feature_idx, interaction_idx, save_path=None):
    """SHAP dependence plot showing feature interactions."""
    fig, ax = plt.subplots(figsize=(8, 6))

    feature_name = FEATURE_LABELS.get(FEATURES[feature_idx], FEATURES[feature_idx])
    interact_name = FEATURE_LABELS.get(FEATURES[interaction_idx], FEATURES[interaction_idx])

    X_display = X.rename(columns=FEATURE_LABELS)
    shap.dependence_plot(
        feature_name, shap_values, X_display,
        interaction_index=interact_name,
        show=False, ax=ax
    )
    ax.set_title(f"SHAP Dependence: {feature_name.replace(chr(10), ' ')} "
                 f"(colored by {interact_name.replace(chr(10), ' ')})",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def generate_shap_report(shap_values, X, df):
    """Generate a structured report of SHAP analysis."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_order = np.argsort(mean_abs)[::-1]

    report = {
        'global_importance': {},
        'per_company_top_factors': [],
    }

    for rank, idx in enumerate(importance_order):
        report['global_importance'][FEATURES[idx]] = {
            'rank': rank + 1,
            'mean_abs_shap': float(mean_abs[idx]),
            'description': FEATURE_LABELS[FEATURES[idx]].replace('\n', ' '),
        }

    seen_tickers = set()
    for i in range(len(df)):
        ticker = df.iloc[i]['ticker']
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)

        top_feature_idx = np.argmax(np.abs(shap_values[i]))
        direction = "increases" if shap_values[i][top_feature_idx] > 0 else "decreases"

        report['per_company_top_factors'].append({
            'ticker': ticker,
            'company': df.iloc[i]['company_name'],
            'rating': df.iloc[i]['rating'],
            'top_feature': FEATURES[top_feature_idx],
            'shap_value': float(shap_values[i][top_feature_idx]),
            'feature_value': float(X.iloc[i].values[top_feature_idx]),
            'effect': f"{direction} speculative-grade probability",
        })

    return report


def main():
    print("=" * 70)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading data and training model...")
    model, X, y, df = load_and_train()
    print(f"   Samples: {len(X)}, Features: {len(FEATURES)}")

    print("\n2. Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X)
    print(f"   SHAP values shape: {shap_values.shape}")
    ev = explainer.expected_value
    if hasattr(ev, '__len__'):
        ev = ev[0] if len(ev) == 1 else ev
    print(f"   Base value (expected value): {ev}")

    print("\n3. Generating plots...")

    plot_global_importance(
        shap_values, X,
        save_path=FIGURES_DIR / "shap_beeswarm.png"
    )

    plot_bar_importance(
        shap_values, X,
        save_path=FIGURES_DIR / "shap_bar_importance.png"
    )

    interesting_companies = []
    seen = set()
    for i in range(len(df)):
        ticker = df.iloc[i]['ticker']
        if ticker not in seen:
            seen.add(ticker)
            interesting_companies.append(i)
        if len(interesting_companies) >= 5:
            break

    for idx in interesting_companies:
        row = df.iloc[idx]
        safe_name = row['ticker'].replace('.', '_')
        plot_waterfall(
            explainer, shap_values, X, idx,
            row.to_dict(),
            save_path=FIGURES_DIR / f"shap_waterfall_{safe_name}.png"
        )

    plot_dependence(
        shap_values, X, 3, 2,
        save_path=FIGURES_DIR / "shap_dependence_leverage_profitab.png"
    )
    plot_dependence(
        shap_values, X, 0, 1,
        save_path=FIGURES_DIR / "shap_dependence_liquid_cumprof.png"
    )

    print("\n4. Generating SHAP report...")
    report = generate_shap_report(shap_values, X, df)

    report_file = RESULTS_DIR / "shap_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"   Saved: {report_file}")

    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE (by mean |SHAP|):")
    print("=" * 70)
    for feat, info in sorted(report['global_importance'].items(),
                              key=lambda x: x[1]['rank']):
        print(f"  #{info['rank']}: {feat:<12} mean|SHAP|={info['mean_abs_shap']:.4f}")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
