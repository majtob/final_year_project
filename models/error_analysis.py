"""
Systematic error analysis for credit rating predictions.

Identifies misclassified companies, analyzes reasons for errors,
examines boundary cases and inter-agency disagreement effects.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_training_data_v2.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES = ['liquid', 'cumprof', 'profitab', 'leverage']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_binary(rating):
    return 0 if RATING_TO_NUMERIC.get(rating, 0) >= 15 else 1


def load_data():
    df = pd.read_csv(DATA_FILE)
    df['target'] = df['rating'].apply(to_binary)
    df['rating_numeric'] = df['rating'].map(RATING_TO_NUMERIC)
    return df


def run_predictions(df):
    """Run GroupKFold CV and return predictions + probabilities."""
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


def analyze_misclassifications(df, y_pred, y_proba):
    """Detailed analysis of every misclassified sample."""
    df = df.copy()
    df['predicted'] = y_pred
    df['correct'] = df['target'] == df['predicted']
    df['confidence'] = [max(p) for p in y_proba]
    df['prob_speculative'] = y_proba[:, 1]

    misclassified = df[~df['correct']].copy()
    correct = df[df['correct']].copy()

    analysis = {
        'summary': {
            'total_samples': len(df),
            'correct': int(df['correct'].sum()),
            'incorrect': int((~df['correct']).sum()),
            'accuracy': float(df['correct'].mean()),
        },
        'misclassified_companies': [],
        'error_patterns': {},
    }

    false_positives = misclassified[
        (misclassified['target'] == 0) & (misclassified['predicted'] == 1)
    ]
    false_negatives = misclassified[
        (misclassified['target'] == 1) & (misclassified['predicted'] == 0)
    ]

    analysis['summary']['false_positives'] = len(false_positives)
    analysis['summary']['false_negatives'] = len(false_negatives)
    analysis['summary']['false_positive_desc'] = (
        "Companies that are actually Investment Grade but predicted as Speculative"
    )
    analysis['summary']['false_negative_desc'] = (
        "Companies that are actually Speculative but predicted as Investment Grade"
    )

    for _, row in misclassified.iterrows():
        actual_class = "Investment Grade" if row['target'] == 0 else "Speculative Grade"
        predicted_class = "Investment Grade" if row['predicted'] == 0 else "Speculative Grade"
        error_type = "False Positive" if row['target'] == 0 else "False Negative"

        is_boundary = row['rating'] in ['A-', 'BBB+']

        same_ticker = df[df['ticker'] == row['ticker']]
        has_agency_disagreement = same_ticker['target'].nunique() > 1

        correct_with_same_ticker = correct[correct['ticker'] == row['ticker']]
        has_correct_predictions_too = len(correct_with_same_ticker) > 0

        mean_ratios = df[FEATURES].mean()
        ratio_comparison = {}
        for feat in FEATURES:
            val = row[feat]
            mean_val = mean_ratios[feat]
            is_outlier = abs(val - mean_val) > 2 * df[feat].std()
            ratio_comparison[feat] = {
                'value': float(val),
                'dataset_mean': float(mean_val),
                'dataset_std': float(df[feat].std()),
                'z_score': float((val - mean_val) / max(df[feat].std(), 1e-10)),
                'is_outlier': bool(is_outlier),
            }

        reasons = []
        if is_boundary:
            reasons.append(f"Boundary rating ({row['rating']}): sits at the A-/BBB+ divide "
                           "between Investment and Speculative grade")
        if has_agency_disagreement:
            agencies_for_ticker = same_ticker[['rating_agency', 'rating']].values.tolist()
            reasons.append(f"Agency disagreement: different agencies rate this company "
                           f"differently ({agencies_for_ticker})")
        outlier_feats = [f for f in FEATURES if ratio_comparison[f]['is_outlier']]
        if outlier_feats:
            reasons.append(f"Outlier ratios: {', '.join(outlier_feats)} are >2 std from mean")
        if row['confidence'] < 0.6:
            reasons.append(f"Low model confidence ({row['confidence']:.1%}), "
                           "indicating the model was uncertain")
        if not reasons:
            reasons.append("No obvious single cause; may reflect genuine difficulty "
                           "in classification from financial ratios alone")

        entry = {
            'ticker': row['ticker'],
            'company': row['company_name'],
            'rating_agency': row['rating_agency'],
            'actual_rating': row['rating'],
            'actual_class': actual_class,
            'predicted_class': predicted_class,
            'error_type': error_type,
            'confidence': float(row['confidence']),
            'prob_speculative': float(row['prob_speculative']),
            'is_boundary_rating': is_boundary,
            'has_agency_disagreement': has_agency_disagreement,
            'ratios': ratio_comparison,
            'likely_reasons': reasons,
        }
        analysis['misclassified_companies'].append(entry)

    boundary_errors = sum(1 for m in analysis['misclassified_companies'] if m['is_boundary_rating'])
    agency_errors = sum(1 for m in analysis['misclassified_companies'] if m['has_agency_disagreement'])
    outlier_errors = sum(1 for m in analysis['misclassified_companies']
                         if any(m['ratios'][f]['is_outlier'] for f in FEATURES))
    low_conf_errors = sum(1 for m in analysis['misclassified_companies'] if m['confidence'] < 0.6)

    total_errors = len(analysis['misclassified_companies'])
    analysis['error_patterns'] = {
        'boundary_ratings': {
            'count': boundary_errors,
            'pct': boundary_errors / max(total_errors, 1),
            'explanation': "Errors on companies rated A- or BBB+, which sit at the exact "
                           "boundary between Investment and Speculative grade",
        },
        'agency_disagreement': {
            'count': agency_errors,
            'pct': agency_errors / max(total_errors, 1),
            'explanation': "Errors on companies where different rating agencies disagree "
                           "on the binary classification",
        },
        'outlier_ratios': {
            'count': outlier_errors,
            'pct': outlier_errors / max(total_errors, 1),
            'explanation': "Errors on companies with at least one financial ratio >2 std "
                           "from the dataset mean",
        },
        'low_confidence': {
            'count': low_conf_errors,
            'pct': low_conf_errors / max(total_errors, 1),
            'explanation': "Errors where model confidence was below 60%",
        },
    }

    return analysis, df


def plot_error_analysis(analysis_df, save_path=None):
    """Scatter plot showing correct vs incorrect predictions in ratio space."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (feat_x, feat_y), title in zip(
        axes,
        [('liquid', 'leverage'), ('cumprof', 'profitab')],
        ['Liquidity vs Leverage', 'Cumulative Profit vs Profitability']
    ):
        correct = analysis_df[analysis_df['correct']]
        incorrect = analysis_df[~analysis_df['correct']]

        ax.scatter(correct[feat_x], correct[feat_y],
                   c=['#2ecc71' if t == 0 else '#3498db' for t in correct['target']],
                   s=60, alpha=0.6, marker='o', edgecolors='gray', linewidth=0.5,
                   label='Correct')
        ax.scatter(incorrect[feat_x], incorrect[feat_y],
                   c='red', s=120, alpha=0.8, marker='X', edgecolors='black',
                   linewidth=1, label='Misclassified', zorder=5)

        for _, row in incorrect.iterrows():
            ax.annotate(row['ticker'].replace('.SR', ''),
                       (row[feat_x], row[feat_y]),
                       fontsize=8, fontweight='bold', color='red',
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel(feat_x.upper(), fontsize=11)
        ax.set_ylabel(feat_y.upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    acc = analysis_df['correct'].mean()
    plt.suptitle(f'Error Analysis: Misclassified Companies in Feature Space '
                 f'(Accuracy: {acc:.1%})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_confidence_distribution(analysis_df, save_path=None):
    """Distribution of model confidence for correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    correct = analysis_df[analysis_df['correct']]['confidence']
    incorrect = analysis_df[~analysis_df['correct']]['confidence']

    bins = np.linspace(0.4, 1.0, 20)
    ax.hist(correct, bins=bins, alpha=0.7, color='#2ecc71',
            label=f'Correct (n={len(correct)})', edgecolor='white')
    ax.hist(incorrect, bins=bins, alpha=0.7, color='#e74c3c',
            label=f'Incorrect (n={len(incorrect)})', edgecolor='white')

    ax.axvline(x=correct.mean(), color='green', linestyle='--', alpha=0.8,
               label=f'Mean correct: {correct.mean():.2f}')
    if len(incorrect) > 0:
        ax.axvline(x=incorrect.mean(), color='red', linestyle='--', alpha=0.8,
                   label=f'Mean incorrect: {incorrect.mean():.2f}')

    ax.set_xlabel('Model Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Model Confidence: Correct vs Incorrect Predictions',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_error_pattern_summary(analysis, save_path=None):
    """Bar chart of error pattern categories."""
    patterns = analysis['error_patterns']

    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(patterns.keys())
    counts = [patterns[n]['count'] for n in names]
    labels = [n.replace('_', ' ').title() for n in names]

    colors = ['#3498db', '#e67e22', '#9b59b6', '#e74c3c']
    bars = ax.bar(range(len(names)), counts, color=colors,
                  edgecolor='gray', linewidth=0.5, width=0.6)

    total_errors = analysis['summary']['incorrect']
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / max(total_errors, 1)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{count}\n({pct:.0%})', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha='right')
    ax.set_ylabel('Number of Errors', fontsize=12)
    ax.set_title(f'Error Pattern Analysis ({total_errors} total errors)',
                 fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("SYSTEMATIC ERROR ANALYSIS")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading data...")
    df = load_data()
    print(f"   Samples: {len(df)}, Companies: {df['ticker'].nunique()}")

    print("\n2. Running Gradient Boosting with GroupKFold CV...")
    y_pred, y_proba = run_predictions(df)
    acc = accuracy_score(df['target'].values, y_pred)
    print(f"   Accuracy: {acc:.1%}")

    print("\n3. Analyzing misclassifications...")
    analysis, analysis_df = analyze_misclassifications(df, y_pred, y_proba)

    print(f"\n   Correct: {analysis['summary']['correct']}")
    print(f"   Incorrect: {analysis['summary']['incorrect']}")
    print(f"   False Positives (IG predicted as SG): {analysis['summary']['false_positives']}")
    print(f"   False Negatives (SG predicted as IG): {analysis['summary']['false_negatives']}")

    print("\n   Error Patterns:")
    for pattern, info in analysis['error_patterns'].items():
        print(f"     {pattern}: {info['count']} ({info['pct']:.0%} of errors)")

    print("\n   Misclassified companies:")
    for m in analysis['misclassified_companies']:
        print(f"     {m['ticker']:<12} {m['company'][:30]:<30} "
              f"actual={m['actual_rating']:<5} pred_class={m['predicted_class']:<20} "
              f"conf={m['confidence']:.1%}")
        for reason in m['likely_reasons']:
            print(f"       -> {reason}")

    print("\n4. Generating error analysis plots...")
    plot_error_analysis(analysis_df, save_path=FIGURES_DIR / "error_scatter.png")
    plot_confidence_distribution(analysis_df, save_path=FIGURES_DIR / "confidence_dist.png")
    plot_error_pattern_summary(analysis, save_path=FIGURES_DIR / "error_patterns.png")

    report_file = RESULTS_DIR / "error_analysis.json"
    with open(report_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n   Error analysis report saved to: {report_file}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)

    if analysis['error_patterns']['boundary_ratings']['count'] > 0:
        pct = analysis['error_patterns']['boundary_ratings']['pct']
        print(f"\n  1. BOUNDARY EFFECT: {pct:.0%} of errors are on A-/BBB+ rated companies.")
        print("     These ratings sit at the exact boundary between Investment and Speculative")
        print("     grade, making them inherently difficult to classify.")

    if analysis['error_patterns']['agency_disagreement']['count'] > 0:
        pct = analysis['error_patterns']['agency_disagreement']['pct']
        print(f"\n  2. AGENCY DISAGREEMENT: {pct:.0%} of errors involve companies where")
        print("     different agencies assign conflicting binary classifications.")
        print("     This reflects genuine uncertainty about these companies' creditworthiness.")

    if analysis['error_patterns']['outlier_ratios']['count'] > 0:
        pct = analysis['error_patterns']['outlier_ratios']['pct']
        print(f"\n  3. OUTLIER RATIOS: {pct:.0%} of errors involve companies with")
        print("     extreme financial ratios (>2 std from mean), suggesting the model")
        print("     struggles with unusual financial profiles.")

    print(f"\n  Overall: Many errors are explainable by rating boundary effects and")
    print(f"  inter-agency disagreement -- not random model failures.")


if __name__ == "__main__":
    main()
