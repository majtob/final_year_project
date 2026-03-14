"""
Publication-quality visualizations for the credit rating project.

Generates confusion matrix, feature importance, rating distribution,
model comparison, decision tree, and dimensionality reduction plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
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


def get_models():
    return {
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  random_state=42, eval_metric='mlogloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                         learning_rate=0.1, random_state=42),
        'SVM (RBF)': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', random_state=42))]),
        'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=3))]),
        'Naive Bayes': GaussianNB(),
    }


def load_data():
    df = pd.read_csv(DATA_FILE)
    df['target'] = df['rating'].apply(to_binary)
    return df


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred)

    labels = ['Investment Grade\n(A- and above)', 'Speculative Grade\n(BBB+ and below)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, annot_kws={'size': 18},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix: {model_name}',
                 fontsize=14, fontweight='bold', pad=15)

    total = cm.sum()
    correct = np.trace(cm)
    ax.text(0.5, -0.15, f'Accuracy: {correct/total:.1%} ({correct}/{total})',
            transform=ax.transAxes, ha='center', fontsize=12,
            style='italic')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_model_comparison(save_path=None):
    """Bar chart comparing all models."""
    df = load_data()
    X = df[FEATURES].values
    y = df['target'].values
    groups = df['ticker'].values

    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)

    results = {}
    models = get_models()

    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
            acc = np.mean(y_pred == y)
            results[name] = acc
        except Exception:
            results[name] = 0.0

    sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names = [m[0] for m in sorted_models]
    accs = [m[1] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), accs, color=colors,
                   edgecolor='gray', linewidth=0.5, height=0.7)

    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline (50%)')
    ax.axvline(x=0.7155, color='blue', linestyle='--', alpha=0.5, label='Paper baseline (71.55%)')

    for i, (name, acc) in enumerate(zip(names, accs)):
        ax.text(acc + 0.005, i, f'{acc:.1%}', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Cross-Validated Accuracy (GroupKFold)', fontsize=12)
    ax.set_title('Model Comparison: Binary Credit Rating Classification\n'
                 '(75 samples, 23 companies, 4 Altman Z\'\' ratios)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, max(accs) + 0.08)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    return results


def plot_rating_distribution(save_path=None):
    """Rating distribution bar plot."""
    df = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
                    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-']
    present_ratings = [r for r in rating_order if r in df['rating'].values]
    counts = df['rating'].value_counts().reindex(present_ratings).fillna(0)

    ig_mask = [RATING_TO_NUMERIC.get(r, 0) >= 15 for r in present_ratings]
    colors = ['#2ecc71' if ig else '#e74c3c' for ig in ig_mask]

    axes[0].bar(range(len(present_ratings)), counts.values, color=colors,
                edgecolor='gray', linewidth=0.5)
    axes[0].set_xticks(range(len(present_ratings)))
    axes[0].set_xticklabels(present_ratings, rotation=45, ha='right', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Rating Distribution (Multi-class)', fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Investment Grade'),
                       Patch(facecolor='#e74c3c', label='Speculative Grade')]
    axes[0].legend(handles=legend_elements, fontsize=10)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    binary_counts = df['target'].value_counts().sort_index()
    labels_bin = ['Investment Grade\n(A- and above)', 'Speculative Grade\n(BBB+ and below)']
    colors_bin = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = axes[1].pie(
        binary_counts.values, labels=labels_bin, colors=colors_bin,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontweight('bold')
        t.set_fontsize(13)
    axes[1].set_title('Binary Classification Split', fontsize=13, fontweight='bold')

    plt.suptitle(f'Dataset: {len(df)} records, {df["ticker"].nunique()} companies, '
                 f'{df["rating_agency"].nunique()} agencies',
                 fontsize=12, y=0.02, style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_distributions(save_path=None):
    """Box plots of financial ratios by rating class."""
    df = load_data()
    df['Rating Class'] = df['target'].map({0: 'Investment Grade', 1: 'Speculative Grade'})

    feature_labels = {
        'liquid': 'Liquidity (WC/TA)',
        'cumprof': 'Cumulative Profit (RE/TA)',
        'profitab': 'Profitability (EBIT/TA)',
        'leverage': 'Leverage (Equity/Liab.)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'Investment Grade': '#2ecc71', 'Speculative Grade': '#e74c3c'}

    for ax, feat in zip(axes.flatten(), FEATURES):
        for cls, color in colors.items():
            data = df[df['Rating Class'] == cls][feat]
            bp = ax.boxplot([data], positions=[list(colors.keys()).index(cls)],
                           widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.7),
                           medianprops=dict(color='black', linewidth=2))

        ax.set_xticklabels(list(colors.keys()), fontsize=10)
        ax.set_title(feature_labels[feat], fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Financial Ratios by Rating Class',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_decision_tree_viz(save_path=None):
    """Decision tree visualization."""
    df = load_data()
    X = df[FEATURES]
    y = df['target'].values

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    tree.fit(X, y)

    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, feature_names=FEATURES,
              class_names=['Investment Grade', 'Speculative Grade'],
              filled=True, rounded=True, fontsize=10,
              proportion=True, ax=ax,
              impurity=True)
    ax.set_title('Decision Tree for Credit Rating Classification\n'
                 '(max_depth=3, trained on full dataset)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    rules = export_text(tree, feature_names=FEATURES,
                        class_names=['Investment Grade', 'Speculative Grade'])
    return rules


def plot_pca_scatter(save_path=None):
    """PCA scatter plot of companies colored by rating."""
    df = load_data()
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    ig_mask = df['target'] == 0
    sg_mask = df['target'] == 1

    ax.scatter(X_pca[ig_mask, 0], X_pca[ig_mask, 1],
               c='#2ecc71', s=80, alpha=0.7, edgecolors='white',
               linewidth=0.5, label='Investment Grade', zorder=3)
    ax.scatter(X_pca[sg_mask, 0], X_pca[sg_mask, 1],
               c='#e74c3c', s=80, alpha=0.7, edgecolors='white',
               linewidth=0.5, label='Speculative Grade', zorder=3)

    seen = set()
    for i in range(len(df)):
        ticker = df.iloc[i]['ticker']
        if ticker not in seen:
            seen.add(ticker)
            ax.annotate(ticker.replace('.SR', ''),
                       (X_pca[i, 0], X_pca[i, 1]),
                       fontsize=7, alpha=0.6,
                       xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: Companies in Financial Ratio Space',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    total_var = sum(pca.explained_variance_ratio_[:2])
    ax.text(0.02, 0.02, f'Total variance explained: {total_var:.1%}',
            transform=ax.transAxes, fontsize=10, style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_agency_disagreement(save_path=None):
    """Show where agencies disagree on the same company."""
    df = load_data()

    multi = df.groupby(['ticker', 'fiscal_year']).filter(lambda x: len(x) > 1)
    if len(multi) == 0:
        print("  No multi-agency records found, skipping.")
        return

    ticker_groups = multi.groupby(['ticker', 'fiscal_year'])

    companies = []
    for (ticker, year), group in ticker_groups:
        ratings = group['rating'].unique()
        binaries = group['target'].unique()
        companies.append({
            'ticker': ticker.replace('.SR', ''),
            'year': int(year),
            'n_agencies': len(group),
            'ratings': ', '.join(sorted(ratings)),
            'agree_on_binary': len(binaries) == 1,
        })

    comp_df = pd.DataFrame(companies)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, row in comp_df.iterrows():
        color = '#2ecc71' if row['agree_on_binary'] else '#e74c3c'
        label = row['ticker'] + f"\n({row['year']})"
        ax.barh(i, row['n_agencies'], color=color, edgecolor='gray',
                linewidth=0.5, height=0.7)
        ax.text(row['n_agencies'] + 0.1, i, row['ratings'],
                va='center', fontsize=9, style='italic')

    ax.set_yticks(range(len(comp_df)))
    ax.set_yticklabels([f"{r['ticker']} ({r['year']})" for _, r in comp_df.iterrows()],
                       fontsize=9)
    ax.set_xlabel('Number of Agency Ratings', fontsize=11)
    ax.set_title('Inter-Agency Rating Disagreement\n'
                 'Green = Agencies agree on Investment/Speculative; '
                 'Red = Agencies disagree',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[FEATURES].values
    y = df['target'].values
    groups = df['ticker'].values

    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)

    print("\n1. Confusion matrix (best model: Gradient Boosting)...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, random_state=42)
    y_pred_gb = cross_val_predict(gb, X, y, cv=cv, groups=groups)
    plot_confusion_matrix(y, y_pred_gb, 'Gradient Boosting',
                         save_path=FIGURES_DIR / "confusion_matrix_gb.png")

    print("\n2. Model comparison bar chart...")
    results = plot_model_comparison(save_path=FIGURES_DIR / "model_comparison.png")

    print("\n3. Rating distribution...")
    plot_rating_distribution(save_path=FIGURES_DIR / "rating_distribution.png")

    print("\n4. Feature distributions by class...")
    plot_feature_distributions(save_path=FIGURES_DIR / "feature_distributions.png")

    print("\n5. Decision tree visualization...")
    rules = plot_decision_tree_viz(save_path=FIGURES_DIR / "decision_tree.png")
    rules_file = RESULTS_DIR / "decision_tree_rules.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(rules_file, 'w') as f:
        f.write(rules)
    print(f"  Decision tree rules saved to: {rules_file}")

    print("\n6. PCA scatter plot...")
    plot_pca_scatter(save_path=FIGURES_DIR / "pca_scatter.png")

    print("\n7. Agency disagreement chart...")
    plot_agency_disagreement(save_path=FIGURES_DIR / "agency_disagreement.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")

    results_file = RESULTS_DIR / "model_comparison_final.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Model comparison results saved to {results_file}")


if __name__ == "__main__":
    main()
