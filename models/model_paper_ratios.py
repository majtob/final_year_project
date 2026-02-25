"""
Model comparison using the EXACT methodology from the reference paper:

Muñoz-Izquierdo et al. (2022) - "Machine learning in corporate credit rating
assessment using the expanded audit report"

Paper methodology:
- 4 Altman Z''-Score ratios: LIQUID, CUMPROF, PROFITAB, LEVERAGE
- Banks and financial institutions EXCLUDED
- Binary classification: A-ratings (0) vs B/C-ratings (1)
- ML techniques: C4.5 Decision Tree, PART, Rough Set, Logistic Regression

Our adaptation:
- Same 4 ratios
- Exclude banks, insurance, financial services
- Historical Fitch dataset (2021-2024) + original Tassnief + expanded data
- 8 ML models including paper's Decision Tree
- Both binary and multi-class classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
HISTORICAL_FILE = PROJECT_ROOT / "data" / "processed" / "historical_fitch_ratings.csv"
EXPANDED_FILE = PROJECT_ROOT / "data" / "processed" / "expanded_ratings_financials.csv"
ORIGINAL_FILE = PROJECT_ROOT / "data" / "processed" / "ratings_with_financials.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "paper_methodology_results.json"

# Paper's 4 Altman Z''-Score ratios
PAPER_RATIOS = ['liquid', 'cumprof', 'profitab', 'leverage']

# Sectors to exclude (following the paper)
EXCLUDED_SECTORS = ['Banks', 'Insurance', 'Financial Services']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_binary(rating):
    """Paper's binary: 0 = A-ratings (low risk), 1 = B-ratings (high risk)."""
    num = RATING_TO_NUMERIC.get(rating, 0)
    return 0 if num >= 15 else 1  # A- and above = 0, BBB+ and below = 1


def to_multiclass(rating):
    num = RATING_TO_NUMERIC.get(rating, 0)
    if num >= 18: return 'AA'
    elif num >= 15: return 'A'
    elif num >= 12: return 'BBB'
    else: return 'BB'


def get_models():
    return {
        'Decision Tree (C4.5)': DecisionTreeClassifier(
            criterion='entropy', max_depth=5, random_state=42,
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss',
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
        ),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42)),
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=3)),
        ]),
        'Naive Bayes': GaussianNB(),
    }


def run_experiment(X, y, groups, feature_names, exp_name, use_group_cv=True):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]} ({', '.join(feature_names)})")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    print(f"Classes: {list(le.classes_)}")
    for cls in le.classes_:
        print(f"  {cls}: {(y == cls).sum()}")

    if use_group_cv and groups is not None:
        n_groups = len(set(groups))
        n_splits = min(5, n_groups)
        cv = GroupKFold(n_splits=n_splits)
        cv_name = f"GroupKFold(k={n_splits})"
        cv_groups = groups
    else:
        min_class = pd.Series(y).value_counts().min()
        if min_class >= 2 and len(X) <= 30:
            cv = LeaveOneOut()
            cv_name = "LOO"
        else:
            n_splits = min(5, min_class)
            n_splits = max(2, n_splits)
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_name = f"StratifiedKFold(k={n_splits})"
        cv_groups = None

    print(f"CV: {cv_name}")

    models = get_models()
    results = {}

    for name, model in models.items():
        try:
            if cv_groups is not None:
                y_pred = cross_val_predict(model, X, y_encoded, cv=cv, groups=cv_groups)
            else:
                y_pred = cross_val_predict(model, X, y_encoded, cv=cv)

            acc = accuracy_score(y_encoded, y_pred)
            report = classification_report(y_encoded, y_pred,
                                           target_names=le.classes_,
                                           output_dict=True, zero_division=0)
            cm = confusion_matrix(y_encoded, y_pred)

            fi = {}
            try:
                model.fit(X, y_encoded)
                if hasattr(model, 'feature_importances_'):
                    fi = dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
            except Exception:
                pass

            results[name] = {
                'accuracy': float(acc),
                'report': report,
                'confusion_matrix': cm.tolist(),
                'feature_importance': fi,
            }
            print(f"  {name:25s} {acc:>7.1%}")

        except Exception as e:
            print(f"  {name:25s} FAILED: {e}")
            results[name] = {'accuracy': 0, 'error': str(e)}

    return results, list(le.classes_)


def load_and_combine():
    """Load all datasets and combine non-financial companies with 4 ratios."""
    all_dfs = []

    # 1. Historical Fitch
    try:
        hist = pd.read_csv(HISTORICAL_FILE)
        hist = hist[hist['has_financials'] == True].copy()
        hist['dataset'] = 'historical_fitch'
        all_dfs.append(hist)
        print(f"Historical Fitch: {len(hist)} records")
    except FileNotFoundError:
        print("Historical Fitch not found")

    # 2. Expanded multi-agency (consensus per company)
    try:
        exp = pd.read_csv(EXPANDED_FILE)
        exp = exp[exp['has_financials'] == True].copy()
        exp['dataset'] = 'expanded_multiagency'
        all_dfs.append(exp)
        print(f"Expanded multi-agency: {len(exp)} records")
    except FileNotFoundError:
        print("Expanded not found")

    # 3. Original Tassnief
    try:
        orig = pd.read_csv(ORIGINAL_FILE)
        orig['rating_agency'] = 'Tassnief'
        orig['dataset'] = 'original_tassnief'
        if 'fiscal_year' not in orig.columns:
            orig['fiscal_year'] = 2024
        all_dfs.append(orig)
        print(f"Original Tassnief: {len(orig)} records")
    except FileNotFoundError:
        print("Original Tassnief not found")

    if not all_dfs:
        raise ValueError("No data found")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Filter: non-financial companies only (following the paper)
    if 'sector' in combined.columns:
        combined = combined[~combined['sector'].isin(EXCLUDED_SECTORS)].copy()

    # Require all 4 ratios
    combined = combined.dropna(subset=PAPER_RATIOS)

    # De-duplicate: keep one record per ticker-year-agency
    if 'rating_agency' in combined.columns:
        combined = combined.drop_duplicates(subset=['ticker', 'fiscal_year', 'rating_agency'])

    print(f"\nCombined non-financial with 4 ratios: {len(combined)} records")
    print(f"Unique companies: {combined['ticker'].nunique()}")

    return combined


def main():
    print("#" * 70)
    print("PAPER METHODOLOGY: 4 Altman Z'' Ratios, Non-Financial Companies")
    print("#" * 70)

    df = load_and_combine()

    all_results = {}

    # === EXPERIMENT 1: Binary classification (paper's approach) ===
    df_bin = df.copy()
    df_bin['target'] = df_bin['rating'].apply(to_binary).astype(str)

    # Check class balance
    class_counts = df_bin['target'].value_counts()
    print(f"\nBinary class distribution:")
    print(f"  0 (A-ratings, low risk): {class_counts.get('0', 0)}")
    print(f"  1 (B-ratings, high risk): {class_counts.get('1', 0)}")

    if class_counts.get('0', 0) >= 2 and class_counts.get('1', 0) >= 2:
        X = df_bin[PAPER_RATIOS].values
        y = df_bin['target'].values
        groups = df_bin['ticker'].values
        res, classes = run_experiment(X, y, groups, PAPER_RATIOS,
                                      "1: Binary (A vs B+C) - Paper's approach")
        all_results['binary'] = {
            'models': res, 'classes': classes,
            'n': len(X), 'n_groups': int(df_bin['ticker'].nunique()),
        }
    else:
        print("SKIPPED: Insufficient class balance for binary")

    # === EXPERIMENT 2: Multi-class (AA/A/BBB/BB) ===
    df_mc = df.copy()
    df_mc['target'] = df_mc['rating'].apply(to_multiclass)

    # Merge rare classes
    counts = df_mc['target'].value_counts()
    merge_map = {}
    for cls in counts.index:
        if counts[cls] < 2:
            merge_map[cls] = 'A' if cls == 'AA' else ('BBB' if cls == 'BB' else cls)
        else:
            merge_map[cls] = cls
    df_mc['target'] = df_mc['target'].map(merge_map)

    X = df_mc[PAPER_RATIOS].values
    y = df_mc['target'].values
    groups = df_mc['ticker'].values
    res, classes = run_experiment(X, y, groups, PAPER_RATIOS,
                                  "2: Multi-class (AA/A/BBB/BB)")
    all_results['multiclass'] = {
        'models': res, 'classes': classes,
        'n': len(X), 'n_groups': int(df_mc['ticker'].nunique()),
    }

    # === EXPERIMENT 3: Fitch-only (our best agency) ===
    fitch_df = df[df.get('rating_agency', '') == 'Fitch'].copy()
    if len(fitch_df) >= 10:
        fitch_df['target'] = fitch_df['rating'].apply(to_multiclass)
        counts = fitch_df['target'].value_counts()
        merge_map = {cls: ('A' if cls == 'AA' else ('BBB' if cls == 'BB' else cls))
                     if counts.get(cls, 0) < 2 else cls for cls in counts.index}
        fitch_df['target'] = fitch_df['target'].map(merge_map)

        X = fitch_df[PAPER_RATIOS].values
        y = fitch_df['target'].values
        groups = fitch_df['ticker'].values
        res, classes = run_experiment(X, y, groups, PAPER_RATIOS,
                                      "3: Fitch-only, 4 Ratios, Multi-class")
        all_results['fitch_4ratios'] = {
            'models': res, 'classes': classes,
            'n': len(X), 'n_groups': int(fitch_df['ticker'].nunique()),
        }

    # === EXPERIMENT 4: Tassnief-only ===
    tassnief_df = df[df.get('dataset', '') == 'original_tassnief'].copy()
    if len(tassnief_df) >= 10:
        tassnief_df['target'] = tassnief_df['rating'].apply(to_multiclass)
        counts = tassnief_df['target'].value_counts()
        merge_map = {cls: ('A' if cls == 'AA' else ('BBB' if cls == 'BB' else cls))
                     if counts.get(cls, 0) < 2 else cls for cls in counts.index}
        tassnief_df['target'] = tassnief_df['target'].map(merge_map)

        X = tassnief_df[PAPER_RATIOS].values
        y = tassnief_df['target'].values
        groups = tassnief_df['ticker'].values if 'ticker' in tassnief_df.columns else None
        res, classes = run_experiment(X, y, groups, PAPER_RATIOS,
                                      "4: Tassnief-only, 4 Ratios")
        all_results['tassnief_4ratios'] = {
            'models': res, 'classes': classes, 'n': len(X),
        }

    # === FINAL LEADERBOARD ===
    print(f"\n{'#'*70}")
    print("FINAL LEADERBOARD - PAPER METHODOLOGY (4 Ratios, Non-Financial)")
    print(f"{'#'*70}")

    all_entries = []
    for ds_name, ds_data in all_results.items():
        for model_name, model_data in ds_data['models'].items():
            if 'accuracy' in model_data and model_data['accuracy'] > 0:
                all_entries.append({
                    'model': model_name,
                    'dataset': ds_name,
                    'accuracy': model_data['accuracy'],
                    'n': ds_data['n'],
                })

    all_entries.sort(key=lambda x: -x['accuracy'])

    print(f"\n{'Rank':>4} {'Model':<25} {'Dataset':<25} {'N':>5} {'Accuracy':>8}")
    print("-" * 70)
    for i, e in enumerate(all_entries[:20], 1):
        print(f"{i:>4} {e['model']:<25} {e['dataset']:<25} {e['n']:>5} {e['accuracy']:>7.1%}")

    print(f"\n--- Reference Paper Results ---")
    print(f"Paper: C4.5 Decision Tree, 4 ratios, 116 companies: 71.55%")
    print(f"Paper: PART algorithm, KAMs only, 116 companies:    74.14%")
    print(f"Paper: PART algorithm, Combined, 116 companies:     84.04%")

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
