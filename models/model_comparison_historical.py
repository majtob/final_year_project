"""
Multi-model comparison on the historical Fitch dataset (2019-2024).

Dataset: 72 Fitch ratings with financials (21 companies, 4 years of data).
Uses GroupKFold (groups=ticker) to prevent same-company data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "historical_fitch_ratings.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "historical_model_comparison.json"

FINANCIAL_COLS = ['liquid', 'cumprof', 'profitab', 'leverage']
BANK_SAFE_COLS = ['cumprof', 'leverage']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_category(rating):
    num = RATING_TO_NUMERIC.get(rating, 0)
    if num >= 18:
        return 'AA'
    elif num >= 15:
        return 'A'
    elif num >= 12:
        return 'BBB'
    else:
        return 'BB'


def get_models():
    return {
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
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42)),
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5)),
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, random_state=42,
        ),
        'Naive Bayes': GaussianNB(),
    }


def run_comparison(X, y, groups, feature_names, dataset_name, n_splits=5):
    print(f"\n{'#'*70}")
    print(f"DATASET: {dataset_name}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Groups: {len(set(groups))}")
    print(f"{'#'*70}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Classes: {list(le.classes_)}")
    for cls in le.classes_:
        print(f"  {cls}: {(y == cls).sum()}")

    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    print(f"CV: GroupKFold (k={n_splits}, groups=ticker)")

    models = get_models()
    results = {}

    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y_encoded, cv=cv, groups=groups)
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
                elif hasattr(model, 'named_steps'):
                    clf = model.named_steps.get('clf')
                    if hasattr(clf, 'coef_'):
                        avg_coef = np.abs(clf.coef_).mean(axis=0)
                        fi = dict(zip(feature_names, [float(x) for x in avg_coef]))
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

    return results, le.classes_


def main():
    df = pd.read_csv(DATA_FILE)
    df = df[df['has_financials'] == True].copy()
    df['target'] = df['rating'].apply(to_category)

    print(f"Historical Fitch Dataset: {len(df)} records, {df['ticker'].nunique()} companies")
    print(f"Years: {sorted(df['fiscal_year'].unique())}")

    all_results = {}

    # === A: All companies, 2 ratios (largest dataset) ===
    all_2r = df.dropna(subset=BANK_SAFE_COLS).copy()
    X = all_2r[BANK_SAFE_COLS].values
    y = all_2r['target'].values
    groups = all_2r['ticker'].values
    res, classes = run_comparison(X, y, groups, BANK_SAFE_COLS,
                                  "A: All Companies (2 ratios, cumprof+leverage)")
    all_results['all_2ratios'] = {
        'models': res, 'classes': list(classes), 'n': len(X),
        'n_groups': int(all_2r['ticker'].nunique()),
    }

    # === B: Non-financial, 4 ratios ===
    non_fin = df[~df['sector'].isin(['Banks', 'Insurance'])].copy()
    non_fin = non_fin.dropna(subset=FINANCIAL_COLS)
    if len(non_fin) >= 10:
        X = non_fin[FINANCIAL_COLS].values
        y = non_fin['target'].values
        groups = non_fin['ticker'].values
        res, classes = run_comparison(X, y, groups, FINANCIAL_COLS,
                                      "B: Non-Financial (4 ratios)")
        all_results['nonfin_4ratios'] = {
            'models': res, 'classes': list(classes), 'n': len(X),
            'n_groups': int(non_fin['ticker'].nunique()),
        }

    # === C: All companies, 2 ratios + year feature ===
    all_yr = df.dropna(subset=BANK_SAFE_COLS).copy()
    feat_cols = BANK_SAFE_COLS + ['fiscal_year']
    X = all_yr[feat_cols].values
    y = all_yr['target'].values
    groups = all_yr['ticker'].values
    res, classes = run_comparison(X, y, groups, feat_cols,
                                  "C: All Companies (2 ratios + year)")
    all_results['all_2ratios_year'] = {
        'models': res, 'classes': list(classes), 'n': len(X),
        'n_groups': int(all_yr['ticker'].nunique()),
    }

    # === LEADERBOARD ===
    print(f"\n{'='*80}")
    print("LEADERBOARD - HISTORICAL FITCH DATASET")
    print(f"{'='*80}")

    all_entries = []
    for ds_name, ds_data in all_results.items():
        for model_name, model_data in ds_data['models'].items():
            if 'accuracy' in model_data and model_data['accuracy'] > 0:
                all_entries.append({
                    'model': model_name,
                    'dataset': ds_name,
                    'accuracy': model_data['accuracy'],
                    'n_samples': ds_data['n'],
                    'n_groups': ds_data['n_groups'],
                })

    all_entries.sort(key=lambda x: -x['accuracy'])

    print(f"{'Rank':>4} {'Model':<25} {'Dataset':<30} {'N':>5} {'Groups':>6} {'Acc':>7}")
    print("-" * 80)
    for i, e in enumerate(all_entries[:15], 1):
        print(f"{i:>4} {e['model']:<25} {e['dataset']:<30} {e['n_samples']:>5} {e['n_groups']:>6} {e['accuracy']:>6.1%}")

    print(f"\n--- Previous Best ---")
    print(f"Decision Tree on Fitch 2024-only (23 samples): 78.3%")
    print(f"XGBoost on Fitch 2024-only (23 samples):       69.6%")

    best = all_entries[0]
    print(f"\nNew Best: {best['model']} on {best['dataset']} = {best['accuracy']:.1%} "
          f"({best['n_samples']} samples, {best['n_groups']} companies)")

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
