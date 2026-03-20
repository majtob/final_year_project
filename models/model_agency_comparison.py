"""
Compare two approaches for handling multi-agency ratings:
  Option 1: Fitch-only (single agency, no conflicts)
  Option 2: All agencies + rating_agency as a feature

Both use the paper's 4 Altman Z'' ratios, non-financial companies only.
Binary classification: A-ratings (0) vs B/C-ratings (1).
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
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_training_data.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "agency_comparison_results.json"

PAPER_RATIOS = ['liquid', 'cumprof', 'profitab', 'leverage']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_binary(rating):
    """Paper's binary: 0 = A-ratings (low risk), 1 = B/C-ratings (high risk)."""
    num = RATING_TO_NUMERIC.get(rating, 0)
    return 0 if num >= 15 else 1


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


def run_experiment(X, y, groups, feature_names, exp_name):
    print(f"\n{'='*70}")
    print(f"{exp_name}")
    print(f"{'='*70}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]} ({', '.join(feature_names)})")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Classes: {list(le.classes_)}")
    for cls in le.classes_:
        count = (y == cls).sum()
        print(f"  {cls}: {count} ({count/len(y)*100:.0f}%)")

    n_groups = len(set(groups))

    if n_groups >= 5:
        cv = GroupKFold(n_splits=min(5, n_groups))
        cv_name = f"GroupKFold(k={min(5, n_groups)}, groups=ticker)"
        cv_groups = groups
    else:
        cv = LeaveOneOut()
        cv_name = "LeaveOneOut"
        cv_groups = None

    print(f"Groups (companies): {n_groups}")
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


def main():
    df = pd.read_csv(DATA_FILE)
    df['binary_target'] = df['rating'].apply(to_binary).astype(str)

    print(f"Full dataset: {len(df)} records, {df['ticker'].nunique()} companies")
    print(f"Agencies: {df['rating_agency'].value_counts().to_dict()}")

    all_results = {}

    # ================================================================
    # OPTION 1: FITCH ONLY
    # ================================================================
    fitch = df[df['rating_agency'] == 'Fitch'].copy()

    print(f"\n{'#'*70}")
    print(f"OPTION 1: FITCH ONLY")
    print(f"{'#'*70}")
    print(f"Records: {len(fitch)}, Companies: {fitch['ticker'].nunique()}")

    X = fitch[PAPER_RATIOS].values
    y = fitch['binary_target'].values
    groups = fitch['ticker'].values

    res, classes = run_experiment(X, y, groups, PAPER_RATIOS,
                                  "Option 1: Fitch Only, 4 Ratios, Binary")
    all_results['fitch_only'] = {
        'models': res, 'classes': classes,
        'n': len(X), 'n_groups': int(fitch['ticker'].nunique()),
    }

    # Also export the Fitch-only training data
    fitch_out = fitch[['ticker', 'company_name', 'rating_agency', 'rating',
                        'fiscal_year', 'binary_target'] + PAPER_RATIOS].copy()
    fitch_out = fitch_out.sort_values(['ticker', 'fiscal_year'])
    fitch_path = PROJECT_ROOT / "data" / "processed" / "model_fitch_only_data.csv"
    fitch_out.to_csv(fitch_path, index=False)
    print(f"\nFitch-only data saved to {fitch_path}")

    # ================================================================
    # OPTION 2: ALL AGENCIES + agency as feature
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"OPTION 2: ALL AGENCIES + rating_agency FEATURE")
    print(f"{'#'*70}")
    print(f"Records: {len(df)}, Companies: {df['ticker'].nunique()}")

    agency_le = LabelEncoder()
    df['agency_encoded'] = agency_le.fit_transform(df['rating_agency'])
    print(f"Agency encoding: {dict(zip(agency_le.classes_, agency_le.transform(agency_le.classes_)))}")

    features_with_agency = PAPER_RATIOS + ['agency_encoded']
    X = df[features_with_agency].values
    y = df['binary_target'].values
    groups = df['ticker'].values

    res, classes = run_experiment(X, y, groups, features_with_agency,
                                  "Option 2: All Agencies + Agency Feature, 4 Ratios, Binary")
    all_results['all_with_agency'] = {
        'models': res, 'classes': classes,
        'n': len(X), 'n_groups': int(df['ticker'].nunique()),
    }

    # ================================================================
    # SIDE-BY-SIDE COMPARISON
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{'#'*70}")
    print(f"\n{'Model':<25}  {'Fitch Only':>12} {'(N='+str(all_results['fitch_only']['n'])+')':>6}"
          f"  {'All+Agency':>12} {'(N='+str(all_results['all_with_agency']['n'])+')':>6}"
          f"  {'Winner':>12}")
    print("-" * 80)

    model_names = list(get_models().keys())
    fitch_wins = 0
    all_wins = 0

    for name in model_names:
        f_acc = all_results['fitch_only']['models'].get(name, {}).get('accuracy', 0)
        a_acc = all_results['all_with_agency']['models'].get(name, {}).get('accuracy', 0)

        if f_acc > a_acc:
            winner = "Fitch"
            fitch_wins += 1
        elif a_acc > f_acc:
            winner = "All+Agency"
            all_wins += 1
        else:
            winner = "Tie"

        print(f"  {name:<25} {f_acc:>11.1%}        {a_acc:>11.1%}        {winner:>10}")

    print(f"\n  Score: Fitch Only wins {fitch_wins}, All+Agency wins {all_wins}")

    best_fitch = max(all_results['fitch_only']['models'].items(), key=lambda x: x[1].get('accuracy', 0))
    best_all = max(all_results['all_with_agency']['models'].items(), key=lambda x: x[1].get('accuracy', 0))

    print(f"\n  Best Fitch-only:  {best_fitch[0]} = {best_fitch[1]['accuracy']:.1%}")
    print(f"  Best All+Agency: {best_all[0]} = {best_all[1]['accuracy']:.1%}")
    print(f"\n  Paper reference: C4.5 Decision Tree, 4 ratios = 71.55%")

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
