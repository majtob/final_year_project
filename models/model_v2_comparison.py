"""
Model comparison on the expanded v2 dataset (75 records, 23 companies).
Binary classification: A-ratings (0) vs B/C-ratings (1).
4 Altman Z'' ratios, non-financial companies only.
GroupKFold CV with groups = ticker.
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
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_V1 = PROJECT_ROOT / "data" / "processed" / "model_training_data.csv"
DATA_V2 = PROJECT_ROOT / "data" / "processed" / "model_training_data_v2.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "model_v2_comparison.json"

FEATURES = ['liquid', 'cumprof', 'profitab', 'leverage']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_binary(rating):
    num = RATING_TO_NUMERIC.get(rating, 0)
    return 0 if num >= 15 else 1


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


def run_experiment(data_path, exp_name):
    df = pd.read_csv(data_path)
    df['target'] = df['rating'].apply(to_binary)

    X = df[FEATURES].values
    y = df['target'].values
    groups = df['ticker'].values

    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    print(f"\n{'='*70}")
    print(f"{exp_name}")
    print(f"{'='*70}")
    print(f"Samples: {len(X)}, Companies: {n_groups}, CV: GroupKFold(k={n_splits})")
    print(f"Class 0 (A-ratings): {(y==0).sum()}, Class 1 (B/C): {(y==1).sum()}")

    results = {}
    models = get_models()

    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
            acc = accuracy_score(y, y_pred)
            results[name] = float(acc)
            print(f"  {name:<25s} {acc:>7.1%}")
        except Exception as e:
            print(f"  {name:<25s} FAILED: {e}")
            results[name] = 0.0

    return results


def main():
    print("#" * 70)
    print("V1 vs V2 DATASET COMPARISON")
    print("#" * 70)

    v1_results = run_experiment(DATA_V1, "V1: Original (60 records, 22 companies)")
    v2_results = run_experiment(DATA_V2, "V2: Expanded (75 records, 23 companies)")

    print(f"\n{'#'*70}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{'#'*70}")
    print(f"\n{'Model':<25}  {'V1 (N=60)':>10}  {'V2 (N=75)':>10}  {'Change':>8}  {'Winner':>8}")
    print("-" * 70)

    v1_wins = v2_wins = 0
    all_results = {}

    for name in get_models().keys():
        a1 = v1_results.get(name, 0)
        a2 = v2_results.get(name, 0)
        diff = a2 - a1

        if a2 > a1:
            winner = "V2"
            v2_wins += 1
        elif a1 > a2:
            winner = "V1"
            v1_wins += 1
        else:
            winner = "Tie"

        print(f"  {name:<25} {a1:>9.1%}  {a2:>9.1%}  {diff:>+7.1%}  {winner:>8}")
        all_results[name] = {'v1': a1, 'v2': a2, 'diff': diff}

    print(f"\n  Score: V1 wins {v1_wins}, V2 wins {v2_wins}")

    best_v1 = max(v1_results.items(), key=lambda x: x[1])
    best_v2 = max(v2_results.items(), key=lambda x: x[1])
    print(f"\n  Best V1: {best_v1[0]} = {best_v1[1]:.1%}")
    print(f"  Best V2: {best_v2[0]} = {best_v2[1]:.1%}")
    print(f"  Paper:   C4.5 Decision Tree = 71.55%")

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'v1': {'n': 60, 'results': v1_results},
            'v2': {'n': 75, 'results': v2_results},
            'comparison': all_results,
        }, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
