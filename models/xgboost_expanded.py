"""
XGBoost model trained on expanded multi-agency rating dataset.

Key improvements over baseline:
- 80 ratings from 5 agencies (vs 23 from Tassnief only)
- GroupKFold cross-validation to prevent same-company data leakage
- Agency indicator feature to capture systematic agency differences
- Two experiments: (A) full 4-ratio model, (B) 2-ratio model including banks
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "expanded_ratings_financials.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "expanded_model_results.json"

FINANCIAL_COLS = ['liquid', 'cumprof', 'profitab', 'leverage']
BANK_SAFE_COLS = ['cumprof', 'leverage']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df[df['has_financials'] == True].copy()
    print(f"Loaded {len(df)} ratings with financials")
    print(f"Unique companies: {df['ticker'].nunique()}")
    print(f"Agencies: {df['rating_agency'].nunique()}")
    return df


def prepare_target(df):
    """Create multi-class target from rating categories."""
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
    
    df = df.copy()
    df['target'] = df['rating'].apply(to_category)
    return df


def run_experiment(df, feature_cols, experiment_name, use_agency_feature=True):
    """Train and evaluate XGBoost with GroupKFold."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    work_df = df.dropna(subset=feature_cols).copy()
    print(f"Samples: {len(work_df)}")
    print(f"Unique companies (groups): {work_df['ticker'].nunique()}")
    print(f"Category distribution:\n{work_df['target'].value_counts().to_string()}")
    
    # Minimum viability check
    n_groups = work_df['ticker'].nunique()
    n_classes = work_df['target'].nunique()
    if n_groups < 5 or len(work_df) < 10:
        print("SKIPPED: Not enough data")
        return None
    
    features = list(feature_cols)
    X = work_df[features].copy()
    
    if use_agency_feature:
        agency_le = LabelEncoder()
        X['agency_encoded'] = agency_le.fit_transform(work_df['rating_agency'])
        features.append('agency_encoded')
    
    le = LabelEncoder()
    y = le.fit_transform(work_df['target'])
    groups = work_df['ticker'].values
    
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    
    # GroupKFold cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=gkf, groups=groups)
    
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    print(f"Confusion Matrix:")
    print(f"Classes: {list(le.classes_)}")
    print(cm)
    
    # Train final model on all data for feature importance
    model.fit(X, y)
    importances = dict(zip(features, [float(x) for x in model.feature_importances_]))
    print(f"\nFeature Importance:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    return {
        'experiment': experiment_name,
        'n_samples': len(work_df),
        'n_companies': int(work_df['ticker'].nunique()),
        'n_agencies': int(work_df['rating_agency'].nunique()),
        'n_folds': n_splits,
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'classes': list(le.classes_),
        'feature_importance': importances,
        'category_distribution': work_df['target'].value_counts().to_dict(),
    }


def main():
    df = load_data()
    df = prepare_target(df)
    
    results = {}
    
    # Exp A: Non-financial companies with all 4 ratios (no agency feature)
    non_fin = df[~df['sector'].isin(['Banks', 'Insurance'])].copy()
    res_a = run_experiment(non_fin, FINANCIAL_COLS, 
                           "A: Non-Financial Companies, 4 Ratios (no agency)", 
                           use_agency_feature=False)
    if res_a:
        results['non_financial_4ratios_no_agency'] = res_a
    
    # Exp B: Non-financial companies with all 4 ratios + agency feature
    res_b = run_experiment(non_fin, FINANCIAL_COLS, 
                           "B: Non-Financial Companies, 4 Ratios + Agency", 
                           use_agency_feature=True)
    if res_b:
        results['non_financial_4ratios_with_agency'] = res_b
    
    # Exp C: All companies with cumprof + leverage only (no agency)
    res_c = run_experiment(df, BANK_SAFE_COLS, 
                           "C: All Companies, 2 Ratios (no agency)", 
                           use_agency_feature=False)
    if res_c:
        results['all_companies_2ratios_no_agency'] = res_c
    
    # Exp D: All companies with cumprof + leverage + agency
    res_d = run_experiment(df, BANK_SAFE_COLS, 
                           "D: All Companies, 2 Ratios + Agency", 
                           use_agency_feature=True)
    if res_d:
        results['all_companies_2ratios_with_agency'] = res_d
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Experiment':<50} {'Samples':>8} {'Accuracy':>10}")
    print("-" * 70)
    for key, res in results.items():
        print(f"{res['experiment']:<50} {res['n_samples']:>8} {res['accuracy']:>9.1%}")
    
    # Compare to old baseline
    print(f"\n--- vs Previous Baseline ---")
    print(f"Previous (Tassnief only, 23 samples): ~65.2% accuracy")
    best = max(results.values(), key=lambda x: x['accuracy'])
    print(f"Best expanded model: {best['accuracy']:.1%} accuracy ({best['experiment']})")
    
    # Save
    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
