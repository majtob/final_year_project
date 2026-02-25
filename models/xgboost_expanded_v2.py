"""
Expanded XGBoost v2 - Consensus ratings and per-agency models.

Addresses issues from v1:
- Inter-agency disagreement was confusing the model
- Same financials mapped to different labels

Approaches:
1. Consensus rating (median across agencies) per company -> eliminates noise
2. Per-agency models (Moody's, Fitch) -> agency-specific criteria
3. Combined Tassnief + expanded data -> leveraging original data
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXPANDED_FILE = PROJECT_ROOT / "data" / "processed" / "expanded_ratings_financials.csv"
ORIGINAL_FILE = PROJECT_ROOT / "data" / "processed" / "ratings_with_financials.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "expanded_v2_results.json"

FINANCIAL_COLS = ['liquid', 'cumprof', 'profitab', 'leverage']

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


def run_model(X, y, labels, model_name, cv_strategy='stratified'):
    """Train XGBoost with specified CV strategy."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f"Classes: {list(le.classes_)}")
    for cls in le.classes_:
        print(f"  {cls}: {(y == cls).sum()}")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
    )
    
    min_class_count = pd.Series(y).value_counts().min()
    
    if cv_strategy == 'loo' and min_class_count >= 2:
        cv = LeaveOneOut()
    else:
        n_splits = min(5, min_class_count)
        n_splits = max(2, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(model, X, y_encoded, cv=cv)
    
    accuracy = accuracy_score(y_encoded, y_pred)
    report = classification_report(y_encoded, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_encoded, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_, zero_division=0))
    print(f"Confusion Matrix (rows=true, cols=pred):")
    print(f"  {list(le.classes_)}")
    print(cm)
    
    model.fit(X, y_encoded)
    importances = dict(zip(labels, [float(x) for x in model.feature_importances_]))
    print(f"\nFeature Importance:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    return {
        'model': model_name,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'cv_strategy': cv_strategy,
        'accuracy': float(accuracy),
        'report': report,
        'confusion_matrix': cm.tolist(),
        'classes': list(le.classes_),
        'feature_importance': importances,
    }


def main():
    # Load expanded data
    df = pd.read_csv(EXPANDED_FILE)
    df = df[df['has_financials'] == True].copy()
    df['target'] = df['rating'].apply(to_category)
    
    results = {}
    
    # ===== EXPERIMENT 1: Consensus ratings (one per company) =====
    print("\n" + "#"*60)
    print("EXPERIMENT 1: CONSENSUS RATINGS")
    print("#"*60)
    
    # For each company, take the median numeric rating, map back to category
    consensus = df.groupby('ticker').agg({
        'company_name': 'first',
        'sector': 'first',
        'rating_numeric': 'median',
        'liquid': 'first',
        'cumprof': 'first',
        'profitab': 'first',
        'leverage': 'first',
    }).reset_index()
    
    def numeric_to_category(num):
        if num >= 18:
            return 'AA'
        elif num >= 15:
            return 'A'
        elif num >= 12:
            return 'BBB'
        else:
            return 'BB'
    
    consensus['target'] = consensus['rating_numeric'].apply(numeric_to_category)
    
    def merge_rare_classes(target_series, min_count=2):
        """Merge classes with fewer than min_count into their neighbor."""
        counts = target_series.value_counts()
        mapping = {}
        for cls in counts.index:
            if counts[cls] < min_count:
                if cls == 'AA':
                    mapping[cls] = 'A'
                elif cls == 'BB':
                    mapping[cls] = 'BBB'
                else:
                    mapping[cls] = cls
            else:
                mapping[cls] = cls
        return target_series.map(mapping)
    
    # 1a: Non-financial with 4 ratios
    non_fin = consensus[~consensus['sector'].isin(['Banks', 'Insurance'])].copy()
    non_fin = non_fin.dropna(subset=FINANCIAL_COLS)
    non_fin['target'] = merge_rare_classes(non_fin['target'])
    
    if len(non_fin) >= 10:
        X = non_fin[FINANCIAL_COLS].values
        y = non_fin['target'].values
        res = run_model(X, y, FINANCIAL_COLS, 
                       "1a: Consensus, Non-Financial, 4 Ratios", 'loo')
        results['consensus_nonfin_4ratios'] = res
    
    # 1b: All companies with cumprof + leverage
    all_cos = consensus.dropna(subset=['cumprof', 'leverage'])
    all_cos = all_cos.copy()
    all_cos['target'] = merge_rare_classes(all_cos['target'])
    if len(all_cos) >= 10:
        X = all_cos[['cumprof', 'leverage']].values
        y = all_cos['target'].values
        res = run_model(X, y, ['cumprof', 'leverage'],
                       "1b: Consensus, All Companies, 2 Ratios", 'loo')
        results['consensus_all_2ratios'] = res
    
    # ===== EXPERIMENT 2: Per-Agency Models =====
    print("\n" + "#"*60)
    print("EXPERIMENT 2: PER-AGENCY MODELS")
    print("#"*60)
    
    for agency in ['Moodys', 'Fitch', 'S&P', 'Financial Analytics']:
        agency_df = df[df['rating_agency'] == agency].copy()
        agency_df = agency_df.dropna(subset=['cumprof', 'leverage'])
        agency_df['target'] = merge_rare_classes(agency_df['target'])
        
        if len(agency_df) < 8:
            print(f"\n{agency}: Only {len(agency_df)} samples, skipping")
            continue
        
        X = agency_df[['cumprof', 'leverage']].values
        y = agency_df['target'].values
        
        if len(set(y)) < 2:
            print(f"\n{agency}: Only 1 class, skipping")
            continue
        
        res = run_model(X, y, ['cumprof', 'leverage'],
                       f"2: {agency} Only, 2 Ratios", 'loo')
        results[f'agency_{agency.lower().replace(" ", "_")}'] = res
    
    # ===== EXPERIMENT 3: Combined Original Tassnief + Expanded =====
    print("\n" + "#"*60)
    print("EXPERIMENT 3: ORIGINAL TASSNIEF + EXPANDED DATA")
    print("#"*60)
    
    # Load original Tassnief data
    try:
        orig_df = pd.read_csv(ORIGINAL_FILE)
        orig_df = orig_df.dropna(subset=FINANCIAL_COLS)
        
        orig_mapping = {
            'AA': 'AA', 'A': 'A', 'BBB': 'BBB', 'BB': 'BB',
            'AA+': 'AA', 'AA-': 'AA',
            'A+': 'A', 'A-': 'A',
            'BBB+': 'BBB', 'BBB-': 'BBB',
            'BB+': 'BB', 'BB-': 'BB',
        }
        orig_df['target'] = orig_df['rating'].map(orig_mapping)
        orig_df = orig_df.dropna(subset=['target'])
        
        orig_rows = orig_df[FINANCIAL_COLS + ['target', 'ticker']].copy()
        orig_rows['source'] = 'tassnief_original'
        
        expanded_non_fin = non_fin[FINANCIAL_COLS + ['target', 'ticker']].copy()
        expanded_non_fin['source'] = 'multi_agency_consensus'
        
        combined = pd.concat([orig_rows, expanded_non_fin], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ticker'], keep='first')
        combined['target'] = merge_rare_classes(combined['target'])
        
        print(f"\nCombined dataset: {len(combined)} unique companies")
        print(f"  From Tassnief original: {(combined['source'] == 'tassnief_original').sum()}")
        print(f"  From multi-agency: {(combined['source'] == 'multi_agency_consensus').sum()}")
        
        if len(combined) >= 10:
            X = combined[FINANCIAL_COLS].values
            y = combined['target'].values
            res = run_model(X, y, FINANCIAL_COLS,
                           "3: Combined Tassnief + Expanded, 4 Ratios", 'loo')
            results['combined_tassnief_expanded'] = res
    except FileNotFoundError:
        print("Original Tassnief file not found, skipping Experiment 3")
    
    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<50} {'N':>5} {'Acc':>8}")
    print("-" * 65)
    for key, res in results.items():
        print(f"{res['model']:<50} {res['n_samples']:>5} {res['accuracy']:>7.1%}")
    
    print(f"\n--- Previous Baselines ---")
    print(f"Tassnief-only (23 samples, StratifiedKFold): ~65.2%")
    print(f"Tassnief + KAMs (26 samples, StratifiedKFold): ~65.2%")
    
    if results:
        best = max(results.values(), key=lambda x: x['accuracy'])
        print(f"\nBest expanded: {best['accuracy']:.1%} ({best['model']})")
    
    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
