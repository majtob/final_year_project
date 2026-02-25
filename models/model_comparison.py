"""
Compare multiple ML models for credit rating prediction.

Models tested:
1. XGBoost
2. Random Forest
3. Gradient Boosting
4. Logistic Regression (ordinal-aware via OvR)
5. SVM (RBF kernel)
6. K-Nearest Neighbors
7. Decision Tree (C4.5 equivalent - reference paper's best model)
8. Naive Bayes

Uses the expanded multi-agency dataset with two configurations:
A. Non-financial companies, 4 ratios (35 samples)
B. All companies, 2 ratios (78 samples)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXPANDED_FILE = PROJECT_ROOT / "data" / "processed" / "expanded_ratings_financials.csv"
ORIGINAL_FILE = PROJECT_ROOT / "data" / "processed" / "ratings_with_financials.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "model_comparison_results.json"

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


def merge_rare(series, min_count=2):
    counts = series.value_counts()
    mapping = {}
    for cls in counts.index:
        if counts[cls] < min_count:
            mapping[cls] = 'A' if cls == 'AA' else ('BBB' if cls == 'BB' else cls)
        else:
            mapping[cls] = cls
    return series.map(mapping)


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
            ('clf', LogisticRegression(
                max_iter=1000, random_state=42,
            )),
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42)),
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=3)),
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, random_state=42,
        ),
        'Naive Bayes': GaussianNB(),
    }


def run_comparison(X, y, feature_names, dataset_name):
    """Run all models on a dataset and compare."""
    print(f"\n{'#'*70}")
    print(f"DATASET: {dataset_name}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"{'#'*70}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Classes: {list(le.classes_)}")
    for cls in le.classes_:
        print(f"  {cls}: {(y == cls).sum()}")
    
    min_class = pd.Series(y).value_counts().min()
    if min_class >= 2:
        cv = LeaveOneOut()
        cv_name = "LOO"
    else:
        n_splits = max(2, min(5, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_name = f"{n_splits}-Fold"
    
    print(f"CV Strategy: {cv_name}")
    
    models = get_models()
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y_encoded, cv=cv)
            acc = accuracy_score(y_encoded, y_pred)
            report = classification_report(y_encoded, y_pred, 
                                          target_names=le.classes_, 
                                          output_dict=True, zero_division=0)
            cm = confusion_matrix(y_encoded, y_pred)
            
            # Get feature importance if available
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
            
            print(f"\n  {name:25s} Accuracy: {acc:.1%}")
            
        except Exception as e:
            print(f"\n  {name:25s} FAILED: {e}")
            results[name] = {'accuracy': 0, 'error': str(e)}
    
    return results, le.classes_


def main():
    # Load expanded data
    df = pd.read_csv(EXPANDED_FILE)
    df = df[df['has_financials'] == True].copy()
    df['target'] = df['rating'].apply(to_category)
    
    # Create consensus (one per company)
    consensus = df.groupby('ticker').agg({
        'company_name': 'first',
        'sector': 'first',
        'rating_numeric': 'median',
        'liquid': 'first',
        'cumprof': 'first',
        'profitab': 'first',
        'leverage': 'first',
    }).reset_index()
    
    consensus['target'] = consensus['rating_numeric'].apply(
        lambda n: 'AA' if n >= 18 else ('A' if n >= 15 else ('BBB' if n >= 12 else 'BB'))
    )
    
    all_results = {}
    
    # === Dataset A: Fitch-only (our best previous) ===
    fitch_df = df[df['rating_agency'] == 'Fitch'].dropna(subset=BANK_SAFE_COLS).copy()
    fitch_df['target'] = merge_rare(fitch_df['target'])
    if len(fitch_df) >= 10 and fitch_df['target'].nunique() >= 2:
        X = fitch_df[BANK_SAFE_COLS].values
        y = fitch_df['target'].values
        res, classes = run_comparison(X, y, BANK_SAFE_COLS, "A: Fitch Only, 2 Ratios")
        all_results['fitch_2ratios'] = {'models': res, 'classes': list(classes), 'n': len(X)}
    
    # === Dataset B: Non-financial consensus, 4 ratios ===
    non_fin = consensus[~consensus['sector'].isin(['Banks', 'Insurance'])].copy()
    non_fin = non_fin.dropna(subset=FINANCIAL_COLS)
    non_fin['target'] = merge_rare(non_fin['target'])
    if len(non_fin) >= 10 and non_fin['target'].nunique() >= 2:
        X = non_fin[FINANCIAL_COLS].values
        y = non_fin['target'].values
        res, classes = run_comparison(X, y, FINANCIAL_COLS, "B: Non-Financial Consensus, 4 Ratios")
        all_results['nonfin_4ratios'] = {'models': res, 'classes': list(classes), 'n': len(X)}
    
    # === Dataset C: All consensus, 2 ratios ===
    all_cos = consensus.dropna(subset=BANK_SAFE_COLS).copy()
    all_cos['target'] = merge_rare(all_cos['target'])
    if len(all_cos) >= 10 and all_cos['target'].nunique() >= 2:
        X = all_cos[BANK_SAFE_COLS].values
        y = all_cos['target'].values
        res, classes = run_comparison(X, y, BANK_SAFE_COLS, "C: All Consensus, 2 Ratios")
        all_results['all_2ratios'] = {'models': res, 'classes': list(classes), 'n': len(X)}
    
    # === Dataset D: Original Tassnief, 4 ratios ===
    try:
        orig = pd.read_csv(ORIGINAL_FILE)
        orig = orig.dropna(subset=FINANCIAL_COLS)
        rating_map = {'AA': 'AA', 'A': 'A', 'BBB': 'BBB', 'BB': 'BB',
                      'AA+': 'AA', 'AA-': 'AA', 'A+': 'A', 'A-': 'A',
                      'BBB+': 'BBB', 'BBB-': 'BBB', 'BB+': 'BB', 'BB-': 'BB',
                      'AAA': 'AA'}
        orig['target'] = orig['rating'].map(rating_map)
        orig = orig.dropna(subset=['target'])
        orig['target'] = merge_rare(orig['target'])
        
        if len(orig) >= 10 and orig['target'].nunique() >= 2:
            X = orig[FINANCIAL_COLS].values
            y = orig['target'].values
            res, classes = run_comparison(X, y, FINANCIAL_COLS, "D: Original Tassnief, 4 Ratios")
            all_results['original_tassnief'] = {'models': res, 'classes': list(classes), 'n': len(X)}
    except FileNotFoundError:
        print("Original Tassnief data not found")
    
    # === FINAL LEADERBOARD ===
    print(f"\n{'='*80}")
    print("FINAL LEADERBOARD - ALL MODELS x ALL DATASETS")
    print(f"{'='*80}")
    print(f"{'Model':<25} ", end="")
    for ds_name in all_results:
        print(f"  {ds_name:>18}", end="")
    print()
    print("-" * (25 + 20 * len(all_results)))
    
    model_names = list(get_models().keys())
    best_overall = (None, None, 0)
    
    for model_name in model_names:
        print(f"{model_name:<25} ", end="")
        for ds_name, ds_data in all_results.items():
            if model_name in ds_data['models'] and 'accuracy' in ds_data['models'][model_name]:
                acc = ds_data['models'][model_name]['accuracy']
                print(f"  {acc:>17.1%}", end="")
                if acc > best_overall[2]:
                    best_overall = (model_name, ds_name, acc)
            else:
                print(f"  {'FAIL':>17}", end="")
        print()
    
    print(f"\n{'='*80}")
    print(f"BEST OVERALL: {best_overall[0]} on {best_overall[1]} = {best_overall[2]:.1%}")
    print(f"{'='*80}")
    
    # Save
    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
