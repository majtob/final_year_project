"""
XGBoost Model with KAMs + Financial Ratios

This model combines:
- 4 Financial ratios (LIQUID, CUMPROF, PROFITAB, LEVERAGE)
- 5 KAM binary features (going_concern, revenue, assets, liabilities, other)
- 1 KAM count feature

Based on Muñoz-Izquierdo et al. (2022) - expecting ~84% accuracy with combined features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FINANCIALS_FILE = PROJECT_ROOT / "data" / "processed" / "ratings_with_financials.csv"
KAMS_FILE = PROJECT_ROOT / "data" / "templates" / "kams_priority.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Rating mappings
RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def load_and_merge_data():
    """Load financials and KAMs, merge them."""
    print("Loading data...")
    
    # Load financials
    fin_df = pd.read_csv(FINANCIALS_FILE)
    print(f"Financials: {len(fin_df)} records")
    
    # Load KAMs
    kams_df = pd.read_csv(KAMS_FILE)
    print(f"KAMs: {len(kams_df)} records")
    
    # Standardize column names for merging
    kams_df = kams_df.rename(columns={'fiscal_year': 'fiscal_year_kam'})
    
    # Merge on ticker and fiscal_year
    # First, ensure fiscal_year types match
    fin_df['fiscal_year'] = fin_df['fiscal_year'].astype(int)
    kams_df['fiscal_year_kam'] = kams_df['fiscal_year_kam'].astype(int)
    
    # Merge
    merged = pd.merge(
        fin_df,
        kams_df[['ticker', 'fiscal_year_kam', 'kam_going_concern', 'kam_revenue', 
                 'kam_assets', 'kam_liabilities', 'kam_other', 'kam_count']],
        left_on=['ticker', 'fiscal_year'],
        right_on=['ticker', 'fiscal_year_kam'],
        how='inner'
    )
    
    print(f"Merged: {len(merged)} records")
    
    return merged


def prepare_features(df):
    """Prepare feature matrix and target."""
    
    # Financial features
    financial_cols = ['liquid', 'cumprof', 'profitab', 'leverage']
    
    # KAM features
    kam_cols = ['kam_going_concern', 'kam_revenue', 'kam_assets', 
                'kam_liabilities', 'kam_other', 'kam_count']
    
    # All features
    all_features = financial_cols + kam_cols
    
    # Check for missing values
    print("\nMissing values:")
    for col in all_features:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing}")
        else:
            print(f"  {col}: COLUMN NOT FOUND")
    
    # Drop rows with missing financial data
    df_clean = df.dropna(subset=financial_cols)
    print(f"\nRecords after dropping NaN: {len(df_clean)}")
    
    # Fill any missing KAM values with 0
    for col in kam_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    # Create target variable
    df_clean['rating_numeric'] = df_clean['rating'].map(RATING_TO_NUMERIC)
    
    # Multi-class categories
    def rating_category(rating):
        if rating in ['AAA', 'AA+', 'AA', 'AA-']:
            return 'AA'
        elif rating in ['A+', 'A', 'A-']:
            return 'A'
        elif rating in ['BBB+', 'BBB', 'BBB-']:
            return 'BBB'
        else:
            return 'BB'
    
    df_clean['rating_category'] = df_clean['rating'].apply(rating_category)
    
    print("\nRating distribution:")
    print(df_clean['rating'].value_counts())
    
    print("\nCategory distribution:")
    print(df_clean['rating_category'].value_counts())
    
    print("\nKAM feature summary:")
    for col in kam_cols:
        if col in df_clean.columns:
            print(f"  {col}: {df_clean[col].sum()} / {len(df_clean)} ({100*df_clean[col].mean():.1f}%)")
    
    return df_clean, financial_cols, kam_cols


def train_financials_only(df, feature_cols):
    """Train model with only financial features (baseline)."""
    print("\n" + "="*60)
    print("MODEL 1: FINANCIALS ONLY (Baseline)")
    print("="*60)
    
    X = df[feature_cols].values
    y_cat = df['rating_category'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_cat)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Cross-validation
    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"\n{n_splits}-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    else:
        cv_scores = [0]
    
    model.fit(X, y)
    
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    return model, le, cv_scores


def train_kams_only(df, kam_cols):
    """Train model with only KAM features."""
    print("\n" + "="*60)
    print("MODEL 2: KAMs ONLY")
    print("="*60)
    
    X = df[kam_cols].values
    y_cat = df['rating_category'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_cat)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"\n{n_splits}-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    else:
        cv_scores = [0]
    
    model.fit(X, y)
    
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(kam_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    return model, le, cv_scores


def train_combined(df, financial_cols, kam_cols):
    """Train model with combined features."""
    print("\n" + "="*60)
    print("MODEL 3: COMBINED (Financials + KAMs)")
    print("="*60)
    
    all_features = financial_cols + kam_cols
    X = df[all_features].values
    y_cat = df['rating_category'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_cat)
    
    print(f"\nClasses: {le.classes_}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Features: {len(all_features)}")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"\n{n_splits}-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    else:
        cv_scores = [0]
    
    model.fit(X, y)
    
    # Full training accuracy
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {train_acc:.2%}")
    
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    print("\nClassification Report (on training data):")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    print(f"Classes: {le.classes_}")
    print(confusion_matrix(y, y_pred))
    
    return model, le, cv_scores, all_features


def save_results(results_dict):
    """Save all results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results_file = RESULTS_DIR / 'combined_model_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("XGBoost WITH KAMs + FINANCIALS")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and merge data
    merged_df = load_and_merge_data()
    
    # Prepare features
    df, financial_cols, kam_cols = prepare_features(merged_df)
    
    if len(df) < 5:
        print("\nERROR: Not enough samples")
        return
    
    # Train models
    fin_model, fin_le, fin_scores = train_financials_only(df, financial_cols)
    kam_model, kam_le, kam_scores = train_kams_only(df, kam_cols)
    combined_model, combined_le, combined_scores, all_features = train_combined(
        df, financial_cols, kam_cols
    )
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'CV Accuracy':<15} {'Paper Reference':<15}")
    print("-"*55)
    print(f"{'Financials Only':<25} {np.mean(fin_scores)*100:>6.2f}%        71.55%")
    print(f"{'KAMs Only':<25} {np.mean(kam_scores)*100:>6.2f}%        74.14%")
    print(f"{'Combined':<25} {np.mean(combined_scores)*100:>6.2f}%        84.04%")
    print("-"*55)
    
    improvement = np.mean(combined_scores) - np.mean(fin_scores)
    print(f"\nImprovement from adding KAMs: {improvement*100:+.2f}%")
    
    # Save results
    results = {
        'date': datetime.now().isoformat(),
        'dataset_size': len(df),
        'features': {
            'financial': financial_cols,
            'kam': kam_cols,
            'total': len(all_features)
        },
        'models': {
            'financials_only': {
                'accuracy': float(np.mean(fin_scores)),
                'std': float(np.std(fin_scores)),
                'paper_reference': 0.7155
            },
            'kams_only': {
                'accuracy': float(np.mean(kam_scores)),
                'std': float(np.std(kam_scores)),
                'paper_reference': 0.7414
            },
            'combined': {
                'accuracy': float(np.mean(combined_scores)),
                'std': float(np.std(combined_scores)),
                'paper_reference': 0.8404
            }
        },
        'improvement': float(improvement)
    }
    
    save_results(results)
    
    return combined_model, results


if __name__ == "__main__":
    main()
