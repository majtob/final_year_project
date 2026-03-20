"""
Baseline XGBoost Model for Credit Rating Prediction

This model uses only financial ratios to predict credit ratings.
Based on Muñoz-Izquierdo et al. (2022) methodology.

Features:
- LIQUID: Working Capital / Total Assets
- CUMPROF: Retained Earnings / Total Assets  
- PROFITAB: EBIT / Total Assets
- LEVERAGE: Book Value of Equity / Total Liabilities

Target: Credit Rating (binary: Investment Grade vs Speculative)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "ratings_with_financials.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


# Rating mappings
RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
    'CCC+': 5, 'CCC': 4, 'CCC-': 3,
    'CC': 2, 'C': 1, 'D': 0
}

# Investment grade: BBB- and above (numeric >= 12)
INVESTMENT_GRADE_THRESHOLD = 12


def load_and_prepare_data():
    """Load data and prepare features/target."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    print(f"Total records: {len(df)}")
    print(f"Records with data: {df['data_available'].sum()}")
    
    # Filter to records with financial data
    df_valid = df[df['data_available'] == True].copy()
    print(f"Using {len(df_valid)} records with financial data")
    
    # Feature columns
    feature_cols = ['liquid', 'cumprof', 'profitab', 'leverage']
    
    # Check for missing values
    print("\nMissing values per feature:")
    for col in feature_cols:
        missing = df_valid[col].isna().sum()
        print(f"  {col}: {missing} ({100*missing/len(df_valid):.1f}%)")
    
    # Drop rows with missing feature values
    df_clean = df_valid.dropna(subset=feature_cols)
    print(f"\nRecords after dropping NaN: {len(df_clean)}")
    
    # Create target variable
    # Convert rating to numeric
    df_clean['rating_numeric'] = df_clean['rating'].map(RATING_TO_NUMERIC)
    
    # Binary classification: Investment Grade (1) vs Speculative (0)
    df_clean['investment_grade'] = (df_clean['rating_numeric'] >= INVESTMENT_GRADE_THRESHOLD).astype(int)
    
    # Multi-class: rating categories
    def rating_category(rating):
        if rating in ['AAA', 'AA+', 'AA', 'AA-']:
            return 'AA'  # High investment grade
        elif rating in ['A+', 'A', 'A-']:
            return 'A'   # Upper medium investment grade
        elif rating in ['BBB+', 'BBB', 'BBB-']:
            return 'BBB' # Lower investment grade
        else:
            return 'BB'  # Speculative (BB+ and below)
    
    df_clean['rating_category'] = df_clean['rating'].apply(rating_category)
    
    print("\nRating distribution:")
    print(df_clean['rating'].value_counts())
    
    print("\nBinary target distribution:")
    print(df_clean['investment_grade'].value_counts())
    print(f"  Investment Grade (1): {df_clean['investment_grade'].sum()}")
    print(f"  Speculative (0): {(df_clean['investment_grade'] == 0).sum()}")
    
    print("\nCategory distribution:")
    print(df_clean['rating_category'].value_counts())
    
    return df_clean, feature_cols


def train_binary_model(df, feature_cols):
    """Train binary classification model (Investment Grade vs Speculative)."""
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION: Investment Grade vs Speculative")
    print("="*60)
    
    X = df[feature_cols].values
    y = df['investment_grade'].values
    
    # Check class balance
    print(f"\nClass distribution: {np.bincount(y)}")
    
    # XGBoost with balanced class weights
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=len(y[y==0]) / len(y[y==1]) if len(y[y==1]) > 0 else 1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Handle case with very few samples
    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        print("Warning: Not enough samples for cross-validation. Using train-test split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Speculative', 'Investment Grade']))
        
        cv_scores = [accuracy]
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\n{n_splits}-Fold Cross-Validation Results:")
        print(f"  Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        # Train final model on all data
        model.fit(X, y)
    
    # Feature importance
    print("\nFeature Importance:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    return model, cv_scores


def train_multiclass_model(df, feature_cols):
    """Train multi-class classification model (AA, A, BBB, BB)."""
    print("\n" + "="*60)
    print("MULTI-CLASS CLASSIFICATION: Rating Categories")
    print("="*60)
    
    X = df[feature_cols].values
    
    # Encode categories
    le = LabelEncoder()
    y = le.fit_transform(df['rating_category'].values)
    
    print(f"\nClasses: {le.classes_}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # XGBoost multi-class
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Cross-validation with fewer splits if needed
    min_class_count = min(np.bincount(y))
    n_splits = min(5, min_class_count)
    
    if n_splits < 2:
        print("Warning: Not enough samples for cross-validation. Using train-test split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        cv_scores = [accuracy]
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\n{n_splits}-Fold Cross-Validation Results:")
        print(f"  Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        # Train final model
        model.fit(X, y)
    
    # Feature importance
    print("\nFeature Importance:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    # Confusion matrix
    y_pred_all = model.predict(X)
    print("\nConfusion Matrix (on training data):")
    cm = confusion_matrix(y, y_pred_all)
    print(f"Classes: {le.classes_}")
    print(cm)
    
    return model, le, cv_scores


def save_results(binary_model, multiclass_model, label_encoder, 
                 binary_scores, multiclass_scores, feature_cols, df):
    """Save model results and metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_name': 'baseline_xgboost_financials_only',
        'created_at': datetime.now().isoformat(),
        'dataset': {
            'total_records': len(df),
            'features': feature_cols,
            'target_binary': 'investment_grade',
            'target_multiclass': 'rating_category'
        },
        'binary_classification': {
            'description': 'Investment Grade (BBB- and above) vs Speculative',
            'cv_accuracy_mean': float(np.mean(binary_scores)),
            'cv_accuracy_std': float(np.std(binary_scores)),
            'cv_scores': [float(s) for s in binary_scores],
            'feature_importance': dict(zip(feature_cols, 
                                          [float(x) for x in binary_model.feature_importances_]))
        },
        'multiclass_classification': {
            'description': 'Rating categories: AA, A, BBB, BB',
            'classes': list(label_encoder.classes_),
            'cv_accuracy_mean': float(np.mean(multiclass_scores)),
            'cv_accuracy_std': float(np.std(multiclass_scores)),
            'cv_scores': [float(s) for s in multiclass_scores],
            'feature_importance': dict(zip(feature_cols,
                                          [float(x) for x in multiclass_model.feature_importances_]))
        },
        'comparison_to_paper': {
            'paper_accuracy': 0.74,
            'paper_method': 'PART algorithm with KAMs only',
            'paper_combined_accuracy': 0.84,
            'note': 'Paper used KAMs + financial ratios for 84% accuracy'
        }
    }
    
    # Save results
    results_file = RESULTS_DIR / 'baseline_financials_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results


def main():
    """Main training pipeline."""
    print("="*60)
    print("BASELINE XGBoost MODEL - FINANCIALS ONLY")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df, feature_cols = load_and_prepare_data()
    
    if len(df) < 5:
        print("\nERROR: Not enough samples to train a model.")
        print("Need at least 5 samples with complete financial data.")
        return
    
    # Train binary model
    binary_model, binary_scores = train_binary_model(df, feature_cols)
    
    # Train multi-class model
    multiclass_model, label_encoder, multiclass_scores = train_multiclass_model(df, feature_cols)
    
    # Save results
    results = save_results(
        binary_model, multiclass_model, label_encoder,
        binary_scores, multiclass_scores, feature_cols, df
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset size: {len(df)} samples")
    print(f"Features: {feature_cols}")
    print(f"\nBinary Classification Accuracy: {np.mean(binary_scores):.2%}")
    print(f"Multi-class Classification Accuracy: {np.mean(multiclass_scores):.2%}")
    print(f"\nReference (Paper): 74% with KAMs only, 84% with KAMs + financials")
    print("\nNext steps:")
    print("1. Add KAM features (5 binary variables)")
    print("2. Add news sentiment features")
    print("3. Implement LLM verdict generation")
    
    return binary_model, multiclass_model, results


if __name__ == "__main__":
    main()
