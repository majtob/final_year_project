"""
XGBoost Model with KAMs + Financial Ratios

This model combines four financial ratios with paper-style KAM dummies (GCKAM, REVKAM,
ASSETKAM, LIABKAM, OTHERKAM) for the combined model; a separate KAM-only benchmark uses
the full kams_processed feature set (aligned with xgboost_kams_only.py).

Based on Muñoz-Izquierdo et al. (2022) - expecting ~84% accuracy with combined features.
"""

import argparse
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

# Paths (paper-style KAM columns: GCKAM, REVKAM, … — see data/processed/kams_processed.csv)
PROJECT_ROOT = Path(__file__).parent.parent
FINANCIALS_FILE = PROJECT_ROOT / "data" / "processed" / "financial_ratios_processed.csv"
# When ratios file is absent (slim pipeline), use multi-agency financials + kams rating.
FINANCIALS_ALT = PROJECT_ROOT / "data" / "processed" / "financials_processed.csv"
KAMS_FILE = PROJECT_ROOT / "data" / "processed" / "kams_processed.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Muñoz-Izquierdo et al. (2022) five KAM dummies (no aggregate KAM_COUNT)
PAPER_KAM_COLS = ["GCKAM", "REVKAM", "ASSETKAM", "LIABKAM", "OTHERKAM"]

# Full KAM / control columns — same as models/xgboost_kams_only.py (Model 2 uses these)
FULL_KAM_FEATURE_COLS = [
    "AUSIZE",
    "AUOP",
    "EMP",
    "GCUP",
    "GCKAM",
    "REVKAM",
    "ASSETKAM",
    "LIABKAM",
    "OTHERKAM",
    "FIRMAGE",
    "FIRMSIZE",
    "INDUSTRY",
]

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

    kams_df = pd.read_csv(KAMS_FILE)
    print(f"KAMs: {len(kams_df)} records")
    kams_df = kams_df.rename(columns={"fiscal_year": "fiscal_year_kam"})
    kams_df["fiscal_year_kam"] = kams_df["fiscal_year_kam"].astype(int)

    if FINANCIALS_FILE.exists():
        fin_df = pd.read_csv(FINANCIALS_FILE)
        print(f"Financials: {len(fin_df)} records (financial_ratios_processed.csv)")
        fin_df["fiscal_year"] = fin_df["fiscal_year"].astype(int)
        merge_kam = ["ticker", "fiscal_year_kam"] + FULL_KAM_FEATURE_COLS
        merge_kam = list(dict.fromkeys(merge_kam))
        merge_kam = [c for c in merge_kam if c in kams_df.columns]
        merged = pd.merge(
            fin_df,
            kams_df[merge_kam],
            left_on=["ticker", "fiscal_year"],
            right_on=["ticker", "fiscal_year_kam"],
            how="inner",
        )
    else:
        if not FINANCIALS_ALT.exists():
            raise FileNotFoundError(
                f"Missing both {FINANCIALS_FILE.name} and {FINANCIALS_ALT.name}"
            )
        fin_df = pd.read_csv(FINANCIALS_ALT)
        print(f"Financials: {len(fin_df)} records (financials_processed.csv, raw)")
        fin_df["fiscal_year"] = pd.to_numeric(
            fin_df["fiscal_year"], errors="coerce"
        ).astype("Int64")
        fin_df = fin_df.dropna(subset=["fiscal_year"])
        fin_df["fiscal_year"] = fin_df["fiscal_year"].astype(int)
        # One row per (ticker, year): prefer Tassnief for ratios
        agency = fin_df["rating_agency"].astype(str).str.strip().str.lower()
        fin_df = fin_df.assign(_pri=(agency == "tassnief").astype(int))
        fin_df = fin_df.sort_values("_pri", ascending=False)
        fin_df = fin_df.drop_duplicates(
            subset=["ticker", "fiscal_year"], keep="first"
        ).drop(columns=["_pri"])
        ratio_cols = ["ticker", "fiscal_year", "liquid", "cumprof", "profitab", "leverage"]
        missing = [c for c in ratio_cols if c not in fin_df.columns]
        if missing:
            raise ValueError(f"financials_processed.csv missing columns: {missing}")
        keep_extra = [c for c in ("company_name", "rating_agency") if c in fin_df.columns]
        fin_df = fin_df[ratio_cols + keep_extra].copy()
        print(f"Financials after dedupe: {len(fin_df)} records")
        merge_kam = ["ticker", "fiscal_year_kam", "rating"] + FULL_KAM_FEATURE_COLS
        merge_kam = list(dict.fromkeys(merge_kam))
        merge_kam = [c for c in merge_kam if c in kams_df.columns]
        merged = pd.merge(
            fin_df,
            kams_df[merge_kam],
            left_on=["ticker", "fiscal_year"],
            right_on=["ticker", "fiscal_year_kam"],
            how="inner",
        )

    print(f"Merged: {len(merged)} records")
    return merged


def prepare_features(df):
    """Prepare feature matrix and target."""
    
    # Financial features
    financial_cols = ['liquid', 'cumprof', 'profitab', 'leverage']

    for c in FULL_KAM_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    full_kam_cols = list(FULL_KAM_FEATURE_COLS)

    # All features for missing-value report
    all_features = financial_cols + full_kam_cols
    
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
    for col in full_kam_cols:
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
    
    print("\nKAM feature summary (paper five dummies):")
    for col in PAPER_KAM_COLS:
        if col in df_clean.columns:
            print(f"  {col}: {df_clean[col].sum()} / {len(df_clean)} ({100*df_clean[col].mean():.1f}%)")
    
    return df_clean, financial_cols, full_kam_cols


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
    """Train model with only KAM features (same columns as xgboost_kams_only.py)."""
    print("\n" + "="*60)
    print("MODEL 2: KAMs ONLY (full kams_processed features, aligned with xgboost_kams_only.py)")
    print("="*60)
    print(f"  KAM features ({len(kam_cols)}): {kam_cols}")
    
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
    parser = argparse.ArgumentParser(
        description="Financial + KAM XGBoost (optionally skip merged KAM-only; use xgboost_kams_only.py first)."
    )
    parser.add_argument(
        "--financial-and-combined-only",
        action="store_true",
        help="Train only financials + combined on merged sample. Run models/xgboost_kams_only.py first for KAM-only.",
    )
    args = parser.parse_args()

    print("="*60)
    print("XGBoost WITH KAMs + FINANCIALS")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.financial_and_combined_only:
        print("Mode: financial + combined only (KAM-only full panel: xgboost_kams_only.py)\n")
    
    # Load and merge data
    merged_df = load_and_merge_data()
    
    # Prepare features
    df, financial_cols, kam_cols = prepare_features(merged_df)
    
    if len(df) < 5:
        print("\nERROR: Not enough samples")
        return
    
    kam_scores = None
    if args.financial_and_combined_only:
        # Order: 2) financials, 3) combined (KAM-only is step 1 via separate script)
        fin_model, fin_le, fin_scores = train_financials_only(df, financial_cols)
        combined_model, combined_le, combined_scores, all_features = train_combined(
            df, financial_cols, kam_cols
        )
    else:
        # Order: KAMs (merged sample) → financials → combined
        kam_model, kam_le, kam_scores = train_kams_only(df, kam_cols)
        fin_model, fin_le, fin_scores = train_financials_only(df, financial_cols)
        combined_model, combined_le, combined_scores, all_features = train_combined(
            df, financial_cols, kam_cols
        )
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS COMPARISON (merged sample)")
    print("="*60)
    print(f"{'Model':<35} {'CV Accuracy':<15} {'Paper Reference':<15}")
    print("-"*65)
    if not args.financial_and_combined_only:
        print(f"{'KAMs only (12 feats, merged n)':<35} {np.mean(kam_scores)*100:>6.2f}%        74.14% *")
    else:
        print(f"{'KAMs only (full panel)':<35} {'—':<15} see kams_only_model_results.json")
    print(f"{'Financials only':<35} {np.mean(fin_scores)*100:>6.2f}%        71.55%")
    print(f"{'KAMs + financials (combined)':<35} {np.mean(combined_scores)*100:>6.2f}%        84.04%")
    print("-"*65)
    if not args.financial_and_combined_only:
        print("* Paper KAM-only row used 6 features; merged KAM model uses 12 like xgboost_kams_only.py.")
    
    improvement = np.mean(combined_scores) - np.mean(fin_scores)
    print(f"\nImprovement (combined vs financials-only on this sample): {improvement*100:+.2f}%")
    
    # Save results
    models_out = {
        'financials_only': {
            'accuracy': float(np.mean(fin_scores)),
            'std': float(np.std(fin_scores)),
            'paper_reference': 0.7155
        },
        'combined': {
            'accuracy': float(np.mean(combined_scores)),
            'std': float(np.std(combined_scores)),
            'paper_reference': 0.8404
        },
    }
    if kam_scores is not None:
        models_out['kams_only_merged_sample'] = {
            'accuracy': float(np.mean(kam_scores)),
            'std': float(np.std(kam_scores)),
            'paper_reference': 0.7414,
            'note': 'Same rows as financial/combined; for full KAM panel use xgboost_kams_only.py',
        }

    results = {
        'date': datetime.now().isoformat(),
        'financials_source': (
            FINANCIALS_FILE.name
            if FINANCIALS_FILE.exists()
            else FINANCIALS_ALT.name
        ),
        'dataset_size': len(df),
        'financial_and_combined_only': args.financial_and_combined_only,
        'kams_only_full_panel_file': str(
            (RESULTS_DIR / 'kams_only_model_results.json').relative_to(PROJECT_ROOT)
        ),
        'features': {
            'financial': financial_cols,
            'kam_full': kam_cols,
            'kam_note': 'Same 12 columns as models/xgboost_kams_only.py',
            'total': len(all_features),
        },
        'models': models_out,
        'improvement': float(improvement)
    }
    
    save_results(results)
    
    return combined_model, results


if __name__ == "__main__":
    main()
