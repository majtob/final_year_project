"""
Full XGBoost Model: Financials + KAMs + News Sentiment

This is the final model combining all three feature sets:
1. Financial ratios (4): LIQUID, CUMPROF, PROFITAB, LEVERAGE
2. KAM features (6): 5 binary categories + count
3. Sentiment features (4): mean, std, pos_pct, neg_pct, news_count

Trains 4 models for comparison:
- Model 1: Financials only (baseline)
- Model 2: Financials + KAMs
- Model 3: Financials + Sentiment
- Model 4: Financials + KAMs + Sentiment (full)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
SENTIMENT_FILE = PROJECT_ROOT / "data" / "processed" / "ratings_financials_sentiment.csv"
KAMS_FILE = PROJECT_ROOT / "data" / "templates" / "kams_priority.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

FINANCIAL_COLS = ['liquid', 'cumprof', 'profitab', 'leverage']
KAM_COLS = ['kam_going_concern', 'kam_revenue', 'kam_assets', 'kam_liabilities', 'kam_other', 'kam_count']
SENTIMENT_COLS = ['sentiment_mean', 'sentiment_std', 'sentiment_pos_pct', 'sentiment_neg_pct', 'news_count']


def load_data():
    """Load and merge all data sources."""
    print("Loading data...")
    
    # Load sentiment+financials
    df = pd.read_csv(SENTIMENT_FILE)
    print(f"Sentiment+Financials: {len(df)} records")
    
    # Load KAMs
    kams = pd.read_csv(KAMS_FILE)
    kams = kams.rename(columns={'fiscal_year': 'fy_kam'})
    kams['fy_kam'] = kams['fy_kam'].astype(int)
    print(f"KAMs: {len(kams)} records")
    
    # Merge KAMs
    df['fiscal_year'] = df['fiscal_year'].astype(int)
    merged = pd.merge(
        df,
        kams[['ticker', 'fy_kam'] + KAM_COLS],
        left_on=['ticker', 'fiscal_year'],
        right_on=['ticker', 'fy_kam'],
        how='inner'
    )
    print(f"Merged (all three sources): {len(merged)} records")
    
    return merged


def prepare_target(df):
    """Create multi-class target variable."""
    def rating_category(r):
        if r in ['AAA', 'AA+', 'AA', 'AA-']:
            return 'AA'
        elif r in ['A+', 'A', 'A-']:
            return 'A'
        elif r in ['BBB+', 'BBB', 'BBB-']:
            return 'BBB'
        else:
            return 'BB'
    
    df['rating_category'] = df['rating'].apply(rating_category)
    return df


def run_model(X, y, le, feature_names, model_name):
    """Train and evaluate a single model configuration."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Samples: {len(y)}")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y)))}")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        print(f"  WARNING: Not enough samples for CV (min class = {min(np.bincount(y))})")
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        cv_scores = [acc]
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        model.fit(X, y)
    
    print(f"\n  CV Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores)*2:.2%})")
    
    # Feature importance
    print(f"\n  Feature Importance:")
    importance = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importance:
        bar = '#' * int(imp * 40)
        print(f"    {feat:<25} {imp:.4f}  {bar}")
    
    # Confusion matrix on training data
    y_pred_all = model.predict(X)
    print(f"\n  Training Accuracy: {accuracy_score(y, y_pred_all):.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred_all, target_names=le.classes_, digits=3))
    
    return model, cv_scores, importance


def main():
    print("=" * 60)
    print("  FULL MODEL COMPARISON")
    print("  Financials vs KAMs vs Sentiment vs Combined")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    df = load_data()
    df = prepare_target(df)
    
    # Drop rows with missing financial data
    all_needed = FINANCIAL_COLS + SENTIMENT_COLS
    df_clean = df.dropna(subset=FINANCIAL_COLS).copy()
    
    # Fill missing sentiment with neutral values
    df_clean['sentiment_mean'] = df_clean['sentiment_mean'].fillna(0)
    df_clean['sentiment_std'] = df_clean['sentiment_std'].fillna(0)
    df_clean['sentiment_pos_pct'] = df_clean['sentiment_pos_pct'].fillna(0)
    df_clean['sentiment_neg_pct'] = df_clean['sentiment_neg_pct'].fillna(0)
    df_clean['news_count'] = df_clean['news_count'].fillna(0)
    
    # Fill missing KAM values
    for col in KAM_COLS:
        df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    print(f"\nUsable records: {len(df_clean)}")
    print(f"\nRating distribution:")
    print(df_clean['rating_category'].value_counts().to_string())
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_clean['rating_category'].values)
    
    # Run all 4 models
    results = {}
    
    # Model 1: Financials only
    X1 = df_clean[FINANCIAL_COLS].values
    m1, s1, imp1 = run_model(X1, y, le, FINANCIAL_COLS, "MODEL 1: Financials Only")
    results['financials_only'] = {'accuracy': float(np.mean(s1)), 'std': float(np.std(s1))}
    
    # Model 2: Financials + KAMs
    feat2 = FINANCIAL_COLS + KAM_COLS
    X2 = df_clean[feat2].values
    m2, s2, imp2 = run_model(X2, y, le, feat2, "MODEL 2: Financials + KAMs")
    results['financials_kams'] = {'accuracy': float(np.mean(s2)), 'std': float(np.std(s2))}
    
    # Model 3: Financials + Sentiment
    feat3 = FINANCIAL_COLS + SENTIMENT_COLS
    X3 = df_clean[feat3].values
    m3, s3, imp3 = run_model(X3, y, le, feat3, "MODEL 3: Financials + Sentiment")
    results['financials_sentiment'] = {'accuracy': float(np.mean(s3)), 'std': float(np.std(s3))}
    
    # Model 4: All features
    feat4 = FINANCIAL_COLS + KAM_COLS + SENTIMENT_COLS
    X4 = df_clean[feat4].values
    m4, s4, imp4 = run_model(X4, y, le, feat4, "MODEL 4: Full (Financials + KAMs + Sentiment)")
    results['full_model'] = {'accuracy': float(np.mean(s4)), 'std': float(np.std(s4))}
    
    # Final comparison
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<40} {'CV Accuracy':<15} {'Paper Ref'}")
    print(f"  {'-'*65}")
    print(f"  {'1. Financials Only':<40} {np.mean(s1)*100:>6.2f}%        71.55%")
    print(f"  {'2. Financials + KAMs':<40} {np.mean(s2)*100:>6.2f}%        84.04%")
    print(f"  {'3. Financials + Sentiment':<40} {np.mean(s3)*100:>6.2f}%        -")
    print(f"  {'4. Full (Fin + KAMs + Sentiment)':<40} {np.mean(s4)*100:>6.2f}%        -")
    print(f"  {'-'*65}")
    
    best = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n  Best model: {best[0]} ({best[1]['accuracy']*100:.2f}%)")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output = {
        'date': datetime.now().isoformat(),
        'dataset_size': len(df_clean),
        'classes': list(le.classes_),
        'class_distribution': dict(zip(le.classes_, [int(x) for x in np.bincount(y)])),
        'models': results,
        'feature_sets': {
            'financial': FINANCIAL_COLS,
            'kam': KAM_COLS,
            'sentiment': SENTIMENT_COLS,
        }
    }
    
    results_file = RESULTS_DIR / 'full_model_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()
