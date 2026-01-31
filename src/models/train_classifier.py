#!/usr/bin/env python3
"""
Credit Rating Classifier Training
Trains XGBoost/Random Forest classifier to predict Tassnief credit ratings.
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "model_training_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


# Feature column groups
FINANCIAL_RATIO_COLS = [
    "current_ratio", "quick_ratio", "cash_ratio",
    "debt_to_equity", "debt_to_assets", "long_term_debt_ratio",
    "interest_coverage", "debt_service_coverage",
    "return_on_assets", "return_on_equity", "net_margin",
    "operating_margin", "gross_margin",
    "asset_turnover", "receivables_turnover",
    "revenue_growth", "net_income_growth"
]

KAM_COLS = [
    "kam_count", "kam_going_concern", "kam_impairment",
    "kam_revenue_recognition", "kam_related_party", "kam_litigation",
    "kam_valuation", "kam_severity_score"
]

NEWS_COLS = [
    "news_count", "avg_sentiment", "sentiment_std",
    "negative_news_ratio", "positive_news_ratio",
    "event_lawsuit", "event_expansion", "event_regulatory", "event_earnings"
]

MARKET_COLS = [
    "avg_price", "price_volatility", "avg_volume",
    "market_cap", "price_return"
]

ALL_FEATURE_COLS = FINANCIAL_RATIO_COLS + KAM_COLS + NEWS_COLS + MARKET_COLS


class RatingClassifier:
    """Credit rating classifier using XGBoost."""
    
    def __init__(
        self,
        model_type: str = "xgboost",
        target_type: str = "multiclass",  # or "binary"
        random_state: int = 42
    ):
        """
        Initialize classifier.
        
        Args:
            model_type: "xgboost" or "random_forest"
            target_type: "multiclass" (full ratings) or "binary" (IG vs Spec)
            random_state: Random seed
        """
        self.model_type = model_type
        self.target_type = target_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.training_metadata = {}
    
    def _create_model(self, n_classes: int):
        """Create the ML model."""
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                if self.target_type == "binary":
                    return xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        use_label_encoder=False,
                        eval_metric="logloss"
                    )
                else:
                    return xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        use_label_encoder=False,
                        eval_metric="mlogloss",
                        objective="multi:softprob",
                        num_class=n_classes
                    )
            except ImportError:
                logger.warning("XGBoost not installed, falling back to Random Forest")
                self.model_type = "random_forest"
        
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight="balanced"
            )
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "tassnief_rating"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X, y)
        """
        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        self.feature_columns = feature_cols
        
        # Filter rows with valid target
        df_valid = df[df[target_col].notna()].copy()
        
        if len(df_valid) == 0:
            raise ValueError("No valid target values found")
        
        logger.info(f"Valid samples: {len(df_valid)} / {len(df)}")
        
        # Prepare target
        if self.target_type == "binary":
            # Use rating_bucket if available, else derive from rating
            if "rating_bucket" in df_valid.columns:
                y = df_valid["rating_bucket"]
            else:
                # Map ratings to binary
                investment_grade = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", 
                                   "BBB+", "BBB", "BBB-"}
                y = df_valid[target_col].apply(
                    lambda x: "investment_grade" if x in investment_grade else "speculative"
                )
        else:
            y = df_valid[target_col]
        
        # Prepare features
        X = df_valid[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Test set proportion
            
        Returns:
            Dictionary with training results
        """
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model(n_classes)
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        
        # Ordinal correlation (Spearman)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring="accuracy"
        )
        
        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            importance = {}
        
        results = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "spearman_correlation": spearman_corr,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "feature_importance": importance,
            "classes": list(self.label_encoder.classes_),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        
        self.training_metadata = {
            "model_type": self.model_type,
            "target_type": self.target_type,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "n_classes": n_classes,
            "training_date": datetime.now().isoformat(),
            "results": {k: v for k, v in results.items() 
                       if k not in ["classification_report", "confusion_matrix", "feature_importance"]}
        }
        
        logger.info(f"Training completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"  Spearman correlation: {spearman_corr:.4f}")
        logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        X_scaled = self.scaler.transform(X[self.feature_columns])
        predictions_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions, probabilities
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save model and artifacts.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save model
        model_path = output_dir / f"{self.model_type}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        paths["model"] = model_path
        
        # Save scaler
        scaler_path = output_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        paths["scaler"] = scaler_path
        
        # Save label encoder
        encoder_path = output_dir / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        paths["encoder"] = encoder_path
        
        # Save metadata
        metadata_path = output_dir / "model_config.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        paths["metadata"] = metadata_path
        
        logger.info(f"Model saved to {output_dir}")
        return paths
    
    def load(self, model_dir: Path):
        """
        Load model and artifacts.
        
        Args:
            model_dir: Directory with saved model
        """
        model_dir = Path(model_dir)
        
        # Load model
        model_files = list(model_dir.glob("*_model.pkl"))
        if model_files:
            with open(model_files[0], 'rb') as f:
                self.model = pickle.load(f)
        
        # Load scaler
        scaler_path = model_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load encoder
        encoder_path = model_dir / "label_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        # Load metadata
        metadata_path = model_dir / "model_config.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)
                self.feature_columns = self.training_metadata.get("feature_columns", [])
        
        logger.info(f"Model loaded from {model_dir}")


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train credit rating classifier")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features" / "features.parquet"),
        help="Path to features file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Output directory for model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "random_forest"],
        default="xgboost",
        help="Model type"
    )
    parser.add_argument(
        "--target-type",
        type=str,
        choices=["multiclass", "binary"],
        default="multiclass",
        help="Target type"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion"
    )
    
    args = parser.parse_args()
    
    # Load features
    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Please run feature extraction first")
        return
    
    if features_path.suffix == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_json(features_path)
    
    logger.info(f"Loaded {len(df)} samples from {features_path}")
    
    # Initialize classifier
    classifier = RatingClassifier(
        model_type=args.model_type,
        target_type=args.target_type
    )
    
    # Prepare data
    X, y = classifier.prepare_data(df)
    
    # Train
    results = classifier.train(X, y, test_size=args.test_size)
    
    # Save
    classifier.save(Path(args.output))
    
    # Save results
    results_path = Path(args.output) / "training_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        results_json = {
            k: v if not isinstance(v, (np.ndarray, np.floating, np.integer)) 
            else (v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in results.items()
        }
        json.dump(results_json, f, indent=2)
    
    logger.success("Training completed successfully!")


if __name__ == "__main__":
    main()
