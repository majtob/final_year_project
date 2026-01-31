#!/usr/bin/env python3
"""
ML Model Evaluation
Evaluates the credit rating classifier against Tassnief ratings.
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# ML imports
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "evaluation" / "model_evaluation_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class ModelEvaluator:
    """Evaluates credit rating classifier."""
    
    def __init__(self, model_dir: Path, output_dir: Path):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory with trained model
            output_dir: Output directory for evaluation results
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model artifacts
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = []
        self._load_model()
    
    def _load_model(self):
        """Load model and artifacts."""
        # Load model
        model_files = list(self.model_dir.glob("*_model.pkl"))
        if model_files:
            with open(model_files[0], 'rb') as f:
                self.model = pickle.load(f)
        
        # Load scaler
        scaler_path = self.model_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load encoder
        encoder_path = self.model_dir / "label_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        # Load metadata
        metadata_path = self.model_dir / "model_config.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata.get("feature_columns", [])
        
        logger.info(f"Loaded model from {self.model_dir}")
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            dataset_name: Name of the dataset (for reporting)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare features
        X_eval = X[self.feature_columns].fillna(X[self.feature_columns].median())
        X_scaled = self.scaler.transform(X_eval)
        
        # Encode true labels
        y_true_encoded = self.label_encoder.transform(y_true)
        
        # Predict
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true_encoded, y_pred_encoded)
        metrics["f1_macro"] = f1_score(y_true_encoded, y_pred_encoded, average="macro")
        metrics["f1_weighted"] = f1_score(y_true_encoded, y_pred_encoded, average="weighted")
        metrics["precision_macro"] = precision_score(y_true_encoded, y_pred_encoded, average="macro")
        metrics["recall_macro"] = recall_score(y_true_encoded, y_pred_encoded, average="macro")
        
        # Ordinal correlation (Spearman)
        spearman_corr, spearman_p = spearmanr(y_true_encoded, y_pred_encoded)
        metrics["spearman_correlation"] = spearman_corr
        metrics["spearman_pvalue"] = spearman_p
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        # Per-class metrics
        per_class_metrics = {}
        for cls in self.label_encoder.classes_:
            if cls in report:
                per_class_metrics[cls] = {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1": report[cls]["f1-score"],
                    "support": report[cls]["support"]
                }
        
        # Investment grade vs speculative (binary)
        investment_grade = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", 
                          "BBB+", "BBB", "BBB-"}
        
        y_true_binary = y_true.apply(lambda x: 1 if x in investment_grade else 0)
        y_pred_binary = pd.Series(y_pred).apply(lambda x: 1 if x in investment_grade else 0)
        
        metrics["binary_accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
        metrics["binary_f1"] = f1_score(y_true_binary, y_pred_binary)
        metrics["binary_precision"] = precision_score(y_true_binary, y_pred_binary)
        metrics["binary_recall"] = recall_score(y_true_binary, y_pred_binary)
        
        # Rating bucket distribution
        true_dist = y_true.value_counts().to_dict()
        pred_dist = pd.Series(y_pred).value_counts().to_dict()
        
        results = {
            "dataset": dataset_name,
            "n_samples": len(y_true),
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "true_distribution": true_dist,
            "pred_distribution": pred_dist,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def analyze_errors(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        company_info: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Analyze prediction errors.
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            company_info: DataFrame with company identifiers
            
        Returns:
            DataFrame with error analysis
        """
        X_eval = X[self.feature_columns].fillna(X[self.feature_columns].median())
        X_scaled = self.scaler.transform(X_eval)
        
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Create error analysis DataFrame
        errors = []
        for i in range(len(y_true)):
            if y_pred[i] != y_true.iloc[i]:
                error_entry = {
                    "index": i,
                    "true_rating": y_true.iloc[i],
                    "predicted_rating": y_pred[i],
                    "confidence": float(np.max(y_pred_proba[i])),
                    "true_numeric": self.label_encoder.transform([y_true.iloc[i]])[0],
                    "pred_numeric": y_pred_encoded[i],
                    "rating_diff": abs(
                        self.label_encoder.transform([y_true.iloc[i]])[0] - y_pred_encoded[i]
                    )
                }
                
                if company_info is not None and i < len(company_info):
                    error_entry["ticker"] = company_info.iloc[i].get("ticker", "")
                    error_entry["company_name"] = company_info.iloc[i].get("company_name", "")
                    error_entry["fiscal_year"] = company_info.iloc[i].get("fiscal_year", "")
                
                errors.append(error_entry)
        
        error_df = pd.DataFrame(errors)
        
        # Sort by rating difference (worst errors first)
        if len(error_df) > 0:
            error_df = error_df.sort_values("rating_diff", ascending=False)
        
        return error_df
    
    def compute_feature_importance(self) -> Dict:
        """Compute and return feature importance."""
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance
        return {}
    
    def save_results(self, results: Dict, error_df: pd.DataFrame = None):
        """
        Save evaluation results.
        
        Args:
            results: Evaluation results dictionary
            error_df: Error analysis DataFrame
        """
        # Save main results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature importance
        importance = self.compute_feature_importance()
        importance_path = self.output_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        
        # Save error analysis
        if error_df is not None and len(error_df) > 0:
            error_path = self.output_dir / "error_analysis.csv"
            error_df.to_csv(error_path, index=False)
            
            error_json_path = self.output_dir / "error_analysis.json"
            error_df.to_json(error_json_path, orient="records", indent=2)
        
        # Generate summary report
        self._generate_report(results, importance, error_df)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_report(
        self,
        results: Dict,
        importance: Dict,
        error_df: pd.DataFrame = None
    ):
        """Generate markdown evaluation report."""
        report_lines = [
            "# Credit Rating Model Evaluation Report",
            "",
            f"**Evaluation Date**: {results.get('evaluation_timestamp', 'N/A')}",
            f"**Dataset**: {results.get('dataset', 'N/A')}",
            f"**Samples**: {results.get('n_samples', 'N/A')}",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        
        metrics = results.get("metrics", {})
        for metric, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"| {metric} | {value:.4f} |")
            else:
                report_lines.append(f"| {metric} | {value} |")
        
        report_lines.extend([
            "",
            "## Binary Classification (Investment Grade vs Speculative)",
            "",
            f"- Accuracy: {metrics.get('binary_accuracy', 'N/A'):.4f}",
            f"- F1 Score: {metrics.get('binary_f1', 'N/A'):.4f}",
            f"- Precision: {metrics.get('binary_precision', 'N/A'):.4f}",
            f"- Recall: {metrics.get('binary_recall', 'N/A'):.4f}",
            "",
            "## Per-Class Performance",
            "",
            "| Rating | Precision | Recall | F1 | Support |",
            "|--------|-----------|--------|-----|---------|"
        ])
        
        per_class = results.get("per_class_metrics", {})
        for cls, cls_metrics in per_class.items():
            report_lines.append(
                f"| {cls} | {cls_metrics['precision']:.3f} | "
                f"{cls_metrics['recall']:.3f} | {cls_metrics['f1']:.3f} | "
                f"{cls_metrics['support']} |"
            )
        
        report_lines.extend([
            "",
            "## Top 10 Feature Importance",
            "",
            "| Feature | Importance |",
            "|---------|------------|"
        ])
        
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            report_lines.append(f"| {feat} | {imp:.4f} |")
        
        if error_df is not None and len(error_df) > 0:
            report_lines.extend([
                "",
                "## Error Analysis",
                "",
                f"Total errors: {len(error_df)}",
                f"Average rating difference: {error_df['rating_diff'].mean():.2f}",
                "",
                "### Worst Errors (Top 10)",
                ""
            ])
            
            for _, row in error_df.head(10).iterrows():
                report_lines.append(
                    f"- {row.get('ticker', 'N/A')} ({row.get('fiscal_year', 'N/A')}): "
                    f"True={row['true_rating']}, Pred={row['predicted_rating']}, "
                    f"Conf={row['confidence']:.2f}"
                )
        
        # Save report
        report_path = self.output_dir / "evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate credit rating model")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features" / "features.parquet"),
        help="Path to features file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Path to model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "evaluation"),
        help="Output directory"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="tassnief_rating",
        help="Target column name"
    )
    
    args = parser.parse_args()
    
    # Load features
    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return
    
    if features_path.suffix == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_json(features_path)
    
    # Filter valid samples
    df_valid = df[df[args.target].notna()]
    
    if len(df_valid) == 0:
        logger.error("No valid samples with target values")
        return
    
    logger.info(f"Evaluating on {len(df_valid)} samples")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_dir=Path(args.model),
        output_dir=Path(args.output)
    )
    
    # Get features and target
    X = df_valid[evaluator.feature_columns]
    y = df_valid[args.target]
    company_info = df_valid[["ticker", "company_name", "fiscal_year"]]
    
    # Evaluate
    results = evaluator.evaluate(X, y, dataset_name="full")
    
    # Error analysis
    error_df = evaluator.analyze_errors(X, y, company_info)
    
    # Save results
    evaluator.save_results(results, error_df)
    
    # Print summary
    metrics = results.get("metrics", {})
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  F1 (macro): {metrics.get('f1_macro', 0):.4f}")
    logger.info(f"  Spearman: {metrics.get('spearman_correlation', 0):.4f}")
    logger.info(f"  Binary F1: {metrics.get('binary_f1', 0):.4f}")
    
    logger.success("Evaluation completed!")


if __name__ == "__main__":
    main()
