#!/usr/bin/env python3
"""
LLM Verdict Evaluation
Evaluates the quality of LLM-generated credit verdicts.
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "evaluation" / "verdict_evaluation_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class VerdictEvaluator:
    """Evaluates LLM-generated credit verdicts."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Output directory for evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_schema_compliance(self, verdict: Dict) -> Dict:
        """
        Check if verdict complies with expected schema.
        
        Args:
            verdict: Verdict dictionary
            
        Returns:
            Schema compliance metrics
        """
        required_fields = [
            "overall_assessment", "risk_level", "strengths",
            "weaknesses", "risk_factors", "prediction_analysis", "outlook"
        ]
        
        valid_risk_levels = ["LOW", "MODERATE", "ELEVATED", "HIGH", "UNKNOWN"]
        valid_outlooks = ["POSITIVE", "STABLE", "NEGATIVE", "UNKNOWN"]
        
        metrics = {
            "has_all_fields": True,
            "missing_fields": [],
            "valid_risk_level": False,
            "valid_outlook": False,
            "has_strengths": False,
            "has_weaknesses": False,
            "has_risk_factors": False
        }
        
        for field in required_fields:
            if field not in verdict or verdict[field] is None:
                metrics["has_all_fields"] = False
                metrics["missing_fields"].append(field)
        
        metrics["valid_risk_level"] = verdict.get("risk_level") in valid_risk_levels
        metrics["valid_outlook"] = verdict.get("outlook") in valid_outlooks
        metrics["has_strengths"] = len(verdict.get("strengths", [])) > 0
        metrics["has_weaknesses"] = len(verdict.get("weaknesses", [])) > 0
        metrics["has_risk_factors"] = len(verdict.get("risk_factors", [])) > 0
        
        # Overall compliance score
        checks = [
            metrics["has_all_fields"],
            metrics["valid_risk_level"],
            metrics["valid_outlook"],
            metrics["has_strengths"],
            metrics["has_weaknesses"],
            metrics["has_risk_factors"]
        ]
        metrics["compliance_score"] = sum(checks) / len(checks)
        
        return metrics
    
    def evaluate_factual_accuracy(
        self,
        verdict: Dict,
        features: Dict
    ) -> Dict:
        """
        Check if verdict cites accurate facts from the data.
        
        Args:
            verdict: Verdict dictionary
            features: Original feature dictionary
            
        Returns:
            Factual accuracy metrics
        """
        metrics = {
            "total_citations": 0,
            "verified_citations": 0,
            "citation_accuracy": 0.0,
            "cited_values": []
        }
        
        # Combine all text from verdict
        text_fields = ["overall_assessment", "prediction_analysis"]
        list_fields = ["strengths", "weaknesses", "risk_factors"]
        
        all_text = ""
        for field in text_fields:
            if field in verdict and verdict[field]:
                all_text += str(verdict[field]) + " "
        
        for field in list_fields:
            if field in verdict and isinstance(verdict[field], list):
                all_text += " ".join(str(item) for item in verdict[field]) + " "
        
        # Extract numeric citations from text
        # Look for patterns like "1.8", "2.1%", "15%", etc.
        number_pattern = r'(\d+\.?\d*)%?'
        cited_numbers = re.findall(number_pattern, all_text)
        
        # Check if cited numbers match features
        feature_values = set()
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Add various representations
                feature_values.add(f"{value:.1f}")
                feature_values.add(f"{value:.2f}")
                feature_values.add(f"{value:.0f}")
                if abs(value) < 1:  # Likely a ratio or percentage
                    feature_values.add(f"{value*100:.1f}")
                    feature_values.add(f"{value*100:.0f}")
        
        verified = 0
        for num in cited_numbers:
            metrics["total_citations"] += 1
            if num in feature_values or f"{float(num):.1f}" in feature_values:
                verified += 1
                metrics["cited_values"].append({"value": num, "verified": True})
            else:
                metrics["cited_values"].append({"value": num, "verified": False})
        
        metrics["verified_citations"] = verified
        if metrics["total_citations"] > 0:
            metrics["citation_accuracy"] = verified / metrics["total_citations"]
        
        return metrics
    
    def evaluate_coherence(self, verdict: Dict) -> Dict:
        """
        Evaluate internal coherence of the verdict.
        
        Args:
            verdict: Verdict dictionary
            
        Returns:
            Coherence metrics
        """
        metrics = {
            "risk_outlook_consistent": False,
            "strengths_weaknesses_balanced": False,
            "assessment_length_adequate": False,
            "prediction_analysis_present": False,
            "coherence_score": 0.0
        }
        
        # Check risk level vs outlook consistency
        risk_level = verdict.get("risk_level", "")
        outlook = verdict.get("outlook", "")
        
        # Generally: LOW risk -> POSITIVE/STABLE, HIGH risk -> NEGATIVE/STABLE
        consistent_pairs = [
            ("LOW", "POSITIVE"), ("LOW", "STABLE"),
            ("MODERATE", "STABLE"), ("MODERATE", "POSITIVE"), ("MODERATE", "NEGATIVE"),
            ("ELEVATED", "STABLE"), ("ELEVATED", "NEGATIVE"),
            ("HIGH", "NEGATIVE"), ("HIGH", "STABLE")
        ]
        metrics["risk_outlook_consistent"] = (risk_level, outlook) in consistent_pairs
        
        # Check balance of strengths vs weaknesses
        n_strengths = len(verdict.get("strengths", []))
        n_weaknesses = len(verdict.get("weaknesses", []))
        if n_strengths > 0 and n_weaknesses > 0:
            ratio = min(n_strengths, n_weaknesses) / max(n_strengths, n_weaknesses)
            metrics["strengths_weaknesses_balanced"] = ratio > 0.3
        
        # Check assessment length
        assessment = verdict.get("overall_assessment", "")
        metrics["assessment_length_adequate"] = len(assessment) >= 50
        
        # Check prediction analysis
        pred_analysis = verdict.get("prediction_analysis", "")
        metrics["prediction_analysis_present"] = len(pred_analysis) >= 20
        
        # Overall coherence score
        checks = [
            metrics["risk_outlook_consistent"],
            metrics["strengths_weaknesses_balanced"],
            metrics["assessment_length_adequate"],
            metrics["prediction_analysis_present"]
        ]
        metrics["coherence_score"] = sum(checks) / len(checks)
        
        return metrics
    
    def evaluate_prediction_alignment(
        self,
        verdict: Dict,
        actual_rating: Optional[str] = None
    ) -> Dict:
        """
        Evaluate alignment between verdict and prediction/actual rating.
        
        Args:
            verdict: Verdict dictionary
            actual_rating: Actual Tassnief rating
            
        Returns:
            Alignment metrics
        """
        metrics = {
            "verdict_supports_prediction": False,
            "verdict_matches_actual": False,
            "risk_level_appropriate": False
        }
        
        predicted_rating = verdict.get("predicted_rating", "")
        risk_level = verdict.get("risk_level", "")
        
        # Map ratings to risk levels
        high_quality_ratings = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-"}
        medium_quality_ratings = {"BBB+", "BBB", "BBB-"}
        low_quality_ratings = {"BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"}
        
        # Expected risk level based on rating
        if predicted_rating in high_quality_ratings:
            expected_risk = ["LOW", "MODERATE"]
        elif predicted_rating in medium_quality_ratings:
            expected_risk = ["MODERATE", "ELEVATED"]
        elif predicted_rating in low_quality_ratings:
            expected_risk = ["ELEVATED", "HIGH"]
        else:
            expected_risk = ["MODERATE"]
        
        metrics["risk_level_appropriate"] = risk_level in expected_risk
        
        # Check prediction analysis sentiment
        pred_analysis = verdict.get("prediction_analysis", "").lower()
        positive_words = ["supports", "aligns", "consistent", "agrees", "confirms"]
        negative_words = ["contradicts", "inconsistent", "disagrees", "does not support"]
        
        has_positive = any(word in pred_analysis for word in positive_words)
        has_negative = any(word in pred_analysis for word in negative_words)
        
        # If analysis has positive language, verdict supports prediction
        metrics["verdict_supports_prediction"] = has_positive and not has_negative
        
        # Check actual rating if provided
        if actual_rating:
            if actual_rating in high_quality_ratings and risk_level in ["LOW", "MODERATE"]:
                metrics["verdict_matches_actual"] = True
            elif actual_rating in medium_quality_ratings and risk_level in ["MODERATE", "ELEVATED"]:
                metrics["verdict_matches_actual"] = True
            elif actual_rating in low_quality_ratings and risk_level in ["ELEVATED", "HIGH"]:
                metrics["verdict_matches_actual"] = True
        
        return metrics
    
    def evaluate_single_verdict(
        self,
        verdict: Dict,
        features: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate a single verdict.
        
        Args:
            verdict: Verdict dictionary
            features: Original features (for factual accuracy check)
            
        Returns:
            Complete evaluation metrics
        """
        evaluation = {
            "ticker": verdict.get("ticker"),
            "fiscal_year": verdict.get("fiscal_year"),
            "predicted_rating": verdict.get("predicted_rating"),
            "actual_rating": verdict.get("actual_rating")
        }
        
        # Schema compliance
        schema_metrics = self.evaluate_schema_compliance(verdict)
        evaluation["schema"] = schema_metrics
        
        # Coherence
        coherence_metrics = self.evaluate_coherence(verdict)
        evaluation["coherence"] = coherence_metrics
        
        # Prediction alignment
        alignment_metrics = self.evaluate_prediction_alignment(
            verdict,
            verdict.get("actual_rating")
        )
        evaluation["alignment"] = alignment_metrics
        
        # Factual accuracy (if features provided)
        if features:
            factual_metrics = self.evaluate_factual_accuracy(verdict, features)
            evaluation["factual"] = factual_metrics
        
        # Overall score
        scores = [
            schema_metrics.get("compliance_score", 0),
            coherence_metrics.get("coherence_score", 0)
        ]
        if features:
            scores.append(factual_metrics.get("citation_accuracy", 0))
        
        evaluation["overall_score"] = np.mean(scores)
        
        return evaluation
    
    def evaluate_all(
        self,
        verdicts: List[Dict],
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Evaluate all verdicts.
        
        Args:
            verdicts: List of verdict dictionaries
            features_df: DataFrame with original features
            
        Returns:
            Aggregate evaluation results
        """
        evaluations = []
        
        for verdict in verdicts:
            # Match features if available
            features = None
            if features_df is not None:
                ticker = verdict.get("ticker")
                year = verdict.get("fiscal_year")
                match = features_df[
                    (features_df["ticker"] == ticker) & 
                    (features_df["fiscal_year"] == year)
                ]
                if len(match) > 0:
                    features = match.iloc[0].to_dict()
            
            eval_result = self.evaluate_single_verdict(verdict, features)
            evaluations.append(eval_result)
        
        # Aggregate metrics
        aggregate = {
            "total_verdicts": len(verdicts),
            "avg_overall_score": np.mean([e["overall_score"] for e in evaluations]),
            "avg_schema_compliance": np.mean([e["schema"]["compliance_score"] for e in evaluations]),
            "avg_coherence_score": np.mean([e["coherence"]["coherence_score"] for e in evaluations]),
            "schema_fully_compliant": sum(1 for e in evaluations if e["schema"]["has_all_fields"]),
            "valid_risk_levels": sum(1 for e in evaluations if e["schema"]["valid_risk_level"]),
            "valid_outlooks": sum(1 for e in evaluations if e["schema"]["valid_outlook"]),
            "risk_appropriate": sum(1 for e in evaluations if e["alignment"]["risk_level_appropriate"]),
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        if features_df is not None:
            factual_scores = [e.get("factual", {}).get("citation_accuracy", 0) for e in evaluations]
            aggregate["avg_citation_accuracy"] = np.mean(factual_scores)
        
        return {
            "aggregate": aggregate,
            "individual": evaluations
        }
    
    def save_results(self, results: Dict):
        """Save evaluation results."""
        # Save full results
        results_path = self.output_dir / "verdict_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save aggregate summary
        summary_path = self.output_dir / "verdict_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results["aggregate"], f, indent=2)
        
        # Generate report
        self._generate_report(results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_report(self, results: Dict):
        """Generate markdown evaluation report."""
        agg = results["aggregate"]
        
        report_lines = [
            "# LLM Verdict Evaluation Report",
            "",
            f"**Evaluation Date**: {agg.get('evaluation_timestamp', 'N/A')}",
            f"**Total Verdicts**: {agg.get('total_verdicts', 0)}",
            "",
            "## Overall Quality",
            "",
            f"- **Average Overall Score**: {agg.get('avg_overall_score', 0):.3f}",
            f"- **Average Schema Compliance**: {agg.get('avg_schema_compliance', 0):.3f}",
            f"- **Average Coherence Score**: {agg.get('avg_coherence_score', 0):.3f}",
            "",
            "## Schema Compliance",
            "",
            f"- Fully Compliant: {agg.get('schema_fully_compliant', 0)} / {agg.get('total_verdicts', 0)}",
            f"- Valid Risk Levels: {agg.get('valid_risk_levels', 0)} / {agg.get('total_verdicts', 0)}",
            f"- Valid Outlooks: {agg.get('valid_outlooks', 0)} / {agg.get('total_verdicts', 0)}",
            "",
            "## Alignment",
            "",
            f"- Risk Level Appropriate: {agg.get('risk_appropriate', 0)} / {agg.get('total_verdicts', 0)}",
            ""
        ]
        
        if "avg_citation_accuracy" in agg:
            report_lines.extend([
                "## Factual Accuracy",
                "",
                f"- Average Citation Accuracy: {agg.get('avg_citation_accuracy', 0):.3f}",
                ""
            ])
        
        # Save report
        report_path = self.output_dir / "verdict_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))


def main():
    """Main entry point for verdict evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate LLM verdicts")
    parser.add_argument(
        "--verdicts",
        type=str,
        default=str(PROJECT_ROOT / "results" / "verdicts" / "all_verdicts.json"),
        help="Path to verdicts file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features" / "features.parquet"),
        help="Path to features file (for factual accuracy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "evaluation"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load verdicts
    verdicts_path = Path(args.verdicts)
    if not verdicts_path.exists():
        logger.error(f"Verdicts file not found: {verdicts_path}")
        return
    
    with open(verdicts_path, 'r') as f:
        verdicts = json.load(f)
    
    logger.info(f"Loaded {len(verdicts)} verdicts")
    
    # Load features (optional)
    features_df = None
    features_path = Path(args.features)
    if features_path.exists():
        if features_path.suffix == ".parquet":
            features_df = pd.read_parquet(features_path)
        else:
            features_df = pd.read_json(features_path)
        logger.info(f"Loaded features for factual accuracy check")
    
    # Initialize evaluator
    evaluator = VerdictEvaluator(output_dir=Path(args.output))
    
    # Evaluate
    results = evaluator.evaluate_all(verdicts, features_df)
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    agg = results["aggregate"]
    logger.info("Verdict Evaluation Results:")
    logger.info(f"  Overall Score: {agg.get('avg_overall_score', 0):.3f}")
    logger.info(f"  Schema Compliance: {agg.get('avg_schema_compliance', 0):.3f}")
    logger.info(f"  Coherence: {agg.get('avg_coherence_score', 0):.3f}")
    
    logger.success("Verdict evaluation completed!")


if __name__ == "__main__":
    main()
