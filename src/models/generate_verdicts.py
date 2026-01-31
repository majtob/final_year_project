#!/usr/bin/env python3
"""
LLM Verdict Generation Pipeline
Generates credit risk verdicts using open-source LLMs (Mistral, Llama).
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
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.prompts import (
    VERDICT_SYSTEM_PROMPT,
    build_verdict_prompt,
    parse_verdict_response,
    validate_verdict
)

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "verdict_generation_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class VerdictGenerator:
    """Generates credit risk verdicts using LLMs."""
    
    def __init__(
        self,
        model_name: str = "mistral",
        backend: str = "ollama",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize verdict generator.
        
        Args:
            model_name: LLM model name (mistral, llama3, etc.)
            backend: Backend to use (ollama, huggingface, transformers)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the LLM backend."""
        if self.backend == "ollama":
            self._init_ollama()
        elif self.backend == "huggingface":
            self._init_huggingface()
        elif self.backend == "transformers":
            self._init_transformers()
        else:
            logger.warning(f"Unknown backend: {self.backend}")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama
            logger.info(f"Initialized Ollama with model: {self.model_name}")
        except ImportError:
            logger.error("Ollama not installed. Install with: pip install ollama")
            logger.info("Make sure Ollama is running: ollama serve")
    
    def _init_huggingface(self):
        """Initialize HuggingFace Inference API."""
        try:
            from huggingface_hub import InferenceClient
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if api_key:
                self.client = InferenceClient(token=api_key)
                logger.info("Initialized HuggingFace Inference API")
            else:
                logger.warning("HUGGINGFACE_API_KEY not set")
        except ImportError:
            logger.error("huggingface_hub not installed")
    
    def _init_transformers(self):
        """Initialize local transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto"
            )
            logger.info("Model loaded successfully")
        except ImportError:
            logger.error("transformers not installed")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def generate(self, prompt: str, system_prompt: str = VERDICT_SYSTEM_PROMPT) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Generated text
        """
        if self.backend == "ollama" and self.client:
            return self._generate_ollama(prompt, system_prompt)
        elif self.backend == "huggingface" and self.client:
            return self._generate_huggingface(prompt, system_prompt)
        elif self.backend == "transformers" and hasattr(self, 'model'):
            return self._generate_transformers(prompt, system_prompt)
        else:
            logger.error("No valid backend initialized")
            return ""
    
    def _generate_ollama(self, prompt: str, system_prompt: str) -> str:
        """Generate using Ollama."""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""
    
    def _generate_huggingface(self, prompt: str, system_prompt: str) -> str:
        """Generate using HuggingFace API."""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self.client.text_generation(
                full_prompt,
                model=self.model_name,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            return ""
    
    def _generate_transformers(self, prompt: str, system_prompt: str) -> str:
        """Generate using local transformers model."""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response = response[len(full_prompt):].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return ""
    
    def generate_verdict(
        self,
        features: Dict,
        predicted_rating: str,
        prediction_confidence: float,
        actual_rating: Optional[str] = None
    ) -> Dict:
        """
        Generate a credit verdict for a company.
        
        Args:
            features: Feature dictionary for the company
            predicted_rating: ML model's predicted rating
            prediction_confidence: Confidence score
            actual_rating: Actual Tassnief rating (if known)
            
        Returns:
            Verdict dictionary
        """
        # Split features by category
        company_data = {
            "company_name": features.get("company_name"),
            "ticker": features.get("ticker"),
            "fiscal_year": features.get("fiscal_year")
        }
        
        financial_ratios = {k: features.get(k) for k in [
            "current_ratio", "quick_ratio", "cash_ratio",
            "debt_to_equity", "debt_to_assets",
            "interest_coverage", "return_on_assets", "return_on_equity",
            "net_margin", "operating_margin", "revenue_growth", "net_income_growth"
        ]}
        
        kam_features = {k: features.get(k) for k in [
            "kam_count", "kam_going_concern", "kam_impairment",
            "kam_revenue_recognition", "kam_related_party",
            "kam_litigation", "kam_valuation", "kam_severity_score"
        ]}
        
        news_features = {k: features.get(k) for k in [
            "news_count", "avg_sentiment", "negative_news_ratio",
            "event_lawsuit", "event_expansion", "event_regulatory"
        ]}
        
        # Build prompt
        prompt = build_verdict_prompt(
            company_data=company_data,
            financial_ratios=financial_ratios,
            kam_features=kam_features,
            news_features=news_features,
            predicted_rating=predicted_rating,
            prediction_confidence=prediction_confidence,
            actual_rating=actual_rating
        )
        
        # Generate response
        response = self.generate(prompt)
        
        # Parse response
        verdict = parse_verdict_response(response)
        
        # Add metadata
        verdict["ticker"] = company_data["ticker"]
        verdict["company_name"] = company_data["company_name"]
        verdict["fiscal_year"] = company_data["fiscal_year"]
        verdict["predicted_rating"] = predicted_rating
        verdict["prediction_confidence"] = prediction_confidence
        verdict["actual_rating"] = actual_rating
        verdict["model_name"] = self.model_name
        verdict["generation_timestamp"] = datetime.now().isoformat()
        verdict["is_valid"] = validate_verdict(verdict)
        
        return verdict


class VerdictPipeline:
    """Pipeline for generating verdicts for multiple companies."""
    
    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        llm_model: str = "mistral",
        llm_backend: str = "ollama"
    ):
        """
        Initialize verdict pipeline.
        
        Args:
            model_dir: Directory with trained ML model
            output_dir: Output directory for verdicts
            llm_model: LLM model name
            llm_backend: LLM backend
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ML model
        self.classifier = self._load_classifier()
        
        # Initialize verdict generator
        self.generator = VerdictGenerator(
            model_name=llm_model,
            backend=llm_backend
        )
    
    def _load_classifier(self):
        """Load the trained classifier."""
        from src.models.train_classifier import RatingClassifier
        classifier = RatingClassifier()
        classifier.load(self.model_dir)
        return classifier
    
    def process_company(self, features: Dict) -> Dict:
        """
        Process a single company.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Verdict dictionary
        """
        # Get ML prediction
        df = pd.DataFrame([features])
        predictions, probabilities = self.classifier.predict(df)
        
        predicted_rating = predictions[0]
        prediction_confidence = float(np.max(probabilities[0]))
        
        # Get actual rating if available
        actual_rating = features.get("tassnief_rating")
        
        # Generate verdict
        verdict = self.generator.generate_verdict(
            features=features,
            predicted_rating=predicted_rating,
            prediction_confidence=prediction_confidence,
            actual_rating=actual_rating
        )
        
        return verdict
    
    def process_all(self, features_df: pd.DataFrame) -> List[Dict]:
        """
        Process all companies.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            List of verdict dictionaries
        """
        verdicts = []
        
        for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Generating verdicts"):
            features = row.to_dict()
            verdict = self.process_company(features)
            verdicts.append(verdict)
            
            # Save individual verdict
            ticker = features.get("ticker", "unknown").replace(".", "_")
            year = features.get("fiscal_year", "unknown")
            verdict_path = self.output_dir / f"{ticker}_{year}_verdict.json"
            with open(verdict_path, 'w', encoding='utf-8') as f:
                json.dump(verdict, f, indent=2, ensure_ascii=False)
        
        # Save all verdicts
        all_verdicts_path = self.output_dir / "all_verdicts.json"
        with open(all_verdicts_path, 'w', encoding='utf-8') as f:
            json.dump(verdicts, f, indent=2, ensure_ascii=False)
        
        # Save summary
        valid_count = sum(1 for v in verdicts if v.get("is_valid", False))
        summary = {
            "total_verdicts": len(verdicts),
            "valid_verdicts": valid_count,
            "invalid_verdicts": len(verdicts) - valid_count,
            "model_used": self.generator.model_name,
            "generation_date": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated {len(verdicts)} verdicts ({valid_count} valid)")
        return verdicts


def main():
    """Main entry point for verdict generation."""
    parser = argparse.ArgumentParser(description="Generate credit risk verdicts")
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
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "verdicts"),
        help="Output directory"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="mistral",
        help="LLM model name"
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["ollama", "huggingface", "transformers"],
        default="ollama",
        help="LLM backend"
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
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Initialize pipeline
    pipeline = VerdictPipeline(
        model_dir=Path(args.model),
        output_dir=Path(args.output),
        llm_model=args.llm_model,
        llm_backend=args.llm_backend
    )
    
    # Generate verdicts
    verdicts = pipeline.process_all(df)
    
    logger.success("Verdict generation completed!")


if __name__ == "__main__":
    main()
