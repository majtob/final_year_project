#!/usr/bin/env python3
"""
Key Audit Matters (KAM) Extraction Pipeline
Extracts KAMs from company annual reports and audit reports.
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "kam_extraction_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


# KAM category keywords
KAM_CATEGORIES = {
    "going_concern": [
        "going concern", "continuity", "ability to continue",
        "material uncertainty", "liquidation"
    ],
    "impairment": [
        "impairment", "goodwill", "intangible assets",
        "recoverable amount", "write-down", "write off"
    ],
    "revenue_recognition": [
        "revenue recognition", "revenue from contracts",
        "percentage of completion", "contract revenue",
        "timing of revenue", "performance obligation"
    ],
    "related_party": [
        "related party", "related parties", "intercompany",
        "transactions with affiliates", "connected parties"
    ],
    "litigation": [
        "litigation", "legal proceedings", "contingent liabilities",
        "provision for claims", "lawsuit", "legal claims"
    ],
    "valuation": [
        "fair value", "valuation", "investment property",
        "financial instruments", "level 3", "unobservable inputs"
    ],
    "inventory": [
        "inventory valuation", "net realizable value",
        "obsolete inventory", "inventory provision"
    ],
    "allowance": [
        "allowance for", "expected credit loss", "ECL",
        "provision for doubtful", "bad debt"
    ]
}


class KAMExtractor:
    """Extracts Key Audit Matters from audit reports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize KAM extractor.
        
        Args:
            output_dir: Directory to save extracted KAMs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kam_output_dir = self.output_dir / "kams"
        self.kam_output_dir.mkdir(parents=True, exist_ok=True)
    
    def identify_kam_section(self, text: str) -> Optional[str]:
        """
        Identify and extract the KAM section from audit report text.
        
        Args:
            text: Full text of audit report
            
        Returns:
            KAM section text or None
        """
        # Common KAM section headers
        kam_headers = [
            r"key\s+audit\s+matter[s]?",
            r"KAM[s]?",
            r"matters\s+that\s+required\s+significant\s+auditor\s+attention"
        ]
        
        # Find KAM section start
        kam_start = None
        for header in kam_headers:
            match = re.search(header, text, re.IGNORECASE)
            if match:
                kam_start = match.start()
                break
        
        if kam_start is None:
            return None
        
        # Find section end (next major section)
        end_markers = [
            r"other\s+information",
            r"responsibilities\s+of\s+management",
            r"auditor['\"]?s?\s+responsibilities",
            r"report\s+on\s+other\s+legal"
        ]
        
        kam_end = len(text)
        for marker in end_markers:
            match = re.search(marker, text[kam_start:], re.IGNORECASE)
            if match:
                potential_end = kam_start + match.start()
                if potential_end < kam_end:
                    kam_end = potential_end
        
        return text[kam_start:kam_end]
    
    def extract_individual_kams(self, kam_section: str) -> List[Dict]:
        """
        Extract individual KAMs from KAM section.
        
        Args:
            kam_section: Text of KAM section
            
        Returns:
            List of KAM dictionaries
        """
        kams = []
        
        # Split by numbered items or common separators
        # This is a simplified approach - actual splitting may need refinement
        patterns = [
            r'\n\s*\d+\.\s+',  # Numbered list
            r'\n\s*[a-z]\)\s+',  # Letter list
            r'\n\s*•\s+',  # Bullet points
            r'\n\s*[-–—]\s+'  # Dashes
        ]
        
        # Try to find individual KAMs
        paragraphs = re.split(r'\n\s*\n', kam_section)
        
        current_kam = None
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip short paragraphs (likely headers)
                if current_kam and para:
                    current_kam["title"] = para
                continue
            
            # Check if this looks like a KAM description
            if any(keyword in para.lower() for keywords in KAM_CATEGORIES.values() for keyword in keywords):
                if current_kam:
                    kams.append(current_kam)
                
                current_kam = {
                    "title": "",
                    "text": para,
                    "category": self.categorize_kam(para)
                }
            elif current_kam:
                current_kam["text"] += "\n" + para
        
        if current_kam:
            kams.append(current_kam)
        
        return kams
    
    def categorize_kam(self, kam_text: str) -> str:
        """
        Categorize a KAM based on keywords.
        
        Args:
            kam_text: Text of the KAM
            
        Returns:
            Category string
        """
        text_lower = kam_text.lower()
        
        scores = {}
        for category, keywords in KAM_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "other"
    
    def extract_from_text(
        self,
        text: str,
        ticker: str,
        fiscal_year: int,
        source_file: str
    ) -> Dict:
        """
        Extract KAMs from audit report text.
        
        Args:
            text: Full text of audit report
            ticker: Company ticker
            fiscal_year: Fiscal year
            source_file: Source file path
            
        Returns:
            Dictionary with extraction results
        """
        kam_section = self.identify_kam_section(text)
        
        if not kam_section:
            logger.warning(f"No KAM section found for {ticker} FY{fiscal_year}")
            return {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "source_file": source_file,
                "kam_section_found": False,
                "kams": [],
                "kam_count": 0
            }
        
        kams = self.extract_individual_kams(kam_section)
        
        # Add metadata to each KAM
        for i, kam in enumerate(kams):
            kam["ticker"] = ticker
            kam["fiscal_year"] = fiscal_year
            kam["source_file"] = source_file
            kam["kam_index"] = i + 1
        
        result = {
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "source_file": source_file,
            "kam_section_found": True,
            "kams": kams,
            "kam_count": len(kams),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Extracted {len(kams)} KAMs for {ticker} FY{fiscal_year}")
        return result
    
    def compute_kam_features(self, kam_result: Dict) -> Dict:
        """
        Compute KAM features for ML model.
        
        Args:
            kam_result: KAM extraction result
            
        Returns:
            Dictionary of KAM features
        """
        kams = kam_result.get("kams", [])
        
        features = {
            "ticker": kam_result.get("ticker"),
            "fiscal_year": kam_result.get("fiscal_year"),
            "kam_count": len(kams),
            "kam_going_concern": 0,
            "kam_impairment": 0,
            "kam_revenue_recognition": 0,
            "kam_related_party": 0,
            "kam_litigation": 0,
            "kam_valuation": 0,
            "kam_inventory": 0,
            "kam_allowance": 0,
            "kam_other": 0
        }
        
        # Count KAMs by category
        for kam in kams:
            category = kam.get("category", "other")
            feature_key = f"kam_{category}"
            if feature_key in features:
                features[feature_key] += 1
            else:
                features["kam_other"] += 1
        
        # Compute severity score (0-1)
        # Higher weight for going concern and litigation
        severity_weights = {
            "going_concern": 1.0,
            "litigation": 0.8,
            "impairment": 0.6,
            "valuation": 0.5,
            "revenue_recognition": 0.4,
            "allowance": 0.4,
            "related_party": 0.3,
            "inventory": 0.3,
            "other": 0.2
        }
        
        if kams:
            severity_sum = sum(
                severity_weights.get(kam.get("category", "other"), 0.2)
                for kam in kams
            )
            features["kam_severity_score"] = min(severity_sum / 3.0, 1.0)  # Normalize
        else:
            features["kam_severity_score"] = 0.0
        
        return features
    
    def save_kam_results(self, results: List[Dict]) -> Tuple[Path, Path]:
        """
        Save KAM extraction results and features.
        
        Args:
            results: List of KAM extraction results
            
        Returns:
            Tuple of (results_path, features_path)
        """
        # Save detailed results
        results_path = self.kam_output_dir / "kam_extractions.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Compute and save features
        features = [self.compute_kam_features(r) for r in results]
        features_path = self.kam_output_dir / "kam_features.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2)
        
        logger.info(f"Saved KAM results to {results_path}")
        logger.info(f"Saved KAM features to {features_path}")
        
        return results_path, features_path


def main():
    """Main entry point for KAM extraction."""
    parser = argparse.ArgumentParser(description="Extract Key Audit Matters from reports")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "interim"),
        help="Directory containing parsed report text"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "interim"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    extractor = KAMExtractor(output_dir=Path(args.output_dir))
    
    # TODO: Implement file discovery and processing
    # This should:
    # 1. Find all parsed audit report text files in input_dir
    # 2. Extract KAMs from each
    # 3. Save results
    
    logger.info("KAM extraction pipeline initialized")
    logger.info("To use: Parse audit reports first, then run extraction on the text files")
    logger.warning("Full implementation requires parsed text files from audit reports")


if __name__ == "__main__":
    main()
