#!/usr/bin/env python3
"""
Tassnief Credit Ratings Collection Pipeline
Collects credit ratings from tassnief.com for Saudi companies.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "ratings_collection_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


# Tassnief rating scale
RATING_SCALE = [
    "AAA", "AA+", "AA", "AA-", 
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC+", "CCC", "CCC-",
    "CC", "C", "D"
]

RATING_TO_NUMERIC = {rating: len(RATING_SCALE) - i for i, rating in enumerate(RATING_SCALE)}

INVESTMENT_GRADE_RATINGS = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"}


def rating_to_bucket(rating: str) -> str:
    """Convert rating to investment_grade or speculative bucket."""
    return "investment_grade" if rating in INVESTMENT_GRADE_RATINGS else "speculative"


class TassnifRatingCollector:
    """Collects credit ratings from Tassnief."""
    
    def __init__(
        self,
        output_dir: Path,
        rate_limit_delay: float = 2.0
    ):
        """
        Initialize rating collector.
        
        Args:
            output_dir: Directory to save ratings data
            rate_limit_delay: Delay between requests (seconds)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://tassnief.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse_rating_from_inventory(self, rating_entry: Dict) -> Dict:
        """
        Process a rating entry from the inventory.
        
        Args:
            rating_entry: Rating entry from data_inventory.json
            
        Returns:
            Processed rating dictionary
        """
        rating = rating_entry.get("rating", "")
        
        return {
            "ticker": rating_entry.get("ticker"),
            "company_name": rating_entry.get("company_name"),
            "rating_date": rating_entry.get("rating_date"),
            "rating": rating,
            "rating_numeric": RATING_TO_NUMERIC.get(rating, 0),
            "rating_bucket": rating_to_bucket(rating),
            "outlook": rating_entry.get("outlook"),
            "rating_type": rating_entry.get("rating_type", "issuer"),
            "source_url": rating_entry.get("source_url"),
            "fiscal_year_applicable": rating_entry.get("fiscal_year_applicable"),
            "processed_timestamp": datetime.now().isoformat()
        }
    
    def fetch_rating_page(self, url: str) -> Optional[str]:
        """
        Fetch a rating page from Tassnief website.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def scrape_company_ratings(self, url: str) -> List[Dict]:
        """
        Scrape ratings from a Tassnief company page.
        
        NOTE: This is a placeholder. Actual implementation depends on 
        Tassnief's website structure. You may need to:
        1. Manually collect ratings if scraping is not feasible
        2. Use PDF parsing if ratings are in PDF reports
        3. Contact Tassnief for data access
        
        Args:
            url: Company rating page URL
            
        Returns:
            List of rating dictionaries
        """
        html = self.fetch_rating_page(url)
        if not html:
            return []
        
        # Placeholder parsing - implement based on actual website structure
        soup = BeautifulSoup(html, 'html.parser')
        
        # TODO: Implement actual parsing based on Tassnief website structure
        # This will depend on how Tassnief presents rating information
        
        logger.warning(f"Scraping not fully implemented for {url}")
        return []
    
    def process_inventory_ratings(self, ratings: List[Dict]) -> List[Dict]:
        """
        Process ratings from inventory file.
        
        Args:
            ratings: List of rating entries from inventory
            
        Returns:
            List of processed rating dictionaries
        """
        processed = []
        for entry in ratings:
            processed_rating = self.parse_rating_from_inventory(entry)
            processed.append(processed_rating)
        return processed
    
    def save_ratings(self, ratings: List[Dict]) -> Path:
        """Save ratings to JSON file."""
        output_path = self.output_dir / "tassnief_ratings.json"
        
        data = {
            "metadata": {
                "source": "tassnief.com",
                "collection_date": datetime.now().isoformat(),
                "total_ratings": len(ratings)
            },
            "ratings": ratings
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(ratings)} ratings to {output_path}")
        return output_path
    
    def create_rating_matrix(self, ratings: List[Dict]) -> Dict:
        """
        Create a company-year rating matrix.
        
        Args:
            ratings: List of rating dictionaries
            
        Returns:
            Dictionary mapping (ticker, year) to rating info
        """
        matrix = {}
        for rating in ratings:
            ticker = rating.get("ticker")
            year = rating.get("fiscal_year_applicable")
            if ticker and year:
                key = f"{ticker}_{year}"
                matrix[key] = {
                    "rating": rating.get("rating"),
                    "rating_numeric": rating.get("rating_numeric"),
                    "rating_bucket": rating.get("rating_bucket"),
                    "outlook": rating.get("outlook")
                }
        
        # Save matrix
        matrix_path = self.output_dir / "rating_matrix.json"
        with open(matrix_path, 'w') as f:
            json.dump(matrix, f, indent=2)
        
        return matrix


def load_inventory(inventory_path: Path) -> Dict:
    """Load data inventory."""
    with open(inventory_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main entry point for ratings collection."""
    parser = argparse.ArgumentParser(description="Collect Tassnief credit ratings")
    parser.add_argument(
        "--inventory",
        type=str,
        default=str(PROJECT_ROOT / "data" / "data_inventory.json"),
        help="Path to data inventory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ratings"),
        help="Output directory"
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Attempt to scrape ratings from website (not fully implemented)"
    )
    
    args = parser.parse_args()
    
    # Load inventory
    inventory = load_inventory(Path(args.inventory))
    ratings_entries = inventory.get("ratings", [])
    
    if not ratings_entries:
        logger.warning("No ratings found in inventory")
        logger.info("Please populate the 'ratings' section in data_inventory.json")
        logger.info("Format: {ticker, company_name, rating_date, rating, outlook, ...}")
        return
    
    # Initialize collector
    collector = TassnifRatingCollector(output_dir=Path(args.output_dir))
    
    # Process ratings from inventory
    processed_ratings = collector.process_inventory_ratings(ratings_entries)
    
    # Save ratings
    collector.save_ratings(processed_ratings)
    
    # Create rating matrix
    matrix = collector.create_rating_matrix(processed_ratings)
    
    logger.success(f"Ratings collection completed. Processed {len(processed_ratings)} ratings.")
    
    # Print summary
    buckets = {}
    for r in processed_ratings:
        bucket = r.get("rating_bucket", "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    
    logger.info("Rating distribution:")
    for bucket, count in buckets.items():
        logger.info(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
