#!/usr/bin/env python3
"""
Tassnief Credit Ratings Scraper
Scrapes credit ratings from https://tassnief.com/find-rating

Usage:
    python scripts/scrape_tassnief.py --output data/templates/ratings_scraped.csv
"""

import os
import sys
import argparse
import re
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "tassnief_scraper_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class TassnifScraper:
    """Scrapes credit ratings from Tassnief website."""
    
    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize scraper.
        
        Args:
            rate_limit: Delay between requests in seconds
        """
        self.base_url = "https://tassnief.com/find-rating"
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def parse_rating(self, rating_text: str) -> Dict:
        """
        Parse rating text to extract rating, pi status, and solicited status.
        
        Args:
            rating_text: Raw rating text like "BBB- (pi) - Unsolicited" or "A+"
            
        Returns:
            Dictionary with parsed rating info
        """
        rating_text = rating_text.strip()
        
        # Check for (pi) indicator
        is_pi = "(pi)" in rating_text.lower()
        
        # Check for unsolicited
        is_unsolicited = "unsolicited" in rating_text.lower()
        
        # Extract the base rating (e.g., "BBB-", "A+", "AA-")
        # Remove (pi), unsolicited, and extra characters
        base_rating = rating_text
        base_rating = re.sub(r'\(pi\)', '', base_rating, flags=re.IGNORECASE)
        base_rating = re.sub(r'-?\s*unsolicited', '', base_rating, flags=re.IGNORECASE)
        base_rating = re.sub(r'\s+', '', base_rating)  # Remove whitespace
        base_rating = base_rating.strip(' -')
        
        return {
            "rating": base_rating,
            "is_solicited": not is_unsolicited,
            "rating_basis": "pi" if is_pi else "full"
        }
    
    def parse_date(self, date_text: str) -> str:
        """
        Parse date text to YYYY-MM-DD format.
        
        Args:
            date_text: Date like "Jan,2026" or "Dec,2025"
            
        Returns:
            Date in YYYY-MM-DD format
        """
        date_text = date_text.strip()
        
        # Handle "Mon,YYYY" format
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        try:
            # Try to parse "Mon,YYYY" format
            match = re.match(r'(\w+),?\s*(\d{4})', date_text, re.IGNORECASE)
            if match:
                month_str = match.group(1).lower()[:3]
                year = match.group(2)
                month = month_map.get(month_str, '01')
                return f"{year}-{month}-15"  # Use 15th as default day
        except:
            pass
        
        return date_text
    
    def extract_fiscal_year(self, date_str: str) -> int:
        """Extract fiscal year from rating date."""
        try:
            year = int(date_str[:4])
            # Ratings issued in early months often apply to previous fiscal year
            month = int(date_str[5:7])
            if month <= 3:  # Jan-Mar ratings often for previous year
                return year - 1
            return year
        except:
            return datetime.now().year
    
    def scrape_page(self, page_num: int = 1) -> List[Dict]:
        """
        Scrape a single page of ratings.
        
        Args:
            page_num: Page number to scrape
            
        Returns:
            List of rating dictionaries
        """
        # The URL might need pagination parameter
        # Try different URL patterns
        urls_to_try = [
            f"{self.base_url}?page={page_num}",
            f"{self.base_url}/page/{page_num}",
            self.base_url if page_num == 1 else f"{self.base_url}?paged={page_num}"
        ]
        
        for url in urls_to_try:
            try:
                logger.info(f"Fetching page {page_num}: {url}")
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    ratings = self._parse_page(response.text)
                    if ratings:
                        return ratings
                        
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                continue
        
        return []
    
    def _parse_page(self, html: str) -> List[Dict]:
        """
        Parse HTML page to extract ratings.
        
        Args:
            html: HTML content
            
        Returns:
            List of rating dictionaries
        """
        soup = BeautifulSoup(html, 'html.parser')
        ratings = []
        
        # Find the ratings table
        table = soup.find('table')
        if not table:
            # Try finding by class or other attributes
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') or t.find('td'):
                    table = t
                    break
        
        if not table:
            logger.warning("No table found on page")
            return []
        
        # Get all rows
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            
            if len(cells) < 5:
                continue
            
            try:
                # Extract data from cells
                entity_name = cells[0].get_text(strip=True)
                rating_type = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                rating_raw = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                rating_action = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                outlook = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                issued_date = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                
                # Get PDF link if available
                pdf_link = ""
                if len(cells) > 6:
                    link = cells[6].find('a')
                    if link and link.get('href'):
                        pdf_link = link['href']
                
                # Skip if no entity name or rating
                if not entity_name or not rating_raw:
                    continue
                
                # Parse rating
                rating_info = self.parse_rating(rating_raw)
                
                # Parse date
                rating_date = self.parse_date(issued_date)
                
                # Handle missing outlook for pi ratings
                if rating_info["rating_basis"] == "pi" or not outlook or outlook == "-":
                    outlook = "N/A"
                
                rating_entry = {
                    "entity_name": entity_name,
                    "rating_type": rating_type,
                    "rating": rating_info["rating"],
                    "rating_action": rating_action,
                    "outlook": outlook,
                    "is_solicited": rating_info["is_solicited"],
                    "rating_basis": rating_info["rating_basis"],
                    "rating_date": rating_date,
                    "fiscal_year_applicable": self.extract_fiscal_year(rating_date),
                    "pdf_link": pdf_link,
                    "source_url": self.base_url
                }
                
                ratings.append(rating_entry)
                logger.debug(f"Parsed: {entity_name} - {rating_info['rating']}")
                
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
        
        return ratings
    
    def scrape_all(self, max_pages: int = 10) -> List[Dict]:
        """
        Scrape all pages of ratings.
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of all rating dictionaries
        """
        all_ratings = []
        
        for page in range(1, max_pages + 1):
            logger.info(f"Scraping page {page}/{max_pages}")
            
            ratings = self.scrape_page(page)
            
            if not ratings:
                logger.info(f"No more ratings found at page {page}")
                break
            
            all_ratings.extend(ratings)
            logger.info(f"Found {len(ratings)} ratings on page {page}")
            
            # Rate limiting
            time.sleep(self.rate_limit)
        
        # Remove duplicates (same entity + same date)
        seen = set()
        unique_ratings = []
        for r in all_ratings:
            key = (r["entity_name"], r["rating_date"], r["rating"])
            if key not in seen:
                seen.add(key)
                unique_ratings.append(r)
        
        logger.info(f"Total unique ratings collected: {len(unique_ratings)}")
        return unique_ratings
    
    def save_to_csv(self, ratings: List[Dict], output_path: Path):
        """
        Save ratings to CSV in template format.
        
        Args:
            ratings: List of rating dictionaries
            output_path: Output file path
        """
        # Convert to DataFrame
        df = pd.DataFrame(ratings)
        
        # Rename columns to match template
        df = df.rename(columns={
            "entity_name": "company_name"
        })
        
        # Add ticker column (will need manual mapping)
        df["ticker"] = ""
        
        # Reorder columns to match template
        columns = [
            "ticker", "company_name", "rating_date", "rating", "outlook",
            "rating_type", "is_solicited", "rating_basis", 
            "fiscal_year_applicable", "source_url", "pdf_link", "rating_action"
        ]
        
        # Only include columns that exist
        columns = [c for c in columns if c in df.columns]
        df = df[columns]
        
        # Convert is_solicited to yes/no
        if "is_solicited" in df.columns:
            df["is_solicited"] = df["is_solicited"].map({True: "yes", False: "no"})
        
        # Save
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} ratings to {output_path}")
        
        # Also save as JSON
        json_path = output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        logger.info(f"Saved JSON to {json_path}")
        
        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scrape Tassnief credit ratings")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "templates" / "ratings_scraped.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages to scrape"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Delay between requests (seconds)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tassnief Credit Ratings Scraper")
    print("=" * 60)
    print(f"Target: https://tassnief.com/find-rating")
    print(f"Output: {args.output}")
    print(f"Max pages: {args.max_pages}")
    print()
    
    # Initialize scraper
    scraper = TassnifScraper(rate_limit=args.rate_limit)
    
    # Scrape ratings
    print("Starting scrape...")
    ratings = scraper.scrape_all(max_pages=args.max_pages)
    
    if not ratings:
        print("\nNo ratings found. The website structure may have changed.")
        print("Try manual collection instead.")
        return
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = scraper.save_to_csv(ratings, output_path)
    
    print()
    print("=" * 60)
    print("Scraping Complete!")
    print("=" * 60)
    print(f"Total ratings collected: {len(ratings)}")
    print(f"Saved to: {output_path}")
    print()
    print("Summary by rating:")
    print(df["rating"].value_counts().to_string())
    print()
    print("Summary by rating basis:")
    print(df["rating_basis"].value_counts().to_string())
    print()
    print("NEXT STEPS:")
    print("1. Open the CSV and add ticker symbols manually")
    print("2. Verify the data looks correct")
    print("3. Copy to ratings_template.csv or merge with existing data")


if __name__ == "__main__":
    main()
