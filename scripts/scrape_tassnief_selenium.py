#!/usr/bin/env python3
"""
Tassnief Credit Ratings Scraper (Selenium Version)
Handles JavaScript-based pagination.

Usage:
    python scripts/scrape_tassnief_selenium.py --output data/templates/ratings_scraped.csv

Requirements:
    pip install selenium webdriver-manager
"""

import os
import sys
import argparse
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import (
        TimeoutException, NoSuchElementException, 
        ElementClickInterceptedException, StaleElementReferenceException
    )
except ImportError:
    print("Selenium not installed. Run: pip install selenium webdriver-manager")
    sys.exit(1)

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    ChromeDriverManager = None
    print("webdriver-manager not installed. Run: pip install webdriver-manager")


PROJECT_ROOT = Path(__file__).parent.parent

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "tassnief_selenium_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG"
)


class TassnifSeleniumScraper:
    """Scrapes credit ratings from Tassnief using Selenium."""
    
    def __init__(self, headless: bool = True, wait_time: float = 3.0):
        """
        Initialize scraper.
        
        Args:
            headless: Run browser in headless mode
            wait_time: Time to wait between actions
        """
        self.base_url = "https://tassnief.com/find-rating"
        self.wait_time = wait_time
        self.driver = None
        self.headless = headless
    
    def _init_driver(self):
        """Initialize Chrome WebDriver."""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            if ChromeDriverManager:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)
            
            self.driver.implicitly_wait(10)
            logger.info("Chrome WebDriver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def _close_driver(self):
        """Close WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _parse_with_beautifulsoup(self, html: str) -> List[Dict]:
        """Parse ratings using BeautifulSoup as fallback."""
        ratings = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        logger.info(f"BeautifulSoup found {len(tables)} tables")
        
        for table in tables:
            rows = table.find_all('tr')
            logger.info(f"Table has {len(rows)} rows")
            
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 5:
                    continue
                
                try:
                    entity_name = cells[0].get_text(strip=True)
                    rating_type = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    rating_raw = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    rating_action = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                    outlook = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                    issued_date = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                    
                    # Get PDF link
                    pdf_link = ""
                    if len(cells) > 6:
                        link = cells[6].find('a')
                        if link and link.get('href'):
                            pdf_link = link['href']
                    
                    if not entity_name or not rating_raw:
                        continue
                    
                    rating_info = self.parse_rating(rating_raw)
                    rating_date = self.parse_date(issued_date)
                    
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
                    logger.debug(f"BS Parsed: {entity_name} - {rating_info['rating']}")
                    
                except Exception as e:
                    logger.warning(f"Error parsing row with BS: {e}")
                    continue
        
        return ratings
    
    def parse_rating(self, rating_text: str) -> Dict:
        """Parse rating text to extract components."""
        rating_text = rating_text.strip()
        
        is_pi = "(pi)" in rating_text.lower()
        is_unsolicited = "unsolicited" in rating_text.lower()
        
        # Extract base rating - preserve +/- suffix
        base_rating = rating_text
        base_rating = re.sub(r'\s*\(pi\)\s*', '', base_rating, flags=re.IGNORECASE)
        base_rating = re.sub(r'\s*-?\s*unsolicited\s*', '', base_rating, flags=re.IGNORECASE)
        base_rating = base_rating.strip()
        
        # Clean any remaining whitespace but keep the rating intact
        # Valid ratings: AAA, AA+, AA, AA-, A+, A, A-, BBB+, BBB, BBB-, BB+, BB, BB-, B+, B, B-, CCC, CC, C, D
        rating_match = re.match(r'^(AAA|AA\+|AA-|AA|A\+|A-|A|BBB\+|BBB-|BBB|BB\+|BB-|BB|B\+|B-|B|CCC\+|CCC-|CCC|CC|C|D)', base_rating, re.IGNORECASE)
        if rating_match:
            base_rating = rating_match.group(1).upper()
        
        return {
            "rating": base_rating,
            "is_solicited": not is_unsolicited,
            "rating_basis": "pi" if is_pi else "full"
        }
    
    def parse_date(self, date_text: str) -> str:
        """Parse date to YYYY-MM-DD format."""
        date_text = date_text.strip()
        
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        try:
            match = re.match(r'(\w+),?\s*(\d{4})', date_text, re.IGNORECASE)
            if match:
                month_str = match.group(1).lower()[:3]
                year = match.group(2)
                month = month_map.get(month_str, '01')
                return f"{year}-{month}-15"
        except:
            pass
        
        return date_text
    
    def extract_fiscal_year(self, date_str: str) -> int:
        """Extract fiscal year from rating date."""
        try:
            year = int(date_str[:4])
            month = int(date_str[5:7])
            if month <= 3:
                return year - 1
            return year
        except:
            return datetime.now().year
    
    def _parse_current_page(self) -> List[Dict]:
        """Parse ratings from current page."""
        ratings = []
        
        try:
            # Wait longer for dynamic content
            time.sleep(5)
            
            # Try multiple selectors for the table
            table = None
            table_selectors = [
                "table",
                ".rating-table",
                "#rating-table", 
                "[class*='table']",
                ".table-responsive table",
                "div.table table",
            ]
            
            for selector in table_selectors:
                try:
                    if selector.startswith(".") or selector.startswith("#") or "[" in selector:
                        table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    else:
                        table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.TAG_NAME, selector))
                        )
                    if table:
                        logger.info(f"Found table with selector: {selector}")
                        break
                except:
                    continue
            
            if not table:
                # Debug: print page source snippet
                page_source = self.driver.page_source
                logger.debug(f"Page source length: {len(page_source)}")
                
                # Look for any table-like structures
                soup = BeautifulSoup(page_source, 'html.parser')
                tables = soup.find_all('table')
                logger.info(f"Found {len(tables)} tables via BeautifulSoup")
                
                if tables:
                    # Parse with BeautifulSoup instead
                    return self._parse_with_beautifulsoup(page_source)
                
                # Try to find rows directly
                rows_found = self.driver.find_elements(By.CSS_SELECTOR, "tr, .row, [class*='row']")
                logger.info(f"Found {len(rows_found)} row-like elements")
                
                return []
            
            rows = table.find_elements(By.TAG_NAME, "tr")
            logger.info(f"Found {len(rows)} rows in table")
            
            # Debug: print first few rows
            for i, row in enumerate(rows[:3]):
                cells = row.find_elements(By.TAG_NAME, "td")
                headers = row.find_elements(By.TAG_NAME, "th")
                logger.info(f"Row {i}: {len(cells)} td cells, {len(headers)} th cells")
                if cells:
                    cell_texts = [c.text[:50] for c in cells[:5]]
                    logger.info(f"Row {i} content: {cell_texts}")
            
            for row in rows[1:]:  # Skip header
                cells = row.find_elements(By.TAG_NAME, "td")
                
                if len(cells) < 5:
                    continue
                
                try:
                    entity_name = cells[0].text.strip()
                    rating_type = cells[1].text.strip() if len(cells) > 1 else ""
                    rating_raw = cells[2].text.strip() if len(cells) > 2 else ""
                    rating_action = cells[3].text.strip() if len(cells) > 3 else ""
                    outlook = cells[4].text.strip() if len(cells) > 4 else ""
                    issued_date = cells[5].text.strip() if len(cells) > 5 else ""
                    
                    # Get PDF link
                    pdf_link = ""
                    if len(cells) > 6:
                        try:
                            link = cells[6].find_element(By.TAG_NAME, "a")
                            pdf_link = link.get_attribute("href") or ""
                        except:
                            pass
                    
                    if not entity_name or not rating_raw:
                        continue
                    
                    rating_info = self.parse_rating(rating_raw)
                    rating_date = self.parse_date(issued_date)
                    
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
                    
        except Exception as e:
            logger.error(f"Error parsing page: {e}")
        
        return ratings
    
    def _get_pagination_info(self) -> tuple:
        """Get current page and total pages."""
        try:
            # Look for pagination elements - adjust selectors based on actual page
            pagination = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='page'], button.page, .pagination a, .pagination button, nav a")
            
            page_numbers = []
            for elem in pagination:
                text = elem.text.strip()
                if text.isdigit():
                    page_numbers.append(int(text))
            
            if page_numbers:
                return 1, max(page_numbers)
            
            # Try alternative: look for page indicator text
            page_text = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Page')]")
            for elem in page_text:
                match = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', elem.text, re.IGNORECASE)
                if match:
                    return int(match.group(1)), int(match.group(2))
                    
        except Exception as e:
            logger.warning(f"Could not get pagination info: {e}")
        
        return 1, 1
    
    def _click_next_page(self, current_page: int) -> bool:
        """
        Click to go to next page.
        
        Returns:
            True if successfully navigated to next page, False otherwise
        """
        try:
            # Try different pagination selectors
            selectors = [
                f"//a[text()='{current_page + 1}']",  # Page number link
                f"//button[text()='{current_page + 1}']",  # Page number button
                "//a[contains(@class, 'next')]",  # Next link
                "//button[contains(@class, 'next')]",  # Next button
                "//a[contains(text(), 'Next')]",
                "//a[contains(text(), '>')]",
                "//a[contains(text(), '›')]",
                "//li[contains(@class, 'next')]/a",
                "//a[@aria-label='Next']",
            ]
            
            for selector in selectors:
                try:
                    next_btn = self.driver.find_element(By.XPATH, selector)
                    if next_btn.is_displayed() and next_btn.is_enabled():
                        # Scroll to element
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                        time.sleep(0.5)
                        
                        # Click
                        try:
                            next_btn.click()
                        except ElementClickInterceptedException:
                            self.driver.execute_script("arguments[0].click();", next_btn)
                        
                        # Wait for page to load
                        time.sleep(self.wait_time)
                        
                        logger.info(f"Navigated to page {current_page + 1}")
                        return True
                except NoSuchElementException:
                    continue
                except StaleElementReferenceException:
                    continue
            
            logger.info("No next page button found")
            return False
            
        except Exception as e:
            logger.error(f"Error clicking next page: {e}")
            return False
    
    def scrape_all(self, max_pages: int = 20) -> List[Dict]:
        """
        Scrape all pages of ratings.
        
        Args:
            max_pages: Maximum pages to scrape
            
        Returns:
            List of all rating dictionaries
        """
        all_ratings = []
        
        try:
            self._init_driver()
            
            # Load initial page
            logger.info(f"Loading {self.base_url}")
            self.driver.get(self.base_url)
            time.sleep(self.wait_time)
            
            # Get pagination info
            _, total_pages = self._get_pagination_info()
            logger.info(f"Detected {total_pages} pages")
            
            pages_to_scrape = min(max_pages, total_pages) if total_pages > 1 else max_pages
            
            current_page = 1
            seen_ratings = set()  # Track unique ratings to detect duplicate pages
            
            while current_page <= pages_to_scrape:
                logger.info(f"Scraping page {current_page}/{pages_to_scrape}")
                
                # Parse current page
                page_ratings = self._parse_current_page()
                
                if not page_ratings:
                    logger.warning(f"No ratings found on page {current_page}")
                    break
                
                # Check for duplicates (same page being returned)
                page_key = tuple((r["entity_name"], r["rating_date"]) for r in page_ratings)
                if page_key in seen_ratings:
                    logger.warning("Duplicate page detected, stopping")
                    break
                seen_ratings.add(page_key)
                
                all_ratings.extend(page_ratings)
                logger.info(f"Found {len(page_ratings)} ratings on page {current_page}")
                
                # Try to go to next page
                if current_page < pages_to_scrape:
                    if not self._click_next_page(current_page):
                        logger.info("Could not navigate to next page, stopping")
                        break
                
                current_page += 1
            
        finally:
            self._close_driver()
        
        # Remove duplicates
        seen = set()
        unique_ratings = []
        for r in all_ratings:
            key = (r["entity_name"], r["rating_date"], r["rating"])
            if key not in seen:
                seen.add(key)
                unique_ratings.append(r)
        
        logger.info(f"Total unique ratings collected: {len(unique_ratings)}")
        return unique_ratings
    
    def save_to_csv(self, ratings: List[Dict], output_path: Path) -> pd.DataFrame:
        """Save ratings to CSV."""
        df = pd.DataFrame(ratings)
        
        df = df.rename(columns={"entity_name": "company_name"})
        df["ticker"] = ""
        
        columns = [
            "ticker", "company_name", "rating_date", "rating", "outlook",
            "rating_type", "is_solicited", "rating_basis",
            "fiscal_year_applicable", "source_url", "pdf_link", "rating_action"
        ]
        columns = [c for c in columns if c in df.columns]
        df = df[columns]
        
        if "is_solicited" in df.columns:
            df["is_solicited"] = df["is_solicited"].map({True: "yes", False: "no"})
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} ratings to {output_path}")
        
        json_path = output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Scrape Tassnief ratings (Selenium)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "templates" / "ratings_scraped.csv"),
        help="Output CSV file"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages to scrape"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode"
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window"
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=3.0,
        help="Wait time between pages (seconds)"
    )
    
    args = parser.parse_args()
    
    headless = not args.no_headless
    
    print("=" * 60)
    print("Tassnief Credit Ratings Scraper (Selenium)")
    print("=" * 60)
    print(f"Target: https://tassnief.com/find-rating")
    print(f"Output: {args.output}")
    print(f"Max pages: {args.max_pages}")
    print(f"Headless: {headless}")
    print()
    
    scraper = TassnifSeleniumScraper(headless=headless, wait_time=args.wait)
    
    print("Starting scrape...")
    ratings = scraper.scrape_all(max_pages=args.max_pages)
    
    if not ratings:
        print("\nNo ratings found.")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = scraper.save_to_csv(ratings, output_path)
    
    print()
    print("=" * 60)
    print("Scraping Complete!")
    print("=" * 60)
    print(f"Total ratings: {len(ratings)}")
    print(f"Saved to: {output_path}")
    print()
    print("Summary by rating:")
    print(df["rating"].value_counts().to_string())
    print()
    print("Summary by solicited/unsolicited:")
    print(df["is_solicited"].value_counts().to_string())


if __name__ == "__main__":
    main()
