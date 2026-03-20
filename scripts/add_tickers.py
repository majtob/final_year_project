#!/usr/bin/env python3
"""
Add ticker symbols to Tassnief ratings data.
Uses manual mapping + Yahoo Finance search for Saudi companies.
Saudi stocks use .SR suffix on Yahoo Finance.
"""

import pandas as pd
from pathlib import Path
from loguru import logger
import time

# Try importing yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")

# Manual mapping of company names to Tadawul tickers
# Format: "company_name_pattern": "TICKER.SR"
TICKER_MAPPING = {
    # Telecom
    "saudi telecom": "7010.SR",
    "stc": "7010.SR",
    "etihad etisalat": "7020.SR",
    "mobily": "7020.SR",
    "zain saudi": "7030.SR",
    "mobile telecommunication": "7030.SR",  # Catches both variations
    
    # Healthcare
    "dr. sulaiman al-habib": "4013.SR",
    "al-habib": "4013.SR",
    "middle east healthcare": "4004.SR",
    "al hammadi": "4007.SR",
    
    # Finance/Insurance
    "quara finance": "9596.SR",  # Listed on NOMU
    "saudi arabian cooperative insurance": "8050.SR",
    "saico": "8050.SR",
    "malath cooperative insurance": "8060.SR",
    "malath": "8060.SR",
    "cooperative reinsurance": "8200.SR",
    "saudi re": "8200.SR",
    "saudi national bank capital": None,  # Subsidiary, not listed
    
    # Real Estate
    "arabian centers": "4321.SR",
    "cenomi centers": "4321.SR",
    "sumou real estate": "4323.SR",  # Listed on NOMU
    "innovest real estate": None,  # Not listed
    "ladun investment": "9535.SR",  # Listed on NOMU
    
    # Industrial/Services
    "catrion": "6004.SR",
    "alkhorayef": "2081.SR",
    "middle east paper": "2120.SR",
    "mepco": "2120.SR",
    "perfect presentation": "7204.SR",  # Listed (2P)
    "2p": "7204.SR",
    "sal saudi logistics": "4263.SR",
    "napco national": "2210.SR",
    
    # Training & Education
    "alkhaleej training": "4290.SR",  # Listed
    "al khaleej training": "4290.SR",
    
    # Construction/Contracting
    "multi business group": "9619.SR",  # Listed on NOMU
    "al kathiri holding": "3008.SR",  # Listed
    "akhc": "3008.SR",
    
    # Not Listed / Private Companies
    "sraco": None,  # Private (part of Saeed Raddad Group)
    "saeed raddad": None,  # Private holding company
    "al-gihaz": None,  # Private
    "rawabi holding": None,  # Private (Rawabi Marketing 9646 is different)
    "safa investment": None,  # Not found on Tadawul
    "atlas elevators": None,  # Not listed
    "mayar holding": None,  # Not listed
    "pharma medical": None,  # Not listed (different from SPIMACO 2070)
    "ejada systems": None,  # IPO pending, not yet listed
}


def normalize_name(name: str) -> str:
    """Normalize company name for matching."""
    name = name.lower()
    # Remove common suffixes
    for suffix in ["company", "co.", "holding", "corporation", "corp", "(unsolicited rating)", 
                   "for general trading and contracting", "for water and power technologies",
                   "catering holding", "medical services group"]:
        name = name.replace(suffix, "")
    return name.strip()


def find_ticker(company_name: str) -> str:
    """Find ticker for a company name."""
    normalized = normalize_name(company_name)
    
    # Check manual mapping first
    for pattern, ticker in TICKER_MAPPING.items():
        if pattern in normalized:
            return ticker if ticker else "NULL"
    
    # Try yfinance search if available
    if HAS_YFINANCE:
        ticker = search_yfinance(company_name)
        if ticker:
            return ticker
    
    return "NULL"


def search_yfinance(company_name: str) -> str:
    """Search Yahoo Finance for Saudi company ticker."""
    try:
        # Try common variations
        search_terms = [
            company_name,
            company_name.split("(")[0].strip(),  # Remove parenthetical
            company_name.replace("Company", "").strip(),
        ]
        
        for term in search_terms:
            # Search for the ticker
            search_query = f"{term} Saudi Arabia"
            
            # Use yfinance Ticker to check if it exists
            # This is a workaround since yfinance doesn't have direct search
            # We'd need to use a web search or the Yahoo Finance API directly
            pass
            
    except Exception as e:
        logger.debug(f"yfinance search failed for {company_name}: {e}")
    
    return None


def verify_ticker(ticker: str) -> bool:
    """Verify that a ticker exists on Yahoo Finance."""
    if not HAS_YFINANCE or ticker == "NULL" or ticker is None:
        return False
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("regularMarketPrice") is not None or info.get("previousClose") is not None
    except Exception:
        return False


def add_tickers_to_ratings(input_file: str, output_file: str = None):
    """Add ticker symbols to ratings CSV."""
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path
    
    # Read CSV
    df = pd.read_csv(input_path)
    logger.info(f"Read {len(df)} ratings from {input_path}")
    
    # Get unique company names
    unique_companies = df["company_name"].unique()
    logger.info(f"Found {len(unique_companies)} unique companies")
    
    # Build ticker mapping for all companies
    company_tickers = {}
    found_count = 0
    null_count = 0
    
    for company in unique_companies:
        ticker = find_ticker(company)
        company_tickers[company] = ticker
        
        if ticker != "NULL":
            found_count += 1
            logger.info(f"  {company} -> {ticker}")
        else:
            null_count += 1
            logger.debug(f"  {company} -> NULL (not found)")
    
    # Verify found tickers
    if HAS_YFINANCE:
        logger.info("\nVerifying tickers on Yahoo Finance...")
        for company, ticker in company_tickers.items():
            if ticker != "NULL":
                if not verify_ticker(ticker):
                    logger.warning(f"  Ticker {ticker} for {company} could not be verified")
                else:
                    logger.debug(f"  {ticker} verified ✓")
                time.sleep(0.5)  # Rate limiting
    
    # Apply tickers to dataframe
    df["ticker"] = df["company_name"].map(company_tickers)
    
    # Save updated CSV
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved updated CSV to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TICKER LOOKUP SUMMARY")
    print("=" * 60)
    print(f"Total unique companies: {len(unique_companies)}")
    print(f"Tickers found: {found_count}")
    print(f"Not found (NULL): {null_count}")
    print(f"\nListed companies:")
    for company, ticker in sorted(company_tickers.items()):
        if ticker != "NULL":
            print(f"  {ticker}: {company}")
    print(f"\nUnlisted/Private companies (NULL):")
    for company, ticker in sorted(company_tickers.items()):
        if ticker == "NULL":
            print(f"  - {company}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add ticker symbols to ratings CSV")
    parser.add_argument(
        "--input", "-i",
        default="data/templates/ratings_scraped.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV file (default: overwrite input)"
    )
    
    args = parser.parse_args()
    
    add_tickers_to_ratings(args.input, args.output)
