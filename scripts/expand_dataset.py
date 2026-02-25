"""
Expand dataset with multi-agency credit ratings.

This script:
1. Loads ratings from Argaam article (S&P, Moody's, Fitch, Tassnief, Financial Analytics)
2. Harmonizes Moody's scale to S&P/Fitch/Tassnief scale
3. Creates one row per company-agency rating
4. Pulls financials for new companies via yfinance
5. Calculates financial ratios
6. Outputs expanded dataset ready for ML

Source: Argaam article (Oct 2025) - https://www.argaam.com/en/article/articledetail/id/1850297
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
MULTI_AGENCY_FILE = PROJECT_ROOT / "data" / "templates" / "multi_agency_ratings.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "expanded_ratings_financials.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "financials"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Moody's long-term to standard scale mapping
MOODYS_MAP = {
    'Aaa': 'AAA',
    'Aa1': 'AA+', 'Aa2': 'AA', 'Aa3': 'AA-',
    'A1': 'A+', 'A2': 'A', 'A3': 'A-',
    'Baa1': 'BBB+', 'Baa2': 'BBB', 'Baa3': 'BBB-',
    'Ba1': 'BB+', 'Ba2': 'BB', 'Ba3': 'BB-',
    'B1': 'B+', 'B2': 'B', 'B3': 'B-',
    'Caa1': 'CCC+', 'Caa2': 'CCC', 'Caa3': 'CCC-',
    'Ca': 'CC', 'C': 'C',
}

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
    'CCC+': 5, 'CCC': 4, 'CCC-': 3,
    'CC': 2, 'C': 1, 'D': 0,
}


def harmonize_moodys(rating_str):
    """Convert Moody's rating to standard scale."""
    if pd.isna(rating_str) or not rating_str.strip():
        return None
    # Handle compound ratings like "Aa3/A1" - take first (long-term)
    parts = str(rating_str).strip().split('/')
    lt_rating = parts[0].strip()
    return MOODYS_MAP.get(lt_rating, lt_rating)


def expand_to_rows(df):
    """
    Convert one row per company (with multiple agency columns)
    into one row per company-agency rating.
    """
    agencies = {
        'moodys': 'Moodys',
        'fitch': 'Fitch',
        'sp': 'S&P',
        'tassnief': 'Tassnief',
        'financial_analytics': 'Financial Analytics',
    }
    
    rows = []
    for _, company in df.iterrows():
        for col, agency_name in agencies.items():
            raw_rating = company.get(col)
            if pd.isna(raw_rating) or not str(raw_rating).strip():
                continue
            
            raw_str = str(raw_rating).strip()
            
            # Harmonize Moody's
            if col == 'moodys':
                rating = harmonize_moodys(raw_str)
            else:
                # Strip national scale suffixes and short-term ratings
                rating = raw_str.split('/')[0].strip()
                # Remove national scale tags
                for suffix in [' AAA (sau)', ' AAA(sau)', ' KsaAAA', ' KSAAA', 
                               ' ksaAA+', ' AA-(sau)', ' A+(Sau)', ' AA-', ' AA']:
                    if rating.endswith(suffix) and len(rating) > len(suffix):
                        pass  # keep the international rating
                rating = rating.split(' ')[0]  # Take just the first part
            
            if not rating or rating not in RATING_TO_NUMERIC:
                logger.warning(f"  Skipping unrecognized rating '{raw_str}' ({agency_name}) for {company['ticker']}")
                continue
            
            rows.append({
                'ticker': company['ticker'],
                'company_name': company['company_name'],
                'sector': company['sector'],
                'rating_agency': agency_name,
                'rating': rating,
                'rating_numeric': RATING_TO_NUMERIC[rating],
                'fiscal_year': 2024,
                'source': company.get('source', 'argaam_oct2025'),
            })
    
    return pd.DataFrame(rows)


def pull_financials(ticker):
    """Pull financial statements and calculate ratios."""
    try:
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
        inc = stock.financials
        
        if bs is None or bs.empty:
            return None
        
        # Get 2024 data (or latest available)
        target_year = 2024
        bs_year = None
        inc_year = None
        
        for col in bs.columns:
            if hasattr(col, 'year') and col.year == target_year:
                bs_year = bs[col]
                break
        
        if inc is not None and not inc.empty:
            for col in inc.columns:
                if hasattr(col, 'year') and col.year == target_year:
                    inc_year = inc[col]
                    break
        
        if bs_year is None:
            return None
        
        def safe_get(series, keys):
            if series is None:
                return None
            for key in keys:
                if key in series.index and pd.notna(series[key]):
                    return series[key]
            return None
        
        total_assets = safe_get(bs_year, ['Total Assets', 'TotalAssets'])
        current_assets = safe_get(bs_year, ['Current Assets', 'CurrentAssets', 'Total Current Assets'])
        current_liabilities = safe_get(bs_year, ['Current Liabilities', 'CurrentLiabilities', 'Total Current Liabilities'])
        retained_earnings = safe_get(bs_year, ['Retained Earnings', 'RetainedEarnings'])
        total_equity = safe_get(bs_year, ['Total Equity Gross Minority Interest', 'Stockholders Equity', 
                                          'Total Equity', 'TotalEquityGrossMinorityInterest'])
        total_liabilities = safe_get(bs_year, ['Total Liabilities Net Minority Interest', 
                                                'Total Liabilities', 'TotalLiabilitiesNetMinorityInterest'])
        ebit = safe_get(inc_year, ['EBIT', 'Operating Income', 'OperatingIncome']) if inc_year is not None else None
        
        ratios = {}
        if total_assets and total_assets != 0:
            if current_assets is not None and current_liabilities is not None:
                ratios['liquid'] = (current_assets - current_liabilities) / total_assets
            if retained_earnings is not None:
                ratios['cumprof'] = retained_earnings / total_assets
            if ebit is not None:
                ratios['profitab'] = ebit / total_assets
        if total_equity is not None and total_liabilities is not None and total_liabilities != 0:
            ratios['leverage'] = total_equity / total_liabilities
        
        ratios['total_assets'] = total_assets
        ratios['total_liabilities'] = total_liabilities
        ratios['total_equity'] = total_equity
        
        return ratios
        
    except Exception as e:
        logger.warning(f"  Error for {ticker}: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("EXPANDING DATASET WITH MULTI-AGENCY RATINGS")
    logger.info("=" * 60)
    
    # Load multi-agency data
    df = pd.read_csv(MULTI_AGENCY_FILE)
    logger.info(f"Loaded {len(df)} companies from Argaam article")
    
    # Expand to one row per rating
    expanded = expand_to_rows(df)
    logger.info(f"Expanded to {len(expanded)} individual ratings")
    logger.info(f"\nRatings by agency:")
    print(expanded['rating_agency'].value_counts().to_string())
    logger.info(f"\nRating distribution:")
    print(expanded['rating'].value_counts().to_string())
    
    # Pull financials for all unique tickers
    unique_tickers = expanded['ticker'].unique()
    logger.info(f"\nPulling financials for {len(unique_tickers)} unique tickers...")
    
    financials_cache = {}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, ticker in enumerate(unique_tickers, 1):
        logger.info(f"[{i}/{len(unique_tickers)}] {ticker}")
        
        # Check if we already have cached data
        clean_ticker = ticker.replace('.', '_')
        cache_file = RAW_DIR / f"{clean_ticker}_financials.json"
        
        if cache_file.exists():
            logger.info(f"  Using cached data")
            ratios = pull_financials(ticker)
        else:
            ratios = pull_financials(ticker)
            time.sleep(0.5)
        
        if ratios:
            financials_cache[ticker] = ratios
            logger.info(f"  OK - {len(ratios)} fields")
        else:
            logger.warning(f"  No 2024 financial data")
    
    # Merge financials with ratings
    records = []
    for _, row in expanded.iterrows():
        record = row.to_dict()
        ticker = row['ticker']
        
        if ticker in financials_cache:
            record.update(financials_cache[ticker])
            record['has_financials'] = True
        else:
            record['has_financials'] = False
        
        records.append(record)
    
    result_df = pd.DataFrame(records)
    
    # Create rating category
    def rating_category(r):
        num = RATING_TO_NUMERIC.get(r, 0)
        if num >= 18:
            return 'AA'
        elif num >= 15:
            return 'A'
        elif num >= 12:
            return 'BBB'
        else:
            return 'BB'
    
    result_df['rating_category'] = result_df['rating'].apply(rating_category)
    
    # Save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    has_fin = result_df['has_financials'] == True
    feature_cols = ['liquid', 'cumprof', 'profitab', 'leverage']
    has_all = result_df[feature_cols].notna().all(axis=1) & has_fin
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPANSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total ratings: {len(result_df)}")
    logger.info(f"With financials: {has_fin.sum()}")
    logger.info(f"With ALL 4 ratios: {has_all.sum()}")
    logger.info(f"Unique companies: {result_df['ticker'].nunique()}")
    logger.info(f"Agencies: {result_df['rating_agency'].nunique()}")
    
    logger.info(f"\nCategory distribution (all with financials):")
    if has_all.any():
        print(result_df.loc[has_all, 'rating_category'].value_counts().to_string())
    
    logger.info(f"\nPrevious dataset: 23 samples")
    logger.info(f"New dataset: {has_all.sum()} samples")
    logger.info(f"Increase: {has_all.sum() - 23} more samples ({(has_all.sum()/23 - 1)*100:.0f}% increase)")
    
    logger.info(f"\nSaved to {OUTPUT_FILE}")
    
    return result_df


if __name__ == "__main__":
    main()
