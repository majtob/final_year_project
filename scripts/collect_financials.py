"""
Collect financial statements for all Tassnief-rated companies.

This script:
1. Reads tickers and fiscal years from ratings_scraped.csv
2. Pulls financial data from yfinance
3. Calculates key financial ratios (from Muñoz-Izquierdo et al., 2022)
4. Outputs a merged dataset ready for ML training

Financial Ratios (Altman Z''-Score components):
- LIQUID: Working Capital / Total Assets
- CUMPROF: Retained Earnings / Total Assets
- PROFITAB: EBIT / Total Assets
- LEVERAGE: Book Value of Equity / Total Liabilities
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import time
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RATINGS_FILE = DATA_DIR / "templates" / "ratings_scraped.csv"
OUTPUT_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw" / "financials"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ratings():
    """Load ratings data and extract unique ticker-year combinations."""
    df = pd.read_csv(RATINGS_FILE)
    
    # Get unique ticker-year combinations
    ticker_years = df[['ticker', 'company_name', 'fiscal_year_applicable', 'rating']].drop_duplicates()
    
    logger.info(f"Loaded {len(df)} ratings for {df['ticker'].nunique()} unique tickers")
    logger.info(f"Fiscal years: {sorted(df['fiscal_year_applicable'].unique())}")
    
    return df, ticker_years


def pull_financials_for_ticker(ticker: str):
    """
    Pull all available financial statements for a ticker.
    
    Args:
        ticker: Stock ticker (e.g., "7010.SR")
        
    Returns:
        dict with balance_sheet, income_statement, cashflow DataFrames
    """
    logger.info(f"Fetching financials for {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        
        data = {
            'ticker': ticker,
            'info': stock.info,
            'balance_sheet': stock.balance_sheet,
            'income_statement': stock.financials,
            'cashflow': stock.cashflow,
        }
        
        # Log what we got
        for key in ['balance_sheet', 'income_statement', 'cashflow']:
            if data[key] is not None and not data[key].empty:
                years = [col.year for col in data[key].columns if hasattr(col, 'year')]
                logger.info(f"  {key}: {len(data[key])} rows, years: {years}")
            else:
                logger.warning(f"  {key}: No data available")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None


def extract_year_data(df: pd.DataFrame, target_year: int):
    """
    Extract data for a specific fiscal year from a financial statement DataFrame.
    
    Args:
        df: Financial statement DataFrame (rows=metrics, cols=dates)
        target_year: The fiscal year to extract
        
    Returns:
        Series with the financial data for that year, or None
    """
    if df is None or df.empty:
        return None
    
    # Find column matching target year
    for col in df.columns:
        if hasattr(col, 'year') and col.year == target_year:
            return df[col]
    
    return None


def calculate_financial_ratios(balance_sheet: pd.Series, income_statement: pd.Series):
    """
    Calculate the 4 key financial ratios from the Muñoz-Izquierdo paper.
    
    Ratios:
    - LIQUID: Working Capital / Total Assets
    - CUMPROF: Retained Earnings / Total Assets
    - PROFITAB: EBIT / Total Assets
    - LEVERAGE: Book Value of Equity / Total Liabilities
    
    Returns:
        dict with calculated ratios
    """
    ratios = {
        'liquid': None,
        'cumprof': None,
        'profitab': None,
        'leverage': None,
    }
    
    if balance_sheet is None:
        return ratios
    
    # Helper to safely get values
    def safe_get(series, keys, default=None):
        """Try multiple possible keys to find a value."""
        if series is None:
            return default
        for key in keys:
            if key in series.index:
                val = series[key]
                if pd.notna(val):
                    return val
        return default
    
    # Extract balance sheet items
    total_assets = safe_get(balance_sheet, ['Total Assets', 'TotalAssets'])
    current_assets = safe_get(balance_sheet, ['Current Assets', 'CurrentAssets', 'Total Current Assets'])
    current_liabilities = safe_get(balance_sheet, ['Current Liabilities', 'CurrentLiabilities', 'Total Current Liabilities'])
    retained_earnings = safe_get(balance_sheet, ['Retained Earnings', 'RetainedEarnings'])
    total_equity = safe_get(balance_sheet, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest', 
                                             'Stockholders Equity', 'Total Equity', 'TotalEquityGrossMinorityInterest'])
    total_liabilities = safe_get(balance_sheet, ['Total Liabilities Net Minority Interest', 'Total Liab',
                                                  'Total Liabilities', 'TotalLiabilitiesNetMinorityInterest'])
    
    # Extract income statement items
    ebit = None
    if income_statement is not None:
        ebit = safe_get(income_statement, ['EBIT', 'Operating Income', 'OperatingIncome'])
    
    # Calculate ratios
    if total_assets and total_assets != 0:
        # LIQUID = Working Capital / Total Assets
        if current_assets is not None and current_liabilities is not None:
            working_capital = current_assets - current_liabilities
            ratios['liquid'] = working_capital / total_assets
        
        # CUMPROF = Retained Earnings / Total Assets
        if retained_earnings is not None:
            ratios['cumprof'] = retained_earnings / total_assets
        
        # PROFITAB = EBIT / Total Assets
        if ebit is not None:
            ratios['profitab'] = ebit / total_assets
    
    # LEVERAGE = Book Value of Equity / Total Liabilities
    if total_equity is not None and total_liabilities is not None and total_liabilities != 0:
        ratios['leverage'] = total_equity / total_liabilities
    
    return ratios


def save_raw_financials(ticker: str, data: dict):
    """Save raw financial data to JSON for reference."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    clean_ticker = ticker.replace('.', '_')
    filepath = RAW_DIR / f"{clean_ticker}_financials.json"
    
    # Convert to JSON-serializable format
    json_data = {
        'ticker': ticker,
        'fetch_date': datetime.now().isoformat(),
        'company_name': data['info'].get('longName', ''),
        'sector': data['info'].get('sector', ''),
        'industry': data['info'].get('industry', ''),
    }
    
    # Convert DataFrames to dict
    for key in ['balance_sheet', 'income_statement', 'cashflow']:
        df = data[key]
        if df is not None and not df.empty:
            # Convert to dict with string keys
            df_dict = {}
            for col in df.columns:
                year_str = str(col.year) if hasattr(col, 'year') else str(col)
                df_dict[year_str] = df[col].to_dict()
            json_data[key] = df_dict
        else:
            json_data[key] = None
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    logger.info(f"Saved raw financials to {filepath.name}")


def process_all_tickers():
    """
    Main function to process all tickers from the ratings file.
    
    Returns:
        DataFrame with ratings + financial ratios
    """
    # Load ratings
    ratings_df, ticker_years = load_ratings()
    
    # Get unique tickers
    unique_tickers = ratings_df['ticker'].unique()
    logger.info(f"\nProcessing {len(unique_tickers)} unique tickers...")
    
    # Store all financial data
    all_financials = {}
    
    # Pull financials for each ticker
    for i, ticker in enumerate(unique_tickers, 1):
        logger.info(f"\n[{i}/{len(unique_tickers)}] Processing {ticker}")
        
        data = pull_financials_for_ticker(ticker)
        if data:
            all_financials[ticker] = data
            save_raw_financials(ticker, data)
        
        # Rate limiting to avoid API blocks
        time.sleep(1)
    
    # Now match financials to ratings by fiscal year
    results = []
    
    for _, row in ratings_df.iterrows():
        ticker = row['ticker']
        fiscal_year = row['fiscal_year_applicable']
        
        result = {
            'ticker': ticker,
            'company_name': row['company_name'],
            'rating_date': row['rating_date'],
            'rating': row['rating'],
            'outlook': row['outlook'],
            'fiscal_year': fiscal_year,
            'rating_action': row['rating_action'],
        }
        
        # Get financial ratios for this ticker/year
        if ticker in all_financials:
            data = all_financials[ticker]
            
            # Extract data for the specific fiscal year
            bs_year = extract_year_data(data['balance_sheet'], fiscal_year)
            is_year = extract_year_data(data['income_statement'], fiscal_year)
            
            # Calculate ratios
            ratios = calculate_financial_ratios(bs_year, is_year)
            result.update(ratios)
            
            # Add raw balance sheet items for reference
            if bs_year is not None:
                result['total_assets'] = bs_year.get('Total Assets', bs_year.get('TotalAssets'))
                result['total_liabilities'] = bs_year.get('Total Liabilities Net Minority Interest', 
                                                          bs_year.get('TotalLiabilitiesNetMinorityInterest'))
                result['total_equity'] = bs_year.get('Total Equity Gross Minority Interest',
                                                     bs_year.get('TotalEquityGrossMinorityInterest'))
            
            result['data_available'] = bs_year is not None
        else:
            result['data_available'] = False
        
        results.append(result)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    return output_df


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("FINANCIAL DATA COLLECTION")
    logger.info("=" * 60)
    
    # Process all tickers
    output_df = process_all_tickers()
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = OUTPUT_DIR / "ratings_with_financials.csv"
    output_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved merged data to {output_file}")
    
    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(output_df)}")
    logger.info(f"Records with financial data: {output_df['data_available'].sum()}")
    logger.info(f"Records missing data: {(~output_df['data_available']).sum()}")
    
    # Show coverage by ratio
    for ratio in ['liquid', 'cumprof', 'profitab', 'leverage']:
        coverage = output_df[ratio].notna().sum()
        logger.info(f"  {ratio.upper()}: {coverage}/{len(output_df)} ({100*coverage/len(output_df):.1f}%)")
    
    # Show sample of data
    logger.info("\nSample output:")
    print(output_df[['ticker', 'fiscal_year', 'rating', 'liquid', 'cumprof', 'profitab', 'leverage']].head(10))
    
    return output_df


if __name__ == "__main__":
    main()
