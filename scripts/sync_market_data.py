#!/usr/bin/env python3
"""
Market Data Collection Script
Fetches stock market data from yfinance for Saudi Exchange tickers (2021-2024).
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "market_data_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class MarketDataCollector:
    """Collects market data from yfinance for Saudi Exchange tickers."""
    
    def __init__(
        self,
        output_dir: Path,
        start_date: str = "2021-01-01",
        end_date: str = "2024-12-31",
        interval: str = "1d"
    ):
        """
        Initialize market data collector.
        
        Args:
            output_dir: Directory to save market data files
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            interval: Data interval (1d, 1wk, 1mo)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.download_log = []
        
    def fetch_ticker_data(
        self,
        ticker: str,
        retry_count: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker.
        
        Args:
            ticker: Stock ticker (e.g., "2222.SR")
            retry_count: Number of retry attempts
            
        Returns:
            DataFrame with market data or None if failed
        """
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching data for {ticker} (attempt {attempt + 1})")
                
                # Create yfinance ticker object
                stock = yf.Ticker(ticker)
                
                # Fetch historical data
                hist = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if hist.empty:
                    logger.warning(f"No data found for {ticker}")
                    return None
                
                # Add ticker column
                hist['ticker'] = ticker
                hist.reset_index(inplace=True)
                
                # Rename columns for consistency
                hist.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Dividends': 'dividends',
                    'Stock Splits': 'stock_splits'
                }, inplace=True)
                
                # Calculate market cap if available
                if 'close' in hist.columns:
                    # Get shares outstanding
                    info = stock.info
                    shares_outstanding = info.get('sharesOutstanding', None)
                    
                    if shares_outstanding:
                        hist['market_cap'] = hist['close'] * shares_outstanding
                    else:
                        hist['market_cap'] = None
                
                logger.success(f"Successfully fetched {len(hist)} records for {ticker}")
                return hist
                
            except Exception as e:
                logger.error(f"Error fetching {ticker} (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
        
        return None
    
    def save_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        format: str = "parquet"
    ) -> Path:
        """
        Save market data to file.
        
        Args:
            df: DataFrame to save
            ticker: Stock ticker
            format: File format (parquet or csv)
            
        Returns:
            Path to saved file
        """
        ticker_clean = ticker.replace(".", "_")
        
        if format == "parquet":
            file_path = self.output_dir / f"{ticker_clean}_market_data.parquet"
            df.to_parquet(file_path, index=False, compression='snappy')
        else:
            file_path = self.output_dir / f"{ticker_clean}_market_data.csv"
            df.to_csv(file_path, index=False)
        
        logger.info(f"Saved data to {file_path}")
        return file_path
    
    def collect_tickers(
        self,
        tickers: List[str],
        save_format: str = "parquet"
    ) -> dict:
        """
        Collect data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            save_format: Format to save files (parquet or csv)
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"Starting market data collection for {len(tickers)} tickers")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        successful = []
        failed = []
        total_records = 0
        
        for ticker in tqdm(tickers, desc="Collecting market data"):
            df = self.fetch_ticker_data(ticker)
            
            if df is not None and not df.empty:
                file_path = self.save_data(df, ticker, save_format)
                successful.append(ticker)
                total_records += len(df)
                
                # Log download metadata
                self.download_log.append({
                    'ticker': ticker,
                    'file_path': str(file_path),
                    'records': len(df),
                    'date_range': {
                        'start': str(df['date'].min()),
                        'end': str(df['date'].max())
                    },
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
            else:
                failed.append(ticker)
                self.download_log.append({
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed'
                })
        
        # Save download log
        log_path = self.output_dir / "download_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.download_log, f, indent=2)
        
        # Generate summary
        summary = {
            'total_tickers': len(tickers),
            'successful': len(successful),
            'failed': len(failed),
            'total_records': total_records,
            'date_range': {
                'start': self.start_date,
                'end': self.end_date
            },
            'interval': self.interval,
            'successful_tickers': successful,
            'failed_tickers': failed,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "collection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nCollection Summary:")
        logger.info(f"  Successful: {len(successful)}/{len(tickers)}")
        logger.info(f"  Failed: {len(failed)}/{len(tickers)}")
        logger.info(f"  Total records: {total_records:,}")
        
        if failed:
            logger.warning(f"Failed tickers: {', '.join(failed)}")
        
        return summary
    
    def verify_coverage(
        self,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Verify data coverage for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with coverage information
        """
        coverage_data = []
        
        for ticker in tickers:
            ticker_clean = ticker.replace(".", "_")
            parquet_path = self.output_dir / f"{ticker_clean}_market_data.parquet"
            csv_path = self.output_dir / f"{ticker_clean}_market_data.csv"
            
            file_path = parquet_path if parquet_path.exists() else csv_path
            
            if file_path.exists():
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                coverage_data.append({
                    'ticker': ticker,
                    'file_exists': True,
                    'records': len(df),
                    'date_start': str(df['date'].min()) if 'date' in df.columns else None,
                    'date_end': str(df['date'].max()) if 'date' in df.columns else None,
                    'missing_dates': None  # Could calculate this
                })
            else:
                coverage_data.append({
                    'ticker': ticker,
                    'file_exists': False,
                    'records': 0,
                    'date_start': None,
                    'date_end': None,
                    'missing_dates': None
                })
        
        coverage_df = pd.DataFrame(coverage_data)
        coverage_path = self.output_dir / "coverage_report.csv"
        coverage_df.to_csv(coverage_path, index=False)
        
        logger.info(f"Coverage report saved to {coverage_path}")
        return coverage_df


def load_tickers_from_inventory(inventory_path: Path) -> List[str]:
    """Load tickers from data inventory file."""
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    tickers = []
    if 'market_data' in inventory and 'tickers' in inventory['market_data']:
        tickers = inventory['market_data']['tickers']
    elif 'companies' in inventory:
        tickers = [comp.get('ticker') for comp in inventory['companies'] if comp.get('ticker')]
    
    return [t for t in tickers if t]  # Remove None values


def main():
    """Main entry point for market data collection."""
    parser = argparse.ArgumentParser(description="Collect market data from yfinance")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of ticker symbols (e.g., 1111.SR 2222.SR)",
        default=[]
    )
    parser.add_argument(
        "--inventory",
        type=str,
        help="Path to data inventory JSON file",
        default=str(PROJECT_ROOT / "data" / "data_inventory.json")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for market data",
        default=str(PROJECT_ROOT / "data" / "market")
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
        default="2021-01-01"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
        default="2024-12-31"
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["1d", "1wk", "1mo"],
        help="Data interval",
        default="1d"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        help="Output file format",
        default="parquet"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data coverage after collection"
    )
    
    args = parser.parse_args()
    
    # Load tickers
    tickers = args.tickers
    if not tickers:
        inventory_path = Path(args.inventory)
        if inventory_path.exists():
            logger.info(f"Loading tickers from {inventory_path}")
            tickers = load_tickers_from_inventory(inventory_path)
        else:
            logger.error(f"Inventory file not found: {inventory_path}")
            logger.info("Please provide tickers via --tickers or populate data_inventory.json")
            return
    
    if not tickers:
        logger.error("No tickers provided")
        return
    
    # Initialize collector
    collector = MarketDataCollector(
        output_dir=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval
    )
    
    # Collect data
    summary = collector.collect_tickers(tickers, save_format=args.format)
    
    # Verify coverage if requested
    if args.verify:
        collector.verify_coverage(tickers)
    
    logger.success("Market data collection completed!")


if __name__ == "__main__":
    main()

