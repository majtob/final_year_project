#!/usr/bin/env python3
"""
Example: How to save data to database (DuckDB or PostgreSQL)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import DataStorage
import pandas as pd
from loguru import logger

# Example 1: Using DuckDB (Recommended)
def example_duckdb():
    """Example using DuckDB - no setup required!"""
    logger.info("=== DuckDB Example ===")
    
    # Initialize DuckDB (creates data/tadawul.duckdb automatically)
    storage = DataStorage(storage_type="duckdb")
    
    # Create tables
    storage.create_tables()
    
    # Save company financial data
    company_data = {
        'company_name': 'Saudi Aramco',
        'ticker': '2222.SR',
        'fiscal_year': 2023,
        'fiscal_period': 'FY',
        'reporting_date': '2023-12-31',
        'total_assets': 1000000000.0,
        'total_liabilities': 600000000.0,
        'total_equity': 400000000.0,
        'revenue': 500000000.0,
        'net_income': 50000000.0,
        'current_ratio': 1.5,
        'debt_to_equity': 1.5,
        'source_file_id': 'file_001',
        'source_url': 'https://example.com/filing.pdf',
        'language': 'en'
    }
    
    record_id = storage.save_company_period(company_data)
    logger.success(f"Saved company period data (ID: {record_id})")
    
    # Save market data
    market_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'open': [100.0, 101.0, 102.0, 101.5, 103.0],
        'high': [101.0, 102.0, 103.0, 102.0, 104.0],
        'low': [99.0, 100.0, 101.0, 101.0, 102.0],
        'close': [100.5, 101.5, 102.5, 101.8, 103.5],
        'volume': [1000000, 1100000, 1200000, 1050000, 1300000],
        'market_cap': [2000000000.0, 2030000000.0, 2050000000.0, 2036000000.0, 2070000000.0]
    })
    
    records_saved = storage.save_market_data(market_data, ticker='2222.SR')
    logger.success(f"Saved {records_saved} market data records")
    
    # Query data
    logger.info("Querying saved data...")
    results = storage.query("""
        SELECT 
            cp.company_name,
            cp.ticker,
            cp.fiscal_year,
            cp.total_assets,
            cp.revenue,
            cp.net_income,
            AVG(md.close) as avg_price,
            SUM(md.volume) as total_volume
        FROM company_periods cp
        LEFT JOIN market_data md ON cp.ticker = md.ticker
        WHERE cp.ticker = '2222.SR'
        GROUP BY cp.id, cp.company_name, cp.ticker, cp.fiscal_year, 
                 cp.total_assets, cp.revenue, cp.net_income
    """)
    
    print("\nQuery Results:")
    print(results)
    
    # Close connection
    storage.close()
    logger.success("DuckDB example completed!")


# Example 2: Using PostgreSQL (if you have it set up)
def example_postgresql():
    """Example using PostgreSQL - requires setup"""
    logger.info("=== PostgreSQL Example ===")
    
    # Initialize PostgreSQL (requires connection string)
    connection_string = "postgresql://user:password@localhost/tadawul_data"
    
    try:
        storage = DataStorage(
            storage_type="postgresql",
            connection_string=connection_string
        )
        
        # Same API as DuckDB!
        storage.create_tables()
        
        # Save data (same as DuckDB example)
        company_data = {
            'company_name': 'Saudi Aramco',
            'ticker': '2222.SR',
            'fiscal_year': 2023,
            'fiscal_period': 'FY',
            'reporting_date': '2023-12-31',
            'total_assets': 1000000000.0,
            'revenue': 500000000.0,
            'net_income': 50000000.0
        }
        
        storage.save_company_period(company_data)
        logger.success("Saved to PostgreSQL!")
        
        storage.close()
        
    except Exception as e:
        logger.error(f"PostgreSQL not available: {e}")
        logger.info("Skipping PostgreSQL example")


if __name__ == "__main__":
    # Run DuckDB example (works out of the box)
    example_duckdb()
    
    # Uncomment to try PostgreSQL (requires setup)
    # example_postgresql()

