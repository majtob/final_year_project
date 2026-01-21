"""
Script to pull all financial statements for Saudi Aramco (2222.SR) using yfinance.
"""

import yfinance as yf
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Aramco ticker symbol on Tadawul (Saudi Stock Exchange)
TICKER = "2222.SR"
OUTPUT_DIR = Path(__file__).parent

def pull_financial_statements(ticker_symbol):
    """
    Pull all financial statements for a given ticker symbol.
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., "2222.SR")
    
    Returns:
        dict: Dictionary containing all financial statements
    """
    print(f"Fetching financial data for {ticker_symbol}...")
    
    # Create ticker object
    stock = yf.Ticker(ticker_symbol)
    
    # Get company info
    info = stock.info
    print(f"\nCompany: {info.get('longName', 'N/A')}")
    print(f"Symbol: {info.get('symbol', 'N/A')}")
    
    # Pull financial statements
    financial_data = {}
    
    # 1. Income Statement (Annual)
    print("\nFetching Annual Income Statement...")
    try:
        income_statement_annual = stock.financials
        financial_data['income_statement_annual'] = income_statement_annual
        print(f"✓ Annual Income Statement: {income_statement_annual.shape}")
    except Exception as e:
        print(f"✗ Error fetching Annual Income Statement: {e}")
        financial_data['income_statement_annual'] = None
    
    # 2. Balance Sheet (Annual)
    print("Fetching Annual Balance Sheet...")
    try:
        balance_sheet_annual = stock.balance_sheet
        financial_data['balance_sheet_annual'] = balance_sheet_annual
        print(f"✓ Annual Balance Sheet: {balance_sheet_annual.shape}")
    except Exception as e:
        print(f"✗ Error fetching Annual Balance Sheet: {e}")
        financial_data['balance_sheet_annual'] = None
    
    # 3. Cash Flow Statement (Annual)
    print("Fetching Annual Cash Flow Statement...")
    try:
        cashflow_annual = stock.cashflow
        financial_data['cashflow_annual'] = cashflow_annual
        print(f"✓ Annual Cash Flow Statement: {cashflow_annual.shape}")
    except Exception as e:
        print(f"✗ Error fetching Annual Cash Flow Statement: {e}")
        financial_data['cashflow_annual'] = None
    
    # 4. Earnings History
    print("Fetching Earnings History...")
    try:
        earnings_history = stock.earnings_history
        financial_data['earnings_history'] = earnings_history
        print(f"✓ Earnings History: {earnings_history.shape if hasattr(earnings_history, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"✗ Error fetching Earnings History: {e}")
        financial_data['earnings_history'] = None
    
    # 5. Earnings Dates
    print("Fetching Earnings Dates...")
    try:
        earnings_dates = stock.earnings_dates
        financial_data['earnings_dates'] = earnings_dates
        print(f"✓ Earnings Dates: {earnings_dates.shape if earnings_dates is not None else 'N/A'}")
    except Exception as e:
        print(f"✗ Error fetching Earnings Dates: {e}")
        financial_data['earnings_dates'] = None
    
    return financial_data, info


def convert_dataframe_to_dict(df):
    """
    Convert pandas DataFrame to JSON-serializable dictionary.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
    
    Returns:
        dict: Dictionary representation of the DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Convert DataFrame to dict with dates as strings
    # Transpose so dates are keys (columns become rows)
    df_dict = df.to_dict(orient='index')
    
    # Convert index (dates) to strings
    result = {}
    for date, row_data in df_dict.items():
        date_str = str(date) if pd.notna(date) else None
        # Convert values, handling NaN
        clean_row = {}
        for key, value in row_data.items():
            if pd.isna(value):
                clean_row[str(key)] = None
            else:
                # Convert numpy types to Python native types
                if hasattr(value, 'item'):
                    clean_row[str(key)] = value.item()
                else:
                    clean_row[str(key)] = value
        result[date_str] = clean_row
    
    return result


def save_financial_statements(financial_data, company_info, ticker_symbol, output_dir):
    """
    Save all financial statements to a single JSON file.
    
    Args:
        financial_data (dict): Dictionary containing financial statements
        company_info (dict): Company information
        ticker_symbol (str): Stock ticker symbol
        output_dir (Path): Directory to save file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean ticker symbol for filename
    clean_ticker = ticker_symbol.replace(".", "_")
    filename = f"{clean_ticker}_financial_statements.json"
    filepath = output_dir / filename
    
    print(f"\nSaving all financial statements to {filename}...")
    
    # Prepare JSON structure
    json_data = {
        "ticker": ticker_symbol,
        "company_info": company_info,
        "fetch_date": datetime.now().isoformat(),
        "financial_statements": {}
    }
    
    # Convert all DataFrames to dictionaries
    for statement_name, statement_data in financial_data.items():
        if statement_data is not None:
            if isinstance(statement_data, pd.DataFrame):
                json_data["financial_statements"][statement_name] = convert_dataframe_to_dict(statement_data)
                print(f"✓ Converted: {statement_name}")
            elif isinstance(statement_data, pd.Series):
                # Handle Series (like earnings_history might be)
                json_data["financial_statements"][statement_name] = statement_data.to_dict()
                print(f"✓ Converted: {statement_name} (Series)")
            else:
                # Try to convert to dict if possible
                try:
                    json_data["financial_statements"][statement_name] = dict(statement_data) if statement_data else None
                    print(f"✓ Converted: {statement_name}")
                except:
                    json_data["financial_statements"][statement_name] = str(statement_data)
                    print(f"✓ Converted: {statement_name} (as string)")
        else:
            json_data["financial_statements"][statement_name] = None
            print(f"✗ Skipped {statement_name}: No data available")
    
    # Save to JSON file with proper formatting
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Saved all financial statements to: {filename}")
    print(f"  File size: {filepath.stat().st_size / 1024:.2f} KB")


def display_summary(financial_data):
    """
    Display a summary of the financial statements.
    
    Args:
        financial_data (dict): Dictionary containing financial statements
    """
    print("\n" + "="*80)
    print("FINANCIAL STATEMENTS SUMMARY")
    print("="*80)
    
    for statement_name, statement_data in financial_data.items():
        if statement_data is not None and not statement_data.empty:
            print(f"\n{statement_name.upper().replace('_', ' ')}:")
            print("-" * 80)
            if isinstance(statement_data, pd.DataFrame):
                print(statement_data)
            else:
                print(statement_data)
        else:
            print(f"\n{statement_name.upper().replace('_', ' ')}: No data available")


def main():
    """Main function to pull and save all financial statements."""
    # Pull all financial statements
    financial_data, company_info = pull_financial_statements(TICKER)
    
    # Display summary
    display_summary(financial_data)
    
    # Save to single JSON file
    save_financial_statements(financial_data, company_info, TICKER, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("Data collection complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
