#!/usr/bin/env python3
"""
Convert CSV templates to data_inventory.json
Run this after filling out the templates in data/templates/
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def load_csv_safe(path: Path) -> pd.DataFrame:
    """Load CSV, return empty DataFrame if file doesn't exist or is empty."""
    if not path.exists():
        print(f"  Warning: {path.name} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        # Remove example rows (those with "Example" in notes)
        if 'notes' in df.columns:
            df = df[~df['notes'].str.contains('Example', na=False)]
        return df
    except Exception as e:
        print(f"  Error reading {path.name}: {e}")
        return pd.DataFrame()


def main():
    templates_dir = PROJECT_ROOT / "data" / "templates"
    output_path = PROJECT_ROOT / "data" / "data_inventory.json"
    
    print("Converting templates to data_inventory.json...")
    print(f"Templates directory: {templates_dir}")
    
    # Load templates
    companies_df = load_csv_safe(templates_dir / "companies_template.csv")
    ratings_df = load_csv_safe(templates_dir / "ratings_template.csv")
    kams_df = load_csv_safe(templates_dir / "kams_template.csv")
    
    # Build inventory structure
    inventory = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": "Data inventory for Saudi Exchange credit rating prediction (2021-2024)",
            "version": "2.0.0",
            "generated_from": "templates"
        },
        "companies": [],
        "filings": [],
        "ratings": [],
        "kams": [],
        "market_data": {
            "source": "yfinance",
            "tickers": [],
            "coverage_period": {
                "start": "2021-01-01",
                "end": "2024-12-31"
            }
        }
    }
    
    # Process companies
    if len(companies_df) > 0:
        print(f"  Processing {len(companies_df)} companies...")
        for _, row in companies_df.iterrows():
            company = {
                "company_name": row.get("company_name", ""),
                "ticker": row.get("ticker", ""),
                "sector": row.get("sector", ""),
                "tassnief_rated": True
            }
            inventory["companies"].append(company)
            
            # Add to market data tickers
            if row.get("ticker"):
                inventory["market_data"]["tickers"].append(row["ticker"])
    
    # Process ratings
    if len(ratings_df) > 0:
        print(f"  Processing {len(ratings_df)} ratings...")
        for _, row in ratings_df.iterrows():
            rating = {
                "ticker": row.get("ticker", ""),
                "company_name": row.get("company_name", ""),
                "rating_date": str(row.get("rating_date", "")),
                "rating": row.get("rating", ""),
                "outlook": row.get("outlook", "Stable"),
                "rating_type": row.get("rating_type", "issuer"),
                "is_solicited": str(row.get("is_solicited", "yes")).lower() == "yes",
                "rating_basis": row.get("rating_basis", "full"),  # "full" or "pi" (public info)
                "fiscal_year_applicable": int(row.get("fiscal_year_applicable", 2023)),
                "source_url": row.get("source_url", "")
            }
            inventory["ratings"].append(rating)
    
    # Process KAMs
    if len(kams_df) > 0:
        print(f"  Processing {len(kams_df)} KAMs...")
        for _, row in kams_df.iterrows():
            kam = {
                "ticker": row.get("ticker", ""),
                "fiscal_year": int(row.get("fiscal_year", 2023)),
                "kam_title": row.get("kam_title", ""),
                "kam_category": row.get("kam_category", "other"),
                "severity": row.get("severity", "medium"),
                "source_file": row.get("source_file", ""),
                "source_page": int(row.get("source_page", 0)) if pd.notna(row.get("source_page")) else None
            }
            inventory["kams"].append(kam)
    
    # Remove duplicates from tickers
    inventory["market_data"]["tickers"] = list(set(inventory["market_data"]["tickers"]))
    
    # Save inventory
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Companies: {len(inventory['companies'])}")
    print(f"  Ratings: {len(inventory['ratings'])}")
    print(f"  KAMs: {len(inventory['kams'])}")
    print(f"  Tickers for market data: {len(inventory['market_data']['tickers'])}")


if __name__ == "__main__":
    main()
