#!/usr/bin/env python3
"""
Merge scraped ratings with ticker mapping.
Adds ticker symbols to scraped ratings based on company name matching.

Usage:
    python scripts/merge_ratings_tickers.py
"""

import sys
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).parent.parent


def similar(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(company_name: str, mapping_df: pd.DataFrame, threshold: float = 0.6) -> str:
    """
    Find best matching ticker for a company name.
    
    Args:
        company_name: Company name to match
        mapping_df: DataFrame with company_name and ticker columns
        threshold: Minimum similarity threshold
        
    Returns:
        Ticker symbol or empty string if no match
    """
    best_match = ""
    best_score = 0
    
    company_name_lower = company_name.lower()
    
    for _, row in mapping_df.iterrows():
        mapping_name = str(row["company_name"]).lower()
        
        # Exact match
        if company_name_lower == mapping_name:
            return row["ticker"]
        
        # Check if one contains the other
        if company_name_lower in mapping_name or mapping_name in company_name_lower:
            return row["ticker"]
        
        # Fuzzy match
        score = similar(company_name, row["company_name"])
        if score > best_score and score >= threshold:
            best_score = score
            best_match = row["ticker"]
    
    return best_match


def main():
    templates_dir = PROJECT_ROOT / "data" / "templates"
    
    # Load files
    scraped_path = templates_dir / "ratings_scraped.csv"
    mapping_path = templates_dir / "ticker_mapping.csv"
    output_path = templates_dir / "ratings_with_tickers.csv"
    
    if not scraped_path.exists():
        print(f"Error: {scraped_path} not found")
        print("Run scrape_tassnief.py first")
        return
    
    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found")
        return
    
    # Load data
    ratings_df = pd.read_csv(scraped_path)
    mapping_df = pd.read_csv(mapping_path)
    
    print(f"Loaded {len(ratings_df)} ratings")
    print(f"Loaded {len(mapping_df)} ticker mappings")
    print()
    
    # Match tickers
    matched = 0
    unmatched = []
    
    for idx, row in ratings_df.iterrows():
        company_name = row["company_name"]
        
        # Skip if already has ticker
        if pd.notna(row.get("ticker")) and row["ticker"]:
            matched += 1
            continue
        
        # Find match
        ticker = find_best_match(company_name, mapping_df)
        
        if ticker:
            ratings_df.at[idx, "ticker"] = ticker
            matched += 1
            print(f"✓ Matched: {company_name} → {ticker}")
        else:
            unmatched.append(company_name)
            print(f"✗ No match: {company_name}")
    
    # Save result
    ratings_df.to_csv(output_path, index=False)
    print()
    print(f"Saved to: {output_path}")
    print(f"Matched: {matched}/{len(ratings_df)}")
    print(f"Unmatched: {len(unmatched)}")
    
    if unmatched:
        print()
        print("Unmatched companies (add to ticker_mapping.csv):")
        for name in set(unmatched):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
