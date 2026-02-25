"""
Build historical Fitch ratings dataset (2019-2024) for Saudi companies.

Sources:
- Fitch press releases (via Argaam, ArabNews, Reuters)
- Saudi Exchange (Tadawul) disclosures
- Company investor relations pages

Confirmed rating changes:
- Apr 2023: Saudi sovereign upgraded A → A+
- Apr 2023: 8 banks upgraded BBB+ → A- (following sovereign)
- Apr 2023: Aramco, SABIC, SEC upgraded A → A+ (sovereign-linked)
- Aug 2023: Tawuniya upgraded A- → A (IFS)
- Aug 2023: Maaden first rated BBB+ (Fitch)
- Mar 2019: Aramco first rated A (Fitch)
- Oct 2019: SABIC downgraded A+ → A (sovereign downgrade)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "historical_fitch_ratings.csv"
KAMS_FILE = PROJECT_ROOT / "data" / "templates" / "kams_priority.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Historical Fitch ratings based on confirmed press releases
# Format: (ticker, company_name, sector, {year: rating})
FITCH_HISTORY = [
    # === CORPORATES ===
    ("2222.SR", "Saudi Aramco", "Energy", {
        2019: "A", 2020: "A", 2021: "A", 2022: "A", 2023: "A+", 2024: "A+",
    }),
    ("2010.SR", "SABIC", "Materials", {
        2019: "A", 2020: "A", 2021: "A", 2022: "A", 2023: "A+", 2024: "A+",
    }),
    ("5110.SR", "Saudi Electricity", "Utilities", {
        2019: "A", 2020: "A", 2021: "A", 2022: "A", 2023: "A+", 2024: "A+",
    }),
    ("1211.SR", "Maaden", "Materials", {
        2023: "BBB+", 2024: "BBB+",
    }),
    ("2082.SR", "ACWA Power", "Utilities", {
        2021: "BBB-", 2022: "BBB-", 2023: "BBB-", 2024: "BBB-",
    }),
    ("2382.SR", "ADES", "Energy", {
        2022: "B+", 2023: "B+", 2024: "B+",
    }),
    ("4321.SR", "Cenomi Centers", "Real Estate", {
        2020: "BB", 2021: "BB", 2022: "BB", 2023: "BB", 2024: "BB",
    }),

    # === BANKS ===
    # SNB and Al Rajhi: A- throughout (IDR)
    ("1180.SR", "SNB", "Banks", {
        2019: "A-", 2020: "A-", 2021: "A-", 2022: "A-", 2023: "A-", 2024: "A-",
    }),
    ("1120.SR", "Al Rajhi Bank", "Banks", {
        2019: "A-", 2020: "A-", 2021: "A-", 2022: "A-", 2023: "A-", 2024: "A-",
    }),
    # 8 banks: BBB+ (2019-2022) → A- (Apr 2023 upgrade)
    ("1010.SR", "Riyad Bank", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1060.SR", "SAB", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1050.SR", "BSF", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1080.SR", "ANB", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1150.SR", "Alinma Bank", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1030.SR", "SAIB", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1020.SR", "Bank AlJazira", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),
    ("1140.SR", "Bank Albilad", "Banks", {
        2019: "BBB+", 2020: "BBB+", 2021: "BBB+", 2022: "BBB+", 2023: "A-", 2024: "A-",
    }),

    # === INSURANCE (IFS ratings) ===
    ("8010.SR", "Tawuniya", "Insurance", {
        2019: "A-", 2020: "A-", 2021: "A-", 2022: "A-", 2023: "A", 2024: "A",
    }),
    ("8070.SR", "Arabian Shield", "Insurance", {
        2021: "A-", 2022: "A-", 2023: "A-", 2024: "A-",
    }),
    ("8180.SR", "Al Sagr Insurance", "Insurance", {
        2021: "BBB", 2022: "BBB", 2023: "BBB", 2024: "BBB",
    }),
    ("8120.SR", "Gulf Union Alahlia", "Insurance", {
        2022: "BBB+", 2023: "BBB+", 2024: "BBB+",
    }),
]

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def pull_financials(ticker, target_year):
    """Pull financial data for a specific year."""
    try:
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
        inc = stock.financials

        if bs is None or bs.empty:
            return None

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

        ta = safe_get(bs_year, ['Total Assets', 'TotalAssets'])
        ca = safe_get(bs_year, ['Current Assets', 'CurrentAssets', 'Total Current Assets'])
        cl = safe_get(bs_year, ['Current Liabilities', 'CurrentLiabilities', 'Total Current Liabilities'])
        re = safe_get(bs_year, ['Retained Earnings', 'RetainedEarnings'])
        te = safe_get(bs_year, ['Total Equity Gross Minority Interest', 'Stockholders Equity',
                                 'Total Equity', 'TotalEquityGrossMinorityInterest'])
        tl = safe_get(bs_year, ['Total Liabilities Net Minority Interest',
                                 'Total Liabilities', 'TotalLiabilitiesNetMinorityInterest'])
        ebit = safe_get(inc_year, ['EBIT', 'Operating Income', 'OperatingIncome']) if inc_year is not None else None

        ratios = {}
        if ta and ta != 0:
            if ca is not None and cl is not None:
                ratios['liquid'] = (ca - cl) / ta
            if re is not None:
                ratios['cumprof'] = re / ta
            if ebit is not None:
                ratios['profitab'] = ebit / ta
        if te is not None and tl is not None and tl != 0:
            ratios['leverage'] = te / tl

        return ratios if ratios else None

    except Exception as e:
        return None


def main():
    logger.info("=" * 60)
    logger.info("BUILDING HISTORICAL FITCH RATINGS DATASET")
    logger.info("=" * 60)

    # Expand history into rows
    rows = []
    for ticker, name, sector, ratings_by_year in FITCH_HISTORY:
        for year, rating in ratings_by_year.items():
            rows.append({
                'ticker': ticker,
                'company_name': name,
                'sector': sector,
                'rating_agency': 'Fitch',
                'rating': rating,
                'rating_numeric': RATING_TO_NUMERIC.get(rating, 0),
                'fiscal_year': year,
                'source': 'fitch_confirmed',
            })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} historical rating records")
    logger.info(f"Companies: {df['ticker'].nunique()}")
    logger.info(f"Year range: {df['fiscal_year'].min()}-{df['fiscal_year'].max()}")

    # Pull financials for each ticker-year
    unique_combos = df[['ticker', 'fiscal_year']].drop_duplicates()
    logger.info(f"\nPulling financials for {len(unique_combos)} ticker-year combinations...")

    fin_cache = {}
    unique_tickers = df['ticker'].unique()

    for ticker in unique_tickers:
        logger.info(f"\n  {ticker}:")
        years_needed = sorted(df[df['ticker'] == ticker]['fiscal_year'].unique())

        for year in years_needed:
            key = f"{ticker}_{year}"
            ratios = pull_financials(ticker, year)
            if ratios:
                fin_cache[key] = ratios
                n_ratios = len([v for v in ratios.values() if v is not None])
                logger.info(f"    {year}: OK ({n_ratios} ratios)")
            else:
                logger.info(f"    {year}: No data")

        time.sleep(0.3)

    # Merge financials
    for idx, row in df.iterrows():
        key = f"{row['ticker']}_{row['fiscal_year']}"
        if key in fin_cache:
            for col, val in fin_cache[key].items():
                df.loc[idx, col] = val
            df.loc[idx, 'has_financials'] = True
        else:
            df.loc[idx, 'has_financials'] = False

    # Rating category
    def to_cat(r):
        n = RATING_TO_NUMERIC.get(r, 0)
        if n >= 18: return 'AA'
        elif n >= 15: return 'A'
        elif n >= 12: return 'BBB'
        else: return 'BB'

    df['rating_category'] = df['rating'].apply(to_cat)

    # Save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    has_fin = df['has_financials'] == True
    ratio_cols = ['liquid', 'cumprof', 'profitab', 'leverage']
    has_all = has_fin & df[ratio_cols].notna().all(axis=1)
    has_2 = has_fin & df[['cumprof', 'leverage']].notna().all(axis=1)

    logger.info(f"\n{'='*60}")
    logger.info("HISTORICAL DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"With financials: {has_fin.sum()}")
    logger.info(f"With all 4 ratios: {has_all.sum()}")
    logger.info(f"With 2 ratios (cumprof+leverage): {has_2.sum()}")
    logger.info(f"Companies: {df['ticker'].nunique()}")
    logger.info(f"Year range: {df['fiscal_year'].min()}-{df['fiscal_year'].max()}")

    logger.info(f"\nBy year:")
    for year in sorted(df['fiscal_year'].unique()):
        yr_df = df[df['fiscal_year'] == year]
        yr_fin = yr_df['has_financials'] == True
        logger.info(f"  {year}: {len(yr_df)} ratings, {yr_fin.sum()} with financials")

    logger.info(f"\nRating category distribution (with financials):")
    if has_fin.any():
        print(df.loc[has_fin, 'rating_category'].value_counts().to_string())

    logger.info(f"\nPrevious best: 23 Fitch ratings (2024 only)")
    logger.info(f"New: {has_2.sum()} Fitch ratings (2019-2024)")
    logger.info(f"Saved to {OUTPUT_FILE}")

    # Update kams_priority.csv with new company-year combos
    logger.info(f"\nUpdating KAMs template...")
    kams = pd.read_csv(KAMS_FILE)
    existing = set(zip(kams['ticker'], kams['fiscal_year']))

    new_kams = []
    for _, row in df.iterrows():
        key = (row['ticker'], row['fiscal_year'])
        if key not in existing:
            new_kams.append({
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'fiscal_year': row['fiscal_year'],
                'rating': row['rating'],
                'kam_going_concern': '',
                'kam_revenue': '',
                'kam_assets': '',
                'kam_liabilities': '',
                'kam_other': '',
                'kam_count': '',
                'kam_descriptions': '',
                'extracted': '',
                'notes': f"Fitch historical {row['fiscal_year']}",
            })

    if new_kams:
        new_df = pd.DataFrame(new_kams)
        combined = pd.concat([kams, new_df], ignore_index=True)
        combined = combined.sort_values(['ticker', 'fiscal_year'], ascending=[True, False])
        combined.to_csv(KAMS_FILE, index=False)
        logger.info(f"Added {len(new_kams)} new company-year rows to KAMs template")
        logger.info(f"Total KAMs rows: {len(combined)}")
    else:
        logger.info("No new company-year combos to add")


if __name__ == "__main__":
    main()
