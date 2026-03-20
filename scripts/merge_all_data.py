"""
Merge all rating datasets into one comprehensive training dataset.

Sources:
1. model_training_data.csv (existing 60 records)
2. tadawul_ratings_clean.csv (new records from Tadawul/Tassnief)

Filters: non-financial companies, all 4 Altman ratios present.
Removes duplicates on (ticker, fiscal_year, rating_agency).
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

FINANCIAL_TICKERS = {
    '1010.SR', '1020.SR', '1050.SR', '1060.SR', '1080.SR', '1120.SR',
    '1140.SR', '1150.SR', '1180.SR', '1182.SR',
    '8010.SR', '8012.SR', '8020.SR', '8030.SR', '8040.SR', '8050.SR',
    '8060.SR', '8070.SR', '8100.SR', '8120.SR', '8150.SR', '8160.SR',
    '8170.SR', '8180.SR', '8190.SR', '8200.SR', '8210.SR', '8230.SR',
    '8240.SR', '8250.SR', '8260.SR', '8270.SR', '8280.SR', '8300.SR',
    '8310.SR', '8311.SR',
    '8141.SR',
    '4291.SR',
}

PAPER_RATIOS = ['liquid', 'cumprof', 'profitab', 'leverage']

RATING_TO_NUMERIC = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
    'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12,
    'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6,
}


def to_binary(rating):
    num = RATING_TO_NUMERIC.get(rating, 0)
    return 0 if num >= 15 else 1


def main():
    print("=" * 70)
    print("MERGING ALL RATING DATA")
    print("=" * 70)

    existing = pd.read_csv(PROCESSED / "model_training_data.csv")
    print(f"\nExisting data: {len(existing)} records, {existing['ticker'].nunique()} companies")

    tadawul_path = PROCESSED / "tadawul_ratings_clean.csv"
    if tadawul_path.exists():
        tadawul = pd.read_csv(tadawul_path)
        print(f"Tadawul data:  {len(tadawul)} records, {tadawul['ticker'].nunique()} companies")
    else:
        print("No Tadawul data found.")
        return

    col_map = {'agency': 'rating_agency'}
    if 'agency' in tadawul.columns:
        tadawul = tadawul.rename(columns=col_map)

    common_cols = ['ticker', 'company_name', 'rating_agency', 'rating',
                   'fiscal_year'] + PAPER_RATIOS
    for col in common_cols:
        if col not in existing.columns:
            existing[col] = np.nan
        if col not in tadawul.columns:
            tadawul[col] = np.nan

    combined = pd.concat([existing[common_cols], tadawul[common_cols]], ignore_index=True)

    combined = combined.dropna(subset=PAPER_RATIOS)

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=['ticker', 'fiscal_year', 'rating_agency', 'rating'],
        keep='first'
    )
    print(f"\nAfter dedup: {len(combined)} (removed {before - len(combined)} duplicates)")

    non_fin = combined[~combined['ticker'].isin(FINANCIAL_TICKERS)].copy()
    print(f"After removing financials: {len(non_fin)} records")

    non_fin = non_fin[non_fin[PAPER_RATIOS].notna().all(axis=1)]
    print(f"With all 4 ratios: {len(non_fin)} records")

    non_fin['binary_target'] = non_fin['rating'].apply(to_binary)

    print(f"\n{'='*70}")
    print(f"MERGED DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total records: {len(non_fin)}")
    print(f"Unique companies: {non_fin['ticker'].nunique()}")
    print(f"Year range: {int(non_fin['fiscal_year'].min())} - {int(non_fin['fiscal_year'].max())}")

    print(f"\nBy agency:")
    for agency, count in non_fin['rating_agency'].value_counts().items():
        print(f"  {agency}: {count}")

    print(f"\nBinary class distribution:")
    b0 = (non_fin['binary_target'] == 0).sum()
    b1 = (non_fin['binary_target'] == 1).sum()
    print(f"  A-ratings (0): {b0} ({b0/len(non_fin)*100:.0f}%)")
    print(f"  B/C-ratings (1): {b1} ({b1/len(non_fin)*100:.0f}%)")

    print(f"\nRating distribution:")
    for rating, count in non_fin['rating'].value_counts().sort_index().items():
        print(f"  {rating}: {count}")

    print(f"\nNew companies added:")
    existing_tickers = set(existing['ticker'].unique())
    new_tickers = set(non_fin['ticker'].unique()) - existing_tickers
    for t in sorted(new_tickers):
        rows = non_fin[non_fin['ticker'] == t]
        print(f"  {t}: {rows['company_name'].iloc[0]} "
              f"({len(rows)} records, {rows['rating'].iloc[0]})")

    out_path = PROCESSED / "model_training_data_v2.csv"
    non_fin.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return non_fin


if __name__ == "__main__":
    main()
