"""Identify and add new companies from multi-agency data to KAMs template."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MULTI_AGENCY = PROJECT_ROOT / "data" / "templates" / "multi_agency_ratings.csv"
KAMS_FILE = PROJECT_ROOT / "data" / "templates" / "kams_priority.csv"

MOODYS_MAP = {
    'Aaa': 'AAA', 'Aa1': 'AA+', 'Aa2': 'AA', 'Aa3': 'AA-',
    'A1': 'A+', 'A2': 'A', 'A3': 'A-',
    'Baa1': 'BBB+', 'Baa2': 'BBB', 'Baa3': 'BBB-',
    'Ba1': 'BB+', 'Ba2': 'BB', 'Ba3': 'BB-',
    'B1': 'B+', 'B2': 'B', 'B3': 'B-',
}

expanded = pd.read_csv(MULTI_AGENCY)
kams = pd.read_csv(KAMS_FILE)
existing_tickers = set(kams['ticker'].unique())

new_rows = []
for _, row in expanded.iterrows():
    if row['ticker'] in existing_tickers:
        continue
    
    # Pick best available rating (prefer international agencies)
    rating = None
    agency_used = None
    for agency_col, agency_name in [('fitch', 'Fitch'), ('sp', 'S&P'), 
                                      ('moodys', 'Moodys'), ('tassnief', 'Tassnief'),
                                      ('financial_analytics', 'Financial Analytics')]:
        val = row.get(agency_col)
        if pd.notna(val) and str(val).strip():
            raw = str(val).strip().split('/')[0].split(' ')[0]
            if agency_col == 'moodys':
                rating = MOODYS_MAP.get(raw, raw)
            else:
                rating = raw
            agency_used = agency_name
            break
    
    if not rating:
        rating = 'N/A'
    
    new_rows.append({
        'ticker': row['ticker'],
        'company_name': row['company_name'],
        'fiscal_year': 2024,
        'rating': rating,
        'kam_going_concern': '',
        'kam_revenue': '',
        'kam_assets': '',
        'kam_liabilities': '',
        'kam_other': '',
        'kam_count': '',
        'kam_descriptions': '',
        'extracted': '',
        'notes': f"{row['sector']} - {agency_used}: {rating}",
    })

new_df = pd.DataFrame(new_rows)
print(f"Existing tickers in KAMs template: {len(existing_tickers)}")
print(f"New companies to add: {len(new_df)}")
print()

for _, r in new_df.iterrows():
    print(f"  {r['ticker']:12s} {r['company_name']:30s} {r['notes']}")

# Append to existing file
combined = pd.concat([kams, new_df], ignore_index=True)
combined.to_csv(KAMS_FILE, index=False)
print(f"\nUpdated {KAMS_FILE}")
print(f"Total rows: {len(combined)}")
