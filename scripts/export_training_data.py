"""Export the exact dataset used by model_paper_ratios.py so the user can inspect it."""
import pandas as pd

hist = pd.read_csv(r'c:\Users\majtob\Desktop\final_project\mxa1438\data\processed\historical_fitch_ratings.csv')
hist = hist[hist['has_financials'] == True].copy()
hist['dataset'] = 'historical_fitch'

exp = pd.read_csv(r'c:\Users\majtob\Desktop\final_project\mxa1438\data\processed\expanded_ratings_financials.csv')
exp = exp[exp['has_financials'] == True].copy()
exp['dataset'] = 'expanded_multiagency'

orig = pd.read_csv(r'c:\Users\majtob\Desktop\final_project\mxa1438\data\processed\ratings_with_financials.csv')
orig['rating_agency'] = 'Tassnief'
orig['dataset'] = 'original_tassnief'
orig['fiscal_year'] = 2024

combined = pd.concat([hist, exp, orig], ignore_index=True)

excluded = ['Banks', 'Insurance', 'Financial Services']
combined = combined[~combined['sector'].isin(excluded)]

ratios = ['liquid', 'cumprof', 'profitab', 'leverage']
combined = combined.dropna(subset=ratios)

combined = combined.drop_duplicates(subset=['ticker', 'fiscal_year', 'rating_agency'])

# Select readable columns
cols = ['ticker', 'company_name', 'sector', 'rating_agency', 'rating', 'fiscal_year',
        'liquid', 'cumprof', 'profitab', 'leverage', 'dataset']
out = combined[[c for c in cols if c in combined.columns]].copy()
out = out.sort_values(['ticker', 'fiscal_year', 'rating_agency'])

out.to_csv(r'c:\Users\majtob\Desktop\final_project\mxa1438\data\processed\model_training_data.csv', index=False)

print(f"Saved {len(out)} records to data/processed/model_training_data.csv")
print(f"Unique companies: {out['ticker'].nunique()}")
print(f"\nBreakdown by dataset:")
print(out['dataset'].value_counts().to_string())
print(f"\nBreakdown by agency:")
print(out['rating_agency'].value_counts().to_string())
print(f"\nAll rows:")
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 15)
print(out[['ticker', 'company_name', 'rating_agency', 'rating', 'fiscal_year', 'dataset']].to_string())
