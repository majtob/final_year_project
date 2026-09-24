# Credit Rating Prediction for Saudi Exchange Companies

**Final Year Project**: Classical ML for Credit Rating Prediction + LLM Verdict Generation

Predict Tassnief credit ratings for Saudi Tadawul-listed companies using public data (financials, KAMs, news), with open-source LLMs generating qualitative credit verdicts.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│  Financials │    KAMs     │    News     │  Tassnief Ratings   │
│  (Tadawul)  │  (Filings)  │ (NewsAPI)   │  (Ground Truth)     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │                 │
       ▼             ▼             ▼                 │
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                           │
│  • Financial ratios (liquidity, leverage, coverage, profit)    │
│  • KAM features (categories, severity)                         │
│  • News features (sentiment, events)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML CLASSIFIER (XGBoost)                      │
│  Input: Feature vector (~30-50 features)                       │
│  Output: Predicted Tassnief rating + confidence                │
│  Target: Tassnief ratings (ground truth)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LLM VERDICT GENERATOR                           │
│  Input: Features + ML prediction                               │
│  Model: Mistral / Llama (open-source)                          │
│  Output: Structured credit verdict (JSON)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION                                 │
│  • ML: Accuracy, F1, Spearman correlation vs Tassnief          │
│  • Verdict: Factual accuracy, coherence, completeness          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Run the Streamlit demo with Docker (examiner-friendly, CPU-only)

```bash
docker compose up --build
# Open http://localhost:8501
```

Uses a slim image (no GPU / no PyTorch). See **[docs/DOCKER.md](docs/DOCKER.md)** for details and troubleshooting.

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate fyp

# Or use pip
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (NewsAPI, etc.)
```

### 3. Rebuild processed datasets

From the repo root (`mxa1438/`), ensure templates and source CSVs are in place (`data/templates/kams_priority.csv`, `data/processed/ratings_with_financials.csv`, `data/processed/ratings_financials_sentiment.csv` as applicable), then:

```bash
python scripts/rebuild_processed_datasets.py
```

Optional: add `--yfinance` to refresh annual financials from Yahoo Finance into `ratings_with_financials.csv` before slicing ratios.

To refresh **news sentiment** from cached `data/raw/news/*_news.json` with **FinBERT** (updates `ratings_financials_sentiment.csv`, then rebuild as above):

```bash
python scripts/collect_news_sentiment_finbert.py
```

See `docs/FINBERT_NEWS_SENTIMENT.md`.

To **fill missing raw news** for every `(ticker, fiscal_year)` in `kams_processed.csv` (MarketAux + optional ticker-alias copies, e.g. `2070.SR` → `7204.SR`):

```bash
# Optional: see gaps without writing
python scripts/fetch_kams_news_marketaux.py --dry-run

export MARKETAUX_API_KEY="your_token"   # or put MARKETAUX_API_KEY in mxa1438/.env
python scripts/fetch_kams_news_marketaux.py

# If some JSON files exist but have zero articles (rate limits, etc.):
python scripts/fetch_kams_news_marketaux.py --refetch-empty
```

Then FinBERT + rebuild as above. **Never commit API keys**; add `mxa1438/.env` to `.gitignore` if you use it (already ignored here).

**Figures missing on Yahoo:** add rows to `data/templates/saudiexchange_financial_supplement.csv` (amounts in the same currency as the annual report, usually SAR). Use the same `ticker` and `fiscal_year` as in `ratings_with_financials.csv`. Columns: `total_assets`, `total_liabilities`, `total_equity`, `current_assets`, `current_liabilities`, `retained_earnings`, `operating_profit` (EBIT or operating income), optional `statement_period_end`, `yfinance_sector`, `yfinance_industry`, `source_url`, `notes`. Rebuild recalculates `liquid`, `cumprof`, `profitab`, `leverage` and merges them in. You can use `ebit` instead of `operating_profit` if you prefer.

Rebuild also drops `outlook`, `rating_action`, and `data_available` from `ratings_with_financials.csv` (they are not used for modeling).

This writes:

- `data/processed/kams_processed.csv`
- `data/processed/financial_ratios_processed.csv`
- `data/processed/news_features_processed.csv`
- `data/processed/merged_multisource_training.csv` (inner join used by the app and LLM prep)

### 4. Train XGBoost (multisource)

```bash
python models/xgboost_with_kams.py    # financial ratios + KAM dummies
python models/xgboost_all_features.py # + news + full KAM/firm block (21 features)
```

On macOS, XGBoost may require `brew install libomp`.

### 5. Streamlit demo

```bash
streamlit run app.py
```

Reads `merged_multisource_training.csv` (rows with complete ratio columns).

### 6. LLM verdicts and QLoRA (optional, local GPU)

```bash
python models/prepare_finetune_data.py
python models/finetune_qwen.py
python models/evaluate_finetune.py
```

See `docs/DOCKER.md` for a CPU-only container that runs `app.py` without PyTorch.

## Project Structure

```
mxa1438/
├── app.py                      # Streamlit demo
├── data/
│   ├── templates/              # kams_priority, ratings_scraped, ticker_mapping, …
│   ├── processed/              # *_processed.csv, merged_multisource_training.csv
│   └── raw/                    # financials/, news/ caches
├── scripts/
│   ├── rebuild_processed_datasets.py
│   ├── collect_news_sentiment_finbert.py
│   └── fetch_kams_news_marketaux.py
├── models/
│   ├── xgboost_with_kams.py
│   ├── xgboost_all_features.py
│   ├── llm_verdict.py
│   ├── prepare_finetune_data.py
│   ├── finetune_qwen.py
│   ├── evaluate_finetune.py
│   ├── improve_verdicts.py
│   └── lora_adapter/
├── results/                    # JSON metrics, verdicts/
├── figures/
└── docs/                       # PROJECT_REPORT.md, DOCKER.md, …
```

## Data Sources

| Source | Description | Collection |
|--------|-------------|------------|
| **Tadawul Filings** | Annual reports, financial statements | Web scraping |
| **Key Audit Matters** | Risk indicators from audit reports | Text extraction |
| **News** | Company news and sentiment | NewsAPI / MarketAux |
| **Market Data** | Price, volume, market cap | yfinance |
| **Tassnief Ratings** | Credit ratings (ground truth) | tassnief.com |

## Feature Categories

### Financial ratios (Altman-style, in processed CSVs)
- `liquid`, `cumprof`, `profitab`, `leverage` (definitions match `app.py` / XGBoost scripts)

### KAM features (paper-aligned dummies)
- Category indicators (e.g. going concern, revenue, assets, liabilities, other) plus counts as built in `kams_processed.csv`

### News features
- `sentiment_mean`, `sentiment_std`, positive/negative proportions, `news_count` from **FinBERT** aggregates (see `news_features_processed.csv`, `docs/FINBERT_NEWS_SENTIMENT.md`)

## Evaluation Metrics

### ML Model
- Accuracy, Macro F1, Weighted F1
- Spearman correlation (ordinal ranking)
- Investment grade vs speculative F1

### LLM Verdicts
- Factual accuracy (citations correct)
- Completeness (covers key factors)
- Coherence (logical reasoning)

## Configuration

### Data Inventory (`data/data_inventory.json`)

```json
{
  "companies": [
    {
      "company_name": "Saudi Aramco",
      "ticker": "2222.SR",
      "sector": "Energy",
      "tassnief_rated": true
    }
  ],
  "filings": [...],
  "ratings": [
    {
      "ticker": "2222.SR",
      "rating_date": "2023-06-15",
      "rating": "A+",
      "outlook": "Stable"
    }
  ],
  "news_config": {
    "sources": ["newsapi", "marketaux"],
    "date_range": {"start": "2021-01-01", "end": "2024-12-31"}
  }
}
```

## Documentation

- `FYP_layers.md` - Detailed project specification
- `PROJECT_SKELETON.md` - Project structure and checkpoints
- `docs/DATA_STORAGE_GUIDE.md` - Data storage conventions
- `docs/data_initialization.md` - Data setup instructions

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black models/ scripts/ app.py
flake8 models/ scripts/ app.py
```

## License

[Add license]

## Contact

[Add contact info]
