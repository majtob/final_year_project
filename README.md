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

### 3. Initialize Data Structure

```bash
python scripts/init_data.py
```

### 4. Collect Data

```bash
# Collect market data
python scripts/sync_market_data.py --tickers 2222.SR 1111.SR

# Collect filings (after populating inventory)
python src/pipelines/ingest_filings.py --inventory data/data_inventory.json

# Collect news
python src/pipelines/collect_news.py --inventory data/data_inventory.json

# Collect Tassnief ratings
python src/pipelines/collect_ratings.py --inventory data/data_inventory.json
```

### 5. Extract Features

```bash
python src/pipelines/extract_features.py --input data/processed --output data/features
```

### 6. Train Model

```bash
python src/models/train_classifier.py --features data/features/features.parquet --output models/
```

### 7. Generate Verdicts

```bash
python src/models/generate_verdicts.py --model models/xgboost_model.pkl --output results/verdicts/
```

### 8. Evaluate

```bash
python src/evaluation/evaluate_model.py --predictions results/predictions.csv --verdicts results/verdicts/
```

## Project Structure

```
mxa1438/
├── data/
│   ├── raw/                    # Original filings (PDFs, HTML, XLS)
│   ├── interim/                # Parsed text, extracted tables
│   ├── processed/              # Clean company-period data
│   ├── market/                 # yfinance market data
│   ├── news/                   # News articles and sentiment
│   ├── ratings/                # Tassnief ratings
│   └── features/               # Extracted feature vectors
│
├── src/
│   ├── config/                 # Configuration and schemas
│   ├── pipelines/              # Data collection and processing
│   │   ├── ingest_filings.py   # Download Tadawul filings
│   │   ├── collect_news.py     # Fetch news from APIs
│   │   ├── collect_ratings.py  # Collect Tassnief ratings
│   │   ├── extract_kams.py     # Extract Key Audit Matters
│   │   └── extract_features.py # Feature engineering pipeline
│   ├── models/
│   │   ├── train_classifier.py # Train XGBoost model
│   │   ├── generate_verdicts.py# LLM verdict generation
│   │   └── prompts.py          # Prompt templates
│   ├── evaluation/
│   │   ├── evaluate_model.py   # ML model evaluation
│   │   ├── evaluate_verdicts.py# Verdict quality assessment
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/
│       ├── io.py               # Data I/O helpers
│       ├── logging.py          # Logging configuration
│       └── validation.py       # Schema validation
│
├── models/                     # Trained model artifacts
├── results/                    # Predictions and verdicts
├── notebooks/                  # Exploration notebooks
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── logs/                       # Pipeline logs
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

### Financial Ratios
- Liquidity: Current ratio, quick ratio
- Leverage: Debt/equity, debt/assets
- Coverage: Interest coverage, DSCR
- Profitability: ROA, ROE, net margin
- Efficiency: Asset turnover
- Growth: Revenue growth

### KAM Features
- KAM count and categories
- Going concern, impairment, revenue recognition flags
- Severity score

### News Features
- Article count, sentiment scores
- Event flags (litigation, expansion, regulatory)

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
black src/ scripts/
flake8 src/ scripts/
```

## License

[Add license]

## Contact

[Add contact info]
