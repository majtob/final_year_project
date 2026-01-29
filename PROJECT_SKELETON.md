# Project Skeleton

## Directory Structure

```
mxa1438/
│
├── README.md                      # Quick start, architecture overview
├── FYP_layers.md                  # Detailed project specification
├── LICENSE
├── .env.example                   # API keys placeholders
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/                       # Original Tadawul filings (PDFs, HTML, XLS)
│   ├── interim/                   # Parsed text, extracted tables, KAM text
│   ├── processed/                 # Clean company-period structured data
│   ├── market/                    # yfinance market data (CSV/Parquet)
│   ├── news/                      # News articles and metadata
│   ├── ratings/                   # Tassnief credit ratings
│   ├── features/                  # Extracted feature vectors for ML
│   └── data_inventory.json        # Master inventory of all data sources
│
├── models/                        # Trained model artifacts
│   ├── xgboost_model.pkl          # Trained XGBoost classifier
│   ├── feature_scaler.pkl         # Feature preprocessing scaler
│   └── model_config.json          # Model hyperparameters
│
├── results/
│   ├── predictions/               # ML model predictions
│   ├── verdicts/                  # LLM-generated verdicts (JSON)
│   └── evaluation/                # Evaluation reports and metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Explore raw data and filings
│   ├── 02_feature_analysis.ipynb  # Feature engineering and EDA
│   ├── 03_model_training.ipynb    # Model development and tuning
│   ├── 04_verdict_analysis.ipynb  # LLM verdict quality review
│   └── 05_evaluation.ipynb        # Final evaluation and reporting
│
├── src/
│   ├── config/
│   │   ├── data_schema.json       # Data validation schemas
│   │   ├── feature_config.yaml    # Feature definitions
│   │   └── model_config.yaml      # Model hyperparameters
│   │
│   ├── pipelines/
│   │   ├── ingest_filings.py      # Download Tadawul filings
│   │   ├── collect_news.py        # Fetch news from APIs
│   │   ├── collect_ratings.py     # Collect Tassnief ratings
│   │   ├── parse_financials.py    # Parse financial statements
│   │   ├── extract_kams.py        # Extract Key Audit Matters
│   │   └── extract_features.py    # Feature engineering pipeline
│   │
│   ├── models/
│   │   ├── train_classifier.py    # Train XGBoost/RF classifier
│   │   ├── generate_verdicts.py   # LLM verdict generation
│   │   ├── prompts.py             # Prompt templates for LLM
│   │   └── inference.py           # Model inference utilities
│   │
│   ├── evaluation/
│   │   ├── evaluate_model.py      # ML model evaluation metrics
│   │   ├── evaluate_verdicts.py   # Verdict quality assessment
│   │   ├── metrics.py             # Custom evaluation metrics
│   │   └── feature_importance.py  # SHAP analysis
│   │
│   └── utils/
│       ├── io.py                  # Data I/O helpers
│       ├── logging.py             # Logging configuration
│       ├── validation.py          # Schema validation
│       └── text_processing.py     # Text cleaning utilities
│
├── scripts/
│   ├── init_data.py               # Initialize data directories
│   ├── init_database.py           # Database setup
│   ├── sync_market_data.py        # Fetch yfinance data
│   └── run_pipeline.py            # End-to-end pipeline runner
│
├── tests/
│   ├── test_feature_extraction.py
│   ├── test_model.py
│   ├── test_verdict_schema.py
│   └── fixtures/
│       └── sample_company_data.json
│
├── docs/
│   ├── DATA_STORAGE_GUIDE.md
│   ├── data_initialization.md
│   ├── feature_dictionary.md      # Feature definitions and sources
│   └── evaluation_report.md       # Final evaluation findings
│
├── configs/
│   ├── logging.yaml
│   └── pipeline.yaml
│
└── logs/
    ├── pipeline/                  # Data collection logs
    └── evaluation/                # Model evaluation logs
```

## Pipeline Stages

### Stage 1: Data Collection

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `sync_market_data.py` | Tickers | `data/market/*.parquet` | Fetch yfinance data |
| `ingest_filings.py` | Inventory | `data/raw/` | Download Tadawul filings |
| `collect_news.py` | Inventory | `data/news/` | Fetch news articles |
| `collect_ratings.py` | Inventory | `data/ratings/` | Collect Tassnief ratings |

### Stage 2: Data Processing

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `parse_financials.py` | `data/raw/` | `data/interim/` | Parse financial tables |
| `extract_kams.py` | `data/raw/` | `data/interim/kams/` | Extract KAMs from audits |

### Stage 3: Feature Engineering

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `extract_features.py` | `data/interim/`, `data/news/` | `data/features/` | Create feature vectors |

### Stage 4: Model Training

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `train_classifier.py` | `data/features/` | `models/` | Train ML classifier |

### Stage 5: Verdict Generation

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `generate_verdicts.py` | Features + Model | `results/verdicts/` | Generate LLM verdicts |

### Stage 6: Evaluation

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `evaluate_model.py` | Predictions | `results/evaluation/` | ML metrics |
| `evaluate_verdicts.py` | Verdicts | `results/evaluation/` | Verdict quality |

## Checkpoints

### Data Collection
- [ ] `data/data_inventory.json` populated with companies, filings, ratings
- [ ] `data/raw/` contains downloaded filings
- [ ] `data/news/` contains news articles
- [ ] `data/ratings/` contains Tassnief ratings
- [ ] `data/market/` contains yfinance data

### Feature Engineering
- [ ] Financial ratios extracted and validated
- [ ] KAM features extracted from audit reports
- [ ] News sentiment computed
- [ ] `data/features/features.parquet` contains complete feature matrix

### Model Training
- [ ] XGBoost model trained
- [ ] Cross-validation completed
- [ ] Feature importance computed
- [ ] `models/xgboost_model.pkl` saved

### Verdict Generation
- [ ] LLM prompts tested
- [ ] Verdicts generated for all companies
- [ ] JSON schema validation passed
- [ ] `results/verdicts/` populated

### Evaluation
- [ ] ML accuracy vs Tassnief computed
- [ ] Confusion matrix generated
- [ ] Verdict factual accuracy assessed
- [ ] `docs/evaluation_report.md` completed

## Data Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Tadawul      │────▶│   data/raw/    │────▶│ data/interim/  │
│   Filings      │     │   (PDFs, XLS)  │     │ (parsed text)  │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
┌────────────────┐     ┌────────────────┐            │
│   NewsAPI      │────▶│   data/news/   │────────────┤
│   MarketAux    │     │   (articles)   │            │
└────────────────┘     └────────────────┘            │
                                                      │
┌────────────────┐     ┌────────────────┐            │
│   yfinance     │────▶│  data/market/  │────────────┤
│                │     │  (prices)      │            │
└────────────────┘     └────────────────┘            │
                                                      │
┌────────────────┐     ┌────────────────┐            │
│   Tassnief     │────▶│ data/ratings/  │────────────┤
│                │     │  (labels)      │            ▼
└────────────────┘     └────────────────┘     ┌────────────────┐
                                              │ data/features/ │
                                              │ (ML input)     │
                                              └───────┬────────┘
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       │                              │                              │
                       ▼                              ▼                              ▼
                ┌────────────┐                ┌────────────┐                ┌────────────┐
                │   Train    │                │  Predict   │                │  Generate  │
                │   Model    │───────────────▶│  Ratings   │───────────────▶│  Verdicts  │
                └────────────┘                └────────────┘                └────────────┘
                       │                              │                              │
                       ▼                              ▼                              ▼
                ┌────────────┐                ┌────────────┐                ┌────────────┐
                │  models/   │                │ results/   │                │ results/   │
                │            │                │predictions/│                │ verdicts/  │
                └────────────┘                └────────────┘                └────────────┘
```

## Configuration Files

### data_inventory.json
Master inventory of all companies, filings, ratings, and data sources.

### feature_config.yaml
Defines all features, their sources, and computation methods.

### model_config.yaml
Model hyperparameters and training configuration.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_feature_extraction.py

# Run with coverage
pytest --cov=src tests/
```

## Logging

All pipeline scripts log to `logs/pipeline/` with rotation.
Evaluation logs go to `logs/evaluation/`.

Configure via `configs/logging.yaml` or environment variables.
