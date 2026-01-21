# Credit Risk Analysis on Saudi Exchange Financials

Final Year Project: Open-Source LLMs for Credit Risk on Saudi Exchange Financials (2021–2024)

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
conda activate fyp

# If environment doesn't exist, create it
conda env create -f environment.yml
conda activate fyp
```

### 2. Initialize Data Structure

```bash
python scripts/init_data.py
```

This creates:
- Directory structure (`data/raw`, `data/interim`, `data/processed`, `data/market`)
- Data schema definitions
- Data inventory template
- Test fixtures

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
```

### 4. Collect Market Data

```bash
# Collect market data for specific tickers
python scripts/sync_market_data.py --tickers 1111.SR 2222.SR 3333.SR

# Or load from inventory
python scripts/sync_market_data.py --inventory data/data_inventory.json

# With custom date range
python scripts/sync_market_data.py --tickers 1111.SR --start-date 2021-01-01 --end-date 2024-12-31
```

### 5. Collect Tadawul Filings

```bash
# First, populate data/data_inventory.json with filing URLs
# Then run:
python src/pipelines/ingest_filings.py --inventory data/data_inventory.json
```

## Project Structure

```
mxa1438/
├── data/
│   ├── raw/              # Original Tadawul filings (PDFs, HTML, XLS)
│   ├── interim/           # OCR/text chunks, parsed tables
│   ├── processed/         # Tidy company-period tables
│   └── market/            # yfinance market data
├── src/
│   ├── pipelines/         # Data ingestion and processing pipelines
│   ├── retrieval/         # Vector store and retrieval
│   ├── models/            # LLM models and prompts
│   ├── evaluation/        # Evaluation metrics and baselines
│   └── utils/             # Utilities (logging, validation, IO)
├── scripts/               # Standalone scripts
├── notebooks/             # Jupyter notebooks for exploration
├── configs/               # Configuration files
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Data Collection

### Market Data Collection

The `sync_market_data.py` script fetches stock market data from yfinance:

```bash
python scripts/sync_market_data.py \
    --tickers 1111.SR 2222.SR \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --interval 1d \
    --format parquet \
    --verify
```

**Options:**
- `--tickers`: List of ticker symbols (space-separated)
- `--inventory`: Path to data inventory JSON file
- `--output-dir`: Output directory (default: `data/market`)
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--interval`: Data interval (`1d`, `1wk`, `1mo`)
- `--format`: Output format (`parquet` or `csv`)
- `--verify`: Verify data coverage after collection

### Filing Collection

The `ingest_filings.py` script downloads Tadawul company filings:

```bash
python src/pipelines/ingest_filings.py \
    --inventory data/data_inventory.json \
    --output-dir data/raw \
    --rate-limit 1.0 \
    --max-retries 3
```

**Options:**
- `--inventory`: Path to data inventory JSON file
- `--output-dir`: Output directory (default: `data/raw`)
- `--rate-limit`: Delay between requests in seconds
- `--max-retries`: Maximum retry attempts

**Data Inventory Format:**

```json
{
  "companies": [
    {
      "company_name": "Company Name",
      "ticker": "1111.SR"
    }
  ],
  "filings": [
    {
      "url": "https://www.tadawul.com.sa/...",
      "company_name": "Company Name",
      "ticker": "1111.SR",
      "fiscal_year": 2023,
      "filing_type": "annual_report",
      "language": "en",
      "publication_date": "2023-03-15"
    }
  ],
  "market_data": {
    "tickers": ["1111.SR", "2222.SR"]
  }
}
```

## Next Steps

1. **Populate Data Inventory**: Add company tickers and filing URLs to `data/data_inventory.json`
2. **Collect Market Data**: Run `sync_market_data.py` for your target companies
3. **Collect Filings**: Add filing URLs to inventory and run `ingest_filings.py`
4. **Preprocessing**: Implement document preprocessing pipeline
5. **RAG Setup**: Build vector store for retrieval-augmented generation
6. **Extraction**: Implement LLM-based financial extraction

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

## Documentation

- `PROJECT_SKELETON.md`: Project structure and checkpoints
- `FYP_layers.md`: Detailed project requirements
- `docs/DATA_INITIALIZATION_OPTIONS.md`: Data initialization guide

## License

[Add your license here]

## Contact

[Add your contact information]
