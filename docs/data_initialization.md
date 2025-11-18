
## Data Initialization

### Option 1: Basic Setup (Recommended for Start)
```bash
python scripts/init_data.py
```
This creates:
- Directory structure (`data/raw`, `data/interim`, `data/processed`, `data/market`)
- Data schema definitions (`src/config/data_schema.json`)
- Data inventory template (`data/data_inventory.json`)
- Test fixtures (`tests/fixtures/`)
- Environment template (`.env.example`)

### Option 2: Initialize with DVC (Data Version Control)
```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add data directories to DVC
dvc add data/raw data/interim data/processed data/market

# Commit DVC config
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Option 3: Download Sample Data
```bash
# Run data collection script (once implemented)
python scripts/sync_market_data.py --tickers 1111.SR 2222.SR --start-date 2021-01-01

# Or manually download filings and place in data/raw/
```

### Data Directory Structure
```
data/
├── raw/              # Original Tadawul filings (PDFs, HTML, XLS)
├── interim/          # OCR/text chunks, parsed tables
├── processed/        # Tidy company-period tables, ratio inputs
└── market/           # yfinance extracts (CSV/Parquet)
```

### Next Steps
1. Add your Tadawul filing URLs to `data/data_inventory.json`
2. Run `scripts/sync_market_data.py` to fetch market data
3. Implement `src/pipelines/ingest_filings.py` to download filings
4. Validate schema with `tests/test_schema.py`
