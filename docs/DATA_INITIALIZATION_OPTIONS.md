# Data Initialization Options

This document outlines all available options for initializing your data for the credit risk analysis project.

## Overview

Your project requires data from two main sources:
1. **Tadawul Filings**: Saudi Exchange company filings (annual reports, financial statements) for 2021-2024
2. **Market Data**: Stock prices, volumes, market cap from yfinance

## Option 1: Basic Directory Structure Setup ✅ (Recommended First Step)

**What it does:**
- Creates all required data directories
- Sets up schema definitions
- Creates data inventory template
- Generates test fixtures

**How to use:**
```bash
python scripts/init_data.py
```

**Output:**
- `data/raw/` - For original filings
- `data/interim/` - For processed text/tables
- `data/processed/` - For tidy structured data
- `data/market/` - For yfinance market data
- `src/config/data_schema.json` - Data schema definitions
- `data/data_inventory.json` - Inventory template
- `.env.example` - Environment configuration template

## Option 2: DVC (Data Version Control) Setup

**What it does:**
- Tracks data files with version control
- Enables data lineage and reproducibility
- Prevents large files from being committed to git

**How to use:**
```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add data directories (after they have some data)
dvc add data/raw data/interim data/processed data/market

# Commit DVC configuration
git add .dvc .dvcignore
git commit -m "Initialize DVC for data versioning"
```

**When to use:**
- When you start collecting real data
- When you need to track data changes over time
- For collaboration and reproducibility

## Option 3: Sample/Test Data Initialization

**What it does:**
- Creates minimal test fixtures for development
- Allows testing pipelines without real data
- Provides schema validation examples

**How to use:**
```bash
# Test fixtures are created by init_data.py
# Located in: tests/fixtures/sample_company_data.json

# Use in tests:
python -m pytest tests/test_schema.py
```

**Test Data Includes:**
- Sample company financial data
- Mock balance sheet and income statement
- Example ratios for validation

## Option 4: Manual Data Collection Setup

**What it does:**
- Prepare for downloading Tadawul filings
- Set up market data collection

**Steps:**

1. **Create Data Inventory:**
   ```bash
   # Edit data/data_inventory.json
   # Add company tickers, filing URLs, dates
   ```

2. **Market Data Collection:**
   ```bash
   # Once sync_market_data.py is implemented:
   python scripts/sync_market_data.py \
     --tickers 1111.SR 2222.SR 3333.SR \
     --start-date 2021-01-01 \
     --end-date 2024-12-31
   ```

3. **Filing Collection:**
   ```bash
   # Once ingest_filings.py is implemented:
   python src/pipelines/ingest_filings.py \
     --inventory data/data_inventory.json \
     --output data/raw/
   ```

## Option 5: Database Initialization (DuckDB/PostgreSQL)

**DuckDB (Recommended for Start):**
```python
# Create database initialization script
import duckdb

conn = duckdb.connect('data/tadawul.duckdb')
conn.execute("""
    CREATE TABLE IF NOT EXISTS company_periods (
        company_name VARCHAR,
        ticker VARCHAR,
        fiscal_year INTEGER,
        fiscal_period VARCHAR,
        reporting_date DATE,
        -- Add other fields from schema
    )
""")
```

**PostgreSQL (For Production):**
```bash
# Create database
createdb tadawul_data

# Run migrations
psql tadawul_data < migrations/001_initial_schema.sql
```

## Option 6: Schema Validation Setup

**What it does:**
- Validates data against defined schemas
- Ensures data quality before processing

**How to use:**
```bash
# Install validation libraries
pip install pandera pydantic

# Validate data
python -m src.utils.validation --input data/processed/ --schema src/config/data_schema.json
```

## Recommended Workflow

1. **Start Here:** Run `python scripts/init_data.py` ✅
2. **Configure:** Copy `.env.example` to `.env` and set paths
3. **Inventory:** Populate `data/data_inventory.json` with your target companies
4. **Collect Market Data:** Run market data sync script
5. **Collect Filings:** Implement and run filing ingestion pipeline
6. **Version Control:** Initialize DVC when you have real data
7. **Validate:** Run schema validation tests

## Data Sources

### Tadawul Filings
- **Source:** https://www.tadawul.com.sa
- **Types:** Annual reports, financial statements, board reports
- **Formats:** PDF, HTML, XLS
- **Period:** 2021-2024
- **Languages:** English, Arabic

### Market Data
- **Source:** yfinance (Yahoo Finance)
- **Tickers:** Format `XXXX.SR` (e.g., `2222.SR`)
- **Data:** Price, volume, market cap
- **Frequency:** Daily or quarterly

## Next Steps After Initialization

1. ✅ Directory structure created
2. ⏭️ Implement `src/pipelines/ingest_filings.py`
3. ⏭️ Implement `scripts/sync_market_data.py`
4. ⏭️ Create preprocessing pipeline
5. ⏭️ Build vector store for RAG
6. ⏭️ Implement extraction pipeline

## Troubleshooting

**Issue:** Directories not created
- **Solution:** Check write permissions, run from project root

**Issue:** Schema validation fails
- **Solution:** Review `src/config/data_schema.json`, ensure data matches schema

**Issue:** DVC not tracking files
- **Solution:** Ensure files exist before running `dvc add`

## Questions?

Refer to:
- `PROJECT_SKELETON.md` - Project structure
- `FYP_layers.md` - Detailed requirements
- `src/config/data_schema.json` - Data schema definitions

