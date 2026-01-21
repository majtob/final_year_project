#!/usr/bin/env python3
"""
Data initialization script for FYP project.
Creates directory structure, sample schemas, and test fixtures.
"""

import os
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def create_directory_structure():
    """Create all required data directories."""
    directories = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/market",
        "notebooks",
        "scripts",
        "src/config",
        "src/pipelines",
        "src/retrieval",
        "src/models",
        "src/evaluation",
        "src/utils",
        "configs",
        "tests/fixtures",
        "docs",
        "logs/pipeline",
        "logs/evaluation",
    ]
    
    for dir_path in directories:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep to preserve empty directories
        (full_path / ".gitkeep").touch(exist_ok=True)
        print(f"✓ Created {dir_path}")


def create_data_schema():
    """Create initial data schema definition."""
    schema = {
        "company_period_schema": {
            "description": "Tidy schema: one record per company-period",
            "fields": {
                "company_name": {"type": "string", "required": True},
                "ticker": {"type": "string", "required": True, "format": "XXXX.SR"},
                "fiscal_year": {"type": "integer", "required": True, "range": [2021, 2024]},
                "fiscal_period": {"type": "string", "required": True, "enum": ["Q1", "Q2", "Q3", "FY"]},
                "reporting_date": {"type": "date", "required": True},
                "source_file_id": {"type": "string", "required": True},
                "source_url": {"type": "string", "required": True},
                "source_section": {"type": "string"},
                "source_page": {"type": "integer"},
                "language": {"type": "string", "enum": ["en", "ar", "both"]},
            },
            "financial_statements": {
                "balance_sheet": {
                    "total_assets": {"type": "float"},
                    "total_liabilities": {"type": "float"},
                    "total_equity": {"type": "float"},
                    "current_assets": {"type": "float"},
                    "current_liabilities": {"type": "float"},
                    "long_term_debt": {"type": "float"},
                },
                "income_statement": {
                    "revenue": {"type": "float"},
                    "net_income": {"type": "float"},
                    "operating_income": {"type": "float"},
                    "interest_expense": {"type": "float"},
                    "ebitda": {"type": "float"},
                },
                "cash_flow": {
                    "operating_cash_flow": {"type": "float"},
                    "investing_cash_flow": {"type": "float"},
                    "financing_cash_flow": {"type": "float"},
                },
            },
            "ratios": {
                "current_ratio": {"type": "float", "description": "Current Assets / Current Liabilities"},
                "debt_to_equity": {"type": "float", "description": "Total Debt / Total Equity"},
                "interest_coverage": {"type": "float", "description": "EBIT / Interest Expense"},
                "return_on_assets": {"type": "float", "description": "Net Income / Total Assets"},
                "net_margin": {"type": "float", "description": "Net Income / Revenue"},
            },
            "market_data": {
                "price": {"type": "float", "description": "Stock price (SAR)"},
                "market_cap": {"type": "float", "description": "Market capitalization (SAR)"},
                "volume": {"type": "float", "description": "Trading volume"},
                "date": {"type": "date"},
            },
        },
        "risk_schema": {
            "description": "Credit risk outputs",
            "fields": {
                "company_name": {"type": "string", "required": True},
                "ticker": {"type": "string", "required": True},
                "fiscal_year": {"type": "integer", "required": True},
                "risk_score": {"type": "float", "range": [0.0, 1.0], "description": "Continuous risk probability"},
                "risk_flag": {"type": "integer", "enum": [0, 1], "description": "0=low risk, 1=elevated risk"},
                "extraction_metadata": {
                    "model_name": {"type": "string"},
                    "prompt_id": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "extraction_timestamp": {"type": "datetime"},
                },
            },
        },
    }
    
    schema_path = PROJECT_ROOT / "src/config/data_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"✓ Created data schema at {schema_path}")


def create_sample_inventory():
    """Create sample data inventory template."""
    inventory = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": "Data inventory for Tadawul filings (2021-2024)",
            "version": "1.0.0",
        },
        "companies": [],
        "filings": [],
        "market_data": {
            "source": "yfinance",
            "tickers": [],
            "coverage_period": {"start": "2021-01-01", "end": "2024-12-31"},
        },
    }
    
    inventory_path = PROJECT_ROOT / "data/data_inventory.json"
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    print(f"✓ Created data inventory template at {inventory_path}")


def create_test_fixtures():
    """Create sample test data fixtures."""
    fixtures_dir = PROJECT_ROOT / "tests/fixtures"
    
    # Sample company data
    sample_company = {
        "company_name": "Sample Company",
        "ticker": "1111.SR",
        "fiscal_year": 2023,
        "fiscal_period": "FY",
        "reporting_date": "2023-12-31",
        "source_file_id": "test_file_001",
        "source_url": "https://example.com/filing.pdf",
        "language": "en",
        "balance_sheet": {
            "total_assets": 1000000.0,
            "total_liabilities": 600000.0,
            "total_equity": 400000.0,
            "current_assets": 300000.0,
            "current_liabilities": 200000.0,
        },
        "income_statement": {
            "revenue": 500000.0,
            "net_income": 50000.0,
            "operating_income": 75000.0,
            "interest_expense": 10000.0,
        },
    }
    
    fixture_path = fixtures_dir / "sample_company_data.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(sample_company, f, indent=2, ensure_ascii=False)
    print(f"✓ Created test fixture at {fixture_path}")


def create_env_example():
    """Create .env.example template."""
    env_example = """# API Keys and Configuration
# Copy this file to .env and fill in your values

# Data Paths
DATA_ROOT=data/
RAW_DATA_PATH=data/raw
PROCESSED_DATA_PATH=data/processed

# Database (if using PostgreSQL)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=tadawul_data
# DB_USER=your_username
# DB_PASSWORD=your_password

# DuckDB (alternative)
DUCKDB_PATH=data/tadawul.duckdb

# LLM Configuration
# LLM_MODEL_NAME=mistral-7b-instruct
# LLM_API_BASE=http://localhost:8000/v1
# LLM_API_KEY=your_api_key_here

# Vector Store
VECTOR_STORE_TYPE=faiss  # Options: faiss, chroma, elasticsearch
VECTOR_STORE_PATH=data/vector_store

# Logging
LOG_LEVEL=INFO
LOG_PATH=logs/

# Tadawul Scraping
TADAWUL_BASE_URL=https://www.tadawul.com.sa
RATE_LIMIT_DELAY=1.0  # seconds between requests
MAX_RETRIES=3

# Market Data (yfinance)
MARKET_DATA_PATH=data/market
"""
    
    env_path = PROJECT_ROOT / ".env.example"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_example)
    print(f"✓ Created .env.example at {env_path}")


def create_readme_data_section():
    """Create data initialization section for README."""
    readme_section = """
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
"""
    
    readme_path = PROJECT_ROOT / "docs/data_initialization.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_section)
    print(f"✓ Created data initialization docs at {readme_path}")


def main():
    """Run all initialization steps."""
    print("Initializing data structure for FYP project...\n")
    
    create_directory_structure()
    print()
    
    create_data_schema()
    create_sample_inventory()
    create_test_fixtures()
    create_env_example()
    create_readme_data_section()
    
    print("\n✅ Data initialization complete!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure")
    print("2. Add Tadawul filing URLs to data/data_inventory.json")
    print("3. Run data collection scripts to populate data/raw/")


if __name__ == "__main__":
    main()

