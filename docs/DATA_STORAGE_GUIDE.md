# Data Storage Guide

This project supports two storage options for structured data:

## Option 1: DuckDB (Recommended for Start) ✅

**Why DuckDB?**
- ✅ No server setup required (file-based)
- ✅ Fast analytical queries
- ✅ Works directly with Pandas
- ✅ Perfect for research projects
- ✅ Single file database (easy to backup/share)

**Setup:**
```bash
# Initialize DuckDB database
python scripts/init_database.py --type duckdb

# Or specify custom path
python scripts/init_database.py --type duckdb --db-path data/my_database.duckdb
```

**Usage:**
```python
from src.utils.io import DataStorage

# Initialize (creates data/tadawul.duckdb)
storage = DataStorage(storage_type="duckdb")

# Create tables
storage.create_tables()

# Save company period data
data = {
    'company_name': 'Saudi Aramco',
    'ticker': '2222.SR',
    'fiscal_year': 2023,
    'fiscal_period': 'FY',
    'reporting_date': '2023-12-31',
    'total_assets': 1000000.0,
    'revenue': 500000.0,
    'net_income': 50000.0
}
storage.save_company_period(data)

# Save market data from DataFrame
import pandas as pd
market_df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02'],
    'close': [100.0, 101.0],
    'volume': [1000000, 1100000]
})
storage.save_market_data(market_df, ticker='2222.SR')

# Query data
df = storage.query("SELECT * FROM company_periods WHERE ticker = '2222.SR'")
print(df)

# Close connection
storage.close()
```

## Option 2: PostgreSQL (For Production/Collaboration)

**Why PostgreSQL?**
- ✅ Better for multiple users/collaboration
- ✅ More robust for production
- ✅ Better concurrent access
- ✅ More features (triggers, functions, etc.)

**Setup:**

1. **Install PostgreSQL:**
   ```bash
   # macOS
   brew install postgresql
   brew services start postgresql
   
   # Linux
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   ```

2. **Create Database:**
   ```bash
   createdb tadawul_data
   ```

3. **Install Python Driver:**
   ```bash
   pip install psycopg2-binary
   ```

4. **Initialize:**
   ```bash
   # Using connection string
   python scripts/init_database.py --type postgresql \
     --connection-string "postgresql://user:password@localhost/tadawul_data"
   
   # Or set environment variable
   export DATABASE_URL="postgresql://user:password@localhost/tadawul_data"
   python scripts/init_database.py --type postgresql
   ```

**Usage:**
```python
from src.utils.io import DataStorage

# Initialize PostgreSQL
storage = DataStorage(
    storage_type="postgresql",
    connection_string="postgresql://user:password@localhost/tadawul_data"
)

# Same API as DuckDB
storage.create_tables()
storage.save_company_period(data)
storage.save_market_data(market_df, ticker='2222.SR')
```

## Storage Architecture

### What Gets Stored Where?

1. **Raw Filings** → Filesystem (`data/raw/`)
   - PDFs, HTML, XLS files
   - Organized by company/ticker/year

2. **Market Data** → Database OR Filesystem
   - Database: Structured tables for queries
   - Filesystem: Parquet/CSV files for backup

3. **Processed Data** → Database
   - Company-period financial data
   - Extracted ratios
   - Risk scores

4. **Interim Data** → Filesystem (`data/interim/`)
   - OCR text chunks
   - Parsed tables
   - Temporary processing files

## Database Schema

### Tables Created:

1. **company_periods** - Main financial data (one row per company-period)
2. **market_data** - Stock price/volume data (daily)
3. **risk_scores** - Credit risk outputs
4. **filings_metadata** - Metadata about downloaded filings

## Migration Between Storage Types

You can easily switch between DuckDB and PostgreSQL:

```python
# Export from DuckDB
duckdb_storage = DataStorage("duckdb")
df = duckdb_storage.query("SELECT * FROM company_periods")

# Import to PostgreSQL
postgres_storage = DataStorage("postgresql", connection_string="...")
for _, row in df.iterrows():
    postgres_storage.save_company_period(row.to_dict())
```

## Best Practices

1. **Start with DuckDB** - Easier setup, perfect for research
2. **Use PostgreSQL** - If you need collaboration or production deployment
3. **Backup Regularly** - DuckDB files can be copied directly
4. **Version Control** - Use DVC for data versioning
5. **Query Optimization** - Create indexes for frequently queried columns

## Environment Configuration

Add to `.env`:
```bash
# DuckDB (default)
DUCKDB_PATH=data/tadawul.duckdb

# PostgreSQL (optional)
DATABASE_URL=postgresql://user:password@localhost/tadawul_data
```

## Troubleshooting

**DuckDB:**
- Database file is created automatically
- Make sure directory exists and is writable
- File can be moved/copied like any file

**PostgreSQL:**
- Check PostgreSQL is running: `pg_isready`
- Verify connection: `psql -d tadawul_data`
- Check user permissions
- Install driver: `pip install psycopg2-binary`

## Recommendation

**For your FYP project, start with DuckDB:**
- ✅ No setup complexity
- ✅ Fast for analytical queries
- ✅ Easy to backup (just copy the file)
- ✅ Perfect for single-user research
- ✅ Can migrate to PostgreSQL later if needed

