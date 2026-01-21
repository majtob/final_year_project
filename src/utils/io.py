"""
Storage utilities for structured data.
Supports both DuckDB (recommended for start) and PostgreSQL (for production).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent


class DataStorage:
    """Unified interface for data storage (DuckDB or PostgreSQL)."""
    
    def __init__(
        self,
        storage_type: str = "duckdb",
        connection_string: Optional[str] = None,
        db_path: Optional[Path] = None
    ):
        """
        Initialize data storage.
        
        Args:
            storage_type: "duckdb" or "postgresql"
            connection_string: PostgreSQL connection string (e.g., "postgresql://user:pass@localhost/dbname")
            db_path: Path to DuckDB database file (default: data/tadawul.duckdb)
        """
        self.storage_type = storage_type.lower()
        
        if self.storage_type == "duckdb":
            import duckdb
            if db_path is None:
                db_path = PROJECT_ROOT / "data" / "tadawul.duckdb"
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to DuckDB: {self.db_path}")
            
        elif self.storage_type == "postgresql":
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
            except ImportError:
                raise ImportError("psycopg2 required for PostgreSQL. Install with: pip install psycopg2-binary")
            
            if connection_string is None:
                connection_string = os.getenv(
                    "DATABASE_URL",
                    "postgresql://localhost/tadawul_data"
                )
            
            self.conn = psycopg2.connect(connection_string)
            logger.info("Connected to PostgreSQL")
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def create_tables(self):
        """Create all required tables based on schema."""
        logger.info("Creating database tables...")
        
        # Company periods table (main financial data)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS company_periods (
                id INTEGER PRIMARY KEY,
                company_name VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                fiscal_year INTEGER NOT NULL,
                fiscal_period VARCHAR NOT NULL,
                reporting_date DATE NOT NULL,
                source_file_id VARCHAR,
                source_url VARCHAR,
                source_section VARCHAR,
                source_page INTEGER,
                language VARCHAR,
                
                -- Balance Sheet
                total_assets FLOAT,
                total_liabilities FLOAT,
                total_equity FLOAT,
                current_assets FLOAT,
                current_liabilities FLOAT,
                long_term_debt FLOAT,
                
                -- Income Statement
                revenue FLOAT,
                net_income FLOAT,
                operating_income FLOAT,
                interest_expense FLOAT,
                ebitda FLOAT,
                
                -- Cash Flow
                operating_cash_flow FLOAT,
                investing_cash_flow FLOAT,
                financing_cash_flow FLOAT,
                
                -- Ratios
                current_ratio FLOAT,
                debt_to_equity FLOAT,
                interest_coverage FLOAT,
                return_on_assets FLOAT,
                net_margin FLOAT,
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(ticker, fiscal_year, fiscal_period)
            )
        """)
        
        # Market data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                market_cap FLOAT,
                dividends FLOAT,
                stock_splits FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(ticker, date)
            )
        """)
        
        # Risk scores table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY,
                company_name VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                fiscal_year INTEGER NOT NULL,
                risk_score FLOAT NOT NULL,
                risk_flag INTEGER NOT NULL,
                model_name VARCHAR,
                prompt_id VARCHAR,
                citations TEXT,
                extraction_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(ticker, fiscal_year)
            )
        """)
        
        # Filings metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS filings_metadata (
                id INTEGER PRIMARY KEY,
                company_name VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                fiscal_year INTEGER,
                filing_type VARCHAR,
                file_path VARCHAR,
                file_name VARCHAR,
                file_size BIGINT,
                file_hash VARCHAR,
                url VARCHAR,
                language VARCHAR,
                publication_date DATE,
                download_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.success("Database tables created successfully")
    
    def save_company_period(
        self,
        data: Dict[str, Any],
        update_if_exists: bool = True
    ) -> int:
        """
        Save or update company period data.
        
        Args:
            data: Dictionary with company period data
            update_if_exists: Update if record already exists
            
        Returns:
            ID of saved/updated record
        """
        if self.storage_type == "duckdb":
            return self._save_company_period_duckdb(data, update_if_exists)
        else:
            return self._save_company_period_postgresql(data, update_if_exists)
    
    def _save_company_period_duckdb(
        self,
        data: Dict[str, Any],
        update_if_exists: bool
    ) -> int:
        """Save company period to DuckDB."""
        # Check if exists
        existing = self.conn.execute("""
            SELECT id FROM company_periods
            WHERE ticker = $1 AND fiscal_year = $2 AND fiscal_period = $3
        """, [data['ticker'], data['fiscal_year'], data['fiscal_period']]).fetchone()
        
        if existing and update_if_exists:
            # Update
            self.conn.execute("""
                UPDATE company_periods SET
                    company_name = $1,
                    reporting_date = $2,
                    total_assets = $3,
                    total_liabilities = $4,
                    total_equity = $5,
                    revenue = $6,
                    net_income = $7,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $8
            """, [
                data.get('company_name'),
                data.get('reporting_date'),
                data.get('total_assets'),
                data.get('total_liabilities'),
                data.get('total_equity'),
                data.get('revenue'),
                data.get('net_income'),
                existing[0]
            ])
            return existing[0]
        else:
            # Insert
            self.conn.execute("""
                INSERT INTO company_periods (
                    company_name, ticker, fiscal_year, fiscal_period, reporting_date,
                    total_assets, total_liabilities, total_equity,
                    revenue, net_income, operating_income, interest_expense, ebitda,
                    operating_cash_flow, investing_cash_flow, financing_cash_flow,
                    current_ratio, debt_to_equity, interest_coverage, return_on_assets, net_margin,
                    source_file_id, source_url, language
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
            """, [
                data.get('company_name'),
                data.get('ticker'),
                data.get('fiscal_year'),
                data.get('fiscal_period'),
                data.get('reporting_date'),
                data.get('total_assets'),
                data.get('total_liabilities'),
                data.get('total_equity'),
                data.get('revenue'),
                data.get('net_income'),
                data.get('operating_income'),
                data.get('interest_expense'),
                data.get('ebitda'),
                data.get('operating_cash_flow'),
                data.get('investing_cash_flow'),
                data.get('financing_cash_flow'),
                data.get('current_ratio'),
                data.get('debt_to_equity'),
                data.get('interest_coverage'),
                data.get('return_on_assets'),
                data.get('net_margin'),
                data.get('source_file_id'),
                data.get('source_url'),
                data.get('language')
            ])
            return self.conn.execute("SELECT LAST_INSERT_ROWID()").fetchone()[0]
    
    def _save_company_period_postgresql(
        self,
        data: Dict[str, Any],
        update_if_exists: bool
    ) -> int:
        """Save company period to PostgreSQL."""
        # Similar implementation for PostgreSQL
        # Using ON CONFLICT for upsert
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO company_periods (
                company_name, ticker, fiscal_year, fiscal_period, reporting_date,
                total_assets, total_liabilities, total_equity,
                revenue, net_income, source_file_id, source_url, language
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, fiscal_year, fiscal_period)
            DO UPDATE SET
                company_name = EXCLUDED.company_name,
                reporting_date = EXCLUDED.reporting_date,
                total_assets = EXCLUDED.total_assets,
                total_liabilities = EXCLUDED.total_liabilities,
                total_equity = EXCLUDED.total_equity,
                revenue = EXCLUDED.revenue,
                net_income = EXCLUDED.net_income,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, [
            data.get('company_name'),
            data.get('ticker'),
            data.get('fiscal_year'),
            data.get('fiscal_period'),
            data.get('reporting_date'),
            data.get('total_assets'),
            data.get('total_liabilities'),
            data.get('total_equity'),
            data.get('revenue'),
            data.get('net_income'),
            data.get('source_file_id'),
            data.get('source_url'),
            data.get('language')
        ])
        result = cursor.fetchone()
        self.conn.commit()
        return result[0]
    
    def save_market_data(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> int:
        """
        Save market data DataFrame.
        
        Args:
            df: DataFrame with market data
            ticker: Stock ticker
            
        Returns:
            Number of records saved
        """
        df['ticker'] = ticker
        
        if self.storage_type == "duckdb":
            # DuckDB can directly insert from DataFrame
            self.conn.execute("""
                INSERT OR REPLACE INTO market_data 
                (ticker, date, open, high, low, close, volume, market_cap, dividends, stock_splits)
                SELECT ticker, date, open, high, low, close, volume, market_cap, dividends, stock_splits
                FROM df
            """)
            return len(df)
        else:
            # PostgreSQL
            from psycopg2.extras import execute_values
            cursor = self.conn.cursor()
            
            records = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'market_cap', 'dividends', 'stock_splits']].to_dict('records')
            
            execute_values(
                cursor,
                """
                INSERT INTO market_data (ticker, date, open, high, low, close, volume, market_cap, dividends, stock_splits)
                VALUES %s
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    market_cap = EXCLUDED.market_cap
                """,
                [(r['ticker'], r['date'], r.get('open'), r.get('high'), r.get('low'), 
                  r.get('close'), r.get('volume'), r.get('market_cap'), 
                  r.get('dividends'), r.get('stock_splits')) for r in records]
            )
            self.conn.commit()
            return len(df)
    
    def query(
        self,
        sql: str,
        params: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with results
        """
        if self.storage_type == "duckdb":
            if params:
                result = self.conn.execute(sql, params).fetchdf()
            else:
                result = self.conn.execute(sql).fetchdf()
        else:
            # PostgreSQL
            cursor = self.conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Create DataFrame
            if rows and columns:
                result = pd.DataFrame(rows, columns=columns)
            else:
                result = pd.DataFrame()
        
        return result
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")


def init_database(storage_type: str = "duckdb", **kwargs) -> DataStorage:
    """
    Initialize database with tables.
    
    Args:
        storage_type: "duckdb" or "postgresql"
        **kwargs: Additional arguments for DataStorage
        
    Returns:
        DataStorage instance
    """
    storage = DataStorage(storage_type=storage_type, **kwargs)
    storage.create_tables()
    return storage

