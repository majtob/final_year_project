#!/usr/bin/env python3
"""
Initialize database (DuckDB or PostgreSQL) with required tables.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import init_database
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Initialize database")
    parser.add_argument(
        "--type",
        type=str,
        choices=["duckdb", "postgresql"],
        default="duckdb",
        help="Database type (default: duckdb)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to DuckDB database file (default: data/tadawul.duckdb)"
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        help="PostgreSQL connection string (e.g., postgresql://user:pass@localhost/dbname)"
    )
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.db_path:
        kwargs['db_path'] = Path(args.db_path)
    if args.connection_string:
        kwargs['connection_string'] = args.connection_string
    
    logger.info(f"Initializing {args.type} database...")
    
    try:
        storage = init_database(storage_type=args.type, **kwargs)
        logger.success(f"Database initialized successfully!")
        logger.info(f"Storage type: {args.type}")
        if args.type == "duckdb":
            logger.info(f"Database path: {storage.db_path}")
        
        storage.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

