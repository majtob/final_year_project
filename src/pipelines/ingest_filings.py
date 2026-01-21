#!/usr/bin/env python3
"""
Tadawul Filings Ingestion Pipeline
Downloads and stores company filings from Saudi Exchange (Tadawul).
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import aiohttp
import asyncio
from loguru import logger
from tqdm import tqdm
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "filings_ingestion_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class TadawulFilingCollector:
    """Collects company filings from Tadawul website."""
    
    def __init__(
        self,
        output_dir: Path,
        base_url: str = "https://www.tadawul.com.sa",
        rate_limit_delay: float = 1.0,
        max_retries: int = 3
    ):
        """
        Initialize filing collector.
        
        Args:
            output_dir: Directory to save downloaded filings
            base_url: Base URL for Tadawul website
            rate_limit_delay: Delay between requests (seconds)
            max_retries: Maximum retry attempts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.download_log = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def download_file(
        self,
        url: str,
        output_path: Path,
        retry_count: int = None
    ) -> bool:
        """
        Download a file from URL.
        
        Args:
            url: URL to download
            output_path: Path to save file
            retry_count: Number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        if retry_count is None:
            retry_count = self.max_retries
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify file was downloaded
                if output_path.exists() and output_path.stat().st_size > 0:
                    file_hash = self.calculate_file_hash(output_path)
                    logger.success(f"Downloaded {output_path.name} ({output_path.stat().st_size:,} bytes)")
                    
                    return True
                else:
                    logger.error(f"File download incomplete: {output_path}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error downloading {url} (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False
        
        return False
    
    def save_filing_metadata(
        self,
        filing_info: Dict,
        file_path: Path,
        file_hash: str
    ) -> Path:
        """
        Save filing metadata.
        
        Args:
            filing_info: Dictionary with filing information
            file_path: Path to downloaded file
            file_hash: SHA256 hash of file
            
        Returns:
            Path to metadata file
        """
        metadata = {
            **filing_info,
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_hash': file_hash,
            'download_timestamp': datetime.now().isoformat(),
            'file_extension': file_path.suffix
        }
        
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_path
    
    def organize_filing_path(
        self,
        company_name: str,
        ticker: str,
        fiscal_year: int,
        filing_type: str,
        file_extension: str
    ) -> Path:
        """
        Generate organized file path for filing.
        
        Args:
            company_name: Company name
            ticker: Stock ticker
            fiscal_year: Fiscal year
            filing_type: Type of filing (annual_report, financial_statement, etc.)
            file_extension: File extension (.pdf, .html, .xls)
            
        Returns:
            Path object for the filing
        """
        # Clean company name for filesystem
        company_clean = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
        company_clean = company_clean.replace(' ', '_')
        
        # Create directory structure: company/ticker/year/
        company_dir = self.output_dir / company_clean / ticker / str(fiscal_year)
        company_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{ticker}_{fiscal_year}_{filing_type}{file_extension}"
        return company_dir / filename
    
    def collect_filing(
        self,
        filing_info: Dict
    ) -> Dict:
        """
        Collect a single filing.
        
        Args:
            filing_info: Dictionary with filing information:
                - url: URL to download
                - company_name: Company name
                - ticker: Stock ticker
                - fiscal_year: Fiscal year
                - filing_type: Type of filing
                - language: Language (en/ar/both)
                - publication_date: Publication date
                
        Returns:
            Dictionary with collection result
        """
        url = filing_info.get('url')
        if not url:
            logger.error("No URL provided in filing_info")
            return {'status': 'failed', 'error': 'No URL'}
        
        # Determine file extension from URL
        file_extension = Path(url).suffix or '.pdf'
        
        # Generate output path
        output_path = self.organize_filing_path(
            company_name=filing_info.get('company_name', 'Unknown'),
            ticker=filing_info.get('ticker', 'UNKNOWN'),
            fiscal_year=filing_info.get('fiscal_year', 2023),
            filing_type=filing_info.get('filing_type', 'filing'),
            file_extension=file_extension
        )
        
        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            file_hash = self.calculate_file_hash(output_path)
            metadata_path = self.save_filing_metadata(filing_info, output_path, file_hash)
            
            return {
                'status': 'skipped',
                'file_path': str(output_path),
                'file_hash': file_hash
            }
        
        # Download file
        success = self.download_file(url, output_path)
        
        if success:
            file_hash = self.calculate_file_hash(output_path)
            metadata_path = self.save_filing_metadata(filing_info, output_path, file_hash)
            
            # Log download
            self.download_log.append({
                **filing_info,
                'file_path': str(output_path),
                'file_hash': file_hash,
                'download_timestamp': datetime.now().isoformat(),
                'status': 'success'
            })
            
            return {
                'status': 'success',
                'file_path': str(output_path),
                'file_hash': file_hash
            }
        else:
            self.download_log.append({
                **filing_info,
                'download_timestamp': datetime.now().isoformat(),
                'status': 'failed'
            })
            
            return {
                'status': 'failed',
                'url': url
            }
    
    def collect_filings(
        self,
        filings: List[Dict]
    ) -> Dict:
        """
        Collect multiple filings.
        
        Args:
            filings: List of filing information dictionaries
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"Starting filing collection for {len(filings)} filings")
        
        successful = 0
        failed = 0
        skipped = 0
        
        for filing in tqdm(filings, desc="Downloading filings"):
            result = self.collect_filing(filing)
            
            if result['status'] == 'success':
                successful += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        # Save download log
        log_path = self.output_dir / "download_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.download_log, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        summary = {
            'total_filings': len(filings),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "collection_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nCollection Summary:")
        logger.info(f"  Successful: {successful}/{len(filings)}")
        logger.info(f"  Failed: {failed}/{len(filings)}")
        logger.info(f"  Skipped: {skipped}/{len(filings)}")
        
        return summary


def load_filings_from_inventory(inventory_path: Path) -> List[Dict]:
    """Load filings from data inventory file."""
    with open(inventory_path, 'r', encoding='utf-8') as f:
        inventory = json.load(f)
    
    filings = []
    if 'filings' in inventory:
        filings = inventory['filings']
    
    return filings


def main():
    """Main entry point for filing ingestion."""
    parser = argparse.ArgumentParser(description="Collect Tadawul company filings")
    parser.add_argument(
        "--inventory",
        type=str,
        help="Path to data inventory JSON file",
        default=str(PROJECT_ROOT / "data" / "data_inventory.json")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for filings",
        default=str(PROJECT_ROOT / "data" / "raw")
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        help="Delay between requests (seconds)",
        default=1.0
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum retry attempts",
        default=3
    )
    
    args = parser.parse_args()
    
    # Load filings from inventory
    inventory_path = Path(args.inventory)
    if not inventory_path.exists():
        logger.error(f"Inventory file not found: {inventory_path}")
        logger.info("Please populate data_inventory.json with filing URLs")
        return
    
    filings = load_filings_from_inventory(inventory_path)
    
    if not filings:
        logger.warning("No filings found in inventory")
        logger.info("Please add filings to data_inventory.json")
        return
    
    # Initialize collector
    collector = TadawulFilingCollector(
        output_dir=Path(args.output_dir),
        rate_limit_delay=args.rate_limit,
        max_retries=args.max_retries
    )
    
    # Collect filings
    summary = collector.collect_filings(filings)
    
    logger.success("Filing collection completed!")


if __name__ == "__main__":
    main()

