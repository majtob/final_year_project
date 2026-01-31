#!/usr/bin/env python3
"""
News Collection Pipeline
Fetches news articles for companies from NewsAPI and MarketAux.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import requests
from loguru import logger
from tqdm import tqdm
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "news_collection_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class NewsCollector:
    """Collects news articles from multiple sources."""
    
    def __init__(
        self,
        output_dir: Path,
        newsapi_key: Optional[str] = None,
        marketaux_key: Optional[str] = None,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize news collector.
        
        Args:
            output_dir: Directory to save news data
            newsapi_key: NewsAPI API key
            marketaux_key: MarketAux API key
            rate_limit_delay: Delay between requests (seconds)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.marketaux_key = marketaux_key or os.getenv("MARKETAUX_KEY")
        self.rate_limit_delay = rate_limit_delay
        
    def fetch_newsapi(
        self,
        query: str,
        from_date: str,
        to_date: str,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Fetch news from NewsAPI.
        
        Args:
            query: Search query (company name or ticker)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_results: Maximum articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(max_results, 100),
            "apiKey": self.newsapi_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "headline": article.get("title"),
                    "description": article.get("description"),
                    "source": article.get("source", {}).get("name"),
                    "published_date": article.get("publishedAt", "")[:10],
                    "url": article.get("url"),
                    "content": article.get("content"),
                    "api_source": "newsapi"
                })
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI for '{query}'")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def fetch_marketaux(
        self,
        symbols: str,
        from_date: str,
        to_date: str,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Fetch news from MarketAux.
        
        Args:
            symbols: Stock symbol(s)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_results: Maximum articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.marketaux_key:
            logger.warning("MarketAux key not configured")
            return []
        
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "symbols": symbols,
            "published_after": from_date,
            "published_before": to_date,
            "language": "en",
            "limit": max_results,
            "api_token": self.marketaux_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("data", []):
                articles.append({
                    "headline": article.get("title"),
                    "description": article.get("description"),
                    "source": article.get("source"),
                    "published_date": article.get("published_at", "")[:10],
                    "url": article.get("url"),
                    "sentiment": article.get("sentiment"),
                    "api_source": "marketaux"
                })
            
            logger.info(f"Fetched {len(articles)} articles from MarketAux for '{symbols}'")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from MarketAux: {e}")
            return []
    
    def collect_company_news(
        self,
        company_name: str,
        ticker: str,
        from_date: str,
        to_date: str,
        max_results: int = 100
    ) -> Dict:
        """
        Collect news for a single company from all sources.
        
        Args:
            company_name: Company name for search
            ticker: Stock ticker
            from_date: Start date
            to_date: End date
            max_results: Max articles per source
            
        Returns:
            Dictionary with company news data
        """
        all_articles = []
        
        # Fetch from NewsAPI
        newsapi_articles = self.fetch_newsapi(
            query=company_name,
            from_date=from_date,
            to_date=to_date,
            max_results=max_results
        )
        all_articles.extend(newsapi_articles)
        time.sleep(self.rate_limit_delay)
        
        # Fetch from MarketAux
        marketaux_articles = self.fetch_marketaux(
            symbols=ticker,
            from_date=from_date,
            to_date=to_date,
            max_results=max_results
        )
        all_articles.extend(marketaux_articles)
        
        # Add ticker to all articles
        for article in all_articles:
            article["ticker"] = ticker
            article["company_name"] = company_name
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "date_range": {"from": from_date, "to": to_date},
            "total_articles": len(all_articles),
            "articles": all_articles,
            "collection_timestamp": datetime.now().isoformat()
        }
    
    def save_company_news(self, news_data: Dict) -> Path:
        """Save company news to JSON file."""
        ticker = news_data["ticker"].replace(".", "_")
        output_path = self.output_dir / f"{ticker}_news.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved news for {news_data['ticker']} to {output_path}")
        return output_path
    
    def collect_all(
        self,
        companies: List[Dict],
        from_date: str,
        to_date: str,
        max_results: int = 100
    ) -> Dict:
        """
        Collect news for all companies.
        
        Args:
            companies: List of company dicts with 'company_name' and 'ticker'
            from_date: Start date
            to_date: End date
            max_results: Max articles per company per source
            
        Returns:
            Summary dictionary
        """
        logger.info(f"Collecting news for {len(companies)} companies")
        
        results = []
        for company in tqdm(companies, desc="Collecting news"):
            news_data = self.collect_company_news(
                company_name=company["company_name"],
                ticker=company["ticker"],
                from_date=from_date,
                to_date=to_date,
                max_results=max_results
            )
            self.save_company_news(news_data)
            results.append({
                "ticker": company["ticker"],
                "articles_collected": news_data["total_articles"]
            })
            time.sleep(self.rate_limit_delay)
        
        summary = {
            "companies_processed": len(companies),
            "total_articles": sum(r["articles_collected"] for r in results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "collection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def load_inventory(inventory_path: Path) -> Dict:
    """Load data inventory."""
    with open(inventory_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main entry point for news collection."""
    parser = argparse.ArgumentParser(description="Collect news for companies")
    parser.add_argument(
        "--inventory",
        type=str,
        default=str(PROJECT_ROOT / "data" / "data_inventory.json"),
        help="Path to data inventory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "news"),
        help="Output directory"
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Max articles per company per source"
    )
    
    args = parser.parse_args()
    
    # Load inventory
    inventory = load_inventory(Path(args.inventory))
    companies = inventory.get("companies", [])
    
    if not companies:
        logger.warning("No companies found in inventory")
        return
    
    # Initialize collector
    collector = NewsCollector(output_dir=Path(args.output_dir))
    
    # Collect news
    summary = collector.collect_all(
        companies=companies,
        from_date=args.from_date,
        to_date=args.to_date,
        max_results=args.max_results
    )
    
    logger.success(f"News collection completed. Total articles: {summary['total_articles']}")


if __name__ == "__main__":
    main()
