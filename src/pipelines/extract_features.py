#!/usr/bin/env python3
"""
Feature Extraction Pipeline
Combines financial ratios, KAM features, and news features into ML-ready feature vectors.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.add(
    PROJECT_ROOT / "logs" / "pipeline" / "feature_extraction_{time}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


class FeatureExtractor:
    """Extracts and combines features from multiple data sources."""
    
    def __init__(
        self,
        processed_dir: Path,
        news_dir: Path,
        ratings_dir: Path,
        market_dir: Path,
        output_dir: Path
    ):
        """
        Initialize feature extractor.
        
        Args:
            processed_dir: Directory with processed financial data
            news_dir: Directory with news data
            ratings_dir: Directory with Tassnief ratings
            market_dir: Directory with market data
            output_dir: Output directory for features
        """
        self.processed_dir = Path(processed_dir)
        self.news_dir = Path(news_dir)
        self.ratings_dir = Path(ratings_dir)
        self.market_dir = Path(market_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_financial_ratios(self, financials: Dict) -> Dict:
        """
        Compute financial ratios from statement data.
        
        Args:
            financials: Dictionary with financial statement data
            
        Returns:
            Dictionary of computed ratios
        """
        bs = financials.get("balance_sheet", {})
        inc = financials.get("income_statement", {})
        cf = financials.get("cash_flow", {})
        
        ratios = {}
        
        # Liquidity ratios
        current_assets = bs.get("current_assets", 0)
        current_liabilities = bs.get("current_liabilities", 1)  # Avoid div by zero
        inventory = bs.get("inventory", 0)
        cash = bs.get("cash_and_equivalents", 0)
        
        ratios["current_ratio"] = self._safe_divide(current_assets, current_liabilities)
        ratios["quick_ratio"] = self._safe_divide(current_assets - inventory, current_liabilities)
        ratios["cash_ratio"] = self._safe_divide(cash, current_liabilities)
        
        # Leverage ratios
        total_assets = bs.get("total_assets", 1)
        total_equity = bs.get("total_equity", 1)
        total_debt = bs.get("long_term_debt", 0) + bs.get("short_term_debt", 0)
        
        ratios["debt_to_equity"] = self._safe_divide(total_debt, total_equity)
        ratios["debt_to_assets"] = self._safe_divide(total_debt, total_assets)
        ratios["long_term_debt_ratio"] = self._safe_divide(bs.get("long_term_debt", 0), total_assets)
        
        # Coverage ratios
        ebit = inc.get("ebit", inc.get("operating_income", 0))
        interest_expense = inc.get("interest_expense", 1)
        operating_cf = cf.get("operating_cash_flow", 0)
        
        ratios["interest_coverage"] = self._safe_divide(ebit, interest_expense)
        ratios["debt_service_coverage"] = self._safe_divide(operating_cf, total_debt) if total_debt > 0 else 0
        
        # Profitability ratios
        net_income = inc.get("net_income", 0)
        revenue = inc.get("revenue", 1)
        operating_income = inc.get("operating_income", 0)
        gross_profit = inc.get("gross_profit", 0)
        
        ratios["return_on_assets"] = self._safe_divide(net_income, total_assets)
        ratios["return_on_equity"] = self._safe_divide(net_income, total_equity)
        ratios["net_margin"] = self._safe_divide(net_income, revenue)
        ratios["operating_margin"] = self._safe_divide(operating_income, revenue)
        ratios["gross_margin"] = self._safe_divide(gross_profit, revenue)
        
        # Efficiency ratios
        receivables = bs.get("accounts_receivable", 1)
        ratios["asset_turnover"] = self._safe_divide(revenue, total_assets)
        ratios["receivables_turnover"] = self._safe_divide(revenue, receivables)
        
        return ratios
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division handling zero denominators."""
        if denominator == 0 or pd.isna(denominator):
            return np.nan
        return numerator / denominator
    
    def compute_growth_ratios(
        self,
        current_financials: Dict,
        previous_financials: Optional[Dict]
    ) -> Dict:
        """
        Compute year-over-year growth ratios.
        
        Args:
            current_financials: Current period financial data
            previous_financials: Previous period financial data
            
        Returns:
            Dictionary of growth ratios
        """
        growth = {}
        
        if not previous_financials:
            growth["revenue_growth"] = np.nan
            growth["net_income_growth"] = np.nan
            return growth
        
        curr_rev = current_financials.get("income_statement", {}).get("revenue", 0)
        prev_rev = previous_financials.get("income_statement", {}).get("revenue", 1)
        growth["revenue_growth"] = self._safe_divide(curr_rev - prev_rev, prev_rev)
        
        curr_ni = current_financials.get("income_statement", {}).get("net_income", 0)
        prev_ni = previous_financials.get("income_statement", {}).get("net_income", 1)
        if prev_ni != 0:
            growth["net_income_growth"] = self._safe_divide(curr_ni - prev_ni, abs(prev_ni))
        else:
            growth["net_income_growth"] = np.nan
        
        return growth
    
    def load_news_features(self, ticker: str, fiscal_year: int) -> Dict:
        """
        Load and compute news features for a company-year.
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year
            
        Returns:
            Dictionary of news features
        """
        features = {
            "news_count": 0,
            "avg_sentiment": 0.0,
            "sentiment_std": 0.0,
            "negative_news_ratio": 0.0,
            "positive_news_ratio": 0.0,
            "event_lawsuit": 0,
            "event_expansion": 0,
            "event_regulatory": 0,
            "event_earnings": 0
        }
        
        # Load news file
        ticker_file = ticker.replace(".", "_")
        news_path = self.news_dir / f"{ticker_file}_news.json"
        
        if not news_path.exists():
            logger.debug(f"No news file found for {ticker}")
            return features
        
        with open(news_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        articles = news_data.get("articles", [])
        
        # Filter articles for the fiscal year
        year_articles = []
        for article in articles:
            pub_date = article.get("published_date", "")
            if pub_date.startswith(str(fiscal_year)):
                year_articles.append(article)
        
        if not year_articles:
            return features
        
        features["news_count"] = len(year_articles)
        
        # Compute sentiment features
        sentiments = []
        for article in year_articles:
            # If sentiment is pre-computed
            sent = article.get("sentiment_score")
            if sent is not None:
                sentiments.append(sent)
        
        if sentiments:
            features["avg_sentiment"] = np.mean(sentiments)
            features["sentiment_std"] = np.std(sentiments) if len(sentiments) > 1 else 0
            features["negative_news_ratio"] = sum(1 for s in sentiments if s < -0.1) / len(sentiments)
            features["positive_news_ratio"] = sum(1 for s in sentiments if s > 0.1) / len(sentiments)
        
        # Count event keywords
        event_keywords = {
            "event_lawsuit": ["lawsuit", "litigation", "sued", "legal action", "court"],
            "event_expansion": ["expansion", "growth", "new market", "acquisition", "merger"],
            "event_regulatory": ["regulatory", "compliance", "fine", "penalty", "investigation"],
            "event_earnings": ["earnings", "profit", "revenue", "quarterly results", "financial results"]
        }
        
        for article in year_articles:
            text = (article.get("headline", "") + " " + article.get("description", "")).lower()
            for event_type, keywords in event_keywords.items():
                if any(kw in text for kw in keywords):
                    features[event_type] += 1
        
        return features
    
    def load_kam_features(self, ticker: str, fiscal_year: int) -> Dict:
        """
        Load KAM features for a company-year.
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year
            
        Returns:
            Dictionary of KAM features
        """
        default_features = {
            "kam_count": 0,
            "kam_going_concern": 0,
            "kam_impairment": 0,
            "kam_revenue_recognition": 0,
            "kam_related_party": 0,
            "kam_litigation": 0,
            "kam_valuation": 0,
            "kam_severity_score": 0.0
        }
        
        kam_features_path = self.processed_dir / "kams" / "kam_features.json"
        
        if not kam_features_path.exists():
            return default_features
        
        with open(kam_features_path, 'r') as f:
            all_features = json.load(f)
        
        for entry in all_features:
            if entry.get("ticker") == ticker and entry.get("fiscal_year") == fiscal_year:
                return {k: entry.get(k, default_features.get(k, 0)) for k in default_features}
        
        return default_features
    
    def load_market_features(self, ticker: str, fiscal_year: int) -> Dict:
        """
        Load market data features for a company-year.
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year
            
        Returns:
            Dictionary of market features
        """
        features = {
            "avg_price": np.nan,
            "price_volatility": np.nan,
            "avg_volume": np.nan,
            "market_cap": np.nan,
            "price_return": np.nan
        }
        
        # Try to find market data file
        ticker_file = ticker.replace(".", "_")
        market_path = self.market_dir / f"{ticker_file}.parquet"
        
        if not market_path.exists():
            market_path = self.market_dir / f"{ticker_file}.csv"
        
        if not market_path.exists():
            return features
        
        try:
            if market_path.suffix == ".parquet":
                df = pd.read_parquet(market_path)
            else:
                df = pd.read_csv(market_path, parse_dates=["Date"])
            
            # Filter for fiscal year
            df["Year"] = pd.to_datetime(df["Date"]).dt.year
            year_df = df[df["Year"] == fiscal_year]
            
            if year_df.empty:
                return features
            
            features["avg_price"] = year_df["Close"].mean()
            features["price_volatility"] = year_df["Close"].std()
            features["avg_volume"] = year_df["Volume"].mean()
            
            if "MarketCap" in year_df.columns:
                features["market_cap"] = year_df["MarketCap"].iloc[-1]
            
            # Compute annual return
            if len(year_df) > 1:
                start_price = year_df["Close"].iloc[0]
                end_price = year_df["Close"].iloc[-1]
                features["price_return"] = (end_price - start_price) / start_price
            
        except Exception as e:
            logger.error(f"Error loading market data for {ticker}: {e}")
        
        return features
    
    def load_rating(self, ticker: str, fiscal_year: int) -> Dict:
        """
        Load Tassnief rating for a company-year.
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year
            
        Returns:
            Dictionary with rating information
        """
        rating_info = {
            "tassnief_rating": None,
            "rating_bucket": None,
            "rating_numeric": None,
            "outlook": None
        }
        
        matrix_path = self.ratings_dir / "rating_matrix.json"
        
        if not matrix_path.exists():
            return rating_info
        
        with open(matrix_path, 'r') as f:
            matrix = json.load(f)
        
        key = f"{ticker}_{fiscal_year}"
        if key in matrix:
            entry = matrix[key]
            rating_info["tassnief_rating"] = entry.get("rating")
            rating_info["rating_bucket"] = entry.get("rating_bucket")
            rating_info["rating_numeric"] = entry.get("rating_numeric")
            rating_info["outlook"] = entry.get("outlook")
        
        return rating_info
    
    def extract_company_features(
        self,
        ticker: str,
        company_name: str,
        fiscal_year: int,
        financials: Dict,
        previous_financials: Optional[Dict] = None
    ) -> Dict:
        """
        Extract all features for a company-year.
        
        Args:
            ticker: Company ticker
            company_name: Company name
            fiscal_year: Fiscal year
            financials: Financial statement data
            previous_financials: Previous year financials (for growth)
            
        Returns:
            Complete feature dictionary
        """
        # Start with identifiers
        features = {
            "ticker": ticker,
            "company_name": company_name,
            "fiscal_year": fiscal_year
        }
        
        # Add financial ratios
        ratios = self.compute_financial_ratios(financials)
        features.update(ratios)
        
        # Add growth ratios
        growth = self.compute_growth_ratios(financials, previous_financials)
        features.update(growth)
        
        # Add KAM features
        kam_features = self.load_kam_features(ticker, fiscal_year)
        features.update(kam_features)
        
        # Add news features
        news_features = self.load_news_features(ticker, fiscal_year)
        features.update(news_features)
        
        # Add market features
        market_features = self.load_market_features(ticker, fiscal_year)
        features.update(market_features)
        
        # Add target (rating)
        rating_info = self.load_rating(ticker, fiscal_year)
        features.update(rating_info)
        
        return features
    
    def save_features(self, features: List[Dict], filename: str = "features") -> Path:
        """
        Save features to parquet and JSON.
        
        Args:
            features: List of feature dictionaries
            filename: Output filename (without extension)
            
        Returns:
            Path to parquet file
        """
        df = pd.DataFrame(features)
        
        # Save as parquet
        parquet_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Save as JSON for readability
        json_path = self.output_dir / f"{filename}.json"
        df.to_json(json_path, orient="records", indent=2)
        
        # Save feature summary
        summary = {
            "total_records": len(df),
            "companies": df["ticker"].nunique() if "ticker" in df.columns else 0,
            "years": sorted(df["fiscal_year"].unique().tolist()) if "fiscal_year" in df.columns else [],
            "feature_columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / f"{filename}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {len(features)} feature vectors to {parquet_path}")
        return parquet_path


def main():
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features for ML model")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory with processed financial data"
    )
    parser.add_argument(
        "--news-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "news"),
        help="Directory with news data"
    )
    parser.add_argument(
        "--ratings-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ratings"),
        help="Directory with ratings data"
    )
    parser.add_argument(
        "--market-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "market"),
        help="Directory with market data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor(
        processed_dir=Path(args.processed_dir),
        news_dir=Path(args.news_dir),
        ratings_dir=Path(args.ratings_dir),
        market_dir=Path(args.market_dir),
        output_dir=Path(args.output_dir)
    )
    
    logger.info("Feature extraction pipeline initialized")
    logger.info("To use: Ensure all data sources are populated, then implement company iteration")
    
    # TODO: Implement iteration over companies and years
    # This should:
    # 1. Load list of companies from inventory
    # 2. For each company-year, load financials and call extract_company_features
    # 3. Save combined features


if __name__ == "__main__":
    main()
