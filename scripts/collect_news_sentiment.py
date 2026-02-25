"""
Collect news articles and compute sentiment for all Tassnief-rated companies.

Pipeline:
1. Fetch news from MarketAux API for each company
2. Score sentiment using VADER (financial-domain adjusted)
3. Aggregate to annual sentiment features per company
4. Merge with existing ratings+financials dataset

Sentiment Features Generated:
- sentiment_mean:   Average sentiment score across articles (-1 to +1)
- sentiment_std:    Sentiment volatility
- sentiment_pos:    Proportion of positive articles
- sentiment_neg:    Proportion of negative articles
- news_count:       Number of news articles found
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RATINGS_FILE = DATA_DIR / "templates" / "ratings_scraped.csv"
FINANCIALS_FILE = DATA_DIR / "processed" / "ratings_with_financials.csv"
NEWS_RAW_DIR = DATA_DIR / "raw" / "news"
OUTPUT_FILE = DATA_DIR / "processed" / "ratings_financials_sentiment.csv"

# MarketAux API
API_KEY = os.getenv("MARKETAUX_API_KEY", "iQAXIrPvqBDP7VVxBlT54NJ8yC8fTQ3HbXEfOt50")
BASE_URL = "https://api.marketaux.com/v1/news/all"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Company search terms mapped to tickers
COMPANY_SEARCH_TERMS = {
    '7010.SR': ['Saudi Telecom', 'STC', 'stc Group'],
    '7020.SR': ['Etihad Etisalat', 'Mobily'],
    '7030.SR': ['Zain Saudi', 'Zain KSA'],
    '2081.SR': ['Alkhorayef'],
    '2070.SR': ['Perfect Presentation', '2P Company'],
    '9596.SR': ['Quara Finance'],
    '9535.SR': ['Ladun Investment'],
    '4290.SR': ['Alkhaleej Training'],
    '9619.SR': ['Multi Business Group'],
    '9511.SR': ['Sumou Real Estate'],
    '3008.SR': ['Al Kathiri Holding', 'AKHC'],
    '2210.SR': ['Napco National'],
    '8200.SR': ['Saudi Re', 'Cooperative Reinsurance'],
    '4263.SR': ['SAL Logistics', 'SAL Saudi Logistics'],
    '6004.SR': ['CATRION', 'Catrion Catering'],
    '4007.SR': ['Al Hammadi'],
    '4013.SR': ['Al Habib', 'Dr Sulaiman Al Habib', 'HMG'],
    '4004.SR': ['Middle East Healthcare', 'MEAHCO'],
    '4321.SR': ['Arabian Centers', 'Cenomi Centers'],
    '9578.SR': ['Atlas Elevators'],
    '9568.SR': ['Mayar Holding'],
    '8050.SR': ['SAICO', 'Saudi Arabian Cooperative Insurance'],
    '8060.SR': ['Malath Insurance', 'Malath Cooperative'],
    '2120.SR': ['MEPCO', 'Middle East Paper'],
}


def fetch_news_for_company(ticker: str, search_terms: list, 
                           year: int, max_articles: int = 50):
    """
    Fetch news articles for a company in a given fiscal year.
    
    Uses the fiscal year as the date range: Jan 1 to Dec 31 of that year.
    """
    articles = []
    seen_urls = set()
    
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"
    
    for term in search_terms:
        try:
            params = {
                'api_token': API_KEY,
                'search': term,
                'countries': 'sa,us,gb,ae',
                'language': 'en',
                'published_after': from_date,
                'published_before': to_date,
                'filter_entities': 'true',
                'limit': 3,
                'page': 1,
            }
            
            response = requests.get(BASE_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"  HTTP {response.status_code} for '{term}'")
                continue
            
            data = response.json()
            
            if 'error' in data:
                logger.warning(f"  API error for '{term}': {data['error']}")
                continue
            
            fetched = data.get('data', [])
            
            for article in fetched:
                url = article.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append({
                        'ticker': ticker,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'snippet': article.get('snippet', ''),
                        'published_at': article.get('published_at', ''),
                        'source_name': article.get('source', ''),
                        'url': url,
                        'search_term': term,
                    })
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"  Error fetching '{term}': {e}")
            continue
    
    return articles


def score_sentiment(articles: list):
    """
    Score sentiment for a list of articles using VADER.
    
    Returns articles with sentiment scores added.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    for article in articles:
        text = f"{article.get('title', '')}. {article.get('description', '')}"
        scores = analyzer.polarity_scores(text)
        article['sentiment_compound'] = scores['compound']
        article['sentiment_pos'] = scores['pos']
        article['sentiment_neg'] = scores['neg']
        article['sentiment_neu'] = scores['neu']
    
    return articles


def aggregate_sentiment(articles: list):
    """
    Aggregate article-level sentiment to company-year level.
    
    Returns dict with aggregated features.
    """
    if not articles:
        return {
            'sentiment_mean': None,
            'sentiment_std': None,
            'sentiment_pos_pct': None,
            'sentiment_neg_pct': None,
            'news_count': 0,
        }
    
    scores = [a['sentiment_compound'] for a in articles]
    
    return {
        'sentiment_mean': np.mean(scores),
        'sentiment_std': np.std(scores) if len(scores) > 1 else 0,
        'sentiment_pos_pct': sum(1 for s in scores if s > 0.05) / len(scores),
        'sentiment_neg_pct': sum(1 for s in scores if s < -0.05) / len(scores),
        'news_count': len(articles),
    }


def save_raw_news(ticker: str, year: int, articles: list):
    """Save raw articles to JSON."""
    NEWS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    clean_ticker = ticker.replace('.', '_')
    filepath = NEWS_RAW_DIR / f"{clean_ticker}_{year}_news.json"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'ticker': ticker,
            'year': year,
            'fetch_date': datetime.now().isoformat(),
            'article_count': len(articles),
            'articles': articles,
        }, f, indent=2, ensure_ascii=False)


def collect_all_sentiment():
    """
    Main pipeline: collect news and sentiment for all company-years.
    """
    logger.info("=" * 60)
    logger.info("NEWS SENTIMENT COLLECTION")
    logger.info("=" * 60)
    
    # Load ratings to know which ticker-year combos we need
    ratings_df = pd.read_csv(RATINGS_FILE)
    
    # Get unique ticker-year pairs
    ticker_years = ratings_df[['ticker', 'fiscal_year_applicable']].drop_duplicates()
    logger.info(f"Need sentiment for {len(ticker_years)} ticker-year pairs")
    
    # Collect sentiment for each
    all_sentiment = []
    total = len(ticker_years)
    
    for i, (_, row) in enumerate(ticker_years.iterrows(), 1):
        ticker = row['ticker']
        year = int(row['fiscal_year_applicable'])
        
        search_terms = COMPANY_SEARCH_TERMS.get(ticker, [])
        if not search_terms:
            logger.warning(f"[{i}/{total}] No search terms for {ticker}, skipping")
            all_sentiment.append({
                'ticker': ticker,
                'fiscal_year': year,
                'sentiment_mean': None,
                'sentiment_std': None,
                'sentiment_pos_pct': None,
                'sentiment_neg_pct': None,
                'news_count': 0,
            })
            continue
        
        logger.info(f"[{i}/{total}] {ticker} ({year}) - searching: {search_terms}")
        
        # Fetch news
        articles = fetch_news_for_company(ticker, search_terms, year)
        logger.info(f"  Found {len(articles)} articles")
        
        # Score sentiment
        if articles:
            articles = score_sentiment(articles)
            save_raw_news(ticker, year, articles)
            
            # Show sample
            for a in articles[:2]:
                logger.info(f"  > \"{a['title'][:60]}...\" sentiment={a['sentiment_compound']:.3f}")
        
        # Aggregate
        agg = aggregate_sentiment(articles)
        agg['ticker'] = ticker
        agg['fiscal_year'] = year
        all_sentiment.append(agg)
        
        # Rate limiting between companies
        time.sleep(1)
    
    # Create sentiment DataFrame
    sentiment_df = pd.DataFrame(all_sentiment)
    
    return sentiment_df


def merge_all_data(sentiment_df):
    """
    Merge sentiment with existing financials+ratings data.
    """
    logger.info("\nMerging all data...")
    
    # Load financials
    fin_df = pd.read_csv(FINANCIALS_FILE)
    
    # Merge
    merged = pd.merge(
        fin_df,
        sentiment_df,
        left_on=['ticker', 'fiscal_year'],
        right_on=['ticker', 'fiscal_year'],
        how='left'
    )
    
    # Save
    merged.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved merged data to {OUTPUT_FILE}")
    
    return merged


def main():
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Collect sentiment
    sentiment_df = collect_all_sentiment()
    
    # Merge with financials
    merged_df = merge_all_data(sentiment_df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(merged_df)}")
    
    has_sentiment = merged_df['news_count'].fillna(0) > 0
    logger.info(f"Records with news: {has_sentiment.sum()}")
    logger.info(f"Records without news: {(~has_sentiment).sum()}")
    
    if has_sentiment.any():
        logger.info(f"\nSentiment statistics (where available):")
        logger.info(f"  Mean sentiment: {merged_df.loc[has_sentiment, 'sentiment_mean'].mean():.3f}")
        logger.info(f"  Avg news count: {merged_df.loc[has_sentiment, 'news_count'].mean():.1f}")
    
    # Show sample
    cols = ['ticker', 'fiscal_year', 'rating', 'sentiment_mean', 'news_count']
    available_cols = [c for c in cols if c in merged_df.columns]
    print("\nSample data:")
    print(merged_df[available_cols].head(15))
    
    return merged_df


if __name__ == "__main__":
    main()
