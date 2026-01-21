"""
Script to pull news for Saudi Aramco using Marketaux API.
Requires: pip install requests
API: https://marketaux.com/
"""

import json
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "aramco_marketaux.json"

# Search terms for Aramco
SEARCH_TERMS = [
    "Aramco",
    "Saudi Aramco",
    "Saudi Arabian Oil Company"
]

# Get API key from environment variable or set it here
API_KEY = os.getenv("MARKETAUX_API_KEY", "iQAXIrPvqBDP7VVxBlT54NJ8yC8fTQ3HbXEfOt50")
BASE_URL = "https://api.marketaux.com/v1/news/all"

def fetch_aramco_news(api_key=None, days_back=30, max_articles=100):
    """
    Fetch news articles about Aramco from Marketaux.
    
    Args:
        api_key (str): Marketaux API key
        days_back (int): Number of days to look back for news
        max_articles (int): Maximum number of articles to fetch
    
    Returns:
        dict: Dictionary containing all fetched news articles
    """
    if api_key is None:
        api_key = API_KEY
    
    if not api_key or api_key == "YOUR_MARKETAUX_API_KEY_HERE":
        print("⚠️  Warning: Please set your Marketaux API key!")
        print("   Set environment variable: export MARKETAUX_API_KEY='your_key'")
        print("   Or edit the API_KEY variable in this script")
        return None
    
    print(f"Fetching Aramco news from Marketaux...")
    print(f"Looking back {days_back} days...")
    
    all_articles = []
    seen_urls = set()
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    # Search for each term
    for term in SEARCH_TERMS:
        print(f"\nSearching for: '{term}'...")
        
        page = 1
        max_pages = 10  # Limit pages to avoid too many requests
        
        while page <= max_pages and len(all_articles) < max_articles:
            try:
                # Prepare request parameters
                params = {
                    'api_token': api_key,
                    'search': term,
                    'countries': 'sa,us,gb',  # Saudi Arabia, US, UK
                    'filter_entities': 'true',
                    'limit': 3,  # Reduced limit based on API plan
                    'page': page
                }
                
                # Make API request
                response = requests.get(BASE_URL, params=params, timeout=30)
                
                # Check HTTP status
                if response.status_code != 200:
                    print(f"  ✗ HTTP Error {response.status_code}: {response.text[:200]}")
                    break
                
                data = response.json()
                
                # Check for errors in response
                if 'error' in data:
                    error_info = data.get('error', {})
                    error_msg = error_info.get('message', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
                    print(f"  ✗ API Error: {error_msg}")
                    break
                
                # Marketaux returns data directly, not wrapped in status
                articles = data.get('data', [])
                
                if articles:
                    count = len(articles)
                    print(f"  Page {page}: Found {count} articles")
                    
                    if count == 0:
                        break  # No more articles
                    
                    # Process articles
                    for article in articles:
                        url = article.get('url', '')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            
                            # Extract entities if available
                            entities = []
                            if 'entities' in article:
                                for entity in article['entities']:
                                    entities.append({
                                        'symbol': entity.get('symbol', ''),
                                        'name': entity.get('name', ''),
                                        'exchange': entity.get('exchange', ''),
                                        'type': entity.get('type', '')
                                    })
                            
                            all_articles.append({
                                'source': 'Marketaux',
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'snippet': article.get('snippet', ''),
                                'url': url,
                                'published_at': article.get('published_at', ''),
                                'source_name': article.get('source', ''),
                                'entities': entities,
                                'search_term': term
                            })
                            
                            if len(all_articles) >= max_articles:
                                break
                    
                    # Check if there are more pages
                    meta = data.get('meta', {})
                    found = meta.get('found', 0)
                    returned = meta.get('returned', 0)
                    if returned == 0 or found <= page * 3:
                        break
                    
                    page += 1
                else:
                    error_msg = data.get('error', {}).get('message', 'Unknown error')
                    print(f"  ✗ API Error: {error_msg}")
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"  ✗ Request error: {e}")
                break
            except Exception as e:
                print(f"  ✗ Error searching for '{term}': {e}")
                break
        
        if len(all_articles) >= max_articles:
            break
    
    # Remove duplicates and sort by date
    unique_articles = []
    seen_titles = set()
    for article in all_articles:
        title_lower = article['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)
    
    # Sort by published date (newest first)
    unique_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
    
    # Prepare output
    output_data = {
        'metadata': {
            'source': 'Marketaux',
            'fetched_at': datetime.now().isoformat(),
            'search_terms': SEARCH_TERMS,
            'total_articles': len(unique_articles),
            'date_range': {
                'from': from_date.isoformat(),
                'to': to_date.isoformat()
            }
        },
        'articles': unique_articles[:max_articles]
    }
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Successfully fetched {len(output_data['articles'])} unique articles")
    print(f"✓ Data saved to: {OUTPUT_FILE}")
    
    return output_data

def print_summary(data):
    """Print a summary of fetched news."""
    if data is None:
        return
    
    print("\n" + "="*60)
    print("NEWS SUMMARY")
    print("="*60)
    print(f"Total Articles: {data['metadata']['total_articles']}")
    print(f"Date Range: {data['metadata']['date_range']['from'][:10]} to {data['metadata']['date_range']['to'][:10]}")
    print("\nRecent Articles:")
    for i, article in enumerate(data['articles'][:5], 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source_name']}")
        print(f"   Published: {article['published_at'][:10] if article.get('published_at') else 'N/A'}")
        if article.get('entities'):
            symbols = [e.get('symbol', '') for e in article['entities'] if e.get('symbol')]
            if symbols:
                print(f"   Entities: {', '.join(symbols)}")
        print(f"   URL: {article['url']}")

if __name__ == "__main__":
    # Fetch news
    news_data = fetch_aramco_news(days_back=30, max_articles=100)
    
    # Print summary
    if news_data:
        print_summary(news_data)
    else:
        print("\n⚠️  No news data fetched. Please check your API key and try again.")
