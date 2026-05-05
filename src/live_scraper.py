import os
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Initialize with your API Key
# Get a free key at: https://newsapi.org/
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

# The "Credible Sources" list - ensuring 1st class quality
TRUSTED_SOURCES = 'reuters,bloomberg,cnbc,the-wall-street-journal,fortune,business-insider'

def fetch_live_nasdaq_news(tickers):
    all_articles = []
    
    # We look for news from the last 30 days (Developer free tier limit)
    from_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
    
    print(f"📡 Fetching live news from {TRUSTED_SOURCES}...")

    for ticker in tickers:
        try:
            # Query NewsAPI for the specific ticker
            response = newsapi.get_everything(
                q=ticker,
                sources=TRUSTED_SOURCES,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=10  # Top 10 latest articles per ticker
            )

            if response['status'] == 'ok':
                articles = response['articles']
                for art in articles:
                    all_articles.append({
                        "ticker": ticker,
                        "date": art['publishedAt'],
                        "headline": art['title'],
                        "source": art['source']['name'],
                        "url": art['url']
                    })
                print(f"Found {len(articles)} articles for {ticker}")
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return pd.DataFrame(all_articles)

if __name__ == "__main__":
    # Use your Big Tech targets
    targets = TARGETS = [
    # --- TECHNOLOGY & SEMIS ---
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'ADBE', 'CRM', 'AMD', 
    'INTC', 'CSCO', 'QCOM', 'ORCL', 'TXN',
    
    # --- FINANCIALS ---
    'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'AXP', 'BLK',
    
    # --- HEALTHCARE ---
    'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'AMGN', 'BMY', 'GILD',
    
    # --- CONSUMER (RETAIL/STAPLES/DISC) ---
    'WMT', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'DIS', 'SBUX', 'TGT', 'HD', 'NFLX',
    
    # --- ENERGY & INDUSTRIALS ---
    'XOM', 'CVX', 'GE', 'BA', 'CAT', 'HON', 'UPS', 'MMM'
]
    
    df_live = fetch_live_nasdaq_news(targets)
    
    if not df_live.empty:
        # Save this to your data folder
        output_file = "../data/live_test_data.csv"
        df_live.to_csv(output_file, index=False)
        print(f"\n💾 Saved {len(df_live)} live articles to {output_file}")