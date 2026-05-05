import pandas as pd
import numpy as np
import os

def build_feature_matrix():
    # 1. Load Data
    news_path = 'data/scored/scored_final_master.csv'
    price_path = 'data/market/QQQ.csv'
    
    news = pd.read_csv(news_path)
    prices = pd.read_csv(price_path)
    
    # 2. Clean Dates (Handling that ' UTC' string)
    news['date'] = pd.to_datetime(news['date'].astype(str).str.replace(' UTC', ''), utc=True).dt.date
    prices['Date'] = pd.to_datetime(prices['Date'], utc=True).dt.date
    
    # 3. Numeric Sentiment
    sent_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    news['s_val'] = news['sentiment'].map(sent_map)
    
    # 4. Binary Regimes (One-Hot Encoding)
    # This turns 'AI' into [1,0,0,0], 'Macro' into [0,1,0,0], etc.
    regime_dummies = pd.get_dummies(news['regime'], prefix='regime').astype(int)
    news = pd.concat([news, regime_dummies], axis=1)
    
    # 5. Daily Aggregation
    agg_rules = {
        's_val': 'mean',
        'confidence': 'mean',
        'impact': 'mean',
        'relevance': 'mean'
    }
    for col in regime_dummies.columns:
        agg_rules[col] = 'sum'
        
    daily_news = news.groupby('date').agg(agg_rules)
    
    # 6. Merge with Prices
    prices = prices.set_index('Date')
    df = prices.join(daily_news).fillna(0)
    
    # 7. Create the Prediction Target (3-Day Forward Return)
    df['forward_return'] = df['Close'].shift(-3) / df['Close'] - 1
    df['target'] = (df['forward_return'] > 0).astype(int)
    
    # Clean up
    df = df.dropna()
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/feature_matrix.csv')
    print(f"✅ Feature Matrix ready: {df.shape[0]} days, {df.shape[1]} features.")

if __name__ == "__main__":
    build_feature_matrix()