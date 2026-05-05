import pandas as pd
import yfinance as yf
import argparse
import os
from datetime import timedelta

def run_fetcher(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"🔍 Analyzing news dates in: {input_path}")
    
    try:
        # 1. Load the scored data
        df = pd.read_csv(input_path)
        
        # 2. Convert and Clean Dates
        df['date'] = pd.to_datetime(df['date'].astype(str).str.replace(' UTC', ''), utc=True)
        
        # 3. Identify the "First" and "Last"
        first_date = df['date'].min()
        last_date = df['date'].max()
        
        # 4. Add "Padding" for better backtesting
        fetch_start = (first_date - timedelta(days=5)).strftime('%Y-%m-%d')
        fetch_end = (last_date + timedelta(days=2)).strftime('%Y-%m-%d')
        
        print(f"First News Headline: {first_date.date()}")
        print(f"Last News Headline:  {last_date.date()}")
        print(f"Fetching QQQ data from {fetch_start} to {fetch_end}...")

        # 5. Download from Yahoo Finance
        ticker = yf.Ticker("QQQ")
        hist = ticker.history(start=fetch_start, end=fetch_end, interval="1d")

        if hist.empty:
            print("Error: No market data retrieved. Check ticker symbol or dates.")
            return

        # 6. Save and Report
        hist.to_csv(output_path)
        print(f"Success! Market data saved to: {output_path}")
        print(f"Total Trading Days Captured: {len(hist)}")

    except Exception as e:
        print(f"Failed to sync market data: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/scored_final_master.csv')
    parser.add_argument('--output', default='data/QQQ.csv')
    args = parser.parse_args()
    
    run_fetcher(args.input, args.output)

if __name__ == "__main__":
    main()