import csv
import os
import sys
from datetime import datetime
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# The 51 Ticker Quantitative Portfolio
TARGETS = {
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AVGO', 'ADBE', 'CRM', 'AMD', 
    'INTC', 'CSCO', 'QCOM', 'ORCL', 'TXN', 'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 
    'PYPL', 'AXP', 'BLK', 'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'AMGN', 'BMY', 'GILD',
    'WMT', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'DIS', 'SBUX', 'TGT', 'HD', 'NFLX',
    'XOM', 'CVX', 'GE', 'BA', 'CAT', 'HON', 'UPS', 'MMM'
}

def run_miner():
    input_file = 'data/nasdaq_exteral_data.csv'
    output_file = 'data/full_context_v2.csv'
    start_date = datetime(2019, 6, 1)
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return

    file_size = os.path.getsize(input_file)
    print(f"🚀 Mining 51 tickers from June 2019 (File Size: {file_size/1e9:.2f} GB)...")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        # Progress bar based on bytes read
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Searching 23GB")
        reader = csv.reader(f_in)
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['ticker', 'date', 'headline', 'summary'])
            
            for row in reader:
                # Update progress bar by the length of the string representation of the row
                pbar.update(len(','.join(row)))
                
                try:
                    if len(row) < 8: continue
                    ticker = row[3].strip().upper()
                    
                    if ticker in TARGETS:
                        raw_date_str = row[1].split(' ')[0]
                        row_date = datetime.strptime(raw_date_str, '%Y-%m-%d')
                        
                        if row_date >= start_date:
                            writer.writerow([ticker, row[1], row[2], row[7]])
                            count += 1
                except:
                    continue
        pbar.close()
            
    print(f"✨ Success! Extracted {count} context-rich rows to {output_file}")

if __name__ == "__main__":
    run_miner()