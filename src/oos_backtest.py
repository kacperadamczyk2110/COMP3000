import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

def run_final_academic_backtest():
    print("⚙️ Initializing Final Out-of-Sample Backtest...")
    
    # 1. DATA PREP 
    df = pd.read_csv('data/processed/feature_matrix.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.copy() # Silence pandas warnings
    
    # Calculate Macro Trend (100-day MA) and 5-Day Sentiment
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['s_val_ma'] = df['s_val'].rolling(window=5).mean()
    df['market_return'] = df['Close'].pct_change()
    
    df = df.dropna().copy()
    
    # Define features (Drop things that give away the future or current price)
    ignore_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'forward_return', 'target', 'market_return']
    features = [c for c in df.columns if c not in ignore_cols]
    
    # 2. STRICT CHRONOLOGICAL SPLIT (70% Train / 30% Test)
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"🧠 Training on {len(train_df)} past days...")
    print(f"🚀 Trading on {len(test_df)} unseen future days...")
    
    # 3. MODEL TRAINING 
    model = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(train_df[features], train_df['target'])
    
    # 4. PREDICTION & SIZING 
    test_df['ai_prob'] = model.predict_proba(test_df[features])[:, 1]
    
    def calculate_realistic_position(row):
        prob = row['ai_prob']
        macro_is_bull = row['Close'] > row['MA100']
        is_crisis = row.get('regime_Crisis', 0) > 0
        
        # INSTITUTIONAL VETO: Panic mode
        if is_crisis and row['impact'] > 5.0:
            return 0.0 # Force Cash
            
        # DYNAMIC SIZING BASED ON CONFIDENCE
        if prob > 0.65 and macro_is_bull and row['s_val_ma'] > 0.1:
            return 1.5 # High Conviction: 150% Leveraged
        elif prob > 0.52:
            return 1.0 # Standard Conviction: 100% Long
        else:
            return 0.0 # Low Conviction: Cash
            
    test_df['target_position'] = test_df.apply(calculate_realistic_position, axis=1)
    
    # 5. REAL-WORLD COSTS 
    FEE = 0.001 
    MARGIN_RATE = 0.06 / 252 
    
    test_df['position_changes'] = test_df['target_position'].diff().abs().fillna(0)
    test_df['borrowed_funds'] = np.where(test_df['target_position'] > 1.0, test_df['target_position'] - 1.0, 0.0)
    
    # Shift(1) is critical: Trade tomorrow based on today's target
    gross_return = test_df['market_return'] * test_df['target_position'].shift(1)
    trading_costs = test_df['position_changes'].shift(1) * FEE
    margin_costs = test_df['borrowed_funds'].shift(1) * MARGIN_RATE
    
    test_df['ai_net_return'] = gross_return - trading_costs - margin_costs
    
    # 6. CALCULATE OOS PERFORMANCE 
    test_df['Buy_Hold'] = (1 + test_df['market_return'].fillna(0)).cumprod()
    test_df['AI_Strategy'] = (1 + test_df['ai_net_return'].fillna(0)).cumprod()
    
    def get_stats(returns):
        returns = returns.dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        cum_ret = (1 + returns).cumprod()
        dd = (cum_ret / cum_ret.cummax() - 1).min() * 100
        return sharpe, dd

    bh_sharpe, bh_dd = get_stats(test_df['market_return'])
    ai_sharpe, ai_dd = get_stats(test_df['ai_net_return'])

    # 7. PLOTTING THE UNSEEN FUTURE
    plt.figure(figsize=(14, 7), dpi=150)
    
    plt.plot(test_df.index, test_df['Buy_Hold'], label=f"NASDAQ Buy & Hold (Sharpe: {bh_sharpe:.2f}, DD: {bh_dd:.1f}%)", color='black', alpha=0.4)
    plt.plot(test_df.index, test_df['AI_Strategy'], label=f"AI Out-of-Sample Alpha (Sharpe: {ai_sharpe:.2f}, DD: {ai_dd:.1f}%)", color='#27ae60', linewidth=2.5)
    
    plt.title('Final Academic Result: Out-of-Sample Performance (Net of Real-World Fees)', fontsize=15, fontweight='bold')
    plt.ylabel('Growth Multiplier (Starting at 1.0)')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.2)
    plt.show()

    print("\n--- OUT-OF-SAMPLE RESULTS (The 'Unseen' 30%) ---")
    print(f"Buy and Hold Multiplier: {test_df['Buy_Hold'].iloc[-1]:.3f}x")
    print(f"AI Strategy Multiplier:  {test_df['AI_Strategy'].iloc[-1]:.3f}x")

if __name__ == "__main__":
    run_final_academic_backtest()