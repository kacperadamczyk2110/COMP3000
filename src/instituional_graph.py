import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

def plot_institutional_engine():
    print("Loading Data and Simulating Institutional Pipeline...")
    
    # 1. Load Data
    df = pd.read_csv('data/processed/feature_matrix.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.copy()
    
    df['market_return'] = df['Close'].pct_change()
    
    # Annualized 10-day volatility (Used for position scaling and PCR simulation)
    df['volatility_10d'] = df['market_return'].rolling(window=10).std() * np.sqrt(252) * 100 
    df = df.dropna().copy()
    
    # 2. Surrogate Deep Learning Brain (Mimicking the Transformer)
    ignore_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'forward_return', 'target', 'market_return']
    features = [c for c in df.columns if c not in ignore_cols]
    
    print("Training Surrogate Probability Engine...")
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(df[features], df['target'])
    
    # Get the raw probabilities
    df['ai_prob'] = model.predict_proba(df[features])[:, 1]
    
    # 3. Simulate Options Data (Put/Call Ratio)
    df['simulated_pcr'] = 0.8 + (df['volatility_10d'] / 100) + np.where(df['market_return'] < -0.015, 0.3, 0.0)

    # 4. The Kelly Criterion Risk Manager
    def apply_institutional_risk(row):
        prob = row['ai_prob']
        pcr = row['simulated_pcr']
        vol = row['volatility_10d']
        
        # VETO: If Put/Call ratio is too high, institutions are panicking. Abort to Cash.
        if pcr > 1.2:
            return 0.0
            
        # VETO: If AI is uncertain (< 51%), stay in Cash
        if prob < 0.51:
            return 0.0
            
        # KELLY FORMULA (Assuming historical Win/Loss Ratio of 1.3)
        W = prob
        R = 1.3
        kelly_fraction = W - ((1 - W) / R)
        
        # Half-Kelly for safety
        optimal_leverage = (kelly_fraction / 2.0) * 10
        
        # Volatility penalty: Reduce leverage in chaotic markets
        if vol > 20.0:
            optimal_leverage *= 0.5
            
        # Cap at 1.5x Margin
        return np.clip(optimal_leverage, 0.0, 1.5)

    df['final_allocation'] = df.apply(apply_institutional_risk, axis=1)

    # 5. Execute Trades with Fees
    FEE = 0.001
    MARGIN_RATE = 0.06 / 252
    
    df['position_changes'] = df['final_allocation'].diff().abs().fillna(0)
    df['borrowed_funds'] = np.where(df['final_allocation'] > 1.0, df['final_allocation'] - 1.0, 0.0)
    
    gross_return = df['market_return'] * df['final_allocation'].shift(1)
    trading_costs = df['position_changes'].shift(1) * FEE
    margin_costs = df['borrowed_funds'].shift(1) * MARGIN_RATE
    
    df['ai_net_return'] = gross_return - trading_costs - margin_costs
    
    df['Buy_Hold'] = (1 + df['market_return'].fillna(0)).cumprod()
    df['Institutional_Engine'] = (1 + df['ai_net_return'].fillna(0)).cumprod()

    # 6. PLOTTING
    print("📊 Generating Enhanced Visualizations...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, dpi=150)
    
    # --- Chart 1: Cumulative Returns ---
    ax1.plot(df.index, df['Buy_Hold'], label=f"NASDAQ (Benchmark)", color='black', alpha=0.3)
    ax1.plot(df.index, df['Institutional_Engine'], label=f"AI Engine", color='#8e44ad', linewidth=2)
    
    # Highlight Veto zones on the main return chart
    veto_zones = df[df['simulated_pcr'] > 1.2].index
    for date in veto_zones:
        ax1.axvspan(date, date + pd.Timedelta(days=1), color='red', alpha=0.1)

    ax1.set_title('Institutional Alpha Architecture: Strategy vs. Benchmark', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Growth Multiplier')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.15)

    # --- Chart 2: Dynamic Allocation & Veto Intensity ---
    ax2.fill_between(df.index, 0, df['final_allocation'], color='#9b59b6', alpha=0.4, label='Portfolio Exposure')
    
    # Use vertical spans for the Veto - much easier to see than 'x' markers
    label_added = False
    for date in veto_zones:
        ax2.axvspan(date, date + pd.Timedelta(days=1), color='red', alpha=0.3, 
                    label='Options Market Veto (Panic)' if not label_added else "")
        label_added = True

    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='100% Equity (No Margin)')
    ax2.set_ylabel('Allocation (1.5 = 150%)')
    ax2.set_ylim(0, 1.7)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_institutional_engine()