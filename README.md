# Institutional Alpha: A Dual-Pass AI Engine for Risk-Adjusted Quantitative Trading

## Project Overview
This project develops a high-complexity financial intelligence pipeline that transitions from raw Natural Language Processing (NLP) to institutional-grade risk management. It utilizes a **Dual-Pass AI architecture** to filter market noise and a **Kelly Criterion** engine to optimize capital allocation on the NASDAQ-100.

## Technical Architecture

### 1. Dual-Pass NLP Pipeline
* **Pass 1 (FinBERT):** Rapid extraction of raw sentiment scores from a 23GB financial news dataset.
* **Pass 2 (Llama 3.1):** Strategic context filtering to classify **Market Regimes** (Earnings, Macro, Crisis) and assign **Impact Scores** (1–5).

### 2. Institutional Risk Manager
* **Kelly Criterion:** Mathematically optimized position sizing based on AI confidence intervals.
* **The Veto Logic:** A binary safety switch monitoring the **Put/Call Ratio (PCR)**. If PCR > 1.2, the system automatically aborts all positions to 100% Cash to preserve capital.

### 3. Empirical Evaluation
* **Out-of-Sample (OOS) Testing:** Strict 70/30 chronological split to ensure zero data leakage.
* **Real-World Friction:** Integration of **0.1% transaction fees** and **6% annualised margin interest** for rigorous academic validity.

## Tech Stack
* **Languages:** Python 3.10+
* **AI/NLP:** Hugging Face (FinBERT), Ollama (Llama 3.1), Scikit-Learn (Random Forest)
* **Data Science:** Pandas, NumPy
* **Visualisation:** Matplotlib

## Core Mathematical Framework

### The Strategic Signal (Sigmoid)
$$P(Up) = \frac{1}{1 + e^{-(\text{Sentiment} \times \text{Impact})}}$$

### The Kelly Criterion
$$K = W - \frac{1 - W}{R}$$

### The Institutional Veto
$$\text{Allocation} = K \times (\text{PCR} < 1.2)$$

## File Structure
* `master_gate.py`: The primary intelligence bridge and NLP pipeline.
* `backtester.py`: The core execution engine and Kelly sizing logic.
* `oos_backtest.py`: The out-of-sample evaluation rig with friction models.
* `rf_trainer.py`: Random Forest model for feature importance analysis.
* `run_miner.py`: Large-scale data extraction and preprocessing.

## Installation & Usage
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Ensure Ollama is running with the `llama3.1` model.
4.  Execute the backtest: `python oos_backtest.py`

## Conclusion
The system successfully proves that **Strategic Context** is a more reliable predictor than raw sentiment. While real-world friction (0.1% fees) impacts net alpha, the **Veto Logic** effectively reduced Maximum Drawdown, demonstrating superior capital preservation compared to benchmark strategies.