import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import spacy

# ---------------------------------------------------------
# MODULE 1: Entity-Specific Sentiment (Cap-Weighting)
# ---------------------------------------------------------
class SentimentEngine:
    def __init__(self):
        # Load NLP model for Named Entity Recognition
        self.nlp = spacy.load("en_core_web_sm")
        
        # Top NASDAQ weights 
        self.weights = {'Apple': 0.11, 'Microsoft': 0.10, 'Nvidia': 0.05, 'DEFAULT': 0.01}

    def extract_weighted_score(self, headline, base_llama_score):
        """Multiplies sentiment by the market cap weight of the company mentioned."""
        doc = self.nlp(headline)
        companies_mentioned = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        # If a mega-cap is mentioned, apply its massive index weight
        for company in companies_mentioned:
            for key_ticker in self.weights.keys():
                if key_ticker.lower() in company.lower():
                    return base_llama_score * self.weights[key_ticker]
                    
        # Otherwise, treat it as minor market noise
        return base_llama_score * self.weights['DEFAULT']

# ---------------------------------------------------------
# MODULE 2: The Deep Learning Brain (Time-Series Transformer)
# ---------------------------------------------------------
class QuantTransformer(nn.Module):
    """
    Replaces the Random Forest. This model looks at a 10-day 
    window of data and understands the chronological sequence of events.
    """
    def __init__(self, num_features=10, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        # Projects our raw features into a higher-dimensional space
        self.embedding = nn.Linear(num_features, d_model)
        
        # The Transformer learns complex patterns over time
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final layer predicts the probability of the market going UP
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x is a tensor of shape [Batch, 10 Days, Features]
        x = self.embedding(x)
        memory = self.transformer(x)
        
        # Take the final day's 'memory' to predict tomorrow
        final_day_state = memory[:, -1, :] 
        prob_up = self.predictor(final_day_state)
        return prob_up

# ---------------------------------------------------------
# MODULE 3: The Execution Logic (Options Data + Kelly Criterion)
# ---------------------------------------------------------
class ExecutionEngine:
    def __init__(self, win_loss_ratio=1.2):
        self.R = win_loss_ratio # Historical Reward/Risk ratio

    def calculate_position(self, ai_probability, put_call_ratio, volatility):
        """
        Calculates exact portfolio leverage using Kelly Criterion, 
        but vetoes the trade if the smart money is buying Puts.
        """
        # 1. THE LEADING INDICATOR VETO (Options Market)
        
        if put_call_ratio > 1.2:
            return 0.0 
            
        # 2. THE KELLY CRITERION SIZING
        # Formula: K = W - [(1 - W) / R]
        W = ai_probability
        kelly_fraction = W - ((1 - W) / self.R)
        
        # Institutional safety parameter: Half-Kelly
        optimal_leverage = kelly_fraction / 2.0
        
        # 3. VOLATILITY SCALING
        if volatility > 20.0:
            optimal_leverage *= 0.5
            
        # Cap our maximum leverage at 150% (1.5x) and floor at 0.0 (No shorting)
        final_position = np.clip(optimal_leverage * 10, 0.0, 1.5)
        
        return final_position

# ---------------------------------------------------------
# THE MASTER PIPELINE
# ---------------------------------------------------------
def run_institutional_pipeline(headline, base_score, historical_tensor, pcr, vix):
    print("Initializing Institutional Pipeline...")
    
    # 1. Process Sentiment
    nlp_engine = SentimentEngine()
    weighted_score = nlp_engine.extract_weighted_score(headline, base_score)
    print(f"Cap-Weighted Sentiment Score: {weighted_score:.4f}")
    
    # 2. Get Deep Learning Probability
    brain = QuantTransformer(num_features=historical_tensor.shape[2])
    
    # In a real system, you'd load the weights: brain.load_state_dict(torch.load('model.pth'))
    with torch.no_grad():
        ai_prob = brain(historical_tensor).item()
    print(f"Transformer Win Probability: {ai_prob:.2%}")
    
    # 3. Execute Trade
    risk_manager = ExecutionEngine(win_loss_ratio=1.3)
    target_position = risk_manager.calculate_position(
        ai_probability=ai_prob, 
        put_call_ratio=pcr, 
        volatility=vix
    )
    
    print(f"Final Portfolio Allocation: {target_position * 100:.1f}%")

if __name__ == "__main__":
    # --- MOCK DATA FOR DEMONSTRATION ---
    sample_headline = "Apple announces breakthrough AI chip, markets rally."
    llama_score = 0.90
    
    # Simulated 10-day history with 10 features (Batch=1, Days=10, Features=10)
    mock_history = torch.rand(1, 10, 10) 
    
    current_pcr = 0.85 # Normal market conditions (No panic)
    current_vix = 14.5 # Low volatility
    
    run_institutional_pipeline(sample_headline, llama_score, mock_history, current_pcr, current_vix)