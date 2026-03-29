import logging
from typing import Dict, Optional
from transformers import pipeline

# 1. Setup Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. Load the FinBERT Model
# We do this globally so the computer only has to load the heavy AI model once, not every time we send a headline.
logger.info("Initializing FinBERT locally... (This might take a moment if downloading for the first time)")
try:
    # 'ProsusAI/finbert' is the industry standard open-source financial sentiment model
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    logger.info("FinBERT successfully loaded into memory!")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load FinBERT. Error: {str(e)}")
    sentiment_analyzer = None

def get_financial_sentiment(headline: str) -> Optional[Dict[str, any]]:
    """
    Passes a factual financial headline to FinBERT.
    Returns a dictionary with the label (positive/negative/neutral) and confidence score.
    """
    if not sentiment_analyzer:
        return None
        
    try:
        # Ask FinBERT to analyze the text
        result = sentiment_analyzer(headline)[0]
        
        return {
            "label": result["label"].upper(),  # POSITIVE, NEGATIVE, or NEUTRAL
            "confidence": round(result["score"], 4) # A percentage of how sure the AI is
        }
    except Exception as e:
        logger.error(f"Failed to analyze headline '{headline}'. Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Task 4: Test FinBERT with some pure, factual financial data
    test_headlines = [
        "Apple reports Q3 earnings of $1.20 per share, vastly exceeding analyst expectations.",
        "The company filed for Chapter 11 bankruptcy after massive accounting fraud was discovered.",
        "The Federal Reserve maintained interest rates at 5.25%."
    ]
    
    logger.info("Testing FinBERT Market Signals...")
    print("-" * 60)
    
    for headline in test_headlines:
        result = get_financial_sentiment(headline)
        if result:
            logger.info(f"Signal: {result['label']:<8} | Certainty: {result['confidence']:.2f} | Headline: '{headline}'")