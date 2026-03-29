import pandas as pd
import logging
import time

# Import your two custom AI engines
from llama_gate import get_sensationalism_score
from finbert_model import get_financial_sentiment

# Setup Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# The Threshold: What score is "Too Sensational" for our trading bot?
# If a headline scores 7 or higher out of 10, we consider it "Fake News / Hype" and block it.
SENSATIONALISM_THRESHOLD = 7 

def run_pipeline():
    logger.info("🚀 Starting the Hybrid AI Trading Pipeline...")
    
    # Let's simulate a raw feed of internet data (The Good, The Bad, and The Hype)
    raw_news_feed = [
        "Federal Reserve maintains interest rates at 5.25%.",
        "CRASH IMMINENT: Wall Street panics as tech stocks face total BLOODBATH! 🚨",
        "Apple reports Q3 earnings of $1.20 per share.",
        "ELON MUSK TWEETS! Dogecoin is going to the MOON! Buy now before you stay poor forever!!!",
        "Tesla vehicle deliveries declined by 4% in the first quarter of 2024."
    ]
    
    processed_data = []
    
    for headline in raw_news_feed:
        logger.info("-" * 60)
        logger.info(f"Processing: '{headline}'")
        
        # --- STAGE 1: THE GATEKEEPER (Llama 3.1) ---
        logger.info("   -> [Stage 1] Checking Sensationalism...")
        score = get_sensationalism_score(headline)
        
        if score is None:
            logger.warning("   -> API Error. Skipping headline.")
            continue
            
        logger.info(f"   -> Llama Score: {score}/10")
        
        # --- THE DECISION ---
        if score >= SENSATIONALISM_THRESHOLD:
            logger.warning(f"   -> REJECTED: Headline is too sensational (Score: {score}). Protecting trading bot.")
            # We record that it was blocked, and move to the next headline
            processed_data.append({
                "Headline": headline,
                "Sensationalism_Score": score,
                "FinBERT_Signal": "BLOCKED",
                "Confidence": 0.0,
                "Action_Taken": "Rejected by Gate"
            })
            continue # Skip Stage 2!
            
        # --- STAGE 2: THE SPECIALIST (FinBERT) ---
        logger.info("   -> APPROVED: Factual news detected. Sending to FinBERT...")
        sentiment_data = get_financial_sentiment(headline)
        
        if sentiment_data:
            logger.info(f"   -> FinBERT Signal: {sentiment_data['label']} (Certainty: {sentiment_data['confidence']:.2f})")
            
            processed_data.append({
                "Headline": headline,
                "Sensationalism_Score": score,
                "FinBERT_Signal": sentiment_data['label'],
                "Confidence": sentiment_data['confidence'],
                "Action_Taken": "Processed successfully"
            })
            
        # Add a tiny 1-second delay so we don't overwhelm the free Groq API limits
        time.sleep(1)

    # --- STAGE 3: GENERATE THE EVIDENCE (CSV) ---
    logger.info("-" * 60)
    logger.info("Pipeline Complete. Generating CSV Report...")
    
    # Convert our results into a Pandas DataFrame (Industry standard data structure)
    df = pd.DataFrame(processed_data)
    
    # Save it to the main project folder
    output_filename = "results_scored.csv"
    df.to_csv(f"../{output_filename}", index=False)
    
    logger.info(f"Success! Results saved to '{output_filename}'")

if __name__ == "__main__":
    run_pipeline()