import os
import logging
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

# 1. Setup Professional Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. Load the secret API key from the .env file
load_dotenv()

def get_sensationalism_score(headline: str) -> Optional[int]:
    """
    Sends a headline to Llama 3.1 via Groq to score its sensationalism from 0 to 10.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("CRITICAL: GROQ_API_KEY not found. Please check your .env file.")
        return None

    try:
        client = Groq(api_key=api_key)
        
        system_prompt = (
            "You are a strict quantitative financial analyst. "
            "Rate the following news headline on a scale of 0 to 10 for 'Sensationalism', "
            "where 10 is pure clickbait, emotional hype, or fear-mongering, and 0 is purely factual and objective. "
            "Return ONLY the integer. Do not provide any explanation, punctuation, or extra text."
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": headline}
            ],
            # UPDATED: Using Groq's newer, active Llama 3.1 model
            model="llama-3.1-8b-instant", 
            temperature=0.0, 
            max_tokens=5,    
        )
        
        score_str = response.choices[0].message.content.strip()
        if score_str.endswith('.'):
            score_str = score_str[:-1]
            
        return int(score_str)
        
    except Exception as e:
        logger.error(f"Failed to score headline '{headline}'. Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the Gate with examples
    test_headlines = [
        "Federal Reserve maintains interest rates at 5.25%.",
        "CRASH IMMINENT: Wall Street panics as tech stocks face total BLOODBATH! 🚨",
        "Apple reports Q3 earnings of $1.20 per share."
    ]
    
    logger.info("Initializing Llama 3.1 Sensationalism Gate via Groq...")
    
    for headline in test_headlines:
        score = get_sensationalism_score(headline)
        
        # Safely handle the print statement if the API fails
        if score is not None:
            logger.info(f"Score: {score:2d}/10 | Headline: '{headline}'")
        else:
            logger.warning(f"Could not calculate score for: '{headline}'")