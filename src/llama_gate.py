import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_sensationalism_score(headline):
    prompt = f"""
    Act as a Senior Quantitative Analyst. Rate the 'Market Impact Potential' 
    of this headline on a scale of 1 to 10.
    1-3: Noise. 5-7: Significant. 10: Market Shifting.
    HEADLINE: "{headline}"
    Return ONLY the integer.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",
            temperature=0.0,
            max_tokens=5
        )
        score = chat_completion.choices[0].message.content.strip()
        return int(score)
    except:
        return 3 # Default to low-impact noise if API fails