import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from tqdm import tqdm
import json

# 1. PASS ONE: FINBERT SENTIMENT 
def run_finbert_pass(input_path, temp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 PASS 1: FinBERT Sentiment Analysis on {device.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    
    df = pd.read_csv(input_path)
    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="FinBERT"):
        inputs = tokenizer(str(row['summary'])[:512], return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        conf, class_idx = torch.max(probs, dim=-1)
        
        results.append({
            'ticker': row['ticker'],
            'date': row['date'],
            'headline': row['headline'],
            'summary': row['summary'],
            'sentiment': model.config.id2label[class_idx.item()].lower(),
            'confidence': conf.item()
        })
    
    pd.DataFrame(results).to_csv(temp_path, index=False)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("✅ Pass 1 Complete. VRAM Cleared.")

# 2. PASS TWO: LLAMA METRICS
def run_llama_pass(temp_path, final_path):
    print("🚀 PASS 2: Local Llama Strategic Metrics...")
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    
    df = pd.read_csv(temp_path)
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Llama"):
        # Check if row is already processed (Resume Logic)
        if os.path.exists(final_path):
             # (Add your resume logic here if needed)
             pass

        context = f"H: {row['headline']} | S: {str(row['summary'])[:400]}"
        prompt = f"Analyze {row['ticker']} news. Return ONLY JSON: {{'relevance': 1-5, 'impact': 1-5, 'regime': 'Earnings/Macro/AI-Hype/Crisis/General'}}. Context: {context}"
        
        try:
            completion = client.chat.completions.create(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            metrics = json.loads(completion.choices[0].message.content)
            
            result = {
                **row.to_dict(),
                'relevance': metrics.get('relevance', 1),
                'impact': metrics.get('impact', 1),
                'regime': metrics.get('regime', 'General')
            }
            
            pd.DataFrame([result]).to_csv(final_path, mode='a', index=False, header=not os.path.exists(final_path))
        except:
            continue

if __name__ == "__main__":
    INPUT = 'data/full_context_clean.csv' 
    TEMP = 'data/temp_finbert_scores.csv'
    FINAL = 'data/scored_final_master.csv'
    
    # Run them sequentially
    run_finbert_pass(INPUT, TEMP)
    run_llama_pass(TEMP, FINAL)