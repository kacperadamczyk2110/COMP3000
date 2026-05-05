from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Singleton pattern: Load once per session
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        
    conf, class_idx = torch.max(probs, dim=-1)
    label = model.config.id2label[class_idx.item()]
    confidence = conf.item()

    # THE CONFIDENCE GUARD
    if confidence < 0.70:
        return {'label': 'neutral', 'confidence': confidence}
        
    return {'label': label.lower(), 'confidence': confidence}