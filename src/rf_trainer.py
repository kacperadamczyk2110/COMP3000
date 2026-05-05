import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_predict():
    df = pd.read_csv('data/processed/feature_matrix.csv', index_col=0)
    
    # 1. Select Features
    feature_cols = [col for col in df.columns if 'regime_' in col] + \
                   ['s_val', 'confidence', 'impact', 'relevance']
    
    X = df[feature_cols]
    y = df['target']
    
    # 2. Time-Series Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 3. Train the Model
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Prediction Accuracy on Test Set: {acc:.2%}")
    print("\nTop Drivers of Price Movement:")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(10))
    
    # Save the model
    joblib.dump(model, 'outputs/news_model.pkl')
    return model, X_test, y_test

if __name__ == "__main__":
    train_and_predict()