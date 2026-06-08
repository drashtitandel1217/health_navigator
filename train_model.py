import os
import django
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess_data import load_and_preprocess_dataset

def train_predictive_ai():
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset()
    
    print("\n Initializing Random Forest AI Engine...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    
    print(" Training the AI model on your 2,001 patient records (running matrix calculations)...")
    model.fit(X_train, y_train)
    print(" Model training complete!")

    print(" Generating predictions on the held-out testing matrix...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n📋 Sample Predictions vs Real Data (First 5 Test Patients):")
    for i in range(5):
        print(f"Patient {i+1} ──► Actual Stay: {y_test[i]} days | AI Predicted Stay: {predictions[i]:.1f} days")

if __name__ == "__main__":
    train_predictive_ai()