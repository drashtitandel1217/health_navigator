import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from collections import Counter

def run_symptom_retraining_pipeline():
    """
    Natively retrains the Random Forest Classifier on historical data 
    extracted from the ExcelPatientRecord table in SQLite.
    """
    # Now it is fully safe to import your database models contextually!
    from patients.models import ExcelPatientRecord

    print("📦 Extracting columns from SQLite database...")
    records = ExcelPatientRecord.objects.values('primary_diagnosis', 'medical_history_summary')
    df = pd.DataFrame(list(records))
    
    # Secure the file path relative to this script file to ensure it lands in the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'symptom_classifier_model.pkl')

    if df.empty:
        print("❌ Error: Your dataset table is empty. Load your Excel file first!")
        return False

    # Clean up missing data gaps safely
    df['medical_history_summary'] = df['medical_history_summary'].fillna("").astype(str)
    df['primary_diagnosis'] = df['primary_diagnosis'].fillna("General Observation").astype(str)

    # 📊 DATA DIAGNOSTIC BLOCK 1: Class Distribution
    print("\n--- 📊 DATASET CLASS DISTRIBUTION ---")
    class_counts = Counter(df['primary_diagnosis'])
    for diagnosis, count in class_counts.most_common(10):
        print(f"🔹 {diagnosis}: {count} records")
    print("--------------------------------------\n")

    # 📊 DATA DIAGNOSTIC BLOCK 2: Check for blank/generic features
    blank_summaries = df[df['medical_history_summary'].str.strip() == ""].shape[0]
    default_summaries = df[df['medical_history_summary'].str.lower().str.contains("no summary", na=False)].shape[0]
    print(f"⚠️ Empty medical summaries: {blank_summaries} rows")
    print(f"⚠️ 'No summary provided' placeholders: {default_summaries} rows")
    print(f"📝 Total valid rows for training: {df.shape[0]}\n")

    # Define X (Input Text) and y (Target Diagnosis Class)
    X = df['medical_history_summary']
    y = df['primary_diagnosis']

    # Split into train/test sets to evaluate accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("🏗️ Initializing NLP Vectorizer & Random Forest Classifier Pipeline...")
    symptom_classifier_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            lowercase=True, 
            max_features=1000,
            ngram_range=(1, 1),        
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{3,}\b' 
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=None,          
            class_weight='balanced', 
            random_state=42
        ))
    ])
    
    print("🏋️‍♂️ Training model to recognize diagnoses based on historical patterns...")
    symptom_classifier_pipeline.fit(X_train, y_train)

    # Calculate verification accuracy score
    try:
        accuracy = symptom_classifier_pipeline.score(X_test, y_test)
        print(f"🎯 Model Training Complete! Testing Accuracy: {accuracy * 100:.1f}%")
    except Exception as eval_err:
        print(f"⚠️ Testing split too small to calculate accuracy score, skipping valuation step.")

    # Save the fresh classification model binary to disk
    joblib.dump(symptom_classifier_pipeline, model_path)
    print(f"✅ Model saved to: {model_path}")
    return True

# 🚀 STANDALONE RUN BACKWARD-COMPATIBILITY
# This block runs ONLY if you type `python train_classifier.py` manually in the terminal
if __name__ == "__main__":
    import django
    print("⚙️ Initializing standalone Django bootstrap environment...")
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_navigator.settings')
    django.setup()
    run_symptom_retraining_pipeline()