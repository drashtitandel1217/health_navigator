import os
import django
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 🚀 BOOTSTRAP SWITCH: Tell Python where your Django settings file is living
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_navigator.settings')
django.setup()

# NOW it is fully safe to import your database models!
from patients.models import ExcelPatientRecord

print("📦 Extracting columns from SQLite database...")
# Fetch only the columns we need to bypass the ghost 'id' field
records = ExcelPatientRecord.objects.values('primary_diagnosis', 'medical_history_summary')
df = pd.DataFrame(list(records))
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'symptom_classifier_model.pkl')

if df.empty:
    print("❌ Error: Your dataset table is empty. Load your Excel file first!")
    exit()

# Clean up missing data gaps safely
df['medical_history_summary'] = df['medical_history_summary'].fillna("").astype(str)
df['primary_diagnosis'] = df['primary_diagnosis'].fillna("General Observation").astype(str)

# Define X (Input Text) and y (Target Diagnosis Class)
X = df['medical_history_summary']
y = df['primary_diagnosis']

# Split into train/test sets to evaluate accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏗️ Initializing NLP Vectorizer & Random Forest Classifier Pipeline...")
symptom_classifier_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
])

print("🏋️‍♂️ Training model to recognize diagnoses based on historical patterns...")
symptom_classifier_pipeline.fit(X_train, y_train)

# Calculate verification accuracy score
accuracy = symptom_classifier_pipeline.score(X_test, y_test)
print(f"🎯 Model Training Complete! Testing Accuracy: {accuracy * 100:.1f}%")

# Save the fresh classification model binary to disk
joblib.dump(symptom_classifier_pipeline, model_path)
print(f"✅ Model saved to: {model_path}")