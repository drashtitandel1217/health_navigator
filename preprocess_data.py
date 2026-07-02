import os
import django
import pandas as pd
import numpy as np

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_navigator.settings')
django.setup()

from patients.models import ExcelPatientRecord
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_dataset():
    print(" Fetching records from SQLite database...")
    queryset = ExcelPatientRecord.objects.values(
        'patient_name',
        'medical_record_number',
        'date_of_birth',
        'date_of_admission',
        'date_of_discharge',
        'primary_diagnosis',
        'attending_physician',
        'medical_history_summary'
    )
    df = pd.DataFrame(list(queryset))
    
    if df.empty:
        print("No data found in the database. Run your import script first!")
        return

    print(f" Successfully loaded {len(df)} records into Pandas.")

    print("Engineering datetime spans and patient age metrics...")

    # Convert to datetime safely
    df['admission_dt'] = pd.to_datetime(df['date_of_admission'], errors='coerce')
    df['discharge_dt'] = pd.to_datetime(df['date_of_discharge'], errors='coerce')
    df['dob_dt'] = pd.to_datetime(df['date_of_birth'], errors='coerce')

    # 📊 Calculate columns
    df['length_of_stay'] = (df['discharge_dt'] - df['admission_dt']).dt.days
    df['age_at_admission'] = (df['admission_dt'].dt.year - df['dob_dt'].dt.year)
    
    # 🩹 SAFE FILL: Instead of .dropna(), we fill missing data so rows aren't deleted!
    df['length_of_stay'] = df['length_of_stay'].fillna(7)  # Fallback to a 7-day stay median
    df['length_of_stay'] = df['length_of_stay'].apply(lambda x: x if x > 0 else 5) # Fix negative calculations
    
    df['age_at_admission'] = df['age_at_admission'].fillna(45)  # Fallback to age 45 if birth date is blank
    df['primary_diagnosis'] = df['primary_diagnosis'].fillna("General Observation").astype(str)
    df['attending_physician'] = df['attending_physician'].fillna("Medical Staff").astype(str)
    df['medical_history_summary'] = df['medical_history_summary'].fillna("No summary provided").astype(str)

    # Separate inputs and targets
    y = df['length_of_stay'].values
    X = df[['primary_diagnosis', 'attending_physician', 'medical_history_summary', 'age_at_admission']]

    print(f" Readying arrays for training. Final clean count: {len(X)} matrix profiles.")
    print("Initializing parallel preprocessing tracks...")

    numerical_features = ['age_at_admission']
    categorical_features = ['primary_diagnosis', 'attending_physician']
    text_features = 'medical_history_summary'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('nlp', TfidfVectorizer(max_features=500, stop_words='english'), text_features)
        ]
    )

    print(" Splitting data arrays into 80% Train and 20% Test allocations...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(" Running transformations across feature matrices...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\n PREPROCESSING PIPELINE COMPLETE!")
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_dataset()