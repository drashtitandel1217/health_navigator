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
    queryset = ExcelPatientRecord.objects.all().values()
    df = pd.DataFrame(list(queryset))
    
    if df.empty:
        print("No data found in the database. Run your import script first!")
        return

    print(f" Successfully loaded {len(df)} records into Pandas.")

    print("Engineering datetime spans and patient age metrics...")

    df['admission_dt'] = pd.to_datetime(df['date_of_admission'], errors='coerce')
    df['discharge_dt'] = pd.to_datetime(df['date_of_discharge'], errors='coerce')
    df['dob_dt'] = pd.to_datetime(df['date_of_birth'], errors='coerce')

    df['length_of_stay'] = (df['discharge_dt'] - df['admission_dt']).dt.days
    df['age_at_admission'] = (df['admission_dt'].dt.year - df['dob_dt'].dt.year)
    
    df['medical_history_summary'] = df['medical_history_summary'].fillna("").astype(str)

    df = df.dropna(subset=['length_of_stay', 'age_at_admission', 'primary_diagnosis', 'attending_physician'])

    y = df['length_of_stay'].values
    X = df.drop(columns=[
        'id', 'medical_record_number', 'patient_name', 'date_of_birth', 
        'date_of_admission', 'date_of_discharge', 'admission_dt', 
        'discharge_dt', 'dob_dt', 'length_of_stay'
    ], errors='ignore')

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
    print(f" Original Input Attributes Shape: {X_train.shape}")
    print(f" Processed High-Dimensional Matrix Shape: {X_train_processed.shape}")
    print("Your data is now 100% numerical and ready to be loaded directly into any AI model matrix.")
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset()