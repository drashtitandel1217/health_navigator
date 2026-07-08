import os
import joblib
import pandas as pd
import numpy as np
import spacy
import re
from .models import ExcelPatientRecord
from django.conf import settings

# 🧠 Safe spaCy Loader Block
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "length_of_stay_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "length_of_stay_model.pkl")

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "data_preprocessor.pkl")
if not os.path.exists(PREPROCESSOR_PATH):
    PREPROCESSOR_PATH = os.path.join(os.path.dirname(BASE_DIR), "data_preprocessor.pkl")

def parse_patient_from_query(user_text):
    if not nlp or not user_text:
        return None
    
    doc = nlp(user_text)

    # 🔍 1. MRN Pattern Fallback Scan (e.g., L999, A123)
    mrn_match = re.search(r"\b[a-zA-Z]\d{3}\b", user_text)
    if mrn_match:
        mrn = mrn_match.group(0).upper()
        return ExcelPatientRecord.objects.filter(medical_record_number=mrn).first()

    # 🧑‍⚕️ 2. Named Entity Recognition for patient names
    extracted_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    for name in extracted_names:
        patient = ExcelPatientRecord.objects.filter(patient_name__icontains=name).first()
        if patient:
            return patient

    # 🕵️‍♂️ 3. Final structural token split search
    for word in user_text.split():
        if len(word) > 2:
            patient = ExcelPatientRecord.objects.filter(patient_name__icontains=word).first()
            if patient:
                return patient

    return None

model = None
preprocessor = None

def load_model_binaries():
    # 📂 1. Get the directory path where utils.py lives (health_navigator_main/patients/)
    current_utils_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 📂 2. Move up one parent level to hit your project root directory (health_navigator_main/)
    project_root_dir = os.path.dirname(current_utils_dir)
    
    # 🎯 3. Lock down the absolute location of the model binary at the root level
    model_path = os.path.join(project_root_dir, 'symptom_classifier_model.pkl')

    # Quick terminal debug log to monitor server behavior
    print(f"🔍 System attempting to load model weights from: {model_path}")

    if os.path.exists(model_path):
        try:
            loaded_pipeline = joblib.load(model_path)
            print("🤖 Success: Random Forest Intent Weights loaded perfectly into memory cache!")
            return loaded_pipeline
        except Exception as read_err:
            print(f"❌ Error while reading .pkl file structure: {str(read_err)}")
            return None
    else:
        print("⚠️ Warning: File lookup failed. symptom_classifier_model.pkl is missing from root.")
        return None

load_model_binaries()

def predict_patient_stay(patient_object):
    """
    Takes a patient object, safely extracts features without invoking the missing 'id' column,
    and runs it through the preprocessor and Random Forest model.
    """
    global model, preprocessor
    if model is None or preprocessor is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                preprocessor = joblib.load(PREPROCESSOR_PATH)
            except Exception as e:
                print(f"Error loading stay forecasting model/preprocessor: {e}")
    if not model or not preprocessor:
        return "Model binaries (.pkl) are missing or not loaded correctly."

    try:
        # 🛡️ CRITICAL BYPASS: Use .values() to fetch only the explicit columns present in SQLite
        patient_data = ExcelPatientRecord.objects.filter(
            medical_record_number=patient_object.medical_record_number
        ).values(
            'primary_diagnosis',
            'attending_physician',
            'medical_history_summary',
            'date_of_admission',
            'date_of_birth'
        ).first()

        if not patient_data:
            return "Could not retrieve clean record parameters."

        # Convert dictionary data to a single-row Pandas DataFrame for the preprocessor
        df = pd.DataFrame([patient_data])

        # 🧼 Process Datetime fields for feature engineering exactly like training
        df['admission_dt'] = pd.to_datetime(df['date_of_admission'], errors='coerce')
        df['dob_dt'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        
        df['age_at_admission'] = df['admission_dt'].dt.year - df['dob_dt'].dt.year
        df['age_at_admission'] = df['age_at_admission'].fillna(45)
        
        df['primary_diagnosis'] = df['primary_diagnosis'].fillna("General Observation").astype(str)
        df['attending_physician'] = df['attending_physician'].fillna("Medical Staff").astype(str)
        df['medical_history_summary'] = df['medical_history_summary'].fillna("No summary provided").astype(str)

        # 🗂️ Isolate model input metrics
        X_input = df[['primary_diagnosis', 'attending_physician', 'medical_history_summary', 'age_at_admission']]

        # 🏗️ Run through the data preprocessor scaler/vectorizer pipeline tracks
        X_processed = preprocessor.transform(X_input)

        # 🔮 Generate final prediction array value
        predicted_days = model.predict(X_processed)[0]
        
        # Return a rounded string value for clean reading
        return round(float(predicted_days), 1)

    except Exception as e:
        return f"Prediction runtime error: {str(e)}"