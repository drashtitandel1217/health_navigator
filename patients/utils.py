import os
import joblib
import pandas as pd
import numpy as np
import spacy
import re
from .models import ExcelPatientRecord

# 🧠 Load the lightweight English NLP pipeline
nlp = spacy.load("en_core_web_sm")


def parse_patient_from_query(user_text):
    """Uses NLP named entity recognition and regex to extract a patient name

    or MRN code from raw user chat text.
    """
    doc = nlp(user_text)

    # 🔍 1. Try Regex tracking line for MRN patterns (e.g., L999, A123)
    mrn_match = re.search(r"\b[a-zA-Z]\d{3}\b", user_text)
    if mrn_match:
        mrn = mrn_match.group(0).upper()
        return (
            ExcelPatientRecord.objects.filter(medical_record_number=mrn)
            .first()
        )

    # 🧑‍⚕️ 2. Use spaCy NER tracking line to extract proper names (PERSON entities)
    extracted_names = [
        ent.text for ent in doc.ents if ent.label_ == "PERSON"
    ]

    if extracted_names:
        for name in extracted_names:
            patient = ExcelPatientRecord.objects.filter(
                patient_name__icontains=name
            ).first()
            if patient:
                return patient

    # 🕵️‍♂️ 3. Fallback: Scan raw text words directly for database matches
    words = user_text.split()
    for word in words:
        if len(word) > 2:
            patient = ExcelPatientRecord.objects.filter(
                patient_name__icontains=word
            ).first()
            if patient:
                return patient

    return None


def predict_patient_stay(patient_record):
    """Takes a single ExcelPatientRecord object and returns a predicted length of stay."""
    # 📁 Locate the saved model files from the root folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "length_of_stay_model.pkl")
    preprocessor_path = os.path.join(base_dir, "data_preprocessor.pkl")

    # Safety check: If model isn't trained yet, return a default fallback
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return "Model files missing"

    # 🧠 Load the AI engine elements
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # 📊 Convert the single Django model instance into a 1-row Pandas DataFrame
    patient_data = {
        "primary_diagnosis": [patient_record.primary_diagnosis],
        "attending_physician": [patient_record.attending_physician],
        "medical_history_summary": [
            str(patient_record.medical_history_summary or "")
        ],
        "age_at_admission": [
            patient_record.date_of_admission.year
            - patient_record.date_of_birth.year
            if patient_record.date_of_admission
            and patient_record.date_of_birth
            else 45
        ],
    }
    df = pd.DataFrame(patient_data)

    # ⚙️ Pass the row through the NLP transformer tracking line
    processed_features = preprocessor.transform(df)

    # 🔮 Generate the live numerical prediction assertion
    prediction = model.predict(processed_features)[0]

    return round(prediction, 1)