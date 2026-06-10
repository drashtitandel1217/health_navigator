from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from .utils import parse_patient_from_query, predict_patient_stay
from django.contrib import messages
import pandas as pd
import json  
import re
import spacy
import os
from django.conf import settings
import joblib

# Create your views here.
from .models import ExcelPatientRecord,ChatbotInquiryLog

try:
    nlp = spacy.load("en_core_web_md")
except:
    nlp = None

def patient_dashboard_view(request):
    # 🧠 Fetch all 2,001 records directly from your database table
    all_patients = ExcelPatientRecord.objects.all()
    
    # Send that data package into your HTML template layout
    context = {
        'patients': all_patients
    }
    return render(request, 'patients/dashboard.html', context)


def upload_excel_view(request):
    if request.method == "POST" and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        
        # 🛡️ Safety check: Ensure it's actually an Excel spreadsheet extension
        if not excel_file.name.endswith(('.xlsx', '.xls')):
            messages.error(request, "Invalid file format. Please upload a valid .xlsx or .xls Excel sheet.")
            return render(request, 'patients/upload.html')
        
        try:
            # Load spreadsheet array elements into Pandas
            df = pd.read_excel(excel_file)
            
            # 🧼 Clean column headers: Strip whitespace and make lowercase to prevent mismatches
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            
            records_created = 0
            
            # Loop through rows and populate your Django schema cleanly
            for _, row in df.iterrows():
                # Extract parameters with safe fallback lookups using .get()
                mrn = str(row.get('medical_record_number', row.get('mrn', ''))).strip().upper()
                name = str(row.get('patient_name', row.get('name', 'Unknown'))).strip()
                
                if not mrn or mrn == 'NAN':
                    continue  # Skip rows without a clear MRN index identity tracking code

                # Convert dates with safe fallback strings to prevent null datetime matrix breakups
                dob = pd.to_datetime(row.get('date_of_birth'), errors='coerce')
                adm = pd.to_datetime(row.get('date_of_admission'), errors='coerce')
                dis = pd.to_datetime(row.get('date_of_discharge'), errors='coerce')
                
                # Formulate structural variables
                diagnosis = str(row.get('primary_diagnosis', 'General Observation')).strip()
                physician = str(row.get('attending_physician', 'Medical Staff')).strip()
                summary = str(row.get('medical_history_summary', 'No summary provided')).strip()

                # 🚀 Save or overwrite row structures into SQLite safely
                ExcelPatientRecord.objects.update_or_create(
                    medical_record_number=mrn,
                    defaults={
                        'patient_name': name,
                        'date_of_birth': dob if pd.notnull(dob) else None,
                        'date_of_admission': adm if pd.notnull(adm) else None,
                        'date_of_discharge': dis if pd.notnull(dis) else None,
                        'primary_diagnosis': diagnosis,
                        'attending_physician': physician,
                        'medical_history_summary': summary
                    }
                )
                records_created += 1
            
            messages.success(request, f"🚀 Success! Successfully parsed and updated {records_created} patient metrics directly into SQLite database.")
            return redirect('patient_dashboard') # Bounces them right back into your beautiful full-screen console screen!

        except Exception as e:
            messages.error(request, f"❌ Excel Parser Parsing Exception Error: {str(e)}")
            
    return render(request, 'patients/upload.html')

import joblib

CLASSIFIER_PATH = os.path.join(
    settings.BASE_DIR,
    "symptom_classifier_model.pkl"
)

print(f"📍 Looking for model at: {CLASSIFIER_PATH}")
print(f"📍 Exists: {os.path.exists(CLASSIFIER_PATH)}")

try:
    symptom_model = joblib.load(CLASSIFIER_PATH)
    print("✅ Symptom classifier loaded successfully")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    symptom_model = None 

# Custom stop words to display transparently in your interface logs
ENGLISH_STOP_WORDS = {'what', 'is', 'the', 'expected', 'for', 'a', 'an', 'and', 'are', 'i', 'have', 'been', 'feeling'}

def chatbot_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            raw_message = data.get('message', '').strip()
            
            if not raw_message:
                return JsonResponse({'error': 'Empty message received'}, status=400)
            
            # 🧼 Manual cleanup step for interface visualization logs
            lower_message = raw_message.lower()
            clean_tokens = re.findall(r'\b\w+\b', lower_message)
            filtered_tokens = [token for token in clean_tokens if token not in ENGLISH_STOP_WORDS]
            processed_message = " ".join(filtered_tokens)

            # 🔮 Run live prediction through your trained model pipeline file
            if symptom_model:
                # The pipeline automatically handles lowercasing and stop words inside its TF-IDF step!
                predicted_diagnosis = symptom_model.predict([raw_message])[0]
                
                reply_text = f"🎯 **Predicted Condition/Department:** `{predicted_diagnosis}`\n\n" \
                             f"• *Extracted Keyword Matrix:* `{processed_message if processed_message else 'Conversational text'}`\n" \
                             f"⚠️ *Disclaimer: Generated via Random Forest Multi-Class Text Classification. Consult a physician for verified advice.*"
            else:
                reply_text = "🤖 **System Alert:** The trained classification matrix binary (`symptom_classifier_model.pkl`) is missing. Please run `python train_classifier.py` in your terminal workspace first."

            # 💾 Store metrics inside database logs
            try:
                ChatbotInquiryLog.objects.create(
                    raw_input_text=raw_message,
                    processed_input_text=processed_message,
                    ai_response_reply=reply_text
                )
            except Exception as log_error:
                print(f"⚠️ Non-blocking Log Error: {log_error}")

            return JsonResponse({'reply': reply_text})
            
        except Exception as e:
            return JsonResponse({'error': f"Internal Server NLP Classifier Exception: {str(e)}"}, status=500)
            
    return JsonResponse({'error': 'Invalid request method.'}, status=400)