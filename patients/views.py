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
from django.db.models import Q

# Create your views here.
from .models import ExcelPatientRecord,ChatbotInquiryLog

try:
    nlp = spacy.load("en_core_web_md")
except:
    nlp = None

def patient_dashboard_view(request):
    if request.method == 'POST':
        return chatbot_view(request)

    # 🧠 Fetch all records directly from your database table
    all_patients = ExcelPatientRecord.objects.all()
    total_records = all_patients.count()
    
    if total_records > 0:
        bladder_count = all_patients.filter(primary_diagnosis__icontains='bladder').count()
        colorectal_count = all_patients.filter(primary_diagnosis__icontains='colorectal').count()
        cervical_count = all_patients.filter(primary_diagnosis__icontains='cervical').count()
        
        bladder_pct = round((bladder_count / total_records) * 100)
        colorectal_pct = round((colorectal_count / total_records) * 100)
        cervical_pct = round((cervical_count / total_records) * 100)
    else:
        bladder_pct = 0
        colorectal_pct = 0
        cervical_pct = 0
    
    # Send that data package into your HTML template layout
    context = {
        'patients': all_patients,
        'total_records': total_records,
        'total_records_formatted': f"{total_records:,}",
        'bladder_pct': bladder_pct,
        'colorectal_pct': colorectal_pct,
        'cervical_pct': cervical_pct,
    }
    return render(request, 'patients/dashboard.html', context)


def upload_excel_view(request):
    if request.method == "POST" and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        
        if not excel_file.name.endswith(('.xlsx', '.xls')):
            messages.error(request, "Invalid file format. Please upload a valid .xlsx or .xls Excel sheet.")
            return render(request, 'patients/upload.html')
        
        try:
            df = pd.read_excel(excel_file)
            
            # Clean column headers: Strip whitespace and make lowercase to prevent mismatches
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            
            records_created = 0
            
            for index, row in df.iterrows():
                # Rebuilt fault-tolerant MRN lookup to process rows even if headers shift
                mrn_source = (
                    row.get('medical_record_number') or 
                    row.get('mrn') or 
                    row.get('patient_id') or 
                    row.get('id')
                )
                
                mrn = str(mrn_source).strip().upper() if pd.notnull(mrn_source) else ''
                name = str(row.get('patient_name', row.get('name', 'Unknown'))).strip()
                
                # Dynamic Safe Identification Generator fallback
                if not mrn or mrn in ['NAN', '', 'NONE']:
                    if name and name != 'Unknown':
                        mrn = f"{name.replace(' ', '_').upper()}_{index}"
                    else:
                        continue  # Discard if the row contains zero patient identifiers
                
                # Dates tracking safely processed
                dob = pd.to_datetime(row.get('date_of_birth'), errors='coerce')
                adm = pd.to_datetime(row.get('date_of_admission'), errors='coerce')
                dis = pd.to_datetime(row.get('date_of_discharge'), errors='coerce')
                
                diagnosis = str(row.get('primary_diagnosis', 'General Observation')).strip()
                physician = str(row.get('attending_physician', 'Medical Staff')).strip()
                summary = str(row.get('medical_history_summary', 'No summary provided')).strip()

                if not physician or physician.lower() in ['nan', 'none', 'unknown']:
                    physician = "Unassigned / General Triage"

                # Update or save new rows safely into SQLite
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
            
            # Retrain the Random Forest model on the expanded dataset
            try:
                from train_model import train_predictive_ai
                from .utils import load_model_binaries
                train_predictive_ai()
                load_model_binaries()
                messages.success(request, f"🚀 Success! Successfully parsed and updated {records_created} patient metrics directly into SQLite database. The Random Forest model has been automatically retrained on the updated dataset.")
            except Exception as train_err:
                messages.success(request, f"🚀 Success! Successfully parsed and updated {records_created} patient metrics directly into SQLite database. (Note: Automatic retraining failed: {str(train_err)})")
            
            return redirect('patient_dashboard')

        except Exception as e:
            messages.error(request, f"❌ Excel Parser Parsing Exception Error: {str(e)}")
            
    return render(request, 'patients/upload.html')

# 📋 1. ROBUST MEDICAL STOP WORDS LIST
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "complain", "complaining", 
    "for", "from", "has", "have", "having", "how", "i", "in", "is", "it", "its", "me", 
    "my", "of", "on", "or", "patient", "that", "the", "this", "to", "was", "with", "you", "your"
}

# 📋 2. CLINICAL SYNONYM DICTIONARY
SYNONYM_MAP = {
    "coughing blood": ["hemoptysis", "blood", "cough"],
    "coughing up blood": ["hemoptysis", "blood", "cough"],
    "shortness of breath": ["dyspnea", "shortness", "breath", "breathing"],
    "chest pain": ["thoracic pain", "chest", "pain"],
    "weight loss": ["unexplained weight loss", "weight", "loss"],
    "night sweats": ["sweats", "drenching"],
    "hoarseness": ["hoarse", "voice"],
    "blood in stool": ["melena", "stool", "blood"],
    "blood in urine": ["hematuria", "urine", "blood"]
}

def chatbot_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            raw_message = data.get('message', '').strip()
            
            if not raw_message:
                return JsonResponse({'error': 'Empty message received'}, status=400)
            
            # Extract text if raw JSON strings are pasted into client fields
            if raw_message.startswith('{') and raw_message.endswith('}'):
                try:
                    inner_data = json.loads(raw_message)
                    raw_message = inner_data.get('message', raw_message)
                except json.JSONDecodeError:
                    pass

            lower_message = raw_message.lower()
            raw_tokens = re.findall(r'\b\w+\b', lower_message)
            is_simple_query = len(raw_tokens) < 10
            
            # ==========================================================
            # 👋 LAYER 0: GREETING INTENT
            # ==========================================================
            greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "greetings"]
            if any(re.search(rf'\b{word}\b', lower_message) for word in greetings):
                reply_text = "Hello! Welcome to the Medical Assistant Chatbot.\n\n"
                ChatbotInquiryLog.objects.create(
                    raw_input_text=raw_message,
                    processed_input_text="greeting",
                    ai_response_reply=reply_text
                )
                return JsonResponse({
                    'reply': reply_text
                })

            # ==========================================================
            # 📋 LAYER 0.5: PATIENT METADATA & LENGTH OF STAY FORECAST
            # ==========================================================
            has_forecast_keywords = any(w in lower_message for w in ["stay", "forecast", "timeline", "discharge", "days", "predict", "hospital", "duration", "length"])
            has_mrn = bool(re.search(r"\b[a-zA-Z]\d{3}\b", raw_message))
            
            # Use spaCy to detect PERSON entities if loaded
            has_person = False
            if nlp:
                doc = nlp(raw_message)
                has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
                
            if has_forecast_keywords or has_mrn or has_person or "patient" in lower_message:
                patient = parse_patient_from_query(raw_message)
                if patient:
                    predicted_stay = predict_patient_stay(patient)
                    if isinstance(predicted_stay, (int, float)):
                        reply_text = (
                            f"📋 **Patient Record Found**:\n"
                            f"• **Patient Name**: {patient.patient_name}\n"
                            f"• **MRN**: `{patient.medical_record_number}`\n"
                            f"• **Primary Diagnosis**: {patient.primary_diagnosis}\n"
                            f"• **Attending Physician**: {patient.attending_physician}\n\n"
                            f"🔮 **Length of Stay Forecast**: `{predicted_stay}` Days\n\n"
                            f"⚠️ *Disclaimer: Prediction generated via Random Forest Regressor Module.*"
                        )
                    else:
                        reply_text = (
                            f"📋 **Patient Record Found**:\n"
                            f"• **Patient Name**: {patient.patient_name}\n"
                            f"• **MRN**: `{patient.medical_record_number}`\n"
                            f"• **Primary Diagnosis**: {patient.primary_diagnosis}\n"
                            f"• **Attending Physician**: {patient.attending_physician}\n\n"
                            f"⚠️ **Stay Forecast Error**: {predicted_stay}"
                        )
                    
                    ChatbotInquiryLog.objects.create(
                        raw_input_text=raw_message,
                        processed_input_text=patient.patient_name,
                        ai_response_reply=reply_text
                    )
                    return JsonResponse({'reply': reply_text})

            # ==========================================================
            # 🎯 LAYER 1: STATIC INTENTS FOR COMMON MINOR ILLNESSES
            # ==========================================================
            if is_simple_query and any(re.search(rf'\b{w}\b', lower_message) for w in ["cough", "fever", "flu", "cold", "sore throat", "chills"]):
                predicted_diagnosis = "Viral Fever / Influenza"
                engine_used = "Static Intent Engine"
                
            elif is_simple_query and any(re.search(rf'\b{w}\b', lower_message) for w in ["headache", "migraine", "temple pain"]):
                predicted_diagnosis = "Migraine / Acute Headache"
                engine_used = "Static Intent Engine"
                
            elif is_simple_query and any(re.search(rf'\b{w}\b', lower_message) for w in ["heartburn", "acid reflux", "acidity", "indigestion"]):
                predicted_diagnosis = "Gastroesophageal Reflux Disease (GERD)"
                engine_used = "Static Intent Engine"
                
            elif is_simple_query and any(re.search(rf'\b{w}\b', lower_message) for w in ["sneezing", "runny nose", "watery eyes", "allergy"]):
                predicted_diagnosis = "Allergic Rhinitis"
                engine_used = "Static Intent Engine"

            # ==========================================================
            # 🗄️ LAYER 2: ADVANCED CLINICAL SEARCH (Runs if Layer 1 fails)
            # ==========================================================
            else:
                engine_used = "Synonym-Expanded Database Search"
                
                # --- FIX 1: Ignore single letters like 't' or 's' from contractions ---
                base_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
                
                search_terms = list(base_tokens)
                for user_phrase, clinical_terms in SYNONYM_MAP.items():
                    if user_phrase in lower_message:
                        search_terms.extend(clinical_terms)
                
                search_terms = list(set(search_terms))

                # Dynamic matching flags
                has_lung_signals = any(k in search_terms for k in ["cough", "blood", "hemoptysis", "chest", "breath", "hoarseness", "lung", "nsclc"])
                has_urinary_signals = any(k in search_terms for k in ["urine", "hematuria", "urination", "bladder"])
                has_bowel_signals = any(k in search_terms for k in ["stool", "poop", "melena", "colorectal", "abdominal"])

                matched_record = None

                # --- FIX 2: Explicit Isolated Query Targets ---
                if has_lung_signals:
                    # Enforce that it MUST contain lung markers and CANNOT contain bladder/cervical markers
                    matched_record = ExcelPatientRecord.objects.filter(
                        (Q(primary_diagnosis__icontains="lung") | Q(primary_diagnosis__icontains="nsclc")),
                        ~Q(primary_diagnosis__icontains="bladder"),
                        ~Q(primary_diagnosis__icontains="cervical")
                    ).first()
                    
                elif has_urinary_signals:
                    matched_record = ExcelPatientRecord.objects.filter(primary_diagnosis__icontains="bladder").first()
                    
                elif has_bowel_signals:
                    matched_record = ExcelPatientRecord.objects.filter(primary_diagnosis__icontains="colorectal").first()

                # --- FIX 3: TARGETED KEYWORD FALLBACK MAPPING ---
                # If target systems yield nothing, search key descriptive terms directly across summaries
                if not matched_record:
                    fallback_filter = Q()
                    # Isolate descriptive tokens that carry precise diagnostic weight
                    critical_landmarks = [t for t in search_terms if t in ["hoarseness", "shortness", "breath", "neck", "collarbone", "chest", "sweats"]]
                    
                    if critical_landmarks:
                        for term in critical_landmarks:
                            fallback_filter |= Q(medical_history_summary__icontains=term)
                            fallback_filter |= Q(primary_diagnosis__icontains=term)
                        
                        # Apply context limits if lung signals were originally caught
                        if has_lung_signals:
                            fallback_filter &= (Q(primary_diagnosis__icontains="lung") | Q(primary_diagnosis__icontains="nsclc"))
                        
                        matched_record = ExcelPatientRecord.objects.filter(fallback_filter).first()

                # Final string processing assignment
                if matched_record:
                    predicted_diagnosis = matched_record.primary_diagnosis
                else:
                    predicted_diagnosis = "General Triage Consultation Required"
                    engine_used = "System Default Fallback"

            # ==========================================================
            # 🎨 LAYER 3: LAYOUT WRAPPER & SEVERE LABEL DISCLAIMER
            # ==========================================================
            severe_labels = ["cancer", "tumor", "carcinoma", "stage iii", "metastasis", "nsclc"]
            is_severe = any(word in predicted_diagnosis.lower() for word in severe_labels)
            
            # Build clean display text token arrays
            display_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
            processed_message = ", ".join(display_tokens) if display_tokens else "None"

            if is_severe and engine_used != "Static Intent Engine":
                reply_text = (
                    f"🩺 Advanced Assessment: `{predicted_diagnosis}`\n\n"
                    f"Next Steps: Please consult an attending specialist immediately for expert verification.\n\n"
                )
            else:
                reply_text = (
                    f"🎯 Suggested Track:`{predicted_diagnosis}`\n\n"
                    f"⚠️ Disclaimer: Generated via {engine_used}. Consult a physician.\n\n"
                    f"• Symptoms: `{processed_message}`"
                )

            # Log the inquiry
            ChatbotInquiryLog.objects.create(
                raw_input_text=raw_message,
                processed_input_text=processed_message,
                ai_response_reply=reply_text
            )

            return JsonResponse({'reply': reply_text})
            
        except Exception as e:
            return JsonResponse({'error': f"NLP Exception: {str(e)}"}, status=500)
            
    return JsonResponse({'error': 'Invalid request method.'}, status=400)