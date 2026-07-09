from sklearn.utils import _metadata_requests
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
from .models import UserSelfCheckMetric

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
        'total_records_formatted': f"{total_records:,}",
        'bladder_pct': bladder_pct,
        'colorectal_pct': colorectal_pct,
        'cervical_pct': cervical_pct,
    }
    return render(request, 'patients/dashboard.html', context)

from .utils import load_model_binaries 

def upload_excel_view(request):
    # Initialize the tracking counter at the absolute top of the scope
    records_created = 0
    
    if request.method == "POST" and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        
        if not excel_file.name.endswith(('.xlsx', '.xls')):
            messages.error(request, "Invalid file format. Please upload a valid .xlsx or .xls Excel sheet.")
            return render(request, 'patients/upload.html')
        
        try:
            df = pd.read_excel(excel_file)
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            
            for index, row in df.iterrows():
                mrn_source = (
                    row.get('medical_record_number') or 
                    row.get('mrn') or 
                    row.get('patient_id') or 
                    row.get('id')
                )
                
                mrn = str(mrn_source).strip().upper() if pd.notnull(mrn_source) else ''
                name = str(row.get('patient_name', row.get('name', 'Unknown'))).strip()
                
                if not mrn or mrn in ['NAN', '', 'NONE']:
                    if name and name != 'Unknown':
                        mrn = f"{name.replace(' ', '_').upper()}_{index}"
                    else:
                        continue
                
                dob = pd.to_datetime(row.get('date_of_birth'), errors='coerce')
                adm = pd.to_datetime(row.get('date_of_admission'), errors='coerce')
                dis = pd.to_datetime(row.get('date_of_discharge'), errors='coerce')
                
                diagnosis = str(row.get('primary_diagnosis', 'General Observation')).strip()
                physician = str(row.get('attending_physician', 'Medical Staff')).strip()
                summary = str(row.get('medical_history_summary', 'No summary provided')).strip()

                if not physician or physician.lower() in ['nan', 'none', 'unknown']:
                    physician = "Unassigned / General Triage"

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
            
            # ==========================================================
            # 🔥 STEP 2: RETRAIN BOTH AI PIPELINES & HOT-RELOAD
            # ==========================================================
            try:
                # 1. Run the length of stay training
                from train_model import train_predictive_ai
                print("🏋️‍♂️ Retraining Length of Stay Regressor...")
                train_predictive_ai()
                
                # 2. 🎯 FORCED RELOAD BYPASS: Clear Python module caching barriers
                print("🏋️‍♂️ Purging cache and reloading Random Forest Symptom Classifier...")
                import importlib
                import train_classifier
                
                # Force Python to reload the module freshly from disk
                importlib.reload(train_classifier)
                
                # Execute the native data extraction and pipeline fitting
                train_classifier.run_symptom_retraining_pipeline()
                
                # 3. Hot-reload the fresh model weights straight into server RAM cache
                load_model_binaries()
                
                messages.success(request, f"🚀 Success! Parsed {records_created} records. Both AI pipelines have been automatically retrained and hot-reloaded!")
            
            except Exception as train_err:
                print(f"❌ Retraining Error details: {str(train_err)}")
                messages.success(request, f"🚀 Success! Parsed {records_created} records. (Note: Multi-model retraining failed: {str(train_err)})")
                return redirect('patient_dashboard')

        except Exception as e:
            messages.error(request, f"❌ Excel Parser Parsing Exception Error: {str(e)}")
            
    return render(request, 'patients/upload.html')

# 📋 1. ROBUST MEDICAL STOP WORDS LIST
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
                return JsonResponse({'reply': reply_text})

            # ==========================================================
            # 📋 LAYER 0.5: PATIENT METADATA & LENGTH OF STAY FORECAST
            # ==========================================================
            has_forecast_keywords = any(w in lower_message for w in ["stay", "forecast", "timeline", "discharge", "days", "predict", "hospital", "duration", "length"])
            has_mrn = bool(re.search(r"\b[a-zA-Z]\d{3}\b", raw_message))
            
            has_person = False
            if 'nlp' in globals() and nlp:
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
            # 🤖 LAYER 2: MACHINE LEARNING INTENT CLASSIFICATION ENGINE
            # ==========================================================
            else:
                try:
                    intent_classifier_pipeline = load_model_binaries()
                    
                    if intent_classifier_pipeline is not None:
                        display_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
                        processed_text = " ".join(display_tokens)
                        
                        if processed_text.strip():
                            prediction = intent_classifier_pipeline.predict([processed_text])[0]
                            predicted_diagnosis = str(prediction)
                            engine_used = "Random Forest Intent Classification Engine"
                        else:
                            predicted_diagnosis = "General Triage Consultation Required"
                            engine_used = "System Default Fallback (Insufficient Input)"
                    else:
                        predicted_diagnosis = "General Triage Consultation Required"
                        engine_used = "System Default Fallback (Model Binary Missing)"
                        
                except Exception as ml_err:
                    print(f"🔥 MACHINE LEARNING ROUTE ERROR: {str(ml_err)}")
                    predicted_diagnosis = "General Triage Consultation Required"
                    engine_used = f"System Default Fallback (ML Processing Error)"

            # ==========================================================
            # 🎨 LAYER 3: LAYOUT WRAPPER & SEVERE LABEL DISCLAIMER
            # ==========================================================
            if "Fallback" in engine_used or predicted_diagnosis == "General Triage Consultation Required":
                reply_text = (
                    "Hello! Thank you for reaching out to the Care Portal.\n\n"
                    "I reviewed the symptoms you described, but I want to be completely thorough and precise."
                    "To give you the most accurate triage recommendation, could you tell me a little more? "
                    "For instance, how long have you felt this way, or are there any other signs you are experiencing?\n\n"
                )
            
            # 2. 🟢 SUCCESS: SYSTEM FOUND A MATCHING METRIC TRACK
            else:
                # Set up structured care content blocks based on the department matched
                track_lower = predicted_diagnosis.lower()
                
                if "orthopedic" in track_lower or "fracture" in track_lower or "tibia" in track_lower:
                    diagnosis_title = "Orthopedic Trauma Triage Protocol"
                    cure_plan = "Immediate physical stabilization. Requires radiological diagnostic imaging (X-Ray/CT Scan) to assess structural displacement, followed by casting or orthopedic surgical intervention."
                    precautions = "Immobilize the affected limb entirely. Avoid placing any weight on the leg. Elevate the extremity above heart level to control swelling, and apply ice packs wrapped in cloth."
                    
                elif "gastro" in track_lower or "gerd" in track_lower or "stomach" in track_lower:
                    diagnosis_title = "Gastroenterology Triage Track"
                    cure_plan = "Clinical evaluation for acid suppression therapy (such as H2 receptor antagonists or Proton Pump Inhibitors like Omeprazole). Dietary mapping to identify gastrointestinal trigger thresholds."
                    precautions = "Avoid lying down for at least 3 hours immediately following a meal. Elevate the head of your bed by 6 inches. Avoid spicy, highly acidic, fatty foods, caffeine, or carbonated beverages."
                    
                elif "neuro" in track_lower or "migraine" in track_lower or "headache" in track_lower:
                    diagnosis_title = "Neurology Consultation Track"
                    cure_plan = "Therapeutic acute relief intervention (such as triptans or targeted NSAIDs). For chronic patterns, prophylactic maintenance therapy may be evaluated by a neurologist."
                    precautions = "Rest in a quiet, completely darkened room at the onset of symptoms. Keep a detailed log of potential lifestyle triggers (e.g., specific foods, shifting sleep patterns, blue-light exposure)."
                    
                else:
                    # Dynamic Default for general database categories (including Oncology rows)
                    diagnosis_title = f"{predicted_diagnosis} Clinical Review"
                    cure_plan = "Requires an advanced clinical workup, formal pathology analysis, and blood panel tracking directed by an attending department specialist."
                    precautions = "Closely document the progression, intensity, and timing of your symptoms. Bring any historical laboratory reports or current medication schedules to your next formal consultation."

                # Compile the clean, elegant markdown message structure
                reply_text = (
                    f"Symptoms of:\n`{diagnosis_title}`\n\n"
                    f"Cure recommended:\n"
                    f"{cure_plan}\n\n"
                    f"Precautions:\n"
                    f"{precautions}\n\n"
                
                )

            display_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
            processed_message = ", ".join(display_tokens) if display_tokens else "None"

            # Record metrics inside historical SQLite database logs safely
            ChatbotInquiryLog.objects.create(
                raw_input_text=raw_message,
                processed_input_text=processed_message, # 👈 Now completely defined!
                ai_response_reply=reply_text
            )

            return JsonResponse({'reply': reply_text})
            
        except Exception as e:
            return JsonResponse({'error': f"NLP Exception Handling Request: {str(e)}"}, status=500)
            
    return JsonResponse({'error': 'Invalid request method.'}, status=400)


def self_check_diet_engine_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))
        age = int(data.get('age', 0))
        gender = data.get('gender', 'M')
        goal = data.get('goal', 'MAINTAIN')
        
        # 1. Math computation equations
        height_m = height / 100.0
        bmi = round(weight / (height_m ** 2), 1)
        
        if bmi < 18.5:
            bmi_tier = "Underweight"
        elif bmi < 25.0:
            bmi_tier = "Normal Weight"
        elif bmi < 30.0:
            bmi_tier = "Overweight"
        else:
            bmi_tier = "Obese"
            
        # 2. Compute BMR
        if gender == 'M':
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            
        # TDEE basic multiplier assuming moderate activity
        tdee = bmr * 1.375
        
        # 3. Target Calorie adjustment based on target goals
        if goal == "LOSE":
            target_calories = int(tdee - 500)
            macro_split = "🥩 High Protein / Moderate Carb split (40% P / 30% C / 30% F)"
            meals = "• Breakfast: Egg whites scramble with spinach\n• Lunch: Grilled chicken breast with broccoli\n• Dinner: Baked salmon with asparagus"
        elif goal == "GAIN":
            target_calories = int(tdee + 400)
            macro_split = "🥑 High Calorie / Clean Bulking split (30% P / 40% C / 30% F)"
            meals = "• Breakfast: Oatmeal with peanut butter and banana\n• Lunch: Lean beef with brown rice\n• Dinner: Roasted turkey breast with sweet potatoes"
        else:
            target_calories = int(tdee)
            macro_split = "🥗 Balanced Nutrition Maintenance profile (30% P / 45% C / 25% F)"
            meals = "• Breakfast: Greek yogurt with mixed berries\n• Lunch: Quinoa salad with mixed greens and tofu\n• Dinner: Lean white fish with mixed vegetables"

        diet_plan = f"📊 **Macronutrient Profile**: {macro_split}\n\n🍏 **Suggested Meal Structure**:\n{meals}"

        # 4. Save into SQLite history trace logs
        UserSelfCheckMetric.objects.create(
            age=age, gender=gender, height_cm=height, weight_kg=weight, fitness_goal=goal,
            calculated_bmi=bmi, bmi_category=bmi_tier, recommended_calories=target_calories, diet_plan_markdown=diet_plan
        )

        return JsonResponse({
            'bmi': bmi,
            'category': bmi_tier,
            'calories': target_calories,
            'diet_plan': diet_plan
        })
        
    return JsonResponse({'error': 'Direct interface browser GET requests not supported on this data endpoint lines.'}, status=400)

