from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from .utils import parse_patient_from_query, predict_patient_stay, load_model_binaries
from django.contrib import messages
import pandas as pd
import json
import re
import random
import spacy
import os
from datetime import datetime
from django.conf import settings
from django.db.models import Q
from .models import UserSelfCheckMetric, ExcelPatientRecord, ChatbotInquiryLog

try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    nlp = None


# ==========================================================
# 📋 MODULE-LEVEL CONSTANTS (built once at import, not per-request)
# ==========================================================

STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "complain", "complaining",
    "for", "from", "has", "have", "having", "how", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "patient", "that", "the", "this", "to", "was", "with", "you", "your"
}

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

# --- Layer 0: greeting / small-talk word banks ---
GREETING_WORDS   = ["hello", "hi", "hey", "greetings", "yo", "sup"]
TIME_GREETING    = ["good morning", "good afternoon", "good evening"]
GRATITUDE_WORDS  = ["thanks", "thank you", "appreciate it", "ty"]
FAREWELL_WORDS   = ["bye", "goodbye", "see you", "later", "exit", "quit"]
HOW_ARE_YOU      = ["how are you", "how's it going", "how you doing", "you good"]
AFFIRM_WORDS     = ["ok", "okay", "cool", "great", "nice", "sounds good"]

GREETING_REPLIES = [
    "Hey there! What's going on?",
    "Hi! How can I help today?",
    "Hey! What brings you in?",
    "Hello! What can I do for you?",
]
GRATITUDE_REPLIES = [
    "Anytime!",
    "You're welcome!",
    "Happy to help.",
    "No problem at all.",
]
FAREWELL_REPLIES = [
    "Take care!",
    "Bye for now — feel better soon.",
    "See you later!",
    "Goodbye, take it easy.",
]
HOW_ARE_YOU_REPLIES = [
    "Doing well, thanks for asking! How about you?",
    "All good here. What's up with you?",
    "I'm running smooth. How are you feeling today?",
]
AFFIRM_REPLIES = [
    "Got it.",
    "Sounds good.",
    "Alright, let me know if you need anything else.",
]

# --- Layer 0.5: patient lookup follow-up word banks ---
FOLLOWUP_DIAGNOSIS_WORDS = ["diagnosis", "condition", "what's wrong", "why is"]
FOLLOWUP_PHYSICIAN_WORDS = ["doctor", "physician", "who's treating", "assigned to"]
FOLLOWUP_STATUS_WORDS    = ["status", "update", "how is", "progress"]
CORRECTION_WORDS         = ["wrong patient", "not them", "different patient", "try again"]

# --- Layer 1: static minor-illness intents ---
RED_FLAG_WORDS = [
    "chest pain", "can't breathe", "cant breathe", "difficulty breathing",
    "severe bleeding", "unconscious", "seizure", "suicidal", "overdose",
    "stroke", "numb face", "slurred speech", "severe allergic reaction"
]

STATIC_INTENTS = {
    "Viral Fever / Influenza": ["cough", "fever", "flu", "cold", "sore throat", "chills"],
    "Migraine / Acute Headache": ["headache", "migraine", "temple pain"],
    "Gastroesophageal Reflux Disease (GERD)": ["heartburn", "acid reflux", "acidity", "indigestion"],
    "Allergic Rhinitis": ["sneezing", "runny nose", "watery eyes", "allergy"],
    "Tension Muscle Strain": ["back pain", "neck pain", "stiff neck", "muscle strain", "sore muscles"],
    "Gastroenteritis": ["diarrhea", "stomach bug", "nausea", "vomiting", "upset stomach"],
    "Insomnia / Sleep Disruption": ["can't sleep", "cant sleep", "insomnia", "trouble sleeping"],
    "Dehydration": ["dizzy", "lightheaded", "dry mouth", "dehydrated"],
    "Contact Dermatitis": ["skin rash", "itchy skin", "hives", "skin irritation"],
    "Tension Eye Strain": ["eye strain", "blurry vision", "tired eyes"],
}

CONCERN_WORDS  = ["worried", "scared", "not sure", "is this serious", "should I be worried"]
DURATION_WORDS = ["days", "weeks", "since yesterday", "started", "for a while"]

# --- Layer 3: care tracks ---
CARE_TRACKS = {
    "orthopedic": {
        "match": ["orthopedic", "fracture", "tibia", "bone", "joint"],
        "title": "Orthopedic Trauma Triage Protocol",
        "cure_plan": "Immediate physical stabilization is recommended, along with radiological imaging (X-Ray/CT) to assess structural involvement. A physician will determine whether casting or surgical intervention is needed.",
        "precautions": "Immobilize the affected limb, avoid weight-bearing, elevate above heart level, and apply a cloth-wrapped ice pack to control swelling.",
    },
    "gastro": {
        "match": ["gastro", "gerd", "stomach", "acid reflux", "heartburn"],
        "title": "Gastroenterology Triage Track",
        "cure_plan": "A clinician may evaluate options for acid-suppression therapy and help map dietary triggers contributing to symptoms.",
        "precautions": "Avoid lying down for ~3 hours after eating, elevate the head of your bed, and limit spicy, fatty, acidic, or carbonated intake.",
    },
    "neuro": {
        "match": ["neuro", "migraine", "headache"],
        "title": "Neurology Consultation Track",
        "cure_plan": "Acute relief options can be discussed with a physician; for recurring patterns, a neurologist may evaluate preventive maintenance care.",
        "precautions": "Rest in a quiet, dark room at symptom onset, and log potential triggers like diet, sleep changes, or screen exposure.",
    },
    "dermatology": {
        "match": ["rash", "hives", "dermatitis", "itchy skin", "skin irritation"],
        "title": "Dermatology Triage Track",
        "cure_plan": "A clinician can help identify the irritant or allergen and recommend appropriate topical care.",
        "precautions": "Avoid scratching the area, keep skin clean and dry, and avoid known irritants (new soaps, detergents, fabrics) until reviewed.",
    },
    "respiratory": {
        "match": ["respiratory", "bronchitis", "wheezing", "shortness of breath", "asthma"],
        "title": "Respiratory Triage Track",
        "cure_plan": "Evaluation of airway function may be needed, along with a review of any inhaler or maintenance therapy currently in use.",
        "precautions": "Avoid known respiratory irritants (smoke, dust, strong fragrances), monitor breathing closely, and seek urgent care if breathing worsens.",
    },
    "ent": {
        "match": ["ent", "ear pain", "sinus", "throat infection", "tonsil"],
        "title": "ENT (Ear, Nose & Throat) Triage Track",
        "cure_plan": "An ENT evaluation can determine whether the cause is infectious, structural, or allergy-related, and guide next steps.",
        "precautions": "Stay hydrated, avoid forceful nose-blowing, and monitor for fever or worsening pain that could indicate infection spread.",
    },
    "urology": {
        "match": ["urology", "bladder", "urinary", "uti"],
        "title": "Urology Triage Track",
        "cure_plan": "A urinalysis and clinical review can help determine whether infection, irritation, or another cause is involved.",
        "precautions": "Increase fluid intake, avoid holding urine for long periods, and note any pain, urgency, or color changes to report to your doctor.",
    },
}

DEFAULT_TRACK = {
    "cure_plan": "This may require an advanced clinical workup, formal pathology analysis, and blood panel tracking directed by an attending department specialist.",
    "precautions": "Closely document the progression, intensity, and timing of your symptoms. Bring any historical lab reports or current medications to your next consultation.",
}


# ==========================================================
# 🔧 MODULE-LEVEL HELPERS
# ==========================================================

def _has_word(words, text):
    """Word-boundary match: True if any word in `words` appears as a whole word/phrase in `text`."""
    return any(re.search(rf'\b{re.escape(w)}\b', text) for w in words)


def _is_pure_affirmation(text, words):
    stripped = text.strip().rstrip("!.")
    return stripped in words or (len(stripped.split()) <= 3 and _has_word(words, stripped))


def _time_of_day_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! How can I help?"
    elif hour < 18:
        return "Good afternoon! What's up?"
    else:
        return "Good evening! What can I do for you?"


def _log_and_reply(raw_message, processed_label, reply_text):
    ChatbotInquiryLog.objects.create(
        raw_input_text=raw_message,
        processed_input_text=processed_label,
        ai_response_reply=reply_text
    )
    return JsonResponse({'reply': reply_text})


def _build_patient_card(patient):
    return (
        f"📋 **Patient Record Found**:\n"
        f"• **Patient Name**: {patient.patient_name}\n"
        f"• **MRN**: `{patient.medical_record_number}`\n"
        f"• **Primary Diagnosis**: {patient.primary_diagnosis}\n"
        f"• **Attending Physician**: {patient.attending_physician}\n\n"
    )


# ==========================================================
# 🖥️ DASHBOARD VIEW
# ==========================================================

def patient_dashboard_view(request):
    if request.method == 'POST':
        return chatbot_view(request)

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

    context = {
        'total_records_formatted': f"{total_records:,}",
        'bladder_pct': bladder_pct,
        'colorectal_pct': colorectal_pct,
        'cervical_pct': cervical_pct,
    }
    return render(request, 'patients/dashboard.html', context)


# ==========================================================
# 📥 EXCEL UPLOAD / INGEST VIEW
# ==========================================================

def upload_excel_view(request):
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
            # 🔥 RETRAIN BOTH AI PIPELINES & HOT-RELOAD
            # ==========================================================
            try:
                from train_model import train_predictive_ai
                print("🏋️‍♂️ Retraining Length of Stay Regressor...")
                train_predictive_ai()

                print("🏋️‍♂️ Purging cache and reloading Random Forest Symptom Classifier...")
                import importlib
                import train_classifier

                importlib.reload(train_classifier)
                train_classifier.run_symptom_retraining_pipeline()

                load_model_binaries()

                messages.success(request, f"🚀 Success! Parsed {records_created} records. Both AI pipelines have been automatically retrained and hot-reloaded!")

            except Exception as train_err:
                print(f"❌ Retraining Error details: {str(train_err)}")
                messages.success(request, f"🚀 Success! Parsed {records_created} records. (Note: Multi-model retraining failed: {str(train_err)})")
                return redirect('patient_dashboard')

        except Exception as e:
            messages.error(request, f"❌ Excel Parser Parsing Exception Error: {str(e)}")

    return render(request, 'patients/upload.html')




# ==========================================================
# 💬 CHATBOT VIEW
# ==========================================================

def chatbot_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method.'}, status=400)

    try:
        data = json.loads(request.body)
        raw_message = data.get('message', '').strip()

        if not raw_message:
            return JsonResponse({'error': 'Empty message received'}, status=400)

        if raw_message.startswith('{') and raw_message.endswith('}'):
            try:
                inner_data = json.loads(raw_message)
                raw_message = inner_data.get('message', raw_message)
            except json.JSONDecodeError:
                pass

        lower_message = raw_message.lower()
        raw_tokens = re.findall(r'\b\w+\b', lower_message)
        is_simple_query = len(raw_tokens) < 10

        # Retrieve active tracking states
        session_state = request.session.get('intake_state', 'IDLE')
        intake_data = request.session.get('intake_data', {})

        # ==========================================================
        # 🏥 STATE 1: FULL DIAGNOSTIC ONBOARDING INTAKE ENGINE (RANDOMIZED MULTI-REPLY)
        # ==========================================================
        if any(w in lower_message for w in ["checkup", "check up", "diagnostic", "fit or not", "evaluate me", "doctor"]):
            request.session['intake_state'] = 'AWAITING_NAME'
            request.session['intake_data'] = {}
            
            init_replies = [
                "👨‍⚕️ **[Clinical Intake Mode Engaged]**\n\nHello, I am your digital attending specialist. Let's run a complete fitness and diagnostic evaluation. To begin our medical record profile, **what is your full name?**",
                "👨‍⚕️ **[Triage Intake Loop Initialized]**\n\nWelcome to the evaluation module. Let's gather your clinical baseline metrics. Please state your **full name** for the session registry.",
                "👨‍⚕️ **[Attending Engine Active]**\n\nGreetings. I will guide you through a physical health review session today. To pull up the proper data matrix, **what name should I log under?**"
            ]
            return _log_and_reply(raw_message, "diagnostic_intake_start", random.choice(init_replies))

        # ==========================================================
        # 🏥 LAYER 0: STATEFUL CLINICAL DIALOGUE INTAKE ENGINE
        # ==========================================================
        
        # Trigger full diagnostic onboarding session checkup
        if any(w in lower_message for w in ["exit", "quit", "cancel", "reset", "stop"]):
            request.session['intake_state'] = 'IDLE'
            request.session['intake_data'] = {}
            reply_text = "🔌 **[Session Terminated]**\n\nActive dialogue pipeline cleared. Returning to primary dashboard telemetry mode."
            return _log_and_reply(raw_message, "session_forced_reset", reply_text)
        
        if any(w in lower_message for w in ["checkup", "check up", "diagnostic", "fit or not", "evaluate me", "doctor"]):
            request.session['intake_state'] = 'AWAITING_NAME'
            request.session['intake_data'] = {}
            reply_text = "Clinical Intake Mode Engaged\n\nHello, I am your digital attending specialist. Let's run a complete fitness and diagnostic evaluation. To begin our medical record profile, what is your full name?**"
            return _log_and_reply(raw_message, "diagnostic_intake_start", reply_text)

        elif session_state == 'AWAITING_NAME':
            clean_name = raw_message.strip()
            # ⛔ VALIDATION: Reject names that match basic greetings, are empty, or contain digits
            if lower_message in GREETING_WORDS or len(clean_name) < 2 or bool(re.search(r'\d', clean_name)):
                name_error_replies = [
                    "📋 **Invalid Record Name**: Please enter a valid name string (letters only) for the medical chart registry.",
                    "⚠️ **Chart Registry Error**: That input does not look like a standard patient name. Please type a valid name.",
                    "🔍 **Intake Alert**: Please state the patient's full name explicitly to provision the diagnostic tracker log."
                ]
                return _log_and_reply(raw_message, "intake_name_validation_failed", random.choice(name_error_replies))
                
            intake_data['name'] = clean_name
            request.session['intake_state'] = 'AWAITING_AGE'
            request.session['intake_data'] = intake_data
            
            name_replies = [
                f"Thank you, {clean_name}. Noted in the chart.\n\n**What is your age?**",
                f"Got it, {clean_name}. Profile baseline registered.\n\n**Please provide your age next:**",
                f"System ledger updated with name: {clean_name}.\n\nMoving forward, **how old are you?**"
            ]
            return _log_and_reply(raw_message, "intake_name", random.choice(name_replies))

        elif session_state == 'AWAITING_AGE':
            age_match = re.search(r'\d+', lower_message)
            # ⛔ VALIDATION: Ensure a realistic numeric age is found
            if not age_match or not (1 <= int(age_match.group()) <= 120):
                age_error_replies = [
                    "🔢 **Invalid Entry**: Please provide a valid numerical value for your age (e.g., 34).",
                    "⚠️ **Numeric Parse Failure**: Age metric must be entered as a positive integer scale. What is your age?",
                    "📊 **Data Entry Alert**: Please state a realistic number for age so we can map your vital chart coordinates."
                ]
                return _log_and_reply(raw_message, "intake_age_validation_failed", random.choice(age_error_replies))
                
            intake_data['age'] = int(age_match.group())
            request.session['intake_state'] = 'AWAITING_HEIGHT'
            request.session['intake_data'] = intake_data
            
            age_replies = [
                "Understood. Now, please state your **height in centimeters (cm)**.",
                "Metric saved. What is your current **height (in cm)**?",
                "Acknowledged. Let's check your height coordinates next. Please enter your **height in centimeters**."
            ]
            return _log_and_reply(raw_message, "intake_age", random.choice(age_replies))

        elif session_state == 'AWAITING_HEIGHT':
            height_match = re.search(r'\d+', lower_message)
            # ⛔ VALIDATION: Ensure a realistic height value is parsed
            if not height_match or not (50 <= float(height_match.group()) <= 270):
                height_error_replies = [
                    "📏 **Invalid Entry**: Please state your height explicitly using numbers (e.g., 175).",
                    "⚠️ **Dimension Parse Error**: I couldn't extract a valid measurement value. What is your current height in centimeters (cm)?",
                    "📊 **Metric Framework Alert**: Please input a pure numeric height value (cm) so we can map your BMI profile coordinates."
                ]
                return _log_and_reply(raw_message, "intake_height_validation_failed", random.choice(height_error_replies))
                
            intake_data['height'] = float(height_match.group())
            request.session['intake_state'] = 'AWAITING_WEIGHT'
            request.session['intake_data'] = intake_data
            
            height_replies = [
                "Acknowledged. Next, please provide your current **weight in kilograms (kg)**.",
                "Height registered in centimeters. Next step: What is your **weight in kilograms (kg)**?",
                "Got your height dimension mapped. Please provide your current **weight (in kg)** to finish the biometrics check."
            ]
            return _log_and_reply(raw_message, "intake_height", random.choice(height_replies))

        elif session_state == 'AWAITING_WEIGHT':
            weight_match = re.search(r'\d+', lower_message)
            # ⛔ VALIDATION: Ensure a realistic weight value is parsed
            if not weight_match or not (10 <= float(weight_match.group()) <= 300):
                weight_error_replies = [
                    "⚖️ **Invalid Entry**: Please provide your weight using standard digits/numbers (e.g., 70).",
                    "⚠️ **Mass Matrix Error**: Could not parse structural numeric value. What is your weight in kilograms (kg)?",
                    "📊 **Data Entry Alert**: Please verify and enter a pure numeric baseline matrix for your weight status parameters."
                ]
                return _log_and_reply(raw_message, "intake_weight_validation_failed", random.choice(weight_error_replies))
                
            intake_data['weight'] = float(weight_match.group())
            request.session['intake_state'] = 'AWAITING_SYMPTOMS'
            request.session['intake_data'] = intake_data
            
            weight_replies = [
                "Got it. Finally, are you currently experiencing any **symptoms or physical discomfort?** (e.g., fatigue, pelvic discomfort, changes in habits, or write 'none')",
                "Weight metrics saved. Let's finish up: What exact **symptoms or physical discomforts** have you noticed lately? (Type 'none' if feeling fine)",
                "Biometrics calculation data loaded. To finish the evaluation tracker, tell me about your current **symptoms or acute discomfort**."
            ]
            return _log_and_reply(raw_message, "intake_weight", random.choice(weight_replies))

        elif session_state == 'AWAITING_SYMPTOMS':
            symptom_lower = raw_message.lower().strip()
            
            # Extract collected biometric variables
            h_m = intake_data['height'] / 100.0
            w_kg = intake_data['weight']
            bmi = round(w_kg / (h_m ** 2), 1)
            
            # Biometric Weight Scale Triage
            if bmi < 18.5:
                fitness_status = "Underweight Status (Nutritional optimization recommended)"
            elif bmi < 25.0:
                fitness_status = "Excellent / Normal Weight Range (Metabolic health optimal)"
            elif bmi < 30.0:
                fitness_status = "Overweight Status (Cardiovascular conditioning advised)"
            else:
                fitness_status = "Obese Range (Clinical lifestyle counseling advised)"

            # Check if user stated they have no symptoms
            is_no_symptoms = symptom_lower in ["no", "none", "nothing", "fine", "no symptoms", "nothin", "healthy"]

            if is_no_symptoms:
                intake_data['symptoms'] = "None Reported (Patient feeling healthy)"
                diagnostic_remark = "🟢 **Clear Ledger**: No acute oncology pathway or system telemetry symptoms flagged today. Maintenance of current routine advised."
                
                reply_text = (
                    f"👨‍⚕️ **[HEALTHY BASELINE LOGGED FOR {intake_data['name'].upper()}]**\n\n"
                    f"📊 **VITAL STATISTICS CHART**:\n"
                    f"• **Age**: {intake_data['age']} years old\n"
                    f"• **Biometrics**: {intake_data['height']} cm / {intake_data['weight']} kg\n"
                    f"• **Calculated BMI**: `{bmi}`\n\n"
                    f"🩺 **CLINICAL FITNESS EVALUATION**:\n"
                    f"• Status: {fitness_status}\n"
                    f"• Symptoms Reported: *\"None\"*\n"
                    f"• Diagnostic Assessment: {diagnostic_remark}\n\n"
                    f"✨ *Intake complete. Clean bill of health tracked. Dialogue session reset to home parameters.*"
                )
            else:
                # Regular processing for active symptom profiles
                intake_data['symptoms'] = raw_message
                if any(w in symptom_lower for w in ["pain", "blood", "pelvic", "discomfort"]):
                    diagnostic_remark = "⚠️ **Pre-clinical Alert**: Noted potential local inflammation or tissue stress indicators. Recommended screening tracking against the Pathology Telemetry matrix."
                else:
                    diagnostic_remark = "📋 **General Observation**: General baseline symptoms recorded. Monitor variance over a 14-day tracking cycle."

                reply_text = (
                    f"👨‍⚕️ **[INSPECTION COMPLETED FOR {intake_data['name'].upper()}]**\n\n"
                    f"📊 **VITAL STATISTICS CHART**:\n"
                    f"• **Age**: {intake_data['age']} years old\n"
                    f"• **Biometrics**: {intake_data['height']} cm / {intake_data['weight']} kg\n"
                    f"• **Calculated BMI**: `{bmi}`\n\n"
                    f"🩺 **CLINICAL FITNESS EVALUATION**:\n"
                    f"• Status: {fitness_status}\n"
                    f"• Symptoms Reported: *\"{intake_data['symptoms']}\"*\n"
                    f"• Diagnostic Assessment: {diagnostic_remark}\n\n"
                    f"⏱️ *Intake complete. Session logs successfully updated in database memory.*"
                )

            # Unconditionally clear the engine state room so the next turn is completely fresh
            request.session['intake_state'] = 'IDLE'
            request.session['intake_data'] = {}
            return _log_and_reply(raw_message, "diagnostic_intake_complete", reply_text)
        # ==========================================================
        # 👋 LAYER 0.2: GREETING / SMALL-TALK INTENTS
        # ==========================================================
        if _has_word(TIME_GREETING, lower_message):
            reply_text = _time_of_day_greeting()
            return _log_and_reply(raw_message, "greeting_time_of_day", reply_text)

        if _has_word(HOW_ARE_YOU, lower_message):
            reply_text = random.choice(HOW_ARE_YOU_REPLIES)
            return _log_and_reply(raw_message, "how_are_you", reply_text)

        if _has_word(GRATITUDE_WORDS, lower_message):
            reply_text = random.choice(GRATITUDE_REPLIES)
            return _log_and_reply(raw_message, "gratitude", reply_text)

        if _has_word(FAREWELL_WORDS, lower_message):
            reply_text = random.choice(FAREWELL_REPLIES)
            return _log_and_reply(raw_message, "farewell", reply_text)

        if _has_word(GREETING_WORDS, lower_message):
            reply_text = random.choice(GREETING_REPLIES)
            return _log_and_reply(raw_message, "greeting", reply_text)

        if _is_pure_affirmation(lower_message, AFFIRM_WORDS):
            reply_text = random.choice(AFFIRM_REPLIES)
            return _log_and_reply(raw_message, "affirmation", reply_text)

        # ==========================================================
        # 🎯 LAYER 1: DYNAMIC SYMPTOM DURATION & SEVERITY TRIAGE
        # ==========================================================
        
        # State 1-B: Capture incoming response to "How long have you felt this way?"
        if session_state == 'TRACKING_DURATION':
            intake_data['duration_text'] = raw_message
            request.session['intake_state'] = 'TRACKING_SEVERITY'
            reply_text = "Understood. On a scale from **1 to 10** (or mild, moderate, severe), how intense or painful is this feeling right now?"
            return _log_and_reply(raw_message, "triage_duration_response", reply_text)

        # State 2-B: Capture severity value, parse duration metrics, and output the tracking result
        elif session_state == 'TRACKING_SEVERITY':
            intake_data['severity_text'] = raw_message
            diagnosis = intake_data.get('suspected_diagnosis', 'General Triage Condition')
            duration_msg = intake_data.get('duration_text', 'unknown duration').lower()
            
            # Smart Chronicity Filter Matrix
            is_chronic = any(w in duration_msg for w in ["weeks", "months", "years", "long time", "chronic", "3 weeks", "14 days"])
            
            chronicity_label = "⚠️ **CHRONIC RISK TRAIL**" if is_chronic else "🟢 **ACUTE / TEMPORARY TIER**"
            timeline_guidance = (
                "This symptom has persisted over a long window. A comprehensive diagnostic screen is advised to rule out deeper systemic changes." 
                if is_chronic else 
                "This appears to be a sudden, acute flare-up. Monitor presentation over the next 48-72 hours."
            )

            reply_text = (
                f"👨‍⚕️ **[TRIAGE SYMPTOM MATRIX EVALUATED]**\n\n"
                f"🧬 **Suspected Indicator**: `{diagnosis}`\n"
                f"⏳ **Timeline Logged**: *\"{intake_data['duration_text']}\"*\n"
                f"⚡ **Reported Severity**: *\"{raw_message}\"* \n\n"
                f"🔬 **CLINICAL CHRONICITY ANALYSIS**:\n"
                f"• Category: {chronicity_label}\n"
                f"• Insight: {timeline_guidance}\n\n"
                f"👉 *To see specialized medical track care steps, type the words 'next step' below.*"
            )
            
            # Reset conversation parameters
            request.session['intake_state'] = 'IDLE'
            request.session['intake_data'] = {}
            return _log_and_reply(raw_message, "triage_severity_complete", reply_text)

        # Red Flag Emergency Filter Check
        if is_simple_query and _has_word(RED_FLAG_WORDS, lower_message):
            reply_text = "That sounds serious — please seek emergency care or call your local emergency number right away. I'm not equipped to help with this."
            return _log_and_reply(raw_message, "red_flag_escalation", reply_text)

        # Intercept simple static diagnoses targets to run the new triage loop
        if is_simple_query:
            for diagnosis, keywords in STATIC_INTENTS.items():
                if _has_word(keywords, lower_message):
                    # Cache the baseline diagnosis guess, then prompt for timing parameters
                    request.session['intake_state'] = 'TRACKING_DURATION'
                    request.session['intake_data'] = {'suspected_diagnosis': diagnosis}
                    
                    reply_text = f"👨‍⚕️ I notice you mentioned indicators that trace closely with **{diagnosis}**.\n\nTo safely analyze this, **from how much time have you been feeling these things?**"
                    return _log_and_reply(raw_message, "triage_intercept_start", reply_text)

        # ==========================================================
        # 📋 LAYER 1.5: PATIENT METADATA & LENGTH OF STAY FORECAST
        # ==========================================================
        # ... (Keep your original patient metadata lookup code here safely) ...
        # ==========================================================
        # 🤖 LAYER 2: MACHINE LEARNING INTENT CLASSIFICATION ENGINE
        # ==========================================================
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
            engine_used = "System Default Fallback (ML Processing Error)"

        # ==========================================================
        # 🎨 LAYER 3: LAYOUT WRAPPER & CARE TRACK DISCLAIMER
        # ==========================================================
        if "Fallback" in engine_used or predicted_diagnosis == "General Triage Consultation Required":
            reply_text = (
                "Hi there! Thanks for reaching out to the Care Portal.\n\n"
                "I want to make sure I understand this correctly before pointing you anywhere. "
                "Could you tell me a bit more — how long you've felt this way, or if there are any other symptoms alongside it?"
            )
        else:
            track_lower = predicted_diagnosis.lower()
            matched_track = None

            for track in CARE_TRACKS.values():
                if _has_word(track["match"], track_lower):
                    matched_track = track
                    break

            if matched_track:
                diagnosis_title = matched_track["title"]
                cure_plan = matched_track["cure_plan"]
                precautions = matched_track["precautions"]
            else:
                diagnosis_title = f"{predicted_diagnosis} Clinical Review"
                cure_plan = DEFAULT_TRACK["cure_plan"]
                precautions = DEFAULT_TRACK["precautions"]

            reply_text = (
                f"Symptoms of:\n`{diagnosis_title}`\n\n"
                f"Recommended next step:\n"
                f"{cure_plan}\n\n"
                f"Precautions:\n"
                f"{precautions}\n\n"
                f"_This is general guidance, not a diagnosis — please confirm with a clinician._"
            )

        display_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
        processed_message = ", ".join(display_tokens) if display_tokens else "None"

        return _log_and_reply(raw_message, processed_message, reply_text)

    except Exception as e:
        return JsonResponse({'error': f"NLP Exception Handling Request: {str(e)}"}, status=500)

# ==========================================================
# 🥗 SELF-CHECK / DIET ENGINE VIEW
# (Restored to module level — this was previously defined *nested inside*
# chatbot_view, after its return statement, which meant Django could never
# import or route to it. The Self-Check Drawer feature was non-functional.)
# ==========================================================

def self_check_diet_engine_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Direct interface browser GET requests not supported on this data endpoint.'}, status=400)

    try:
        data = json.loads(request.body)
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))
        age = int(data.get('age', 0))
        gender = str(data.get('gender', 'M')).strip().upper()[:1]  # normalizes 'Male'/'male' -> 'M'
        goal = data.get('goal', 'MAINTAIN')
        # 🥦 Extract dietary preference parameter (defaulting to 'VEG')
        diet_pref = str(data.get('diet_preference', 'VEG')).strip().upper()
    except (TypeError, ValueError, json.JSONDecodeError):
        return JsonResponse({'error': 'Invalid or missing input values.'}, status=400)

    if height <= 0 or weight <= 0 or age <= 0:
        return JsonResponse({'error': 'Height, weight, and age must be positive numbers.'}, status=400)

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

    # Calculate BMR and TDEE based on biological gender bounds
    if gender == 'M':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    tdee = bmr * 1.375

    # Determine Calorie Targets & Macro Split profiles based on Goal
    if goal == "LOSE":
        target_calories = int(tdee - 500)
        macro_split = "High Protein / Deficit Taper (40% P / 30% C / 30% F)"
    elif goal == "GAIN":
        target_calories = int(tdee + 400)
        macro_split = "Clean Mass Gain / Surplus Matrix (30% P / 40% C / 30% F)"
    else:
        target_calories = int(tdee)
        macro_split = "Balanced Nutrition Maintenance Profile (30% P / 45% C / 25% F)"

    # ==========================================================
    # 🥗 CLINICAL DYNAMIC MEAL GENERATOR TREE (BMI + GENDER + DIET)
    # ==========================================================
    
    # 🛑 GUARD CONDITION: If the category is Normal Weight, skip the meal plan completely
    if bmi_tier == "Normal Weight":
        diet_plan = (
            f"✨ **Status**: Excellent / Normal Weight Range\n"
            f"🩺 **Clinical Target**: {bmi_tier} ({gender})\n\n"
            f"💡 **Guidance**: Your current biometrics fall within a highly optimal metabolic weight range. "
            f"No corrective therapeutic diet plan is required at this time. Focus on intuitive, well-balanced eating "
            f"and maintaining your active lifestyle to sustain these current telemetry baselines!"
        )
    
    # 🟠 CATEGORY 1: OVERWEIGHT / OBESE PROFILES
    elif bmi_tier in ["Overweight", "Obese"]:
        if diet_pref in ["NON_VEG", "NON-VEG"]:
            if gender == 'F':
                meals = "• Breakfast: Egg white wrap with spinach and iron-rich kale\n• Lunch: Grilled lemon chicken breast with fiber-dense broccoli\n• Dinner: Baked cod with asparagus and a side of mixed greens"
            else:  # Male
                meals = "• Breakfast: 4 Egg white scramble with grilled mushrooms\n• Lunch: Baked turkey breast with broccoli and zero-calorie shirataki noodles\n• Dinner: Broiled lean sirloin steak with asparagus and roasted zucchini"
        else:  # Vegetarian
            if gender == 'F':
                meals = "• Breakfast: Low-fat paneer bhurji with bell peppers and fortified soy milk\n• Lunch: Tofu stir-fry with mixed greens, broccoli, and light sesame dressing\n• Dinner: Lentil and mung bean sprouts salad with lemon zest and steamed cauliflower"
            else:  # Male
                meals = "• Breakfast: Thick tofu scramble with spinach and unsweetened almond milk\n• Lunch: High-protein soya chunks curry with steamed broccoli florets\n• Dinner: Grilled tempeh steak with roasted cauliflower and sautéed asparagus"

        # Assemble structural markdown string for overweight categories
        diet_plan = (
            f"📊 **Macronutrient Profile**: {macro_split}\n"
            f"🍏 **Dietary Preference**: {diet_pref}\n"
            f"🩺 **Target Weight Class**: {bmi_tier} ({gender})\n\n"
            f"💡 **Tailored Meal Structure**:\n{meals}"
        )

    # 🔵 CATEGORY 2: UNDERWEIGHT PROFILES
    else:  # Underweight
        if diet_pref in ["NON_VEG", "NON-VEG"]:
            if gender == 'F':
                meals = "• Breakfast: Whole egg omelet with avocado and whole-milk chia seed pudding\n• Lunch: Salmon fillet with olive oil dressing, sweet potatoes, and walnuts\n• Dinner: Rich chicken thigh curry cooked in coconut oil with basmati rice"
            else:  # Male
                meals = "• Breakfast: 4 Whole eggs with peanut butter oatmeal and sliced bananas\n• Lunch: Lean beef chunks or chicken thighs with a double portion of white rice\n• Dinner: Seared tuna steak with mashed potatoes cooked in grass-fed butter"
        else:  # Vegetarian
            if gender == 'F':
                meals = "• Breakfast: Oatmeal cooked in whole milk with peanut butter, dates, and hemp seeds\n• Lunch: Full-fat paneer cubes and chickpea curry with paratha or brown rice\n• Dinner: Thick lentil stew with roasted almonds, edamame salad, and sweet potato fries"
            else:  # Male
                meals = "• Breakfast: High-calorie plant protein shake with oats, peanut butter, banana, and oat milk\n• Lunch: Kidney bean (Rajma) and paneer thick curry with white basmati rice\n• Dinner: Stuffed tempeh wraps with avocado mash, roasted cashews, and olive oil dressing"

        # Assemble structural markdown string for underweight categories
        diet_plan = (
            f"📊 **Macronutrient Profile**: {macro_split}\n"
            f"🍏 **Dietary Preference**: {diet_pref}\n"
            f"🩺 **Target Weight Class**: {bmi_tier} ({gender})\n\n"
            f"💡 **Tailored Meal Structure**:\n{meals}"
        )

    # Save tracking parameters to the local metric model ledger table regardless of category
    UserSelfCheckMetric.objects.create(
        age=age, gender=gender, height_cm=height, weight_kg=weight, fitness_goal=goal,
        calculated_bmi=bmi, bmi_category=bmi_tier, recommended_calories=target_calories,
        diet_plan_markdown=diet_plan
    )

    return JsonResponse({
        'bmi': bmi,
        'category': bmi_tier,
        'calories': target_calories,
        'diet_plan': diet_plan
    })