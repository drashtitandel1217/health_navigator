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

        # ==========================================================
        # 👋 LAYER 0: GREETING / SMALL-TALK INTENTS
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

        # Only the guarded "pure affirmation" check — the unguarded broad
        # substring version was removed since it made this one unreachable
        # and risked swallowing real symptom messages that start with "ok".
        if _is_pure_affirmation(lower_message, AFFIRM_WORDS):
            reply_text = random.choice(AFFIRM_REPLIES)
            return _log_and_reply(raw_message, "affirmation", reply_text)

        # ==========================================================
        # 📋 LAYER 0.5: PATIENT METADATA & LENGTH OF STAY FORECAST
        # ==========================================================
        has_forecast_keywords = _has_word(
            ["stay", "forecast", "timeline", "discharge", "days", "predict", "hospital", "duration", "length"],
            lower_message
        )
        has_mrn = bool(re.search(r"\b[a-zA-Z]\d{3}\b", raw_message))
        has_followup = _has_word(
            FOLLOWUP_DIAGNOSIS_WORDS + FOLLOWUP_PHYSICIAN_WORDS + FOLLOWUP_STATUS_WORDS,
            lower_message
        )
        has_correction = _has_word(CORRECTION_WORDS, lower_message)

        has_person = False
        if nlp:
            doc = nlp(raw_message)
            has_person = any(ent.label_ == "PERSON" for ent in doc.ents)

        if has_correction:
            reply_text = "No problem — who's the correct patient? Name or MRN works."
            return _log_and_reply(raw_message, "patient_lookup_correction", reply_text)

        if has_forecast_keywords or has_mrn or has_person or "patient" in lower_message or has_followup:
            patient = parse_patient_from_query(raw_message)

            if not patient:
                reply_text = "I couldn't find a matching patient record. Could you give me a name or MRN?"
                return _log_and_reply(raw_message, "patient_not_found", reply_text)

            if has_followup and not (has_forecast_keywords or has_mrn):
                if _has_word(FOLLOWUP_DIAGNOSIS_WORDS, lower_message):
                    reply_text = f"{patient.patient_name}'s primary diagnosis is {patient.primary_diagnosis}."
                    return _log_and_reply(raw_message, patient.patient_name, reply_text)

                if _has_word(FOLLOWUP_PHYSICIAN_WORDS, lower_message):
                    reply_text = f"Dr. {patient.attending_physician} is attending on this case."
                    return _log_and_reply(raw_message, patient.patient_name, reply_text)

                if _has_word(FOLLOWUP_STATUS_WORDS, lower_message):
                    reply_text = "Let me pull the latest — want the full stay forecast too?"
                    return _log_and_reply(raw_message, patient.patient_name, reply_text)

            predicted_stay = predict_patient_stay(patient)

            if isinstance(predicted_stay, (int, float)):
                reply_text = (
                    _build_patient_card(patient) +
                    f"🔮 **Length of Stay Forecast**: `{predicted_stay}` Days\n\n"
                    f"⚠️ *Disclaimer: Prediction generated via Random Forest Regressor Module.*"
                )
            else:
                reply_text = (
                    _build_patient_card(patient) +
                    f"⚠️ **Stay Forecast Error**: {predicted_stay}"
                )

            return _log_and_reply(raw_message, patient.patient_name, reply_text)

        # ==========================================================
        # 🎯 LAYER 1: STATIC INTENTS FOR COMMON MINOR ILLNESSES
        # ==========================================================
        if is_simple_query and _has_word(RED_FLAG_WORDS, lower_message):
            reply_text = (
                "That sounds serious — please seek emergency care or call your local "
                "emergency number right away. I'm not equipped to help with this."
            )
            return _log_and_reply(raw_message, "red_flag_escalation", reply_text)

        if is_simple_query:
            for diagnosis, keywords in STATIC_INTENTS.items():
                if _has_word(keywords, lower_message):
                    predicted_diagnosis = diagnosis
                    engine_used = "Static Intent Engine"

                    if _has_word(CONCERN_WORDS, lower_message):
                        prefix = "I hear you, that's a fair thing to worry about. "
                    elif _has_word(DURATION_WORDS, lower_message):
                        prefix = "Thanks for the detail on timing. "
                    else:
                        prefix = ""

                    reply_text = f"{prefix}This sounds like it could be {diagnosis}."
                    return _log_and_reply(raw_message, diagnosis, reply_text)

        # ==========================================================
        # 🤖 LAYER 2: MACHINE LEARNING INTENT CLASSIFICATION ENGINE
        # ==========================================================
        # NOTE: This must run unconditionally whenever Layer 1 didn't already
        # return above — NOT only when is_simple_query is False. Previously
        # this was wired as `else:` against `if is_simple_query:`, which meant
        # short messages that matched no static keyword fell through with
        # engine_used/predicted_diagnosis never assigned, causing an
        # UnboundLocalError on every such message.
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

        ChatbotInquiryLog.objects.create(
            raw_input_text=raw_message,
            processed_input_text=processed_message,
            ai_response_reply=reply_text
        )

        return JsonResponse({'reply': reply_text})

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

    if gender == 'M':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    tdee = bmr * 1.375

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