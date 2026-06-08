from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from .utils import parse_patient_from_query, predict_patient_stay
# Create your views here.
from .models import ExcelPatientRecord

def patient_dashboard_view(request):
    # 🧠 Fetch all 2,001 records directly from your database table
    all_patients = ExcelPatientRecord.objects.all()
    
    # Send that data package into your HTML template layout
    context = {
        'patients': all_patients
    }
    return render(request, 'patients/dashboard.html', context)


@csrf_protect
def chatbot_response_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({'reply': "I didn't catch that. Could you please provide a patient name or MRN?"})
            
        # 🧠 Execute NLP extraction track
        patient = parse_patient_from_query(user_message)
        
        if patient:
            # 🔮 Call your Random Forest machine learning prediction utility
            predicted_days = predict_patient_stay(patient)
            
            # 💬 Structure an intelligent conversational response context block
            reply = (
                f"✨ <b>NLP Analysis Complete:</b> Identified record for patient <b>{patient.patient_name}</b> (MRN: {patient.medical_record_number}).<br><br>"
                f"📋 <b>Primary Diagnosis:</b> {patient.primary_diagnosis}<br>"
                f"🔮 <b>AI Predicted Stay Length:</b> <span class='text-emerald-600 font-bold'>{predicted_days} Days</span>.<br>"
                f"🩺 <i>Attending Physician: {patient.attending_physician}</i>"
            )
        else:
            reply = (
                "Sorry, my NLP model couldn't map any patient name or MRN from your query. "
                "Try phrasing it like: <i>'What is the length of stay for Sherry Emery?'</i> or <i>'Check record L999'</i>."
            )
            
        return JsonResponse({'reply': reply})
        
    return JsonResponse({'error': 'Invalid request method'}, status=400)