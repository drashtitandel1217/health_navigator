from django.shortcuts import render

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