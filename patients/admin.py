from django.contrib import admin

# Register your models here.
from .models import PatientMaster, HospitalEncounter, DiagnosticReport, DischargeCarePlan, ExcelPatientRecord, ChatbotInquiryLog

# Register your tables so they appear on the admin panel
admin.site.register(PatientMaster)
admin.site.register(HospitalEncounter)
admin.site.register(DiagnosticReport)
admin.site.register(DischargeCarePlan)

@admin.register(ChatbotInquiryLog)
class ChatbotInquiryLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'raw_input_text', 'processed_input_text', 'ai_response_reply')
    search_fields = ('raw_input_text', 'processed_input_text', 'ai_response_reply')
    list_filter = ('timestamp',)

@admin.register(ExcelPatientRecord)
class ExcelPatientRecordAdmin(admin.ModelAdmin):
    # Sets up professional table headers matching your spreadsheet layout
    list_display = ('medical_record_number', 'patient_name', 'date_of_birth', 'primary_diagnosis', 'attending_physician')
    search_fields = ('patient_name', 'medical_record_number', 'primary_diagnosis')
    list_filter = ('attending_physician',)