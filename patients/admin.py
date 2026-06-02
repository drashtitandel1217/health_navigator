from django.contrib import admin

# Register your models here.
from .models import PatientMaster, HospitalEncounter, DiagnosticReport, DischargeCarePlan

# Register your tables so they appear on the admin panel
admin.site.register(PatientMaster)
admin.site.register(HospitalEncounter)
admin.site.register(DiagnosticReport)
admin.site.register(DischargeCarePlan)