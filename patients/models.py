from django.db import models

# Create your models here.

class PatientMaster(models.Model):
    patient_id = models.CharField(max_length=50, unique=True, primary_key=True)
    name = models.CharField(max_length=100)
    date_of_birth = models.DateField()
    gender = models.CharField(max_length=20, default="Unknown")
    chronic_conditions = models.TextField(blank=True, null=True)
    allergies = models.CharField(max_length=250, default="No known drug allergies")
    family_medical_history = models.TextField(blank=True, null=True)
    lifestyle_habits = models.TextField(blank=True, null=True)

    # 🛑 ADD THESE 2 LINES HERE:
    class Meta:
        app_label = 'patients'

    def __str__(self):
        return self.name

class HospitalEncounter(models.Model):
    patient = models.ForeignKey(PatientMaster, on_delete=models.CASCADE, related_name='encounters')
    date_of_admission = models.DateField()
    date_of_discharge = models.DateField()
    attending_physician = models.CharField(max_length=100)
    primary_diagnosis = models.CharField(max_length=255)
    reason_for_admission = models.TextField()
    hospital_course_summary = models.TextField()

    def __str__(self):
        return f"Encounter {self.id} - {self.patient.name}"

class DiagnosticReport(models.Model):
    encounter = models.ForeignKey(HospitalEncounter, on_delete=models.CASCADE, related_name='diagnostics')
    patient = models.ForeignKey(PatientMaster, on_delete=models.CASCADE)
    test_type = models.CharField(max_length=100)  # e.g., CT Scan, Biopsy
    tumor_size_location = models.CharField(max_length=255, blank=True, null=True)
    pathology_details = models.TextField(blank=True, null=True)
    lab_abnormalities = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.test_type} for {self.patient.name}"

class DischargeCarePlan(models.Model):
    encounter = models.OneToOneField(HospitalEncounter, on_delete=models.CASCADE, primary_key=True)
    patient = models.ForeignKey(PatientMaster, on_delete=models.CASCADE)
    surgical_history = models.TextField(blank=True, null=True)
    discharge_medications = models.TextField()
    chemotherapy_regimen = models.TextField(blank=True, null=True)
    radiation_regimen = models.TextField(blank=True, null=True)
    follow_up_appointments = models.TextField()
    activity_restrictions = models.TextField()
    dietary_hydration_directives = models.TextField()
    urgent_warning_signs = models.TextField()

    def __str__(self):
        return f"Care Plan for {self.patient.name}"
    

class ExcelPatientRecord(models.Model):
    # Use the MRN as the unique primary identifier
    medical_record_number = models.CharField(max_length=50, primary_key=True, verbose_name="MRN")
    patient_name = models.CharField(max_length=255)
    date_of_birth = models.CharField(max_length=20)  # Keeping as string to match your text format safely
    date_of_admission = models.CharField(max_length=20)
    date_of_discharge = models.CharField(max_length=20)
    attending_physician = models.CharField(max_length=255)
    primary_diagnosis = models.CharField(max_length=255)
    
    # Using TextField for long clinical notes and summaries
    medical_history_summary = models.TextField(blank=True, null=True)
    treatment_plan = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.patient_name} ({self.medical_record_number})"

    class Meta:
        verbose_name = "Excel Patient Record"
        verbose_name = "Excel Patient Records"