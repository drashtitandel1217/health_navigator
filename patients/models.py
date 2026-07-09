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

    class Meta:
        app_label = 'patients'

    def __str__(self):
        return self.name

class HospitalEncounter(models.Model):
    patient = models.ForeignKey(PatientMaster, on_delete=models.CASCADE, related_name='encounters')
    date_of_admission = models.DateField()
    date_of_discharge = models.DateField()
    attending_physician = models.CharField(max_length=100, default="General Staff")
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
    # 🛠️ FIX 1: Set primary_key=True so Django drops the ghost 'id' field tracking requirement!
    medical_record_number = models.CharField(max_length=100, unique=True, primary_key=True)
    patient_name = models.CharField(max_length=255)
    date_of_birth = models.DateField(blank=True, null=True)
    date_of_admission = models.DateField(blank=True, null=True)
    date_of_discharge = models.DateField(blank=True, null=True)
    primary_diagnosis = models.CharField(max_length=255)
    
    # 🛠️ FIX 2: Added blank=True and null=True so it's not compulsory in the Add Form screen!
    attending_physician = models.CharField(max_length=255, blank=True, null=True, default="Unassigned")
    medical_history_summary = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.patient_name} ({self.medical_record_number})"
    
    # 🛠️ FIX 3: Placed the correct naming parameters onto the right model tracking metadata block
    class Meta:
        verbose_name = "Excel Patient Record"
        verbose_name_plural = "Excel Patient Records"
    
class ChatbotInquiryLog(models.Model):
    raw_input_text = models.TextField()
    processed_input_text = models.TextField()
    ai_response_reply = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query on {self.timestamp.strftime('%Y-%m-%d %H:%M')}: {self.raw_input_text[:30]}..."

    class Meta:
        verbose_name = "Chatbot Inquiry Log"
        verbose_name_plural = "Chatbot Inquiry Logs"

class UserSelfCheckMetric(models.Model):
    GENDER_CHOICES = [('M', 'Male'), ('F', 'Female'), ('O', 'Other')]
    GOAL_CHOICES = [('LOSE', 'Weight Loss'), ('MAINTAIN', 'Maintenance'), ('GAIN', 'Muscle Gain')]

    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    height_cm = models.FloatField()
    weight_kg = models.FloatField()
    fitness_goal = models.CharField(max_length=10, choices=GOAL_CHOICES)
    
    # AI System Generated Outputs
    calculated_bmi = models.FloatField(blank=True, null=True)
    bmi_category = models.CharField(max_length=30, blank=True, null=True)
    recommended_calories = models.IntegerField(blank=True, null=True)
    diet_plan_markdown = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Self Check - {self.timestamp.strftime('%Y-%m-%d')}"
