# patients/management/commands/load_data.py
import os
import re
from datetime import datetime
from django.core.management.base import BaseCommand
from patients.models import PatientMaster, HospitalEncounter, DiagnosticReport, DischargeCarePlan

def extract_field(pattern, text, default_value="Not Specified"):
    """Helper function to find specific text headers in the files using Regex"""
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else default_value

# Django looks for this specific class to run your code
class Command(BaseCommand):
    help = 'Loads text patient records from Downloads folder into the database'

    def handle(self, *args, **options):
        # Using your updated downloads path
        dataset_path = "/Users/drashtitandel12/my_dataset_folder/llama"
        
        self.stdout.write(f"Scanning dataset folder: '{dataset_path}'...")
        
        if not os.path.exists(dataset_path):
            self.stdout.write(self.style.ERROR(f"Error: The folder '{dataset_path}' was not found."))
            return

        imported_count = 0

        # Loop through every text file in the dataset directory
        for filename in os.listdir(dataset_path):
            if filename.endswith(".txt") or filename.endswith(".txt.txt"):
                file_path = os.path.join(dataset_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                
                p_name = extract_field(r"•Name:\s*(.*)", text_content, "Unknown Patient")
                p_dob_str = extract_field(r"•Date of Birth:\s*(.*)", text_content, "01/01/1990")
                p_mrn = extract_field(r"•Medical Record Number:\s*(.*)", text_content, f"MRN-{imported_count}")
                p_adm_str = extract_field(r"•Date of Admission:\s*(.*)", text_content, "01/01/2023")
                p_dis_str = extract_field(r"•Date of Discharge:\s*(.*)", text_content, "01/01/2023")
                p_doctor = extract_field(r"•Attending Physician:\s*(.*)", text_content, "Unknown Doctor")
                p_diagnosis = extract_field(r"•Primary Diagnosis:\s*(.*)", text_content, "Not Specified")
                
                try:
                    p_dob = datetime.strptime(p_dob_str, '%m/%d/%Y').date()
                    p_adm = datetime.strptime(p_adm_str, '%m/%d/%Y').date()
                    p_dis = datetime.strptime(p_dis_str, '%m/%d/%Y').date()
                except ValueError:
                    p_dob = p_adm = p_dis = datetime.now().date()

                med_history = extract_field(r"Medical History:\s*(.*?)(?=Diagnostic Findings:|$)", text_content)
                diagnostics = extract_field(r"Diagnostic Findings:\s*(.*?)(?=Treatment Plan:|$)", text_content)
                hospital_course = extract_field(r"Hospital Course:\s*(.*?)(?=Follow-Up Plan:|$)", text_content)
                care_plan_text = extract_field(r"Follow-Up Plan:\s*(.*?)(?=Patient Education:|$)", text_content)

                # --- INSERT DATA DYNAMICALLY VIA DJANGO ORM ---
                patient, created = PatientMaster.objects.get_or_create(
                    patient_id=p_mrn,
                    defaults={
                        'name': p_name,
                        'date_of_birth': p_dob,
                        'chronic_conditions': med_history,
                    }
                )

                encounter, created = HospitalEncounter.objects.get_or_create(
                    patient=patient,
                    date_of_admission=p_adm,
                    date_of_discharge=p_dis,
                    defaults={
                        'attending_physician': p_doctor,
                        'primary_diagnosis': p_diagnosis,
                        'reason_for_admission': f"Presented for evaluation under MRN {p_mrn}.",
                        'hospital_course_summary': hospital_course
                    }
                )

                DiagnosticReport.objects.get_or_create(
                    encounter=encounter,
                    patient=patient,
                    test_type='Admission Diagnostic Baseline Summary',
                    defaults={'pathology_details': diagnostics}
                )

                DischargeCarePlan.objects.get_or_create(
                    encounter=encounter,
                    defaults={
                        'patient': patient,
                        'discharge_medications': "Refer to text instructions below",
                        'follow_up_appointments': care_plan_text,
                        'activity_restrictions': care_plan_text,
                        'dietary_hydration_directives': care_plan_text,
                        'urgent_warning_signs': care_plan_text
                    }
                )

                imported_count += 1
                self.stdout.write(f"Successfully processed: {p_name} ({p_mrn})")

        self.stdout.write(self.style.SUCCESS(f"\nCompleted! {imported_count} files read and mapped successfully."))