import os
import pandas as pd
from django.core.management.base import BaseCommand
from patients.models import ExcelPatientRecord

class Command(BaseCommand):
    help = "Reads the parsed Excel database from Desktop and migrates it into the Django SQLite Database"

    def handle(self, *args, **options):
        # 📂 Target your spreadsheet path on your Desktop
        excel_path = "/Users/drashtitandel12/Desktop/patient_database.xlsx"

        if not os.path.exists(excel_path):
            self.stdout.write(self.style.ERROR(f"Excel file not found at '{excel_path}'. Run export_to_excel first."))
            return

        self.stdout.write(self.style.SUCCESS(f"Reading data from: {excel_path}"))

        # 📊 Read the Excel file using Pandas
        df = pd.read_excel(excel_path)
        
        # Replace empty/NaN values with empty strings or default text safely
        df = df.fillna("")

        count = 0
        # 🔄 Loop through each row in the spreadsheet and save it to the database
        for index, row in df.iterrows():
            mrn = str(row["Medical Record Number (MRN)"]).strip()
            
            # Use update_or_create so it updates records cleanly if run multiple times without duplicating
            record, created = ExcelPatientRecord.objects.update_or_create(
                medical_record_number=mrn,
                defaults={
                    "patient_name": str(row["Patient Name"]).strip(),
                    "date_of_birth": str(row["Date of Birth"]).strip(),
                    "date_of_admission": str(row["Date of Admission"]).strip(),
                    "date_of_discharge": str(row["Date of Discharge"]).strip(),
                    "attending_physician": str(row["Attending Physician"]).strip(),
                    "primary_diagnosis": str(row["Primary Diagnosis"]).strip(),
                    "medical_history_summary": str(row["Medical History Summary"]).strip(),
                    "treatment_plan": str(row["Treatment Plan"]).strip(),
                }
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"🎉 SUCCESS! Migrated {count} rows from Excel straight into your Django Database."))