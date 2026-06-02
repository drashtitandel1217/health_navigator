import os
import re
import pandas as pd
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Scans raw text dataset files, splits strings by known headers, and exports perfectly to Excel"

    def handle(self, *args, **options):
        dataset_path = "/Users/drashtitandel12/my_dataset_folder/llama"
        
        if not os.path.exists(dataset_path):
            self.stdout.write(self.style.ERROR(f"Folder '{dataset_path}' not found."))
            return

        self.stdout.write(self.style.SUCCESS(f"Scanning files for Excel conversion in: '{dataset_path}'"))
        patient_records = []

        for filename in os.listdir(dataset_path):
            if filename.endswith(".txt") or filename.endswith(".txt.txt"):
                file_path = os.path.join(dataset_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    # 🧼 Remove potential leading bullets or spaces at the very start of the file
                    clean_content = content.strip().lstrip('•').strip()

                    # 🔍 1. EXTRACT PATIENT NAME (Everything before 'Date of Birth:')
                    p_name = "Unknown"
                    if "Date of Birth:" in clean_content:
                        raw_name_block = clean_content.split("Date of Birth:")[0].strip()
                        
                        # 🧼 If it contains the Patient Information block header, peel it away!
                        if "•Name:" in raw_name_block:
                            p_name = raw_name_block.split("•Name:")[1].strip()
                        elif "Patient Information:" in raw_name_block:
                            p_name = raw_name_block.split("Patient Information:")[1].strip()
                        else:
                            p_name = raw_name_block.rstrip('•').strip()
                            
                    # 🔍 2. EXTRACT PRIMARY DIAGNOSIS (Everything between 'Primary Diagnosis:' and 'Reason for Admission:')
                    p_diag = "Unknown"
                    if "Primary Diagnosis:" in clean_content:
                        after_diag = clean_content.split("Primary Diagnosis:")[1]
                        # Cut off at the next logical section header
                        for header in ["Reason for Admission:", "•", "Medical History:"]:
                            if header in after_diag:
                                after_diag = after_diag.split(header)[0]
                        p_diag = after_diag.strip().rstrip('•').strip()

                    # 🔍 3. STANDARD REGEX FOR KEY FIELDS (These are already working perfectly for you!)
                    dob_match = re.search(r"Date of Birth:\s*([\d/]+)", content)
                    mrn_match = re.search(r"Medical Record Number:\s*(\w+)", content)
                    admission_match = re.search(r"Date of Admission:\s*([\d/]+)", content)
                    discharge_match = re.search(r"Date of Discharge:\s*([\d/]+)", content)
                    physician_match = re.search(r"Attending Physician:\s*([A-Za-z\s\.\-]+?)(?=•|Primary Diagnosis:)", content)

                    # 🔍 4. BLOCK PARSERS FOR LARGE TEXT
                    history_match = re.search(r"Medical History:\s*(.*?)(?=Diagnostic Findings:|$)", content, re.DOTALL)
                    plan_match = re.search(r"Treatment Plan:\s*(.*?)(?=Hospital Course:|$)", content, re.DOTALL)

                    record = {
                        "Medical Record Number (MRN)": mrn_match.group(1).strip() if mrn_match else "Unknown",
                        "Patient Name": p_name,
                        "Date of Birth": dob_match.group(1).strip() if dob_match else "Unknown",
                        "Date of Admission": admission_match.group(1).strip() if admission_match else "Unknown",
                        "Date of Discharge": discharge_match.group(1).strip() if discharge_match else "Unknown",
                        "Attending Physician": physician_match.group(1).strip() if physician_match else "Unknown",
                        "Primary Diagnosis": p_diag,
                        "Medical History Summary": history_match.group(1).strip() if history_match else "N/A",
                        "Treatment Plan": plan_match.group(1).strip() if plan_match else "N/A"
                    }
                    
                    patient_records.append(record)

        # 📊 Convert records to a DataFrame and save to your Desktop
        df = pd.DataFrame(patient_records)
        output_excel_path = "/Users/drashtitandel12/Desktop/patient_database.xlsx"
        df.to_excel(output_excel_path, index=False, sheet_name="Patient Master Data")

        self.stdout.write(self.style.SUCCESS(
            f"\n🎉 SUCCESS! Excel database updated cleanly at: {output_excel_path}"
        ))