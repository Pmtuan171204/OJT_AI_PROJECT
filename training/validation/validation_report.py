"""
==========================================================
Validation Report
OJT AI Project

Generate validation reports after dataset validation.
==========================================================
"""

import os
import sys
from datetime import datetime

import pandas as pd

# ======================================================
# Add Project Root
# ======================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ======================================================
# Import Paths
# ======================================================

from config.paths import REPORT_FOLDER, VALIDATION_REPORT

# ======================================================
# Validation Report
# ======================================================


class ValidationReport:

    def __init__(self, results):

        self.results = results

    # ==================================================

    def calculate_score(self):

        passed = sum(result["status"] for result in self.results)

        total = len(self.results)

        score = round(passed / total * 100, 2)

        return passed, total, score

    # ==================================================

    def dataset_status(self, score):

        if score == 100:
            return "READY FOR TRAINING"

        if score >= 90:
            return "READY (Minor Issues)"

        return "NOT READY"

    # ==================================================

    def export_txt(self):

        os.makedirs(REPORT_FOLDER, exist_ok=True)

        passed, total, score = self.calculate_score()

        status = self.dataset_status(score)

        with open(VALIDATION_REPORT, "w", encoding="utf-8") as f:

            f.write("=" * 60 + "\n")
            f.write("OJT DATASET VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated Time : " f"{datetime.now()}\n\n")

            f.write(f"Passed Rules : {passed}\n")

            f.write(f"Failed Rules : {total-passed}\n")

            f.write(f"Data Quality Score : {score}%\n")

            f.write(f"Dataset Status : {status}\n\n")

            f.write("=" * 60 + "\n")

            for result in self.results:

                text = "PASS" if result["status"] else "FAIL"

                f.write(f"[{text}] " f"{result['rule']}\n")

                f.write(f"{result['message']}\n")

                f.write("-" * 60 + "\n")

        print()

        print("=" * 60)
        print("Validation Report Generated")
        print("=" * 60)

        print(VALIDATION_REPORT)

    # ==================================================

    def export_excel(self):

        os.makedirs(REPORT_FOLDER, exist_ok=True)

        excel_path = os.path.join(REPORT_FOLDER, "validation_report.xlsx")

        rows = []

        for result in self.results:

            rows.append(
                {
                    "Rule": result["rule"],
                    "Status": "PASS" if result["status"] else "FAIL",
                    "Message": result["message"],
                }
            )

        df = pd.DataFrame(rows)

        df.to_excel(excel_path, index=False)

        print(excel_path)

    # ==================================================

    def generate(self):

        self.export_txt()

        self.export_excel()
