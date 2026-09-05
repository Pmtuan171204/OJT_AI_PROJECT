"""
==========================================================
Dataset Validator
OJT AI Project

Validate dataset before Machine Learning training.
==========================================================
"""

import os
import sys
import pandas as pd

# ======================================================
# Add Project Root into Python Path
# ======================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ======================================================
# Import Project Config
# ======================================================

from config.paths import RAW_DATASET_XLSX

# ======================================================
# Import Validation Rules
# ======================================================

from validation_rules import *

# ======================================================
# Import Validation Report
# ======================================================

from validation_report import ValidationReport

# ======================================================
# Dataset Validator
# ======================================================


class DatasetValidator:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.df = None

        self.results = []

    # ==================================================
    # Load Dataset
    # ==================================================

    def load_dataset(self):

        self.df = pd.read_excel(self.dataset_path)

    # ==================================================
    # Run All Validation Rules
    # ==================================================

    def validate(self):

        print("=" * 60)
        print("Running Dataset Validation...")
        print("=" * 60)

        # -----------------------------
        # File Exists
        # -----------------------------

        self.results.append(check_file_exists(self.dataset_path))

        if not self.results[-1]["status"]:
            return

        # -----------------------------
        # Load Dataset
        # -----------------------------

        self.load_dataset()

        # -----------------------------
        # Dataset Integrity
        # -----------------------------

        self.results.append(check_missing_values(self.df))

        self.results.append(check_duplicate_student_id(self.df))

        # -----------------------------
        # Academic Validation
        # -----------------------------

        self.results.append(check_gpa_range(self.df))

        self.results.append(check_completed_credits(self.df))

        self.results.append(check_remaining_credits(self.df))

        self.results.append(check_completion_rate(self.df))

        self.results.append(check_remaining_to_ojt(self.df))

        # -----------------------------
        # AI Validation
        # -----------------------------

        self.results.append(check_risk_score(self.df))

        self.results.append(check_risk_level(self.df))

        self.results.append(check_delay_risk(self.df))

        self.results.append(check_ojt_eligible(self.df))

        self.results.append(check_readiness(self.df))

        self.results.append(check_ai_recommendation(self.df))

        self.results.append(check_student_profile(self.df))

    # ==================================================
    # Print Validation Result
    # ==================================================

    def print_result(self):

        print()

        print("=" * 60)
        print("Validation Result")
        print("=" * 60)

        passed = 0
        failed = 0

        for result in self.results:

            status = "PASS" if result["status"] else "FAIL"

            print(f"[{status}] " f"{result['rule']} " f"-> {result['message']}")

            if result["status"]:
                passed += 1
            else:
                failed += 1

        print()

        print("=" * 60)
        print(f"Passed : {passed}")
        print(f"Failed : {failed}")
        print(f"Total  : {len(self.results)}")
        print("=" * 60)

    # ==================================================
    # Return Result
    # ==================================================

    def get_results(self):

        return self.results


# ======================================================
# Main
# ======================================================


def main():

    print()

    print("Dataset Path:")
    print(RAW_DATASET_XLSX)
    print()

    validator = DatasetValidator(RAW_DATASET_XLSX)

    validator.validate()

    validator.print_result()

    # ==================================================
    # Generate Validation Report
    # ==================================================

    report = ValidationReport(validator.get_results())

    report.generate()

    # ==================================================
    # Pipeline Quality Gate
    # ==================================================

    results = validator.get_results()

    failed = [result for result in results if not result["status"]]

    print()

    if len(failed) > 0:

        print("=" * 60)
        print("VALIDATION FAILED")
        print("=" * 60)

        print(f"{len(failed)} validation rule(s) failed.")

        print("Pipeline execution stopped.")

        print("=" * 60)

        return

    print("=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)

    print("Dataset is ready for preprocessing and training.")

    print("=" * 60)


if __name__ == "__main__":
    main()
