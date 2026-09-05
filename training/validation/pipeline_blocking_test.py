"""
==========================================================
Pipeline Blocking Test
OJT AI Project

Test whether the validation pipeline correctly blocks
invalid datasets before preprocessing/training.
==========================================================
"""

import os
import sys
import pandas as pd

# ==========================================================
# Add Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==========================================================
# Import Config
# ==========================================================

from config.paths import RAW_DATASET_XLSX

# ==========================================================
# Import Validator
# ==========================================================

from validator import DatasetValidator

# ==========================================================
# Main Test
# ==========================================================


def main():

    print()
    print("=" * 60)
    print("PIPELINE BLOCKING TEST")
    print("=" * 60)

    print()
    print("Dataset:")
    print(RAW_DATASET_XLSX)

    # ======================================================
    # Load Original Dataset
    # ======================================================

    print()
    print("=" * 60)
    print("Creating Invalid Test Dataset...")
    print("=" * 60)

    df = pd.read_excel(RAW_DATASET_XLSX)

    print(f"Original Dataset Rows : {len(df)}")

    # ======================================================
    # Inject Validation Error
    # ======================================================

    # Create an invalid Remaining_To_OJT value.
    #
    # Correct formula:
    #
    # Remaining_To_OJT =
    # max(100 - Credits_Completed, 0)
    #
    # We intentionally set the first record to 999.

    original_value = df.loc[0, "Remaining_To_OJT"]

    df.loc[0, "Remaining_To_OJT"] = 999

    print()
    print("Injected Error:")

    print("Remaining_To_OJT = 999")

    print(f"Original Value = {original_value}")

    # ======================================================
    # Save Temporary Invalid Dataset
    # ======================================================

    test_dataset_path = os.path.join(CURRENT_DIR, "temp_invalid_dataset.xlsx")

    df.to_excel(test_dataset_path, index=False)

    print()
    print("Temporary invalid dataset created:")

    print(test_dataset_path)

    # ======================================================
    # Run Validator
    # ======================================================

    print()
    print("=" * 60)
    print("Running Validator...")
    print("=" * 60)

    validator = DatasetValidator(test_dataset_path)

    validator.validate()

    validator.print_result()

    # ======================================================
    # Check Validation Result
    # ======================================================

    results = validator.get_results()

    failed = [result for result in results if not result["status"]]

    # ======================================================
    # Quality Gate
    # ======================================================

    print()
    print("=" * 60)
    print("PIPELINE BLOCKING RESULT")
    print("=" * 60)

    if len(failed) > 0:

        print("[PASS] Validation failure detected.")

        print(f"[PASS] Failed Rules : {len(failed)}")

        print("[PASS] Pipeline would be blocked.")

        print("=" * 60)

        print()
        print("PIPELINE BLOCKING TEST PASSED.")

    else:

        print("[FAIL] Invalid dataset was not detected.")

        print("[FAIL] Pipeline would NOT be blocked.")

        print("=" * 60)

        print()
        print("PIPELINE BLOCKING TEST FAILED.")

        # Exit with failure code
        sys.exit(1)

    # ======================================================
    # Cleanup Temporary Dataset
    # ======================================================

    if os.path.exists(test_dataset_path):

        os.remove(test_dataset_path)

        print()
        print("Temporary test dataset removed.")


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
