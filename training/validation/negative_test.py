"""
==========================================================
Validation Negative Tests
OJT AI Project

Purpose:
Test whether validation rules can detect intentionally
incorrect dataset values.

The original dataset is NEVER modified.
==========================================================
"""

import os
import sys
import pandas as pd


# ==========================================================
# PROJECT ROOT
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# CONFIG
# ==========================================================

from config.paths import RAW_DATASET_XLSX


# ==========================================================
# VALIDATION RULES
# ==========================================================

from validation_rules import (
    check_remaining_to_ojt,
    check_risk_score,
    check_risk_level,
    check_delay_risk,
    check_ojt_eligible,
    check_readiness,
    check_ai_recommendation
)


# ==========================================================
# TEST RESULT
# ==========================================================

def print_test_result(
    test_id,
    description,
    result
):

    # Negative test PASS means:
    # Validation correctly detected the error.

    if result["status"] is False:

        print(
            f"[PASS] {test_id} -> "
            f"{description}"
        )

        return True

    print(
        f"[FAIL] {test_id} -> "
        f"{description}"
    )

    print(
        f"       Validator incorrectly returned: "
        f"{result['message']}"
    )

    return False


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)
    print("VALIDATION NEGATIVE TESTING")
    print("=" * 60)

    print()
    print("Dataset:")
    print(RAW_DATASET_XLSX)

    # ======================================================
    # Load original dataset
    # ======================================================

    df = pd.read_excel(
        RAW_DATASET_XLSX
    )

    print()
    print(
        f"Dataset Loaded: {len(df)} rows"
    )

    print()
    print("=" * 60)
    print("Creating Negative Test Cases...")
    print("=" * 60)

    results = []

    # ======================================================
    # TEST 01
    # Remaining To OJT
    # ======================================================

    test_df = df.copy()

    original_value = test_df.loc[
        0,
        "Remaining_To_OJT"
    ]

    test_df.loc[
        0,
        "Remaining_To_OJT"
    ] = 999

    result = check_remaining_to_ojt(
        test_df
    )

    results.append(
        print_test_result(
            "TC01",
            "Incorrect Remaining_To_OJT = 999",
            result
        )
    )

    # ======================================================
    # TEST 02
    # Risk Score
    # ======================================================

    test_df = df.copy()

    test_df.loc[
        0,
        "Risk_Score"
    ] = 999

    result = check_risk_score(
        test_df
    )

    results.append(
        print_test_result(
            "TC02",
            "Incorrect Risk_Score = 999",
            result
        )
    )

    # ======================================================
    # TEST 03
    # Risk Level
    # ======================================================

    test_df = df.copy()

    test_df.loc[
        0,
        "Risk_Score"
    ] = 0

    test_df.loc[
        0,
        "Risk_Level"
    ] = "High"

    result = check_risk_level(
        test_df
    )

    results.append(
        print_test_result(
            "TC03",
            "Risk_Score = 0 but Risk_Level = High",
            result
        )
    )

    # ======================================================
    # TEST 04
    # Delay Risk
    # ======================================================

    test_df = df.copy()

    test_df.loc[
        0,
        "Risk_Score"
    ] = 0

    test_df.loc[
        0,
        "OJT_Delay_Risk"
    ] = 1

    result = check_delay_risk(
        test_df
    )

    results.append(
        print_test_result(
            "TC04",
            "Risk_Score = 0 but OJT_Delay_Risk = 1",
            result
        )
    )

    # ======================================================
    # TEST 05
    # OJT Eligible
    # ======================================================

    test_df = df.copy()

    test_df.loc[
        0,
        "Credits_Completed"
    ] = 10

    test_df.loc[
        0,
        "Missing_Prerequisite_Courses"
    ] = 0

    test_df.loc[
        0,
        "OJT_Eligible"
    ] = True

    result = check_ojt_eligible(
        test_df
    )

    results.append(
        print_test_result(
            "TC05",
            "Credits_Completed = 10 but OJT_Eligible = True",
            result
        )
    )

    # ======================================================
    # TEST 06
    # Readiness
    # ======================================================

    test_df = df.copy()

    # Force an obviously incorrect readiness value.
    # The validator should recalculate the expected
    # business-rule value and detect the mismatch.

    test_df.loc[
        0,
        "OJT_Readiness"
    ] = 99

    result = check_readiness(
        test_df
    )

    results.append(
        print_test_result(
            "TC06",
            "Incorrect OJT_Readiness = 99",
            result
        )
    )

    # ======================================================
    # TEST 07
    # AI Recommendation
    # ======================================================

    test_df = df.copy()

    test_df.loc[
        0,
        "AI_Recommendation"
    ] = ""

    result = check_ai_recommendation(
        test_df
    )

    results.append(
        print_test_result(
            "TC07",
            "Empty AI_Recommendation",
            result
        )
    )

    # ======================================================
    # SUMMARY
    # ======================================================

    passed = sum(results)

    failed = len(results) - passed

    print()
    print("=" * 60)
    print("NEGATIVE TEST SUMMARY")
    print("=" * 60)

    print(
        f"Passed : {passed}"
    )

    print(
        f"Failed : {failed}"
    )

    print(
        f"Total  : {len(results)}"
    )

    print("=" * 60)

    # ======================================================
    # FINAL STATUS
    # ======================================================

    if failed == 0:

        print()
        print(
            "ALL NEGATIVE TESTS PASSED."
        )

        print(
            "Validation rules correctly "
            "detected all injected errors."
        )

    else:

        print()
        print(
            "NEGATIVE TESTING FAILED."
        )

        print(
            "One or more validation rules "
            "failed to detect injected errors."
        )

    print()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()