import os
import sys
import json
import traceback

# ------------------------------------------------------------------
# Project Root
# ------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

import numpy as np

from training.prediction.prediction_pipeline import predict_student

# ------------------------------------------------------------------
# Test Configuration
# ------------------------------------------------------------------

TOTAL_TESTS = 7

passed_tests = 0
failed_tests = 0


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def pass_test(test_name, message):
    global passed_tests

    passed_tests += 1

    print(f"[PASS] {test_name}")
    print(f"       {message}")


def fail_test(test_name, message):
    global failed_tests

    failed_tests += 1

    print(f"[FAIL] {test_name}")
    print(f"       {message}")


# ------------------------------------------------------------------
# Test Student Data
# ------------------------------------------------------------------

GOOD_STUDENT = {
    "Student_Profile": "Excellent",
    "Current_Semester": 5,
    "GPA_Cumulative": 8.8,
    "Total_Credits": 120,
    "Credits_Completed": 96,
    "Credits_Remaining": 24,
    "Completion_Rate": 96 / 120,
    "Average_Credits_Per_Semester": 19.2,
    "Remaining_To_OJT": 4,
    "Failed_Courses": 0,
    "Retake_Count": 0,
    "Missing_Prerequisite_Courses": 0,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


AVERAGE_STUDENT = {
    "Student_Profile": "Average",
    "Current_Semester": 5,
    "GPA_Cumulative": 6.8,
    "Total_Credits": 120,
    "Credits_Completed": 82,
    "Credits_Remaining": 38,
    "Completion_Rate": 82 / 120,
    "Average_Credits_Per_Semester": 16.4,
    "Remaining_To_OJT": 18,
    "Failed_Courses": 2,
    "Retake_Count": 1,
    "Missing_Prerequisite_Courses": 1,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


HIGH_RISK_STUDENT = {
    "Student_Profile": "Critical",
    "Current_Semester": 5,
    "GPA_Cumulative": 4.8,
    "Total_Credits": 120,
    "Credits_Completed": 65,
    "Credits_Remaining": 55,
    "Completion_Rate": 65 / 120,
    "Average_Credits_Per_Semester": 13.0,
    "Remaining_To_OJT": 35,
    "Failed_Courses": 6,
    "Retake_Count": 4,
    "Missing_Prerequisite_Courses": 3,
    "Academic_Warning_Count": 2,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


CREDIT_GAP_STUDENT = {
    "Student_Profile": "LateStarter",
    "Current_Semester": 5,
    "GPA_Cumulative": 6.2,
    "Total_Credits": 120,
    "Credits_Completed": 60,
    "Credits_Remaining": 60,
    "Completion_Rate": 60 / 120,
    "Average_Credits_Per_Semester": 12.0,
    "Remaining_To_OJT": 40,
    "Failed_Courses": 3,
    "Retake_Count": 2,
    "Missing_Prerequisite_Courses": 2,
    "Academic_Warning_Count": 1,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


NEAR_READY_STUDENT = {
    "Student_Profile": "Good",
    "Current_Semester": 6,
    "GPA_Cumulative": 7.8,
    "Total_Credits": 120,
    "Credits_Completed": 100,
    "Credits_Remaining": 20,
    "Completion_Rate": 100 / 120,
    "Average_Credits_Per_Semester": 16.67,
    "Remaining_To_OJT": 0,
    "Failed_Courses": 0,
    "Retake_Count": 0,
    "Missing_Prerequisite_Courses": 0,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


INVALID_GPA_STUDENT = {
    "Student_Profile": "Average",
    "Current_Semester": 5,
    "GPA_Cumulative": 15.0,
    "Total_Credits": 120,
    "Credits_Completed": 82,
    "Credits_Remaining": 38,
    "Completion_Rate": 82 / 120,
    "Average_Credits_Per_Semester": 16.4,
    "Remaining_To_OJT": 18,
    "Failed_Courses": 2,
    "Retake_Count": 1,
    "Missing_Prerequisite_Courses": 1,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


INVALID_PROFILE_STUDENT = {
    "Student_Profile": "UnknownProfile",
    "Current_Semester": 5,
    "GPA_Cumulative": 6.8,
    "Total_Credits": 120,
    "Credits_Completed": 82,
    "Credits_Remaining": 38,
    "Completion_Rate": 82 / 120,
    "Average_Credits_Per_Semester": 16.4,
    "Remaining_To_OJT": 18,
    "Failed_Courses": 2,
    "Retake_Count": 1,
    "Missing_Prerequisite_Courses": 1,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


# ------------------------------------------------------------------
# Prediction Output Validation
# ------------------------------------------------------------------


def validate_prediction_result(result):
    """
    Validate the structure and values returned by predict_student().
    """

    if not isinstance(result, dict):
        raise ValueError("Prediction result must be a dictionary.")

    required_keys = [
        "prediction_class",
        "prediction_label",
        "delay_probability",
        "no_delay_probability",
    ]

    for key in required_keys:

        if key not in result:
            raise ValueError(f"Missing prediction result field: {key}")

    prediction_class = result["prediction_class"]
    delay_probability = result["delay_probability"]
    no_delay_probability = result["no_delay_probability"]

    # --------------------------------------------------------------
    # Prediction Class
    # --------------------------------------------------------------

    if prediction_class not in [0, 1]:
        raise ValueError(
            f"Prediction class must be 0 or 1, " f"received {prediction_class}"
        )

    # --------------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------------

    if not 0 <= delay_probability <= 1:
        raise ValueError(f"Invalid delay probability: {delay_probability}")

    if not 0 <= no_delay_probability <= 1:
        raise ValueError(f"Invalid no-delay probability: {no_delay_probability}")

    # --------------------------------------------------------------
    # Probability Sum
    # --------------------------------------------------------------

    if not np.isclose(delay_probability + no_delay_probability, 1.0, atol=1e-6):
        raise ValueError("Prediction probabilities do not sum to 1.")

    return True


# ------------------------------------------------------------------
# Test 1
# ------------------------------------------------------------------


def test_good_student():

    test_name = "Test 1 - Good Student"

    try:

        result = predict_student(GOOD_STUDENT)

        validate_prediction_result(result)

        pass_test(
            test_name,
            (
                f"Prediction={result['prediction_label']}, "
                f"Delay Probability="
                f"{result['delay_probability']:.4f}"
            ),
        )

    except Exception as e:

        fail_test(test_name, str(e))


# ------------------------------------------------------------------
# Test 2
# ------------------------------------------------------------------


def test_average_student():

    test_name = "Test 2 - Average Student"

    try:

        result = predict_student(AVERAGE_STUDENT)

        validate_prediction_result(result)

        pass_test(
            test_name,
            (
                f"Prediction={result['prediction_label']}, "
                f"Delay Probability="
                f"{result['delay_probability']:.4f}"
            ),
        )

    except Exception as e:

        fail_test(test_name, str(e))


# ------------------------------------------------------------------
# Test 3
# ------------------------------------------------------------------


def test_high_risk_student():

    test_name = "Test 3 - High Risk Student"

    try:

        result = predict_student(HIGH_RISK_STUDENT)

        validate_prediction_result(result)

        pass_test(
            test_name,
            (
                f"Prediction={result['prediction_label']}, "
                f"Delay Probability="
                f"{result['delay_probability']:.4f}"
            ),
        )

    except Exception as e:

        fail_test(test_name, str(e))


# ------------------------------------------------------------------
# Test 4
# ------------------------------------------------------------------


def test_credit_gap_student():

    test_name = "Test 4 - Large Credit Gap"

    try:

        result = predict_student(CREDIT_GAP_STUDENT)

        validate_prediction_result(result)

        pass_test(
            test_name,
            (
                f"Prediction={result['prediction_label']}, "
                f"Delay Probability="
                f"{result['delay_probability']:.4f}"
            ),
        )

    except Exception as e:

        fail_test(test_name, str(e))


# ------------------------------------------------------------------
# Test 5
# ------------------------------------------------------------------


def test_near_ready_student():

    test_name = "Test 5 - OJT Ready Student"

    try:

        result = predict_student(NEAR_READY_STUDENT)

        validate_prediction_result(result)

        pass_test(
            test_name,
            (
                f"Prediction={result['prediction_label']}, "
                f"Delay Probability="
                f"{result['delay_probability']:.4f}"
            ),
        )

    except Exception as e:

        fail_test(test_name, str(e))


# ------------------------------------------------------------------
# Test 6
# ------------------------------------------------------------------


def test_invalid_gpa():

    test_name = "Test 6 - Invalid GPA"

    try:

        predict_student(INVALID_GPA_STUDENT)

        fail_test(
            test_name, "Invalid GPA was accepted. Expected input validation error."
        )

    except Exception:

        pass_test(test_name, "Invalid GPA was correctly rejected.")


# ------------------------------------------------------------------
# Test 7
# ------------------------------------------------------------------


def test_invalid_profile():

    test_name = "Test 7 - Invalid Student Profile"

    try:

        predict_student(INVALID_PROFILE_STUDENT)

        fail_test(
            test_name,
            (
                "Invalid Student_Profile was accepted. "
                "Expected input validation error."
            ),
        )

    except Exception:

        pass_test(test_name, "Invalid Student_Profile was correctly rejected.")


# ------------------------------------------------------------------
# Main Test Runner
# ------------------------------------------------------------------


def main():

    print()
    print("=" * 70)
    print("OJT AI PROJECT - PREDICTION TEST V2")
    print("=" * 70)

    print()
    print("Purpose:")
    print("- Test multiple valid student scenarios")
    print("- Test invalid student inputs")
    print("- Validate prediction class")
    print("- Validate prediction probabilities")
    print("- Confirm Prediction Pipeline stability")

    # --------------------------------------------------------------
    # Run Tests
    # --------------------------------------------------------------

    print_header("Running Prediction Tests...")

    test_good_student()

    test_average_student()

    test_high_risk_student()

    test_credit_gap_student()

    test_near_ready_student()

    test_invalid_gpa()

    test_invalid_profile()

    # --------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTION TEST SUMMARY")
    print("=" * 70)

    print(f"Passed : {passed_tests}")
    print(f"Failed : {failed_tests}")
    print(f"Total  : {passed_tests + failed_tests}")

    print()

    if failed_tests == 0:

        print("PREDICTION TEST: PASS")
        print()
        print("Prediction Pipeline is stable and ready " "for Recommendation Engine.")

    else:

        print("PREDICTION TEST: FAIL")
        print()
        print("Prediction Pipeline requires further investigation.")


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
