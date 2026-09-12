import os
import sys

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# IMPORT RECOMMENDATION ENGINE
# ==========================================================

from training.recommendation.recommendation_engine import generate_recommendation

# ==========================================================
# TEST PREDICTION RESULTS
# ==========================================================
# These prediction results simulate the output that would
# normally come from the Prediction Pipeline.

LOW_RISK_PREDICTION = {
    "prediction_class": 0,
    "prediction_label": "No Delay",
    "delay_probability": 0.10,
    "no_delay_probability": 0.90,
}

MEDIUM_RISK_PREDICTION = {
    "prediction_class": 1,
    "prediction_label": "OJT Delay",
    "delay_probability": 0.50,
    "no_delay_probability": 0.50,
}

HIGH_RISK_PREDICTION = {
    "prediction_class": 1,
    "prediction_label": "OJT Delay",
    "delay_probability": 0.759,
    "no_delay_probability": 0.241,
}


# ==========================================================
# TEST STUDENTS
# ==========================================================

GOOD_STUDENT = {
    "Student_Profile": "Excellent",
    "Current_Semester": 5,
    "GPA_Cumulative": 9.0,
    "Total_Credits": 120,
    "Credits_Completed": 100,
    "Credits_Remaining": 20,
    "Failed_Courses": 0,
    "Retake_Count": 0,
    "Missing_Prerequisite_Courses": 0,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 7,
}


AVERAGE_STUDENT = {
    "Student_Profile": "Average",
    "Current_Semester": 5,
    "GPA_Cumulative": 6.8,
    "Total_Credits": 120,
    "Credits_Completed": 82,
    "Credits_Remaining": 38,
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
    "Failed_Courses": 5,
    "Retake_Count": 3,
    "Missing_Prerequisite_Courses": 2,
    "Academic_Warning_Count": 2,
    "Suspension_Count": 1,
    "Planned_OJT_Semester": 6,
}


LARGE_CREDIT_GAP_STUDENT = {
    "Student_Profile": "LateStarter",
    "Current_Semester": 5,
    "GPA_Cumulative": 7.0,
    "Total_Credits": 120,
    "Credits_Completed": 60,
    "Credits_Remaining": 60,
    "Failed_Courses": 1,
    "Retake_Count": 1,
    "Missing_Prerequisite_Courses": 0,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


MISSING_PREREQUISITE_STUDENT = {
    "Student_Profile": "Average",
    "Current_Semester": 5,
    "GPA_Cumulative": 7.2,
    "Total_Credits": 120,
    "Credits_Completed": 85,
    "Credits_Remaining": 35,
    "Failed_Courses": 0,
    "Retake_Count": 0,
    "Missing_Prerequisite_Courses": 2,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


OJT_APPROACHING_STUDENT = {
    "Student_Profile": "Good",
    "Current_Semester": 5,
    "GPA_Cumulative": 8.0,
    "Total_Credits": 120,
    "Credits_Completed": 90,
    "Credits_Remaining": 30,
    "Failed_Courses": 0,
    "Retake_Count": 0,
    "Missing_Prerequisite_Courses": 0,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


# ==========================================================
# TEST HELPER
# ==========================================================

passed = 0
failed = 0


def run_test(test_name, test_function):
    """
    Execute one test and print the result.
    """

    global passed, failed

    try:
        test_function()

        print(f"[PASS] {test_name}")
        passed += 1

    except AssertionError as error:
        print(f"[FAIL] {test_name}")
        print(f"       {error}")
        failed += 1

    except Exception as error:
        print(f"[FAIL] {test_name}")
        print(f"       Unexpected error: {error}")
        failed += 1


# ==========================================================
# TEST 1 - GOOD STUDENT
# ==========================================================
def test_good_student():

    result = generate_recommendation(GOOD_STUDENT, LOW_RISK_PREDICTION)

    assert "prediction" in result, "Missing prediction section."

    assert "current_metrics" in result, "Missing current_metrics section."

    assert "issues" in result, "Missing issues section."

    assert "recommendations" in result, "Missing recommendations section."

    assert "roadmap" in result, "Missing roadmap section."

    # Prediction validation
    assert result["prediction"]["prediction_class"] in [
        0,
        1,
    ], "Prediction class must be 0 or 1."

    assert (
        0 <= result["prediction"]["delay_probability"] <= 1
    ), "Delay probability must be between 0 and 1."

    assert result["prediction"]["risk_level"] in [
        "Low",
        "Medium",
        "High",
        "Critical",
    ], "Risk level must be valid."

    # Good student's current metrics
    assert result["current_metrics"]["OJT_Credit_Gap"] == 0, (
        "Expected OJT_Credit_Gap=0, "
        f"got {result['current_metrics']['OJT_Credit_Gap']}"
    )

    assert isinstance(
        result["recommendations"], list
    ), "Recommendations should be a list."

    assert (
        len(result["recommendations"]) > 0
    ), "Good student should still receive recommendations."


# ==========================================================
# TEST 2 - AVERAGE STUDENT
# ==========================================================


def test_average_student():

    result = generate_recommendation(AVERAGE_STUDENT, HIGH_RISK_PREDICTION)

    assert result["prediction"]["prediction_label"] == "OJT Delay"

    assert result["current_metrics"]["OJT_Credit_Gap"] == 18

    assert result["current_metrics"]["OJT_Semester_Gap"] == 1

    assert len(result["issues"]) > 0

    assert len(result["recommendations"]) > 0

    assert len(result["roadmap"]) > 0


# ==========================================================
# TEST 3 - HIGH RISK STUDENT
# ==========================================================


def test_high_risk_student():

    result = generate_recommendation(HIGH_RISK_STUDENT, HIGH_RISK_PREDICTION)

    assert result["current_metrics"]["OJT_Credit_Gap"] == 35

    assert len(result["issues"]) >= 5

    assert len(result["recommendations"]) >= 5

    assert len(result["roadmap"]) > 0


# ==========================================================
# TEST 4 - LARGE CREDIT GAP
# ==========================================================


def test_large_credit_gap():

    result = generate_recommendation(LARGE_CREDIT_GAP_STUDENT, MEDIUM_RISK_PREDICTION)

    gap = result["current_metrics"]["OJT_Credit_Gap"]

    assert gap == 40

    assert gap > 0

    assert len(result["issues"]) > 0

    assert len(result["recommendations"]) > 0

    # Convert recommendation objects to text
    issue_text = str(result["issues"]).lower()
    recommendation_text = str(result["recommendations"]).lower()

    assert (
        "credit" in issue_text
    ), "Recommendation Engine should identify the OJT credit gap."

    assert (
        "credit" in recommendation_text
    ), "Recommendation Engine should provide credit-related advice."


# ==========================================================
# TEST 5 - MISSING PREREQUISITE
# ==========================================================


def test_missing_prerequisite():

    result = generate_recommendation(
        MISSING_PREREQUISITE_STUDENT, MEDIUM_RISK_PREDICTION
    )

    assert len(result["issues"]) > 0

    assert len(result["recommendations"]) > 0

    # Convert recommendation objects to text
    issue_text = str(result["issues"]).lower()
    recommendation_text = str(result["recommendations"]).lower()

    assert "prerequisite" in issue_text, "Missing prerequisite should be identified."

    assert (
        "prerequisite" in recommendation_text
    ), "Prerequisite recommendation should be generated."


# ==========================================================
# TEST 6 - OJT APPROACHING
# ==========================================================


def test_ojt_approaching():

    result = generate_recommendation(OJT_APPROACHING_STUDENT, MEDIUM_RISK_PREDICTION)

    semester_gap = result["current_metrics"]["OJT_Semester_Gap"]

    assert semester_gap == 1

    approaching_issue_found = any(
        "approaching" in issue["title"].lower() for issue in result["issues"]
    )

    assert approaching_issue_found, "OJT approaching issue should be identified."


# ==========================================================
# TEST 7 - INVALID GPA
# ==========================================================


def test_invalid_gpa():

    invalid_student = GOOD_STUDENT.copy()

    invalid_student["GPA_Cumulative"] = 11

    try:

        generate_recommendation(invalid_student, LOW_RISK_PREDICTION)

    except ValueError:

        return

    raise AssertionError("Invalid GPA should be rejected.")


# ==========================================================
# TEST 8 - INVALID STUDENT PROFILE
# ==========================================================


def test_invalid_student_profile():

    invalid_student = GOOD_STUDENT.copy()

    invalid_student["Student_Profile"] = "UnknownProfile"

    try:

        generate_recommendation(invalid_student, LOW_RISK_PREDICTION)

    except ValueError:

        return

    raise AssertionError("Invalid Student_Profile should be rejected.")


# ==========================================================
# TEST 9 - INVALID NEGATIVE CREDITS
# ==========================================================


def test_negative_credits():

    invalid_student = GOOD_STUDENT.copy()

    invalid_student["Credits_Completed"] = -10

    try:

        generate_recommendation(invalid_student, LOW_RISK_PREDICTION)

    except ValueError:

        return

    raise AssertionError("Negative Credits_Completed should be rejected.")


# ==========================================================
# TEST 10 - RECOMMENDATION STRUCTURE
# ==========================================================


def test_recommendation_structure():

    result = generate_recommendation(AVERAGE_STUDENT, HIGH_RISK_PREDICTION)

    # Main sections

    required_sections = [
        "prediction",
        "current_metrics",
        "issues",
        "recommendations",
        "roadmap",
    ]

    for section in required_sections:

        assert section in result, f"Missing section: {section}"

    # Prediction structure

    prediction = result["prediction"]

    required_prediction_fields = [
        "prediction_class",
        "prediction_label",
        "delay_probability",
        "risk_level",
    ]

    for field in required_prediction_fields:

        assert field in prediction, f"Missing prediction field: {field}"

    # Probability validation

    probability = prediction["delay_probability"]

    assert 0 <= probability <= 1

    # Issues

    assert isinstance(result["issues"], list)

    # Recommendations

    assert isinstance(result["recommendations"], list)

    # Roadmap

    assert isinstance(result["roadmap"], list)


# ==========================================================
# RUN TESTS
# ==========================================================

print()
print("=" * 70)
print("OJT AI PROJECT - RECOMMENDATION ENGINE TEST V2")
print("=" * 70)
print()


run_test("Test 1 - Good Student", test_good_student)

run_test("Test 2 - Average Student", test_average_student)

run_test("Test 3 - High Risk Student", test_high_risk_student)

run_test("Test 4 - Large Credit Gap", test_large_credit_gap)

run_test("Test 5 - Missing Prerequisite", test_missing_prerequisite)

run_test("Test 6 - OJT Approaching", test_ojt_approaching)

run_test("Test 7 - Invalid GPA", test_invalid_gpa)

run_test("Test 8 - Invalid Student Profile", test_invalid_student_profile)

run_test("Test 9 - Negative Credits", test_negative_credits)

run_test("Test 10 - Recommendation Structure", test_recommendation_structure)


# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print(f"Passed : {passed}")
print(f"Failed : {failed}")
print(f"Total  : {passed + failed}")

print()

if failed == 0:

    print("RECOMMENDATION TEST: PASS")
    print("Recommendation Engine is ready for API Integration.")

else:

    print("RECOMMENDATION TEST: FAIL")
    print("Please fix failed tests before API Integration.")

print("=" * 70)
