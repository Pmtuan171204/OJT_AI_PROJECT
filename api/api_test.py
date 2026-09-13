import json
import urllib.request
import urllib.error

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_URL = "http://127.0.0.1:8000"


# ==========================================================
# TEST DATA
# ==========================================================

AVERAGE_STUDENT = {
    "Student_Profile": "Average",
    "Current_Semester": 5,
    "GPA_Cumulative": 6.8,
    "Total_Credits": 120,
    "Credits_Completed": 82,
    "Credits_Remaining": 38,
    "Completion_Rate": 0.6833,
    "Average_Credits_Per_Semester": 16.4,
    "Remaining_To_OJT": 18,
    "Failed_Courses": 2,
    "Retake_Count": 1,
    "Missing_Prerequisite_Courses": 1,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 6,
}


GOOD_STUDENT = {
    "Student_Profile": "Excellent",
    "Current_Semester": 5,
    "GPA_Cumulative": 9.0,
    "Total_Credits": 120,
    "Credits_Completed": 100,
    "Credits_Remaining": 20,
    "Completion_Rate": 0.8333,
    "Average_Credits_Per_Semester": 20.0,
    "Remaining_To_OJT": 0,
    "Failed_Courses": 0,
    "Retake_Count": 0,
    "Missing_Prerequisite_Courses": 0,
    "Academic_Warning_Count": 0,
    "Suspension_Count": 0,
    "Planned_OJT_Semester": 7,
}


HIGH_RISK_STUDENT = {
    "Student_Profile": "Critical",
    "Current_Semester": 5,
    "GPA_Cumulative": 4.8,
    "Total_Credits": 120,
    "Credits_Completed": 65,
    "Credits_Remaining": 55,
    "Completion_Rate": 0.5417,
    "Average_Credits_Per_Semester": 13.0,
    "Remaining_To_OJT": 35,
    "Failed_Courses": 5,
    "Retake_Count": 3,
    "Missing_Prerequisite_Courses": 2,
    "Academic_Warning_Count": 2,
    "Suspension_Count": 1,
    "Planned_OJT_Semester": 6,
}


# ==========================================================
# TEST COUNTERS
# ==========================================================

passed = 0
failed = 0


# ==========================================================
# HTTP HELPERS
# ==========================================================


def get_request(endpoint):
    """
    Send GET request and return status code + response.
    """

    url = BASE_URL + endpoint

    try:

        with urllib.request.urlopen(url) as response:

            status_code = response.status

            body = response.read().decode("utf-8")

            data = json.loads(body)

            return status_code, data

    except urllib.error.HTTPError as error:

        body = error.read().decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}

        return error.code, data


def post_request(endpoint, payload):
    """
    Send POST request and return status code + response.
    """

    url = BASE_URL + endpoint

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:

        with urllib.request.urlopen(request) as response:

            status_code = response.status

            body = response.read().decode("utf-8")

            data = json.loads(body)

            return status_code, data

    except urllib.error.HTTPError as error:

        body = error.read().decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}

        return error.code, data


# ==========================================================
# TEST RUNNER
# ==========================================================


def run_test(test_name, test_function):

    global passed, failed

    try:

        test_function()

        print(f"[PASS] {test_name}")

        passed += 1

    except AssertionError as error:

        print(f"[FAIL] {test_name}")

        if str(error):
            print(f"       {error}")

        failed += 1

    except Exception as error:

        print(f"[FAIL] {test_name}")
        print(f"       Unexpected error: {error}")

        failed += 1


# ==========================================================
# TEST 1 - ROOT ENDPOINT
# ==========================================================


def test_root():

    status, data = get_request("/")

    assert status == 200, f"Expected HTTP 200, got {status}"

    assert data["project"] == "OJT AI Project"

    assert data["status"] == "running"


# ==========================================================
# TEST 2 - HEALTH CHECK
# ==========================================================


def test_health():

    status, data = get_request("/health")

    assert status == 200, f"Expected HTTP 200, got {status}"

    assert data["status"] == "healthy"


# ==========================================================
# TEST 3 - VALID PREDICTION
# ==========================================================


def test_valid_prediction():

    status, data = post_request("/predict", AVERAGE_STUDENT)

    assert status == 200, f"Expected HTTP 200, got {status}"

    assert "prediction" in data

    assert "current_metrics" in data

    assert "issues" in data

    assert "recommendations" in data

    assert "roadmap" in data


# ==========================================================
# TEST 4 - GOOD STUDENT
# ==========================================================


def test_good_student():

    status, data = post_request("/predict", GOOD_STUDENT)

    assert status == 200, f"Expected HTTP 200, got {status}"

    prediction = data["prediction"]

    assert "prediction_class" in prediction

    assert "delay_probability" in prediction

    assert "risk_level" in prediction

    probability = prediction["delay_probability"]

    assert 0 <= probability <= 1


# ==========================================================
# TEST 5 - HIGH RISK STUDENT
# ==========================================================


def test_high_risk_student():

    status, data = post_request("/predict", HIGH_RISK_STUDENT)

    assert status == 200, f"Expected HTTP 200, got {status}"

    prediction = data["prediction"]

    assert prediction["prediction_class"] in [0, 1]

    probability = prediction["delay_probability"]

    assert 0 <= probability <= 1

    assert len(data["issues"]) > 0

    assert len(data["recommendations"]) > 0

    assert len(data["roadmap"]) > 0


# ==========================================================
# TEST 6 - INVALID GPA
# ==========================================================


def test_invalid_gpa():

    student = AVERAGE_STUDENT.copy()

    student["GPA_Cumulative"] = 15

    status, data = post_request("/predict", student)

    assert status == 422, f"Expected HTTP 422 for invalid GPA, got {status}"


# ==========================================================
# TEST 7 - INVALID STUDENT PROFILE
# ==========================================================


def test_invalid_student_profile():

    student = AVERAGE_STUDENT.copy()

    student["Student_Profile"] = "UnknownProfile"

    status, data = post_request("/predict", student)

    assert status == 400, f"Expected HTTP 400 for invalid profile, got {status}"


# ==========================================================
# TEST 8 - MISSING REQUIRED FIELD
# ==========================================================


def test_missing_required_field():

    student = AVERAGE_STUDENT.copy()

    del student["GPA_Cumulative"]

    status, data = post_request("/predict", student)

    assert status == 422, f"Expected HTTP 422 for missing field, got {status}"


# ==========================================================
# TEST 9 - INVALID OJT SEMESTER
# ==========================================================


def test_invalid_ojt_semester():

    student = AVERAGE_STUDENT.copy()

    student["Planned_OJT_Semester"] = 0

    status, data = post_request("/predict", student)

    assert status == 422, f"Expected HTTP 422 for invalid semester, got {status}"


# ==========================================================
# TEST 10 - RESPONSE STRUCTURE
# ==========================================================


def test_response_structure():

    status, data = post_request("/predict", AVERAGE_STUDENT)

    assert status == 200

    # ------------------------------------------------------
    # Main sections
    # ------------------------------------------------------

    required_sections = [
        "prediction",
        "current_metrics",
        "issues",
        "recommendations",
        "roadmap",
    ]

    for section in required_sections:

        assert section in data, f"Missing response section: {section}"

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = data["prediction"]

    prediction_fields = [
        "prediction_class",
        "prediction_label",
        "delay_probability",
        "risk_level",
    ]

    for field in prediction_fields:

        assert field in prediction, f"Missing prediction field: {field}"

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    metrics = data["current_metrics"]

    metric_fields = [
        "OJT_Credit_Gap",
        "OJT_Credit_Progress",
        "OJT_Semester_Gap",
        "Required_Average_Credits_Per_Semester",
        "Credit_Completion_Efficiency",
        "Completion_Rate",
    ]

    for field in metric_fields:

        assert field in metrics, f"Missing metric field: {field}"

    # ------------------------------------------------------
    # Lists
    # ------------------------------------------------------

    assert isinstance(data["issues"], list)

    assert isinstance(data["recommendations"], list)

    assert isinstance(data["roadmap"], list)


# ==========================================================
# MAIN
# ==========================================================

print()
print("=" * 70)
print("OJT AI PROJECT - API TEST V2")
print("=" * 70)
print()

print(f"API Base URL : {BASE_URL}")
print()


run_test("Test 1 - Root Endpoint", test_root)

run_test("Test 2 - Health Check", test_health)

run_test("Test 3 - Valid Prediction", test_valid_prediction)

run_test("Test 4 - Good Student", test_good_student)

run_test("Test 5 - High Risk Student", test_high_risk_student)

run_test("Test 6 - Invalid GPA", test_invalid_gpa)

run_test("Test 7 - Invalid Student Profile", test_invalid_student_profile)

run_test("Test 8 - Missing Required Field", test_missing_required_field)

run_test("Test 9 - Invalid OJT Semester", test_invalid_ojt_semester)

run_test("Test 10 - Response Structure", test_response_structure)


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

    print("API TEST: PASS")
    print("API Integration is ready for Frontend Integration.")

else:

    print("API TEST: FAIL")
    print("Please fix failed tests before Frontend Integration.")

print("=" * 70)
