import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from data.generator.profiles import PROFILES

from data.generator.business_rules import (
    calculate_risk_score,
    get_risk_level,
    check_ojt_eligibility,
    calculate_ojt_readiness,
    generate_ai_recommendation,
)

# ==========================================================
# Helper Function
# ==========================================================


def create_result(rule, status, message):
    """
    Standard format for every validation result.
    """

    return {"rule": rule, "status": status, "message": message}


# ==========================================================
# Rule 1
# File Exists
# ==========================================================


def check_file_exists(file_path):

    if os.path.exists(file_path):

        return create_result("Dataset File", True, "Dataset file found.")

    return create_result("Dataset File", False, "Dataset file does not exist.")


# ==========================================================
# Rule 2
# Missing Values
# ==========================================================


def check_missing_values(df):

    missing = df.isnull().sum().sum()

    if missing == 0:

        return create_result("Missing Values", True, "No missing values found.")

    return create_result("Missing Values", False, f"Found {missing} missing values.")


# ==========================================================
# Rule 3
# Duplicate MSSV
# ==========================================================


def check_duplicate_student_id(df):

    duplicate = df["MSSV"].duplicated().sum()

    if duplicate == 0:

        return create_result("Duplicate Student ID", True, "No duplicate student IDs.")

    return create_result(
        "Duplicate Student ID", False, f"Found {duplicate} duplicated student IDs."
    )


# ==========================================================
# Rule 4
# GPA Validation
# ==========================================================


def check_gpa_range(df):

    invalid = df[(df["GPA_Cumulative"] < 0) | (df["GPA_Cumulative"] > 10)]

    if len(invalid) == 0:

        return create_result("GPA Range", True, "All GPA values are valid.")

    return create_result("GPA Range", False, f"{len(invalid)} invalid GPA values.")


# ==========================================================
# Rule 5
# Credits Completed
# ==========================================================


def check_completed_credits(df):

    invalid = df[
        (df["Credits_Completed"] < 0) | (df["Credits_Completed"] > df["Total_Credits"])
    ]

    if len(invalid) == 0:

        return create_result("Completed Credits", True, "Completed credits are valid.")

    return create_result(
        "Completed Credits", False, f"{len(invalid)} invalid completed credits."
    )


# ==========================================================
# Rule 6
# Remaining Credits
# ==========================================================


def check_remaining_credits(df):

    expected = df["Total_Credits"] - df["Credits_Completed"]

    invalid = df[df["Credits_Remaining"] != expected]

    if len(invalid) == 0:

        return create_result(
            "Remaining Credits", True, "Remaining credits are correct."
        )

    return create_result(
        "Remaining Credits", False, f"{len(invalid)} invalid remaining credits."
    )


# ==========================================================
# Rule 7
# Completion Rate
# ==========================================================


def check_completion_rate(df):

    expected = ((df["Credits_Completed"] / df["Total_Credits"]) * 100).round(2)

    invalid = df[abs(df["Completion_Rate"] - expected) > 0.01]

    if len(invalid) == 0:

        return create_result("Completion Rate", True, "Completion rate is correct.")

    return create_result(
        "Completion Rate", False, f"{len(invalid)} invalid completion rates."
    )


# ==========================================================
# Rule 8
# Remaining To OJT
# ==========================================================


def check_remaining_to_ojt(df):

    expected = (100 - df["Credits_Completed"]).clip(lower=0)

    invalid = df[df["Remaining_To_OJT"] != expected]

    if len(invalid) == 0:

        return create_result(
            "Remaining To OJT", True, "Remaining credits to OJT are correct."
        )

    return create_result(
        "Remaining To OJT", False, f"{len(invalid)} invalid Remaining_To_OJT values."
    )


# ==========================================================
# Rule 9
# Risk Score
# ==========================================================


def check_risk_score(df):

    invalid_count = 0

    for _, student in df.iterrows():

        expected = calculate_risk_score(student)

        actual = student["Risk_Score"]

        if int(actual) != int(expected):

            invalid_count += 1

    if invalid_count == 0:

        return create_result("Risk Score", True, "Risk scores are valid.")

    return create_result("Risk Score", False, f"{invalid_count} invalid risk scores.")


# ==========================================================
# Rule 10
# Risk Level
# ==========================================================


def check_risk_level(df):

    invalid_count = 0

    for _, student in df.iterrows():

        expected = get_risk_level(student["Risk_Score"])

        actual = student["Risk_Level"]

        if str(actual) != str(expected):

            invalid_count += 1

    if invalid_count == 0:

        return create_result("Risk Level", True, "Risk levels are valid.")

    return create_result("Risk Level", False, f"{invalid_count} invalid risk levels.")


# ==========================================================
# Rule 11
# Delay Risk Label
# ==========================================================


def check_delay_risk(df):

    invalid_count = 0

    for _, student in df.iterrows():

        expected = 0 if student["Risk_Score"] <= 10 else 1

        actual = int(student["OJT_Delay_Risk"])

        if actual != expected:

            invalid_count += 1

    if invalid_count == 0:

        return create_result("Delay Risk", True, "Delay risk labels are valid.")

    return create_result(
        "Delay Risk", False, f"{invalid_count} invalid delay risk labels."
    )


# ==========================================================
# Rule 12
# OJT Eligible
# ==========================================================


def check_ojt_eligible(df):

    invalid_count = 0

    for _, student in df.iterrows():

        expected = check_ojt_eligibility(student)

        actual = student["OJT_Eligible"]

        if bool(actual) != bool(expected):

            invalid_count += 1

    if invalid_count == 0:

        return create_result("OJT Eligible", True, "OJT eligibility values are valid.")

    return create_result(
        "OJT Eligible", False, f"{invalid_count} invalid OJT eligibility values."
    )


# ==========================================================
# Rule 13
# Readiness
# ==========================================================


def check_readiness(df):

    invalid_count = 0

    for _, student in df.iterrows():

        expected = calculate_ojt_readiness(student)

        actual = student["OJT_Readiness"]

        if abs(float(actual) - float(expected)) > 0.01:

            invalid_count += 1

    if invalid_count == 0:

        return create_result("Readiness", True, "Readiness values are valid.")

    return create_result(
        "Readiness", False, f"{invalid_count} invalid readiness values."
    )


# ==========================================================
# Rule 14
# AI Recommendation
# ==========================================================


def check_ai_recommendation(df):

    invalid_count = 0

    for _, student in df.iterrows():

        expected = generate_ai_recommendation(student)

        actual = student["AI_Recommendation"]

        if str(actual).strip() != str(expected).strip():

            invalid_count += 1

    if invalid_count == 0:

        return create_result("AI Recommendation", True, "AI recommendations are valid.")

    return create_result(
        "AI Recommendation", False, f"{invalid_count} invalid AI recommendations."
    )


# ==========================================================
# Rule 15
# Student Profile
# ==========================================================


def check_student_profile(df):
    """
    Validate student profile values.
    Profiles are loaded directly from profiles.py
    """

    valid_profiles = list(PROFILES.keys())

    invalid = df[~df["Student_Profile"].isin(valid_profiles)]

    if len(invalid) == 0:

        return create_result("Student Profile", True, "Student profiles are valid.")

    return create_result(
        "Student Profile", False, f"{len(invalid)} invalid student profiles."
    )


# ==========================================================
# Rule 16
# Future Credits At OJT
# ==========================================================


def check_future_credits(df):

    invalid = df[
        (df["Future_Credits_At_OJT"] < 0)
        | (df["Future_Credits_At_OJT"] > df["Total_Credits"])
    ]

    if len(invalid) == 0:

        return create_result("Future Credits", True, "Future credits at OJT are valid.")

    return create_result(
        "Future Credits", False, f"{len(invalid)} invalid future credit values."
    )


# ==========================================================
# Rule 17
# Future Failed Courses
# ==========================================================


def check_future_failed_courses(df):

    invalid = df[df["Future_Failed_Courses"] < df["Failed_Courses"]]

    if len(invalid) == 0:

        return create_result(
            "Future Failed Courses", True, "Future failed course values are valid."
        )

    return create_result(
        "Future Failed Courses",
        False,
        f"{len(invalid)} invalid future failed course values.",
    )


# ==========================================================
# Rule 18
# Future Missing Prerequisites
# ==========================================================


def check_future_missing_prerequisites(df):

    invalid = df[
        (df["Future_Missing_Prerequisites"] < 0)
        | (df["Future_Missing_Prerequisites"] > df["Missing_Prerequisite_Courses"])
    ]

    if len(invalid) == 0:

        return create_result(
            "Future Missing Prerequisites",
            True,
            "Future missing prerequisite values are valid.",
        )

    return create_result(
        "Future Missing Prerequisites",
        False,
        f"{len(invalid)} invalid future prerequisite values.",
    )


# ==========================================================
# Rule 19
# Future Academic Warning
# ==========================================================


def check_future_academic_warning(df):

    invalid = df[df["Future_Academic_Warning"] < df["Academic_Warning_Count"]]

    if len(invalid) == 0:

        return create_result(
            "Future Academic Warning", True, "Future academic warning values are valid."
        )

    return create_result(
        "Future Academic Warning",
        False,
        f"{len(invalid)} invalid future academic warning values.",
    )


# ==========================================================
# Rule 20
# Future OJT Eligible
# ==========================================================


def check_future_ojt_eligible(df):

    expected = (df["Future_Credits_At_OJT"] >= 100) & (
        df["Future_Missing_Prerequisites"] == 0
    )

    actual = df["Future_OJT_Eligible"].astype(bool)

    invalid = df[actual != expected]

    if len(invalid) == 0:

        return create_result(
            "Future OJT Eligible", True, "Future OJT eligibility values are valid."
        )

    return create_result(
        "Future OJT Eligible",
        False,
        f"{len(invalid)} invalid future OJT eligibility values.",
    )


# ==========================================================
# Rule 21
# OJT Delay Outcome
# ==========================================================


def check_ojt_delay_outcome(df):

    expected = (~df["Future_OJT_Eligible"].astype(bool)).astype(int)

    actual = df["OJT_Delay_Outcome"].astype(int)

    invalid = df[actual != expected]

    if len(invalid) == 0:

        return create_result("OJT Delay Outcome", True, "OJT delay outcomes are valid.")

    return create_result(
        "OJT Delay Outcome", False, f"{len(invalid)} invalid OJT delay outcomes."
    )
