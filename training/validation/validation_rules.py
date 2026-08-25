"""
==========================================================
Validation Rules
OJT AI Project

This module contains all validation rules
used before training Machine Learning models.
==========================================================
"""

import os
import sys

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

from data.generator.profiles import PROFILES

# ==========================================================
# Helper Function
# ==========================================================

def create_result(rule, status, message):
    """
    Standard format for every validation result.
    """

    return {
        "rule": rule,
        "status": status,
        "message": message
    }


# ==========================================================
# Rule 1
# File Exists
# ==========================================================

def check_file_exists(file_path):

    if os.path.exists(file_path):

        return create_result(
            "Dataset File",
            True,
            "Dataset file found."
        )

    return create_result(
        "Dataset File",
        False,
        "Dataset file does not exist."
    )


# ==========================================================
# Rule 2
# Missing Values
# ==========================================================

def check_missing_values(df):

    missing = df.isnull().sum().sum()

    if missing == 0:

        return create_result(
            "Missing Values",
            True,
            "No missing values found."
        )

    return create_result(
        "Missing Values",
        False,
        f"Found {missing} missing values."
    )


# ==========================================================
# Rule 3
# Duplicate MSSV
# ==========================================================

def check_duplicate_student_id(df):

    duplicate = df["MSSV"].duplicated().sum()

    if duplicate == 0:

        return create_result(
            "Duplicate Student ID",
            True,
            "No duplicate student IDs."
        )

    return create_result(
        "Duplicate Student ID",
        False,
        f"Found {duplicate} duplicated student IDs."
    )


# ==========================================================
# Rule 4
# GPA Validation
# ==========================================================

def check_gpa_range(df):

    invalid = df[
        (df["GPA_Cumulative"] < 0)
        |
        (df["GPA_Cumulative"] > 10)
    ]

    if len(invalid) == 0:

        return create_result(
            "GPA Range",
            True,
            "All GPA values are valid."
        )

    return create_result(
        "GPA Range",
        False,
        f"{len(invalid)} invalid GPA values."
    )


# ==========================================================
# Rule 5
# Credits Completed
# ==========================================================

def check_completed_credits(df):

    invalid = df[
        (df["Credits_Completed"] < 0)
        |
        (df["Credits_Completed"] > df["Total_Credits"])
    ]

    if len(invalid) == 0:

        return create_result(
            "Completed Credits",
            True,
            "Completed credits are valid."
        )

    return create_result(
        "Completed Credits",
        False,
        f"{len(invalid)} invalid completed credits."
    )


# ==========================================================
# Rule 6
# Remaining Credits
# ==========================================================

def check_remaining_credits(df):

    expected = (
        df["Total_Credits"]
        -
        df["Credits_Completed"]
    )

    invalid = df[
        df["Credits_Remaining"] != expected
    ]

    if len(invalid) == 0:

        return create_result(
            "Remaining Credits",
            True,
            "Remaining credits are correct."
        )

    return create_result(
        "Remaining Credits",
        False,
        f"{len(invalid)} invalid remaining credits."
    )


# ==========================================================
# Rule 7
# Completion Rate
# ==========================================================

def check_completion_rate(df):

    expected = (
        (
            df["Credits_Completed"]
            /
            df["Total_Credits"]
        ) * 100
    ).round(2)

    invalid = df[
        abs(
            df["Completion_Rate"] - expected
        ) > 0.01
    ]

    if len(invalid) == 0:

        return create_result(
            "Completion Rate",
            True,
            "Completion rate is correct."
        )

    return create_result(
        "Completion Rate",
        False,
        f"{len(invalid)} invalid completion rates."
    )


# ==========================================================
# Rule 8
# Remaining To OJT
# ==========================================================

def check_remaining_to_ojt(df):

    invalid = df[
        df["Remaining_To_OJT"] < 0
    ]

    if len(invalid) == 0:

        return create_result(
            "Remaining To OJT",
            True,
            "Remaining credits to OJT are valid."
        )

    return create_result(
        "Remaining To OJT",
        False,
        f"{len(invalid)} invalid values."
    )


# ==========================================================
# Rule 9
# Risk Score
# ==========================================================

def check_risk_score(df):

    invalid = df[
        df["Risk_Score"] < 0
    ]

    if len(invalid) == 0:

        return create_result(
            "Risk Score",
            True,
            "Risk score is valid."
        )

    return create_result(
        "Risk Score",
        False,
        f"{len(invalid)} invalid risk scores."
    )


# ==========================================================
# Rule 10
# Risk Level
# ==========================================================

def check_risk_level(df):

    valid = [
        "Low",
        "Medium",
        "High"
    ]

    invalid = df[
        ~df["Risk_Level"].isin(valid)
    ]

    if len(invalid) == 0:

        return create_result(
            "Risk Level",
            True,
            "Risk levels are valid."
        )

    return create_result(
        "Risk Level",
        False,
        f"{len(invalid)} invalid risk levels."
    )


# ==========================================================
# Rule 11
# Delay Risk Label
# ==========================================================

def check_delay_risk(df):

    valid = [0, 1]

    invalid = df[
        ~df["OJT_Delay_Risk"].isin(valid)
    ]

    if len(invalid) == 0:

        return create_result(
            "Delay Risk",
            True,
            "Delay risk labels are valid."
        )

    return create_result(
        "Delay Risk",
        False,
        f"{len(invalid)} invalid labels."
    )


# ==========================================================
# Rule 12
# OJT Eligible
# ==========================================================

def check_ojt_eligible(df):

    valid = [True, False]

    invalid = df[
        ~df["OJT_Eligible"].isin(valid)
    ]

    if len(invalid) == 0:

        return create_result(
            "OJT Eligible",
            True,
            "Eligibility values are valid."
        )

    return create_result(
        "OJT Eligible",
        False,
        f"{len(invalid)} invalid eligibility values."
    )


# ==========================================================
# Rule 13
# Readiness
# ==========================================================

def check_readiness(df):

    invalid = df[
        (df["OJT_Readiness"] < 0)
        |
        (df["OJT_Readiness"] > 100)
    ]

    if len(invalid) == 0:

        return create_result(
            "Readiness",
            True,
            "Readiness values are valid."
        )

    return create_result(
        "Readiness",
        False,
        f"{len(invalid)} invalid readiness values."
    )


# ==========================================================
# Rule 14
# AI Recommendation
# ==========================================================

def check_ai_recommendation(df):

    invalid = df[
        df["AI_Recommendation"].isnull()
    ]

    if len(invalid) == 0:

        return create_result(
            "AI Recommendation",
            True,
            "Recommendation column is valid."
        )

    return create_result(
        "AI Recommendation",
        False,
        f"{len(invalid)} missing recommendations."
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

    invalid = df[
        ~df["Student_Profile"].isin(valid_profiles)
    ]

    if len(invalid) == 0:

        return create_result(
            "Student Profile",
            True,
            "Student profiles are valid."
        )

    return create_result(
        "Student Profile",
        False,
        f"{len(invalid)} invalid student profiles."
    )