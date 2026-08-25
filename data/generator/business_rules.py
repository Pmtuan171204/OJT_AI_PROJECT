"""
==========================================================
BUSINESS RULES
OJT AI Project

Business logic for OJT Risk Prediction
==========================================================
"""

from config import (
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    REQUIRED_CREDITS_FOR_OJT
)

# ==========================================================
# RISK SCORE
# ==========================================================

def calculate_risk_score(student):
    """
    Calculate risk score based on business rules.
    Higher score = Higher OJT Delay Risk.
    """

    score = 0

    # ======================================================
    # GPA
    # ======================================================

    gpa = student["GPA_Cumulative"]

    if gpa < 5:
        score += 4

    elif gpa < 6:
        score += 3

    elif gpa < 7:
        score += 1

    # ======================================================
    # Failed Courses
    # ======================================================

    failed = student["Failed_Courses"]

    if failed >= 5:
        score += 3

    elif failed >= 3:
        score += 2

    elif failed >= 1:
        score += 1

    # ======================================================
    # Retake Count
    # ======================================================

    retake = student["Retake_Count"]

    if retake >= 5:
        score += 3

    elif retake >= 3:
        score += 2

    elif retake >= 1:
        score += 1

    # ======================================================
    # Missing Prerequisite
    # ======================================================

    missing = student["Missing_Prerequisite_Courses"]

    if missing >= 3:
        score += 4

    elif missing >= 2:
        score += 3

    elif missing == 1:
        score += 2

    # ======================================================
    # Academic Warning
    # ======================================================

    warning = student["Academic_Warning_Count"]

    if warning >= 2:
        score += 3

    elif warning == 1:
        score += 2

    # ======================================================
    # Suspension
    # ======================================================

    suspension = student["Suspension_Count"]

    if suspension >= 2:
        score += 4

    elif suspension == 1:
        score += 2

    # ======================================================
    # Credits Remaining
    # ======================================================

    remain = student["Credits_Remaining"]

    if remain > 50:
        score += 3

    elif remain > 35:
        score += 2

    elif remain > 20:
        score += 1

    return score


# ==========================================================
# RISK LEVEL
# ==========================================================

def get_risk_level(score):
    """
    Convert Risk Score to Risk Level.
    """

    if score <= LOW_RISK_MAX:
        return "Low"

    elif score <= MEDIUM_RISK_MAX:
        return "Medium"

    return "High"


# ==========================================================
# BINARY LABEL
# ==========================================================

def assign_risk_label(score):
    """
    Binary Classification Label.

    0 = Safe

    1 = At Risk
    """

    if score <= MEDIUM_RISK_MAX:
        return 0

    return 1


# ==========================================================
# OJT ELIGIBILITY
# ==========================================================

def check_ojt_eligibility(student):
    """
    Check whether student satisfies OJT conditions.
    """

    if student["Credits_Completed"] < REQUIRED_CREDITS_FOR_OJT:
        return False

    if student["Missing_Prerequisite_Courses"] > 0:
        return False

    return True


# ==========================================================
# OJT READINESS
# ==========================================================

def calculate_ojt_readiness(student):
    """
    Return readiness percentage.
    """

    score = 100

    score -= max(0, student["Credits_Remaining"] * 0.5)

    score -= student["Failed_Courses"] * 3

    score -= student["Retake_Count"] * 2

    score -= student["Academic_Warning_Count"] * 5

    score -= student["Suspension_Count"] * 8

    score -= student["Missing_Prerequisite_Courses"] * 6

    return max(0, round(score, 2))


# ==========================================================
# AI RECOMMENDATION
# ==========================================================

def generate_ai_recommendation(student):
    """
    Generate recommendation for student.
    """

    recommendations = []

    if student["Credits_Completed"] < REQUIRED_CREDITS_FOR_OJT:

        remain = REQUIRED_CREDITS_FOR_OJT - student["Credits_Completed"]

        recommendations.append(
            f"Complete at least {remain} more credits."
        )

    if student["Missing_Prerequisite_Courses"] > 0:

        recommendations.append(
            "Complete all prerequisite courses."
        )

    if student["Failed_Courses"] > 0:

        recommendations.append(
            "Retake failed courses as soon as possible."
        )

    if student["Academic_Warning_Count"] > 0:

        recommendations.append(
            "Improve GPA to remove academic warning."
        )

    if len(recommendations) == 0:

        recommendations.append(
            "You are ready for OJT."
        )

    return " ".join(recommendations)