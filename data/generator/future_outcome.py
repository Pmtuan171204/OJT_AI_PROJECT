"""
==========================================================
FUTURE OUTCOME SIMULATOR
OJT AI Project

Simulate future academic progress and determine
the actual OJT delay outcome.

Important:
This module does NOT use Risk_Score to generate
the OJT delay target.
==========================================================
"""

import random

from config import TOTAL_CREDITS, REQUIRED_CREDITS_FOR_OJT, RANDOM_SEED

# ==========================================================
# RANDOM SEED
# ==========================================================

future_random = random.Random(RANDOM_SEED)


# ==========================================================
# PROFILE PROGRESSION
# ==========================================================

PROFILE_PROGRESS = {
    "Excellent": {
        "credit_range": (18, 24),
        "failure_probability": 0.03,
        "warning_probability": 0.01,
        "prerequisite_completion_probability": 0.90,
    },
    "Good": {
        "credit_range": (17, 23),
        "failure_probability": 0.07,
        "warning_probability": 0.02,
        "prerequisite_completion_probability": 0.82,
    },
    "Average": {
        "credit_range": (15, 21),
        "failure_probability": 0.14,
        "warning_probability": 0.05,
        "prerequisite_completion_probability": 0.70,
    },
    "AtRisk": {
        "credit_range": (12, 19),
        "failure_probability": 0.25,
        "warning_probability": 0.12,
        "prerequisite_completion_probability": 0.55,
    },
    "Critical": {
        "credit_range": (9, 17),
        "failure_probability": 0.38,
        "warning_probability": 0.20,
        "prerequisite_completion_probability": 0.40,
    },
    "Recovery": {
        "credit_range": (15, 21),
        "failure_probability": 0.12,
        "warning_probability": 0.04,
        "prerequisite_completion_probability": 0.72,
    },
    "LateStarter": {
        "credit_range": (13, 20),
        "failure_probability": 0.20,
        "warning_probability": 0.08,
        "prerequisite_completion_probability": 0.60,
    },
}


# ==========================================================
# GPA ADJUSTMENT
# ==========================================================


def calculate_gpa_factor(gpa):
    """
    Adjust future academic performance based on GPA.
    """

    if gpa >= 8.0:
        return 0.85

    if gpa >= 7.0:
        return 0.95

    if gpa >= 6.0:
        return 1.00

    if gpa >= 5.0:
        return 1.10

    return 1.20


# ==========================================================
# SIMULATE ONE FUTURE SEMESTER
# ==========================================================


def simulate_future_semester(
    profile,
    current_gpa,
    current_failed,
    current_warnings,
    current_missing_prerequisites,
):
    """
    Simulate one future academic semester.

    Returns:
        credits_earned
        new_failed_courses
        new_academic_warning
        future_missing_prerequisites
    """

    settings = PROFILE_PROGRESS[profile]

    # ------------------------------------------------------
    # Credit progression
    # ------------------------------------------------------

    credit_low, credit_high = settings["credit_range"]

    base_credits = future_random.randint(credit_low, credit_high)

    gpa_factor = calculate_gpa_factor(current_gpa)

    credits_earned = round(base_credits / gpa_factor)

    credits_earned = max(0, credits_earned)

    # ------------------------------------------------------
    # Failed courses
    # ------------------------------------------------------

    failure_probability = settings["failure_probability"]

    # Existing academic problems increase
    # probability of another failed course.

    failure_probability += min(current_failed * 0.015, 0.15)

    if current_gpa < 6:
        failure_probability += 0.05

    failure_probability = min(failure_probability, 0.60)

    new_failed = 0

    if future_random.random() < failure_probability:

        new_failed = future_random.randint(1, 2)

    # ------------------------------------------------------
    # Academic warning
    # ------------------------------------------------------

    warning_probability = settings["warning_probability"]

    if current_gpa < 6:
        warning_probability += 0.08

    if current_failed >= 3:
        warning_probability += 0.05

    warning_probability = min(warning_probability, 0.50)

    new_warning = 0

    if future_random.random() < warning_probability:
        new_warning = 1

    # ------------------------------------------------------
    # Prerequisite completion
    # ------------------------------------------------------

    future_missing = current_missing_prerequisites

    if future_missing > 0:

        completion_probability = settings["prerequisite_completion_probability"]

        # Poor academic performance makes
        # prerequisite completion harder.

        if current_gpa < 6:
            completion_probability -= 0.10

        if current_failed >= 3:
            completion_probability -= 0.10

        completion_probability = max(0.10, completion_probability)

        if future_random.random() < completion_probability:

            future_missing -= 1

    return (credits_earned, new_failed, new_warning, future_missing)


# ==========================================================
# SIMULATE FUTURE OJT OUTCOME
# ==========================================================


def simulate_ojt_outcome(student):
    """
    Simulate the student's future academic progress
    until the planned OJT semester.

    Returns a dictionary containing future state
    and OJT delay outcome.
    """

    profile = student["Student_Profile"]

    current_semester = student["Current_Semester"]

    planned_ojt_semester = student["Planned_OJT_Semester"]

    future_credits = student["Credits_Completed"]

    current_gpa = student["GPA_Cumulative"]

    future_failed = student["Failed_Courses"]

    future_warnings = student["Academic_Warning_Count"]

    future_missing = student["Missing_Prerequisite_Courses"]

    # ------------------------------------------------------
    # If current semester is already beyond
    # planned OJT semester, evaluate current state.
    # ------------------------------------------------------

    if current_semester < planned_ojt_semester:

        semesters_to_simulate = planned_ojt_semester - current_semester

        for _ in range(semesters_to_simulate):

            credits_earned, new_failed, new_warning, future_missing = (
                simulate_future_semester(
                    profile=profile,
                    current_gpa=current_gpa,
                    current_failed=future_failed,
                    current_warnings=future_warnings,
                    current_missing_prerequisites=future_missing,
                )
            )

            # --------------------------------------------------
            # Update future academic state
            # --------------------------------------------------

            future_credits += credits_earned

            future_failed += new_failed

            future_warnings += new_warning

            # --------------------------------------------------
            # Keep credits within valid range
            # --------------------------------------------------

            future_credits = min(future_credits, TOTAL_CREDITS)

    # ------------------------------------------------------
    # Determine future OJT eligibility
    # ------------------------------------------------------

    future_eligible = future_credits >= REQUIRED_CREDITS_FOR_OJT and future_missing == 0

    # ------------------------------------------------------
    # Determine actual future outcome
    # ------------------------------------------------------

    if future_eligible:

        delay_outcome = 0

    else:

        delay_outcome = 1

    return {
        "Future_Credits_At_OJT": future_credits,
        "Future_Failed_Courses": future_failed,
        "Future_Missing_Prerequisites": future_missing,
        "Future_Academic_Warning": future_warnings,
        "Future_OJT_Eligible": future_eligible,
        "OJT_Delay_Outcome": delay_outcome,
    }
