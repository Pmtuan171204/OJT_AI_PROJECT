"""
==========================================================
UTILITY FUNCTIONS
OJT AI Project

Utility functions for Dataset Generator
==========================================================
"""

import random
import numpy as np

from config import (
    RANDOM_SEED,
    TOTAL_CREDITS,
    REQUIRED_CREDITS_FOR_OJT
)

from profiles import PROFILES

# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================================
# STUDENT ID
# ==========================================================

def generate_student_id(index: int) -> str:
    """
    Generate student ID.

    Example:
        1 -> SE1800001
    """

    return f"SE18{index:05d}"


# ==========================================================
# PROFILE
# ==========================================================

def choose_profile() -> str:
    """
    Randomly choose a student profile
    based on predefined probability.
    """

    profile_names = list(PROFILES.keys())

    probabilities = [
        profile["probability"]
        for profile in PROFILES.values()
    ]

    return random.choices(
        profile_names,
        weights=probabilities,
        k=1
    )[0]


# ==========================================================
# SEMESTER
# ==========================================================

SEMESTER_WEIGHTS = {

    1: 5,
    2: 10,
    3: 18,
    4: 22,
    5: 22,
    6: 13,
    7: 7,
    8: 3

}

def generate_current_semester():

    semesters = list(SEMESTER_WEIGHTS.keys())

    weights = list(SEMESTER_WEIGHTS.values())

    return random.choices(
        semesters,
        weights=weights,
        k=1
    )[0]


# ==========================================================
# GPA
# ==========================================================

def generate_gpa(profile):

    low, high = PROFILES[profile]["gpa"]

    return round(
        random.uniform(low, high),
        2
    )


# ==========================================================
# COMPLETED CREDITS
# ==========================================================

CREDIT_MATRIX = {

    "Excellent": {

        1:(18,24),
        2:(36,48),
        3:(54,66),
        4:(72,84),
        5:(90,105),
        6:(105,115),
        7:(112,120),
        8:(115,120)

    },

    "Good": {

        1:(18,22),
        2:(34,46),
        3:(50,64),
        4:(68,82),
        5:(88,100),
        6:(100,112),
        7:(108,118),
        8:(112,120)

    },

    "Average": {

        1:(16,20),
        2:(30,42),
        3:(45,58),
        4:(60,75),
        5:(80,95),
        6:(92,108),
        7:(100,115),
        8:(108,120)

    },

    "AtRisk": {

        1:(14,18),
        2:(24,36),
        3:(36,50),
        4:(50,68),
        5:(68,85),
        6:(82,98),
        7:(92,108),
        8:(100,115)

    },

    "Critical": {

        1:(10,16),
        2:(18,30),
        3:(28,42),
        4:(40,58),
        5:(55,75),
        6:(65,88),
        7:(78,98),
        8:(88,105)

    },

    "Recovery": {

        1:(15,19),
        2:(28,40),
        3:(42,55),
        4:(58,72),
        5:(78,90),
        6:(90,104),
        7:(102,114),
        8:(108,118)

    },

    "LateStarter": {

        1:(12,18),
        2:(22,34),
        3:(35,50),
        4:(55,72),
        5:(78,95),
        6:(92,108),
        7:(102,115),
        8:(108,120)

    }

}


def generate_completed_credits(profile, semester):

    low, high = CREDIT_MATRIX[profile][semester]

    return random.randint(
        low,
        high
    )


# ==========================================================
# INTEGER FEATURES
# ==========================================================

def generate_integer_value(profile, feature):

    low, high = PROFILES[profile][feature]

    return random.randint(
        low,
        high
    )


# ==========================================================
# PLANNED OJT SEMESTER
# ==========================================================

def generate_planned_ojt_semester(current_semester):

    if current_semester <= 4:
        return random.choice([5,6])

    elif current_semester <=6:
        return random.choice([6,7])

    else:
        return 8


# ==========================================================
# DERIVED FEATURES
# ==========================================================

def calculate_remaining_credits(completed_credits):

    return TOTAL_CREDITS - completed_credits


def calculate_completion_rate(completed_credits):

    return round(
        completed_credits / TOTAL_CREDITS * 100,
        2
    )


def calculate_average_credits_per_semester(
    completed_credits,
    semester
):

    if semester == 0:
        return 0

    return round(
        completed_credits / semester,
        2
    )


# ==========================================================
# OJT FEATURES
# ==========================================================

def calculate_remaining_to_ojt(completed_credits):

    remain = REQUIRED_CREDITS_FOR_OJT - completed_credits

    return max(remain,0)


def check_ojt_credit_requirement(completed_credits):

    return completed_credits >= REQUIRED_CREDITS_FOR_OJT


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def safe_percentage(value, total):

    if total == 0:
        return 0

    return round(
        value / total * 100,
        2
    )