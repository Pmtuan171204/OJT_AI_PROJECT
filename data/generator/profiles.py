"""
==========================================================
STUDENT PROFILES
OJT AI Project

Each profile represents a typical learning pattern
of university students.
==========================================================
"""

PROFILES = {

    # ======================================================
    # Excellent Student
    # ======================================================

    "Excellent": {

        "probability": 0.10,

        "gpa": (8.5, 10.0),

        "failed_courses": (0, 0),

        "retake_count": (0, 0),

        "academic_warning": (0, 0),

        "suspension": (0, 0),

        "missing_prerequisite": (0, 0)
    },

    # ======================================================
    # Good Student
    # ======================================================

    "Good": {

        "probability": 0.30,

        "gpa": (7.0, 8.49),

        "failed_courses": (0, 1),

        "retake_count": (0, 2),

        "academic_warning": (0, 0),

        "suspension": (0, 0),

        "missing_prerequisite": (0, 1)
    },

    # ======================================================
    # Average Student
    # ======================================================

    "Average": {

        "probability": 0.30,

        "gpa": (6.0, 6.99),

        "failed_courses": (1, 3),

        "retake_count": (1, 4),

        "academic_warning": (0, 1),

        "suspension": (0, 0),

        "missing_prerequisite": (0, 2)
    },

    # ======================================================
    # At Risk Student
    # ======================================================

    "AtRisk": {

        "probability": 0.15,

        "gpa": (5.0, 5.99),

        "failed_courses": (3, 5),

        "retake_count": (3, 6),

        "academic_warning": (1, 2),

        "suspension": (0, 1),

        "missing_prerequisite": (1, 3)
    },

    # ======================================================
    # Critical Student
    # ======================================================

    "Critical": {

        "probability": 0.05,

        "gpa": (3.5, 4.99),

        "failed_courses": (5, 8),

        "retake_count": (5, 8),

        "academic_warning": (2, 3),

        "suspension": (1, 2),

        "missing_prerequisite": (2, 4)
    },

    # ======================================================
    # Recovery Student
    # ======================================================

    "Recovery": {

        "probability": 0.05,

        "gpa": (5.8, 6.8),

        "failed_courses": (1, 2),

        "retake_count": (2, 4),

        "academic_warning": (0, 1),

        "suspension": (0, 0),

        "missing_prerequisite": (0, 1)
    },

    # ======================================================
    # Late Starter
    # ======================================================

    "LateStarter": {

        "probability": 0.05,

        "gpa": (6.2, 7.2),

        "failed_courses": (2, 4),

        "retake_count": (2, 5),

        "academic_warning": (0, 1),

        "suspension": (0, 0),

        "missing_prerequisite": (0, 2)
    }

}