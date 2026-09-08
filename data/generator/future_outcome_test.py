"""
==========================================================
FUTURE OUTCOME TEST
OJT AI Project

Test Future Outcome Simulator independently
before integrating it into the main dataset generator.
==========================================================
"""

from future_outcome import simulate_ojt_outcome

# ==========================================================
# TEST STUDENTS
# ==========================================================

TEST_STUDENTS = [
    {
        "MSSV": "TEST001",
        "Student_Profile": "Excellent",
        "Current_Semester": 4,
        "GPA_Cumulative": 9.0,
        "Credits_Completed": 75,
        "Failed_Courses": 0,
        "Retake_Count": 0,
        "Missing_Prerequisite_Courses": 0,
        "Academic_Warning_Count": 0,
        "Suspension_Count": 0,
        "Planned_OJT_Semester": 6,
    },
    {
        "MSSV": "TEST002",
        "Student_Profile": "Average",
        "Current_Semester": 4,
        "GPA_Cumulative": 6.3,
        "Credits_Completed": 65,
        "Failed_Courses": 2,
        "Retake_Count": 2,
        "Missing_Prerequisite_Courses": 1,
        "Academic_Warning_Count": 1,
        "Suspension_Count": 0,
        "Planned_OJT_Semester": 6,
    },
    {
        "MSSV": "TEST003",
        "Student_Profile": "Critical",
        "Current_Semester": 4,
        "GPA_Cumulative": 4.2,
        "Credits_Completed": 45,
        "Failed_Courses": 6,
        "Retake_Count": 6,
        "Missing_Prerequisite_Courses": 3,
        "Academic_Warning_Count": 2,
        "Suspension_Count": 1,
        "Planned_OJT_Semester": 6,
    },
    {
        "MSSV": "TEST004",
        "Student_Profile": "Good",
        "Current_Semester": 6,
        "GPA_Cumulative": 7.8,
        "Credits_Completed": 102,
        "Failed_Courses": 0,
        "Retake_Count": 0,
        "Missing_Prerequisite_Courses": 0,
        "Academic_Warning_Count": 0,
        "Suspension_Count": 0,
        "Planned_OJT_Semester": 6,
    },
]


# ==========================================================
# VALIDATE RESULT
# ==========================================================


def validate_result(student, result):

    errors = []

    # ------------------------------------------------------
    # Future credits
    # ------------------------------------------------------

    future_credits = result["Future_Credits_At_OJT"]

    if future_credits < 0:
        errors.append("Future credits cannot be negative.")

    if future_credits > 120:
        errors.append("Future credits cannot exceed 120.")

    # ------------------------------------------------------
    # Future missing prerequisites
    # ------------------------------------------------------

    future_missing = result["Future_Missing_Prerequisites"]

    if future_missing < 0:
        errors.append("Future missing prerequisites cannot be negative.")

    # ------------------------------------------------------
    # Future eligibility
    # ------------------------------------------------------

    expected_eligibility = future_credits >= 100 and future_missing == 0

    actual_eligibility = result["Future_OJT_Eligible"]

    if actual_eligibility != expected_eligibility:
        errors.append("Future OJT eligibility is incorrect.")

    # ------------------------------------------------------
    # Delay outcome
    # ------------------------------------------------------

    outcome = result["OJT_Delay_Outcome"]

    if outcome not in [0, 1]:
        errors.append("OJT_Delay_Outcome must be 0 or 1.")

    expected_outcome = 0 if expected_eligibility else 1

    if outcome != expected_outcome:
        errors.append("OJT_Delay_Outcome does not match " "future OJT eligibility.")

    return errors


# ==========================================================
# MAIN TEST
# ==========================================================


def main():

    print("=" * 60)
    print("FUTURE OUTCOME GENERATOR TEST")
    print("=" * 60)

    passed = 0
    failed = 0

    for student in TEST_STUDENTS:

        print()
        print("-" * 60)

        print("Student:", student["MSSV"])

        result = simulate_ojt_outcome(student)

        print("Current Credits:", student["Credits_Completed"])

        print("Planned OJT Semester:", student["Planned_OJT_Semester"])

        print("Future Credits:", result["Future_Credits_At_OJT"])

        print("Future Failed Courses:", result["Future_Failed_Courses"])

        print("Future Missing Prerequisites:", result["Future_Missing_Prerequisites"])

        print("Future OJT Eligible:", result["Future_OJT_Eligible"])

        print("OJT Delay Outcome:", result["OJT_Delay_Outcome"])

        errors = validate_result(student, result)

        if len(errors) == 0:

            print("[PASS]", student["MSSV"])

            passed += 1

        else:

            print("[FAIL]", student["MSSV"])

            for error in errors:
                print(" -", error)

            failed += 1

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print("Passed:", passed)
    print("Failed:", failed)
    print("Total :", len(TEST_STUDENTS))

    if failed == 0:

        print()
        print("ALL FUTURE OUTCOME TESTS PASSED.")

    else:

        print()
        print("FUTURE OUTCOME TEST FAILED.")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()
