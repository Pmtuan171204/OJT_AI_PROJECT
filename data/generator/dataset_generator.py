"""
==========================================================
DATASET GENERATOR
OJT AI Project

Generate Synthetic Dataset for OJT Risk Prediction
==========================================================
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from config import (
    TOTAL_STUDENTS,
    TOTAL_CREDITS
)

from utils import (
    generate_student_id,
    choose_profile,
    generate_current_semester,
    generate_gpa,
    generate_completed_credits,
    generate_integer_value,
    generate_planned_ojt_semester,
    calculate_remaining_credits,
    calculate_completion_rate,
    calculate_average_credits_per_semester,
    calculate_remaining_to_ojt
)

from business_rules import (
    calculate_risk_score,
    get_risk_level,
    assign_risk_label,
    check_ojt_eligibility,
    calculate_ojt_readiness,
    generate_ai_recommendation
)


# =====================================================
# Generate One Student
# =====================================================

def generate_student(index):

    student = {}

    # ==============================
    # Basic Information
    # ==============================

    student["MSSV"] = generate_student_id(index)

    profile = choose_profile()

    student["Student_Profile"] = profile

    student["Current_Semester"] = generate_current_semester()

    student["GPA_Cumulative"] = generate_gpa(profile)

    # ==============================
    # Credits
    # ==============================

    completed = generate_completed_credits(
        profile,
        student["Current_Semester"]
    )

    student["Total_Credits"] = TOTAL_CREDITS

    student["Credits_Completed"] = completed

    student["Credits_Remaining"] = calculate_remaining_credits(
        completed
    )

    student["Completion_Rate"] = calculate_completion_rate(
        completed
    )

    student["Average_Credits_Per_Semester"] = (
        calculate_average_credits_per_semester(
            completed,
            student["Current_Semester"]
        )
    )

    student["Remaining_To_OJT"] = (
        calculate_remaining_to_ojt(
            completed
        )
    )

    # ==============================
    # Academic Information
    # ==============================

    student["Failed_Courses"] = generate_integer_value(
        profile,
        "failed_courses"
    )

    student["Retake_Count"] = generate_integer_value(
        profile,
        "retake_count"
    )

    student["Missing_Prerequisite_Courses"] = generate_integer_value(
        profile,
        "missing_prerequisite"
    )

    student["Academic_Warning_Count"] = generate_integer_value(
        profile,
        "academic_warning"
    )

    student["Suspension_Count"] = generate_integer_value(
        profile,
        "suspension"
    )

    student["Planned_OJT_Semester"] = (
        generate_planned_ojt_semester(
            student["Current_Semester"]
        )
    )

    # ==============================
    # Business Rules
    # ==============================

    score = calculate_risk_score(student)

    student["Risk_Score"] = score

    student["Risk_Level"] = get_risk_level(score)

    student["OJT_Delay_Risk"] = assign_risk_label(score)

    student["OJT_Eligible"] = check_ojt_eligibility(student)

    student["OJT_Readiness"] = calculate_ojt_readiness(student)

    student["AI_Recommendation"] = (
        generate_ai_recommendation(student)
    )

    return student


# =====================================================
# Generate Dataset
# =====================================================

def generate_dataset():

    students = []

    print("=" * 60)
    print("Generating Dataset...")
    print("=" * 60)

    for i in tqdm(range(1, TOTAL_STUDENTS + 1)):

        students.append(
            generate_student(i)
        )

    df = pd.DataFrame(students)

    return df


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df):

    current_file = Path(__file__).resolve()

    generator_folder = current_file.parent

    data_folder = generator_folder.parent

    raw_folder = data_folder / "raw"

    raw_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    excel_path = raw_folder / "ojt_dataset.xlsx"

    csv_path = raw_folder / "ojt_dataset.csv"

    df.to_excel(
        excel_path,
        index=False
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print()

    print("=" * 60)
    print("Dataset Saved Successfully")
    print("=" * 60)

    print("Excel :", excel_path)

    print("CSV   :", csv_path)


# =====================================================
# Show Statistics
# =====================================================

def show_statistics(df):

    print()

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("Total Students :", len(df))

    print()

    print(
        "Average GPA :",
        round(df["GPA_Cumulative"].mean(), 2)
    )

    print(
        "Average Completed Credits :",
        round(df["Credits_Completed"].mean(), 2)
    )

    print()

    print("Risk Level Distribution")

    print(df["Risk_Level"].value_counts())

    print()

    print("Risk Label Distribution")

    print(df["OJT_Delay_Risk"].value_counts())

    print()

    print("OJT Eligibility")

    print(df["OJT_Eligible"].value_counts())

    print()

    print("Student Profiles")

    print(df["Student_Profile"].value_counts())

    print("=" * 60)


# =====================================================
# Main
# =====================================================

def main():

    dataset = generate_dataset()

    save_dataset(dataset)

    show_statistics(dataset)


if __name__ == "__main__":

    main()