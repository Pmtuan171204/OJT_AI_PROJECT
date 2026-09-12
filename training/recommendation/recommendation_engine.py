import os
import sys
import json

# ------------------------------------------------------------------
# Project Root
# ------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OJT_REQUIRED_CREDITS = 100

RISK_THRESHOLDS = {"LOW": 0.30, "MEDIUM": 0.60, "HIGH": 0.80}


# ------------------------------------------------------------------
# Risk Interpretation
# ------------------------------------------------------------------


def determine_risk_level(delay_probability):
    """
    Convert ML delay probability into an interpretable risk level.

    < 30%       -> Low
    30% - <60%  -> Medium
    60% - <80%  -> High
    >= 80%      -> Critical
    """

    if not 0 <= delay_probability <= 1:
        raise ValueError("delay_probability must be between 0 and 1.")

    if delay_probability < RISK_THRESHOLDS["LOW"]:
        return "Low"

    elif delay_probability < RISK_THRESHOLDS["MEDIUM"]:
        return "Medium"

    elif delay_probability < RISK_THRESHOLDS["HIGH"]:
        return "High"

    else:
        return "Critical"


# ------------------------------------------------------------------
# Prediction Interpretation
# ------------------------------------------------------------------


def interpret_prediction(prediction_class, delay_probability):
    """
    Interpret the ML prediction.
    """

    if prediction_class not in [0, 1]:
        raise ValueError("prediction_class must be 0 or 1.")

    risk_level = determine_risk_level(delay_probability)

    if prediction_class == 1:
        prediction_label = "OJT Delay"
    else:
        prediction_label = "No OJT Delay"

    return {
        "prediction_class": prediction_class,
        "prediction_label": prediction_label,
        "delay_probability": round(delay_probability, 4),
        "risk_level": risk_level,
    }


# ------------------------------------------------------------------
# Input Validation
# ------------------------------------------------------------------


def validate_student_data(student):
    """
    Validate current student information required by the
    Recommendation Engine.
    """

    required_fields = [
        "Student_Profile",
        "Current_Semester",
        "GPA_Cumulative",
        "Total_Credits",
        "Credits_Completed",
        "Credits_Remaining",
        "Failed_Courses",
        "Retake_Count",
        "Missing_Prerequisite_Courses",
        "Academic_Warning_Count",
        "Suspension_Count",
        "Planned_OJT_Semester",
    ]

    for field in required_fields:

        if field not in student:
            raise ValueError(f"Missing required field: {field}")

    # ------------------------------------------------------
    # Validate Student Profile
    # ------------------------------------------------------

    VALID_PROFILES = {
        "Excellent",
        "Good",
        "Average",
        "AtRisk",
        "Critical",
        "Recovery",
        "LateStarter"
    }

    if student["Student_Profile"] not in VALID_PROFILES:
        raise ValueError(
            f"Invalid Student_Profile: "
            f"{student['Student_Profile']}"
        )

    # --------------------------------------------------------------
    # GPA
    # --------------------------------------------------------------

    gpa = student["GPA_Cumulative"]

    if not isinstance(gpa, (int, float)):
        raise ValueError("GPA_Cumulative must be numeric.")

    if not 0 <= gpa <= 10:
        raise ValueError("GPA_Cumulative must be between 0 and 10.")

    # --------------------------------------------------------------
    # Semester
    # --------------------------------------------------------------

    current_semester = student["Current_Semester"]
    planned_semester = student["Planned_OJT_Semester"]

    if current_semester < 1:
        raise ValueError("Current semester must be >= 1.")

    if planned_semester < current_semester:
        raise ValueError("Planned OJT semester cannot be before current semester.")

    # --------------------------------------------------------------
    # Credits
    # --------------------------------------------------------------

    credits_completed = student["Credits_Completed"]
    total_credits = student["Total_Credits"]

    if credits_completed < 0:
        raise ValueError("Credits_Completed cannot be negative.")

    if total_credits <= 0:
        raise ValueError("Total_Credits must be greater than 0.")

    if credits_completed > total_credits:
        raise ValueError("Credits_Completed cannot exceed Total_Credits.")

    # --------------------------------------------------------------
    # Academic Indicators
    # --------------------------------------------------------------

    non_negative_fields = [
        "Credits_Remaining",
        "Failed_Courses",
        "Retake_Count",
        "Missing_Prerequisite_Courses",
        "Academic_Warning_Count",
        "Suspension_Count",
    ]

    for field in non_negative_fields:

        if student[field] < 0:
            raise ValueError(f"{field} cannot be negative.")

    return True


# ------------------------------------------------------------------
# Calculate Current Metrics
# ------------------------------------------------------------------


def calculate_current_metrics(student):
    """
    Calculate current academic and OJT planning metrics.
    """

    current_semester = student["Current_Semester"]
    planned_ojt_semester = student["Planned_OJT_Semester"]

    credits_completed = student["Credits_Completed"]

    credit_gap = max(OJT_REQUIRED_CREDITS - credits_completed, 0)

    credit_progress = credits_completed / OJT_REQUIRED_CREDITS

    semester_gap = max(planned_ojt_semester - current_semester, 0)

    if semester_gap > 0:
        required_average_credits = credit_gap / semester_gap
    else:
        required_average_credits = credit_gap

    if current_semester > 0:
        credit_completion_efficiency = credits_completed / current_semester
    else:
        credit_completion_efficiency = 0

    completion_rate = credits_completed / student["Total_Credits"]

    return {
        "OJT_Credit_Gap": credit_gap,
        "OJT_Credit_Progress": round(credit_progress, 4),
        "OJT_Semester_Gap": semester_gap,
        "Required_Average_Credits_Per_Semester": round(required_average_credits, 2),
        "Credit_Completion_Efficiency": round(credit_completion_efficiency, 2),
        "Completion_Rate": round(completion_rate, 4),
    }


# ------------------------------------------------------------------
# Identify Issues
# ------------------------------------------------------------------


def identify_issues(student, metrics, risk_level):
    """
    Identify the student's current academic and OJT planning issues.
    """

    issues = []

    # --------------------------------------------------------------
    # GPA
    # --------------------------------------------------------------

    gpa = student["GPA_Cumulative"]

    if gpa < 5:
        issues.append(
            {
                "code": "LOW_GPA",
                "severity": "Critical",
                "title": "Very low GPA",
                "description": ("The student's cumulative GPA is below 5.0."),
            }
        )

    elif gpa < 6:
        issues.append(
            {
                "code": "LOW_GPA",
                "severity": "High",
                "title": "Low GPA",
                "description": ("The student's cumulative GPA is below 6.0."),
            }
        )

    elif gpa < 7:
        issues.append(
            {
                "code": "MODERATE_GPA",
                "severity": "Medium",
                "title": "Moderate GPA",
                "description": (
                    "The student's GPA may require improvement "
                    "to maintain academic stability."
                ),
            }
        )

    # --------------------------------------------------------------
    # Failed Courses
    # --------------------------------------------------------------

    if student["Failed_Courses"] > 0:

        issues.append(
            {
                "code": "FAILED_COURSES",
                "severity": "High",
                "title": "Failed courses",
                "description": (
                    f"{student['Failed_Courses']} failed course(s) "
                    "need to be addressed."
                ),
            }
        )

    # --------------------------------------------------------------
    # Retakes
    # --------------------------------------------------------------

    if student["Retake_Count"] > 0:

        issues.append(
            {
                "code": "RETAKE_COURSES",
                "severity": "Medium",
                "title": "Course retakes",
                "description": (
                    f"{student['Retake_Count']} course retake(s) " "have been recorded."
                ),
            }
        )

    # --------------------------------------------------------------
    # Prerequisites
    # --------------------------------------------------------------

    if student["Missing_Prerequisite_Courses"] > 0:

        issues.append(
            {
                "code": "MISSING_PREREQUISITES",
                "severity": "High",
                "title": "Missing prerequisite courses",
                "description": (
                    f"{student['Missing_Prerequisite_Courses']} "
                    "prerequisite course(s) are still missing."
                ),
            }
        )

    # --------------------------------------------------------------
    # Academic Warning
    # --------------------------------------------------------------

    if student["Academic_Warning_Count"] > 0:

        issues.append(
            {
                "code": "ACADEMIC_WARNING",
                "severity": "High",
                "title": "Academic warnings",
                "description": (
                    f"{student['Academic_Warning_Count']} "
                    "academic warning(s) have been recorded."
                ),
            }
        )

    # --------------------------------------------------------------
    # Suspension
    # --------------------------------------------------------------

    if student["Suspension_Count"] > 0:

        issues.append(
            {
                "code": "SUSPENSION_HISTORY",
                "severity": "Critical",
                "title": "Suspension history",
                "description": (
                    f"{student['Suspension_Count']} "
                    "suspension record(s) have been recorded."
                ),
            }
        )

    # --------------------------------------------------------------
    # OJT Credit Gap
    # --------------------------------------------------------------

    if metrics["OJT_Credit_Gap"] > 0:

        issues.append(
            {
                "code": "OJT_CREDIT_GAP",
                "severity": "High",
                "title": "OJT credit gap",
                "description": (
                    f"{metrics['OJT_Credit_Gap']} credit(s) "
                    "are still needed to reach the OJT credit threshold."
                ),
            }
        )

    # --------------------------------------------------------------
    # OJT Planning
    # --------------------------------------------------------------

    if metrics["OJT_Semester_Gap"] == 0:

        if metrics["OJT_Credit_Gap"] > 0:

            issues.append(
                {
                    "code": "URGENT_OJT_PLANNING",
                    "severity": "Critical",
                    "title": "OJT is approaching immediately",
                    "description": (
                        "The planned OJT semester has been reached "
                        "while the student still has an OJT credit gap."
                    ),
                }
            )

    elif metrics["OJT_Semester_Gap"] == 1:

        issues.append(
            {
                "code": "OJT_APPROACHING",
                "severity": "High",
                "title": "OJT is approaching",
                "description": ("The planned OJT semester is only one semester away."),
            }
        )

    # --------------------------------------------------------------
    # ML Risk
    # --------------------------------------------------------------

    if risk_level == "Critical":

        issues.append(
            {
                "code": "ML_CRITICAL_RISK",
                "severity": "Critical",
                "title": "Critical predicted OJT delay risk",
                "description": (
                    "The machine learning model predicts a very high "
                    "probability of OJT delay."
                ),
            }
        )

    elif risk_level == "High":

        issues.append(
            {
                "code": "ML_HIGH_RISK",
                "severity": "High",
                "title": "High predicted OJT delay risk",
                "description": (
                    "The machine learning model predicts a high "
                    "probability of OJT delay."
                ),
            }
        )

    elif risk_level == "Medium":

        issues.append(
            {
                "code": "ML_MEDIUM_RISK",
                "severity": "Medium",
                "title": "Moderate predicted OJT delay risk",
                "description": (
                    "The machine learning model indicates a moderate "
                    "probability of OJT delay."
                ),
            }
        )

    return issues


# ------------------------------------------------------------------
# Generate Recommendations
# ------------------------------------------------------------------


def generate_recommendations(student, metrics, issues, risk_level):
    """
    Generate recommendations based on current student conditions.
    """

    recommendations = []

    issue_codes = {issue["code"] for issue in issues}

    # --------------------------------------------------------------
    # GPA Recommendation
    # --------------------------------------------------------------

    if "LOW_GPA" in issue_codes:

        recommendations.append(
            {
                "priority": "High",
                "area": "Academic Performance",
                "action": (
                    "Prioritize improving GPA by focusing on "
                    "core and difficult courses."
                ),
            }
        )

    elif "MODERATE_GPA" in issue_codes:

        recommendations.append(
            {
                "priority": "Medium",
                "area": "Academic Performance",
                "action": (
                    "Maintain consistent study performance and "
                    "prioritize courses with higher academic impact."
                ),
            }
        )

    # --------------------------------------------------------------
    # Failed Courses
    # --------------------------------------------------------------

    if "FAILED_COURSES" in issue_codes:

        recommendations.append(
            {
                "priority": "High",
                "area": "Failed Courses",
                "action": (
                    "Prioritize retaking failed courses to prevent "
                    "further delay in academic progress."
                ),
            }
        )

    # --------------------------------------------------------------
    # Retakes
    # --------------------------------------------------------------

    if "RETAKE_COURSES" in issue_codes:

        recommendations.append(
            {
                "priority": "Medium",
                "area": "Course Planning",
                "action": (
                    "Include retake courses in the semester plan "
                    "without creating an excessive course load."
                ),
            }
        )

    # --------------------------------------------------------------
    # Prerequisites
    # --------------------------------------------------------------

    if "MISSING_PREREQUISITES" in issue_codes:

        recommendations.append(
            {
                "priority": "High",
                "area": "Prerequisites",
                "action": (
                    "Complete missing prerequisite courses as early "
                    "as possible before OJT registration."
                ),
            }
        )

    # --------------------------------------------------------------
    # Academic Warning
    # --------------------------------------------------------------

    if "ACADEMIC_WARNING" in issue_codes:

        recommendations.append(
            {
                "priority": "High",
                "area": "Academic Stability",
                "action": (
                    "Monitor academic performance closely and avoid "
                    "additional academic warnings."
                ),
            }
        )

    # --------------------------------------------------------------
    # Suspension
    # --------------------------------------------------------------

    if "SUSPENSION_HISTORY" in issue_codes:

        recommendations.append(
            {
                "priority": "Critical",
                "area": "Academic Stability",
                "action": (
                    "Review academic progress carefully and consult "
                    "academic support before planning OJT."
                ),
            }
        )

    # --------------------------------------------------------------
    # Credit Gap
    # --------------------------------------------------------------

    if metrics["OJT_Credit_Gap"] > 0:

        recommendations.append(
            {
                "priority": "High",
                "area": "OJT Credits",
                "action": (
                    f"Plan to complete the remaining "
                    f"{metrics['OJT_Credit_Gap']} OJT-related "
                    "credit gap before the planned OJT semester."
                ),
            }
        )

    # --------------------------------------------------------------
    # Required Credit Pace
    # --------------------------------------------------------------

    if (
        metrics["Required_Average_Credits_Per_Semester"] > 0
        and metrics["OJT_Semester_Gap"] > 0
    ):

        recommendations.append(
            {
                "priority": "Medium",
                "area": "Semester Planning",
                "action": (
                    f"Target approximately "
                    f"{metrics['Required_Average_Credits_Per_Semester']:.1f} "
                    "credits per remaining semester to close the "
                    "current OJT credit gap."
                ),
            }
        )

    # --------------------------------------------------------------
    # High Risk
    # --------------------------------------------------------------

    if risk_level in ["High", "Critical"]:

        recommendations.append(
            {
                "priority": "High",
                "area": "OJT Risk Management",
                "action": (
                    "Review the current OJT plan and monitor academic "
                    "progress every semester because the predicted "
                    "delay risk is elevated."
                ),
            }
        )

    # --------------------------------------------------------------
    # Low Risk
    # --------------------------------------------------------------

    elif risk_level == "Low":

        recommendations.append(
            {
                "priority": "Low",
                "area": "OJT Preparation",
                "action": (
                    "Maintain the current academic progress and "
                    "continue monitoring OJT readiness."
                ),
            }
        )

    return recommendations


# ------------------------------------------------------------------
# Generate Roadmap
# ------------------------------------------------------------------


def generate_roadmap(student, metrics, recommendations):
    """
    Generate a simple actionable OJT roadmap.
    """

    roadmap = []

    # --------------------------------------------------------------
    # Immediate Actions
    # --------------------------------------------------------------

    immediate_actions = []

    if student["Missing_Prerequisite_Courses"] > 0:

        immediate_actions.append("Review and prioritize missing prerequisite courses.")

    if student["Failed_Courses"] > 0:

        immediate_actions.append("Create a recovery plan for failed courses.")

    if student["Academic_Warning_Count"] > 0:

        immediate_actions.append("Monitor academic performance and warning status.")

    if not immediate_actions:

        immediate_actions.append("Maintain current academic performance.")

    roadmap.append({"stage": "Immediate", "actions": immediate_actions})

    # --------------------------------------------------------------
    # Short Term
    # --------------------------------------------------------------

    short_term_actions = []

    if metrics["OJT_Credit_Gap"] > 0:

        short_term_actions.append(
            f"Complete credits needed to reduce the "
            f"{metrics['OJT_Credit_Gap']}-credit OJT gap."
        )

    short_term_actions.append(
        "Monitor GPA, failed courses, and prerequisite completion."
    )

    roadmap.append({"stage": "Short Term", "actions": short_term_actions})

    # --------------------------------------------------------------
    # Before OJT
    # --------------------------------------------------------------

    before_ojt_actions = [
        "Verify academic requirements before OJT registration.",
        "Confirm prerequisite completion.",
        "Review planned OJT semester and remaining credits.",
    ]

    roadmap.append({"stage": "Before OJT", "actions": before_ojt_actions})

    # --------------------------------------------------------------
    # Monitoring
    # --------------------------------------------------------------

    roadmap.append(
        {
            "stage": "Continuous Monitoring",
            "actions": [
                "Update academic data after each semester.",
                "Run the prediction again when academic status changes.",
                "Review recommendations before final OJT registration.",
            ],
        }
    )

    return roadmap


# ------------------------------------------------------------------
# Main Recommendation Function
# ------------------------------------------------------------------


def generate_recommendation(student, prediction_result):
    """
    Main Recommendation Engine function.

    Parameters:
        student: current student information
        prediction_result: output from Prediction Pipeline

    Returns:
        Complete recommendation dictionary.
    """

    # --------------------------------------------------------------
    # Validate Inputs
    # --------------------------------------------------------------

    validate_student_data(student)

    if not isinstance(prediction_result, dict):
        raise ValueError("prediction_result must be a dictionary.")

    required_prediction_fields = [
        "prediction_class",
        "delay_probability",
        "no_delay_probability",
    ]

    for field in required_prediction_fields:

        if field not in prediction_result:
            raise ValueError(f"Missing prediction field: {field}")

    prediction_class = prediction_result["prediction_class"]

    delay_probability = prediction_result["delay_probability"]

    no_delay_probability = prediction_result["no_delay_probability"]

    # --------------------------------------------------------------
    # Prediction Validation
    # --------------------------------------------------------------

    if prediction_class not in [0, 1]:

        raise ValueError("prediction_class must be 0 or 1.")

    if not 0 <= delay_probability <= 1:

        raise ValueError("delay_probability must be between 0 and 1.")

    if not 0 <= no_delay_probability <= 1:

        raise ValueError("no_delay_probability must be between 0 and 1.")

    # --------------------------------------------------------------
    # Risk
    # --------------------------------------------------------------

    prediction = interpret_prediction(prediction_class, delay_probability)

    risk_level = prediction["risk_level"]

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    metrics = calculate_current_metrics(student)

    # --------------------------------------------------------------
    # Identify Issues
    # --------------------------------------------------------------

    issues = identify_issues(student, metrics, risk_level)

    # --------------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------------

    recommendations = generate_recommendations(student, metrics, issues, risk_level)

    # --------------------------------------------------------------
    # Roadmap
    # --------------------------------------------------------------

    roadmap = generate_roadmap(student, metrics, recommendations)

    # --------------------------------------------------------------
    # Final Result
    # --------------------------------------------------------------

    result = {
        "prediction": prediction,
        "current_metrics": metrics,
        "issues": issues,
        "recommendations": recommendations,
        "roadmap": roadmap,
    }

    return result


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------


def main():

    print()
    print("=" * 70)
    print("OJT AI PROJECT - RECOMMENDATION ENGINE V2")
    print("=" * 70)

    # --------------------------------------------------------------
    # Sample Student
    # --------------------------------------------------------------

    sample_student = {
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

    # --------------------------------------------------------------
    # Prediction Result
    #
    # This matches the current prediction pipeline result.
    # --------------------------------------------------------------

    prediction_result = {
        "prediction_class": 1,
        "prediction_label": "OJT Delay",
        "delay_probability": 0.7590,
        "no_delay_probability": 0.2410,
    }

    # --------------------------------------------------------------
    # Generate Recommendation
    # --------------------------------------------------------------

    result = generate_recommendation(sample_student, prediction_result)

    # --------------------------------------------------------------
    # Display Result
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOMMENDATION RESULT")
    print("=" * 70)

    print()
    print("Prediction")
    print("-" * 70)

    print(f"Label           : " f"{result['prediction']['prediction_label']}")

    print(
        f"Delay Probability: " f"{result['prediction']['delay_probability'] * 100:.2f}%"
    )

    print(f"Risk Level      : " f"{result['prediction']['risk_level']}")

    print()
    print("Current Metrics")
    print("-" * 70)

    for key, value in result["current_metrics"].items():

        print(f"{key:<45}: {value}")

    print()
    print("Identified Issues")
    print("-" * 70)

    for issue in result["issues"]:

        print(f"[{issue['severity']}] " f"{issue['title']}")

        print(f"  {issue['description']}")

    print()
    print("Recommendations")
    print("-" * 70)

    for index, recommendation in enumerate(result["recommendations"], start=1):

        print(
            f"{index}. " f"[{recommendation['priority']}] " f"{recommendation['area']}"
        )

        print(f"   {recommendation['action']}")

    print()
    print("OJT Roadmap")
    print("-" * 70)

    for stage in result["roadmap"]:

        print()
        print(stage["stage"])

        for action in stage["actions"]:

            print(f"  - {action}")

    print()
    print("=" * 70)
    print("RECOMMENDATION ENGINE: SUCCESS")
    print("=" * 70)


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
