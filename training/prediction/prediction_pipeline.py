import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# PROJECT PATHS
# ==========================================================

from config.paths import MODEL_DIR

# ==========================================================
# ARTIFACT PATHS
# ==========================================================

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

PREPROCESSING_PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")

BEST_MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "best_model_selection.json")

PREPROCESSING_METADATA_PATH = os.path.join(
    MODEL_DIR, "preprocessing_pipeline_metadata.json"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TARGET_COLUMN = "OJT_Delay_Outcome"

EXPECTED_INPUT_FEATURE_COUNT = 23
EXPECTED_ENCODED_FEATURE_COUNT = 38


# ==========================================================
# OFFICIAL MODEL FEATURES
# ==========================================================

MODEL_FEATURES = [
    "Student_Profile",
    "Current_Semester",
    "GPA_Cumulative",
    "Total_Credits",
    "Credits_Completed",
    "Credits_Remaining",
    "Completion_Rate",
    "Average_Credits_Per_Semester",
    "Remaining_To_OJT",
    "Failed_Courses",
    "Retake_Count",
    "Missing_Prerequisite_Courses",
    "Academic_Warning_Count",
    "Suspension_Count",
    "Planned_OJT_Semester",
    "Credit_Progress_Category",
    "OJT_Credit_Gap",
    "OJT_Credit_Progress",
    "OJT_Eligibility_Gap_Category",
    "OJT_Semester_Gap",
    "OJT_Planning_Status",
    "Required_Average_Credits_Per_Semester",
    "Credit_Completion_Efficiency",
]


# ==========================================================
# RAW INPUT FEATURES
# ==========================================================

RAW_INPUT_FEATURES = [
    "Student_Profile",
    "Current_Semester",
    "GPA_Cumulative",
    "Total_Credits",
    "Credits_Completed",
    "Credits_Remaining",
    "Completion_Rate",
    "Average_Credits_Per_Semester",
    "Remaining_To_OJT",
    "Failed_Courses",
    "Retake_Count",
    "Missing_Prerequisite_Courses",
    "Academic_Warning_Count",
    "Suspension_Count",
    "Planned_OJT_Semester",
]


# ==========================================================
# CATEGORICAL FEATURES
# ==========================================================

CATEGORICAL_FEATURES = [
    "Student_Profile",
    "Credit_Progress_Category",
    "OJT_Eligibility_Gap_Category",
    "OJT_Planning_Status",
]


# ==========================================================
# NUMERICAL FEATURES
# ==========================================================

NUMERICAL_FEATURES = [
    feature for feature in MODEL_FEATURES if feature not in CATEGORICAL_FEATURES
]


# ==========================================================
# PROFILE VALUES
# ==========================================================

ALLOWED_PROFILES = [
    "Excellent",
    "Good",
    "Average",
    "AtRisk",
    "Critical",
    "Recovery",
    "LateStarter",
]


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def safe_float(value, field_name):
    """
    Convert a value to float and validate it.
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric.")

    if pd.isna(result):
        raise ValueError(f"{field_name} cannot be missing.")

    return result


def safe_int(value, field_name):
    """
    Convert a value to integer and validate it.
    """

    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer.")

    return result


# ==========================================================
# VALIDATE RAW INPUT
# ==========================================================


def validate_raw_input(student_data):

    print("=" * 70)
    print("Validating Raw Student Input...")
    print("=" * 70)

    if not isinstance(student_data, dict):
        raise TypeError("student_data must be a dictionary.")

    # ------------------------------------------------------
    # Required fields
    # ------------------------------------------------------

    missing_fields = [
        field for field in RAW_INPUT_FEATURES if field not in student_data
    ]

    if missing_fields:
        raise ValueError("Missing required input fields: " f"{missing_fields}")

    print("Required Fields     : PASS")

    # ------------------------------------------------------
    # Student Profile
    # ------------------------------------------------------

    if student_data["Student_Profile"] not in ALLOWED_PROFILES:
        raise ValueError(
            "Invalid Student_Profile. " f"Allowed values: {ALLOWED_PROFILES}"
        )

    print("Student Profile     : PASS")

    # ------------------------------------------------------
    # Current Semester
    # ------------------------------------------------------

    current_semester = safe_int(student_data["Current_Semester"], "Current_Semester")

    if not 1 <= current_semester <= 8:
        raise ValueError("Current_Semester must be between 1 and 8.")

    print("Current Semester    : PASS")

    # ------------------------------------------------------
    # Planned OJT Semester
    # ------------------------------------------------------

    planned_ojt_semester = safe_int(
        student_data["Planned_OJT_Semester"], "Planned_OJT_Semester"
    )

    if not 1 <= planned_ojt_semester <= 8:
        raise ValueError("Planned_OJT_Semester must be between 1 and 8.")

    print("Planned OJT Semester: PASS")

    # ------------------------------------------------------
    # GPA
    # ------------------------------------------------------

    gpa = safe_float(student_data["GPA_Cumulative"], "GPA_Cumulative")

    if not 0 <= gpa <= 10:
        raise ValueError("GPA_Cumulative must be between 0 and 10.")

    print("GPA                 : PASS")

    # ------------------------------------------------------
    # Credits
    # ------------------------------------------------------

    total_credits = safe_float(student_data["Total_Credits"], "Total_Credits")

    credits_completed = safe_float(
        student_data["Credits_Completed"], "Credits_Completed"
    )

    credits_remaining = safe_float(
        student_data["Credits_Remaining"], "Credits_Remaining"
    )

    if total_credits <= 0:
        raise ValueError("Total_Credits must be greater than 0.")

    if credits_completed < 0:
        raise ValueError("Credits_Completed cannot be negative.")

    if credits_remaining < 0:
        raise ValueError("Credits_Remaining cannot be negative.")

    if credits_completed + credits_remaining > total_credits:
        raise ValueError(
            "Credits_Completed + Credits_Remaining " "cannot exceed Total_Credits."
        )

    print("Credit Information  : PASS")

    # ------------------------------------------------------
    # Completion Rate
    # ------------------------------------------------------

    completion_rate = safe_float(student_data["Completion_Rate"], "Completion_Rate")

    if not 0 <= completion_rate <= 1:
        raise ValueError("Completion_Rate must be between 0 and 1.")

    print("Completion Rate     : PASS")

    # ------------------------------------------------------
    # Average Credits
    # ------------------------------------------------------

    average_credits = safe_float(
        student_data["Average_Credits_Per_Semester"], "Average_Credits_Per_Semester"
    )

    if average_credits < 0:
        raise ValueError("Average_Credits_Per_Semester " "cannot be negative.")

    print("Average Credits     : PASS")

    # ------------------------------------------------------
    # Remaining To OJT
    # ------------------------------------------------------

    remaining_to_ojt = safe_float(student_data["Remaining_To_OJT"], "Remaining_To_OJT")

    if remaining_to_ojt < 0:
        raise ValueError("Remaining_To_OJT cannot be negative.")

    print("Remaining To OJT    : PASS")

    # ------------------------------------------------------
    # Count-based features
    # ------------------------------------------------------

    count_features = [
        "Failed_Courses",
        "Retake_Count",
        "Missing_Prerequisite_Courses",
        "Academic_Warning_Count",
        "Suspension_Count",
    ]

    for feature in count_features:

        value = safe_int(student_data[feature], feature)

        if value < 0:
            raise ValueError(f"{feature} cannot be negative.")

    print("Academic Indicators : PASS")

    print()
    print("Raw Input Validation : PASS")
    print()


# ==========================================================
# CREATE DATAFRAME
# ==========================================================


def create_raw_dataframe(student_data):

    data = {feature: student_data[feature] for feature in RAW_INPUT_FEATURES}

    df = pd.DataFrame([data])

    return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================


def engineer_features(df):
    """
    Generate the 8 engineered features required by
    the official model feature list.

    These features correspond to the Feature Engineering V2
    stage used during model training.
    """

    print("=" * 70)
    print("Generating Prediction Features...")
    print("=" * 70)

    # ------------------------------------------------------
    # 1. Credit Progress Category
    # ------------------------------------------------------

    completion_rate = df["Completion_Rate"].iloc[0]

    if completion_rate < 0.40:
        credit_progress_category = "Early"

    elif completion_rate < 0.70:
        credit_progress_category = "Developing"

    elif completion_rate < 0.90:
        credit_progress_category = "Advanced"

    else:
        credit_progress_category = "Near_Completion"

    df["Credit_Progress_Category"] = credit_progress_category

    # ------------------------------------------------------
    # 2. OJT Credit Gap
    # ------------------------------------------------------

    df["OJT_Credit_Gap"] = 100 - df["Credits_Completed"]

    # ------------------------------------------------------
    # 3. OJT Credit Progress
    # ------------------------------------------------------

    df["OJT_Credit_Progress"] = df["Credits_Completed"] / 100

    # ------------------------------------------------------
    # 4. OJT Eligibility Gap Category
    # ------------------------------------------------------

    ojt_credit_gap = df["OJT_Credit_Gap"].iloc[0]

    if ojt_credit_gap <= 0:
        eligibility_gap_category = "Eligible"

    elif ojt_credit_gap <= 10:
        eligibility_gap_category = "Near_Eligible"

    elif ojt_credit_gap <= 30:
        eligibility_gap_category = "Moderate_Gap"

    else:
        eligibility_gap_category = "Far_From_Eligible"

    df["OJT_Eligibility_Gap_Category"] = eligibility_gap_category

    # ------------------------------------------------------
    # 5. OJT Semester Gap
    # ------------------------------------------------------

    df["OJT_Semester_Gap"] = df["Planned_OJT_Semester"] - df["Current_Semester"]

    # ------------------------------------------------------
    # 6. OJT Planning Status
    # ------------------------------------------------------

    semester_gap = df["OJT_Semester_Gap"].iloc[0]

    if semester_gap <= 0:
        planning_status = "OJT_Ready_Stage"

    elif semester_gap == 1:
        planning_status = "Approaching_OJT"

    elif semester_gap <= 2:
        planning_status = "Moderate_Planning_Gap"

    else:
        planning_status = "Long_Term_Planning"

    df["OJT_Planning_Status"] = planning_status

    # ------------------------------------------------------
    # 7. Required Average Credits Per Semester
    # ------------------------------------------------------

    semester_gap_for_calculation = max(float(semester_gap), 1.0)

    df["Required_Average_Credits_Per_Semester"] = (
        df["OJT_Credit_Gap"] / semester_gap_for_calculation
    )

    # ------------------------------------------------------
    # 8. Credit Completion Efficiency
    # ------------------------------------------------------

    current_semester = df["Current_Semester"].iloc[0]

    if current_semester > 0:

        df["Credit_Completion_Efficiency"] = df["Credits_Completed"] / current_semester

    else:

        df["Credit_Completion_Efficiency"] = 0.0

    print("Engineered Features : 8")

    print()
    print("Feature Engineering : PASS")
    print()

    return df


# ==========================================================
# PREPARE MODEL INPUT
# ==========================================================


def prepare_model_input(df):

    print("=" * 70)
    print("Preparing Model Input...")
    print("=" * 70)

    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in df.columns
    ]

    if missing_features:
        raise ValueError("Missing model features: " f"{missing_features}")

    model_input = df[MODEL_FEATURES].copy()

    if model_input.shape[1] != (EXPECTED_INPUT_FEATURE_COUNT):
        raise ValueError(
            "Unexpected model input feature count: " f"{model_input.shape[1]}"
        )

    print(f"Model Input Features : " f"{model_input.shape[1]}")

    print()
    print("Model Input Preparation : PASS")
    print()

    return model_input


# ==========================================================
# LOAD ARTIFACTS
# ==========================================================


def load_artifacts():

    print("=" * 70)
    print("Loading Prediction Artifacts...")
    print("=" * 70)

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found:\n" f"{BEST_MODEL_PATH}")

    best_model = joblib.load(BEST_MODEL_PATH)

    print(f"Best Model : {BEST_MODEL_PATH}")

    # ------------------------------------------------------
    # Preprocessing Pipeline
    # ------------------------------------------------------

    if not os.path.exists(PREPROCESSING_PIPELINE_PATH):
        raise FileNotFoundError(
            "Preprocessing pipeline not found:\n" f"{PREPROCESSING_PIPELINE_PATH}"
        )

    preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)

    print(f"Preprocessor : " f"{PREPROCESSING_PIPELINE_PATH}")

    print()
    print("Prediction Artifacts Loading : PASS")
    print()

    return (best_model, preprocessing_pipeline)


# ==========================================================
# TRANSFORM PREDICTION INPUT
# ==========================================================


def transform_prediction_input(model_input, preprocessing_pipeline, best_model):
    print("=" * 70)
    print("Transforming Prediction Input...")
    print("=" * 70)
    encoded_array = preprocessing_pipeline.transform(model_input)

    encoded_feature_names = preprocessing_pipeline.get_feature_names_out()

    clean_feature_names = [
        name.replace("numerical__", "").replace("categorical__", "")
        for name in encoded_feature_names
    ]

    encoded_input = pd.DataFrame(
        encoded_array, columns=clean_feature_names, index=model_input.index
    )

    if encoded_input.shape[1] != 38:
        raise ValueError(
            f"Expected 38 encoded features, " f"but received {encoded_input.shape[1]}"
        )

    if not hasattr(best_model, "feature_names_in_"):
        raise ValueError(
            "Best model does not contain feature_names_in_. "
            "The model may not have been trained with named features."
        )

    model_feature_names = list(best_model.feature_names_in_)

    if set(encoded_input.columns) != set(model_feature_names):
        missing = set(model_feature_names) - set(encoded_input.columns)
        extra = set(encoded_input.columns) - set(model_feature_names)

        raise ValueError(
            f"Encoded feature mismatch.\n"
            f"Missing features: {sorted(missing)}\n"
            f"Extra features: {sorted(extra)}"
        )

    encoded_input = encoded_input[model_feature_names]

    if list(encoded_input.columns) != model_feature_names:
        raise ValueError("Encoded feature order does not match model feature order.")

    if encoded_input.isnull().any().any():
        raise ValueError("Encoded prediction input contains missing values.")

    if not np.isfinite(encoded_input.to_numpy()).all():
        raise ValueError("Encoded prediction input contains infinite values.")

    print(f"Encoded Features     : {encoded_input.shape[1]}")
    print("Feature Order        : MATCHED WITH MODEL")
    print("Prediction Transformation : PASS")

    return encoded_input


# ==========================================================
# MAKE PREDICTION
# ==========================================================


def make_prediction(best_model, encoded_input):

    print("=" * 70)
    print("Making OJT Delay Prediction...")
    print("=" * 70)

    prediction = best_model.predict(encoded_input)

    prediction_class = int(prediction[0])

    # ------------------------------------------------------
    # Probability
    # ------------------------------------------------------

    if hasattr(best_model, "predict_proba"):

        probabilities = best_model.predict_proba(encoded_input)[0]

        probability_no_delay = float(probabilities[0])

        probability_delay = float(probabilities[1])

    else:

        probability_no_delay = None
        probability_delay = None

    # ------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------

    if prediction_class == 1:

        prediction_label = "OJT Delay"

    else:

        prediction_label = "No OJT Delay"

    print(f"Prediction Class : " f"{prediction_class}")

    print(f"Prediction Label : " f"{prediction_label}")

    if probability_delay is not None:

        print(f"Delay Probability : " f"{probability_delay:.4f}")

        print(f"No-Delay Probability : " f"{probability_no_delay:.4f}")

    print()
    print("Prediction : PASS")
    print()

    return {
        "prediction_class": prediction_class,
        "prediction_label": prediction_label,
        "delay_probability": probability_delay,
        "no_delay_probability": probability_no_delay,
    }


# ==========================================================
# SAVE PREDICTION RESULT
# ==========================================================


def save_prediction_result(student_data, result):

    output = {
        "project": "OJT AI Project",
        "target": TARGET_COLUMN,
        "prediction": result,
        "input": student_data,
        "model": "Random Forest",
        "artifact": BEST_MODEL_PATH,
        "dataset_type": ("synthetic_development_dataset"),
        "note": (
            "This prediction is generated by a model "
            "trained on synthetic development data and "
            "should not be interpreted as validated "
            "real-world OJT prediction performance."
        ),
    }

    output_path = os.path.join(MODEL_DIR, "latest_prediction.json")

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"Prediction Result : {output_path}")

    return output_path


# ==========================================================
# COMPLETE PIPELINE
# ==========================================================


def predict_student(student_data):
    """
    Complete prediction pipeline.

    Input:
        Dictionary containing raw student data.

    Output:
        Prediction result dictionary.
    """

    # 1. Validate raw input
    validate_raw_input(student_data)

    # 2. Create raw dataframe
    raw_df = create_raw_dataframe(student_data)

    # 3. Feature engineering
    feature_df = engineer_features(raw_df)

    # 4. Prepare model input
    model_input = prepare_model_input(feature_df)

    # 5. Load model + preprocessing
    best_model, preprocessing_pipeline = load_artifacts()

    # 6. Transform
    encoded_input = transform_prediction_input(model_input, preprocessing_pipeline, best_model)

    # 7. Predict
    result = make_prediction(best_model, encoded_input)

    # 8. Save
    save_prediction_result(student_data, result)

    return result


# ==========================================================
# TEST / DEMO
# ==========================================================


def main():

    print()

    print("=" * 70)
    print("OJT AI PROJECT - PREDICTION PIPELINE V2")
    print("=" * 70)

    print()

    # ------------------------------------------------------
    # Demo Student
    # ------------------------------------------------------
    #
    # This is a development/test student only.
    # Values are used to verify the complete pipeline.
    #

    sample_student = {
        "Student_Profile": "Average",
        "Current_Semester": 5,
        "GPA_Cumulative": 6.8,
        "Total_Credits": 120,
        "Credits_Completed": 82,
        "Credits_Remaining": 38,
        "Completion_Rate": 82 / 120,
        "Average_Credits_Per_Semester": 16.4,
        "Remaining_To_OJT": 18,
        "Failed_Courses": 2,
        "Retake_Count": 1,
        "Missing_Prerequisite_Courses": 1,
        "Academic_Warning_Count": 0,
        "Suspension_Count": 0,
        "Planned_OJT_Semester": 6,
    }

    result = predict_student(sample_student)

    print("=" * 70)
    print("PREDICTION PIPELINE: SUCCESS")
    print("=" * 70)

    print()

    print("Final Prediction")
    print("----------------")
    print(f"Class : " f"{result['prediction_class']}")

    print(f"Label : " f"{result['prediction_label']}")

    if result["delay_probability"] is not None:

        print(f"Delay Probability : " f"{result['delay_probability']:.2%}")

        print(f"No-Delay Probability : " f"{result['no_delay_probability']:.2%}")

    print()

    print("Next Step:")
    print("Build Prediction Test.")

    print("=" * 70)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()
