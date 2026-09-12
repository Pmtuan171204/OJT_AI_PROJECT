import os
import sys
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

from config.paths import X_TRAIN_PATH, X_TEST_PATH, MODEL_DIR

# ==========================================================
# OUTPUT PATHS
# ==========================================================

PREPROCESSING_PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")

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
# FORBIDDEN / LEAKAGE FEATURES
# ==========================================================

FORBIDDEN_FEATURES = [
    "MSSV",
    "Risk_Score",
    "Risk_Level",
    "OJT_Delay_Risk",
    "OJT_Eligible",
    "OJT_Readiness",
    "AI_Recommendation",
    "Future_Credits_At_OJT",
    "Future_Failed_Courses",
    "Future_Missing_Prerequisites",
    "Future_Academic_Warning",
    "Future_OJT_Eligible",
    "OJT_Delay_Outcome",
]


# ==========================================================
# LOAD DATA
# ==========================================================


def load_data():

    print("=" * 70)
    print("Loading Training Data...")
    print("=" * 70)

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(f"X_train not found:\n{X_TRAIN_PATH}")

    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(f"X_test not found:\n{X_TEST_PATH}")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    print(f"X_train Rows : {len(X_train)}")
    print(f"X_test Rows  : {len(X_test)}")
    print()

    return X_train, X_test


# ==========================================================
# VALIDATE INPUT DATA
# ==========================================================


def validate_input_data(X_train, X_test):

    print("=" * 70)
    print("Validating Input Data...")
    print("=" * 70)

    # ------------------------------------------------------
    # 1. Feature count
    # ------------------------------------------------------

    if X_train.shape[1] != EXPECTED_INPUT_FEATURE_COUNT:
        raise ValueError("Unexpected X_train feature count: " f"{X_train.shape[1]}")

    if X_test.shape[1] != EXPECTED_INPUT_FEATURE_COUNT:
        raise ValueError("Unexpected X_test feature count: " f"{X_test.shape[1]}")

    print("Feature Count       : PASS")

    # ------------------------------------------------------
    # 2. Feature names
    # ------------------------------------------------------

    if list(X_train.columns) != MODEL_FEATURES:
        raise ValueError(
            "X_train feature names/order do not match "
            "the official model feature list."
        )

    if list(X_test.columns) != MODEL_FEATURES:
        raise ValueError(
            "X_test feature names/order do not match "
            "the official model feature list."
        )

    print("Feature Names       : PASS")

    # ------------------------------------------------------
    # 3. Target separation
    # ------------------------------------------------------

    if TARGET_COLUMN in X_train.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' found in X_train.")

    if TARGET_COLUMN in X_test.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' found in X_test.")

    print("Target Separation   : PASS")

    # ------------------------------------------------------
    # 4. Forbidden features
    # ------------------------------------------------------

    forbidden_found = [
        feature for feature in FORBIDDEN_FEATURES if feature in X_train.columns
    ]

    if forbidden_found:
        raise ValueError(
            "Forbidden/leakage features found in X_train: " f"{forbidden_found}"
        )

    forbidden_found_test = [
        feature for feature in FORBIDDEN_FEATURES if feature in X_test.columns
    ]

    if forbidden_found_test:
        raise ValueError(
            "Forbidden/leakage features found in X_test: " f"{forbidden_found_test}"
        )

    print("Leakage Check       : PASS")

    # ------------------------------------------------------
    # 5. Missing values
    # ------------------------------------------------------

    if X_train.isnull().any().any():
        raise ValueError("Missing values found in X_train.")

    if X_test.isnull().any().any():
        raise ValueError("Missing values found in X_test.")

    print("Missing Values      : PASS")

    # ------------------------------------------------------
    # 6. Numerical columns
    # ------------------------------------------------------

    for column in NUMERICAL_FEATURES:

        if not pd.api.types.is_numeric_dtype(X_train[column]):
            raise ValueError(f"Numerical feature '{column}' " "is not numeric.")

        if not pd.api.types.is_numeric_dtype(X_test[column]):
            raise ValueError(
                f"Numerical feature '{column}' " "is not numeric in X_test."
            )

    print("Numerical Features  : PASS")

    # ------------------------------------------------------
    # 7. Categorical columns
    # ------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        if column not in X_train.columns:
            raise ValueError(f"Categorical feature '{column}' " "missing from X_train.")

        if column not in X_test.columns:
            raise ValueError(f"Categorical feature '{column}' " "missing from X_test.")

    print("Categorical Features: PASS")

    print()
    print("Input Data Validation : PASS")
    print()


# ==========================================================
# BUILD PREPROCESSOR
# ==========================================================


def build_preprocessor():

    print("=" * 70)
    print("Building Preprocessing Pipeline...")
    print("=" * 70)

    print(f"Categorical Features : {len(CATEGORICAL_FEATURES)}")

    print(f"Numerical Features   : {len(NUMERICAL_FEATURES)}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    print()
    print("Preprocessor Created : PASS")
    print()

    return preprocessor


# ==========================================================
# FIT ON TRAINING DATA
# ==========================================================


def fit_preprocessor(preprocessor, X_train):

    print("=" * 70)
    print("Fitting Preprocessor on Training Data...")
    print("=" * 70)

    preprocessor.fit(X_train)

    print("Fit Dataset : X_train")
    print("X_test      : NOT USED FOR FITTING")
    print()
    print("Preprocessor Fitting : PASS")
    print()

    return preprocessor


# ==========================================================
# TRANSFORM DATA
# ==========================================================


def transform_data(preprocessor, X_train, X_test):

    print("=" * 70)
    print("Transforming Training and Testing Data...")
    print("=" * 70)

    X_train_encoded = preprocessor.transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    print(f"X_train Encoded Shape : {X_train_encoded.shape}")

    print(f"X_test Encoded Shape  : {X_test_encoded.shape}")

    print()

    return X_train_encoded, X_test_encoded


# ==========================================================
# VALIDATE TRANSFORMED DATA
# ==========================================================


def validate_transformed_data(X_train_encoded, X_test_encoded):

    print("=" * 70)
    print("Validating Transformed Data...")
    print("=" * 70)

    # ------------------------------------------------------
    # Shape
    # ------------------------------------------------------

    if X_train_encoded.shape[1] != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected encoded feature count in X_train: "
            f"{X_train_encoded.shape[1]}"
        )

    if X_test_encoded.shape[1] != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected encoded feature count in X_test: " f"{X_test_encoded.shape[1]}"
        )

    print("Encoded Feature Count : PASS")

    # ------------------------------------------------------
    # Missing values
    # ------------------------------------------------------

    if pd.isna(X_train_encoded).any():
        raise ValueError("Missing values found in encoded X_train.")

    if pd.isna(X_test_encoded).any():
        raise ValueError("Missing values found in encoded X_test.")

    print("Missing Values        : PASS")

    # ------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(X_train_encoded.dtype):
        raise ValueError("Encoded X_train is not numeric.")

    if not pd.api.types.is_numeric_dtype(X_test_encoded.dtype):
        raise ValueError("Encoded X_test is not numeric.")

    print("Numeric Encoded Data  : PASS")

    # ------------------------------------------------------
    # Infinite values
    # ------------------------------------------------------

    if (
        not pd.DataFrame(X_train_encoded)
        .apply(
            lambda column: column.map(lambda value: abs(value) == float("inf")).any()
        )
        .any()
    ):

        train_inf_ok = True

    else:
        train_inf_ok = False

    if (
        not pd.DataFrame(X_test_encoded)
        .apply(
            lambda column: column.map(lambda value: abs(value) == float("inf")).any()
        )
        .any()
    ):

        test_inf_ok = True

    else:
        test_inf_ok = False

    if not train_inf_ok or not test_inf_ok:
        raise ValueError("Infinite values found in transformed data.")

    print("Infinite Values       : PASS")

    print()
    print("Transformed Data Validation : PASS")
    print()


# ==========================================================
# SAVE PIPELINE
# ==========================================================


def save_pipeline(preprocessor):

    print("=" * 70)
    print("Saving Preprocessing Pipeline...")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSING_PIPELINE_PATH)

    if not os.path.exists(PREPROCESSING_PIPELINE_PATH):
        raise RuntimeError("Preprocessing pipeline file was not created.")

    print(f"Pipeline : {PREPROCESSING_PIPELINE_PATH}")

    print()
    print("Preprocessing Pipeline Artifact : PASS")
    print()


# ==========================================================
# GET FEATURE NAMES
# ==========================================================


def get_encoded_feature_names(preprocessor):

    encoded_feature_names = preprocessor.get_feature_names_out()

    return [
        name.replace("categorical__", "").replace("numerical__", "")
        for name in encoded_feature_names
    ]


# ==========================================================
# SAVE METADATA
# ==========================================================


def save_metadata(preprocessor, encoded_feature_names):

    print("=" * 70)
    print("Saving Preprocessing Metadata...")
    print("=" * 70)

    metadata = {
        "project": "OJT AI Project",
        "pipeline_version": "2.0",
        "target": TARGET_COLUMN,
        "input_feature_count": len(MODEL_FEATURES),
        "encoded_feature_count": len(encoded_feature_names),
        "categorical_features": (CATEGORICAL_FEATURES),
        "numerical_features": (NUMERICAL_FEATURES),
        "input_features": MODEL_FEATURES,
        "encoded_features": (encoded_feature_names),
        "encoder": {
            "type": "OneHotEncoder",
            "handle_unknown": "ignore",
            "fit_on": "X_train",
        },
        "scaler": {"type": "StandardScaler", "fit_on": "X_train"},
        "leakage_policy": {
            "target_excluded": True,
            "risk_features_excluded": True,
            "business_rule_features_excluded": True,
            "future_outcome_features_excluded": True,
            "student_id_excluded": True,
        },
        "dataset_type": ("synthetic_development_dataset"),
        "note": (
            "This preprocessing artifact is fitted "
            "only on X_train and must be reused for "
            "future prediction data. It must not be "
            "refitted on individual prediction inputs."
        ),
    }

    with open(PREPROCESSING_METADATA_PATH, "w", encoding="utf-8") as file:

        json.dump(metadata, file, indent=4, ensure_ascii=False)

    if not os.path.exists(PREPROCESSING_METADATA_PATH):
        raise RuntimeError("Preprocessing metadata was not created.")

    print("Metadata : PASS")
    print(f"Metadata File : {PREPROCESSING_METADATA_PATH}")

    print()


# ==========================================================
# MAIN
# ==========================================================


def main():

    print()

    print("=" * 70)
    print("OJT AI PROJECT - PREPROCESSING PIPELINE V2")
    print("=" * 70)

    print()

    # 1. Load
    X_train, X_test = load_data()

    # 2. Validate
    validate_input_data(X_train, X_test)

    # 3. Build
    preprocessor = build_preprocessor()

    # 4. Fit ONLY on X_train
    preprocessor = fit_preprocessor(preprocessor, X_train)

    # 5. Transform
    X_train_encoded, X_test_encoded = transform_data(preprocessor, X_train, X_test)

    # 6. Validate transformed data
    validate_transformed_data(X_train_encoded, X_test_encoded)

    # 7. Get encoded feature names
    encoded_feature_names = get_encoded_feature_names(preprocessor)

    # 8. Validate feature names
    if len(encoded_feature_names) != (EXPECTED_ENCODED_FEATURE_COUNT):
        raise ValueError(
            "Unexpected number of encoded feature names: "
            f"{len(encoded_feature_names)}"
        )

    # 9. Save pipeline
    save_pipeline(preprocessor)

    # 10. Save metadata
    save_metadata(preprocessor, encoded_feature_names)

    print("=" * 70)
    print("PREPROCESSING PIPELINE: SUCCESS")
    print("=" * 70)

    print()
    print(f"Input Features   : " f"{EXPECTED_INPUT_FEATURE_COUNT}")

    print(f"Encoded Features : " f"{EXPECTED_ENCODED_FEATURE_COUNT}")

    print()
    print("Generated Files:")

    print(f"1. {PREPROCESSING_PIPELINE_PATH}")

    print(f"2. {PREPROCESSING_METADATA_PATH}")

    print()
    print("Preprocessing is now deployment-ready.")

    print()
    print("Next Step:")
    print("Build Prediction Pipeline.")

    print("=" * 70)


if __name__ == "__main__":
    main()
