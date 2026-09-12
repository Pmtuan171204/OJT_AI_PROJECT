# ==========================================================
# 1. IMPORTS
# ==========================================================

import os
import sys
import json
import shutil

# ==========================================================
# 2. PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# 3. PROJECT CONFIG
# ==========================================================

from config.paths import (
    MODEL_DIR,
    EVALUATION_REPORT_DIR,
    LOGISTIC_REGRESSION_MODEL,
    RANDOM_FOREST_MODEL,
)

# ==========================================================
# 4. CONFIGURATION
# ==========================================================

COMPARISON_JSON = os.path.join(EVALUATION_REPORT_DIR, "model_comparison.json")

BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")

BEST_MODEL_METADATA_FILE = os.path.join(MODEL_DIR, "best_model_selection.json")

TARGET_COLUMN = "OJT_Delay_Outcome"

SELECTION_METRIC = "Test F1"


# ==========================================================
# 5. MODEL REGISTRY
# ==========================================================

MODEL_REGISTRY = {
    "Logistic Regression": {
        "algorithm": "LogisticRegression",
        "model_path": LOGISTIC_REGRESSION_MODEL,
    },
    "Random Forest": {
        "algorithm": "RandomForestClassifier",
        "model_path": RANDOM_FOREST_MODEL,
    },
}


# ==========================================================
# 6. LOAD COMPARISON RESULT
# ==========================================================


def load_comparison_result():
    """
    Load the official model comparison result.
    """

    print("=" * 70)
    print("Loading Model Comparison Result...")
    print("=" * 70)

    if not os.path.exists(COMPARISON_JSON):
        raise FileNotFoundError(
            "Model comparison JSON not found:\n" f"{COMPARISON_JSON}"
        )

    with open(COMPARISON_JSON, "r", encoding="utf-8") as file:

        comparison_data = json.load(file)

    print(f"Comparison File : {COMPARISON_JSON}")

    print("Comparison Loading : PASS")

    print()

    return comparison_data


# ==========================================================
# 7. VALIDATE COMPARISON RESULT
# ==========================================================


def validate_comparison_result(comparison_data):
    """
    Validate the model comparison result before
    selecting a model.
    """

    print("=" * 70)
    print("Validating Model Comparison Result...")
    print("=" * 70)

    # ------------------------------------------------------
    # Required fields
    # ------------------------------------------------------

    required_fields = [
        "project",
        "target",
        "selection_metric",
        "best_model_candidate",
        "models",
    ]

    for field in required_fields:

        if field not in comparison_data:

            raise ValueError(f"Missing comparison field: {field}")

    # ------------------------------------------------------
    # Target validation
    # ------------------------------------------------------

    if comparison_data["target"] != TARGET_COLUMN:

        raise ValueError("Unexpected target column: " f"{comparison_data['target']}")

    print("Target              : PASS")

    # ------------------------------------------------------
    # Selection metric
    # ------------------------------------------------------

    if comparison_data["selection_metric"] != SELECTION_METRIC:

        raise ValueError(
            "Unexpected selection metric: " f"{comparison_data['selection_metric']}"
        )

    print("Selection Metric    : PASS")

    # ------------------------------------------------------
    # Model list
    # ------------------------------------------------------

    models = comparison_data["models"]

    if not isinstance(models, list):

        raise ValueError("'models' must be a list.")

    if len(models) < 2:

        raise ValueError("At least two models are required " "for model selection.")

    print("Model Count         : PASS")

    # ------------------------------------------------------
    # Validate model records
    # ------------------------------------------------------

    for model in models:

        if "Model" not in model:

            raise ValueError("Model name missing from comparison result.")

        if "Test F1" not in model:

            raise ValueError("Test F1 missing from comparison result.")

    print("Model Records       : PASS")
    print()

    print("Comparison Validation : PASS")

    print()


# ==========================================================
# 8. VERIFY BEST MODEL CANDIDATE
# ==========================================================


def verify_best_model_candidate(comparison_data):
    """
    Independently verify that the selected candidate
    actually has the highest Test F1.
    """

    print("=" * 70)
    print("Verifying Best Model Candidate...")
    print("=" * 70)

    models = comparison_data["models"]

    # ------------------------------------------------------
    # Calculate highest Test F1
    # ------------------------------------------------------

    highest_f1_model = max(models, key=lambda model: model["Test F1"])

    calculated_best_model = highest_f1_model["Model"]

    reported_best_model = comparison_data["best_model_candidate"]

    print(f"Reported Best Model : " f"{reported_best_model}")

    print(f"Calculated Best     : " f"{calculated_best_model}")

    # ------------------------------------------------------
    # Compare
    # ------------------------------------------------------

    if reported_best_model != calculated_best_model:

        raise ValueError(
            "Best model candidate does not match " "the model with the highest Test F1."
        )

    print()
    print("Best Model Verification : PASS")

    print()

    return calculated_best_model


# ==========================================================
# 9. VERIFY MODEL FILE
# ==========================================================


def verify_model_file(selected_model):
    """
    Verify that the selected trained model file exists.
    """

    print("=" * 70)
    print("Verifying Selected Model File...")
    print("=" * 70)

    if selected_model not in MODEL_REGISTRY:

        raise ValueError(f"Model '{selected_model}' " "is not registered.")

    model_info = MODEL_REGISTRY[selected_model]

    model_path = model_info["model_path"]

    print(f"Selected Model : {selected_model}")

    print(f"Algorithm      : " f"{model_info['algorithm']}")

    print(f"Model Path     : {model_path}")

    if not os.path.exists(model_path):

        raise FileNotFoundError("Selected model file does not exist:\n" f"{model_path}")

    print()
    print("Selected Model File : PASS")

    print()

    return model_info


# ==========================================================
# 10. COPY SELECTED MODEL
# ==========================================================


def create_best_model(selected_model, model_info):
    """
    Copy the selected trained model to the standard
    best_model.pkl location.
    """

    print("=" * 70)
    print("Creating Best Model Artifact...")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    source_path = model_info["model_path"]

    shutil.copy2(source_path, BEST_MODEL_FILE)

    if not os.path.exists(BEST_MODEL_FILE):

        raise RuntimeError("Best model file was not created.")

    print(f"Source Model : {source_path}")

    print(f"Best Model   : {BEST_MODEL_FILE}")

    print()
    print("Best Model Artifact : PASS")

    print()


# ==========================================================
# 11. SAVE SELECTION METADATA
# ==========================================================


def save_selection_metadata(comparison_data, selected_model, model_info):
    """
    Save official best model selection metadata.
    """

    print("=" * 70)
    print("Saving Best Model Selection Metadata...")
    print("=" * 70)

    selected_result = None

    for model in comparison_data["models"]:

        if model["Model"] == selected_model:

            selected_result = model
            break

    if selected_result is None:

        raise ValueError("Selected model performance record " "could not be found.")

    metadata = {
        "project": "OJT AI Project",
        "selection_version": "1.0",
        "target": TARGET_COLUMN,
        "selection_metric": SELECTION_METRIC,
        "selected_model": selected_model,
        "algorithm": model_info["algorithm"],
        "source_model": model_info["model_path"],
        "best_model_artifact": BEST_MODEL_FILE,
        "test_performance": {
            "accuracy": selected_result["Test Accuracy"],
            "precision": selected_result["Test Precision"],
            "recall": selected_result["Test Recall"],
            "f1": selected_result["Test F1"],
            "roc_auc": selected_result["Test ROC-AUC"],
        },
        "ranking": selected_result.get("F1 Rank"),
        "dataset_type": "synthetic_development_dataset",
        "note": (
            "The selected model is based on Test F1 "
            "from the model comparison stage. "
            "The dataset is synthetic development data, "
            "so model performance should not be interpreted "
            "as real-world OJT prediction performance."
        ),
    }

    with open(BEST_MODEL_METADATA_FILE, "w", encoding="utf-8") as file:

        json.dump(metadata, file, indent=4, ensure_ascii=False)

    if not os.path.exists(BEST_MODEL_METADATA_FILE):

        raise RuntimeError("Best model selection metadata " "was not created.")

    print("Selection Metadata : PASS")

    print(f"Metadata : " f"{BEST_MODEL_METADATA_FILE}")

    print()


# ==========================================================
# 12. MAIN
# ==========================================================


def main():

    print()
    print("=" * 70)
    print("OJT AI PROJECT - BEST MODEL SELECTION V2")
    print("=" * 70)
    print()

    # ------------------------------------------------------
    # Step 1: Load comparison
    # ------------------------------------------------------

    comparison_data = load_comparison_result()

    # ------------------------------------------------------
    # Step 2: Validate comparison
    # ------------------------------------------------------

    validate_comparison_result(comparison_data)

    # ------------------------------------------------------
    # Step 3: Verify best candidate
    # ------------------------------------------------------

    selected_model = verify_best_model_candidate(comparison_data)

    # ------------------------------------------------------
    # Step 4: Verify model file
    # ------------------------------------------------------

    model_info = verify_model_file(selected_model)

    # ------------------------------------------------------
    # Step 5: Create best model artifact
    # ------------------------------------------------------

    create_best_model(selected_model, model_info)

    # ------------------------------------------------------
    # Step 6: Save metadata
    # ------------------------------------------------------

    save_selection_metadata(comparison_data, selected_model, model_info)

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print("=" * 70)
    print("BEST MODEL SELECTION: SUCCESS")
    print("=" * 70)

    print(f"Selected Model : {selected_model}")

    print(f"Algorithm      : " f"{model_info['algorithm']}")

    print(f"Selection Metric : {SELECTION_METRIC}")

    print()

    print("Generated Files:")

    print(f"1. {BEST_MODEL_FILE}")

    print(f"2. {BEST_MODEL_METADATA_FILE}")

    print()

    print("Next Step:")

    print("Build Prediction Pipeline.")

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
