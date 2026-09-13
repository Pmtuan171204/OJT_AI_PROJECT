import os
import sys
import json
import pandas as pd

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

from config.paths import MODEL_DIR, EVALUATION_REPORT_DIR

# ==========================================================
# 4. CONFIGURATION
# ==========================================================

LOGISTIC_METADATA_FILE = os.path.join(MODEL_DIR, "logistic_regression_metadata.json")

RANDOM_FOREST_METADATA_FILE = os.path.join(MODEL_DIR, "random_forest_metadata.json")

COMPARISON_CSV = os.path.join(EVALUATION_REPORT_DIR, "model_comparison.csv")

COMPARISON_JSON = os.path.join(EVALUATION_REPORT_DIR, "model_comparison.json")

COMPARISON_TXT = os.path.join(EVALUATION_REPORT_DIR, "model_comparison.txt")

TARGET_COLUMN = "OJT_Delay_Outcome"

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# ==========================================================
# 5. LOAD METADATA
# ==========================================================


def load_metadata(file_path, model_name):
    """
    Load model metadata from JSON file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{model_name} metadata not found:\n" f"{file_path}")

    with open(file_path, "r", encoding="utf-8") as file:

        metadata = json.load(file)

    return metadata


# ==========================================================
# 6. VALIDATE METADATA
# ==========================================================


def validate_metadata(metadata, expected_model_name):
    """
    Validate required metadata fields before comparison.
    """

    required_fields = [
        "model_name",
        "algorithm",
        "target",
        "training_samples",
        "feature_count",
        "features",
        "metrics",
    ]

    for field in required_fields:

        if field not in metadata:
            raise ValueError(f"Missing metadata field: {field}")

    if metadata["target"] != TARGET_COLUMN:
        raise ValueError(f"Unexpected target: " f"{metadata['target']}")

    if metadata["model_name"] != expected_model_name:
        raise ValueError(
            f"Expected model '{expected_model_name}', "
            f"but found '{metadata['model_name']}'."
        )

    if "train" not in metadata["metrics"]:
        raise ValueError(f"Training metrics missing for " f"{expected_model_name}.")

    if "test" not in metadata["metrics"]:
        raise ValueError(f"Testing metrics missing for " f"{expected_model_name}.")

    for metric in METRICS:

        if metric not in metadata["metrics"]["train"]:
            raise ValueError(
                f"Missing train metric '{metric}' " f"for {expected_model_name}."
            )

        if metric not in metadata["metrics"]["test"]:
            raise ValueError(
                f"Missing test metric '{metric}' " f"for {expected_model_name}."
            )


# ==========================================================
# 7. CHECK MODEL COMPATIBILITY
# ==========================================================


def validate_model_compatibility(logistic_metadata, random_forest_metadata):
    """
    Ensure both models were evaluated on compatible
    datasets and target definitions.
    """

    print("=" * 70)
    print("Checking Model Compatibility...")
    print("=" * 70)

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    if logistic_metadata["target"] != random_forest_metadata["target"]:
        raise ValueError("Models use different target columns.")

    # ------------------------------------------------------
    # Training sample count
    # ------------------------------------------------------

    if (
        logistic_metadata["training_samples"]
        != random_forest_metadata["training_samples"]
    ):
        raise ValueError(
            "Models were trained on different " "numbers of training samples."
        )

    # ------------------------------------------------------
    # Feature count
    # ------------------------------------------------------

    if logistic_metadata["feature_count"] != random_forest_metadata["feature_count"]:
        raise ValueError("Models use different feature counts.")

    # ------------------------------------------------------
    # Feature names
    # ------------------------------------------------------

    logistic_features = logistic_metadata["features"]
    random_forest_features = random_forest_metadata["features"]

    if logistic_features != random_forest_features:

        raise ValueError("Models do not use the same encoded " "feature columns.")

    print("Target              : PASS")
    print("Training Samples    : PASS")
    print("Feature Count       : PASS")
    print("Feature Names       : PASS")
    print()
    print("Model Compatibility : PASS")
    print()


# ==========================================================
# 8. BUILD COMPARISON TABLE
# ==========================================================


def build_comparison_table(logistic_metadata, random_forest_metadata):
    """
    Create a comparison table containing train/test
    metrics for both models.
    """

    rows = []

    models = [logistic_metadata, random_forest_metadata]

    for metadata in models:

        model_name = metadata["model_name"]

        test_metrics = metadata["metrics"]["test"]

        train_metrics = metadata["metrics"]["train"]

        rows.append(
            {
                "Model": model_name,
                "Train Accuracy": train_metrics["accuracy"],
                "Test Accuracy": test_metrics["accuracy"],
                "Train Precision": train_metrics["precision"],
                "Test Precision": test_metrics["precision"],
                "Train Recall": train_metrics["recall"],
                "Test Recall": test_metrics["recall"],
                "Train F1": train_metrics["f1"],
                "Test F1": test_metrics["f1"],
                "Train ROC-AUC": train_metrics["roc_auc"],
                "Test ROC-AUC": test_metrics["roc_auc"],
            }
        )

    comparison_df = pd.DataFrame(rows)

    return comparison_df


# ==========================================================
# 9. CALCULATE TRAIN-TEST GAPS
# ==========================================================


def calculate_gaps(comparison_df):
    """
    Calculate train-test performance gaps.
    """

    comparison_df["Accuracy Gap"] = (
        comparison_df["Train Accuracy"] - comparison_df["Test Accuracy"]
    )

    comparison_df["Precision Gap"] = (
        comparison_df["Train Precision"] - comparison_df["Test Precision"]
    )

    comparison_df["Recall Gap"] = (
        comparison_df["Train Recall"] - comparison_df["Test Recall"]
    )

    comparison_df["F1 Gap"] = comparison_df["Train F1"] - comparison_df["Test F1"]

    comparison_df["ROC-AUC Gap"] = (
        comparison_df["Train ROC-AUC"] - comparison_df["Test ROC-AUC"]
    )

    return comparison_df


# ==========================================================
# 10. DETERMINE BEST MODEL
# ==========================================================


def determine_best_model(comparison_df):
    """
    Determine the best model based on Test F1.

    F1 is selected as the primary metric because the
    project needs a balance between detecting delayed
    students and avoiding excessive false alarms.
    """

    best_index = comparison_df["Test F1"].idxmax()

    best_model = comparison_df.loc[best_index, "Model"]

    return best_model


# ==========================================================
# 11. ADD RANKING
# ==========================================================


def add_ranking(comparison_df):
    """
    Rank models according to Test F1.
    """

    comparison_df["F1 Rank"] = (
        comparison_df["Test F1"].rank(ascending=False, method="min").astype(int)
    )

    comparison_df = comparison_df.sort_values(
        by="Test F1", ascending=False
    ).reset_index(drop=True)

    return comparison_df


# ==========================================================
# 12. DISPLAY COMPARISON
# ==========================================================


def display_comparison(comparison_df, best_model):
    """
    Display the comparison results.
    """

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    display_columns = [
        "Model",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Test ROC-AUC",
    ]

    display_df = comparison_df[display_columns].copy()

    for column in display_columns[1:]:

        display_df[column] = display_df[column].map(lambda x: f"{x:.4f}")

    print(display_df.to_string(index=False))

    print()
    print("=" * 70)
    print("BEST MODEL CANDIDATE")
    print("=" * 70)

    print(f"Selected based on Test F1 : {best_model}")

    print()
    print("Primary Selection Metric : Test F1")

    print("Reason : Balance between Precision and Recall")

    print()


# ==========================================================
# 13. SAVE CSV
# ==========================================================


def save_csv(comparison_df):
    """
    Save comparison results as CSV.
    """

    os.makedirs(EVALUATION_REPORT_DIR, exist_ok=True)

    comparison_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

    if not os.path.exists(COMPARISON_CSV):
        raise RuntimeError("Comparison CSV was not created.")

    print(f"CSV Saved : {COMPARISON_CSV}")


# ==========================================================
# 14. SAVE JSON
# ==========================================================


def save_json(comparison_df, best_model):
    """
    Save structured comparison results as JSON.
    """

    output = {
        "project": "OJT AI Project",
        "target": TARGET_COLUMN,
        "selection_metric": "Test F1",
        "best_model_candidate": best_model,
        "dataset_type": ("synthetic_development_dataset"),
        "note": (
            "Model comparison is based on synthetic "
            "development data. Results should not be "
            "interpreted as real-world OJT prediction "
            "performance."
        ),
        "models": comparison_df.to_dict(orient="records"),
    }

    with open(COMPARISON_JSON, "w", encoding="utf-8") as file:

        json.dump(output, file, indent=4, ensure_ascii=False)

    if not os.path.exists(COMPARISON_JSON):
        raise RuntimeError("Comparison JSON was not created.")

    print(f"JSON Saved : {COMPARISON_JSON}")


# ==========================================================
# 15. SAVE TEXT REPORT
# ==========================================================


def save_text_report(comparison_df, best_model):
    """
    Save human-readable comparison report.
    """

    with open(COMPARISON_TXT, "w", encoding="utf-8") as file:

        file.write("==================================================\n")

        file.write("OJT AI PROJECT - MODEL COMPARISON REPORT\n")

        file.write("==================================================\n\n")

        file.write(f"Target: {TARGET_COLUMN}\n")

        file.write("Selection Metric: Test F1\n")

        file.write(f"Best Model Candidate: {best_model}\n\n")

        file.write("--------------------------------------------------\n")

        file.write("MODEL PERFORMANCE\n")

        file.write("--------------------------------------------------\n")

        for _, row in comparison_df.iterrows():

            file.write(f"\nModel: {row['Model']}\n")

            file.write(f"Test Accuracy : " f"{row['Test Accuracy']:.4f}\n")

            file.write(f"Test Precision: " f"{row['Test Precision']:.4f}\n")

            file.write(f"Test Recall   : " f"{row['Test Recall']:.4f}\n")

            file.write(f"Test F1       : " f"{row['Test F1']:.4f}\n")

            file.write(f"Test ROC-AUC  : " f"{row['Test ROC-AUC']:.4f}\n")

            file.write(f"F1 Gap        : " f"{row['F1 Gap']:.4f}\n")

        file.write("\n--------------------------------------------------\n")

        file.write("SELECTION\n")

        file.write("--------------------------------------------------\n")

        file.write(f"Best Model Candidate: {best_model}\n")

        file.write("Primary Metric: Test F1\n")

        file.write("Reason: F1 balances Precision and Recall.\n")

        file.write("\nIMPORTANT NOTE:\n")

        file.write("The dataset is a synthetic development dataset.\n")

        file.write("The reported performance must not be interpreted\n")

        file.write("as real-world OJT prediction performance.\n")

    if not os.path.exists(COMPARISON_TXT):
        raise RuntimeError("Comparison TXT report was not created.")

    print(f"TXT Saved : {COMPARISON_TXT}")


# ==========================================================
# 16. MAIN
# ==========================================================


def main():

    print()
    print("=" * 70)
    print("OJT AI PROJECT - MODEL COMPARISON V2")
    print("=" * 70)
    print()

    # ------------------------------------------------------
    # Step 1: Load metadata
    # ------------------------------------------------------

    logistic_metadata = load_metadata(LOGISTIC_METADATA_FILE, "Logistic Regression")

    random_forest_metadata = load_metadata(RANDOM_FOREST_METADATA_FILE, "Random Forest")

    print("Metadata Loading : PASS")
    print()

    # ------------------------------------------------------
    # Step 2: Validate metadata
    # ------------------------------------------------------

    validate_metadata(logistic_metadata, "Logistic Regression")

    validate_metadata(random_forest_metadata, "Random Forest")

    print("Metadata Validation : PASS")
    print()

    # ------------------------------------------------------
    # Step 3: Check compatibility
    # ------------------------------------------------------

    validate_model_compatibility(logistic_metadata, random_forest_metadata)

    # ------------------------------------------------------
    # Step 4: Build comparison
    # ------------------------------------------------------

    comparison_df = build_comparison_table(logistic_metadata, random_forest_metadata)

    # ------------------------------------------------------
    # Step 5: Calculate train-test gaps
    # ------------------------------------------------------

    comparison_df = calculate_gaps(comparison_df)

    # ------------------------------------------------------
    # Step 6: Determine best model
    # ------------------------------------------------------

    best_model = determine_best_model(comparison_df)

    # ------------------------------------------------------
    # Step 7: Add ranking
    # ------------------------------------------------------

    comparison_df = add_ranking(comparison_df)

    # ------------------------------------------------------
    # Step 8: Display
    # ------------------------------------------------------

    display_comparison(comparison_df, best_model)

    # ------------------------------------------------------
    # Step 9: Save reports
    # ------------------------------------------------------

    print("=" * 70)
    print("Saving Comparison Reports...")
    print("=" * 70)

    save_csv(comparison_df)

    save_json(comparison_df, best_model)

    save_text_report(comparison_df, best_model)

    print()

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print("=" * 70)
    print("MODEL COMPARISON: SUCCESS")
    print("=" * 70)

    print("Generated Files:")
    print(f"1. {COMPARISON_CSV}")
    print(f"2. {COMPARISON_JSON}")
    print(f"3. {COMPARISON_TXT}")

    print()

    print(f"Best Model Candidate : {best_model}")

    print("Next Step : Best Model Selection / " "Prediction Pipeline")

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
