# ==========================================================
# 1. IMPORTS
# ==========================================================

import os
import sys
import json
import joblib
import pandas as pd

# ----------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# 2. MACHINE LEARNING IMPORTS
# ==========================================================

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================================
# 3. PROJECT CONFIG
# ==========================================================

from config.paths import (
    X_TRAIN_ENCODED_PATH,
    X_TEST_ENCODED_PATH,
    Y_TRAIN_ENCODED_PATH,
    Y_TEST_ENCODED_PATH,
    MODEL_DIR,
    RANDOM_FOREST_MODEL,
)

# ==========================================================
# 4. CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_FEATURES = "sqrt"

MIN_SAMPLES_LEAF = 2

N_JOBS = -1

TARGET_COLUMN = "OJT_Delay_Outcome"

MODEL_VERSION = "1.0"

METADATA_FILE = os.path.join(MODEL_DIR, "random_forest_metadata.json")


# ==========================================================
# 5. LOAD DATA
# ==========================================================


def load_data():
    """
    Load encoded training/testing datasets.
    """

    print("=" * 70)
    print("Loading Encoded Dataset...")
    print("=" * 70)

    X_train = pd.read_csv(X_TRAIN_ENCODED_PATH)
    X_test = pd.read_csv(X_TEST_ENCODED_PATH)

    y_train = pd.read_csv(Y_TRAIN_ENCODED_PATH)
    y_test = pd.read_csv(Y_TEST_ENCODED_PATH)

    print(f"X_train Rows : {len(X_train)}")
    print(f"X_test Rows  : {len(X_test)}")
    print(f"Features     : {X_train.shape[1]}")
    print()

    return X_train, X_test, y_train, y_test


# ==========================================================
# 6. VALIDATE DATASET
# ==========================================================


def validate_dataset(X_train, X_test, y_train, y_test):
    """
    Validate data before model training.

    Checks:
        - Row alignment
        - Feature alignment
        - Missing values
        - Numeric features
        - Target column
        - Target values
    """

    print("=" * 70)
    print("Validating Dataset...")
    print("=" * 70)

    # ------------------------------------------------------
    # Check row alignment
    # ------------------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test row counts do not match.")

    # ------------------------------------------------------
    # Check feature alignment
    # ------------------------------------------------------

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test feature columns do not match.")

    # ------------------------------------------------------
    # Check target column
    # ------------------------------------------------------

    if TARGET_COLUMN not in y_train.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' " "not found in y_train.")

    if TARGET_COLUMN not in y_test.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' " "not found in y_test.")

    # ------------------------------------------------------
    # Check missing values
    # ------------------------------------------------------

    if X_train.isnull().any().any():
        raise ValueError("X_train contains missing values.")

    if X_test.isnull().any().any():
        raise ValueError("X_test contains missing values.")

    if y_train.isnull().any().any():
        raise ValueError("y_train contains missing values.")

    if y_test.isnull().any().any():
        raise ValueError("y_test contains missing values.")

    # ------------------------------------------------------
    # Check numeric features
    # ------------------------------------------------------

    non_numeric_train = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    non_numeric_test = X_test.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_train:
        raise ValueError(
            f"Non-numeric features found in X_train: " f"{non_numeric_train}"
        )

    if non_numeric_test:
        raise ValueError(
            f"Non-numeric features found in X_test: " f"{non_numeric_test}"
        )

    # ------------------------------------------------------
    # Check target values
    # ------------------------------------------------------

    train_target_values = set(y_train[TARGET_COLUMN].unique())

    test_target_values = set(y_test[TARGET_COLUMN].unique())

    allowed_values = {0, 1}

    if not train_target_values.issubset(allowed_values):
        raise ValueError("Unexpected values found in y_train.")

    if not test_target_values.issubset(allowed_values):
        raise ValueError("Unexpected values found in y_test.")

    # ------------------------------------------------------
    # Check infinite values
    # ------------------------------------------------------

    if X_train.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("X_train contains infinite values.")

    if X_test.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("X_test contains infinite values.")

    print("Dataset Validation : PASS")
    print()


# ==========================================================
# 7. CREATE RANDOM FOREST MODEL
# ==========================================================


def create_model():
    """
    Create Random Forest classifier.

    Hyperparameters are intentionally kept as a strong
    baseline rather than optimized against the test set.
    """

    print("=" * 70)
    print("Creating Random Forest Model...")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    print(f"n_estimators     : {N_ESTIMATORS}")
    print(f"max_features     : {MAX_FEATURES}")
    print(f"min_samples_leaf : {MIN_SAMPLES_LEAF}")
    print(f"random_state     : {RANDOM_STATE}")
    print(f"n_jobs           : {N_JOBS}")
    print()

    return model


# ==========================================================
# 8. TRAIN MODEL
# ==========================================================


def train_model(model, X_train, y_train):
    """
    Train Random Forest model.
    """

    print("=" * 70)
    print("Training Random Forest...")
    print("=" * 70)

    model.fit(X_train, y_train[TARGET_COLUMN])

    print("Training Completed Successfully.")
    print()

    return model


# ==========================================================
# 9. EVALUATE MODEL
# ==========================================================


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate Random Forest on training and testing data.

    Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1
        - ROC-AUC
    """

    print("=" * 70)
    print("Evaluating Random Forest...")
    print("=" * 70)

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # ------------------------------------------------------
    # Prediction probabilities
    # ------------------------------------------------------

    train_probabilities = model.predict_proba(X_train)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------

    train_accuracy = accuracy_score(y_train[TARGET_COLUMN], train_predictions)

    train_precision = precision_score(
        y_train[TARGET_COLUMN], train_predictions, zero_division=0
    )

    train_recall = recall_score(
        y_train[TARGET_COLUMN], train_predictions, zero_division=0
    )

    train_f1 = f1_score(y_train[TARGET_COLUMN], train_predictions, zero_division=0)

    train_roc_auc = roc_auc_score(y_train[TARGET_COLUMN], train_probabilities)

    # ------------------------------------------------------
    # Testing metrics
    # ------------------------------------------------------

    test_accuracy = accuracy_score(y_test[TARGET_COLUMN], test_predictions)

    test_precision = precision_score(
        y_test[TARGET_COLUMN], test_predictions, zero_division=0
    )

    test_recall = recall_score(y_test[TARGET_COLUMN], test_predictions, zero_division=0)

    test_f1 = f1_score(y_test[TARGET_COLUMN], test_predictions, zero_division=0)

    test_roc_auc = roc_auc_score(y_test[TARGET_COLUMN], test_probabilities)

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print("TRAINING RESULTS")
    print("-" * 70)
    print(f"Accuracy  : {train_accuracy:.4f}")
    print(f"Precision : {train_precision:.4f}")
    print(f"Recall    : {train_recall:.4f}")
    print(f"F1 Score  : {train_f1:.4f}")
    print(f"ROC-AUC   : {train_roc_auc:.4f}")
    print()

    print("TEST RESULTS")
    print("-" * 70)
    print(f"Accuracy  : {test_accuracy:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"ROC-AUC   : {test_roc_auc:.4f}")
    print()

    metrics = {
        "train": {
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_recall,
            "f1": train_f1,
            "roc_auc": train_roc_auc,
        },
        "test": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
        },
    }

    return metrics


# ==========================================================
# 10. SAVE MODEL
# ==========================================================


def save_model(model):
    """
    Save trained Random Forest model.
    """

    print("=" * 70)
    print("Saving Random Forest Model...")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, RANDOM_FOREST_MODEL)

    if not os.path.exists(RANDOM_FOREST_MODEL):
        raise RuntimeError("Random Forest model file was not created.")

    print("Model Saved Successfully.")
    print(f"Model : {RANDOM_FOREST_MODEL}")
    print()


# ==========================================================
# 11. SAVE METADATA
# ==========================================================


def save_metadata(model, metrics, X_train):
    """
    Save model configuration, features and metrics.
    """

    print("=" * 70)
    print("Saving Random Forest Metadata...")
    print("=" * 70)

    metadata = {
        "model_name": "Random Forest",
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "target": TARGET_COLUMN,
        "target_classes": {"0": "No OJT delay", "1": "OJT delay"},
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "max_features": MAX_FEATURES,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "n_jobs": N_JOBS,
        "class_weight": None,
        "training_samples": len(X_train),
        "feature_count": X_train.shape[1],
        "features": X_train.columns.tolist(),
        "metrics": metrics,
        "dataset_type": "synthetic_development_dataset",
        "note": (
            "OJT_Delay_Outcome is generated by a synthetic "
            "future-outcome simulation. Current model "
            "performance should not be interpreted as "
            "real-world OJT prediction performance."
        ),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:

        json.dump(metadata, file, indent=4, ensure_ascii=False)

    if not os.path.exists(METADATA_FILE):
        raise RuntimeError("Random Forest metadata file was not created.")

    print("Metadata Saved Successfully.")
    print(f"Metadata : {METADATA_FILE}")
    print()

    return metadata


# ==========================================================
# 12. MAIN
# ==========================================================


def main():

    print()
    print("=" * 70)
    print("OJT AI PROJECT - RANDOM FOREST TRAINING V2")
    print("=" * 70)
    print()

    # ------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    # ------------------------------------------------------
    # Step 2: Validate data
    # ------------------------------------------------------

    validate_dataset(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------
    # Step 3: Create model
    # ------------------------------------------------------

    model = create_model()

    # ------------------------------------------------------
    # Step 4: Train model
    # ------------------------------------------------------

    model = train_model(model, X_train, y_train)

    # ------------------------------------------------------
    # Step 5: Evaluate model
    # ------------------------------------------------------

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    # ------------------------------------------------------
    # Step 6: Save model
    # ------------------------------------------------------

    save_model(model)

    # ------------------------------------------------------
    # Step 7: Save metadata
    # ------------------------------------------------------

    save_metadata(model, metrics, X_train)

    # ------------------------------------------------------
    # Final verification
    # ------------------------------------------------------

    print("=" * 70)
    print("RANDOM FOREST TRAINING: SUCCESS")
    print("=" * 70)

    print("Generated Files:")

    print(f"1. {RANDOM_FOREST_MODEL}")

    print(f"2. {METADATA_FILE}")

    print()

    print("Next Step:")
    print("Compare Logistic Regression and Random Forest.")

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
