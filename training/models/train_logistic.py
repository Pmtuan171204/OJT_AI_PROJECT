import os
import sys
import json
import joblib
import pandas as pd

# ==========================================================
# PROJECT ROOT SETUP
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# MACHINE LEARNING IMPORTS
# ==========================================================

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================================
# PROJECT CONFIG IMPORTS
# ==========================================================

from config.paths import (
    X_TRAIN_ENCODED_PATH,
    X_TEST_ENCODED_PATH,
    Y_TRAIN_ENCODED_PATH,
    Y_TEST_ENCODED_PATH,
    MODEL_DIR,
    LOGISTIC_REGRESSION_MODEL,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config.paths import (
    X_TRAIN_ENCODED_PATH,
    X_TEST_ENCODED_PATH,
    Y_TRAIN_ENCODED_PATH,
    Y_TEST_ENCODED_PATH,
    MODEL_DIR,
    LOGISTIC_REGRESSION_MODEL,
)

# ==========================================================
# CONFIGURATION
# ==========================================================

TARGET_COLUMN = "OJT_Delay_Outcome"

RANDOM_STATE = 42

MAX_ITER = 1000

MODEL_NAME = "Logistic Regression"

MODEL_VERSION = "1.0"


# ==========================================================
# LOAD DATA
# ==========================================================


def load_data():

    print("\n" + "=" * 60)
    print("Loading Encoded Dataset...")
    print("=" * 60)

    X_train = pd.read_csv(X_TRAIN_ENCODED_PATH)
    X_test = pd.read_csv(X_TEST_ENCODED_PATH)

    y_train_df = pd.read_csv(Y_TRAIN_ENCODED_PATH)
    y_test_df = pd.read_csv(Y_TEST_ENCODED_PATH)

    # ------------------------------------------------------
    # Check target
    # ------------------------------------------------------

    if TARGET_COLUMN not in y_train_df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' not found in y_train.")

    if TARGET_COLUMN not in y_test_df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' not found in y_test.")

    y_train = y_train_df[TARGET_COLUMN]
    y_test = y_test_df[TARGET_COLUMN]

    print("\nEncoded Dataset Loaded Successfully.")

    print(f"X_train Rows    : {len(X_train)}")
    print(f"X_test Rows     : {len(X_test)}")
    print(f"Features        : {X_train.shape[1]}")
    print(f"y_train Rows    : {len(y_train)}")
    print(f"y_test Rows     : {len(y_test)}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# VALIDATE DATASET
# ==========================================================


def validate_dataset(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print("Validating Training Dataset...")
    print("=" * 60)

    # ------------------------------------------------------
    # Empty check
    # ------------------------------------------------------

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if X_test.empty:
        raise ValueError("X_test is empty.")

    if y_train.empty:
        raise ValueError("y_train is empty.")

    if y_test.empty:
        raise ValueError("y_test is empty.")

    # ------------------------------------------------------
    # Row alignment
    # ------------------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test row counts do not match.")

    # ------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test features do not match.")

    # ------------------------------------------------------
    # Missing values
    # ------------------------------------------------------

    if X_train.isnull().any().any():
        raise ValueError("X_train contains missing values.")

    if X_test.isnull().any().any():
        raise ValueError("X_test contains missing values.")

    if y_train.isnull().any():
        raise ValueError("y_train contains missing values.")

    if y_test.isnull().any():
        raise ValueError("y_test contains missing values.")

    # ------------------------------------------------------
    # Numeric feature check
    # ------------------------------------------------------

    non_numeric_train = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    non_numeric_test = X_test.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_train:
        raise ValueError(f"Non-numeric features in X_train: " f"{non_numeric_train}")

    if non_numeric_test:
        raise ValueError(f"Non-numeric features in X_test: " f"{non_numeric_test}")

    # ------------------------------------------------------
    # Target check
    # ------------------------------------------------------

    train_classes = sorted(y_train.unique().tolist())

    test_classes = sorted(y_test.unique().tolist())

    if train_classes != [0, 1]:
        raise ValueError(f"Unexpected training classes: {train_classes}")

    if test_classes != [0, 1]:
        raise ValueError(f"Unexpected testing classes: {test_classes}")

    print("\nDataset Validation Passed.")

    print(f"Training Shape : {X_train.shape}")
    print(f"Testing Shape  : {X_test.shape}")
    print(f"Target Classes : {train_classes}")


# ==========================================================
# CREATE MODEL
# ==========================================================


def create_model():

    print("\n" + "=" * 60)
    print("Creating Logistic Regression Model...")
    print("=" * 60)

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER)

    print("\nLogistic Regression Model Created.")

    print("class_weight : None")
    print(f"random_state : {RANDOM_STATE}")
    print(f"max_iter     : {MAX_ITER}")

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================


def train_model(model, X_train, y_train):

    print("\n" + "=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)

    model.fit(X_train, y_train)

    print("\nTraining Completed Successfully.")

    return model


# ==========================================================
# EVALUATE MODEL
# ==========================================================


def evaluate_model(model, X_train, y_train, X_test, y_test):

    print("\n" + "=" * 60)
    print("Evaluating Logistic Regression...")
    print("=" * 60)

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # ------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------

    train_accuracy = accuracy_score(y_train, train_pred)

    train_precision = precision_score(y_train, train_pred, zero_division=0)

    train_recall = recall_score(y_train, train_pred, zero_division=0)

    train_f1 = f1_score(y_train, train_pred, zero_division=0)

    train_roc_auc = roc_auc_score(y_train, train_prob)

    # ------------------------------------------------------
    # Testing metrics
    # ------------------------------------------------------

    test_accuracy = accuracy_score(y_test, test_pred)

    test_precision = precision_score(y_test, test_pred, zero_division=0)

    test_recall = recall_score(y_test, test_pred, zero_division=0)

    test_f1 = f1_score(y_test, test_pred, zero_division=0)

    test_roc_auc = roc_auc_score(y_test, test_prob)

    # ------------------------------------------------------
    # Display training metrics
    # ------------------------------------------------------

    print("\nTraining Metrics")
    print("-" * 40)

    print(f"Accuracy  : {train_accuracy:.4f}")
    print(f"Precision : {train_precision:.4f}")
    print(f"Recall    : {train_recall:.4f}")
    print(f"F1-Score  : {train_f1:.4f}")
    print(f"ROC-AUC   : {train_roc_auc:.4f}")

    # ------------------------------------------------------
    # Display testing metrics
    # ------------------------------------------------------

    print("\nTesting Metrics")
    print("-" * 40)

    print(f"Accuracy  : {test_accuracy:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1-Score  : {test_f1:.4f}")
    print(f"ROC-AUC   : {test_roc_auc:.4f}")

    # ------------------------------------------------------
    # Metrics dictionary
    # ------------------------------------------------------

    metrics = {
        "train": {
            "accuracy": float(train_accuracy),
            "precision": float(train_precision),
            "recall": float(train_recall),
            "f1": float(train_f1),
            "roc_auc": float(train_roc_auc),
        },
        "test": {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1": float(test_f1),
            "roc_auc": float(test_roc_auc),
        },
    }

    return metrics


# ==========================================================
# SAVE MODEL
# ==========================================================


def save_model(model):

    print("\n" + "=" * 60)
    print("Saving Logistic Regression Model...")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, LOGISTIC_REGRESSION_MODEL)

    print("\nModel Saved Successfully.")

    print(f"Model : {LOGISTIC_REGRESSION_MODEL}")

    # ------------------------------------------------------
    # Verify
    # ------------------------------------------------------

    if not os.path.exists(LOGISTIC_REGRESSION_MODEL):
        raise RuntimeError("Model file was not created.")

    print("Model File Verification : PASS")


# ==========================================================
# SAVE METADATA
# ==========================================================


def save_metadata(model, metrics, X_train):

    print("\n" + "=" * 60)
    print("Saving Model Metadata...")
    print("=" * 60)

    metadata_path = os.path.join(MODEL_DIR, "logistic_regression_metadata.json")

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": "LogisticRegression",
        "target": TARGET_COLUMN,
        "target_classes": {"0": "No OJT delay", "1": "OJT delay"},
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
        "class_weight": None,
        "training_samples": int(len(X_train)),
        "feature_count": int(X_train.shape[1]),
        "features": X_train.columns.tolist(),
        "metrics": metrics,
        "dataset_type": "synthetic_development_dataset",
        "note": (
            "OJT_Delay_Outcome is generated by "
            "a synthetic future-outcome simulation. "
            "Current model performance should not "
            "be interpreted as real-world OJT "
            "prediction performance."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:

        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\nMetadata Saved Successfully.")

    print(f"Metadata : {metadata_path}")

    # ------------------------------------------------------
    # Verify
    # ------------------------------------------------------

    if not os.path.exists(metadata_path):
        raise RuntimeError("Metadata file was not created.")

    print("Metadata File Verification : PASS")


# ==========================================================
# MAIN
# ==========================================================


def main():

    print("\n")
    print("=" * 70)
    print("OJT AI PROJECT - " "LOGISTIC REGRESSION TRAINING V2")
    print("=" * 70)

    # ------------------------------------------------------
    # Step 1
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    # ------------------------------------------------------
    # Step 2
    # ------------------------------------------------------

    validate_dataset(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------
    # Step 3
    # ------------------------------------------------------

    model = create_model()

    # ------------------------------------------------------
    # Step 4
    # ------------------------------------------------------

    model = train_model(model, X_train, y_train)

    # ------------------------------------------------------
    # Step 5
    # ------------------------------------------------------

    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------
    # Step 6
    # ------------------------------------------------------

    save_model(model)

    # ------------------------------------------------------
    # Step 7
    # ------------------------------------------------------

    save_metadata(model, metrics, X_train)

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION TRAINING: SUCCESS")
    print("=" * 70)

    print("\nGenerated Files:")

    print(f"1. {LOGISTIC_REGRESSION_MODEL}")

    print(f"2. {os.path.join(MODEL_DIR, 'logistic_regression_metadata.json')}")

    print("\nNext Step:")
    print("Train Random Forest and compare models.")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
