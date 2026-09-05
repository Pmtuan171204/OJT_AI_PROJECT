"""
==========================================================
Model Evaluation
OJT AI Project

Evaluate Logistic Regression performance for
OJT Delay Risk Prediction.
==========================================================
"""

import os
import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================================
# Add Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Project Paths
# ==========================================================

from config.paths import PROCESSED_DATA_DIR


# ==========================================================
# Dataset Paths
# ==========================================================

ENCODED_DIR = os.path.join(
    PROCESSED_DATA_DIR,
    "encoded"
)

X_TEST_PATH = os.path.join(
    ENCODED_DIR,
    "X_test_encoded.csv"
)

Y_TEST_PATH = os.path.join(
    ENCODED_DIR,
    "y_test.csv"
)


# ==========================================================
# Model Path
# ==========================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "logistic_regression.pkl"
)


# ==========================================================
# Evaluation Class
# ==========================================================

class ModelEvaluator:

    def __init__(self):

        self.model = None

        self.X_test = None
        self.y_test = None

        self.y_pred = None
        self.y_probability = None

        self.metrics = {}

    # ======================================================
    # Load Model
    # ======================================================

    def load_model(self):

        print()
        print("=" * 60)
        print("Loading Logistic Regression Model...")
        print("=" * 60)

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        self.model = joblib.load(
            MODEL_PATH
        )

        print()
        print(
            "Model Loaded Successfully."
        )

        print(
            f"Model : {MODEL_PATH}"
        )

    # ======================================================
    # Load Test Dataset
    # ======================================================

    def load_test_dataset(self):

        print()
        print("=" * 60)
        print("Loading Test Dataset...")
        print("=" * 60)

        if not os.path.exists(X_TEST_PATH):

            raise FileNotFoundError(
                f"X_test file not found: {X_TEST_PATH}"
            )

        if not os.path.exists(Y_TEST_PATH):

            raise FileNotFoundError(
                f"y_test file not found: {Y_TEST_PATH}"
            )

        self.X_test = pd.read_csv(
            X_TEST_PATH
        )

        self.y_test = pd.read_csv(
            Y_TEST_PATH
        ).iloc[:, 0]

        print()
        print(
            "Test Dataset Loaded Successfully."
        )

        print(
            f"Test Samples : {len(self.X_test)}"
        )

        print(
            f"Features     : {self.X_test.shape[1]}"
        )

    # ======================================================
    # Validate Evaluation Dataset
    # ======================================================

    def validate_dataset(self):

        print()
        print("=" * 60)
        print("Validating Evaluation Dataset...")
        print("=" * 60)

        if len(self.X_test) != len(self.y_test):

            raise ValueError(
                "X_test and y_test row counts do not match."
            )

        if self.X_test.isnull().sum().sum() > 0:

            raise ValueError(
                "X_test contains missing values."
            )

        valid_targets = {0, 1}

        if not set(
            self.y_test.unique()
        ).issubset(valid_targets):

            raise ValueError(
                "y_test contains invalid target values."
            )

        print()
        print(
            "Evaluation Dataset Validation Passed."
        )

    # ======================================================
    # Generate Predictions
    # ======================================================

    def predict(self):

        print()
        print("=" * 60)
        print("Generating Predictions...")
        print("=" * 60)

        self.y_pred = self.model.predict(
            self.X_test
        )

        self.y_probability = (
            self.model.predict_proba(
                self.X_test
            )[:, 1]
        )

        print()
        print(
            "Predictions Generated Successfully."
        )

        print(
            f"Predictions : {len(self.y_pred)}"
        )

    # ======================================================
    # Calculate Metrics
    # ======================================================

    def calculate_metrics(self):

        print()
        print("=" * 60)
        print("Calculating Evaluation Metrics...")
        print("=" * 60)

        accuracy = accuracy_score(
            self.y_test,
            self.y_pred
        )

        precision = precision_score(
            self.y_test,
            self.y_pred,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            self.y_pred,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            self.y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            self.y_test,
            self.y_probability
        )

        self.metrics = {

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1-Score": f1,

            "ROC-AUC": roc_auc

        }

    # ======================================================
    # Display Metrics
    # ======================================================

    def display_metrics(self):

        print()
        print("=" * 60)
        print("LOGISTIC REGRESSION EVALUATION")
        print("=" * 60)

        print(
            f"Accuracy  : {self.metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision : {self.metrics['Precision']:.4f}"
        )

        print(
            f"Recall    : {self.metrics['Recall']:.4f}"
        )

        print(
            f"F1-Score  : {self.metrics['F1-Score']:.4f}"
        )

        print(
            f"ROC-AUC   : {self.metrics['ROC-AUC']:.4f}"
        )

    # ======================================================
    # Display Confusion Matrix
    # ======================================================

    def display_confusion_matrix(self):

        print()
        print("=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)

        matrix = confusion_matrix(
            self.y_test,
            self.y_pred
        )

        print()
        print(
            "                 Predicted"
        )

        print(
            "                 0       1"
        )

        print(
            f"Actual 0     {matrix[0][0]:5d}   "
            f"{matrix[0][1]:5d}"
        )

        print(
            f"Actual 1     {matrix[1][0]:5d}   "
            f"{matrix[1][1]:5d}"
        )

        print()
        print(
            "TN =", matrix[0][0]
        )

        print(
            "FP =", matrix[0][1]
        )

        print(
            "FN =", matrix[1][0]
        )

        print(
            "TP =", matrix[1][1]
        )

    # ======================================================
    # Run Evaluation
    # ======================================================

    def run(self):

        self.load_model()

        self.load_test_dataset()

        self.validate_dataset()

        self.predict()

        self.calculate_metrics()

        self.display_metrics()

        self.display_confusion_matrix()


# ==========================================================
# Main
# ==========================================================

def main():

    evaluator = ModelEvaluator()

    evaluator.run()

    print()
    print("=" * 60)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 60)


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":
    main()