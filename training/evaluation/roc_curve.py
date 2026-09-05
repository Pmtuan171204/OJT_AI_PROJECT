"""
============================================================
ROC CURVE EVALUATION
OJT AI Project

Evaluate Logistic Regression using:
- ROC Curve
- ROC-AUC Score
- False Positive Rate
- True Positive Rate

============================================================
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# ==========================================================
# PROJECT ROOT
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# CONFIG PATHS
# ==========================================================

from config.paths import ENCODED_DATA_DIR, MODEL_DIR, EVALUATION_REPORT_DIR

# ==========================================================
# FILE PATHS
# ==========================================================

X_TEST_PATH = os.path.join(ENCODED_DATA_DIR, "X_test_encoded.csv")

Y_TEST_PATH = os.path.join(ENCODED_DATA_DIR, "y_test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

ROC_CSV_PATH = os.path.join(EVALUATION_REPORT_DIR, "logistic_roc_curve.csv")

ROC_IMAGE_PATH = os.path.join(EVALUATION_REPORT_DIR, "logistic_roc_curve.png")


# ==========================================================
# ROC CURVE EVALUATOR
# ==========================================================


class ROCCurveEvaluator:

    def __init__(self):

        self.X_test = None
        self.y_test = None
        self.model = None

        self.y_probability = None

        self.fpr = None
        self.tpr = None
        self.thresholds = None

        self.roc_auc = None

    # ======================================================
    # LOAD DATASET
    # ======================================================

    def load_dataset(self):

        print("=" * 60)
        print("Loading Test Dataset...")
        print("=" * 60)

        self.X_test = pd.read_csv(X_TEST_PATH)

        self.y_test = pd.read_csv(Y_TEST_PATH).squeeze()

        print()
        print("Test Dataset Loaded Successfully.")
        print(f"Test Samples : {len(self.X_test)}")
        print(f"Features     : {self.X_test.shape[1]}")

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def load_model(self):

        print()
        print("=" * 60)
        print("Loading Logistic Regression Model...")
        print("=" * 60)

        import joblib

        self.model = joblib.load(MODEL_PATH)

        print()
        print("Model Loaded Successfully.")
        print(f"Model : {MODEL_PATH}")

    # ======================================================
    # VALIDATE DATASET
    # ======================================================

    def validate_dataset(self):

        print()
        print("=" * 60)
        print("Validating ROC Evaluation Dataset...")
        print("=" * 60)

        if len(self.X_test) != len(self.y_test):

            raise ValueError("X_test and y_test row counts do not match.")

        if self.X_test.isnull().sum().sum() > 0:

            raise ValueError("X_test contains missing values.")

        if self.y_test.isnull().sum() > 0:

            raise ValueError("y_test contains missing values.")

        if not set(self.y_test.unique()).issubset({0, 1}):

            raise ValueError("Target contains invalid class values.")

        print()
        print("ROC Evaluation Dataset Validation Passed.")

    # ======================================================
    # GENERATE PROBABILITY PREDICTIONS
    # ======================================================

    def generate_probability_predictions(self):

        print()
        print("=" * 60)
        print("Generating Probability Predictions...")
        print("=" * 60)

        self.y_probability = self.model.predict_proba(self.X_test)[:, 1]

        print()
        print("Probability Predictions Generated Successfully.")

        print(f"Predictions : {len(self.y_probability)}")

    # ======================================================
    # CALCULATE ROC
    # ======================================================

    def calculate_roc(self):

        print()
        print("=" * 60)
        print("Calculating ROC Curve...")
        print("=" * 60)

        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_probability)

        self.roc_auc = roc_auc_score(self.y_test, self.y_probability)

        print()
        print("ROC Curve Calculated Successfully.")

    # ======================================================
    # PRINT ROC-AUC
    # ======================================================

    def print_result(self):

        print()
        print("=" * 60)
        print("LOGISTIC REGRESSION ROC-AUC")
        print("=" * 60)

        print(f"ROC-AUC : {self.roc_auc:.4f}")

        print(f"ROC Points : {len(self.fpr)}")

        print()

        if self.roc_auc >= 0.90:

            print("Performance : Excellent")

        elif self.roc_auc >= 0.80:

            print("Performance : Good")

        elif self.roc_auc >= 0.70:

            print("Performance : Acceptable")

        else:

            print("Performance : Poor")

    # ======================================================
    # SAVE ROC DATA
    # ======================================================

    def save_roc_data(self):

        print()
        print("=" * 60)
        print("Saving ROC Curve Data...")
        print("=" * 60)

        os.makedirs(EVALUATION_REPORT_DIR, exist_ok=True)

        roc_data = pd.DataFrame(
            {
                "False_Positive_Rate": self.fpr,
                "True_Positive_Rate": self.tpr,
                "Threshold": self.thresholds,
            }
        )

        roc_data.to_csv(ROC_CSV_PATH, index=False)

        print()
        print("ROC Data Saved Successfully.")

        print(f"CSV : {ROC_CSV_PATH}")

    # ======================================================
    # GENERATE ROC CURVE IMAGE
    # ======================================================

    def generate_roc_plot(self):

        print()
        print("=" * 60)
        print("Generating ROC Curve Image...")
        print("=" * 60)

        plt.figure(figsize=(8, 6))

        plt.plot(
            self.fpr, self.tpr, label=f"Logistic Regression (AUC = {self.roc_auc:.4f})"
        )

        plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve - Logistic Regression")

        plt.legend(loc="lower right")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(ROC_IMAGE_PATH, dpi=300)

        plt.close()

        print()
        print("ROC Curve Image Saved Successfully.")

        print(f"Image : {ROC_IMAGE_PATH}")

    # ======================================================
    # RUN EVALUATION
    # ======================================================

    def evaluate(self):

        self.load_dataset()

        self.load_model()

        self.validate_dataset()

        self.generate_probability_predictions()

        self.calculate_roc()

        self.print_result()

        self.save_roc_data()

        self.generate_roc_plot()


# ==========================================================
# MAIN
# ==========================================================


def main():

    evaluator = ROCCurveEvaluator()

    evaluator.evaluate()

    print()
    print("=" * 60)
    print("ROC CURVE ANALYSIS COMPLETED")
    print("=" * 60)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
