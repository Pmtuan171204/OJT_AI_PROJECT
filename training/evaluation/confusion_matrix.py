"""
============================================================
Confusion Matrix Analysis
OJT AI Project

Generate and analyze the confusion matrix for
Logistic Regression OJT Delay Risk Prediction.
============================================================
"""

import os
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# ============================================================
# Add Project Root
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# Project Paths
# ============================================================

from config.paths import PROCESSED_DATA_DIR

# ============================================================
# Input Paths
# ============================================================

ENCODED_DIR = os.path.join(PROCESSED_DATA_DIR, "encoded")

X_TEST_PATH = os.path.join(ENCODED_DIR, "X_test_encoded.csv")

Y_TEST_PATH = os.path.join(ENCODED_DIR, "y_test.csv")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logistic_regression.pkl")


# ============================================================
# Output Directory
# ============================================================

EVALUATION_DIR = os.path.join(PROJECT_ROOT, "reports", "evaluation")

os.makedirs(EVALUATION_DIR, exist_ok=True)


# ============================================================
# Output Paths
# ============================================================

CONFUSION_MATRIX_IMAGE = os.path.join(EVALUATION_DIR, "logistic_confusion_matrix.png")

CONFUSION_MATRIX_CSV = os.path.join(EVALUATION_DIR, "logistic_confusion_matrix.csv")


# ============================================================
# Confusion Matrix Analyzer
# ============================================================


class ConfusionMatrixAnalyzer:

    def __init__(self):

        self.model = None

        self.X_test = None
        self.y_test = None

        self.y_pred = None

        self.matrix = None

    # ========================================================
    # Load Model
    # ========================================================

    def load_model(self):

        print()
        print("=" * 60)
        print("Loading Logistic Regression Model...")
        print("=" * 60)

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.model = joblib.load(MODEL_PATH)

        print()
        print("Model Loaded Successfully.")

        print(f"Model : {MODEL_PATH}")

    # ========================================================
    # Load Test Dataset
    # ========================================================

    def load_dataset(self):

        print()
        print("=" * 60)
        print("Loading Test Dataset...")
        print("=" * 60)

        if not os.path.exists(X_TEST_PATH):

            raise FileNotFoundError(f"X_test not found: {X_TEST_PATH}")

        if not os.path.exists(Y_TEST_PATH):

            raise FileNotFoundError(f"y_test not found: {Y_TEST_PATH}")

        self.X_test = pd.read_csv(X_TEST_PATH)

        self.y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

        print()
        print("Test Dataset Loaded Successfully.")

        print(f"Test Samples : {len(self.X_test)}")

        print(f"Features     : {self.X_test.shape[1]}")

    # ========================================================
    # Validate Dataset
    # ========================================================

    def validate_dataset(self):

        print()
        print("=" * 60)
        print("Validating Test Dataset...")
        print("=" * 60)

        if len(self.X_test) != len(self.y_test):

            raise ValueError("X_test and y_test row counts do not match.")

        if self.X_test.isnull().sum().sum() > 0:

            raise ValueError("X_test contains missing values.")

        valid_targets = {0, 1}

        if not set(self.y_test.unique()).issubset(valid_targets):

            raise ValueError("y_test contains invalid target values.")

        print()
        print("Test Dataset Validation Passed.")

    # ========================================================
    # Generate Predictions
    # ========================================================

    def generate_predictions(self):

        print()
        print("=" * 60)
        print("Generating Predictions...")
        print("=" * 60)

        self.y_pred = self.model.predict(self.X_test)

        print()
        print("Predictions Generated Successfully.")

        print(f"Predictions : {len(self.y_pred)}")

    # ========================================================
    # Calculate Confusion Matrix
    # ========================================================

    def calculate_matrix(self):

        print()
        print("=" * 60)
        print("Calculating Confusion Matrix...")
        print("=" * 60)

        self.matrix = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])

        print()
        print("Confusion Matrix Calculated Successfully.")

    # ========================================================
    # Display Matrix
    # ========================================================

    def display_matrix(self):

        tn = self.matrix[0][0]
        fp = self.matrix[0][1]
        fn = self.matrix[1][0]
        tp = self.matrix[1][1]

        print()
        print("=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)

        print()
        print("                 Predicted")

        print("                 0       1")

        print(f"Actual 0     {tn:5d}   {fp:5d}")

        print(f"Actual 1     {fn:5d}   {tp:5d}")

        print()
        print(f"True Negative  (TN) : {tn}")

        print(f"False Positive (FP) : {fp}")

        print(f"False Negative (FN) : {fn}")

        print(f"True Positive  (TP) : {tp}")

    # ========================================================
    # Analyze Prediction Errors
    # ========================================================

    def analyze_errors(self):

        tn = self.matrix[0][0]
        fp = self.matrix[0][1]
        fn = self.matrix[1][0]
        tp = self.matrix[1][1]

        print()
        print("=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)

        print()
        print(f"False Positive : {fp}")

        print("Students predicted as At Risk " "but actually Safe.")

        print()
        print(f"False Negative : {fn}")

        print("Students predicted as Safe " "but actually At Risk.")

        print()

        if fn <= 5:

            print("Status : LOW False Negative count.")

        else:

            print("Status : REVIEW False Negative count.")

        print()

        print(
            "For OJT Risk Prediction, "
            "False Negative is the critical error "
            "because an At Risk student may be "
            "predicted as Safe."
        )

    # ========================================================
    # Save Matrix CSV
    # ========================================================

    def save_matrix_csv(self):

        matrix_df = pd.DataFrame(
            self.matrix,
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"],
        )

        matrix_df.to_csv(CONFUSION_MATRIX_CSV)

        print()
        print("=" * 60)
        print("Saving Confusion Matrix Data...")
        print("=" * 60)

        print()
        print("CSV Saved Successfully.")

        print(f"CSV : {CONFUSION_MATRIX_CSV}")

    # ========================================================
    # Save Matrix Image
    # ========================================================

    def save_matrix_image(self):

        print()
        print("=" * 60)
        print("Generating Confusion Matrix Image...")
        print("=" * 60)

        fig, ax = plt.subplots()

        image = ax.imshow(self.matrix)

        ax.set_xlabel("Predicted Label")

        ax.set_ylabel("Actual Label")

        ax.set_title("Logistic Regression - Confusion Matrix")

        ax.set_xticks([0, 1])

        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Safe (0)", "At Risk (1)"])

        ax.set_yticklabels(["Safe (0)", "At Risk (1)"])

        for i in range(2):

            for j in range(2):

                ax.text(j, i, self.matrix[i, j], ha="center", va="center")

        fig.colorbar(image, ax=ax)

        fig.tight_layout()

        fig.savefig(CONFUSION_MATRIX_IMAGE, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print()
        print("Confusion Matrix Image Saved Successfully.")

        print(f"Image : {CONFUSION_MATRIX_IMAGE}")

    # ========================================================
    # Run Analysis
    # ========================================================

    def run(self):

        self.load_model()

        self.load_dataset()

        self.validate_dataset()

        self.generate_predictions()

        self.calculate_matrix()

        self.display_matrix()

        self.analyze_errors()

        self.save_matrix_csv()

        self.save_matrix_image()


# ============================================================
# Main
# ============================================================


def main():

    analyzer = ConfusionMatrixAnalyzer()

    analyzer.run()

    print()
    print("=" * 60)
    print("CONFUSION MATRIX ANALYSIS COMPLETED")
    print("=" * 60)


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()
