"""
============================================================
Classification Report
OJT AI Project

Generate detailed classification metrics for
Logistic Regression OJT Delay Risk Prediction.
============================================================
"""

import os
import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    accuracy_score
)


# ============================================================
# Add Project Root
# ============================================================

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


# ============================================================
# Project Paths
# ============================================================

from config.paths import PROCESSED_DATA_DIR


# ============================================================
# Input Paths
# ============================================================

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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "logistic_regression.pkl"
)


# ============================================================
# Output Directory
# ============================================================

EVALUATION_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "evaluation"
)

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)


# ============================================================
# Output Paths
# ============================================================

REPORT_TXT_PATH = os.path.join(
    EVALUATION_DIR,
    "logistic_classification_report.txt"
)

REPORT_CSV_PATH = os.path.join(
    EVALUATION_DIR,
    "logistic_classification_report.csv"
)


# ============================================================
# Classification Report Analyzer
# ============================================================

class ClassificationReportAnalyzer:

    def __init__(self):

        self.model = None

        self.X_test = None
        self.y_test = None

        self.y_pred = None

        self.report_dict = None
        self.report_text = None

        self.accuracy = None

    # ========================================================
    # Load Model
    # ========================================================

    def load_model(self):

        print()
        print("=" * 60)
        print("Loading Logistic Regression Model...")
        print("=" * 60)

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
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

    # ========================================================
    # Load Test Dataset
    # ========================================================

    def load_dataset(self):

        print()
        print("=" * 60)
        print("Loading Test Dataset...")
        print("=" * 60)

        if not os.path.exists(X_TEST_PATH):

            raise FileNotFoundError(
                f"X_test not found: {X_TEST_PATH}"
            )

        if not os.path.exists(Y_TEST_PATH):

            raise FileNotFoundError(
                f"y_test not found: {Y_TEST_PATH}"
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

    # ========================================================
    # Validate Dataset
    # ========================================================

    def validate_dataset(self):

        print()
        print("=" * 60)
        print("Validating Test Dataset...")
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
            "Test Dataset Validation Passed."
        )

    # ========================================================
    # Generate Predictions
    # ========================================================

    def generate_predictions(self):

        print()
        print("=" * 60)
        print("Generating Predictions...")
        print("=" * 60)

        self.y_pred = self.model.predict(
            self.X_test
        )

        print()
        print(
            "Predictions Generated Successfully."
        )

        print(
            f"Predictions : {len(self.y_pred)}"
        )

    # ========================================================
    # Generate Classification Report
    # ========================================================

    def generate_report(self):

        print()
        print("=" * 60)
        print("Generating Classification Report...")
        print("=" * 60)

        self.report_text = classification_report(
            self.y_test,
            self.y_pred,
            labels=[0, 1],
            target_names=[
                "Safe",
                "At Risk"
            ],
            digits=4,
            zero_division=0
        )

        self.report_dict = classification_report(
            self.y_test,
            self.y_pred,
            labels=[0, 1],
            target_names=[
                "Safe",
                "At Risk"
            ],
            output_dict=True,
            zero_division=0
        )

        self.accuracy = accuracy_score(
            self.y_test,
            self.y_pred
        )

        print()
        print(
            "Classification Report Generated Successfully."
        )

    # ========================================================
    # Display Report
    # ========================================================

    def display_report(self):

        print()
        print("=" * 60)
        print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
        print("=" * 60)

        print()
        print(
            self.report_text
        )

    # ========================================================
    # Analyze At Risk Class
    # ========================================================

    def analyze_at_risk_class(self):

        at_risk = self.report_dict["At Risk"]

        print()
        print("=" * 60)
        print("AT RISK CLASS ANALYSIS")
        print("=" * 60)

        print()
        print(
            f"Precision : {at_risk['precision']:.4f}"
        )

        print(
            f"Recall    : {at_risk['recall']:.4f}"
        )

        print(
            f"F1-Score  : {at_risk['f1-score']:.4f}"
        )

        print(
            f"Support   : {int(at_risk['support'])}"
        )

        print()

        print(
            "Recall is especially important because "
            "the system should identify students "
            "who are at risk of OJT delay."
        )

    # ========================================================
    # Save TXT Report
    # ========================================================

    def save_txt_report(self):

        print()
        print("=" * 60)
        print("Saving Classification Report...")
        print("=" * 60)

        with open(
            REPORT_TXT_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "LOGISTIC REGRESSION "
                "CLASSIFICATION REPORT\n"
            )

            file.write(
                "=" * 60
                + "\n\n"
            )

            file.write(
                f"Accuracy : {self.accuracy:.4f}\n\n"
            )

            file.write(
                self.report_text
            )

            file.write(
                "\n\n"
            )

            file.write(
                "At Risk Class Analysis\n"
            )

            file.write(
                "-" * 30
                + "\n"
            )

            at_risk = self.report_dict["At Risk"]

            file.write(
                f"Precision : "
                f"{at_risk['precision']:.4f}\n"
            )

            file.write(
                f"Recall    : "
                f"{at_risk['recall']:.4f}\n"
            )

            file.write(
                f"F1-Score  : "
                f"{at_risk['f1-score']:.4f}\n"
            )

            file.write(
                f"Support   : "
                f"{int(at_risk['support'])}\n"
            )

        print()
        print(
            "TXT Report Saved Successfully."
        )

        print(
            f"TXT : {REPORT_TXT_PATH}"
        )

    # ========================================================
    # Save CSV Report
    # ========================================================

    def save_csv_report(self):

        rows = []

        for label in [
            "Safe",
            "At Risk"
        ]:

            metrics = self.report_dict[label]

            rows.append({

                "Class": label,

                "Precision":
                    round(
                        metrics["precision"],
                        4
                    ),

                "Recall":
                    round(
                        metrics["recall"],
                        4
                    ),

                "F1_Score":
                    round(
                        metrics["f1-score"],
                        4
                    ),

                "Support":
                    int(
                        metrics["support"]
                    )

            })

        rows.append({

            "Class": "Accuracy",

            "Precision": "",

            "Recall": "",

            "F1_Score":
                round(
                    self.accuracy,
                    4
                ),

            "Support":
                len(self.y_test)

        })

        macro = self.report_dict[
            "macro avg"
        ]

        rows.append({

            "Class": "Macro Avg",

            "Precision":
                round(
                    macro["precision"],
                    4
                ),

            "Recall":
                round(
                    macro["recall"],
                    4
                ),

            "F1_Score":
                round(
                    macro["f1-score"],
                    4
                ),

            "Support":
                int(
                    macro["support"]
                )

        })

        weighted = self.report_dict[
            "weighted avg"
        ]

        rows.append({

            "Class": "Weighted Avg",

            "Precision":
                round(
                    weighted["precision"],
                    4
                ),

            "Recall":
                round(
                    weighted["recall"],
                    4
                ),

            "F1_Score":
                round(
                    weighted["f1-score"],
                    4
                ),

            "Support":
                int(
                    weighted["support"]
                )

        })

        report_df = pd.DataFrame(
            rows
        )

        report_df.to_csv(
            REPORT_CSV_PATH,
            index=False
        )

        print()
        print(
            "CSV Report Saved Successfully."
        )

        print(
            f"CSV : {REPORT_CSV_PATH}"
        )

    # ========================================================
    # Run Analysis
    # ========================================================

    def run(self):

        self.load_model()

        self.load_dataset()

        self.validate_dataset()

        self.generate_predictions()

        self.generate_report()

        self.display_report()

        self.analyze_at_risk_class()

        self.save_txt_report()

        self.save_csv_report()


# ============================================================
# Main
# ============================================================

def main():

    analyzer = ClassificationReportAnalyzer()

    analyzer.run()

    print()
    print("=" * 60)
    print(
        "CLASSIFICATION REPORT ANALYSIS COMPLETED"
    )
    print("=" * 60)


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()