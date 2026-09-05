"""
==========================================================
Data Leakage Check
OJT AI Project

Check whether training features contain direct or
indirect information about the target variable.

Target:
    OJT_Delay_Risk

This module is used before finalizing ML models.
==========================================================
"""

import os
import sys

import pandas as pd

# ==========================================================
# PROJECT ROOT
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# PATHS
# ==========================================================

from config.paths import ENCODED_DATA_DIR, EVALUATION_REPORT_DIR

# ==========================================================
# DATA FILES
# ==========================================================

X_TRAIN_PATH = os.path.join(ENCODED_DATA_DIR, "X_train_encoded.csv")

Y_TRAIN_PATH = os.path.join(ENCODED_DATA_DIR, "y_train.csv")


# ==========================================================
# TARGET
# ==========================================================

TARGET_COLUMN = "OJT_Delay_Risk"


# ==========================================================
# KNOWN LEAKAGE FEATURES
# ==========================================================

DIRECT_LEAKAGE_FEATURES = ["OJT_Delay_Risk", "Risk_Score", "Risk_Level"]


# ==========================================================
# TARGET-RELATED FEATURES
# ==========================================================

TARGET_RELATED_FEATURES = [
    "Risk_Score",
    "Risk_Level",
    "OJT_Delay_Risk",
    "Academic_Risk_Index",
    "Academic_Stability",
]


# ==========================================================
# CORRELATION THRESHOLD
# ==========================================================

HIGH_CORRELATION_THRESHOLD = 0.90


# ==========================================================
# DATA LEAKAGE CHECKER
# ==========================================================


class DataLeakageChecker:

    def __init__(self):

        self.X_train = None
        self.y_train = None

        self.results = []

        self.leakage_features = []
        self.high_correlation_features = []

    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_dataset(self):

        print()
        print("=" * 60)
        print("Loading Training Dataset...")
        print("=" * 60)

        if not os.path.exists(X_TRAIN_PATH):
            raise FileNotFoundError(f"X_train file not found:\n{X_TRAIN_PATH}")

        if not os.path.exists(Y_TRAIN_PATH):
            raise FileNotFoundError(f"y_train file not found:\n{Y_TRAIN_PATH}")

        self.X_train = pd.read_csv(X_TRAIN_PATH)

        self.y_train = pd.read_csv(Y_TRAIN_PATH)

        print()
        print("Training Dataset Loaded Successfully.")
        print(f"X_train Rows : {len(self.X_train)}")
        print(f"X_train Features : {self.X_train.shape[1]}")
        print(f"y_train Rows : {len(self.y_train)}")

    # ======================================================
    # CHECK TARGET SEPARATION
    # ======================================================

    def check_target_separation(self):

        print()
        print("=" * 60)
        print("Checking Target Separation...")
        print("=" * 60)

        if TARGET_COLUMN in self.X_train.columns:

            self.results.append(
                ("Target Separation", False, "Target column is present in X_train.")
            )

            self.leakage_features.append(TARGET_COLUMN)

            print("[FAIL] Target Separation")
            print(f"Target '{TARGET_COLUMN}' found in X_train.")

        else:

            self.results.append(
                ("Target Separation", True, "Target column is correctly separated.")
            )

            print("[PASS] Target Separation")
            print(f"Target '{TARGET_COLUMN}' is not in X_train.")

    # ======================================================
    # CHECK KNOWN LEAKAGE FEATURES
    # ======================================================

    def check_direct_leakage(self):

        print()
        print("=" * 60)
        print("Checking Direct Leakage Features...")
        print("=" * 60)

        found = []

        for feature in DIRECT_LEAKAGE_FEATURES:

            if feature in self.X_train.columns:

                found.append(feature)

        if found:

            self.leakage_features.extend(found)

            self.results.append(
                ("Direct Leakage", False, f"Potential leakage features found: {found}")
            )

            print("[FAIL] Direct Leakage")

            for feature in found:
                print(f" - {feature}")

        else:

            self.results.append(
                ("Direct Leakage", True, "No known direct leakage features found.")
            )

            print("[PASS] Direct Leakage")

    # ======================================================
    # CHECK TARGET RELATED FEATURES
    # ======================================================

    def check_target_related_features(self):

        print()
        print("=" * 60)
        print("Checking Target-Related Features...")
        print("=" * 60)

        found = []

        for feature in TARGET_RELATED_FEATURES:

            if feature in self.X_train.columns:

                found.append(feature)

        if found:

            self.results.append(
                (
                    "Target Related Features",
                    False,
                    f"Target-related features found: {found}",
                )
            )

            print("[WARNING] Target Related Features")

            for feature in found:

                print(f" - {feature}")

        else:

            self.results.append(
                ("Target Related Features", True, "No target-related features found.")
            )

            print("[PASS] Target Related Features")

    # ======================================================
    # CHECK NUMERICAL CORRELATION
    # ======================================================

    def check_correlation(self):

        print()
        print("=" * 60)
        print("Checking Feature Correlation...")
        print("=" * 60)

        data = self.X_train.copy()

        data[TARGET_COLUMN] = self.y_train.iloc[:, 0].values

        numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()

        if TARGET_COLUMN in numerical_columns:
            numerical_columns.remove(TARGET_COLUMN)

        correlation = data[numerical_columns + [TARGET_COLUMN]].corr()[TARGET_COLUMN]

        correlation = correlation.drop(TARGET_COLUMN)

        high_corr = correlation[
            correlation.abs() >= HIGH_CORRELATION_THRESHOLD
        ].sort_values(key=abs, ascending=False)

        print()
        print(f"Correlation Threshold : " f"{HIGH_CORRELATION_THRESHOLD}")

        if len(high_corr) == 0:

            self.results.append(
                (
                    "Feature Correlation",
                    True,
                    "No feature has suspiciously high correlation " "with the target.",
                )
            )

            print("[PASS] Feature Correlation")

        else:

            self.high_correlation_features = high_corr.index.tolist()

            self.results.append(
                ("Feature Correlation", False, "High correlation features detected.")
            )

            print("[WARNING] High Correlation Features")

            for feature, value in high_corr.items():

                print(f" - {feature}: {value:.4f}")

    # ======================================================
    # CHECK DATASET CONSISTENCY
    # ======================================================

    def check_row_alignment(self):

        print()
        print("=" * 60)
        print("Checking Row Alignment...")
        print("=" * 60)

        if len(self.X_train) == len(self.y_train):

            self.results.append(
                ("Row Alignment", True, "X_train and y_train row counts match.")
            )

            print("[PASS] Row Alignment")

        else:

            self.results.append(
                ("Row Alignment", False, "X_train and y_train row counts do not match.")
            )

            print("[FAIL] Row Alignment")

    # ======================================================
    # RUN ALL CHECKS
    # ======================================================

    def run_checks(self):

        self.load_dataset()

        self.check_row_alignment()

        self.check_target_separation()

        self.check_direct_leakage()

        self.check_target_related_features()

        self.check_correlation()

    # ======================================================
    # PRINT SUMMARY
    # ======================================================

    def print_summary(self):

        print()
        print("=" * 60)
        print("DATA LEAKAGE CHECK SUMMARY")
        print("=" * 60)

        passed = 0
        failed = 0

        for rule, status, message in self.results:

            if status:

                print(f"[PASS] {rule} -> {message}")

                passed += 1

            else:

                print(f"[WARNING/FAIL] {rule} -> {message}")

                failed += 1

        print()
        print(f"Passed : {passed}")

        print(f"Warnings/Failed : {failed}")

        print(f"Total : {len(self.results)}")

        print()

        if self.leakage_features:

            print("Potential Leakage Features:")

            for feature in set(self.leakage_features):

                print(f" - {feature}")

        else:

            print("Potential Direct Leakage Features : None")

        print()

        if self.high_correlation_features:

            print("High Correlation Features:")

            for feature in self.high_correlation_features:

                print(f" - {feature}")

        else:

            print("High Correlation Features : None")

        print()

        if TARGET_COLUMN not in self.X_train.columns and not self.leakage_features:

            print("STATUS : NO DIRECT DATA LEAKAGE DETECTED")

        else:

            print("STATUS : POTENTIAL DATA LEAKAGE DETECTED")

        print("=" * 60)

    # ======================================================
    # SAVE REPORT
    # ======================================================

    def save_report(self):

        os.makedirs(EVALUATION_REPORT_DIR, exist_ok=True)

        report_path = os.path.join(EVALUATION_REPORT_DIR, "data_leakage_report.txt")

        with open(report_path, "w", encoding="utf-8") as file:

            file.write("DATA LEAKAGE CHECK REPORT\n")

            file.write("=" * 60 + "\n\n")

            for rule, status, message in self.results:

                status_text = "PASS" if status else "WARNING/FAIL"

                file.write(f"[{status_text}] " f"{rule} -> {message}\n")

            file.write("\n")

            file.write("Potential Leakage Features:\n")

            if self.leakage_features:

                for feature in set(self.leakage_features):

                    file.write(f" - {feature}\n")

            else:

                file.write(" - None\n")

            file.write("\n")

            file.write("High Correlation Features:\n")

            if self.high_correlation_features:

                for feature in self.high_correlation_features:

                    file.write(f" - {feature}\n")

            else:

                file.write(" - None\n")

        print()
        print("Leakage Report Saved Successfully.")

        print(f"Report : {report_path}")


# ==========================================================
# MAIN
# ==========================================================


def main():

    print()
    print("=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)

    checker = DataLeakageChecker()

    checker.run_checks()

    checker.print_summary()

    checker.save_report()

    print()
    print("=" * 60)
    print("DATA LEAKAGE CHECK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
