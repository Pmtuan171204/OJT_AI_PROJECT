"""
==========================================================
ENCODING VALIDATOR
OJT AI Project

Validate encoded datasets before Machine Learning training.
==========================================================
"""

import os
import sys

import pandas as pd


# ==========================================================
# Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Config
# ==========================================================

from config.paths import (
    PROCESSED_DATA_DIR
)


# ==========================================================
# Encoded Dataset Paths
# ==========================================================

ENCODED_DIR = os.path.join(
    PROCESSED_DATA_DIR,
    "encoded"
)

X_TRAIN_PATH = os.path.join(
    ENCODED_DIR,
    "X_train_encoded.csv"
)

X_TEST_PATH = os.path.join(
    ENCODED_DIR,
    "X_test_encoded.csv"
)

Y_TRAIN_PATH = os.path.join(
    ENCODED_DIR,
    "y_train.csv"
)

Y_TEST_PATH = os.path.join(
    ENCODED_DIR,
    "y_test.csv"
)


# ==========================================================
# Result Helper
# ==========================================================

def create_result(
    test_name,
    status,
    message
):

    return {
        "test": test_name,
        "status": status,
        "message": message
    }


# ==========================================================
# Encoding Validator
# ==========================================================

class EncodingValidator:

    def __init__(self):

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.results = []


    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        print()
        print("=" * 60)
        print("Loading Encoded Dataset...")
        print("=" * 60)

        self.X_train = pd.read_csv(
            X_TRAIN_PATH
        )

        self.X_test = pd.read_csv(
            X_TEST_PATH
        )

        self.y_train = pd.read_csv(
            Y_TRAIN_PATH
        )

        self.y_test = pd.read_csv(
            Y_TEST_PATH
        )

        print()

        print(
            "Encoded Dataset Loaded Successfully."
        )

        print(
            f"X_train Rows : {len(self.X_train)}"
        )

        print(
            f"X_test Rows  : {len(self.X_test)}"
        )

        print(
            f"X_train Features : "
            f"{len(self.X_train.columns)}"
        )

        print(
            f"X_test Features  : "
            f"{len(self.X_test.columns)}"
        )


    # ======================================================
    # Test 1
    # File Exists
    # ======================================================

    def check_files_exist(self):

        print()
        print("=" * 60)
        print("Checking Encoded Dataset Files...")
        print("=" * 60)

        files = {
            "X_train_encoded.csv": X_TRAIN_PATH,
            "X_test_encoded.csv": X_TEST_PATH,
            "y_train.csv": Y_TRAIN_PATH,
            "y_test.csv": Y_TEST_PATH
        }

        missing_files = []

        for name, path in files.items():

            if not os.path.exists(path):

                missing_files.append(name)

        if len(missing_files) == 0:

            result = create_result(
                "Encoded Files",
                True,
                "All encoded dataset files exist."
            )

        else:

            result = create_result(
                "Encoded Files",
                False,
                "Missing files: "
                + ", ".join(missing_files)
            )

        self.results.append(result)


    # ======================================================
    # Test 2
    # Missing Values
    # ======================================================

    def check_missing_values(self):

        train_missing = (
            self.X_train.isnull().sum().sum()
        )

        test_missing = (
            self.X_test.isnull().sum().sum()
        )

        y_train_missing = (
            self.y_train.isnull().sum().sum()
        )

        y_test_missing = (
            self.y_test.isnull().sum().sum()
        )

        total_missing = (
            train_missing
            + test_missing
            + y_train_missing
            + y_test_missing
        )

        if total_missing == 0:

            result = create_result(
                "Missing Values",
                True,
                "No missing values found."
            )

        else:

            result = create_result(
                "Missing Values",
                False,
                f"{total_missing} missing values found."
            )

        self.results.append(result)


    # ======================================================
    # Test 3
    # Train/Test Feature Count
    # ======================================================

    def check_feature_count(self):

        train_count = len(
            self.X_train.columns
        )

        test_count = len(
            self.X_test.columns
        )

        if train_count == test_count:

            result = create_result(
                "Feature Count",
                True,
                f"Both datasets contain "
                f"{train_count} features."
            )

        else:

            result = create_result(
                "Feature Count",
                False,
                f"Train has {train_count} features "
                f"but test has {test_count}."
            )

        self.results.append(result)


    # ======================================================
    # Test 4
    # Train/Test Feature Names
    # ======================================================

    def check_feature_names(self):

        train_columns = (
            self.X_train.columns.tolist()
        )

        test_columns = (
            self.X_test.columns.tolist()
        )

        if train_columns == test_columns:

            result = create_result(
                "Feature Names",
                True,
                "Train and test feature columns match."
            )

        else:

            missing_in_test = list(
                set(train_columns)
                - set(test_columns)
            )

            missing_in_train = list(
                set(test_columns)
                - set(train_columns)
            )

            result = create_result(
                "Feature Names",
                False,
                f"Column mismatch. "
                f"Missing in test: {missing_in_test}. "
                f"Missing in train: {missing_in_train}."
            )

        self.results.append(result)


    # ======================================================
    # Test 5
    # Numeric Features Only
    # ======================================================

    def check_numeric_features(self):

        non_numeric_train = (
            self.X_train.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
        )

        non_numeric_test = (
            self.X_test.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
        )

        if (
            len(non_numeric_train) == 0
            and len(non_numeric_test) == 0
        ):

            result = create_result(
                "Numeric Features",
                True,
                "All X features are numeric."
            )

        else:

            result = create_result(
                "Numeric Features",
                False,
                f"Non-numeric features found. "
                f"Train: {non_numeric_train}. "
                f"Test: {non_numeric_test}."
            )

        self.results.append(result)


    # ======================================================
    # Test 6
    # Target Values
    # ======================================================

    def check_target_values(self):

        train_values = set(
            self.y_train.iloc[:, 0]
            .dropna()
            .unique()
        )

        test_values = set(
            self.y_test.iloc[:, 0]
            .dropna()
            .unique()
        )

        valid_values = {0, 1}

        train_valid = (
            train_values.issubset(
                valid_values
            )
        )

        test_valid = (
            test_values.issubset(
                valid_values
            )
        )

        if train_valid and test_valid:

            result = create_result(
                "Target Values",
                True,
                "Target contains only 0 and 1."
            )

        else:

            result = create_result(
                "Target Values",
                False,
                f"Invalid target values. "
                f"Train: {train_values}. "
                f"Test: {test_values}."
            )

        self.results.append(result)


    # ======================================================
    # Test 7
    # Data Leakage - Risk Columns
    # ======================================================

    def check_risk_columns(self):

        forbidden_columns = [
            "Risk_Score",
            "Risk_Level",
            "OJT_Delay_Risk"
        ]

        found_columns = [
            column
            for column in forbidden_columns
            if column in self.X_train.columns
            or column in self.X_test.columns
        ]

        if len(found_columns) == 0:

            result = create_result(
                "Data Leakage",
                True,
                "Risk and target columns are "
                "not present in X."
            )

        else:

            result = create_result(
                "Data Leakage",
                False,
                f"Potential leakage columns found: "
                f"{found_columns}"
            )

        self.results.append(result)


    # ======================================================
    # Test 8
    # Target Separation
    # ======================================================

    def check_target_separation(self):

        target_column = "OJT_Delay_Risk"

        if target_column not in self.X_train.columns:

            result = create_result(
                "Target Separation",
                True,
                "Target is correctly separated from X."
            )

        else:

            result = create_result(
                "Target Separation",
                False,
                "OJT_Delay_Risk is still present in X."
            )

        self.results.append(result)


    # ======================================================
    # Test 9
    # Train/Test Row Count
    # ======================================================

    def check_row_count(self):

        train_rows = len(
            self.X_train
        )

        test_rows = len(
            self.X_test
        )

        y_train_rows = len(
            self.y_train
        )

        y_test_rows = len(
            self.y_test
        )

        train_match = (
            train_rows == y_train_rows
        )

        test_match = (
            test_rows == y_test_rows
        )

        if train_match and test_match:

            result = create_result(
                "Row Alignment",
                True,
                "X and y row counts match."
            )

        else:

            result = create_result(
                "Row Alignment",
                False,
                f"Row mismatch. "
                f"X_train={train_rows}, "
                f"y_train={y_train_rows}, "
                f"X_test={test_rows}, "
                f"y_test={y_test_rows}."
            )

        self.results.append(result)


    # ======================================================
    # Test 10
    # Infinite Values
    # ======================================================

    def check_infinite_values(self):

        train_infinite = (
            self.X_train
            .select_dtypes(include=["number"])
            .isin([float("inf"), float("-inf")])
            .sum()
            .sum()
        )

        test_infinite = (
            self.X_test
            .select_dtypes(include=["number"])
            .isin([float("inf"), float("-inf")])
            .sum()
            .sum()
        )

        total_infinite = (
            train_infinite
            + test_infinite
        )

        if total_infinite == 0:

            result = create_result(
                "Infinite Values",
                True,
                "No infinite values found."
            )

        else:

            result = create_result(
                "Infinite Values",
                False,
                f"{total_infinite} infinite values found."
            )

        self.results.append(result)


    # ======================================================
    # Run All Tests
    # ======================================================

    def validate(self):

        self.check_files_exist()

        if not self.results[-1]["status"]:

            return

        self.load_dataset()

        self.check_missing_values()

        self.check_feature_count()

        self.check_feature_names()

        self.check_numeric_features()

        self.check_target_values()

        self.check_risk_columns()

        self.check_target_separation()

        self.check_row_count()

        self.check_infinite_values()


    # ======================================================
    # Print Result
    # ======================================================

    def print_result(self):

        print()

        print("=" * 60)
        print("ENCODING VALIDATION RESULT")
        print("=" * 60)

        passed = 0
        failed = 0

        for result in self.results:

            status = (
                "PASS"
                if result["status"]
                else "FAIL"
            )

            print(
                f"[{status}] "
                f"{result['test']} "
                f"-> {result['message']}"
            )

            if result["status"]:

                passed += 1

            else:

                failed += 1

        print()

        print("=" * 60)

        print(
            f"Passed : {passed}"
        )

        print(
            f"Failed : {failed}"
        )

        print(
            f"Total  : {len(self.results)}"
        )

        print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    print()

    print("Encoded Dataset Path:")
    print(ENCODED_DIR)

    validator = EncodingValidator()

    validator.validate()

    validator.print_result()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()