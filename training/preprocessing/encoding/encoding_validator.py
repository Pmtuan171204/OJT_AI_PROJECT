"""
==========================================================
ENCODING VALIDATOR V2
OJT AI Project

Sprint 3 - Encoding Validation

Purpose:
    Validate encoded datasets before Machine Learning training.

Validation includes:
    1. Encoded files exist
    2. Missing values
    3. Train/Test feature count
    4. Train/Test feature names
    5. Numeric features only
    6. Target values
    7. Data leakage - forbidden columns
    8. Target separation
    9. Train/Test row alignment
    10. Infinite values
    11. Expected encoded feature count
    12. Dataset shape validation

Target:
    OJT_Delay_Outcome

Expected:
    X_train : 4000 rows
    X_test  : 1000 rows
    Original model features : 23
    Encoded features        : 38

==========================================================
"""

import os
import sys

import pandas as pd

# ==========================================================
# Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

if PROJECT_ROOT not in sys.path:

    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Config
# ==========================================================

from config.paths import PROCESSED_DATA_DIR

# ==========================================================
# Constants
# ==========================================================

TARGET_COLUMN = "OJT_Delay_Outcome"

EXPECTED_TRAIN_ROWS = 4000

EXPECTED_TEST_ROWS = 1000

EXPECTED_ORIGINAL_FEATURES = 23

EXPECTED_ENCODED_FEATURES = 38


# ==========================================================
# Encoded Dataset Paths
# ==========================================================

ENCODED_DIR = os.path.join(PROCESSED_DATA_DIR, "encoded")

X_TRAIN_PATH = os.path.join(ENCODED_DIR, "X_train_encoded.csv")

X_TEST_PATH = os.path.join(ENCODED_DIR, "X_test_encoded.csv")

Y_TRAIN_PATH = os.path.join(ENCODED_DIR, "y_train.csv")

Y_TEST_PATH = os.path.join(ENCODED_DIR, "y_test.csv")


# ==========================================================
# Forbidden / Leakage Columns
# ==========================================================

FORBIDDEN_COLUMNS = [
    # ------------------------------------------------------
    # ID / Identifier
    # ------------------------------------------------------
    "MSSV",
    # ------------------------------------------------------
    # Risk / Business Rule Outputs
    # ------------------------------------------------------
    "Risk_Score",
    "Risk_Level",
    "OJT_Delay_Risk",
    "OJT_Eligible",
    "OJT_Readiness",
    "AI_Recommendation",
    # ------------------------------------------------------
    # Future Outcome Columns
    # ------------------------------------------------------
    "Future_Credits_At_OJT",
    "Future_Failed_Courses",
    "Future_Missing_Prerequisites",
    "Future_Academic_Warning",
    "Future_OJT_Eligible",
    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------
    "OJT_Delay_Outcome",
]


# ==========================================================
# Result Helper
# ==========================================================


def create_result(test_name, status, message):

    return {"test": test_name, "status": status, "message": message}


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

        # --------------------------------------------------
        # Check all files first
        # --------------------------------------------------

        files = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]

        for file_path in files:

            if not os.path.exists(file_path):

                raise FileNotFoundError(
                    "Required encoded dataset file " "not found:\n" f"{file_path}"
                )

        # --------------------------------------------------
        # Load datasets
        # --------------------------------------------------

        self.X_train = pd.read_csv(X_TRAIN_PATH)

        self.X_test = pd.read_csv(X_TEST_PATH)

        self.y_train = pd.read_csv(Y_TRAIN_PATH)

        self.y_test = pd.read_csv(Y_TEST_PATH)

        print()

        print("Encoded Dataset Loaded Successfully.")

        print(f"X_train Rows : " f"{len(self.X_train)}")

        print(f"X_test Rows  : " f"{len(self.X_test)}")

        print(f"X_train Features : " f"{len(self.X_train.columns)}")

        print(f"X_test Features  : " f"{len(self.X_test.columns)}")

    # ======================================================
    # Test 1
    # Encoded Files Exist
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
            "y_test.csv": Y_TEST_PATH,
        }

        missing_files = []

        for name, path in files.items():

            if not os.path.exists(path):

                missing_files.append(name)

        if len(missing_files) == 0:

            result = create_result(
                "Encoded Files", True, "All encoded dataset files exist."
            )

        else:

            result = create_result(
                "Encoded Files", False, "Missing files: " + ", ".join(missing_files)
            )

        self.results.append(result)

    # ======================================================
    # Test 2
    # Missing Values
    # ======================================================

    def check_missing_values(self):

        train_missing = self.X_train.isnull().sum().sum()

        test_missing = self.X_test.isnull().sum().sum()

        y_train_missing = self.y_train.isnull().sum().sum()

        y_test_missing = self.y_test.isnull().sum().sum()

        total_missing = train_missing + test_missing + y_train_missing + y_test_missing

        if total_missing == 0:

            result = create_result("Missing Values", True, "No missing values found.")

        else:

            result = create_result(
                "Missing Values", False, f"{total_missing} " "missing values found."
            )

        self.results.append(result)

    # ======================================================
    # Test 3
    # Train/Test Feature Count
    # ======================================================

    def check_feature_count(self):

        train_count = len(self.X_train.columns)

        test_count = len(self.X_test.columns)

        if train_count == test_count:

            result = create_result(
                "Feature Count",
                True,
                f"Both datasets contain " f"{train_count} encoded features.",
            )

        else:

            result = create_result(
                "Feature Count",
                False,
                f"Train has "
                f"{train_count} features "
                f"but test has "
                f"{test_count}.",
            )

        self.results.append(result)

    # ======================================================
    # Test 4
    # Train/Test Feature Names
    # ======================================================

    def check_feature_names(self):

        train_columns = self.X_train.columns.tolist()

        test_columns = self.X_test.columns.tolist()

        if train_columns == test_columns:

            result = create_result(
                "Feature Names",
                True,
                "Train and test encoded " "feature columns match.",
            )

        else:

            missing_in_test = [
                column for column in train_columns if column not in test_columns
            ]

            missing_in_train = [
                column for column in test_columns if column not in train_columns
            ]

            result = create_result(
                "Feature Names",
                False,
                f"Column mismatch. "
                f"Missing in test: "
                f"{missing_in_test}. "
                f"Missing in train: "
                f"{missing_in_train}.",
            )

        self.results.append(result)

    # ======================================================
    # Test 5
    # Numeric Features Only
    # ======================================================

    def check_numeric_features(self):

        non_numeric_train = self.X_train.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        non_numeric_test = self.X_test.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        if len(non_numeric_train) == 0 and len(non_numeric_test) == 0:

            result = create_result(
                "Numeric Features", True, "All encoded X features " "are numeric."
            )

        else:

            result = create_result(
                "Numeric Features",
                False,
                "Non-numeric features found. "
                f"Train: {non_numeric_train}. "
                f"Test: {non_numeric_test}.",
            )

        self.results.append(result)

    # ======================================================
    # Test 6
    # Target Values
    # ======================================================

    def check_target_values(self):

        # --------------------------------------------------
        # Check target column names
        # --------------------------------------------------

        train_target_name = self.y_train.columns[0]

        test_target_name = self.y_test.columns[0]

        # --------------------------------------------------
        # Check target name
        # --------------------------------------------------

        if train_target_name != TARGET_COLUMN or test_target_name != TARGET_COLUMN:

            result = create_result(
                "Target Values",
                False,
                "Target column name is incorrect. "
                f"Expected: {TARGET_COLUMN}. "
                f"Train: {train_target_name}. "
                f"Test: {test_target_name}.",
            )

            self.results.append(result)

            return

        # --------------------------------------------------
        # Get target values
        # --------------------------------------------------

        train_values = set(self.y_train[TARGET_COLUMN].dropna().unique())

        test_values = set(self.y_test[TARGET_COLUMN].dropna().unique())

        valid_values = {0, 1}

        train_valid = train_values.issubset(valid_values)

        test_valid = test_values.issubset(valid_values)

        if train_valid and test_valid:

            result = create_result(
                "Target Values", True, f"{TARGET_COLUMN} " "contains only 0 and 1."
            )

        else:

            result = create_result(
                "Target Values",
                False,
                f"Invalid target values. "
                f"Train: {train_values}. "
                f"Test: {test_values}.",
            )

        self.results.append(result)

    # ======================================================
    # Test 7
    # Data Leakage - Forbidden Columns
    # ======================================================

    def check_forbidden_columns(self):

        # --------------------------------------------------
        # Check columns in encoded X
        # --------------------------------------------------

        all_columns = self.X_train.columns.tolist() + self.X_test.columns.tolist()

        found_columns = [
            column for column in FORBIDDEN_COLUMNS if column in all_columns
        ]

        if len(found_columns) == 0:

            result = create_result(
                "Data Leakage",
                True,
                "No forbidden risk, business-rule, "
                "future-outcome, ID, or target "
                "columns are present in encoded X.",
            )

        else:

            result = create_result(
                "Data Leakage",
                False,
                "Potential leakage columns found: " f"{found_columns}",
            )

        self.results.append(result)

    # ======================================================
    # Test 8
    # Target Separation
    # ======================================================

    def check_target_separation(self):

        train_contains_target = TARGET_COLUMN in self.X_train.columns

        test_contains_target = TARGET_COLUMN in self.X_test.columns

        if not train_contains_target and not test_contains_target:

            result = create_result(
                "Target Separation",
                True,
                f"{TARGET_COLUMN} " "is correctly separated from X.",
            )

        else:

            result = create_result(
                "Target Separation", False, f"{TARGET_COLUMN} " "is still present in X."
            )

        self.results.append(result)

    # ======================================================
    # Test 9
    # Train/Test Row Count
    # ======================================================

    def check_row_count(self):

        train_rows = len(self.X_train)

        test_rows = len(self.X_test)

        y_train_rows = len(self.y_train)

        y_test_rows = len(self.y_test)

        train_match = train_rows == y_train_rows

        test_match = test_rows == y_test_rows

        if train_match and test_match:

            result = create_result("Row Alignment", True, "X and y row counts match.")

        else:

            result = create_result(
                "Row Alignment",
                False,
                f"Row mismatch. "
                f"X_train={train_rows}, "
                f"y_train={y_train_rows}, "
                f"X_test={test_rows}, "
                f"y_test={y_test_rows}.",
            )

        self.results.append(result)

    # ======================================================
    # Test 10
    # Infinite Values
    # ======================================================

    def check_infinite_values(self):

        train_infinite = self.X_train.isin([float("inf"), float("-inf")]).sum().sum()

        test_infinite = self.X_test.isin([float("inf"), float("-inf")]).sum().sum()

        total_infinite = train_infinite + test_infinite

        if total_infinite == 0:

            result = create_result("Infinite Values", True, "No infinite values found.")

        else:

            result = create_result(
                "Infinite Values", False, f"{total_infinite} " "infinite values found."
            )

        self.results.append(result)

    # ======================================================
    # Test 11
    # Expected Encoded Feature Count
    # ======================================================

    def check_expected_feature_count(self):

        actual_count = len(self.X_train.columns)

        if actual_count == EXPECTED_ENCODED_FEATURES:

            result = create_result(
                "Encoded Feature Count",
                True,
                f"Encoded dataset contains "
                f"the expected "
                f"{EXPECTED_ENCODED_FEATURES} features.",
            )

        else:

            result = create_result(
                "Encoded Feature Count",
                False,
                f"Expected "
                f"{EXPECTED_ENCODED_FEATURES} "
                f"encoded features but found "
                f"{actual_count}.",
            )

        self.results.append(result)

    # ======================================================
    # Test 12
    # Expected Dataset Shape
    # ======================================================

    def check_expected_dataset_shape(self):

        train_rows = len(self.X_train)

        test_rows = len(self.X_test)

        if train_rows == EXPECTED_TRAIN_ROWS and test_rows == EXPECTED_TEST_ROWS:

            result = create_result(
                "Dataset Shape",
                True,
                f"Expected shape confirmed: "
                f"train={EXPECTED_TRAIN_ROWS}, "
                f"test={EXPECTED_TEST_ROWS}.",
            )

        else:

            result = create_result(
                "Dataset Shape",
                False,
                f"Unexpected shape. "
                f"Expected train={EXPECTED_TRAIN_ROWS}, "
                f"test={EXPECTED_TEST_ROWS}; "
                f"found train={train_rows}, "
                f"test={test_rows}.",
            )

        self.results.append(result)

    # ======================================================
    # Run All Tests
    # ======================================================

    def validate(self):

        # --------------------------------------------------
        # Test 1 must run first
        # --------------------------------------------------

        self.check_files_exist()

        # --------------------------------------------------
        # Stop if files do not exist
        # --------------------------------------------------

        if not self.results[-1]["status"]:

            return

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------

        self.load_dataset()

        # --------------------------------------------------
        # Run remaining tests
        # --------------------------------------------------

        self.check_missing_values()

        self.check_feature_count()

        self.check_feature_names()

        self.check_numeric_features()

        self.check_target_values()

        self.check_forbidden_columns()

        self.check_target_separation()

        self.check_row_count()

        self.check_infinite_values()

        self.check_expected_feature_count()

        self.check_expected_dataset_shape()

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

            status = "PASS" if result["status"] else "FAIL"

            print(f"[{status}] " f"{result['test']} " f"-> {result['message']}")

            if result["status"]:

                passed += 1

            else:

                failed += 1

        print()

        print("=" * 60)

        print(f"Passed : {passed}")

        print(f"Failed : {failed}")

        print(f"Total  : {len(self.results)}")

        print("=" * 60)

        # --------------------------------------------------
        # Final status
        # --------------------------------------------------

        print()

        if failed == 0:

            print("ENCODING VALIDATION: PASS")

            print("Encoded dataset is ready " "for Machine Learning training.")

        else:

            print("ENCODING VALIDATION: FAIL")

            print("Please fix the failed checks " "before Machine Learning training.")


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
