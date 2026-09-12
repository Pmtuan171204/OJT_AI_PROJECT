"""
==========================================================
DATA SPLITTING V2
OJT AI Project

Sprint 3 - Data Splitting

Purpose:
    Split the feature-engineered dataset into:

        X_train
        X_test
        y_train
        y_test

Target:
    OJT_Delay_Outcome

Approach:
    Genuine Predictive AI

Important:
    The model must NOT use:
        - MSSV
        - Risk_Score
        - Risk_Level
        - OJT_Delay_Risk
        - OJT_Eligible
        - OJT_Readiness
        - AI_Recommendation
        - Future outcome columns
        - Target column

    The model uses only current-state features
    and approved engineered features.

Split:
    Training = 80%
    Testing  = 20%

Random State:
    42

==========================================================
"""

import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================================
# Add Project Root Into Python Path
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

if PROJECT_ROOT not in sys.path:

    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Import Project Paths
# ==========================================================

from config.paths import (
    FEATURE_DATASET_XLSX,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    SPLIT_DATA_DIR,
)

# ==========================================================
# Configuration
# ==========================================================

TARGET_COLUMN = "OJT_Delay_Outcome"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ==========================================================
# Data Splitter
# ==========================================================


class DataSplitter:

    def __init__(self):

        # --------------------------------------------------
        # Input
        # --------------------------------------------------

        self.input_file = FEATURE_DATASET_XLSX

        # --------------------------------------------------
        # Output Folder
        # --------------------------------------------------

        self.output_folder = SPLIT_DATA_DIR

        # --------------------------------------------------
        # Target
        # --------------------------------------------------

        self.target_column = TARGET_COLUMN

        # --------------------------------------------------
        # Split Configuration
        # --------------------------------------------------

        self.test_size = TEST_SIZE

        self.random_state = RANDOM_STATE

        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------

        self.df = None

        self.X = None

        self.y = None

        # --------------------------------------------------
        # Split Dataset
        # --------------------------------------------------

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

    # ======================================================
    # Load Feature Dataset
    # ======================================================

    def load_dataset(self):

        print()

        print("=" * 60)

        print("Loading Feature Dataset...")

        print("=" * 60)

        # --------------------------------------------------
        # Check input file
        # --------------------------------------------------

        if not os.path.exists(self.input_file):

            raise FileNotFoundError(
                "Feature dataset not found.\n" f"Expected path:\n" f"{self.input_file}"
            )

        # --------------------------------------------------
        # Load Excel
        # --------------------------------------------------

        self.df = pd.read_excel(self.input_file)

        print()

        print("Feature Dataset Loaded Successfully.")

        print(f"Rows    : {len(self.df)}")

        print(f"Columns : {len(self.df.columns)}")

    # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_dataset(self):

        print()

        print("=" * 60)

        print("Validating Feature Dataset...")

        print("=" * 60)

        # --------------------------------------------------
        # Check empty dataset
        # --------------------------------------------------

        if self.df.empty:

            raise ValueError("Feature dataset is empty.")

        # --------------------------------------------------
        # Check duplicate columns
        # --------------------------------------------------

        duplicated_columns = self.df.columns[self.df.columns.duplicated()].tolist()

        if len(duplicated_columns) > 0:

            raise ValueError(
                "Duplicated columns found:\n"
                + "\n".join(f" - {column}" for column in duplicated_columns)
            )

        # --------------------------------------------------
        # Check missing values
        # --------------------------------------------------

        missing_values = self.df.isnull().sum()

        missing_columns = missing_values[missing_values > 0]

        if len(missing_columns) > 0:

            print()

            print("Missing Values Detected:")

            print(missing_columns)

            raise ValueError("Feature dataset contains " "missing values.")

        print()

        print("Dataset validation passed.")

    # ======================================================
    # Validate Target Column
    # ======================================================

    def validate_target(self):

        print()

        print("=" * 60)

        print("Validating Target Column...")

        print("=" * 60)

        # --------------------------------------------------
        # Target exists
        # --------------------------------------------------

        if self.target_column not in self.df.columns:

            raise ValueError(
                f"Target column " f"'{self.target_column}' " f"does not exist."
            )

        # --------------------------------------------------
        # Target missing values
        # --------------------------------------------------

        if self.df[self.target_column].isnull().any():

            raise ValueError("Target column contains " "missing values.")

        # --------------------------------------------------
        # Target values
        # --------------------------------------------------

        unique_values = sorted(self.df[self.target_column].unique())

        # --------------------------------------------------
        # Binary classification
        # --------------------------------------------------

        if not set(unique_values).issubset({0, 1}):

            raise ValueError(
                "Target column must contain " "only binary values 0 and 1."
            )

        print()

        print(f"Target Column : " f"{self.target_column}")

        print(f"Unique Values  : " f"{unique_values}")

        print()

        print("Target Distribution:")

        print(self.df[self.target_column].value_counts().sort_index())

    # ======================================================
    # Define Official Model Features
    # ======================================================

    def get_model_features(self):
        """
        Official model feature list.

        These features represent the student's
        current academic state and engineered
        current-state indicators.

        Target-related and future information
        is explicitly excluded.
        """

        model_features = [
            # ------------------------------------------------
            # Current Academic State
            # ------------------------------------------------
            "Student_Profile",
            "Current_Semester",
            "GPA_Cumulative",
            # ------------------------------------------------
            # Credit Progress
            # ------------------------------------------------
            "Total_Credits",
            "Credits_Completed",
            "Credits_Remaining",
            "Completion_Rate",
            "Average_Credits_Per_Semester",
            "Remaining_To_OJT",
            # ------------------------------------------------
            # Academic Performance
            # ------------------------------------------------
            "Failed_Courses",
            "Retake_Count",
            "Missing_Prerequisite_Courses",
            "Academic_Warning_Count",
            "Suspension_Count",
            # ------------------------------------------------
            # OJT Planning
            # ------------------------------------------------
            "Planned_OJT_Semester",
            # ------------------------------------------------
            # Engineered Features
            # ------------------------------------------------
            "Credit_Progress_Category",
            "OJT_Credit_Gap",
            "OJT_Credit_Progress",
            "OJT_Eligibility_Gap_Category",
            "OJT_Semester_Gap",
            "OJT_Planning_Status",
            "Required_Average_Credits_Per_Semester",
            "Credit_Completion_Efficiency",
        ]

        return model_features

    # ======================================================
    # Separate Features And Target
    # ======================================================

    def separate_features_target(self):

        print()

        print("=" * 60)

        print("Separating Features and Target...")

        print("=" * 60)

        # --------------------------------------------------
        # Official Model Features
        # --------------------------------------------------

        model_features = self.get_model_features()

        # --------------------------------------------------
        # Check missing model features
        # --------------------------------------------------

        missing_features = [
            feature for feature in model_features if feature not in self.df.columns
        ]

        if len(missing_features) > 0:

            raise ValueError(
                "Missing model features:\n"
                + "\n".join(f" - {feature}" for feature in missing_features)
            )

        # --------------------------------------------------
        # Create X
        # --------------------------------------------------

        self.X = self.df[model_features].copy()

        # --------------------------------------------------
        # Create y
        # --------------------------------------------------

        self.y = self.df[self.target_column].copy()

        # --------------------------------------------------
        # Convert target to integer
        # --------------------------------------------------

        self.y = self.y.astype(int)

        print()

        print(f"Feature Columns : " f"{len(self.X.columns)}")

        print(f"Target Column   : " f"{self.target_column}")

        print()

        print("Model Features:")

        for index, column in enumerate(self.X.columns, start=1):

            print(f"{index:02d}. {column}")

    # ======================================================
    # Leakage Check
    # ======================================================

    def check_leakage(self):

        print()

        print("=" * 60)

        print("Checking Data Leakage...")

        print("=" * 60)

        # --------------------------------------------------
        # Columns that must NEVER enter X
        # --------------------------------------------------

        forbidden_columns = [
            # Identity
            "MSSV",
            # Existing risk outputs
            "Risk_Score",
            "Risk_Level",
            "OJT_Delay_Risk",
            # Existing business-rule outputs
            "OJT_Eligible",
            "OJT_Readiness",
            "AI_Recommendation",
            # Future outcome information
            "Future_Credits_At_OJT",
            "Future_Failed_Courses",
            "Future_Missing_Prerequisites",
            "Future_Academic_Warning",
            "Future_OJT_Eligible",
            # Target
            "OJT_Delay_Outcome",
        ]

        leakage_found = [
            column for column in forbidden_columns if column in self.X.columns
        ]

        if len(leakage_found) > 0:

            print()

            print("LEAKAGE DETECTED:")

            for column in leakage_found:

                print(f" - {column}")

            raise ValueError("Forbidden columns were included " "in model features.")

        # --------------------------------------------------
        # Check target separation
        # --------------------------------------------------

        if self.target_column in self.X.columns:

            raise ValueError("Target column exists inside X.")

        # --------------------------------------------------
        # Check X/y row alignment
        # --------------------------------------------------

        if len(self.X) != len(self.y):

            raise ValueError("X and y row counts do not match.")

        print()

        print("[PASS] Forbidden feature check")

        print("[PASS] Target separation")

        print("[PASS] X/y row alignment")

        print()

        print("Data leakage check passed.")

    # ======================================================
    # Split Dataset
    # ======================================================

    def split_dataset(self):

        print()

        print("=" * 60)

        print("Splitting Dataset...")

        print("=" * 60)

        # --------------------------------------------------
        # Train / Test Split
        # --------------------------------------------------

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        print()

        print("Dataset Split Successfully.")

        print(f"Training Samples : " f"{len(self.X_train)}")

        print(f"Testing Samples  : " f"{len(self.X_test)}")

        print()

        print(f"Training Ratio : " f"{len(self.X_train) / len(self.df):.2%}")

        print(f"Testing Ratio  : " f"{len(self.X_test) / len(self.df):.2%}")

    # ======================================================
    # Validate Split Distribution
    # ======================================================

    def validate_split_distribution(self):

        print()

        print("=" * 60)

        print("Validating Split Distribution...")

        print("=" * 60)

        # --------------------------------------------------
        # Full dataset distribution
        # --------------------------------------------------

        full_distribution = self.y.value_counts(normalize=True).sort_index()

        # --------------------------------------------------
        # Training distribution
        # --------------------------------------------------

        train_distribution = self.y_train.value_counts(normalize=True).sort_index()

        # --------------------------------------------------
        # Testing distribution
        # --------------------------------------------------

        test_distribution = self.y_test.value_counts(normalize=True).sort_index()

        print()

        print("Full Dataset:")

        print(full_distribution)

        print()

        print("Training Set:")

        print(train_distribution)

        print()

        print("Testing Set:")

        print(test_distribution)

        # --------------------------------------------------
        # Check classes
        # --------------------------------------------------

        expected_classes = {0, 1}

        train_classes = set(self.y_train.unique())

        test_classes = set(self.y_test.unique())

        if train_classes != expected_classes:

            raise ValueError("Training set does not contain " "both target classes.")

        if test_classes != expected_classes:

            raise ValueError("Testing set does not contain " "both target classes.")

        print()

        print("[PASS] Both classes exist in training set.")

        print("[PASS] Both classes exist in testing set.")

    # ======================================================
    # Save Split Dataset
    # ======================================================

    def save_split_dataset(self):

        print()

        print("=" * 60)

        print("Saving Split Dataset...")

        print("=" * 60)

        # --------------------------------------------------
        # Create Output Folder
        # --------------------------------------------------

        os.makedirs(self.output_folder, exist_ok=True)

        # --------------------------------------------------
        # Save X Train
        # --------------------------------------------------

        self.X_train.to_csv(X_TRAIN_PATH, index=False, encoding="utf-8-sig")

        # --------------------------------------------------
        # Save X Test
        # --------------------------------------------------

        self.X_test.to_csv(X_TEST_PATH, index=False, encoding="utf-8-sig")

        # --------------------------------------------------
        # Save y Train
        # --------------------------------------------------

        self.y_train.to_csv(Y_TRAIN_PATH, index=False, encoding="utf-8-sig")

        # --------------------------------------------------
        # Save y Test
        # --------------------------------------------------

        self.y_test.to_csv(Y_TEST_PATH, index=False, encoding="utf-8-sig")

        print()

        print("Split Dataset Saved Successfully.")

        print()

        print(f"X_train : {X_TRAIN_PATH}")

        print(f"X_test  : {X_TEST_PATH}")

        print(f"y_train : {Y_TRAIN_PATH}")

        print(f"y_test  : {Y_TEST_PATH}")

    # ======================================================
    # Print Split Summary
    # ======================================================

    def print_summary(self):

        print()

        print("=" * 60)

        print("DATA SPLITTING SUMMARY")

        print("=" * 60)

        print()

        print(f"Original Rows : " f"{len(self.df)}")

        print(f"Training Rows : " f"{len(self.X_train)}")

        print(f"Testing Rows  : " f"{len(self.X_test)}")

        print()

        print(f"Feature Columns : " f"{len(self.X.columns)}")

        print(f"Target Column   : " f"{self.target_column}")

        print()

        print("Target Distribution - Full Dataset")

        print(self.y.value_counts(normalize=True).sort_index())

        print()

        print("Target Distribution - Training Set")

        print(self.y_train.value_counts(normalize=True).sort_index())

        print()

        print("Target Distribution - Testing Set")

        print(self.y_test.value_counts(normalize=True).sort_index())

        print()

        print("Output Folder:")

        print(self.output_folder)

        print()

        print("Official Model Features:")

        for index, feature in enumerate(self.X.columns, start=1):

            print(f"{index:02d}. {feature}")

        print()

        print("Target:")

        print(f" - {self.target_column}")

        print("=" * 60)

    # ======================================================
    # Run Data Splitting Pipeline
    # ======================================================

    def transform(self):

        # --------------------------------------------------
        # Step 1: Load Dataset
        # --------------------------------------------------

        self.load_dataset()

        # --------------------------------------------------
        # Step 2: Validate Dataset
        # --------------------------------------------------

        self.validate_dataset()

        # --------------------------------------------------
        # Step 3: Validate Target
        # --------------------------------------------------

        self.validate_target()

        # --------------------------------------------------
        # Step 4: Separate Features and Target
        # --------------------------------------------------

        self.separate_features_target()

        # --------------------------------------------------
        # Step 5: Check Leakage
        # --------------------------------------------------

        self.check_leakage()

        # --------------------------------------------------
        # Step 6: Split Dataset
        # --------------------------------------------------

        self.split_dataset()

        # --------------------------------------------------
        # Step 7: Validate Split Distribution
        # --------------------------------------------------

        self.validate_split_distribution()

        # --------------------------------------------------
        # Step 8: Save Split Dataset
        # --------------------------------------------------

        self.save_split_dataset()

        # --------------------------------------------------
        # Step 9: Print Summary
        # --------------------------------------------------

        self.print_summary()


# ==========================================================
# Main
# ==========================================================


def main():

    data_splitter = DataSplitter()

    data_splitter.transform()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
