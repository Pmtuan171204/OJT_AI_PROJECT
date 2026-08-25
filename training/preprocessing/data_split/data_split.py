"""
==========================================================
DATA SPLITTING
OJT AI Project

Split feature-engineered dataset into training
and testing datasets for Machine Learning.
==========================================================
"""

import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split


# ==========================================================
# Add Project Root Into Python Path
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

    sys.path.append(
        PROJECT_ROOT
    )


# ==========================================================
# Import Project Paths
# ==========================================================

from config.paths import (
    PROCESSED_DATA_FOLDER
)


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = os.path.join(
    PROCESSED_DATA_FOLDER,
    "feature_dataset.xlsx"
)

OUTPUT_FOLDER = os.path.join(
    PROCESSED_DATA_FOLDER,
    "split"
)

TARGET_COLUMN = "OJT_Delay_Risk"

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

        self.input_file = INPUT_FILE

        # --------------------------------------------------
        # Output Folder
        # --------------------------------------------------

        self.output_folder = OUTPUT_FOLDER

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

        print(
            "Loading Feature Dataset..."
        )

        print("=" * 60)

        self.df = pd.read_excel(
            self.input_file
        )

        print()

        print(
            "Feature Dataset Loaded Successfully."
        )

        print(
            f"Rows    : {len(self.df)}"
        )

        print(
            f"Columns : {len(self.df.columns)}"
        )

    # ======================================================
    # Validate Target Column
    # ======================================================

    def validate_target(self):

        print()

        print("=" * 60)

        print(
            "Validating Target Column..."
        )

        print("=" * 60)

        if self.target_column not in self.df.columns:

            raise ValueError(
                f"Target column "
                f"'{self.target_column}' "
                f"does not exist."
            )

        if self.df[
            self.target_column
        ].isnull().any():

            raise ValueError(
                "Target column contains "
                "missing values."
            )

        print()

        print(
            f"Target Column : "
            f"{self.target_column}"
        )

        print()

        print(
            "Target Distribution:"
        )

        print(
            self.df[
                self.target_column
            ].value_counts()
        )

    # ======================================================
    # Separate Features and Target
    # ======================================================

    def separate_features_target(self):

        print()

        print("=" * 60)

        print(
            "Separating Features and Target..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Columns Not Used For Training
        # --------------------------------------------------

        columns_to_drop = [
            "MSSV",

            "Student_Profile",

            "Risk_Score",

             "Risk_Level",

            "AI_Recommendation",

        self.target_column
           
        ]

        # --------------------------------------------------
        # Create X
        # --------------------------------------------------

        self.X = self.df.drop(
            columns=columns_to_drop,
            errors="ignore"
        )

        # --------------------------------------------------
        # Create y
        # --------------------------------------------------

        self.y = self.df[
            self.target_column
        ]

        print()

        print(
            f"Feature Columns : "
            f"{len(self.X.columns)}"
        )

        print(
            f"Target Column   : "
            f"{self.target_column}"
        )

        print()

        print(
            "Features:"
        )

        for column in self.X.columns:

            print(
                f" - {column}"
            )

    # ======================================================
    # Split Dataset
    # ======================================================

    def split_dataset(self):

        print()

        print("=" * 60)

        print(
            "Splitting Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Train / Test Split
        # --------------------------------------------------

        self.X_train, self.X_test, \
        self.y_train, self.y_test = train_test_split(

            self.X,

            self.y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=self.y
        )

        print()

        print(
            "Dataset Split Successfully."
        )

        print(
            f"Training Samples : "
            f"{len(self.X_train)}"
        )

        print(
            f"Testing Samples  : "
            f"{len(self.X_test)}"
        )

        print()

        print(
            f"Training Ratio : "
            f"{len(self.X_train) / len(self.df):.2%}"
        )

        print(
            f"Testing Ratio  : "
            f"{len(self.X_test) / len(self.df):.2%}"
        )

    # ======================================================
    # Save Split Dataset
    # ======================================================

    def save_split_dataset(self):

        print()

        print("=" * 60)

        print(
            "Saving Split Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Create Output Folder
        # --------------------------------------------------

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

        # --------------------------------------------------
        # Output Paths
        # --------------------------------------------------

        X_train_path = os.path.join(
            self.output_folder,
            "X_train.csv"
        )

        X_test_path = os.path.join(
            self.output_folder,
            "X_test.csv"
        )

        y_train_path = os.path.join(
            self.output_folder,
            "y_train.csv"
        )

        y_test_path = os.path.join(
            self.output_folder,
            "y_test.csv"
        )

        # --------------------------------------------------
        # Save X Train
        # --------------------------------------------------

        self.X_train.to_csv(
            X_train_path,
            index=False
        )

        # --------------------------------------------------
        # Save X Test
        # --------------------------------------------------

        self.X_test.to_csv(
            X_test_path,
            index=False
        )

        # --------------------------------------------------
        # Save y Train
        # --------------------------------------------------

        self.y_train.to_csv(
            y_train_path,
            index=False
        )

        # --------------------------------------------------
        # Save y Test
        # --------------------------------------------------

        self.y_test.to_csv(
            y_test_path,
            index=False
        )

        print()

        print(
            "Split Dataset Saved Successfully."
        )

        print()

        print(
            f"X_train : {X_train_path}"
        )

        print(
            f"X_test  : {X_test_path}"
        )

        print(
            f"y_train : {y_train_path}"
        )

        print(
            f"y_test  : {y_test_path}"
        )

    # ======================================================
    # Print Split Summary
    # ======================================================

    def print_summary(self):

        print()

        print("=" * 60)

        print(
            "DATA SPLITTING SUMMARY"
        )

        print("=" * 60)

        print()

        print(
            f"Original Rows : "
            f"{len(self.df)}"
        )

        print(
            f"Training Rows : "
            f"{len(self.X_train)}"
        )

        print(
            f"Testing Rows  : "
            f"{len(self.X_test)}"
        )

        print()

        print(
            f"Feature Columns : "
            f"{len(self.X.columns)}"
        )

        print(
            f"Target Column   : "
            f"{self.target_column}"
        )

        print()

        print(
            "Target Distribution - Full Dataset"
        )

        print(
            self.y.value_counts(
                normalize=True
            ).sort_index()
        )

        print()

        print(
            "Target Distribution - Training Set"
        )

        print(
            self.y_train.value_counts(
                normalize=True
            ).sort_index()
        )

        print()

        print(
            "Target Distribution - Testing Set"
        )

        print(
            self.y_test.value_counts(
                normalize=True
            ).sort_index()
        )

        print()

        print(
            "Output Folder:"
        )

        print(
            self.output_folder
        )

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
        # Step 2: Validate Target
        # --------------------------------------------------

        self.validate_target()

        # --------------------------------------------------
        # Step 3: Separate Features and Target
        # --------------------------------------------------

        self.separate_features_target()

        # --------------------------------------------------
        # Step 4: Split Dataset
        # --------------------------------------------------

        self.split_dataset()

        # --------------------------------------------------
        # Step 5: Save Split Dataset
        # --------------------------------------------------

        self.save_split_dataset()

        # --------------------------------------------------
        # Step 6: Print Summary
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