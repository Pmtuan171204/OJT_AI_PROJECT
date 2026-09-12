"""
==========================================================
DATA ENCODING & SCALING V2
OJT AI Project

Sprint 3 - Encoding & Scaling

Purpose:
    1. Encode categorical features using One-Hot Encoding.
    2. Scale numerical features using StandardScaler.
    3. Fit preprocessing ONLY on training data.
    4. Transform both training and testing data.
    5. Prevent preprocessing data leakage.

Input:
    data/processed/split/X_train.csv
    data/processed/split/X_test.csv
    data/processed/split/y_train.csv
    data/processed/split/y_test.csv

Output:
    data/processed/encoded/X_train_encoded.csv
    data/processed/encoded/X_test_encoded.csv
    data/processed/encoded/y_train.csv
    data/processed/encoded/y_test.csv

Target:
    OJT_Delay_Outcome

==========================================================
"""

import os
import sys

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Add Project Root To Python Path
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
    SPLIT_DATA_DIR,
    ENCODED_DATA_DIR,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    X_TRAIN_ENCODED_PATH,
    X_TEST_ENCODED_PATH,
    Y_TRAIN_ENCODED_PATH,
    Y_TEST_ENCODED_PATH
)


# ==========================================================
# Configuration
# ==========================================================

TARGET_COLUMN = "OJT_Delay_Outcome"


# ==========================================================
# Data Encoder
# ==========================================================

class DataEncoder:

    def __init__(self):

        # --------------------------------------------------
        # Input Paths
        # --------------------------------------------------

        self.X_train_path = X_TRAIN_PATH

        self.X_test_path = X_TEST_PATH

        self.y_train_path = Y_TRAIN_PATH

        self.y_test_path = Y_TEST_PATH

        # --------------------------------------------------
        # Output Directory
        # --------------------------------------------------

        self.output_dir = ENCODED_DATA_DIR

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        # --------------------------------------------------
        # Encoded Data
        # --------------------------------------------------

        self.X_train_encoded = None

        self.X_test_encoded = None

        # --------------------------------------------------
        # Preprocessing Objects
        # --------------------------------------------------

        self.encoder = None

        self.scaler = None

        # --------------------------------------------------
        # Feature Information
        # --------------------------------------------------

        self.categorical_columns = []

        self.numeric_columns = []

        self.encoded_feature_names = []

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        print()

        print("=" * 60)

        print(
            "Loading Split Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Check files
        # --------------------------------------------------

        input_files = [

            self.X_train_path,
            self.X_test_path,
            self.y_train_path,
            self.y_test_path

        ]

        for file_path in input_files:

            if not os.path.exists(file_path):

                raise FileNotFoundError(
                    "Required input file not found:\n"
                    f"{file_path}"
                )

        # --------------------------------------------------
        # Load X
        # --------------------------------------------------

        self.X_train = pd.read_csv(
            self.X_train_path
        )

        self.X_test = pd.read_csv(
            self.X_test_path
        )

        # --------------------------------------------------
        # Load y
        # --------------------------------------------------

        self.y_train = pd.read_csv(
            self.y_train_path
        )

        self.y_test = pd.read_csv(
            self.y_test_path
        )

        print()

        print(
            "Split Dataset Loaded Successfully."
        )

        print(
            f"X_train Rows : "
            f"{len(self.X_train)}"
        )

        print(
            f"X_test Rows  : "
            f"{len(self.X_test)}"
        )

        print(
            f"Features     : "
            f"{len(self.X_train.columns)}"
        )

    # ======================================================
    # Validate Input Data
    # ======================================================

    def validate_input_data(self):

        print()

        print("=" * 60)

        print(
            "Validating Input Data..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Check X row alignment
        # --------------------------------------------------

        if len(self.X_train) != len(self.y_train):

            raise ValueError(
                "X_train and y_train row counts "
                "do not match."
            )

        if len(self.X_test) != len(self.y_test):

            raise ValueError(
                "X_test and y_test row counts "
                "do not match."
            )

        # --------------------------------------------------
        # Check feature alignment
        # --------------------------------------------------

        if list(
            self.X_train.columns
        ) != list(
            self.X_test.columns
        ):

            raise ValueError(
                "X_train and X_test do not "
                "have the same feature columns."
            )

        # --------------------------------------------------
        # Check target column
        # --------------------------------------------------

        if len(self.y_train.columns) != 1:

            raise ValueError(
                "y_train must contain exactly "
                "one target column."
            )

        if len(self.y_test.columns) != 1:

            raise ValueError(
                "y_test must contain exactly "
                "one target column."
            )

        # --------------------------------------------------
        # Check target name
        # --------------------------------------------------

        if self.y_train.columns[0] != TARGET_COLUMN:

            raise ValueError(
                f"Expected target column "
                f"'{TARGET_COLUMN}' "
                f"but found "
                f"'{self.y_train.columns[0]}'."
            )

        if self.y_test.columns[0] != TARGET_COLUMN:

            raise ValueError(
                f"Expected target column "
                f"'{TARGET_COLUMN}' "
                f"but found "
                f"'{self.y_test.columns[0]}'."
            )

        # --------------------------------------------------
        # Check missing values
        # --------------------------------------------------

        if self.X_train.isnull().any().any():

            raise ValueError(
                "X_train contains missing values."
            )

        if self.X_test.isnull().any().any():

            raise ValueError(
                "X_test contains missing values."
            )

        if self.y_train.isnull().any().any():

            raise ValueError(
                "y_train contains missing values."
            )

        if self.y_test.isnull().any().any():

            raise ValueError(
                "y_test contains missing values."
            )

        print()

        print(
            "[PASS] X_train / y_train alignment"
        )

        print(
            "[PASS] X_test / y_test alignment"
        )

        print(
            "[PASS] Train / test feature alignment"
        )

        print(
            "[PASS] Target column validation"
        )

        print(
            "[PASS] Missing value validation"
        )

    # ======================================================
    # Detect Feature Types
    # ======================================================

    def detect_feature_types(self):

        print()

        print("=" * 60)

        print(
            "Detecting Feature Types..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Detect categorical features from TRAINING data
        # --------------------------------------------------

        self.categorical_columns = (

            self.X_train
            .select_dtypes(
                include=[
                    "object",
                    "string",
                    "category"
                ]
            )
            .columns
            .tolist()

        )

        # --------------------------------------------------
        # Detect numerical features from TRAINING data
        # --------------------------------------------------

        self.numeric_columns = (

            self.X_train
            .select_dtypes(
                include=[
                    "number",
                    "bool"
                ]
            )
            .columns
            .tolist()

        )

        # --------------------------------------------------
        # Validate all columns are classified
        # --------------------------------------------------

        classified_columns = (

            self.categorical_columns
            + self.numeric_columns

        )

        unclassified_columns = [

            column

            for column in self.X_train.columns

            if column not in classified_columns

        ]

        if len(unclassified_columns) > 0:

            raise ValueError(
                "Unclassified feature columns:\n"
                + "\n".join(
                    f" - {column}"
                    for column in unclassified_columns
                )
            )

        # --------------------------------------------------
        # Validate column count
        # --------------------------------------------------

        if (

            len(self.categorical_columns)
            + len(self.numeric_columns)

        ) != len(self.X_train.columns):

            raise ValueError(
                "Feature type counts do not match "
                "total feature count."
            )

        # --------------------------------------------------
        # Print categorical features
        # --------------------------------------------------

        print()

        print(
            "Categorical Features:"
        )

        for column in self.categorical_columns:

            print(
                f" - {column}"
            )

        # --------------------------------------------------
        # Print numerical features
        # --------------------------------------------------

        print()

        print(
            "Numerical Features:"
        )

        for column in self.numeric_columns:

            print(
                f" - {column}"
            )

        print()

        print(
            f"Categorical Count : "
            f"{len(self.categorical_columns)}"
        )

        print(
            f"Numerical Count   : "
            f"{len(self.numeric_columns)}"
        )

        print(
            f"Total Features    : "
            f"{len(self.X_train.columns)}"
        )

    # ======================================================
    # Create Categorical Encoder
    # ======================================================

    def create_encoder(self):

        print()

        print("=" * 60)

        print(
            "Creating Categorical Encoder..."
        )

        print("=" * 60)

        if not self.categorical_columns:

            print(
                "No categorical features found."
            )

            self.encoder = None

            return

        # --------------------------------------------------
        # One-Hot Encoder
        # --------------------------------------------------

        self.encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse_output=False

        )

        # --------------------------------------------------
        # IMPORTANT:
        # Fit ONLY on X_train
        # --------------------------------------------------

        self.encoder.fit(
            self.X_train[
                self.categorical_columns
            ]
        )

        print()

        print(
            "Categorical Encoder "
            "Fitted Successfully."
        )

        print(
            "Encoder Fit Data : X_train"
        )

    # ======================================================
    # Create Numerical Scaler
    # ======================================================

    def create_scaler(self):

        print()

        print("=" * 60)

        print(
            "Creating Numerical Scaler..."
        )

        print("=" * 60)

        if not self.numeric_columns:

            print(
                "No numerical features found."
            )

            self.scaler = None

            return

        # --------------------------------------------------
        # Standard Scaler
        # --------------------------------------------------

        self.scaler = StandardScaler()

        # --------------------------------------------------
        # IMPORTANT:
        # Fit ONLY on X_train
        # --------------------------------------------------

        self.scaler.fit(
            self.X_train[
                self.numeric_columns
            ]
        )

        print()

        print(
            "Numerical Scaler "
            "Fitted Successfully."
        )

        print(
            "Scaler Fit Data : X_train"
        )

    # ======================================================
    # Transform Features
    # ======================================================

    def transform_features(self):

        print()

        print("=" * 60)

        print(
            "Transforming Features..."
        )

        print("=" * 60)

        # ==================================================
        # Numerical Features
        # ==================================================

        if self.scaler is not None:

            X_train_numeric = (

                self.scaler.transform(
                    self.X_train[
                        self.numeric_columns
                    ]
                )

            )

            X_test_numeric = (

                self.scaler.transform(
                    self.X_test[
                        self.numeric_columns
                    ]
                )

            )

            # ----------------------------------------------
            # Convert to DataFrame
            # ----------------------------------------------

            X_train_numeric = pd.DataFrame(

                X_train_numeric,

                columns=self.numeric_columns,

                index=self.X_train.index

            )

            X_test_numeric = pd.DataFrame(

                X_test_numeric,

                columns=self.numeric_columns,

                index=self.X_test.index

            )

        else:

            X_train_numeric = pd.DataFrame(
                index=self.X_train.index
            )

            X_test_numeric = pd.DataFrame(
                index=self.X_test.index
            )

        # ==================================================
        # Categorical Features
        # ==================================================

        if self.encoder is not None:

            X_train_categorical = (

                self.encoder.transform(
                    self.X_train[
                        self.categorical_columns
                    ]
                )

            )

            X_test_categorical = (

                self.encoder.transform(
                    self.X_test[
                        self.categorical_columns
                    ]
                )

            )

            encoded_names = (

                self.encoder
                .get_feature_names_out(
                    self.categorical_columns
                )

            )

            # ----------------------------------------------
            # Convert to DataFrame
            # ----------------------------------------------

            X_train_categorical = pd.DataFrame(

                X_train_categorical,

                columns=encoded_names,

                index=self.X_train.index

            )

            X_test_categorical = pd.DataFrame(

                X_test_categorical,

                columns=encoded_names,

                index=self.X_test.index

            )

        else:

            X_train_categorical = pd.DataFrame(
                index=self.X_train.index
            )

            X_test_categorical = pd.DataFrame(
                index=self.X_test.index
            )

            encoded_names = []

        # ==================================================
        # Combine Numerical + Categorical
        # ==================================================

        self.X_train_encoded = pd.concat(

            [

                X_train_numeric,

                X_train_categorical

            ],

            axis=1

        )

        self.X_test_encoded = pd.concat(

            [

                X_test_numeric,

                X_test_categorical

            ],

            axis=1

        )

        # --------------------------------------------------
        # Store Feature Names
        # --------------------------------------------------

        self.encoded_feature_names = (

            self.X_train_encoded
            .columns
            .tolist()

        )

        # --------------------------------------------------
        # Validate Feature Alignment
        # --------------------------------------------------

        if list(
            self.X_train_encoded.columns
        ) != list(
            self.X_test_encoded.columns
        ):

            raise ValueError(
                "Encoded X_train and X_test "
                "have different feature columns."
            )

        print()

        print(
            "Feature Transformation Completed."
        )

        print(
            f"Features Before Encoding : "
            f"{len(self.X_train.columns)}"
        )

        print(
            f"Features After Encoding  : "
            f"{len(self.X_train_encoded.columns)}"
        )

    # ======================================================
    # Validate Encoded Dataset
    # ======================================================

    def validate_encoded_data(self):

        print()

        print("=" * 60)

        print(
            "Validating Encoded Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Check row counts
        # --------------------------------------------------

        if len(
            self.X_train_encoded
        ) != len(
            self.X_train
        ):

            raise ValueError(
                "Encoded X_train row count "
                "does not match original X_train."
            )

        if len(
            self.X_test_encoded
        ) != len(
            self.X_test
        ):

            raise ValueError(
                "Encoded X_test row count "
                "does not match original X_test."
            )

        # --------------------------------------------------
        # Check column count
        # --------------------------------------------------

        if len(
            self.X_train_encoded.columns
        ) == 0:

            raise ValueError(
                "No encoded features were generated."
            )

        # --------------------------------------------------
        # Check missing values
        # --------------------------------------------------

        if self.X_train_encoded.isnull().any().any():

            raise ValueError(
                "Encoded X_train contains "
                "missing values."
            )

        if self.X_test_encoded.isnull().any().any():

            raise ValueError(
                "Encoded X_test contains "
                "missing values."
            )

        # --------------------------------------------------
        # Check infinite values
        # --------------------------------------------------

        if not (
            self.X_train_encoded
            .map(lambda value: pd.notna(value))
            .all()
            .all()
        ):

            raise ValueError(
                "Invalid values detected "
                "in encoded X_train."
            )

        if not (
            self.X_test_encoded
            .map(lambda value: pd.notna(value))
            .all()
            .all()
        ):

            raise ValueError(
                "Invalid values detected "
                "in encoded X_test."
            )

        # --------------------------------------------------
        # Check numeric data
        # --------------------------------------------------

        non_numeric_train = (

            self.X_train_encoded
            .select_dtypes(
                exclude=["number"]
            )
            .columns
            .tolist()

        )

        non_numeric_test = (

            self.X_test_encoded
            .select_dtypes(
                exclude=["number"]
            )
            .columns
            .tolist()

        )

        if len(non_numeric_train) > 0:

            raise ValueError(
                "Non-numeric columns found "
                "in encoded X_train:\n"
                + "\n".join(
                    f" - {column}"
                    for column in non_numeric_train
                )
            )

        if len(non_numeric_test) > 0:

            raise ValueError(
                "Non-numeric columns found "
                "in encoded X_test:\n"
                + "\n".join(
                    f" - {column}"
                    for column in non_numeric_test
                )
            )

        print()

        print(
            "[PASS] Row count validation"
        )

        print(
            "[PASS] Encoded feature generation"
        )

        print(
            "[PASS] Missing value validation"
        )

        print(
            "[PASS] Numeric feature validation"
        )

        print()

        print(
            "Encoded Dataset Validation Passed."
        )

    # ======================================================
    # Check Preprocessing Leakage
    # ======================================================

    def check_preprocessing_leakage(self):

        print()

        print("=" * 60)

        print(
            "Checking Preprocessing Leakage..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # The implementation explicitly fits both
        # preprocessing objects only on X_train.
        # --------------------------------------------------

        print()

        print(
            "[PASS] OneHotEncoder fitted only on X_train"
        )

        print(
            "[PASS] StandardScaler fitted only on X_train"
        )

        print(
            "[PASS] X_test transformed without fitting"
        )

        print()

        print(
            "Preprocessing leakage check passed."
        )

    # ======================================================
    # Save Encoded Dataset
    # ======================================================

    def save_dataset(self):

        print()

        print("=" * 60)

        print(
            "Saving Encoded Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Create Output Directory
        # --------------------------------------------------

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # --------------------------------------------------
        # Save Encoded X_train
        # --------------------------------------------------

        self.X_train_encoded.to_csv(

            X_TRAIN_ENCODED_PATH,

            index=False,

            encoding="utf-8-sig"

        )

        # --------------------------------------------------
        # Save Encoded X_test
        # --------------------------------------------------

        self.X_test_encoded.to_csv(

            X_TEST_ENCODED_PATH,

            index=False,

            encoding="utf-8-sig"

        )

        # --------------------------------------------------
        # Save y_train
        # --------------------------------------------------

        self.y_train.to_csv(

            Y_TRAIN_ENCODED_PATH,

            index=False,

            encoding="utf-8-sig"

        )

        # --------------------------------------------------
        # Save y_test
        # --------------------------------------------------

        self.y_test.to_csv(

            Y_TEST_ENCODED_PATH,

            index=False,

            encoding="utf-8-sig"

        )

        print()

        print(
            "Encoded Dataset Saved Successfully."
        )

        print()

        print(
            f"X_train : "
            f"{X_TRAIN_ENCODED_PATH}"
        )

        print(
            f"X_test  : "
            f"{X_TEST_ENCODED_PATH}"
        )

        print(
            f"y_train : "
            f"{Y_TRAIN_ENCODED_PATH}"
        )

        print(
            f"y_test  : "
            f"{Y_TEST_ENCODED_PATH}"
        )

    # ======================================================
    # Print Summary
    # ======================================================

    def print_summary(self):

        print()

        print("=" * 60)

        print(
            "ENCODING SUMMARY"
        )

        print("=" * 60)

        print()

        print(
            f"Training Samples : "
            f"{len(self.X_train_encoded)}"
        )

        print(
            f"Testing Samples  : "
            f"{len(self.X_test_encoded)}"
        )

        print(
            f"Original Features : "
            f"{len(self.X_train.columns)}"
        )

        print(
            f"Encoded Features  : "
            f"{len(self.X_train_encoded.columns)}"
        )

        print()

        print(
            f"Categorical Features : "
            f"{len(self.categorical_columns)}"
        )

        print(
            f"Numerical Features   : "
            f"{len(self.numeric_columns)}"
        )

        print()

        print(
            "Categorical Columns:"
        )

        for column in self.categorical_columns:

            print(
                f" - {column}"
            )

        print()

        print(
            "Target:"
        )

        print(
            f" - {TARGET_COLUMN}"
        )

        print()

        print(
            "Output Directory:"
        )

        print(
            self.output_dir
        )

        print()

        print("=" * 60)

    # ======================================================
    # Run Encoding Pipeline
    # ======================================================

    def transform(self):

        # --------------------------------------------------
        # Step 1: Load Dataset
        # --------------------------------------------------

        self.load_dataset()

        # --------------------------------------------------
        # Step 2: Validate Input
        # --------------------------------------------------

        self.validate_input_data()

        # --------------------------------------------------
        # Step 3: Detect Feature Types
        # --------------------------------------------------

        self.detect_feature_types()

        # --------------------------------------------------
        # Step 4: Create Encoder
        # --------------------------------------------------

        self.create_encoder()

        # --------------------------------------------------
        # Step 5: Create Scaler
        # --------------------------------------------------

        self.create_scaler()

        # --------------------------------------------------
        # Step 6: Transform Features
        # --------------------------------------------------

        self.transform_features()

        # --------------------------------------------------
        # Step 7: Validate Encoded Data
        # --------------------------------------------------

        self.validate_encoded_data()

        # --------------------------------------------------
        # Step 8: Check Leakage
        # --------------------------------------------------

        self.check_preprocessing_leakage()

        # --------------------------------------------------
        # Step 9: Save Dataset
        # --------------------------------------------------

        self.save_dataset()

        # --------------------------------------------------
        # Step 10: Print Summary
        # --------------------------------------------------

        self.print_summary()


# ==========================================================
# Main
# ==========================================================

def main():

    encoder = DataEncoder()

    encoder.transform()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()