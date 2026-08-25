"""
==========================================================
DATA ENCODING & SCALING
OJT AI Project

Encode categorical features and scale numerical features
before Machine Learning training.
==========================================================
"""

import os
import sys

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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

from config.paths import SPLIT_DATA_DIR, PROCESSED_DATA_DIR


class DataEncoder:

    def __init__(self):

        # ==================================================
        # Input Paths
        # ==================================================

        self.X_train_path = os.path.join(SPLIT_DATA_DIR, "X_train.csv")

        self.X_test_path = os.path.join(SPLIT_DATA_DIR, "X_test.csv")

        self.y_train_path = os.path.join(SPLIT_DATA_DIR, "y_train.csv")

        self.y_test_path = os.path.join(SPLIT_DATA_DIR, "y_test.csv")

        # ==================================================
        # Output Directory
        # ==================================================

        self.output_dir = os.path.join(PROCESSED_DATA_DIR, "encoded")

        # ==================================================
        # Data
        # ==================================================

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        # ==================================================
        # Encoders
        # ==================================================

        self.encoder = None
        self.scaler = None

        # ==================================================
        # Feature Information
        # ==================================================

        self.categorical_columns = []
        self.numeric_columns = []

        self.encoded_feature_names = []

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        print()

        print("=" * 60)
        print("Loading Split Dataset...")
        print("=" * 60)

        self.X_train = pd.read_csv(self.X_train_path)

        self.X_test = pd.read_csv(self.X_test_path)

        self.y_train = pd.read_csv(self.y_train_path)

        self.y_test = pd.read_csv(self.y_test_path)

        print()

        print("Split Dataset Loaded Successfully.")

        print(f"X_train Rows : {len(self.X_train)}")

        print(f"X_test Rows  : {len(self.X_test)}")

        print(f"Features     : {len(self.X_train.columns)}")

    # ======================================================
    # Detect Feature Types
    # ======================================================

    def detect_feature_types(self):

        print()

        print("=" * 60)
        print("Detecting Feature Types...")
        print("=" * 60)

        self.categorical_columns = self.X_train.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        self.numeric_columns = self.X_train.select_dtypes(
            include=["int64", "float64", "bool"]
        ).columns.tolist()

        print()

        print("Categorical Features:")

        for column in self.categorical_columns:

            print(f" - {column}")

        print()

        print("Numerical Features:")

        for column in self.numeric_columns:

            print(f" - {column}")

    # ======================================================
    # Create Encoder
    # ======================================================

    def create_encoder(self):

        print()

        print("=" * 60)
        print("Creating Categorical Encoder...")
        print("=" * 60)

        if not self.categorical_columns:

            print("No categorical features found.")

            self.encoder = None

            return

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.encoder.fit(self.X_train[self.categorical_columns])

        print()

        print("Categorical Encoder Fitted " "Successfully.")

    # ======================================================
    # Create Scaler
    # ======================================================

    def create_scaler(self):

        print()

        print("=" * 60)
        print("Creating Numerical Scaler...")
        print("=" * 60)

        self.scaler = StandardScaler()

        self.scaler.fit(self.X_train[self.numeric_columns])

        print()

        print("Numerical Scaler Fitted " "Successfully.")

    # ======================================================
    # Transform Features
    # ======================================================

    def transform_features(self):

        print()

        print("=" * 60)
        print("Transforming Features...")
        print("=" * 60)

        # --------------------------------------------------
        # Numerical Features
        # --------------------------------------------------

        X_train_numeric = self.scaler.transform(self.X_train[self.numeric_columns])

        X_test_numeric = self.scaler.transform(self.X_test[self.numeric_columns])

        # --------------------------------------------------
        # Categorical Features
        # --------------------------------------------------

        if self.encoder is not None:

            X_train_categorical = self.encoder.transform(
                self.X_train[self.categorical_columns]
            )

            X_test_categorical = self.encoder.transform(
                self.X_test[self.categorical_columns]
            )

            encoded_names = self.encoder.get_feature_names_out(self.categorical_columns)

        else:

            X_train_categorical = pd.DataFrame(index=self.X_train.index)

            X_test_categorical = pd.DataFrame(index=self.X_test.index)

            encoded_names = []

        # --------------------------------------------------
        # Convert Numerical DataFrame
        # --------------------------------------------------

        X_train_numeric = pd.DataFrame(
            X_train_numeric, columns=self.numeric_columns, index=self.X_train.index
        )

        X_test_numeric = pd.DataFrame(
            X_test_numeric, columns=self.numeric_columns, index=self.X_test.index
        )

        # --------------------------------------------------
        # Convert Categorical DataFrame
        # --------------------------------------------------

        X_train_categorical = pd.DataFrame(
            X_train_categorical, columns=encoded_names, index=self.X_train.index
        )

        X_test_categorical = pd.DataFrame(
            X_test_categorical, columns=encoded_names, index=self.X_test.index
        )

        # --------------------------------------------------
        # Combine
        # --------------------------------------------------

        self.X_train_encoded = pd.concat([X_train_numeric, X_train_categorical], axis=1)

        self.X_test_encoded = pd.concat([X_test_numeric, X_test_categorical], axis=1)

        self.encoded_feature_names = self.X_train_encoded.columns.tolist()

        print()

        print("Feature Transformation Completed.")

        print(f"Features Before Encoding : " f"{len(self.X_train.columns)}")

        print(f"Features After Encoding  : " f"{len(self.X_train_encoded.columns)}")

    # ======================================================
    # Save Encoded Dataset
    # ======================================================

    def save_dataset(self):

        print()

        print("=" * 60)
        print("Saving Encoded Dataset...")
        print("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)

        X_train_output = os.path.join(self.output_dir, "X_train_encoded.csv")

        X_test_output = os.path.join(self.output_dir, "X_test_encoded.csv")

        y_train_output = os.path.join(self.output_dir, "y_train.csv")

        y_test_output = os.path.join(self.output_dir, "y_test.csv")

        self.X_train_encoded.to_csv(X_train_output, index=False)

        self.X_test_encoded.to_csv(X_test_output, index=False)

        self.y_train.to_csv(y_train_output, index=False)

        self.y_test.to_csv(y_test_output, index=False)

        print()

        print("Encoded Dataset Saved Successfully.")

        print()

        print(f"X_train : {X_train_output}")

        print(f"X_test  : {X_test_output}")

        print(f"y_train : {y_train_output}")

        print(f"y_test  : {y_test_output}")

    # ======================================================
    # Print Summary
    # ======================================================

    def print_summary(self):

        print()

        print("=" * 60)
        print("ENCODING SUMMARY")
        print("=" * 60)

        print()

        print(f"Training Samples : " f"{len(self.X_train_encoded)}")

        print(f"Testing Samples  : " f"{len(self.X_test_encoded)}")

        print(f"Original Features : " f"{len(self.X_train.columns)}")

        print(f"Encoded Features  : " f"{len(self.X_train_encoded.columns)}")

        print()

        print("Categorical Features:")

        for column in self.categorical_columns:

            print(f" - {column}")

        print()

        print("Output Directory:")

        print(self.output_dir)

        print()

        print("=" * 60)

    # ======================================================
    # Run Encoding Pipeline
    # ======================================================

    def transform(self):

        self.load_dataset()

        self.detect_feature_types()

        self.create_encoder()

        self.create_scaler()

        self.transform_features()

        self.save_dataset()

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