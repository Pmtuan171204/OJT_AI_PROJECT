"""
==========================================================
Logistic Regression Training
OJT AI Project

Baseline Machine Learning Model for
OJT Delay Risk Prediction.
==========================================================
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

# ==========================================================
# Add Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Project Paths
# ==========================================================

from config.paths import PROCESSED_DATA_DIR

# ==========================================================
# Dataset Paths
# ==========================================================

ENCODED_DIR = os.path.join(PROCESSED_DATA_DIR, "encoded")

X_TRAIN_PATH = os.path.join(ENCODED_DIR, "X_train_encoded.csv")

X_TEST_PATH = os.path.join(ENCODED_DIR, "X_test_encoded.csv")

Y_TRAIN_PATH = os.path.join(ENCODED_DIR, "y_train.csv")

Y_TEST_PATH = os.path.join(ENCODED_DIR, "y_test.csv")


# ==========================================================
# Model Output Directory
# ==========================================================

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")


# ==========================================================
# Logistic Regression Trainer
# ==========================================================


class LogisticRegressionTrainer:

    def __init__(self):

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.model = None

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        print()
        print("=" * 60)
        print("Loading Encoded Dataset...")
        print("=" * 60)

        self.X_train = pd.read_csv(X_TRAIN_PATH)

        self.X_test = pd.read_csv(X_TEST_PATH)

        self.y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]

        self.y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

        print()
        print("Encoded Dataset Loaded Successfully.")

        print(f"X_train Rows    : {len(self.X_train)}")

        print(f"X_test Rows     : {len(self.X_test)}")

        print(f"Features        : {self.X_train.shape[1]}")

        print(f"y_train Rows    : {len(self.y_train)}")

        print(f"y_test Rows     : {len(self.y_test)}")

    # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_dataset(self):

        print()
        print("=" * 60)
        print("Validating Training Dataset...")
        print("=" * 60)

        # --------------------------------------------------
        # Feature Count
        # --------------------------------------------------

        if self.X_train.shape[1] != self.X_test.shape[1]:

            raise ValueError("X_train and X_test feature counts do not match.")

        # --------------------------------------------------
        # Row Alignment
        # --------------------------------------------------

        if len(self.X_train) != len(self.y_train):

            raise ValueError("X_train and y_train row counts do not match.")

        if len(self.X_test) != len(self.y_test):

            raise ValueError("X_test and y_test row counts do not match.")

        # --------------------------------------------------
        # Missing Values
        # --------------------------------------------------

        if self.X_train.isnull().sum().sum() > 0:

            raise ValueError("X_train contains missing values.")

        if self.X_test.isnull().sum().sum() > 0:

            raise ValueError("X_test contains missing values.")

        # --------------------------------------------------
        # Target Validation
        # --------------------------------------------------

        valid_targets = {0, 1}

        if not set(self.y_train.unique()).issubset(valid_targets):

            raise ValueError("y_train contains invalid target values.")

        if not set(self.y_test.unique()).issubset(valid_targets):

            raise ValueError("y_test contains invalid target values.")

        print()
        print("Dataset Validation Passed.")

        print(f"Training Shape : {self.X_train.shape}")

        print(f"Testing Shape  : {self.X_test.shape}")

    # ======================================================
    # Create Model
    # ======================================================

    def create_model(self):

        print()
        print("=" * 60)
        print("Creating Logistic Regression Model...")
        print("=" * 60)

        self.model = LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        )

        print()
        print("Logistic Regression Model Created.")

        print("class_weight : balanced")

        print("random_state : 42")

        print("max_iter     : 1000")

    # ======================================================
    # Train Model
    # ======================================================

    def train(self):

        print()
        print("=" * 60)
        print("Training Logistic Regression...")
        print("=" * 60)

        self.model.fit(self.X_train, self.y_train)

        print()
        print("Training Completed Successfully.")

    # ======================================================
    # Training Information
    # ======================================================

    def display_model_information(self):

        print()
        print("=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)

        print(f"Algorithm          : Logistic Regression")

        print(f"Training Samples   : {len(self.X_train)}")

        print(f"Testing Samples    : {len(self.X_test)}")

        print(f"Features           : {self.X_train.shape[1]}")

        print(f"Classes            : {sorted(self.y_train.unique())}")

        print("Class Weight       : balanced")

    # ======================================================
    # Save Model
    # ======================================================

    def save_model(self):

        print()
        print("=" * 60)
        print("Saving Logistic Regression Model...")
        print("=" * 60)

        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(self.model, MODEL_PATH)

        print()
        print("Model Saved Successfully.")

        print(f"Model : {MODEL_PATH}")

    # ======================================================
    # Run Training Pipeline
    # ======================================================

    def run(self):

        self.load_dataset()

        self.validate_dataset()

        self.create_model()

        self.train()

        self.display_model_information()

        self.save_model()


# ==========================================================
# Main
# ==========================================================


def main():

    trainer = LogisticRegressionTrainer()

    trainer.run()

    print()
    print("=" * 60)
    print("LOGISTIC REGRESSION TRAINING COMPLETED")
    print("=" * 60)


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
