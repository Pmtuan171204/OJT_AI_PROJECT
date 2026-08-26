"""
==========================================================
Class Distribution Check
OJT AI Project

Check target class distribution before model training.
==========================================================
"""

import os
import sys
import pandas as pd

# ==========================================================
# Add Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Paths
# ==========================================================

from config.paths import PROCESSED_DATA_DIR

ENCODED_DIR = os.path.join(PROCESSED_DATA_DIR, "encoded")

Y_TRAIN_PATH = os.path.join(ENCODED_DIR, "y_train.csv")

Y_TEST_PATH = os.path.join(ENCODED_DIR, "y_test.csv")


# ==========================================================
# Load Target
# ==========================================================


def load_target():

    print()
    print("=" * 60)
    print("Loading Target Dataset...")
    print("=" * 60)

    y_train = pd.read_csv(Y_TRAIN_PATH)

    y_test = pd.read_csv(Y_TEST_PATH)

    # Make sure target is a Series
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    print()
    print("Target Dataset Loaded Successfully.")

    print(f"Training Samples : {len(y_train)}")

    print(f"Testing Samples  : {len(y_test)}")

    return y_train, y_test


# ==========================================================
# Display Distribution
# ==========================================================


def display_distribution(y, dataset_name):

    print()
    print("=" * 60)
    print(f"{dataset_name} CLASS DISTRIBUTION")
    print("=" * 60)

    counts = y.value_counts().sort_index()

    percentages = y.value_counts(normalize=True).sort_index() * 100

    for label in counts.index:

        print(
            f"Class {label}: "
            f"{counts[label]} samples "
            f"({percentages[label]:.2f}%)"
        )


# ==========================================================
# Check Imbalance
# ==========================================================


def check_imbalance(y, dataset_name):

    counts = y.value_counts()

    majority = counts.max()
    minority = counts.min()

    ratio = majority / minority if minority > 0 else float("inf")

    print()
    print(f"{dataset_name} Imbalance Ratio : " f"{ratio:.2f}")

    if ratio <= 1.5:

        print("Status : BALANCED")

    elif ratio <= 3:

        print("Status : MODERATE IMBALANCE")

    else:

        print("Status : HIGH IMBALANCE")

    return ratio


# ==========================================================
# Main
# ==========================================================


def main():

    y_train, y_test = load_target()

    display_distribution(y_train, "TRAINING DATA")

    display_distribution(y_test, "TEST DATA")

    train_ratio = check_imbalance(y_train, "Training Data")

    test_ratio = check_imbalance(y_test, "Testing Data")

    print()
    print("=" * 60)
    print("CLASS DISTRIBUTION SUMMARY")
    print("=" * 60)

    print(f"Training Imbalance Ratio : " f"{train_ratio:.2f}")

    print(f"Testing Imbalance Ratio  : " f"{test_ratio:.2f}")

    print("=" * 60)


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
