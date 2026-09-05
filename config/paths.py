"""
==========================================================
Project Paths
OJT AI Project

Centralized management of all project paths.
==========================================================
"""

import os

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==========================================================
# DATA DIRECTORIES
# ==========================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

SPLIT_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "split")

ENCODED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "encoded")


# ==========================================================
# TRAINING DIRECTORIES
# ==========================================================

TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

PREPROCESSING_DIR = os.path.join(TRAINING_DIR, "preprocessing")

VALIDATION_DIR = os.path.join(TRAINING_DIR, "validation")

MODELS_TRAINING_DIR = os.path.join(TRAINING_DIR, "models")

EVALUATION_DIR = os.path.join(TRAINING_DIR, "evaluation")


# ==========================================================
# MODEL DIRECTORY
# ==========================================================

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ==========================================================
# REPORT DIRECTORIES
# ==========================================================

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

EVALUATION_REPORT_DIR = os.path.join(REPORTS_DIR, "evaluation")

EDA_REPORT_DIR = os.path.join(REPORTS_DIR, "eda")

FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


# ==========================================================
# LOG DIRECTORY
# ==========================================================

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ==========================================================
# RAW DATASET FILES
# ==========================================================

RAW_DATASET_XLSX = os.path.join(RAW_DATA_DIR, "ojt_dataset.xlsx")

RAW_DATASET_CSV = os.path.join(RAW_DATA_DIR, "ojt_dataset.csv")


# ==========================================================
# PROCESSED DATASET FILES
# ==========================================================

CLEANED_DATASET_XLSX = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.xlsx")

CLEANED_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")

FEATURE_DATASET_XLSX = os.path.join(PROCESSED_DATA_DIR, "feature_dataset.xlsx")

FEATURE_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "feature_dataset.csv")


# ==========================================================
# SPLIT DATASET FILES
# ==========================================================

X_TRAIN_PATH = os.path.join(SPLIT_DATA_DIR, "X_train.csv")

X_TEST_PATH = os.path.join(SPLIT_DATA_DIR, "X_test.csv")

Y_TRAIN_PATH = os.path.join(SPLIT_DATA_DIR, "y_train.csv")

Y_TEST_PATH = os.path.join(SPLIT_DATA_DIR, "y_test.csv")


# ==========================================================
# ENCODED DATASET FILES
# ==========================================================

X_TRAIN_ENCODED_PATH = os.path.join(ENCODED_DATA_DIR, "X_train_encoded.csv")

X_TEST_ENCODED_PATH = os.path.join(ENCODED_DATA_DIR, "X_test_encoded.csv")

Y_TRAIN_ENCODED_PATH = os.path.join(ENCODED_DATA_DIR, "y_train.csv")

Y_TEST_ENCODED_PATH = os.path.join(ENCODED_DATA_DIR, "y_test.csv")


# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

LOGISTIC_REGRESSION_MODEL = os.path.join(MODEL_DIR, "logistic_regression.pkl")

RANDOM_FOREST_MODEL = os.path.join(MODEL_DIR, "random_forest.pkl")


# ==========================================================
# VALIDATION REPORTS
# ==========================================================

VALIDATION_REPORT_TXT = os.path.join(REPORTS_DIR, "validation_report.txt")

VALIDATION_REPORT_XLSX = os.path.join(REPORTS_DIR, "validation_report.xlsx")


# ==========================================================
# EVALUATION REPORT FILES
# ==========================================================

LOGISTIC_CONFUSION_MATRIX_CSV = os.path.join(
    EVALUATION_REPORT_DIR, "logistic_confusion_matrix.csv"
)

LOGISTIC_CONFUSION_MATRIX_PNG = os.path.join(
    EVALUATION_REPORT_DIR, "logistic_confusion_matrix.png"
)

LOGISTIC_CLASSIFICATION_REPORT_TXT = os.path.join(
    EVALUATION_REPORT_DIR, "logistic_classification_report.txt"
)

LOGISTIC_CLASSIFICATION_REPORT_CSV = os.path.join(
    EVALUATION_REPORT_DIR, "logistic_classification_report.csv"
)

LOGISTIC_ROC_CURVE_PNG = os.path.join(EVALUATION_REPORT_DIR, "logistic_roc_curve.png")

LOGISTIC_ROC_DATA_CSV = os.path.join(EVALUATION_REPORT_DIR, "logistic_roc_data.csv")
