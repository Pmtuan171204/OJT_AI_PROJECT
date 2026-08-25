"""
==========================================================
Project Paths
OJT AI Project

Centralized management of all project paths.
==========================================================
"""

import os

# ======================================================
# Project Root
# ======================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ======================================================
# Data Folder
# ======================================================

DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data"
)

RAW_DATA_FOLDER = os.path.join(
    DATA_FOLDER,
    "raw"
)

PROCESSED_DATA_FOLDER = os.path.join(
    DATA_FOLDER,
    "processed"
)

GENERATOR_FOLDER = os.path.join(
    DATA_FOLDER,
    "generator"
)

# ======================================================
# Training Folder
# ======================================================

TRAINING_FOLDER = os.path.join(
    PROJECT_ROOT,
    "training"
)

PREPROCESSING_FOLDER = os.path.join(
    TRAINING_FOLDER,
    "preprocessing"
)

VALIDATION_FOLDER = os.path.join(
    TRAINING_FOLDER,
    "validation"
)

MODEL_TRAINING_FOLDER = os.path.join(
    TRAINING_FOLDER,
    "models"
)

# ======================================================
# Models
# ======================================================

MODELS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "models"
)

# ======================================================
# Reports
# ======================================================

REPORT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "reports"
)

EDA_REPORT_FOLDER = os.path.join(
    REPORT_FOLDER,
    "eda"
)

FIGURE_FOLDER = os.path.join(
    REPORT_FOLDER,
    "figures"
)

# ======================================================
# Logs
# ======================================================

LOG_FOLDER = os.path.join(
    PROJECT_ROOT,
    "logs"
)

# ======================================================
# Dataset Files
# ======================================================

RAW_DATASET_XLSX = os.path.join(
    RAW_DATA_FOLDER,
    "ojt_dataset.xlsx"
)

RAW_DATASET_CSV = os.path.join(
    RAW_DATA_FOLDER,
    "ojt_dataset.csv"
)

PROCESSED_DATASET = os.path.join(
    PROCESSED_DATA_FOLDER,
    "ojt_dataset_clean.csv"
)

# ======================================================
# Machine Learning Models
# ======================================================

RANDOM_FOREST_MODEL = os.path.join(
    MODELS_FOLDER,
    "random_forest.pkl"
)

LOGISTIC_MODEL = os.path.join(
    MODELS_FOLDER,
    "logistic.pkl"
)

# ======================================================
# Validation Reports
# ======================================================

VALIDATION_REPORT = os.path.join(
    REPORT_FOLDER,
    "validation_report.txt"
)

INVALID_ROWS = os.path.join(
    REPORT_FOLDER,
    "invalid_rows.xlsx"
)

# ==========================================================
# DATA DIRECTORIES
# ==========================================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

RAW_DATA_DIR = os.path.join(
    DATA_DIR,
    "raw"
)

PROCESSED_DATA_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

SPLIT_DATA_DIR = os.path.join(
    PROCESSED_DATA_DIR,
    "split"
)