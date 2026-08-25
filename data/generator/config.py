"""
==========================================================
CONFIGURATION
OJT AI Project

Central configuration for Dataset Generator
==========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT PATH
# ==========================================================

# data/generator/
GENERATOR_DIR = Path(__file__).resolve().parent

# data/
DATA_DIR = GENERATOR_DIR.parent

# data/raw/
RAW_DATA_DIR = DATA_DIR / "raw"

# data/processed/
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Tự tạo folder nếu chưa tồn tại
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# OUTPUT FILE
# ==========================================================

RAW_DATASET_FILE = RAW_DATA_DIR / "ojt_dataset.xlsx"

PROCESSED_DATASET_FILE = PROCESSED_DATA_DIR / "ojt_dataset_clean.xlsx"

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# DATASET SIZE
# ==========================================================

TOTAL_STUDENTS = 5000

# ==========================================================
# OJT CONFIGURATION
# ==========================================================

TOTAL_CREDITS = 120

REQUIRED_CREDITS_FOR_OJT = 100

MAX_SEMESTER = 8

# ==========================================================
# GPA CONFIGURATION
# ==========================================================

MIN_GPA = 0.0

MAX_GPA = 10.0

# ==========================================================
# RISK THRESHOLD
# ==========================================================

LOW_RISK_MAX = 5

MEDIUM_RISK_MAX = 10

HIGH_RISK_MAX = 999

# ==========================================================
# FEATURE CONFIGURATION
# ==========================================================

USE_SYNTHETIC_DATA = True

SAVE_EXCEL = True

SAVE_CSV = True