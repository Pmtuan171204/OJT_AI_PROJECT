"""
==========================================================
Data Cleaning Module
OJT AI Project

Sprint 3 - Module 1

This module is responsible for cleaning the raw dataset
before Feature Engineering and Machine Learning.

Author : OJT AI Project
==========================================================
"""

import os
import sys

import pandas as pd

# ==========================================================
# Add Project Root
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==========================================================
# Project Paths
# ==========================================================

from config.paths import (
    RAW_DATASET_XLSX,
    PROCESSED_DATA_FOLDER
)

# ==========================================================
# Data Cleaner
# ==========================================================

class DataCleaner:
    """
    Data Cleaning Pipeline

    Workflow

    Raw Dataset
          │
          ▼
    Load Dataset
          │
          ▼
    Remove Duplicate
          │
          ▼
    Handle Missing Values
          │
          ▼
    Validate Data Types
          │
          ▼
    Validate Numeric Range
          │
          ▼
    Standardize Text
          │
          ▼
    Save Clean Dataset
    """

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self):

        self.input_file = RAW_DATASET_XLSX

        self.output_excel = os.path.join(
            PROCESSED_DATA_FOLDER,
            "cleaned_dataset.xlsx"
        )

        self.output_csv = os.path.join(
            PROCESSED_DATA_FOLDER,
            "cleaned_dataset.csv"
        )

        self.df = None

        # Statistics

        self.before_rows = 0

        self.after_rows = 0

        self.duplicate_removed = 0

        self.missing_fixed = 0

        self.numeric_fixed = 0

        self.text_standardized = 0

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):
        """
        Load raw dataset into DataFrame.
        """

        print()

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        if not os.path.exists(
            self.input_file
        ):

            raise FileNotFoundError(

                f"Dataset not found:\n"

                f"{self.input_file}"

            )

        self.df = pd.read_excel(
            self.input_file
        )

        self.before_rows = len(
            self.df
        )

        print()

        print(
            f"Dataset Loaded Successfully"
        )

        print(
            f"Rows : {self.before_rows}"
        )

        print(
            f"Columns : {len(self.df.columns)}"
        )

        print()

        print("Columns")

        for column in self.df.columns:

            print(
                f" - {column}"
            )

    # ======================================================
    # Remove Duplicate Records
    # ======================================================

    def remove_duplicates(self):
        """
        Remove duplicated student records.
        """

        print()

        print("=" * 60)
        print("Removing Duplicate Records...")
        print("=" * 60)

        before = len(self.df)

        self.df = self.df.drop_duplicates()

        after = len(self.df)

        self.duplicate_removed = before - after

        print(
            f"Duplicate Records Removed : "
            f"{self.duplicate_removed}"
        )

    # ======================================================
    # Handle Missing Values
    # ======================================================

    def handle_missing_values(self):
        """
        Fill missing values using suitable defaults.
        """

        print()

        print("=" * 60)
        print("Handling Missing Values...")
        print("=" * 60)

        total_missing = int(
            self.df.isnull().sum().sum()
        )

        self.missing_fixed = total_missing

        if total_missing == 0:

            print("No Missing Values Found.")

            return

        # ------------------------------
        # Numeric Columns
        # ------------------------------

        numeric_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        for column in numeric_columns:

            self.df[column] = self.df[column].fillna(
                0
            )

        # ------------------------------
        # Boolean Columns
        # ------------------------------

        boolean_columns = self.df.select_dtypes(
            include=["bool"]
        ).columns

        for column in boolean_columns:

            self.df[column] = self.df[column].fillna(
                False
            )

        # ------------------------------
        # Object Columns
        # ------------------------------

        object_columns = self.df.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:

            self.df[column] = self.df[column].fillna(
                "Unknown"
            )

        print(
            f"Missing Values Fixed : "
            f"{self.missing_fixed}"
        )

    # ======================================================
    # Validate Data Types
    # ======================================================

    def validate_data_types(self):
        """
        Convert columns into correct data types.
        """

        print()

        print("=" * 60)
        print("Validating Data Types...")
        print("=" * 60)

        # ------------------------------
        # Integer Columns
        # ------------------------------

        integer_columns = [

            "Current_Semester",

            "Total_Credits",

            "Credits_Completed",

            "Credits_Remaining",

            "Remaining_To_OJT",

            "Failed_Courses",

            "Retake_Count",

            "Missing_Prerequisite_Courses",

            "Academic_Warning_Count",

            "Suspension_Count",

            "Risk_Score",

            "Planned_OJT_Semester"

        ]

        for column in integer_columns:

            if column in self.df.columns:

                self.df[column] = (
                    pd.to_numeric(
                        self.df[column],
                        errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                )

        # ------------------------------
        # Float Columns
        # ------------------------------

        float_columns = [

            "GPA_Cumulative",

            "Completion_Rate",

            "Average_Credits_Per_Semester",

            "OJT_Readiness"

        ]

        for column in float_columns:

            if column in self.df.columns:

                self.df[column] = (
                    pd.to_numeric(
                        self.df[column],
                        errors="coerce"
                    )
                    .fillna(0)
                    .astype(float)
                )

        # ------------------------------
        # Boolean Columns
        # ------------------------------

        if "OJT_Eligible" in self.df.columns:

            self.df["OJT_Eligible"] = (
                self.df["OJT_Eligible"]
                .astype(bool)
            )

        print("Data Types Validated Successfully.")

        print()

        print(self.df.dtypes)

        # ======================================================
    # Validate Numeric Range
    # ======================================================

    def validate_numeric_range(self):
        """
        Validate and correct numeric values
        to ensure they stay within acceptable ranges.
        """

        print()

        print("=" * 60)
        print("Validating Numeric Range...")
        print("=" * 60)

        fixed = 0

        # --------------------------------------------------
        # GPA (0 -> 10)
        # --------------------------------------------------

        if "GPA_Cumulative" in self.df.columns:

            before = self.df["GPA_Cumulative"].copy()

            self.df["GPA_Cumulative"] = (
                self.df["GPA_Cumulative"]
                .clip(0, 10)
            )

            fixed += (
                before !=
                self.df["GPA_Cumulative"]
            ).sum()

        # --------------------------------------------------
        # Credits
        # --------------------------------------------------

        credit_columns = [

            "Total_Credits",

            "Credits_Completed",

            "Credits_Remaining",

            "Remaining_To_OJT"

        ]

        for column in credit_columns:

            if column in self.df.columns:

                before = self.df[column].copy()

                self.df[column] = (
                    self.df[column]
                    .clip(0, 120)
                )

                fixed += (
                    before !=
                    self.df[column]
                ).sum()

        # --------------------------------------------------
        # Completion Rate
        # --------------------------------------------------

        if "Completion_Rate" in self.df.columns:

            before = self.df[
                "Completion_Rate"
            ].copy()

            self.df["Completion_Rate"] = (

                self.df[
                    "Completion_Rate"
                ].clip(0, 100)

            )

            fixed += (

                before !=

                self.df[
                    "Completion_Rate"
                ]

            ).sum()

        # --------------------------------------------------
        # Readiness
        # --------------------------------------------------

        if "OJT_Readiness" in self.df.columns:

            before = self.df[
                "OJT_Readiness"
            ].copy()

            self.df["OJT_Readiness"] = (

                self.df[
                    "OJT_Readiness"
                ].clip(0, 100)

            )

            fixed += (

                before !=

                self.df[
                    "OJT_Readiness"
                ]

            ).sum()

        # --------------------------------------------------
        # Semester
        # --------------------------------------------------

        if "Current_Semester" in self.df.columns:

            before = self.df[
                "Current_Semester"
            ].copy()

            self.df[
                "Current_Semester"
            ] = (

                self.df[
                    "Current_Semester"
                ].clip(1, 8)

            )

            fixed += (

                before !=

                self.df[
                    "Current_Semester"
                ]

            ).sum()

        self.numeric_fixed = int(fixed)

        print()

        print(

            f"Numeric Values Corrected : "

            f"{self.numeric_fixed}"

        )

    # ======================================================
    # Standardize Text Columns
    # ======================================================

    def standardize_text_columns(self):
        """
        Standardize all text columns.
        """

        print()

        print("=" * 60)
        print("Standardizing Text Columns...")
        print("=" * 60)

        text_columns = self.df.select_dtypes(
            include=["object", "string"]
        ).columns

        count = 0

        for column in text_columns:

            before = self.df[column].copy()

            self.df[column] = (

                self.df[column]

                .astype(str)

                .str.strip()

            )

            # ------------------------------------------
            # Student Profile
            # ------------------------------------------

            if column == "Student_Profile":

                profile_map = {

                    "excellent": "Excellent",

                    "good": "Good",

                    "average": "Average",

                    "atrisk": "AtRisk",

                    "critical": "Critical",

                    "recovery": "Recovery",

                    "latestarter": "LateStarter"

                }

                self.df[column] = (

                    self.df[column]

                    .str.lower()

                    .replace(profile_map)

                )

            # ------------------------------------------
            # Risk Level
            # ------------------------------------------

            elif column == "Risk_Level":

                risk_map = {

                    "low": "Low",

                    "medium": "Medium",

                    "high": "High"

                }

                self.df[column] = (

                    self.df[column]

                    .str.lower()

                    .replace(risk_map)

                )

            # ------------------------------------------
            # Recommendation
            # ------------------------------------------

            elif column == "AI_Recommendation":

                self.df[column] = (

                    self.df[column]

                    .str.replace(
                        r"\s+",
                        " ",
                        regex=True
                    )

                )

            else:

                self.df[column] = (

                    self.df[column]

                    .str.strip()

                )

        count += int(
            (before != self.df[column]).sum()
        )

        self.text_standardized = count

        print()

        print(

            f"Text Values Standardized : "

            f"{self.text_standardized}"

        )

        print()

        print("Text Cleaning Completed.")   

        # ======================================================
    # Save Clean Dataset
    # ======================================================

    def save_dataset(self):
        """
        Save cleaned dataset into processed folder.
        """

        print()

        print("=" * 60)
        print("Saving Clean Dataset...")
        print("=" * 60)

        os.makedirs(
            PROCESSED_DATA_FOLDER,
            exist_ok=True
        )

        self.df.to_excel(
            self.output_excel,
            index=False
        )

        self.df.to_csv(
            self.output_csv,
            index=False
        )

        self.after_rows = len(
            self.df
        )

        print()

        print("Dataset Saved Successfully.")

        print(
            f"Excel : {self.output_excel}"
        )

        print(
            f"CSV    : {self.output_csv}"
        )

    # ======================================================
    # Print Summary
    # ======================================================

    def print_summary(self):
        """
        Display cleaning summary.
        """

        print()

        print("=" * 60)
        print("DATA CLEANING SUMMARY")
        print("=" * 60)

        print(
            f"Rows Before Cleaning      : {self.before_rows}"
        )

        print(
            f"Rows After Cleaning       : {self.after_rows}"
        )

        print(
            f"Duplicate Removed         : {self.duplicate_removed}"
        )

        print(
            f"Missing Values Fixed      : {self.missing_fixed}"
        )

        print(
            f"Numeric Values Corrected  : {self.numeric_fixed}"
        )

        print(
            f"Text Values Standardized  : {self.text_standardized}"
        )

        print()

        print("Output Files")

        print(
            f"Excel : {self.output_excel}"
        )

        print(
            f"CSV   : {self.output_csv}"
        )

        print("=" * 60)

    # ======================================================
    # Run Cleaning Pipeline
    # ======================================================

    def clean(self):
        """
        Execute complete cleaning pipeline.
        """

        self.load_dataset()

        self.remove_duplicates()

        self.handle_missing_values()

        self.validate_data_types()

        self.validate_numeric_range()

        self.standardize_text_columns()

        self.save_dataset()

        self.print_summary()


# ==========================================================
# Main
# ==========================================================

def main():

    cleaner = DataCleaner()

    cleaner.clean()


if __name__ == "__main__":

    main()             