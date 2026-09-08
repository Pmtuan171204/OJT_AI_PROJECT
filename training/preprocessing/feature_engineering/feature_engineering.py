"""
==========================================================
Feature Engineering Module
OJT AI Project

Sprint 3 - Feature Engineering V2

Purpose:
    Create meaningful predictive features for:

    1. OJT Delay Outcome Prediction
    2. Academic Progress Analysis
    3. OJT Progress Analysis
    4. OJT Planning Analysis
    5. Study Efficiency Analysis

Important:
    This version follows the Genuine Predictive AI approach.

    The model must learn from the student's current
    academic state instead of reproducing an existing
    Risk Score / Risk Label.

Input:
    data/processed/cleaned_dataset.xlsx

Output:
    data/processed/feature_dataset.xlsx
    data/processed/feature_dataset.csv

Target:
    OJT_Delay_Outcome

Leakage-related columns excluded from model features:
    Risk_Score
    Risk_Level
    OJT_Delay_Risk
    AI_Recommendation
    Academic_Risk_Index
    Academic_Stability
    OJT_Eligible
    OJT_Readiness

==========================================================
"""

import os
import sys

import pandas as pd

# ==========================================================
# Add Project Root to Python Path
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# Import Project Paths
# ==========================================================

from config.paths import (
    PROCESSED_DATA_DIR,
    CLEANED_DATASET_XLSX,
    FEATURE_DATASET_XLSX,
    FEATURE_DATASET_CSV,
)

# ==========================================================
# Import Generator Configuration
# ==========================================================

from data.generator.config import REQUIRED_CREDITS_FOR_OJT

# ==========================================================
# Feature Engineer
# ==========================================================


class FeatureEngineer:
    """
    Feature Engineering Pipeline V2.

    Workflow:

        Cleaned Dataset
              |
              v
        Academic Progress Features
              |
              v
        OJT Progress Features
              |
              v
        OJT Planning Features
              |
              v
        Study Efficiency Features
              |
              v
        Leakage Check
              |
              v
        Feature Dataset
    """

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self):

        # --------------------------------------------------
        # Input Dataset
        # --------------------------------------------------

        self.input_file = CLEANED_DATASET_XLSX

        # --------------------------------------------------
        # Output Dataset
        # --------------------------------------------------

        self.output_excel = FEATURE_DATASET_XLSX

        self.output_csv = FEATURE_DATASET_CSV

        # --------------------------------------------------
        # DataFrame
        # --------------------------------------------------

        self.df = None

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.before_columns = 0

        self.after_columns = 0

        self.features_created = 0

        self.original_columns = []

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):
        """
        Load cleaned dataset from processed folder.
        """

        print()
        print("=" * 60)
        print("Loading Cleaned Dataset...")
        print("=" * 60)

        # --------------------------------------------------
        # Check Dataset
        # --------------------------------------------------

        if not os.path.exists(self.input_file):

            raise FileNotFoundError(
                "Cleaned dataset not found.\n" f"Expected path:\n{self.input_file}"
            )

        # --------------------------------------------------
        # Read Excel Dataset
        # --------------------------------------------------

        self.df = pd.read_excel(self.input_file)

        # --------------------------------------------------
        # Store Initial Column Count
        # --------------------------------------------------

        self.before_columns = len(self.df.columns)

        self.original_columns = list(self.df.columns)

        # --------------------------------------------------
        # Print Dataset Information
        # --------------------------------------------------

        print()
        print("Cleaned Dataset Loaded Successfully.")

        print(f"Rows : {len(self.df)}")

        print(f"Columns : {self.before_columns}")

        print()

        print("Dataset Columns:")

        for column in self.df.columns:

            print(f" - {column}")

    # ======================================================
    # Validate Required Columns
    # ======================================================

    def validate_required_columns(self):
        """
        Check whether all columns required for feature
        engineering are available.
        """

        print()
        print("=" * 60)
        print("Validating Required Columns...")
        print("=" * 60)

        required_columns = [
            # Basic academic information
            "MSSV",
            "Student_Profile",
            "Current_Semester",
            "GPA_Cumulative",
            # Credit information
            "Total_Credits",
            "Credits_Completed",
            "Credits_Remaining",
            "Completion_Rate",
            "Average_Credits_Per_Semester",
            "Remaining_To_OJT",
            # Academic information
            "Failed_Courses",
            "Retake_Count",
            "Missing_Prerequisite_Courses",
            "Academic_Warning_Count",
            "Suspension_Count",
            # OJT planning
            "Planned_OJT_Semester",
            # Target
            "OJT_Delay_Outcome",
        ]

        missing_columns = [
            column for column in required_columns if column not in self.df.columns
        ]

        if len(missing_columns) > 0:

            raise ValueError(
                "Missing required columns:\n"
                + "\n".join(f" - {column}" for column in missing_columns)
            )

        print("All required columns are available.")

    # ======================================================
    # Create Academic Progress Features
    # ======================================================

    def create_progress_features(self):
        """
        Create features related to student's current
        academic progress.
        """

        print()
        print("=" * 60)
        print("Creating Academic Progress Features...")
        print("=" * 60)

        # --------------------------------------------------
        # Credit Progress Category
        # --------------------------------------------------

        def classify_progress(rate):

            if rate < 30:

                return "Early"

            elif rate < 60:

                return "Developing"

            elif rate < 80:

                return "Advanced"

            else:

                return "Near_Completion"

        self.df["Credit_Progress_Category"] = self.df["Completion_Rate"].apply(
            classify_progress
        )

        print("Created: Credit_Progress_Category")

    # ======================================================
    # Create OJT Progress Features
    # ======================================================

    def create_ojt_progress_features(self):
        """
        Create features describing the student's current
        progress toward the OJT credit requirement.

        Important:
            These features are based only on the current
            academic state.

            They do not use future outcome information.
        """

        print()
        print("=" * 60)
        print("Creating OJT Progress Features...")
        print("=" * 60)

        # --------------------------------------------------
        # OJT Credit Gap
        # --------------------------------------------------

        self.df["OJT_Credit_Gap"] = (
            REQUIRED_CREDITS_FOR_OJT - self.df["Credits_Completed"]
        ).clip(lower=0)

        print("Created: OJT_Credit_Gap")

        # --------------------------------------------------
        # OJT Credit Progress
        # --------------------------------------------------

        self.df["OJT_Credit_Progress"] = (
            (self.df["Credits_Completed"] / REQUIRED_CREDITS_FOR_OJT * 100)
            .clip(upper=100)
            .round(2)
        )

        print("Created: OJT_Credit_Progress")

        # --------------------------------------------------
        # OJT Eligibility Gap Category
        # --------------------------------------------------

        def classify_ojt_gap(gap):

            if gap == 0:

                return "Eligible"

            elif gap <= 20:

                return "Near_Eligible"

            elif gap <= 50:

                return "Moderate_Gap"

            else:

                return "Far_From_Eligible"

        self.df["OJT_Eligibility_Gap_Category"] = self.df["OJT_Credit_Gap"].apply(
            classify_ojt_gap
        )

        print("Created: OJT_Eligibility_Gap_Category")

    # ======================================================
    # Create OJT Planning Features
    # ======================================================

    def create_planning_features(self):
        """
        Create features related to OJT planning
        and semester progress.

        All features are based on current/planned
        semester information.
        """

        print()
        print("=" * 60)
        print("Creating OJT Planning Features...")
        print("=" * 60)

        # --------------------------------------------------
        # OJT Semester Gap
        # --------------------------------------------------

        self.df["OJT_Semester_Gap"] = (
            self.df["Planned_OJT_Semester"] - self.df["Current_Semester"]
        ).clip(lower=0)

        print("Created: OJT_Semester_Gap")

        # --------------------------------------------------
        # OJT Planning Status
        # --------------------------------------------------

        def classify_planning_status(gap):

            if gap == 0:

                return "OJT_Ready_Stage"

            elif gap == 1:

                return "Approaching_OJT"

            elif gap <= 2:

                return "Moderate_Planning_Gap"

            else:

                return "Long_Term_Planning"

        self.df["OJT_Planning_Status"] = self.df["OJT_Semester_Gap"].apply(
            classify_planning_status
        )

        print("Created: OJT_Planning_Status")

    # ======================================================
    # Create Study Efficiency Features
    # ======================================================

    def create_efficiency_features(self):
        """
        Create features related to the student's
        current study/credit completion efficiency.
        """

        print()
        print("=" * 60)
        print("Creating Study Efficiency Features...")
        print("=" * 60)

        # --------------------------------------------------
        # Remaining Semesters Until Planned OJT
        # --------------------------------------------------

        remaining_semesters = (
            self.df["Planned_OJT_Semester"] - self.df["Current_Semester"]
        ).clip(lower=1)

        # --------------------------------------------------
        # Required Average Credits Per Semester
        # --------------------------------------------------

        self.df["Required_Average_Credits_Per_Semester"] = (
            self.df["OJT_Credit_Gap"] / remaining_semesters
        ).round(2)

        print("Created: " "Required_Average_Credits_Per_Semester")

        # --------------------------------------------------
        # Credit Completion Efficiency
        # --------------------------------------------------

        denominator = (
            self.df["Current_Semester"] * self.df["Average_Credits_Per_Semester"]
        ).replace(0, 1)

        self.df["Credit_Completion_Efficiency"] = (
            (self.df["Credits_Completed"] / denominator * 100).clip(upper=100).round(2)
        )

        print("Created: Credit_Completion_Efficiency")

    # ======================================================
    # Check Feature Leakage
    # ======================================================

    def check_feature_leakage(self):
        """
        Verify that forbidden risk/target-derived
        columns are not treated as engineered features.

        These columns must NOT be used as model inputs
        in the Genuine Predictive AI approach.
        """

        print()
        print("=" * 60)
        print("Checking Feature Leakage...")
        print("=" * 60)

        forbidden_features = [
            "Risk_Score",
            "Risk_Level",
            "OJT_Delay_Risk",
            "AI_Recommendation",
            "Academic_Risk_Index",
            "Academic_Stability",
            "OJT_Eligible",
            "OJT_Readiness",
            "Future_Credits_At_OJT",
            "Future_Failed_Courses",
            "Future_Missing_Prerequisites",
            "Future_Academic_Warning",
            "Future_OJT_Eligible",
        ]

        # --------------------------------------------------
        # Target is allowed to remain in dataset.
        # It will be separated later by Data Split.
        # --------------------------------------------------

        allowed_target = "OJT_Delay_Outcome"

        forbidden_present = [
            column for column in forbidden_features if column in self.df.columns
        ]

        # --------------------------------------------------
        # Important:
        # Forbidden columns can remain in the feature
        # dataset for reference, but they must NOT be
        # included in the model feature list.
        #
        # Therefore we only report their presence here.
        # --------------------------------------------------

        if allowed_target not in self.df.columns:

            raise ValueError("Target column " "'OJT_Delay_Outcome' " "was not found.")

        print("Target found: OJT_Delay_Outcome")

        if len(forbidden_present) > 0:

            print()
            print("Columns excluded from model features:")

            for column in forbidden_present:

                print(f" - {column}")

        else:

            print("No forbidden feature columns found.")

        print()
        print("Feature leakage check completed.")

    # ======================================================
    # Define Model Features
    # ======================================================

    def get_model_features(self):
        """
        Return the official feature list for the
        Genuine Predictive AI model.

        Only current-state information is allowed.
        """

        model_features = [
            # ------------------------------------------------
            # Academic State
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
    # Validate Model Feature List
    # ======================================================

    def validate_model_features(self):
        """
        Validate the official model feature list.
        """

        print()
        print("=" * 60)
        print("Validating Model Feature List...")
        print("=" * 60)

        model_features = self.get_model_features()

        missing_features = [
            feature for feature in model_features if feature not in self.df.columns
        ]

        if len(missing_features) > 0:

            raise ValueError(
                "Missing model features:\n"
                + "\n".join(f" - {feature}" for feature in missing_features)
            )

        # --------------------------------------------------
        # Explicit forbidden feature check
        # --------------------------------------------------

        forbidden_features = [
            "Risk_Score",
            "Risk_Level",
            "OJT_Delay_Risk",
            "AI_Recommendation",
            "Academic_Risk_Index",
            "Academic_Stability",
            "OJT_Eligible",
            "OJT_Readiness",
            "Future_Credits_At_OJT",
            "Future_Failed_Courses",
            "Future_Missing_Prerequisites",
            "Future_Academic_Warning",
            "Future_OJT_Eligible",
            "OJT_Delay_Outcome",
        ]

        leakage_features = [
            feature for feature in model_features if feature in forbidden_features
        ]

        if len(leakage_features) > 0:

            raise ValueError(
                "Forbidden leakage features detected:\n"
                + "\n".join(f" - {feature}" for feature in leakage_features)
            )

        print(f"Model Features : {len(model_features)}")

        print("No forbidden leakage features found " "in the model feature list.")

    # ======================================================
    # Save Feature Dataset
    # ======================================================

    def save_dataset(self):
        """
        Save the feature-engineered dataset.

        Output:
            data/processed/feature_dataset.xlsx
            data/processed/feature_dataset.csv
        """

        print()
        print("=" * 60)
        print("Saving Feature Dataset...")
        print("=" * 60)

        # --------------------------------------------------
        # Create Processed Folder
        # --------------------------------------------------

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # --------------------------------------------------
        # Save Excel
        # --------------------------------------------------

        self.df.to_excel(self.output_excel, index=False)

        # --------------------------------------------------
        # Save CSV
        # --------------------------------------------------

        self.df.to_csv(self.output_csv, index=False, encoding="utf-8-sig")

        print()
        print("Feature Dataset Saved Successfully.")

        print(f"Excel : {self.output_excel}")

        print(f"CSV   : {self.output_csv}")

    # ======================================================
    # Print Feature Engineering Summary
    # ======================================================

    def print_summary(self):
        """
        Print summary after feature engineering.
        """

        self.after_columns = len(self.df.columns)

        self.features_created = self.after_columns - self.before_columns

        print()
        print("=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)

        print(f"Rows : {len(self.df)}")

        print(f"Columns Before : " f"{self.before_columns}")

        print(f"Columns After  : " f"{self.after_columns}")

        print(f"New Features Created : " f"{self.features_created}")

        print()
        print("New Features:")

        for column in self.df.columns:

            if column not in self.original_columns:

                print(f" - {column}")

        print()

        print("Official Model Features:")

        for feature in self.get_model_features():

            print(f" - {feature}")

        print()

        print("Target:")

        print(" - OJT_Delay_Outcome")

        print()

        print("Output Files:")

        print(f"Excel : {self.output_excel}")

        print(f"CSV   : {self.output_csv}")

        print("=" * 60)

    # ======================================================
    # Run Feature Engineering Pipeline
    # ======================================================

    def transform(self):
        """
        Run the complete feature engineering pipeline.
        """

        # --------------------------------------------------
        # Step 1: Load Dataset
        # --------------------------------------------------

        self.load_dataset()

        # --------------------------------------------------
        # Step 2: Validate Required Columns
        # --------------------------------------------------

        self.validate_required_columns()

        # --------------------------------------------------
        # Step 3: Create Academic Progress Features
        # --------------------------------------------------

        self.create_progress_features()

        # --------------------------------------------------
        # Step 4: Create OJT Progress Features
        # --------------------------------------------------

        self.create_ojt_progress_features()

        # --------------------------------------------------
        # Step 5: Create OJT Planning Features
        # --------------------------------------------------

        self.create_planning_features()

        # --------------------------------------------------
        # Step 6: Create Study Efficiency Features
        # --------------------------------------------------

        self.create_efficiency_features()

        # --------------------------------------------------
        # Step 7: Check Feature Leakage
        # --------------------------------------------------

        self.check_feature_leakage()

        # --------------------------------------------------
        # Step 8: Validate Model Feature List
        # --------------------------------------------------

        self.validate_model_features()

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

    feature_engineer = FeatureEngineer()

    feature_engineer.transform()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
