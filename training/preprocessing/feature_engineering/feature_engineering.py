"""
==========================================================
Feature Engineering Module
OJT AI Project

Sprint 3 - Module 2

Purpose:
    Create meaningful features for:

    1. OJT Delay Risk Prediction
    2. OJT Eligibility Analysis
    3. OJT Roadmap Recommendation
    4. Academic Progress Analysis

Input:
    data/processed/cleaned_dataset.xlsx

Output:
    data/processed/feature_dataset.xlsx
    data/processed/feature_dataset.csv

==========================================================
"""

import os
import sys

import pandas as pd


# ==========================================================
# Add Project Root to Python Path
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
# Import Generator Configuration
# ==========================================================

from data.generator.config import (
    REQUIRED_CREDITS_FOR_OJT
)


# ==========================================================
# Feature Engineer
# ==========================================================

class FeatureEngineer:
    """
    Feature Engineering Pipeline.

    Workflow:

        Cleaned Dataset
              |
              v
        Academic Progress Features
              |
              v
        OJT Eligibility Features
              |
              v
        Academic Risk Features
              |
              v
        OJT Planning Features
              |
              v
        Study Efficiency Features
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

        self.input_file = os.path.join(
            PROCESSED_DATA_FOLDER,
            "cleaned_dataset.xlsx"
        )

        # --------------------------------------------------
        # Output Dataset
        # --------------------------------------------------

        self.output_excel = os.path.join(
            PROCESSED_DATA_FOLDER,
            "feature_dataset.xlsx"
        )

        self.output_csv = os.path.join(
            PROCESSED_DATA_FOLDER,
            "feature_dataset.csv"
        )

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

        print(
            "Loading Cleaned Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Check Dataset
        # --------------------------------------------------

        if not os.path.exists(
            self.input_file
        ):

            raise FileNotFoundError(

                "Cleaned dataset not found.\n"

                f"Expected path:\n"
                f"{self.input_file}"

            )

        # --------------------------------------------------
        # Read Excel Dataset
        # --------------------------------------------------

        self.df = pd.read_excel(
            self.input_file
        )

        # --------------------------------------------------
        # Store Initial Column Count
        # --------------------------------------------------

        self.before_columns = len(
            self.df.columns
        )

        self.original_columns = list(
            self.df.columns
        )

        # --------------------------------------------------
        # Print Dataset Information
        # --------------------------------------------------

        print()

        print(
            "Cleaned Dataset Loaded Successfully."
        )

        print(
            f"Rows : {len(self.df)}"
        )

        print(
            f"Columns : {self.before_columns}"
        )

        print()

        print(
            "Dataset Columns:"
        )

        for column in self.df.columns:

            print(
                f" - {column}"
            )

    # ======================================================
    # Create Academic Progress Features
    # ======================================================

    def create_progress_features(self):
        """
        Create features related to student's academic progress.
        """

        print()

        print("=" * 60)

        print(
            "Creating Academic Progress Features..."
        )

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

        self.df[
            "Credit_Progress_Category"
        ] = self.df[
            "Completion_Rate"
        ].apply(
            classify_progress
        )

        print(
            "Created: Credit_Progress_Category"
        )

    # ======================================================
    # Create OJT Eligibility Features
    # ======================================================

    def create_ojt_eligibility_features(self):
        """
        Create features related to OJT eligibility.

        Note:
            GPA is NOT used to determine OJT eligibility.

            The main OJT credit requirement is:
                REQUIRED_CREDITS_FOR_OJT = 100
        """

        print()

        print("=" * 60)

        print(
            "Creating OJT Eligibility Features..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # OJT Credit Gap
        # --------------------------------------------------

        self.df[
            "OJT_Credit_Gap"
        ] = (

            REQUIRED_CREDITS_FOR_OJT

            - self.df[
                "Credits_Completed"
            ]

        ).clip(
            lower=0
        )

        print(
            "Created: OJT_Credit_Gap"
        )

        # --------------------------------------------------
        # OJT Credit Progress
        # --------------------------------------------------

        self.df[
            "OJT_Credit_Progress"
        ] = (

            self.df[
                "Credits_Completed"
            ]

            / REQUIRED_CREDITS_FOR_OJT

            * 100

        ).clip(
            upper=100
        ).round(
            2
        )

        print(
            "Created: OJT_Credit_Progress"
        )

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

        self.df[
            "OJT_Eligibility_Gap_Category"
        ] = self.df[
            "OJT_Credit_Gap"
        ].apply(
            classify_ojt_gap
        )

        print(
            "Created: OJT_Eligibility_Gap_Category"
        )

    # ======================================================
    # Create Academic Risk Features
    # ======================================================

    def create_risk_features(self):
        """
        Create additional features related to academic risk.

        Important:
            OJT_Delay_Risk is NOT used to create input features.

            This helps prevent data leakage when training
            the OJT delay risk prediction model.
        """

        print()

        print("=" * 60)

        print(
            "Creating Academic Risk Features..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Academic Risk Index
        # --------------------------------------------------

        self.df[
            "Academic_Risk_Index"
        ] = (

            self.df[
                "Failed_Courses"
            ]

            + self.df[
                "Retake_Count"
            ]

            + self.df[
                "Missing_Prerequisite_Courses"
            ]

            + self.df[
                "Academic_Warning_Count"
            ]

            + self.df[
                "Suspension_Count"
            ]

        )

        print(
            "Created: Academic_Risk_Index"
        )

        # --------------------------------------------------
        # Academic Stability
        # --------------------------------------------------

        self.df[
            "Academic_Stability"
        ] = (

            100

            - (

                self.df[
                    "Academic_Risk_Index"
                ]

                * 10

            )

        ).clip(
            lower=0,
            upper=100
        )

        print(
            "Created: Academic_Stability"
        )

    # ======================================================
    # Create OJT Planning Features
    # ======================================================

    def create_planning_features(self):
        """
        Create features related to OJT planning
        and semester progress.
        """

        print()

        print("=" * 60)

        print(
            "Creating OJT Planning Features..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # OJT Semester Gap
        # --------------------------------------------------

        self.df[
            "OJT_Semester_Gap"
        ] = (

            self.df[
                "Planned_OJT_Semester"
            ]

            - self.df[
                "Current_Semester"
            ]

        ).clip(
            lower=0
        )

        print(
            "Created: OJT_Semester_Gap"
        )

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

        self.df[
            "OJT_Planning_Status"
        ] = self.df[
            "OJT_Semester_Gap"
        ].apply(
            classify_planning_status
        )

        print(
            "Created: OJT_Planning_Status"
        )

    # ======================================================
    # Create Study Efficiency Features
    # ======================================================

    def create_efficiency_features(self):
        """
        Create features related to credit completion
        efficiency and required academic progress.
        """

        print()

        print("=" * 60)

        print(
            "Creating Study Efficiency Features..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Required Average Credits Per Semester
        # --------------------------------------------------

        remaining_semesters = (

            self.df[
                "Planned_OJT_Semester"
            ]

            - self.df[
                "Current_Semester"
            ]

        ).clip(
            lower=1
        )

        self.df[
            "Required_Average_Credits_Per_Semester"
        ] = (

            self.df[
                "OJT_Credit_Gap"
            ]

            / remaining_semesters

        ).round(
            2
        )

        print(
            "Created: "
            "Required_Average_Credits_Per_Semester"
        )

        # --------------------------------------------------
        # Credit Completion Efficiency
        # --------------------------------------------------

        self.df[
            "Credit_Completion_Efficiency"
        ] = (

            self.df[
                "Credits_Completed"
            ]

            / (

                self.df[
                    "Current_Semester"
                ]

                * self.df[
                    "Average_Credits_Per_Semester"
                ]

            ).replace(
                0,
                1
            )

            * 100

        ).clip(
            upper=100
        ).round(
            2
        )

        print(
            "Created: Credit_Completion_Efficiency"
        )

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

        print(
            "Saving Feature Dataset..."
        )

        print("=" * 60)

        # --------------------------------------------------
        # Create Processed Folder If Not Exists
        # --------------------------------------------------

        os.makedirs(
            PROCESSED_DATA_FOLDER,
            exist_ok=True
        )

        # --------------------------------------------------
        # Save Excel
        # --------------------------------------------------

        self.df.to_excel(
            self.output_excel,
            index=False
        )

        # --------------------------------------------------
        # Save CSV
        # --------------------------------------------------

        self.df.to_csv(
            self.output_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print()

        print(
            "Feature Dataset Saved Successfully."
        )

        print(
            f"Excel : {self.output_excel}"
        )

        print(
            f"CSV   : {self.output_csv}"
        )

    # ======================================================
    # Print Feature Engineering Summary
    # ======================================================

    def print_summary(self):
        """
        Print summary after feature engineering.
        """

        self.after_columns = len(
            self.df.columns
        )

        self.features_created = (

            self.after_columns

            - self.before_columns

        )

        print()

        print("=" * 60)

        print(
            "FEATURE ENGINEERING SUMMARY"
        )

        print("=" * 60)

        print(
            f"Rows : {len(self.df)}"
        )

        print(
            f"Columns Before : "
            f"{self.before_columns}"
        )

        print(
            f"Columns After  : "
            f"{self.after_columns}"
        )

        print(
            f"New Features Created : "
            f"{self.features_created}"
        )

        print()

        print(
            "New Features:"
        )

        for column in self.df.columns:

            if column not in self.original_columns:

                print(
                    f" - {column}"
                )

        print()

        print(
            "Output Files"
        )

        print(
            f"Excel : {self.output_excel}"
        )

        print(
            f"CSV   : {self.output_csv}"
        )

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
        # Step 2: Create Academic Progress Features
        # --------------------------------------------------

        self.create_progress_features()

        # --------------------------------------------------
        # Step 3: Create OJT Eligibility Features
        # --------------------------------------------------

        self.create_ojt_eligibility_features()

        # --------------------------------------------------
        # Step 4: Create Academic Risk Features
        # --------------------------------------------------

        self.create_risk_features()

        # --------------------------------------------------
        # Step 5: Create OJT Planning Features
        # --------------------------------------------------

        self.create_planning_features()

        # --------------------------------------------------
        # Step 6: Create Study Efficiency Features
        # --------------------------------------------------

        self.create_efficiency_features()

        # --------------------------------------------------
        # Step 7: Save Dataset
        # --------------------------------------------------

        self.save_dataset()

        # --------------------------------------------------
        # Step 8: Print Summary
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