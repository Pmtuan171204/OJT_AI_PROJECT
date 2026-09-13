from typing import List
from pydantic import BaseModel, Field

# ==========================================================
# PREDICTION REQUEST
# ==========================================================


class StudentPredictionRequest(BaseModel):

    Student_Profile: str

    Current_Semester: int = Field(ge=1, le=8)

    GPA_Cumulative: float = Field(ge=0, le=10)

    Total_Credits: int = Field(ge=0)

    Credits_Completed: int = Field(ge=0)

    Credits_Remaining: int = Field(ge=0)

    Completion_Rate: float = Field(ge=0, le=1)

    Average_Credits_Per_Semester: float = Field(ge=0)

    Remaining_To_OJT: int = Field(ge=0)

    Failed_Courses: int = Field(ge=0)

    Retake_Count: int = Field(ge=0)

    Missing_Prerequisite_Courses: int = Field(ge=0)

    Academic_Warning_Count: int = Field(ge=0)

    Suspension_Count: int = Field(ge=0)

    Planned_OJT_Semester: int = Field(ge=1, le=8)


# ==========================================================
# API RESPONSE
# ==========================================================


class PredictionResponse(BaseModel):

    prediction: dict

    current_metrics: dict

    issues: List[dict]

    recommendations: List[dict]

    roadmap: List[dict]
