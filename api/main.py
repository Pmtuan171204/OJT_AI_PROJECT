import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# FASTAPI
# ==========================================================

from fastapi import FastAPI, HTTPException

# ==========================================================
# PROJECT MODULES
# ==========================================================

from api.schemas import StudentPredictionRequest, PredictionResponse

from training.prediction.prediction_pipeline import predict_student

from training.recommendation.recommendation_engine import generate_recommendation

# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="OJT AI Prediction & Recommendation API",
    description=(
        "AI system for predicting OJT delay risk "
        "and generating personalized recommendations."
    ),
    version="2.0.0",
)


# ==========================================================
# ROOT ENDPOINT
# ==========================================================


@app.get("/")
def root():

    return {
        "project": "OJT AI Project",
        "service": "OJT Prediction & Recommendation API",
        "version": "2.0.0",
        "status": "running",
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================


@app.get("/health")
def health_check():

    return {"status": "healthy", "service": "ojt-ai-api"}


# ==========================================================
# PREDICTION ENDPOINT
# ==========================================================


@app.post("/predict", response_model=PredictionResponse)
def predict(request: StudentPredictionRequest):

    try:

        # --------------------------------------------------
        # Convert Pydantic object to dictionary
        # --------------------------------------------------

        student = request.model_dump()

        # --------------------------------------------------
        # Prediction Pipeline
        # --------------------------------------------------

        prediction_result = predict_student(student)

        # --------------------------------------------------
        # Recommendation Engine
        # --------------------------------------------------

        recommendation_result = generate_recommendation(student, prediction_result)

        # --------------------------------------------------
        # Return combined result
        # --------------------------------------------------

        return recommendation_result

    except ValueError as error:

        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:

        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(error)}"
        )
