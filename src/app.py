import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.ML_TRACKING import train

app = FastAPI(title="Student Pass/Fail API")


class Student(BaseModel):
    study_hours: float = Field(ge=0, le=24, description="Hours studied per day (0-24)")
    attendance: float = Field(ge=0, le=100, description="Attendance percentage (0-100)")
    previous_marks: float = Field(ge=0, le=100, description="Previous marks (0-100)")


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")  # Swagger UI


@app.post("/predict", summary="Predict Pass or Fail")
def predict(student: Student):
    try:
        model = joblib.load("models/model.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found. Call /train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load the model: {e}")

    try:
        features = [[student.study_hours, student.attendance, student.previous_marks]]
        result = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"result": result}


@app.post("/train", summary="Train the models")
def train_models():
    # trains the models, logs them to MLflow and saves the best one to models/model.pkl
    try:
        return {"runs": train()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Training data not found: {e.filename}")
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Column missing in the dataset: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
