# PFDECTECTOR

A small machine learning project that predicts whether a student will **Pass** or **Fail**.

It uses:
- **Python + scikit-learn** to train the model
- **MLflow** to record every training run
- **FastAPI** to serve the model as an API
- **Docker** to package it, and **Render** to host it

## Project structure

```
ml/
├── src/
│   ├── app.py            FastAPI app (/predict, /train)
│   └── ML_TRACKING.py    training code, logs runs to MLflow
├── dataset/              student data (student_cleaned.csv is used for training)
├── models/               saved models (model.pkl is the one the API uses)
├── tests/
│   └── test_api.py       sends one test request to the API
├── notebooks/            experiments
├── Dockerfile            builds the image
├── .dockerignore         files kept out of the image
├── requirements.txt      Python packages
└── README.md
```

Run every command below from the project folder (`ml/`).

## 1. Setup (one time)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell says activation is blocked, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 2. Train the model

```powershell
python -m src.ML_TRACKING
```

This trains 3 models with different settings and prints the accuracy of each.
The best model is saved as `models/model.pkl`.

## 3. See the results in MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 and click the `students` experiment.

## 4. Run the API

```powershell
python -m uvicorn src.app:app --reload
```

Open http://127.0.0.1:8000/docs and try `/predict`.

Test it from code:

```powershell
python tests/test_api.py
```

## 5. Run with Docker

```powershell
docker build -t student-prediction-api .
docker run -d -p 10000:10000 --name student-api-container student-prediction-api
```

Open http://localhost:10000/docs. Stop and remove it with `docker rm -f student-api-container`.

## 6. Deploy on Render

1. Push the repo to GitHub.
2. In Render choose **New, Web Service**, connect the repo and pick **Docker**.
3. Dockerfile Path: `./Dockerfile`. Health Check Path: `/openapi.json`.

Render sets the `PORT` variable itself and the Dockerfile reads it.
