# PFDECTECTOR

A small machine learning project that predicts whether a student will **Pass** or **Fail**.

It uses:
- **Python + scikit-learn** to train the model
- **MLflow** to record every training run
- **FastAPI** to use the model from a web page

## 1. Setup (one time)

Open PowerShell in the project folder and run:

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
python ML_TRACKING.py
```

This trains 3 models with different settings and prints the accuracy of each.
The best model is saved as `models/model.pkl`.

## 3. See the results in MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 in your browser and click the `students` experiment.

## 4. Use the model in the browser

```powershell
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000 to enter a student's details and see Pass or Fail on a scatter plot.
The API docs are at http://127.0.0.1:8000/docs.

## 5. Project files

| File or folder | What it is |
|---|---|
| `ML_TRACKING.py` | the training code |
| `app.py` | the FastAPI app |
| `static/index.html` | the web page with the scatter plot |
| `dataset/` | the student data (`student_cleaned.csv` is used for training) |
| `models/` | saved models |
| `requirements.txt` | the Python packages needed |
