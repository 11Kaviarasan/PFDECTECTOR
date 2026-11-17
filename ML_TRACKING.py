import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset/student_cleaned.csv")

x = data[["study_hours", "attendance", "previous_marks"]]
y = data["result"]

mlflow.set_experiment("students")

# (train fraction, n_estimators) for each run
runs = [(0.5, 10), (0.8, 50), (0.3, 100)]
best_acc = 0

for i, (train_size, n_estimators) in enumerate(runs, start=1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

    with mlflow.start_run(run_name=f"train={train_size}_trees={n_estimators}"):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("train_size", train_size)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name=f"v{i}", serialization_format="cloudpickle")

    print(f"train={train_size}  trees={n_estimators}  accuracy={acc * 100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        joblib.dump(model, "models/model.pkl")
