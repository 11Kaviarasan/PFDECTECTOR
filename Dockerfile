FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ML_TRACKING.py ./
COPY dataset/student_cleaned.csv dataset/
COPY models/model.pkl models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10000/openapi.json')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
