FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn joblib fastapi uvicorn mlflow evidently

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]