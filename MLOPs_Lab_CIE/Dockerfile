FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4 joblib

COPY src/predict_cli.py src/predict_cli.py
COPY models/ridge_model.pkl models/ridge_model.pkl

ENTRYPOINT ["python", "src/predict_cli.py"]