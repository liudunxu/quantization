FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir catboost scikit-learn numpy pandas scipy \
    akshare yfinance baostock statsmodels joblib \
    python-dotenv PyYAML fastapi uvicorn \
    || true

COPY . .

RUN mkdir -p cache data logs models

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "scripts/predict.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
