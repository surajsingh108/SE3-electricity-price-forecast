FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt \
    && find /root/.local/lib -type d -name "tests" -exec rm -rf {} + 2>/dev/null; true \
    && find /root/.local/lib -type d -name "test"  -exec rm -rf {} + 2>/dev/null; true \
    && find /root/.local/lib -name "*.pyi"         -delete              2>/dev/null; true \
    && find /root/.local/lib -path "*/plotly/package_data*" -name "*.json" \
       -delete 2>/dev/null; true \
    && find /root/.local/lib -path "*/streamlit/static*" -name "*.map" \
       -delete 2>/dev/null; true

COPY pipeline.py ml.py ml_imbalance.py dashboard.py api.py retrain.py retrain_imbalance.py ./
COPY data_sources.py feature_engineering.py ./
COPY agent ./agent

RUN mkdir -p /app/data /app/model

ENV SE3_DB_PATH=/app/data/se3_cache.duckdb \
    MODEL_DIR=/app/model \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY supervisord.conf .

RUN pip install --no-cache-dir supervisor

EXPOSE 8080

# Run both api.py and dashboard.py via supervisord
CMD ["supervisord", "-c", "/app/supervisord.conf"]
