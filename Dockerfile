FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && find /root/.local/lib -type d -name "tests" -exec rm -rf {} + 2>/dev/null; true \
    && find /root/.local/lib -type d -name "test"  -exec rm -rf {} + 2>/dev/null; true \
    && find /root/.local/lib -name "*.pyi"         -delete              2>/dev/null; true \
    && find /root/.local/lib -path "*/plotly/package_data*" -name "*.json" \
       -delete 2>/dev/null; true \
    && find /root/.local/lib -path "*/streamlit/static*" -name "*.map" \
       -delete 2>/dev/null; true

COPY pipeline.py ml.py dashboard.py ./

RUN mkdir -p /app/data /app/model

ENV SE3_DB_PATH=/app/data/se3_cache.duckdb \
    MODEL_DIR=/app/model \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Default: dashboard. Jobs override CMD at deploy time.
CMD streamlit run dashboard.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
