FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY pipeline.py ml.py api.py dashboard.py ./
COPY supervisord.conf /etc/supervisor/conf.d/se3.conf

# ── Create directories that will be overridden by GCS mount in Cloud Run ──────
RUN mkdir -p /app/data /app/model

# ── Cloud Run exposes one port — streamlit on 8080, api on 8000 (internal) ────
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/se3.conf"]
