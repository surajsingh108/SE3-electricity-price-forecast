FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && find /root/.local -name "*.pyc" -delete \
    && find /root/.local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

COPY pipeline.py ml.py api.py dashboard.py ./
COPY supervisord.conf /etc/supervisor/conf.d/se3.conf

RUN mkdir -p /app/data /app/model

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8080
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/se3.conf"]
