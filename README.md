# ⚡ SE3 Electricity Price Forecast

A production-grade day-ahead electricity price forecasting system for the **SE3 bidding area** (Sweden), built with LightGBM and Streamlit — deployed on Google Cloud Run.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-quantile-green)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)
![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4)

**Live dashboard:** https://se3-dashboard-458800722354.europe-north1.run.app

---

## What it does

- Forecasts SE3 electricity prices **24 hours ahead** with 90% confidence intervals
- Syncs data hourly from ENTSO-E and Open-Meteo
- Retrains the model weekly on fresh data
- Reads data directly from DuckDB — no API layer needed

---

## Dashboard pages

| Page | Description |
|---|---|
| 📈 **Forecast** | Next 24h price forecast with confidence band, hourly table |
| 📊 **Performance** | MAE, RMSE, MAPE, PI coverage, MAE by hour of day |
| 📉 **Backtesting** | Actual vs forecast for any date range, error distribution |
| 🗃️ **Data** | Raw prices, weather, generation with CSV export |

---

## Architecture

```
GCS bucket: se3-cache
├── se3_cache.duckdb      ← all fetched data + saved forecasts
├── models.pkl            ← trained LightGBM models
├── pretrained_models.pkl
└── metrics.json + other artifacts

Cloud Run SERVICE: se3-dashboard   ← Streamlit reads DuckDB via GCS volume mount
Cloud Run JOBS:
  ├── pipeline-job   ← hourly data sync (ENTSO-E + Open-Meteo)
  ├── forecast-job   ← daily forecast generation (10:00 Stockholm)
  └── train-job      ← weekly posttrain (Sunday 02:00 Stockholm)
Cloud Scheduler: 3 cron jobs triggering the above
```

### Data sources

| Source | Data |
|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Day-ahead prices, generation, load, nuclear output, cross-border flows |
| [Open-Meteo](https://open-meteo.com) | Historical + forecast weather (temp, wind 10m/100m, cloud, solar) |

---

## ML model

**LightGBM** with quantile regression (p05, p50, p95) trained on 78 features.

### Feature groups

| Group | Examples |
|---|---|
| Price anchors | EWA same-hour mean, lag 24h/48h/168h, rolling 7d/30d stats |
| Calendar | Fourier harmonics k1–k3 (daily, weekly, annual), hour/month/DOW |
| Weather | windspeed_100m, wind_surprise, heating_degree, cloudcover |
| Generation | wind_gen_lag24, load_lag24, load_residual |
| Nuclear | nuclear_gen_lag24, nuclear_shortfall (outage proxy) |
| Cross-border flows | flow_no1_lag24, flow_se4_lag24, net_pos_roll7d |

### Key design decisions
- **EWA anchor** — exponentially weighted average adapts faster to price regime shifts than a simple rolling mean
- **Fourier k2 harmonics** — captures the double morning+evening peak structure
- **V2 feature neutralization** — weather features orthogonalized against seasonal Fourier basis
- **V3 target neutralization** — LightGBM trains on residuals from a Ridge linear baseline
- **Zero data leakage** — all features shifted ≥ 24h

---

## Project structure

```
se3-forecast/
├── pipeline.py          # Data sync — fetches ENTSO-E + weather into DuckDB
├── ml.py                # ML — feature engineering, train, predict, evaluate
├── dashboard.py         # Streamlit dashboard (4 pages, reads DuckDB directly)
├── Dockerfile           # Single image used for dashboard + all Cloud Run Jobs
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── notebooks/           # Exploratory analysis
```

---

## Quickstart (local)

### 1. Clone and install

```bash
git clone https://github.com/surajsingh108/SE3-electricity-price-forecast.git
cd SE3-electricity-price-forecast
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your ENTSO-E API key — free at https://transparency.entsoe.eu
```

### 3. Sync data and train

```bash
python pipeline.py          # fetch all data (~10 min first run)
python ml.py --pretrain     # full pretrain on all historical data
python ml.py --forecast     # generate first forecast and save to DB
```

### 4. Run the dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501`.

---

## Docker (local test)

```bash
docker build -t se3:test .

# Dashboard
docker run -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  se3:test

# Run a job manually
docker run --rm --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  se3:test python pipeline.py
```

---

## GCP Deployment summary

One image, one bucket, three jobs.

```bash
# 1. Build and push
gcloud builds submit --tag=REGION-docker.pkg.dev/PROJECT/REPO/se3 .

# 2. Deploy dashboard (GCS volume mount)
gcloud run deploy se3-dashboard --image=IMAGE \
  --add-volume=name=gcs-data,type=cloud-storage,bucket=se3-cache \
  --add-volume-mount=volume=gcs-data,mount-path=/app/data \
  --add-volume-mount=volume=gcs-data,mount-path=/app/model \
  --set-env-vars=SE3_DB_PATH=/app/data/se3_cache.duckdb,MODEL_DIR=/app/model

# 3. Deploy jobs (same image, different CMD)
gcloud run jobs deploy pipeline-job --image=IMAGE --command=python --args=pipeline.py
gcloud run jobs deploy forecast-job --image=IMAGE --command=python --args="ml.py,--forecast"
gcloud run jobs deploy train-job    --image=IMAGE --command=python --args="ml.py,--posttrain"
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ENTSOE_API_KEY` | ✅ | ENTSO-E Transparency Platform API key |
| `SE3_DB_PATH` | | Path to DuckDB file (default: `data/se3_cache.duckdb`) |
| `MODEL_DIR` | | Path to model artifacts directory (default: `model`) |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data storage | [DuckDB](https://duckdb.org) — columnar, zero-setup, single file |
| ML model | [LightGBM](https://lightgbm.readthedocs.io) — quantile regression |
| Dashboard | [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com) |
| Container | Docker → GCP Cloud Run |
| Data pipeline | [entsoe-py](https://github.com/EnergieID/entsoe-py) + [Open-Meteo](https://open-meteo.com) |
| Scheduling | GCP Cloud Scheduler + Cloud Run Jobs |
| Storage | GCP Cloud Storage (GCS FUSE volume mount) |

---

## License

MIT
