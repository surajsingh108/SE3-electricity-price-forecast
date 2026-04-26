# ⚡ SE3 Electricity Price Forecast Dashboard

A production-grade day-ahead electricity price forecasting system for the **SE3 bidding area** (Sweden), built with LightGBM, FastAPI, and Streamlit — deployed on GCP free tier.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-quantile-green)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)
![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4)

---

## What it does

- Forecasts SE3 electricity prices **24 hours ahead** with 90% confidence intervals
- Retrains automatically every day on fresh data
- Syncs data hourly from ENTSO-E and Open-Meteo
- Serves forecasts via a REST API
- Displays results in an interactive Streamlit dashboard

---

## Dashboard pages

| Page | Description |
|---|---|
| 📈 **Forecast** | Next 24h price forecast with confidence band, hourly table |
| 📊 **Performance** | MAE, RMSE, MAPE, PI coverage, MAE by hour of day |
| 🔍 **Backtesting** | Actual vs forecast for any date range, error distribution |
| 🗄️ **Data explorer** | Raw prices, weather, generation data with CSV export |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Cloud Run (GCP)                   │
│                                                     │
│   supervisord                                       │
│   ├── uvicorn api:app        :8000  (internal)      │
│   └── streamlit dashboard    :8080  (public)        │
└──────────────────┬──────────────────────────────────┘
                   │
          Cloud Storage (GCP)
          ├── data/se3_cache.duckdb   ← all fetched data
          └── model/                  ← trained artifacts

Cloud Scheduler (GCP)
├── Hourly  → pipeline.py   (incremental data sync)
└── Daily   → POST /retrain (model retrain at 03:00)
```

### Data sources

| Source | Data | Auth |
|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Day-ahead prices, generation, load, nuclear output, cross-border flows | Free account |
| [Open-Meteo](https://open-meteo.com) | Historical + forecast weather (temp, wind 10m/100m, cloud, solar) | None |

---

## ML model

**LightGBM** with quantile regression (p05, p50, p95) trained on 78 features.

### Feature groups

| Group | Examples |
|---|---|
| Price anchors | EWA same-hour mean, lag 24h/48h/168h, rolling 7d/30d stats |
| Calendar | Fourier harmonics k1+k2+k3 (daily, weekly, annual), cyclical hour/month/DOW |
| Weather | windspeed_100m, wind_surprise, heating_degree, cloudcover |
| Generation | wind_gen_lag24, load_lag24, load_residual |
| Nuclear | nuclear_gen_lag24, nuclear_shortfall (outage proxy) |
| Cross-border flows | flow_no1_lag24, flow_se4_lag24, net_pos_roll7d |

### Key design decisions
- **EWA anchor** instead of simple 7-day mean — adapts faster to price regime shifts
- **Fourier k2 harmonics** — captures double morning+evening peak structure
- **V2 feature neutralization** — weather features orthogonalized against seasonal Fourier basis
- **V3 target neutralization** — LightGBM trains on residuals from a Ridge linear baseline
- **Zero data leakage** — all features shifted ≥ 24h; no same-hour nuclear or flow values

---

## Project structure

```
se3-forecast/
├── pipeline.py          # Module 1 — data sync, DuckDB cache, scheduler
├── ml.py                # Module 2 — feature engineering, train, predict, evaluate
├── api.py               # Module 3 — FastAPI backend (6 endpoints)
├── dashboard.py         # Module 4 — Streamlit dashboard (4 pages)
├── Dockerfile           # Combined API + dashboard container
├── supervisord.conf     # Process supervisor config
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore
├── .dockerignore
├── DEPLOY.md            # Step-by-step GCP deployment guide
└── se3_forecast_architecture.md  # Full architecture reference
```

---

## Quickstart (local)

### 1. Clone and install

```bash
git clone https://github.com/YOUR-USERNAME/se3-forecast.git
cd se3-forecast
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ENTSO-E API key
# Get a free key at: https://transparency.entsoe.eu/usrm/user/myAccountSettings
```

### 3. Fetch data and train model

```bash
python pipeline.py          # fetch all data (~10 min first run)
python ml.py --train        # train model and save artifacts
```

### 4. Run the full stack

```bash
# Terminal 1
python api.py

# Terminal 2
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Docker (local test)

```bash
docker build -t se3-forecast .

docker run -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  se3-forecast
```

Open `http://localhost:8080`.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/forecast` | Next 24h price forecast (cached 5 min) |
| GET | `/metrics` | Latest model performance metrics |
| GET | `/history` | Actuals vs saved forecasts for a date range |
| GET | `/prices` | Raw hourly SE3 prices |
| GET | `/weather` | Raw hourly weather data |
| GET | `/generation` | Raw hourly generation data |
| POST | `/retrain` | Trigger model retrain (requires `X-Secret` header) |

Interactive API docs at `http://localhost:8000/docs`.

---

## GCP Deployment

See **[DEPLOY.md](DEPLOY.md)** for the full step-by-step guide.

**Summary:**
```bash
# Build and push image
gcloud builds submit --tag $IMAGE .

# Deploy to Cloud Run (free tier)
gcloud run deploy se3-forecast --image=$IMAGE --region=europe-north1

# Schedule daily retrain
gcloud scheduler jobs create http se3-daily-retrain \
  --schedule="0 3 * * *" --uri="$SERVICE_URL/retrain" \
  --http-method=POST --headers="X-Secret=$RETRAIN_SECRET"
```

---

## GCP Free Tier Usage

| Service | Limit | This project |
|---|---|---|
| Cloud Run | 2M requests/month | ~3k/month |
| Cloud Storage | 5 GB | ~500 MB |
| Cloud Scheduler | 3 jobs | 2 jobs |
| Artifact Registry | 0.5 GB | ~500 MB |
| **Cost** | | **$0/month** |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ENTSOE_API_KEY` | ✅ | ENTSO-E Transparency Platform API key |
| `RETRAIN_SECRET` | ✅ | Secret for `POST /retrain` endpoint |
| `API_URL` | ✅ | URL of the FastAPI backend (default: `http://localhost:8000`) |
| `SE3_DB_PATH` | | Path to DuckDB file (default: `data/se3_cache.duckdb`) |
| `MODEL_DIR` | | Path to model artifacts (default: `model/`) |
| `FORECAST_CACHE_TTL` | | Forecast cache TTL in seconds (default: `300`) |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data storage | [DuckDB](https://duckdb.org) — columnar, zero-setup, single file |
| ML model | [LightGBM](https://lightgbm.readthedocs.io) — quantile regression |
| Feature importance | [SHAP](https://shap.readthedocs.io) |
| API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| Dashboard | [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com) |
| Process management | [Supervisor](http://supervisord.org) |
| Container | Docker → GCP Cloud Run |
| Data pipeline | [entsoe-py](https://github.com/EnergieID/entsoe-py) + [Open-Meteo](https://open-meteo.com) |
| Scheduling | GCP Cloud Scheduler |

---

## License

MIT
