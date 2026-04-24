# SE3 Electricity Price Forecast Dashboard — Architecture Plan

## Overview
A fully free-tier electricity price forecast dashboard for the SE3 bidding
area (Sweden), built on GCP always-free services with a Streamlit frontend
and a FastAPI backend.

---

## Current Status (local development — complete)

| File | Role | Status |
|---|---|---|
| `pipeline.py` | Data sync, DuckDB cache, scheduler | ✅ Done |
| `ml.py` | Feature engineering, train, predict, evaluate | ✅ Done |
| `api.py` | FastAPI backend — 6 endpoints | ✅ Done |
| `dashboard.py` | Streamlit UI — 4 pages | ✅ Done |
| `data/se3_cache.duckdb` | Local data cache | ✅ Populated |
| `model/` | Trained model artifacts | ✅ Saved |

---

## GCP Services Used (all always-free)

| Service | Free Limit | Role |
|---|---|---|
| Cloud Run | 2M requests/month · 360k GB-seconds | Hosts API + dashboard |
| Cloud Scheduler | 3 jobs/month | Triggers daily retrain + hourly sync |
| Cloud Storage | 5 GB | Model artifacts (model/*.pkl, config.json) |
| BigQuery | 10 GB storage · 1 TB queries/month | Optional upgrade from DuckDB |
| Artifact Registry | 0.5 GB | Docker image |

> New GCP accounts get $300 free credit for 90 days as a buffer.

---

## Target Architecture (GCP deployment)

```
Cloud Scheduler (3 cron jobs)
    │
    ├── hourly-sync  (Cloud Run Job)
    │       python pipeline.py
    │       → incremental fetch of prices, weather, nuclear, flows
    │       → writes to DuckDB on Cloud Storage (or BigQuery)
    │
    └── daily-retrain  (Cloud Run Job)
            python ml.py --train
            → trains model, saves artifacts to Cloud Storage
            → calls POST /retrain on API to hot-reload

Cloud Storage
    ├── data/se3_cache.duckdb     ← shared between jobs and API
    └── model/
        ├── models.pkl
        ├── linear_baseline.pkl
        ├── neutralizers.pkl
        ├── config.json
        └── metrics.json

Cloud Run — API (api.py via uvicorn)
    ├── GET /health
    ├── GET /forecast          → runs ml.predict(), caches 5 min
    ├── GET /metrics           → reads model/metrics.json
    ├── GET /history           → actuals vs saved forecasts
    ├── GET /prices
    ├── GET /weather
    ├── GET /generation
    └── POST /retrain          → protected by X-Secret header

Cloud Run — Dashboard (dashboard.py via streamlit)
    ├── 📈 Forecast            → calls /forecast
    ├── 📊 Performance         → calls /metrics
    ├── 🔍 Backtesting         → calls /history with date picker
    └── 🗄️  Data explorer       → calls /prices, /weather, /generation

Artifact Registry
    └── se3-forecast:latest    ← Docker image containing all four modules
```

---

## Data Sources

| Source | What | Auth |
|---|---|---|
| ENTSO-E Transparency Platform | Prices, generation, load, nuclear, cross-border flows | Free account — API key in `.env` |
| Open-Meteo archive | Historical weather (temp, wind 10m/100m, cloud, solar) | None — no key needed |
| Open-Meteo forecast | 48h weather forecast for live prediction | None — no key needed |
| `holidays` Python library | Swedish public holidays | Offline |

---

## ML Model — Summary of Learnings

### Features (78 total, zero data leakage)

**Price anchors**
- EWA same-hour anchor (weights: 35/25/17/11/6/4/2%) — adapts faster than 7d mean
- 3d mean, EWA deviation, rolling 168h/720h stats, regime×vol

**Calendar**
- Cyclical sin/cos encodings for hour, day-of-week, month
- Fourier harmonics k1+k2+k3 daily, weekly, annual
- k2 captures the double morning+evening peak structure
- `lag24_x_hour_sin/cos` — fixes midnight anchor problem

**Weather (feature-neutralized against seasonal Fourier basis)**
- windspeed_100m (hub height), wind_surprise, wind_7d_mean
- temperature, heating_degree (non-linear: clip at 15°C)
- cloudcover, cloud×solar interaction

**Generation (all lagged ≥ 24h)**
- wind_gen_lag24, load_lag24, wind_load_ratio, load_residual
- Nuclear: nuclear_gen_lag24, nuclear_shortfall (outage proxy), nuclear_x_peak
- Flows: flow_no1_lag24, flow_se4_lag24, net_pos_roll7d (NO1 was top-4 SHAP)

### Model pipeline
1. **V2 feature neutralization** — weather/gen features orthogonalized vs Fourier time basis
2. **V3 target neutralization** — Ridge linear baseline absorbs trivial anchor+Fourier component
3. **LightGBM** trained on residuals only (what the linear model gets wrong)
4. **Early stopping** — 2000 tree ceiling, stops ~400-600 trees in practice
5. **Three quantile models** — p05, p50, p95

### Key lessons
- `price_roll_24h_mean` caused over-smoothing (35 EUR/MWh SHAP) — removed
- `price_lag_1h` is data leakage for day-ahead forecast — removed
- Raw `nuclear_gen_mw` and raw `flow_*_mw` are same-hour leakage — use lags only
- `query_generation` gives actual nuclear output; `query_installed_generation_capacity` returns static values
- Fourier k2 critical for double-peak — single sin/cos (k1 only) misses morning+evening structure
- EWA anchor updates faster after regime shifts than simple 7d mean
- Target neutralization (V3) works best when combined with EWA (anchor is real signal, not noise)

---

## Project File Structure

```
se3-forecast/
├── pipeline.py          # Module 1 — data pipeline
├── ml.py                # Module 2 — ML train/predict/evaluate
├── api.py               # Module 3 — FastAPI backend
├── dashboard.py         # Module 4 — Streamlit dashboard
├── .env                 # ENTSOE_API_KEY, API_URL, RETRAIN_SECRET
├── data/
│   └── se3_cache.duckdb # DuckDB local cache (all fetched data)
├── model/
│   ├── models.pkl
│   ├── linear_baseline.pkl
│   ├── neutralizers.pkl
│   ├── config.json
│   └── metrics.json
├── Dockerfile           # ← TODO (GCP deployment)
└── cloudbuild.yaml      # ← TODO (GCP deployment)
```

---

## Local Usage

```bash
# One-time: populate cache + train model
python pipeline.py
python ml.py --train

# Daily development
python pipeline.py          # incremental sync (fast)
python ml.py --forecast     # print next 24h to terminal

# Run the full stack
python api.py               # terminal 1 — keeps running
streamlit run dashboard.py  # terminal 2 — opens browser

# Keep data fresh automatically (runs forever)
python pipeline.py --schedule
```

---

## GCP Deployment Checklist (next step)

- [ ] Create GCP account (free)
- [ ] Enable billing (required, stays within free tier)
- [ ] Install `gcloud` CLI
- [ ] Create GCP project: `gcloud projects create se3-forecast`
- [ ] Enable APIs: Cloud Run, Cloud Scheduler, Cloud Storage, Artifact Registry
- [ ] Write `Dockerfile` combining api.py + dashboard.py
- [ ] Build and push image to Artifact Registry
- [ ] Deploy API to Cloud Run (scale to zero — no idle cost)
- [ ] Deploy dashboard to Cloud Run
- [ ] Mount Cloud Storage bucket for DuckDB file sharing
- [ ] Create Cloud Scheduler jobs: hourly sync + daily retrain
- [ ] Set env vars in Cloud Run: `ENTSOE_API_KEY`, `RETRAIN_SECRET`, `API_URL`

---

## Next Steps (remaining work)

- [ ] **Dockerfile** — package all four modules into one image
- [ ] **GCP deployment** — Cloud Run + Cloud Scheduler
- [ ] **Optuna tuning** — likely 5–10% further MAE reduction
- [ ] **Alert system** — email/Slack when rolling MAE drifts above threshold
- [ ] **SHAP explorer page** in dashboard — already stubbed in architecture
- [ ] **Extend training data** back to 2018 for better low-price regime coverage
