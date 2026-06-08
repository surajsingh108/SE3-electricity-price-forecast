# SE3 Electricity Market Forecasting & Analysis

A live forecasting system for the SE3 Swedish electricity market
(Stockholm/Mälardalen price zone). Combines a LangGraph AI agent,
three LightGBM models, and a real-time dashboard built on 8 data sources.

**Live demo:** https://tinyurl.com/se3-forecast

---

## What it does

**Dashboard** — six pages covering spot price forecasts, imbalance price
forecasts, spike detection, model performance, backtesting, raw data
explorer, and a natural language agent.

**Models**
- Spot price regression — day-ahead P05/P50/P95 for next 24h (MAE ~20 EUR/MWh)
- Imbalance price regression — 15-min P05/P50/P95 for next 24h (MAE ~26 EUR/MWh)
- Spike detector — binary classifier P(spike) for each 15-min period (AUC 0.910)

**Agent** — ask anything about SE3 in plain English:
- "Why is SE3 electricity expensive right now?"
- "What will prices look like tomorrow?"
- "Is there a spike risk this evening?"
- "Should I charge my battery now?"

---

## Architecture

```
                         RAW DATA LAYER
                              │
     eSett · ENTSO-E · Open-Meteo · SMHI · ENTSO-E Nuclear
                              │
                         pipeline.py
                        (DuckDB on GCS)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ml.py          ml_imbalance.py   hindcast_imbalance.py
    (spot forecast)   (imbalance + spike)  (historical forecasts)
              │               │
              └───────────────┘
                      │
                 dashboard.py          api.py + agent/
                 (Streamlit)           (FastAPI + LangGraph)
```

```
User question (Streamlit / API)
        │
        ▼
  LangGraph Agent (graph.py)
        │
  ┌─────┼──────┐
  ▼     ▼      ▼
Price  Forecast Weather
Tool   Tool     Tool
        │
  Synthesis Node (Gemini 3.5 Flash)
        │
        ▼
  Structured answer + sources + confidence
```

---

## Tech stack

| Component | Choice |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Gemini 3.5 Flash (Google AI Studio) |
| Spot price data | ENTSO-E Transparency API |
| Imbalance price data | eSett Open Data API |
| Weather (archive) | Open-Meteo archive + historical forecast |
| Weather (live) | SMHI snow1g forecast + Mesan analysis |
| Nuclear availability | ENTSO-E unavailability API |
| Generation forecast | ENTSO-E wind/solar generation forecast |
| Cross-border flows | ENTSO-E physical flows |
| Forecasting models | LightGBM (quantile regression + binary classification) |
| Data layer | DuckDB (GCS-mounted at gs://se3-cache/) |
| API | FastAPI |
| Frontend | Streamlit |
| Deployment | GCP Cloud Run (europe-north1) |

---

## Data sources

| Source | Endpoint | Auth | Resolution | Coverage |
|--------|----------|------|------------|----------|
| eSett imbalance prices | api.opendata.esett.com/EXP14/Prices | None | 15-min | 2021-11 → live |
| ENTSO-E spot prices | web-api.tp.entsoe.eu | API key | Hourly | Historical |
| ENTSO-E cross-border flows | web-api.tp.entsoe.eu | API key | 15-min | Historical |
| ENTSO-E nuclear unavailability | web-api.tp.entsoe.eu | API key | Variable | Historical |
| ENTSO-E generation forecast | web-api.tp.entsoe.eu | API key | 15-min | Historical |
| Open-Meteo archive | archive-api.open-meteo.com | None | Hourly | Historical |
| Open-Meteo hist. forecast | historical-forecast-api.open-meteo.com | None | Hourly | Historical |
| SMHI snow1g forecast | opendata-download-metfcst.smhi.se | None | Hourly | 83h ahead |
| SMHI Mesan analysis | opendata-download-metanalys.smhi.se | None | Hourly | Last 24h |

> **Note:** SMHI deprecated the old `pmp3g` endpoint on 31 March 2026.
> All SMHI calls now use `snow1g` (forecast) and `mesan2g` (analysis).

---

## Models

### Spot price forecaster (`ml.py`)
- LightGBM quantile regression (P05/P50/P95)
- 24-hour ahead forecast at hourly resolution
- Features: weather, calendar Fourier harmonics, cross-border flows,
  generation mix, lagged imbalance prices
- Test MAE: ~20 EUR/MWh

### Imbalance price forecaster (`ml_imbalance.py` — regression)
- LightGBM quantile regression (P05/P50/P95)
- 24-hour ahead at 15-min resolution
- 95 features including regulation spread, direction lags, volatility
- Test MAE: ~26 EUR/MWh, PI coverage: 88%

### Spike detector (`ml_imbalance.py` — classifier)
- Binary LightGBM: P(price > 200 EUR/MWh or < -100 EUR/MWh)
- Trained on 4.5 years of SE3 data (120,434 rows)
- AUC: 0.910, Walk-forward recall: 0.42
- Production threshold: 0.8118 (profit-optimal from full dataset)
---

## Repo structure

```
se3-agent/
├── agent/
│   ├── graph.py              # LangGraph StateGraph
│   ├── state.py              # AgentState dataclass
│   ├── nodes.py              # router, synthesiser nodes
│   └── tools/
│       ├── price.py          # ENTSO-E price fetcher
│       ├── forecast.py       # spot forecast (DuckDB)
│       └── weather.py        # Open-Meteo weather
├── notebooks/
│   ├── se3_imbalance_forecasting.ipynb  # regression model dev
│   ├── se3_spike_detection.ipynb        # spike classifier dev
│   └── se3_normal_regime.ipynb          # regime classifier dev
├── model/                    # trained model pickle files (gitignored)
│   ├── pretrained_models.pkl
│   ├── imbalance_models.pkl
│   └── spike_model.pkl
├── api.py                    # FastAPI — /ask, /retrain endpoints
├── dashboard.py              # Streamlit frontend (6 pages)
├── pipeline.py               # data sync — all 8 sources → DuckDB
├── ml.py                     # spot price LightGBM model
├── ml_imbalance.py           # imbalance regression + spike detector
├── data_sources.py           # fetch functions for all data sources
├── feature_engineering.py    # shared feature engineering (95 features)
├── hindcast_imbalance.py     # generate historical forecasts for backtesting
├── retrain.py                # spot forecast refresh (called by scheduler)
├── retrain_imbalance.py      # full retrain from scratch (local only)
├── supervisord.conf          # runs api (8000) + dashboard (8080)
├── Dockerfile
└── requirements.txt
```

---

## Dashboard pages

| Page | Description |
|------|-------------|
| 📈 Forecast | Spot price P05/P50/P95 next 24h with metric cards |
| ⚡ Imbalance & Spike | Live imbalance price + spike probability + regime |
| 📊 Performance | Model evaluation metrics, MAE by hour, error distribution |
| 📉 Backtesting | Spot price and imbalance price backtesting with date picker |
| 🗃️ Data | Raw data explorer — prices, weather, generation, imbalance |
| 💬 Ask SE3 | Natural language questions via LangGraph + Gemini |

---

## Running locally

```bash
# 1. Clone and set environment variables
cp .env.example .env
# Required:
#   GEMINI_API_KEY      — Google AI Studio key
#   ENTSOE_API_KEY      — ENTSO-E Transparency Platform (free registration)
#   SE3_DB_PATH         — path to DuckDB file (default: data/se3_cache.duckdb)
#   MODEL_DIR           — path to model/ directory (default: model/)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Sync all data sources (first run takes ~5 min for full backfill)
python pipeline.py

# 4. Train models (only needed on first run or after data changes)
python ml.py --train                     # spot price model
python ml_imbalance.py --train-all       # imbalance regression + spike detector

# 5. Generate forecasts
python ml.py --forecast                  # spot price forecast
python ml_imbalance.py --forecast        # imbalance + spike forecast

# 6. Generate hindcast for backtesting (Jun 2026 onwards)
python hindcast_imbalance.py --start 2026-06-01

# 7. Start both services
supervisord -c supervisord.conf
# Dashboard at http://localhost:8080
# API at http://localhost:8000
```

---

## Commands reference

### Data pipeline

```bash
# Sync all data sources (incremental — only fetches new data)
python pipeline.py

# Check what's in the DB
python pipeline.py --status

# Sync + refresh both forecasts (what the scheduler runs)
python retrain.py
```

### Spot price model

```bash
# Train spot price model
python ml.py --train

# Generate spot price forecast (writes to DB)
python ml.py --forecast

# Evaluate model on test set
python ml.py --evaluate
```

### Imbalance + spike models

```bash
# Train both imbalance regression and spike detector
python ml_imbalance.py --train-all

# Train imbalance regression only
python ml_imbalance.py --train-imbalance

# Train spike detector only
python ml_imbalance.py --train-spike

# Generate 24h imbalance + spike forecast (writes to DB)
python ml_imbalance.py --forecast

# Evaluate both models on test set
python ml_imbalance.py --evaluate

# Show next 24h forecast in terminal
python ml_imbalance.py --show
```

### Hindcast (backtesting data)

```bash
# Generate hindcast from a specific date (runs in ~1 min per month)
python hindcast_imbalance.py --start 2026-06-01

# Generate hindcast for the last N days only
python hindcast_imbalance.py --last-days 30

# Overwrite existing hindcast rows (re-run after model retrain)
python hindcast_imbalance.py --start 2026-06-01 --overwrite

# Full history from model training end (takes 20-40 min)
python hindcast_imbalance.py --start 2026-04-08

# Run in background
nohup python hindcast_imbalance.py --start 2026-04-08 \
  > hindcast.log 2>&1 &
tail -f hindcast.log
```

### Full refresh (sync + forecast + hindcast)

```bash
# What the Cloud Scheduler runs 4x daily
python retrain.py

# Full retrain from scratch (local only — takes 15-30 min)
python pipeline.py
python ml.py --train
python ml_imbalance.py --train-all
python ml.py --forecast
python ml_imbalance.py --forecast
python hindcast_imbalance.py --last-days 7
```

---

### Build and deploy

```bash
# Build and push image
IMAGE="gcr.io/$(gcloud config get-value project)/se3-agent:$(date +%Y%m%d-%H%M)"
docker build -t $IMAGE .
docker push $IMAGE

# Deploy to Cloud Run
gcloud run deploy se3-dashboard \
  --image $IMAGE \
  --region europe-north1 \
  --platform managed

# Check deployment status
gcloud run revisions list \
  --service=se3-dashboard \
  --region=europe-north1 \
  --limit=5
```

### View logs

```bash
# Recent errors
gcloud logging read \
  "resource.type=cloud_run_revision \
   AND resource.labels.service_name=se3-dashboard \
   AND severity>=ERROR" \
  --limit=20 --format="value(timestamp,textPayload)"

# All recent logs
gcloud logging read \
  "resource.type=cloud_run_revision \
   AND resource.labels.service_name=se3-dashboard" \
  --limit=50 --format="value(timestamp,textPayload)"
```

### Cloud Scheduler

The forecast refresh job runs 4x daily at times aligned with SE3 market
activity and eSett settlement windows (6-12h lag):

```
Schedule: 0 6,12,18,23 * * *  (06:00, 12:00, 18:00, 23:00 CET)
Job:      forecast-job (Cloud Run Job, europe-north1)
```

Each run: syncs all data sources → refreshes spot forecast →
refreshes imbalance forecast → backfills last 7 days of hindcast.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google AI Studio API key |
| `ENTSOE_API_KEY` | Yes | ENTSO-E Transparency Platform key |
| `SE3_DB_PATH` | No | DuckDB path (default: `data/se3_cache.duckdb`) |
| `MODEL_DIR` | No | Model directory (default: `model/`) |


Built by [Suraj Singh](https://github.com/surajsingh108)
