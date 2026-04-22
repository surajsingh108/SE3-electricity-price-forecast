# SE3 Electricity Price Forecast Dashboard — Architecture Plan

## Overview
A fully free-tier electricity price forecast dashboard for the SE3 bidding area (Sweden),
built on GCP always-free services with a Streamlit frontend.

---

## GCP Services Used (all always-free)

| Service | Free Limit | Role |
|---|---|---|
| Cloud Scheduler | 3 jobs/month | Triggers ingestion and retraining |
| Cloud Functions | 2M invocations/month | Ingestion + retrain jobs |
| BigQuery | 10 GB storage · 1 TB queries/month | Time-series data store |
| Cloud Storage | 5 GB | Model pickle + feature config |
| Cloud Run | 2M requests/month · 360k GB-seconds | Streamlit app hosting |
| Artifact Registry | 0.5 GB | Docker image for app |

> New GCP accounts also get $300 free credit for 90 days as a buffer.

---

## Pipeline Architecture

```
Cloud Scheduler (3 cron jobs)
    │
    ├── ingest-entso-e (Cloud Function)
    │       ENTSO-E Transparency API (free account)
    │       → generation, load, day-ahead prices, cross-border flows
    │       → writes to BigQuery
    │
    ├── ingest-weather (Cloud Function)
    │       Open-Meteo API (no key, no cost)
    │       → temperature, wind speed, solar radiation over SE3 region
    │       → writes to BigQuery
    │
    └── retrain-model (Cloud Function, daily)
            reads features from BigQuery
            trains LightGBM model
            saves model.pkl → Cloud Storage

BigQuery
    ├── prices table       (hourly spot prices SE3)
    ├── features table     (engineered feature matrix)
    ├── forecasts table    (model output, written after each retrain)
    └── actuals table      (for rolling backtesting)

Cloud Storage
    └── model.pkl + feature_config.json

Cloud Run (Streamlit app, scales to zero)
    ├── queries BigQuery for latest data and forecasts
    ├── loads model from Cloud Storage
    └── serves public URL
```

---

## Data Sources

| Source | What | Auth |
|---|---|---|
| ENTSO-E Transparency Platform | Generation, load, day-ahead prices, cross-border flows | Free account at transparency.entsoe.eu |
| Open-Meteo | Temperature, wind, solar radiation | None — no key needed |
| `holidays` Python library | Swedish public holidays | Offline, no API |

---

## Feature Engineering

**Lag features**
- Spot price t-1h, t-24h, t-168h (same hour last week)
- Rolling mean and std over 24h, 72h, 168h windows

**Calendar features**
- Hour of day, day of week, month
- Swedish public holiday flag
- Peak hours flag (07–09, 17–20)
- Season (winter/spring/summer/autumn)

**External signals**
- Hydro reservoir levels (ENTSO-E)
- Wind + solar generation (ENTSO-E)
- Nuclear planned/unplanned outages (ENTSO-E)
- Temperature (SE3 region centroid, Open-Meteo)
- Cross-border flow capacity: SE3↔SE4, SE3↔NO1, SE3↔DK1

---

## Model Stack

**Primary model:** LightGBM with quantile regression
- Fast training, handles tabular features well
- Quantile outputs (q10, q50, q90) give prediction intervals
- SHAP values for free → explainability in dashboard

**Tuning:** Optuna (free, open source)

**Tracking:** MLflow local instance (free, self-hosted inside Cloud Run or locally)

**Optional upgrade:** Temporal Fusion Transformer via Darts
- Better seasonal pattern learning
- Worth adding once LightGBM baseline is stable

---

## Streamlit Dashboard Pages

1. **Forecast** — 24h and 7-day ahead price chart with confidence band (q10–q90)
2. **Feature importance** — SHAP values, top drivers of today's forecast
3. **Model performance** — rolling MAE, MAPE, RMSE vs actuals
4. **Actual vs forecast** — backtesting view, last 30/90 days
5. **Data explorer** — raw prices, generation mix, weather signals

---

## Project File Structure (to scaffold)

```
se3-forecast/
├── ingestion/
│   ├── ingest_entso_e/
│   │   ├── main.py          # Cloud Function entrypoint
│   │   └── requirements.txt
│   ├── ingest_weather/
│   │   ├── main.py
│   │   └── requirements.txt
│   └── retrain_model/
│       ├── main.py
│       └── requirements.txt
├── features/
│   └── feature_pipeline.py  # shared feature engineering logic
├── model/
│   ├── train.py
│   └── predict.py
├── app/
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── infra/
│   └── setup.sh             # gcloud commands to provision everything
├── bigquery/
│   └── schema.sql           # table definitions
└── README.md
```

---

## Setup Checklist

- [ ] Create GCP account (free)
- [ ] Enable billing (required, but stays within free tier)
- [ ] Install `gcloud` CLI locally
- [ ] Register at transparency.entsoe.eu (free ENTSO-E account)
- [ ] Create GCP project
- [ ] Enable APIs: BigQuery, Cloud Functions, Cloud Run, Cloud Scheduler, Artifact Registry
- [ ] Run `infra/setup.sh` to provision all resources
- [ ] Deploy Cloud Functions
- [ ] Build and push Docker image to Artifact Registry
- [ ] Deploy Streamlit app to Cloud Run

---

## Next Steps
When resuming: ask Claude to generate the full project scaffold including
Dockerfiles, Cloud Function code, BigQuery schema SQL, and Streamlit app skeleton.
