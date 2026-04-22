# Architecture — Sthlm Electricity Usage

Hourly pipeline that ingests Swedish electricity data, engineers features, trains probabilistic 48-hour forecasts with LightGBM, and serves a Streamlit dashboard. One command starts everything.

---

## Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 15 (Docker, port 5433) |
| Pipeline | Python + APScheduler (hourly cron) |
| ML | LightGBM quantile regression |
| Dashboard | Streamlit + Plotly |
| Data sources | ENTSO-E Transparency API, Open-Meteo, MET Norway |

---

## Data Flow

```
ENTSO-E API  ──► ingest_prices.py  ──► raw_prices
Open-Meteo   ──► ingest_weather.py ──► raw_weather     ──► analyse.py ──► features_hourly
MET Norway   ──► ingest_weather.py ──┘                                ──► features_correlation
ENTSO-E A75  ──► ingest_carbon.py  ──► raw_generation                 ──► features_best_hours
                                                                             │
                                                                       predict_48h()
                                                                             │
                                                                       dashboard_state  ──► Streamlit
```

Each phase is independent and re-runnable. All writes are upserts on natural keys.

---

## Pipeline Phases

### Phase 1 — Ingest
Fetches raw JSON from three APIs, writes to `data/raw/`. No transformation.

### Phase 2 — Parse
Reads JSON, normalises to tabular rows, upserts into raw tables. Deduplication by UNIQUE constraints on `(zone, price_time)`, `(source, city, forecast_time)`, etc.

### Phase 3 (runner.py) — Orchestration
Wraps phases 1–5 in `run_with_retry()`: 3 attempts, 30 s delay, every run logged to `pipeline_runs(run_at, status, sources_ok, sources_total, notes)`.

### Phase 4 — Feature Engineering (`analyse.py`)
For each zone (SE1–SE4):

| Feature | Method |
|---|---|
| `rolling_avg_6h`, `rolling_avg_24h` | `pandas.rolling` on `price_eur_mwh` |
| `price_level` | `'low'` if price < 0.85 × 24 h avg, `'high'` if > 1.15 ×, else `'medium'` |
| `greenness_score` | Low-carbon MW (hydro B12, nuclear B14, wind B18/B19, solar B16) ÷ total MW × 100. SE3 only; proxied for SE1/SE2/SE4. |
| `appliance_signal` | `'run_now'` if low price + green, `'avoid'` if high + dirty, else `'wait'` |
| `combined_score` | `(1 − norm_price) + norm_greenness` — used to rank hours of day |

Ends by calling `populate_dashboard_state()`.

### Phase 5 — Dashboard State (`dashboard_state.py`)
Pre-renders one row per zone into `dashboard_state`. Dashboard reads this single row at request time — no joins, no computation.

Computes:
- Weekly deltas (current vs 7-day average)
- Optimal appliance windows (cheapest hour in look-ahead window)
- Recommendation text + action (`run_now` | `wait`)
- Savings in kr (clamped ≥ 0 to avoid negative hero card)

---

## Machine Learning

### Task
Probabilistic 48-hour forecasting of `price_eur_mwh` and `greenness_score`.

### Algorithm
LightGBM gradient-boosted decision trees with `objective='quantile'` at α ∈ {0.1, 0.5, 0.9}.
Six models total: 2 targets × 3 quantiles.

```
models/
  price_eur_mwh_q01.txt   # P10
  price_eur_mwh_q05.txt   # P50 (median)
  price_eur_mwh_q09.txt   # P90
  greenness_score_q01.txt
  greenness_score_q05.txt
  greenness_score_q09.txt
```

Saved in native LightGBM text format. Loaded lazily at inference time.

### Feature Matrix (15 features)

| Group | Features |
|---|---|
| Price lags | `price_lag_1h`, `price_lag_24h`, `price_lag_168h` |
| Rolling averages | `rolling_avg_6h`, `rolling_avg_24h` |
| Weather (Open-Meteo) | `temperature_c`, `windspeed_ms`, `radiation_wm2` |
| Weather (MET Norway) | `temperature_c_met`, `windspeed_ms_met` |
| Multi-source disagreement | `wind_disagreement = |OM − MET|` |
| ENTSO-E A69 forecast | `a69_wind_mw`, `a69_solar_mw` (SE3 only) |
| Temporal | `hour_of_day`, `day_of_week` |
| Calendar | `is_holiday` (Swedish public holidays via `holidays` package) |

### Hyperparameters

```python
boosting_type    = "gbdt"
learning_rate    = 0.05
num_leaves       = 31
n_estimators     = 500
min_child_samples = 20
subsample        = 0.8
colsample_bytree = 0.8
early_stopping   = 50 rounds on 10 % held-out tail of each fold
```

### Validation — Walk-Forward Cross-Validation
- Minimum training window: 4 weeks (672 h)
- Fold step: 1 week (168 h), expanding window
- Baseline: seasonal-naive — predict same hour one week prior (t − 168 h)
- Metrics: MAE, MAPE, calibration check (P10/P90 coverage)
- SHAP feature importance computed post-training via `evaluate.py`

### Training
```bash
python scripts/train_forecast.py --zone SE3
```
Runs walk-forward CV, prints fold metrics, trains final models on all available data, saves to `models/`.

### Inference — Fallback Chain
```
predict_48h(zone, conn)
  ├── models/*.txt exist?
  │     Yes → LightGBM predict P10/P50/P90
  └── No  → ENTSO-E day-ahead prices from raw_prices
              + heuristic ±15 % price interval
              + historical hour-of-day greenness ± 10 pp
```
Dashboard always gets a 48-row DataFrame regardless of model availability.

### Known Limitations
- Day-ahead prices publish at 13:00 CET — forecast is sparse before then each day
- Greenness only directly measured for SE3; SE1/SE2/SE4 use SE3 as proxy
- Extreme cold-snap demand spikes underrepresented in short training windows
- DST transitions not encoded as features

---

## Database Schema

```sql
-- Raw ingestion targets
raw_prices        (zone, price_time)               UNIQUE (zone, price_time)
raw_weather       (source, city, forecast_time)    UNIQUE (source, city, forecast_time)
raw_generation    (document_type, psr_type, gen_time)

-- Feature tables (analyse.py output)
features_hourly       PK (zone, hour)   -- all engineered features
features_correlation  PK (zone, metric_a, metric_b)
features_best_hours   PK (zone, hour_of_day)

-- Pre-rendered dashboard payload (one row per zone)
dashboard_state   PK zone
  current_price_eur_mwh, price_delta_pct
  greenness_score, greenness_delta_pp, windspeed_ms
  recommendation_text, recommendation_action
  hours_to_wait, optimal_hour_utc, optimal_price_eur_mwh
  savings_kr_per_kwh, recommendation_savings_kr
  forecast_48h      JSONB  -- [{hour, price, price_p10, price_p90, greenness, …}]
  appliance_windows JSONB  -- {dishwasher: {hour_offset, hour_utc, price_ore}, …}
  horizon_hours, computed_at

-- Health log
pipeline_runs (run_at, status, sources_ok, sources_total, notes)
```

Schema is idempotent — all DDL uses `IF NOT EXISTS`. Migrations in `db/migrations/` run automatically on startup via `pipeline/db.py`.

---

## Deployment

### Local (development)
```bash
cp .env.example .env        # add ENTSOE_API_KEY
python start.py             # starts everything
```

`start.py` sequence:
1. `docker-compose up -d` — PostgreSQL 15 on port 5433
2. Poll DB readiness (60 s timeout)
3. Run first full pipeline (phases 1–5)
4. Launch `pipeline/scheduler.py` as a subprocess
5. `streamlit run dashboard/app.py` in foreground (port 8501)
6. `Ctrl-C` terminates scheduler cleanly

### Scheduler
```python
BlockingScheduler(timezone="UTC")
CronTrigger(minute=0)        # fires :00 every hour
max_instances = 1            # prevents pile-up if a run overruns
misfire_grace_time = 300     # allows up to 5-min startup slack
```

### Dashboard read path
`app.py` → `queries.fetch_dashboard_state(zone)` → single `SELECT` on `dashboard_state` → `charts.py` renders Plotly figures. All `@st.cache_data(ttl=3600)` — no re-query within an hour.

### Environment Variables
```
ENTSOE_API_KEY    required   # transparency.entsoe.eu
DB_HOST           localhost
DB_PORT           5432
DB_NAME           sweden_energy
DB_USER           pipeline
DB_PASSWORD       pipeline
```

### Appliance Configuration (dashboard_state.py)
```python
_APPLIANCES = {
    "dishwasher": (24, 1.5),   # look-ahead hours, kWh per cycle
    "laundry":    (24, 2.0),
    "ev":         (12, 10.0),
}
_MIN_WAIT_SAVINGS_KR = 1.0     # minimum saving to recommend waiting
_EUR_TO_SEK          = 11.5    # conversion factor (hardcoded)
```

---

## Error Handling Summary

| Scenario | Behaviour |
|---|---|
| API down | Returns empty DataFrame, logs WARNING, pipeline continues |
| Partial ingest | Per-source exception caught; successful sources still parsed |
| Missing ML models | Falls back to ENTSO-E day-ahead + heuristic intervals |
| Pipeline crash | Retry × 3 (30 s delay); failure row written to `pipeline_runs` |
| Consecutive failures | `alerts.py` notifies after N failures in a row |
| Zero/null forecast prices | Filtered out before appliance window search and hero card |
| Negative savings | Clamped to 0 — `max(raw_saving, 0.0)` |
