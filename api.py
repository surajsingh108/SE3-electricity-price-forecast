"""
api.py — SE3 forecast backend (Module 3)

FastAPI app exposing forecast, metrics, history and raw data.
Sits between the ML module and the Streamlit dashboard.

Endpoints
---------
  GET /health                          → service health check
  GET /forecast                        → next 24h price forecast
  GET /metrics                         → latest model performance metrics
  GET /history?from_date=&to_date=     → actuals vs saved forecasts
  GET /prices?from_date=&to_date=      → raw hourly prices
  GET /weather?from_date=&to_date=     → raw hourly weather
  POST /retrain                        → trigger model retrain (protected)

Usage
-----
  uvicorn api:app --reload             # dev
  python api.py                        # prod (via uvicorn programmatically)
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("api")

# ── Config ────────────────────────────────────────────────────────────────────

ENTSO_E_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
RETRAIN_SECRET  = os.environ.get("RETRAIN_SECRET", "change-me")  # protect /retrain
MODEL_DIR       = Path(os.environ.get("MODEL_DIR", "model"))
CACHE_TTL_SEC   = int(os.environ.get("FORECAST_CACHE_TTL", "300"))  # 5 min

# ── App ───────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load artifacts. Shutdown: log."""
    log.info("API starting up...")
    yield
    log.info("API shutting down")

app = FastAPI(
    title="SE3 Price Forecast API",
    version="1.0.0",
    description="Day-ahead electricity price forecasting for SE3 (Sweden)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Shared state (loaded once at startup) ─────────────────────────────────────

_artifacts: dict | None = None
_last_forecast: dict | None = None          # cached forecast response
_last_forecast_ts: datetime | None = None   # when it was generated


def get_artifacts() -> dict:
    """Load ML artifacts once, reuse on every request."""
    global _artifacts
    if _artifacts is None:
        if not (MODEL_DIR / "models.pkl").exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run: python ml.py --train"
            )
        from ml import load_artifacts
        _artifacts = load_artifacts(MODEL_DIR)
        log.info("Artifacts loaded (%d features)", len(_artifacts["feature_cols"]))
    return _artifacts


# Pre-load at import time so first request is instant
try:
    get_artifacts()
except Exception as _e:
    log.warning("Model not pre-loaded: %s — will load on first /forecast request", _e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_range(from_date: str, to_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse date strings to tz-aware Timestamps."""
    try:
        start = pd.Timestamp(from_date, tz="Europe/Stockholm")
        end   = pd.Timestamp(to_date,   tz="Europe/Stockholm") + pd.Timedelta(days=1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    return start, end


def _clean(obj):
    """Recursively replace NaN / inf / -inf with None so json.dumps never crashes."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-serialisable dicts."""
    df = df.copy()
    df.index = df.index.astype(str)
    records = df.reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records")
    return _clean(records)


def _load_data_for_predict() -> dict:
    """Load all raw data needed for prediction from the pipeline cache."""
    from pipeline import (
        sync_all, fetch_weather_forecast,
        get_conn, _load,
    )
    import holidays

    start = pd.Timestamp("2020-01-01", tz="Europe/Stockholm")
    end   = pd.Timestamp.now("Europe/Stockholm").floor("h")

    conn    = get_conn()
    prices  = _load(conn, "prices",      start, end)
    weather = _load(conn, "weather",     start, end)
    gen     = _load(conn, "generation",  start, end)

    nuc_df  = _load(conn, "nuclear_gen", start, end)
    nuclear = nuc_df["nuclear_gen_mw"] if "nuclear_gen_mw" in nuc_df.columns \
              else pd.Series(dtype=float, name="nuclear_gen_mw")

    # Load flows
    from pipeline import BORDERS
    flow_series = {}
    for name in BORDERS:
        df_b = conn.execute(
            "SELECT timestamp, flow_mw FROM cross_border_flows "
            "WHERE border = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            [name, start, end],
        ).df()
        if not df_b.empty:
            df_b["timestamp"] = pd.to_datetime(df_b["timestamp"]).dt.tz_convert("Europe/Stockholm")
            s = df_b.set_index("timestamp")["flow_mw"]
            s.name = f"flow_{name.lower()}_mw"
            flow_series[name] = s

    conn.close()

    flows_df = pd.DataFrame()
    if flow_series:
        flows_df = pd.concat(flow_series.values(), axis=1)
        flows_df.columns = [f"flow_{n.lower()}_mw" for n in flow_series]
        flows_df["net_position_mw"] = flows_df.sum(axis=1)
        flows_df = flows_df.ffill(limit=3)

    weather_fc = fetch_weather_forecast()

    return {
        "prices":      prices,
        "weather":     weather,
        "gen":         gen,
        "nuclear_gen": nuclear,
        "flows_df":    flows_df,
        "weather_fc":  weather_fc,
    }


# ── Response models ───────────────────────────────────────────────────────────

class HourlyForecast(BaseModel):
    timestamp: str
    p05: float
    p50: float
    p95: float


class ForecastResponse(BaseModel):
    generated_at: str
    forecast_from: str
    forecast_to: str
    currency: str = "EUR/MWh"
    hours: list[HourlyForecast]


class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    mape: float
    coverage_q5_q95: float
    night_mae: float
    peak_mae: float
    spike_mae: Optional[float]
    n_spikes: int
    test_from: str
    test_to: str
    mae_by_hour: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cache_db: str
    uptime_seconds: float


# ── Startup ───────────────────────────────────────────────────────────────────

_start_time = datetime.now()



# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    from pipeline import DB_PATH
    return HealthResponse(
        status       = "ok",
        model_loaded = _artifacts is not None,
        cache_db     = DB_PATH,
        uptime_seconds = (datetime.now() - _start_time).total_seconds(),
    )


@app.get("/forecast", response_model=ForecastResponse)
def forecast(force_refresh: bool = Query(False)):
    """
    Return the next 24h price forecast for SE3.

    Cached for FORECAST_CACHE_TTL seconds (default 5 min) to avoid
    re-running prediction on every dashboard refresh.
    Use ?force_refresh=true to bypass the cache.
    """
    global _last_forecast, _last_forecast_ts

    # Return cached response if fresh enough
    if (
        not force_refresh
        and _last_forecast is not None
        and _last_forecast_ts is not None
        and (datetime.now() - _last_forecast_ts).total_seconds() < CACHE_TTL_SEC
    ):
        log.info("Returning cached forecast (age %.0fs)",
                 (datetime.now() - _last_forecast_ts).total_seconds())
        return _last_forecast

    artifacts = get_artifacts()
    from ml import predict

    try:
        data       = _load_data_for_predict()
        fc_df      = predict(
            artifacts,
            prices      = data["prices"],
            weather_hist= data["weather"],
            weather_fc  = data["weather_fc"],
            gen         = data["gen"],
            nuclear_gen = data.get("nuclear_gen"),
            flows_df    = data.get("flows_df"),
        )
    except Exception as e:
        log.error("Forecast failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    # Save forecast to DB for backtesting
    try:
        from pipeline import save_forecast
        save_df = fc_df.rename(columns={"p05": "p05", "p50": "p50", "p95": "p95"})
        save_forecast(save_df)
    except Exception as e:
        log.warning("Could not save forecast to DB: %s", e)

    now      = datetime.now(tz=pd.Timestamp.now("Europe/Stockholm").tzinfo)
    response = ForecastResponse(
        generated_at  = now.isoformat(),
        forecast_from = str(fc_df.index[0]),
        forecast_to   = str(fc_df.index[-1]),
        hours         = [
            HourlyForecast(
                timestamp = str(ts),
                p05       = round(float(row.p05), 2),
                p50       = round(float(row.p50), 2),
                p95       = round(float(row.p95), 2),
            )
            for ts, row in fc_df.iterrows()
        ],
    )

    # Cache it
    _last_forecast    = response
    _last_forecast_ts = datetime.now()
    log.info("Forecast generated: %s → %s", fc_df.index[0], fc_df.index[-1])
    return response


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """
    Return the latest model performance metrics from the saved config.
    Re-reads the file on every call so metrics update after a retrain.
    """
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run: python ml.py --train"
        )
    import json
    with open(metrics_path) as f:
        m = json.load(f)
    return MetricsResponse(**m)


@app.get("/history")
def history(
    from_date: str = Query(
        default=(date.today() - timedelta(days=30)).isoformat(),
        description="Start date YYYY-MM-DD",
    ),
    to_date: str = Query(
        default=date.today().isoformat(),
        description="End date YYYY-MM-DD",
    ),
):
    """
    Return hourly actuals alongside the most recent saved forecast for each hour.
    Used by the backtesting page.
    """
    start, end = _ts_range(from_date, to_date)

    from pipeline import get_conn, _load, load_forecasts

    conn    = get_conn()
    actuals = _load(conn, "prices", start, end)
    conn.close()

    try:
        forecasts = load_forecasts(start, end)
    except Exception as e:
        log.warning("Could not load forecasts: %s", e)
        forecasts = pd.DataFrame()

    if actuals.empty:
        raise HTTPException(status_code=404, detail="No price data for this range.")

    # For each hour pick the most recently generated forecast
    if not forecasts.empty:
        # Keep only the latest generated_at per timestamp
        latest = (
            forecasts.sort_values("generated_at")
            .groupby("timestamp")
            .last()
            .reset_index()
            .set_index("timestamp")
        )
        merged = actuals.join(latest[["p05","p50","p95"]], how="left")
    else:
        merged = actuals.copy()
        merged[["p05","p50","p95"]] = np.nan

    merged = merged.rename(columns={"price_eur_mwh": "actual"})

    # Compute per-hour errors where we have both
    has_fc = merged["p50"].notna()
    merged.loc[has_fc, "error"]    = (merged.loc[has_fc, "p50"] - merged.loc[has_fc, "actual"]).round(2)
    merged.loc[has_fc, "abs_error"]= merged.loc[has_fc, "error"].abs().round(2)

    return _clean({
        "from":    from_date,
        "to":      to_date,
        "n_hours": int(len(merged)),
        "mae":     round(float(merged["abs_error"].mean()), 2) if has_fc.any() else None,
        "records": _df_to_records(merged.round(2)),
    })


@app.get("/prices")
def prices(
    from_date: str = Query(
        default=(date.today() - timedelta(days=7)).isoformat()
    ),
    to_date: str = Query(
        default=date.today().isoformat()
    ),
):
    """Raw hourly SE3 day-ahead prices."""
    start, end = _ts_range(from_date, to_date)
    from pipeline import get_conn, _load
    conn = get_conn()
    df   = _load(conn, "prices", start, end)
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="No price data for this range.")
    return _clean({
        "from":    from_date,
        "to":      to_date,
        "n_hours": len(df),
        "mean_eur_mwh": round(float(df["price_eur_mwh"].mean()), 2),
        "records": _df_to_records(df.round(2)),
    })


@app.get("/weather")
def weather(
    from_date: str = Query(
        default=(date.today() - timedelta(days=7)).isoformat()
    ),
    to_date: str = Query(
        default=date.today().isoformat()
    ),
):
    """Raw hourly weather (temperature, wind, solar, cloud cover)."""
    start, end = _ts_range(from_date, to_date)
    from pipeline import get_conn, _load
    conn = get_conn()
    df   = _load(conn, "weather", start, end)
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="No weather data for this range.")
    return _clean({
        "from":    from_date,
        "to":      to_date,
        "n_hours": len(df),
        "records": _df_to_records(df.round(2)),
    })


@app.get("/generation")
def generation(
    from_date: str = Query(
        default=(date.today() - timedelta(days=7)).isoformat()
    ),
    to_date: str = Query(
        default=date.today().isoformat()
    ),
):
    """Raw hourly generation data (wind, load, nuclear)."""
    start, end = _ts_range(from_date, to_date)
    from pipeline import get_conn, _load
    conn = get_conn()
    gen_df  = _load(conn, "generation",  start, end)
    nuc_df  = _load(conn, "nuclear_gen", start, end)
    conn.close()

    if gen_df.empty:
        raise HTTPException(status_code=404, detail="No generation data for this range.")

    if not nuc_df.empty:
        gen_df = gen_df.join(nuc_df, how="left")

    return _clean({
        "from":    from_date,
        "to":      to_date,
        "n_hours": len(gen_df),
        "records": _df_to_records(gen_df.round(1)),
    })


@app.post("/retrain")
def retrain(x_secret: str = Header(..., alias="X-Secret")):
    """
    Trigger a model retrain. Protected by X-Secret header.
    Set RETRAIN_SECRET env var to a strong secret.

    Example:
      curl -X POST http://localhost:8000/retrain -H "X-Secret: your-secret"
    """
    if x_secret != RETRAIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret.")

    global _artifacts, _last_forecast, _last_forecast_ts

    log.info("Retrain triggered via API")
    try:
        from pipeline import sync_all
        from ml import train, evaluate, save_artifacts
        import json

        data      = sync_all(ENTSO_E_API_KEY)
        artifacts = train(data)
        metrics   = evaluate(artifacts)

        # Save metrics alongside model artifacts
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        save_artifacts(artifacts, MODEL_DIR)

        # Reset in-memory caches so next request uses the new model
        _artifacts        = None
        _last_forecast    = None
        _last_forecast_ts = None

        log.info("Retrain complete — MAE=%.2f", metrics["mae"])
        return {
            "status":  "ok",
            "metrics": {k: v for k, v in metrics.items() if k != "mae_by_hour"},
        }
    except Exception as e:
        log.error("Retrain failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = int(os.environ.get("PORT", 8000)),
        reload  = os.environ.get("ENV", "prod") == "dev",
        workers = 1,   # keep to 1 — LightGBM is not fork-safe
    )
