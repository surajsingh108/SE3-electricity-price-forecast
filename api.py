"""
api.py — SE3 REST API

Provides endpoints for price data, forecast data, and the LangGraph agent.
Reads directly from DuckDB (no API layer).
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import duckdb
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from agent.graph import agent_graph

load_dotenv()

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

app = FastAPI(title="SE3 API")


class AskRequest(BaseModel):
    question: str


def get_db():
    """Get DuckDB connection."""
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return duckdb.connect(DB_PATH, read_only=True)


@app.get("/prices")
async def get_prices():
    """Return recent prices (last 24 hours + metadata)."""
    try:
        conn = get_db()
        df = conn.execute("""
            SELECT timestamp, price_eur_mwh AS price
            FROM prices
            ORDER BY timestamp DESC
            LIMIT 24
        """).df()
        conn.close()

        if df.empty:
            return {"prices": [], "error": "no data"}

        prices = [
            {
                "timestamp": str(row["timestamp"]),
                "price": float(row["price"])
            }
            for _, row in df.iterrows()
        ]
        prices.reverse()

        return {"prices": prices}
    except Exception as e:
        return {"prices": [], "error": str(e)}


@app.get("/forecast")
async def get_forecast():
    """Return latest forecast with current actual price."""
    try:
        conn = get_db()

        forecast_df = conn.execute("""
            SELECT timestamp, p05, p50, p95
            FROM forecasts
            WHERE generated_at = (SELECT MAX(generated_at) FROM forecasts)
            ORDER BY timestamp
            LIMIT 24
        """).df()

        if not forecast_df.empty:
            # Get current actual price
            actual_df = conn.execute("""
                SELECT price_eur_mwh FROM prices
                ORDER BY timestamp DESC LIMIT 1
            """).df()
            current_actual = float(actual_df.iloc[0]["price_eur_mwh"]) if not actual_df.empty else None
        else:
            current_actual = None

        conn.close()

        forecasts = [
            {
                "timestamp": str(row["timestamp"]),
                "p05": float(row["p05"]),
                "p50": float(row["p50"]),
                "p95": float(row["p95"])
            }
            for _, row in forecast_df.iterrows()
        ]

        return {
            "forecasts": forecasts,
            "current_actual": current_actual
        }
    except Exception as e:
        return {"forecasts": [], "error": str(e)}


@app.get("/weather")
async def get_weather():
    """Return latest weather data."""
    try:
        conn = get_db()
        df = conn.execute("""
            SELECT timestamp, temperature, windspeed_100m, cloudcover, solar_radiation
            FROM weather
            ORDER BY timestamp DESC
            LIMIT 24
        """).df()
        conn.close()

        if df.empty:
            return {"weather": {}, "error": "no data"}

        weather = [
            {
                "timestamp": str(row["timestamp"]),
                "temperature": float(row["temperature"]) if row["temperature"] else None,
                "windspeed_100m": float(row["windspeed_100m"]) if row["windspeed_100m"] else None,
                "cloudcover": float(row["cloudcover"]) if row["cloudcover"] else None,
                "solar_radiation": float(row["solar_radiation"]) if row["solar_radiation"] else None,
            }
            for _, row in df.iterrows()
        ]
        weather.reverse()

        return {"weather": weather}
    except Exception as e:
        return {"weather": {}, "error": str(e)}


@app.post("/ask")
async def ask(req: AskRequest):
    """Answer a natural language question about SE3 using the LangGraph agent."""
    state = {
        "question": req.question,
        "tool_results": {},
        "answer": "",
        "confidence": 0.0,
        "sources": [],
        "tools_called": [],
        "tools_failed": []
    }
    result = agent_graph.invoke(state)
    price_data = result["tool_results"].get("price", {})
    forecast_data = result["tool_results"].get("forecast", {})
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "tools_called": result["tools_called"],
        "tools_failed": result["tools_failed"],
        "current_price_eur_mwh": price_data.get("current_price_eur_mwh"),
        "forecast_p50_next_hour": forecast_data.get("next_hour_p50"),
        "forecast_delta": forecast_data.get("forecast_vs_current_delta")
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/retrain")
async def retrain():
    """Run the full forecast refresh pipeline (no model retraining)."""
    import subprocess
    import sys
    import time

    t0             = time.time()
    steps_done: list[str] = []
    logs: list[str]       = []

    def _run(label: str, cmd: list[str], timeout: int = 300) -> bool:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            logs.append(f"[{label}] exit={r.returncode} {r.stdout[-300:].strip()}")
            if r.returncode == 0:
                steps_done.append(label)
                return True
            logs.append(f"[{label}] stderr={r.stderr[-200:].strip()}")
            return False
        except subprocess.TimeoutExpired:
            logs.append(f"[{label}] TIMEOUT after {timeout}s")
            return False
        except Exception as exc:
            logs.append(f"[{label}] ERROR {exc}")
            return False

    _run("pipeline", [sys.executable, "pipeline.py"], timeout=300)
    _run("spot_forecast", [sys.executable, "ml.py", "--forecast"], timeout=120)
    _run("imbalance_forecast", [sys.executable, "ml_imbalance.py", "--forecast"], timeout=120)
    _run("hindcast", [sys.executable, "hindcast_imbalance.py", "--last-days", "7"], timeout=300)
    _run("threshold_diagnosis", [sys.executable, "diagnose_threshold.py", "--days", "30"], timeout=120)

    # Fetch latest generated_at for each forecast table
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
        spot_gen = conn.execute(
            "SELECT MAX(generated_at) FROM forecasts"
        ).fetchone()[0]
        imbl_gen = conn.execute(
            "SELECT MAX(generated_at) FROM imbalance_forecasts"
        ).fetchone()[0]
        thr_rec = conn.execute(
            "SELECT decision, action, recommended_threshold "
            "FROM threshold_decisions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
    except Exception:
        spot_gen = imbl_gen = thr_rec = None

    return {
        "status":                          "ok" if steps_done else "error",
        "steps_completed":                 steps_done,
        "spot_forecast_generated_at":      str(spot_gen) if spot_gen else None,
        "imbalance_forecast_generated_at": str(imbl_gen) if imbl_gen else None,
        "threshold_decision":              (
            f"{thr_rec[1]} → {thr_rec[2]:.4f} ({thr_rec[0]})"
            if thr_rec else None
        ),
        "duration_seconds":                round(time.time() - t0, 1),
        "log":                             "\n".join(logs[-20:]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
