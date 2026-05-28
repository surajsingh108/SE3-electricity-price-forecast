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
    """Trigger model retraining."""
    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, "retrain.py"],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode == 0:
            return {"status": "success", "log": result.stdout[-2000:]}
        else:
            return {"status": "error", "log": result.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "log": "Training exceeded 30 minutes"}
    except Exception as e:
        return {"status": "error", "log": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
