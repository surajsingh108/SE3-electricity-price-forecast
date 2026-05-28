import duckdb, os
import pandas as pd

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

def get_forecast_data() -> dict:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute("""
            SELECT timestamp, p05, p50, p95
            FROM forecasts
            ORDER BY timestamp ASC
            LIMIT 12
        """).df()
        con.close()
        if df.empty:
            return {"error": "no forecast data"}
        next_hour = df.iloc[0]
        return {
            "next_hour_p05": round(float(next_hour["p05"]), 2),
            "next_hour_p50": round(float(next_hour["p50"]), 2),
            "next_hour_p95": round(float(next_hour["p95"]), 2),
            "avg_p50_next_12h": round(float(df["p50"].mean()), 2),
            "source": "SE3 LightGBM model"
        }
    except Exception as e:
        return {"error": str(e)}
