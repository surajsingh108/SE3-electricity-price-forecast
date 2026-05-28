import duckdb, os
import pandas as pd

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

def get_weather_data() -> dict:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute("""
            SELECT * FROM weather
            ORDER BY timestamp DESC
            LIMIT 1
        """).df()
        con.close()
        if df.empty:
            return {"error": "no weather data"}
        row = df.iloc[0].to_dict()
        row = {k: round(float(v), 2) if isinstance(v, float) else str(v)
               for k, v in row.items()}
        row["source"] = "Open-Meteo"
        return row
    except Exception as e:
        return {"error": str(e)}
