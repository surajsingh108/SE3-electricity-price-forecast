import duckdb, os
import pandas as pd

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

def get_price_data() -> dict:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute("""
            SELECT timestamp, price_eur_mwh as price
            FROM prices
            ORDER BY timestamp DESC
            LIMIT 25
        """).df()
        con.close()
        if df.empty:
            return {"error": "no price data"}
        current = df.iloc[0]
        last_24 = df.head(24)
        return {
            "current_price_eur_mwh": round(float(current["price"]), 2),
            "timestamp": str(current["timestamp"]),
            "avg_last_24h_eur_mwh": round(float(last_24["price"].mean()), 2),
            "min_last_24h": round(float(last_24["price"].min()), 2),
            "max_last_24h": round(float(last_24["price"].max()), 2),
            "source": "ENTSO-E"
        }
    except Exception as e:
        return {"error": str(e)}
