import duckdb, os
from zoneinfo import ZoneInfo

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
STOCKHOLM = ZoneInfo("Europe/Stockholm")

def get_forecast_data() -> dict:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)

        # Next 12h forecasts
        df_fc = con.execute("""
            SELECT timestamp, p05, p50, p95
            FROM forecasts
            ORDER BY timestamp ASC
            LIMIT 12
        """).df()

        # Full 48h forecast series for point lookups
        df_series = con.execute("""
            SELECT timestamp, p05, p50, p95
            FROM forecasts
            WHERE timestamp >= NOW()
            ORDER BY timestamp ASC
            LIMIT 48
        """).df()

        # Forecast vs actual accuracy for last 24h
        df_acc = con.execute("""
            SELECT
                f.timestamp,
                f.p50 AS forecast_p50,
                p.price_eur_mwh AS actual_price,
                ABS(f.p50 - p.price_eur_mwh) AS abs_error,
                f.p50 - p.price_eur_mwh AS signed_error
            FROM forecasts f
            JOIN prices p ON f.timestamp = p.timestamp
            WHERE f.timestamp >= NOW() - INTERVAL '24 hours'
              AND f.timestamp < NOW()
            ORDER BY f.timestamp DESC
        """).df()

        con.close()

        if df_fc.empty:
            return {"error": "no forecast data"}

        next_hour = df_fc.iloc[0]
        avg_p50 = round(float(df_fc["p50"].mean()), 2)

        result = {
            "next_hour_p05": round(float(next_hour["p05"]), 2),
            "next_hour_p50": round(float(next_hour["p50"]), 2),
            "next_hour_p95": round(float(next_hour["p95"]), 2),
            "avg_p50_next_12h": avg_p50,
            "timezone": "Europe/Stockholm (CEST, UTC+2 in summer)",
            "source": "SE3 LightGBM model"
        }

        # Add forecast accuracy stats
        if not df_acc.empty:
            result["forecast_accuracy_last_24h"] = {
                "avg_abs_error_eur_mwh": round(float(df_acc["abs_error"].mean()), 2),
                "max_abs_error_eur_mwh": round(float(df_acc["abs_error"].max()), 2),
                "systematic_bias": round(float(df_acc["signed_error"].mean()), 2),
                "n_hours_compared": len(df_acc),
                "accuracy_rating": (
                    "good" if df_acc["abs_error"].mean() < 10 else
                    "moderate" if df_acc["abs_error"].mean() < 25 else
                    "poor"
                )
            }
        else:
            result["forecast_accuracy_last_24h"] = {
                "note": "insufficient historical overlap to compute accuracy"
            }

        # Add full forecast series for point lookups
        if not df_series.empty:
            result["forecast_series_48h"] = [
                {
                    "timestamp": str(row["timestamp"].astimezone(STOCKHOLM))
                             if hasattr(row["timestamp"], 'astimezone')
                             else str(row["timestamp"]),
                    "p05": round(float(row["p05"]), 2),
                    "p50": round(float(row["p50"]), 2),
                    "p95": round(float(row["p95"]), 2)
                }
                for _, row in df_series.iterrows()
            ]

        return result

    except Exception as e:
        return {"error": str(e)}
