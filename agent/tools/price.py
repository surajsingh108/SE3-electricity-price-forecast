import duckdb, os
from zoneinfo import ZoneInfo

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
STOCKHOLM = ZoneInfo("Europe/Stockholm")

def get_price_data() -> dict:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)

        # Last 168h for stats (7 days)
        df = con.execute("""
            SELECT timestamp, price_eur_mwh as price
            FROM prices
            ORDER BY timestamp DESC
            LIMIT 168
        """).df()

        # Full 72h hourly series for point lookups
        df_series = con.execute("""
            SELECT timestamp, price_eur_mwh as price
            FROM prices
            WHERE timestamp >= NOW() - INTERVAL '72 hours'
            ORDER BY timestamp ASC
        """).df()

        con.close()

        if df.empty:
            return {"error": "no price data"}

        current = df.iloc[0]
        last_24 = df.head(24)
        last_168 = df

        current_price = float(current["price"])
        mean_7d = float(last_168["price"].mean())
        std_7d = float(last_168["price"].std())

        # Anomaly detection
        z_score = (current_price - mean_7d) / std_7d if std_7d > 0 else 0
        if abs(z_score) > 2:
            anomaly = "high" if z_score > 0 else "low"
            anomaly_note = f"Current price is {abs(z_score):.1f} standard deviations {anomaly} vs 7-day mean"
        else:
            anomaly = "normal"
            anomaly_note = "Price is within normal range"

        # Spike detection in last 48h
        df_48 = df_series.tail(48) if len(df_series) >= 48 else df_series
        peak_row = df_48.loc[df_48["price"].idxmax()] if not df_48.empty else None

        current_ts = current["timestamp"]
        if hasattr(current_ts, 'astimezone'):
            current_ts = current_ts.astimezone(STOCKHOLM)

        result = {
            "current_price_eur_mwh": round(current_price, 2),
            "timestamp": str(current_ts),
            "timezone": "Europe/Stockholm (CEST, UTC+2 in summer)",
            "avg_last_24h_eur_mwh": round(float(last_24["price"].mean()), 2),
            "min_last_24h": round(float(last_24["price"].min()), 2),
            "max_last_24h": round(float(last_24["price"].max()), 2),
            "mean_7d_eur_mwh": round(mean_7d, 2),
            "std_7d_eur_mwh": round(std_7d, 2),
            "z_score": round(z_score, 2),
            "anomaly_status": anomaly,
            "anomaly_note": anomaly_note,
            "source": "ENTSO-E"
        }

        # Add spike info
        if peak_row is not None:
            peak_ts = peak_row["timestamp"]
            if hasattr(peak_ts, 'astimezone'):
                peak_ts = peak_ts.astimezone(STOCKHOLM)
            result["peak_price_48h"] = round(float(peak_row["price"]), 2)
            result["peak_timestamp_48h"] = str(peak_ts)

        # Add full 72h hourly series for point lookups
        if not df_series.empty:
            result["hourly_series_72h"] = [
                {
                    "timestamp": str(row["timestamp"].astimezone(STOCKHOLM))
                             if hasattr(row["timestamp"], 'astimezone')
                             else str(row["timestamp"]),
                    "price": round(float(row["price"]), 2)
                }
                for _, row in df_series.iterrows()
            ]

        return result

    except Exception as e:
        return {"error": str(e)}
