import duckdb, os
import pandas as pd

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

def get_weather_data() -> dict:
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        tables = con.execute("SHOW TABLES").df()["name"].tolist()
        has_smhi = "smhi_wind_forecast" in tables
        has_fcst = "weather_forecast" in tables

        # Build query based on available tables
        select_smhi = ""
        join_smhi = ""
        if has_smhi:
            select_smhi = """
                s.smhi_wind_speed_ms,
                s.smhi_wind_gust_ms,
                s.smhi_cloud_fraction,
                s.smhi_temperature_c,
                s.smhi_precip_prob,
                s.smhi_symbol_code,
            """
            join_smhi = "LEFT JOIN smhi_wind_forecast s ON w.timestamp = s.timestamp"

        select_fcst = ""
        join_fcst = ""
        if has_fcst:
            select_fcst = """
                f.fcst_wind_100m,
                f.fcst_wind_10m,
                f.fcst_cloud_cover,
                f.fcst_temperature,
            """
            join_fcst = "LEFT JOIN weather_forecast f ON w.timestamp = f.timestamp"

        query = f"""
            SELECT
                w.*,
                {select_smhi}
                {select_fcst}
                NULL as _dummy
            FROM weather w
            {join_smhi}
            {join_fcst}
            ORDER BY w.timestamp DESC
            LIMIT 1
        """

        df = con.execute(query).df()
        con.close()

        if df.empty:
            return {"error": "no weather data"}

        row = df.iloc[0].to_dict()
        # Remove the dummy column and convert values
        row.pop("_dummy", None)
        row = {k: round(float(v), 2) if isinstance(v, float) else str(v)
               for k, v in row.items() if v is not None}
        sources = []
        if has_smhi:
            sources.append("SMHI")
        if has_fcst:
            sources.append("Open-Meteo Forecast")
        row["source"] = " + ".join(sources) if sources else "Open-Meteo"
        return row
    except Exception as e:
        return {"error": str(e)}
