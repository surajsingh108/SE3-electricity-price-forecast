"""
pipeline.py — SE3 data pipeline

Fetches and caches all data needed by the ML module into DuckDB.
Incremental — only fetches the missing tail on each run.

Usage:
  python pipeline.py          # one-time sync to today
  python pipeline.py --status # print cache status and exit
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("pipeline")

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
SE3            = "10Y1001A1001A46L"
WEATHER_LAT    = float(os.environ.get("SE3_LAT", "59.33"))
WEATHER_LON    = float(os.environ.get("SE3_LON", "18.07"))
TRAIN_START    = os.environ.get("SE3_TRAIN_START", "2020-01-01")
DB_PATH        = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

BORDERS = {
    "SE4": "10Y1001A1001A47J",
    "NO1": "10YNO-1--------2",
    "DK1": "10YDK-1--------W",
    "SE2": "10Y1001A1001A45N",
}


def get_conn() -> duckdb.DuckDBPyConnection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    _create_tables(conn)
    return conn


def _create_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            timestamp TIMESTAMPTZ PRIMARY KEY, price_eur_mwh DOUBLE)""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            timestamp TIMESTAMPTZ PRIMARY KEY, temperature DOUBLE,
            windspeed_10m DOUBLE, windspeed_100m DOUBLE,
            solar_radiation DOUBLE, cloudcover DOUBLE)""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation (
            timestamp TIMESTAMPTZ PRIMARY KEY,
            wind_gen_mw DOUBLE, nuclear_cap_mw DOUBLE, load_mw DOUBLE)""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nuclear_gen (
            timestamp TIMESTAMPTZ PRIMARY KEY, nuclear_gen_mw DOUBLE)""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cross_border_flows (
            timestamp TIMESTAMPTZ, border VARCHAR, flow_mw DOUBLE,
            PRIMARY KEY (timestamp, border))""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            timestamp TIMESTAMPTZ, generated_at TIMESTAMPTZ,
            p05 DOUBLE, p50 DOUBLE, p95 DOUBLE,
            PRIMARY KEY (timestamp, generated_at))""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS smhi_wind_forecast (
            timestamp             TIMESTAMPTZ PRIMARY KEY,
            smhi_wind_speed_ms    DOUBLE,
            smhi_wind_gust_ms     DOUBLE,
            smhi_cloud_fraction   DOUBLE,
            smhi_temperature_c    DOUBLE,
            smhi_precip_prob      DOUBLE,
            smhi_symbol_code      INTEGER,
            fetched_at            TIMESTAMPTZ
        )""")


def _cached_max(conn, table, border=None):
    if border:
        r = conn.execute(
            "SELECT MAX(timestamp), COUNT(*) FROM cross_border_flows WHERE border=?",
            [border]).fetchone()
    else:
        r = conn.execute(
            f"SELECT MAX(timestamp), COUNT(*) FROM {table}").fetchone()
    mx, n = r
    if n:
        mx = pd.Timestamp(mx).tz_convert("Europe/Stockholm")
    return mx, n or 0


def _gap(conn, table, start, end, border=None):
    label = f"{table}{'/' + border if border else ''}"
    mx, n = _cached_max(conn, table, border)
    if n == 0:
        log.info("  %s: empty, fetching full range", label)
        return start, end
    if mx >= end:
        log.info("  ✓ %s: fully cached (%d rows)", label, n)
        return None
    fs = mx + pd.Timedelta(hours=1)
    log.info("  ↻ %s: fetching %s → %s", label, fs.date(), end.date())
    return fs, end


def _upsert(conn, table, df):
    if df.empty:
        return
    conn.register("_tmp", df)
    conn.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM _tmp")
    conn.unregister("_tmp")
    log.info("    → wrote %d rows to %s", len(df), table)


def _load(conn, table, start, end, cols="*"):
    df = conn.execute(
        f"SELECT {cols} FROM {table} WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
        [start, end]).df()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
        df = df.set_index("timestamp")
    return df


def sync_prices(client, start, end):
    conn = get_conn()
    gap = _gap(conn, "prices", start, end)
    if gap:
        fs, fe = gap
        raw = client.query_day_ahead_prices(SE3, start=fs, end=fe)
        raw = raw.rename("price_eur_mwh").to_frame()
        raw.index = raw.index.tz_convert("Europe/Stockholm")
        raw.index.name = "timestamp"
        _upsert(conn, "prices", raw.reset_index())
    df = _load(conn, "prices", start, end)
    conn.close()
    return df


def sync_weather(start, end):
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    conn = get_conn()
    gap = _gap(conn, "weather", start, end)
    if gap:
        fs, fe = gap
        # Archive API lags ~1 day — cap end date
        fe = min(fe, pd.Timestamp.now(tz="Europe/Stockholm").normalize()
                 - pd.Timedelta(days=1))
        if fe >= fs:
            sess = requests_cache.CachedSession(".weather_cache", expire_after=-1)
            om = openmeteo_requests.Client(session=retry(sess, retries=5))
            params = {
                "latitude": WEATHER_LAT, "longitude": WEATHER_LON,
                "start_date": fs.strftime("%Y-%m-%d"),
                "end_date":   fe.strftime("%Y-%m-%d"),
                "hourly": ["temperature_2m", "windspeed_10m", "windspeed_100m",
                           "direct_radiation", "cloudcover"],
                "timezone": "Europe/Stockholm",
            }
            resp = om.weather_api(
                "https://archive-api.open-meteo.com/v1/archive", params=params)[0]
            h = resp.Hourly()
            raw = pd.DataFrame({
                "timestamp": pd.date_range(
                    start=pd.to_datetime(h.Time(), unit="s", utc=True),
                    end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=h.Interval()),
                    inclusive="left").tz_convert("Europe/Stockholm"),
                "temperature":     h.Variables(0).ValuesAsNumpy(),
                "windspeed_10m":   h.Variables(1).ValuesAsNumpy(),
                "windspeed_100m":  h.Variables(2).ValuesAsNumpy(),
                "solar_radiation": h.Variables(3).ValuesAsNumpy(),
                "cloudcover":      h.Variables(4).ValuesAsNumpy(),
            })
            _upsert(conn, "weather", raw)
    df = _load(conn, "weather", start, end)
    conn.close()
    return df


def fetch_weather_forecast():
    """Fetch 48h weather forecast from Open-Meteo (not stored in DB)."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    sess = requests_cache.CachedSession(".weather_forecast_cache", expire_after=3600)
    om = openmeteo_requests.Client(session=retry(sess, retries=3))
    params = {
        "latitude": WEATHER_LAT, "longitude": WEATHER_LON,
        "hourly": ["temperature_2m", "windspeed_10m", "windspeed_100m",
                   "direct_radiation", "cloudcover"],
        "forecast_days": 2, "timezone": "Europe/Stockholm",
    }
    resp = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    h = resp.Hourly()
    return pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(h.Time(), unit="s", utc=True),
            end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=h.Interval()),
            inclusive="left").tz_convert("Europe/Stockholm"),
        "temperature":     h.Variables(0).ValuesAsNumpy(),
        "windspeed_10m":   h.Variables(1).ValuesAsNumpy(),
        "windspeed_100m":  h.Variables(2).ValuesAsNumpy(),
        "solar_radiation": h.Variables(3).ValuesAsNumpy(),
        "cloudcover":      h.Variables(4).ValuesAsNumpy(),
    }).set_index("timestamp")


SMHI_LOCATIONS = {
    "stockholm":  (59.33, 18.07),
    "sundsvall":  (62.39, 17.31),
    "gothenburg": (57.71, 11.97),
    "vasteras":   (59.62, 16.54),
}


def fetch_smhi_wind_forecast() -> pd.DataFrame:
    """
    Fetches wind speed forecasts from SMHI Open Data API (SNOW1gv1)
    for key SE3 locations. No API key required.
    Returns hourly DataFrame with avg wind forecast across locations.
    """
    import requests

    SMHI_BASE = (
        "https://opendata-download-metfcst.smhi.se/api/category/snow1g"
        "/version/1/geotype/point/lon/{lon}/lat/{lat}/data.json"
    )

    all_frames = []

    for location, (lat, lon) in SMHI_LOCATIONS.items():
        try:
            url = SMHI_BASE.format(lat=lat, lon=lon)
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()

            rows = []
            for entry in data.get("timeSeries", []):
                ts = pd.Timestamp(entry["time"])
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                d = entry.get("data", {})
                rows.append({
                    "timestamp":            ts,
                    "location":             location,
                    "wind_speed_ms":        d.get("wind_speed",              None),
                    "wind_dir_deg":         d.get("wind_from_direction",     None),
                    "wind_gust_ms":         d.get("wind_speed_of_gust",      None),
                    "cloud_fraction":       d.get("cloud_area_fraction",     None),
                    "temperature_c":        d.get("air_temperature",         None),
                    "precip_probability":   d.get("probability_of_precipitation", None),
                    "symbol_code":          d.get("symbol_code",             None),
                })

            df_loc = pd.DataFrame(rows)
            all_frames.append(df_loc)
            log.info(f"SMHI {location}: {len(df_loc)} forecast hours")

        except Exception as e:
            log.warning(f"SMHI fetch failed for {location}: {e}")

    if not all_frames:
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)

    # Average across SE3 locations – one row per timestamp
    df_avg = (
        df_all
        .groupby("timestamp")[["wind_speed_ms", "wind_gust_ms", "cloud_fraction",
                                "temperature_c", "precip_probability"]]
        .mean()
        .reset_index()
        .rename(columns={
            "wind_speed_ms":      "smhi_wind_speed_ms",
            "wind_gust_ms":       "smhi_wind_gust_ms",
            "cloud_fraction":     "smhi_cloud_fraction",
            "temperature_c":      "smhi_temperature_c",
            "precip_probability": "smhi_precip_prob",
        })
    )
    # Mode for symbol_code (most common value)
    df_symbol = df_all.groupby("timestamp")["symbol_code"].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
    df_symbol.rename(columns={"symbol_code": "smhi_symbol_code"}, inplace=True)
    df_avg = df_avg.merge(df_symbol, on="timestamp", how="left")
    df_avg["fetched_at"] = pd.Timestamp.now("UTC")
    return df_avg


def sync_generation(client, start, end):
    conn = get_conn()
    gap = _gap(conn, "generation", start, end)
    if gap:
        fs, fe = gap

        def _fetch(fn, label):
            try:
                raw = fn()
                s = raw.sum(axis=1) if isinstance(raw, pd.DataFrame) else raw
                s = s.tz_convert("Europe/Stockholm").resample("h").mean()
                s.name = label
                return s
            except Exception as exc:
                log.warning("  %s failed: %s", label, exc)
                return pd.Series(dtype=float, name=label)

        wind    = _fetch(lambda: client.query_wind_and_solar_forecast(
                         SE3, start=fs, end=fe)["Wind Onshore"], "wind_gen_mw")
        nuclear = _fetch(lambda: client.query_installed_generation_capacity(
                         SE3, start=fs, end=fe)["Nuclear"], "nuclear_cap_mw")
        load    = _fetch(lambda: client.query_load(SE3, start=fs, end=fe), "load_mw")

        raw = pd.concat([wind, nuclear, load], axis=1).ffill(limit=3)
        for col in ["wind_gen_mw", "nuclear_cap_mw", "load_mw"]:
            if col not in raw.columns:
                raw[col] = np.nan
        _upsert(conn, "generation",
                raw.reset_index().rename(columns={"index": "timestamp"}))
    df = _load(conn, "generation", start, end)
    conn.close()
    return df


def sync_nuclear(client, start, end):
    conn = get_conn()
    gap = _gap(conn, "nuclear_gen", start, end)
    if gap:
        fs, fe = gap
        chunk = fs
        while chunk < fe:
            c_end = min(chunk + pd.DateOffset(years=1), fe)
            try:
                raw = client.query_generation(SE3, start=chunk, end=c_end)
                nuc_col = next(
                    (c for c in raw.columns if "nuclear" in str(c).lower()), None)
                if nuc_col is None:
                    chunk = c_end
                    continue
                nuc = raw[nuc_col]
                if isinstance(nuc, pd.DataFrame):
                    nuc = nuc.get("Actual Aggregated", nuc.iloc[:, 0])
                nuc = (nuc.tz_convert("Europe/Stockholm")
                           .resample("h").mean().ffill(limit=3))
                nuc.name = "nuclear_gen_mw"
                df_c = nuc.reset_index()
                df_c.columns = ["timestamp", "nuclear_gen_mw"]
                _upsert(conn, "nuclear_gen", df_c)
            except Exception as exc:
                log.warning("  nuclear chunk failed: %s", exc)
            chunk = c_end
    df = _load(conn, "nuclear_gen", start, end)
    conn.close()
    return df["nuclear_gen_mw"] if "nuclear_gen_mw" in df.columns else pd.Series(
        dtype=float, name="nuclear_gen_mw")


def sync_flows(client, start, end):
    conn = get_conn()
    series = {}
    for name, code in BORDERS.items():
        gap = _gap(conn, "cross_border_flows", start, end, border=name)
        if gap:
            fs, fe = gap
            chunk = fs
            while chunk < fe:
                c_end = min(chunk + pd.DateOffset(months=6), fe)
                try:
                    exp = client.query_crossborder_flows(
                        country_code_from=SE3, country_code_to=code,
                        start=chunk, end=c_end)
                    imp = client.query_crossborder_flows(
                        country_code_from=code, country_code_to=SE3,
                        start=chunk, end=c_end)
                    exp = exp.tz_convert("Europe/Stockholm").resample("h").mean()
                    imp = imp.tz_convert("Europe/Stockholm").resample("h").mean()
                    net = exp.subtract(imp, fill_value=0)
                    df_c = pd.DataFrame({
                        "timestamp": net.index, "border": name,
                        "flow_mw": net.values})
                    _upsert(conn, "cross_border_flows", df_c)
                except Exception as exc:
                    log.warning("  flow SE3→%s failed: %s", name, exc)
                chunk = c_end
        df_b = conn.execute(
            "SELECT timestamp, flow_mw FROM cross_border_flows "
            "WHERE border=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
            [name, start, end]).df()
        df_b["timestamp"] = pd.to_datetime(
            df_b["timestamp"]).dt.tz_convert("Europe/Stockholm")
        s = df_b.set_index("timestamp")["flow_mw"]
        s.name = f"flow_{name.lower()}_mw"
        series[name] = s
    conn.close()
    if not series:
        return pd.DataFrame()
    flows = pd.concat(series.values(), axis=1)
    flows.columns = [f"flow_{n.lower()}_mw" for n in series]
    flows["net_position_mw"] = flows.sum(axis=1)
    return flows.ffill(limit=3)


def save_forecast(forecast_df: pd.DataFrame) -> None:
    conn = get_conn()
    now = pd.Timestamp.now("UTC")
    df = forecast_df.copy()
    df.index.name = "timestamp"
    df = df.reset_index()
    df["generated_at"] = now
    _upsert(conn, "forecasts", df[["timestamp", "generated_at", "p05", "p50", "p95"]])
    conn.close()


def sync_all(api_key=None) -> dict:
    from entsoe import EntsoePandasClient
    key = api_key or ENTSOE_API_KEY
    client = EntsoePandasClient(api_key=key)
    start = pd.Timestamp(TRAIN_START, tz="Europe/Stockholm")
    end   = pd.Timestamp.now("Europe/Stockholm").floor("h")
    log.info("Syncing %s → %s", start.date(), end.date())
    prices      = sync_prices(client, start, end)
    weather     = sync_weather(start, end)

    # SMHI wind forecast
    log.info("Fetching SMHI wind forecast...")
    df_smhi = fetch_smhi_wind_forecast()
    if not df_smhi.empty:
        conn = get_conn()
        conn.register("_smhi", df_smhi)
        conn.execute("""
            INSERT OR REPLACE INTO smhi_wind_forecast
            SELECT timestamp, smhi_wind_speed_ms, smhi_wind_gust_ms,
                   smhi_cloud_fraction, smhi_temperature_c, smhi_precip_prob,
                   smhi_symbol_code, fetched_at
            FROM _smhi
        """)
        conn.unregister("_smhi")
        conn.close()
        log.info(f"SMHI: stored {len(df_smhi)} forecast rows")

    gen         = sync_generation(client, start, end)
    nuclear_gen = sync_nuclear(client, start, end)
    flows_df    = sync_flows(client, start, end)
    log.info("Sync complete — prices:%d weather:%d gen:%d",
             len(prices), len(weather), len(gen))
    return {"prices": prices, "weather": weather, "gen": gen,
            "nuclear_gen": nuclear_gen, "flows_df": flows_df,
            "start": start, "end": end}


def cache_status() -> dict:
    conn = get_conn()
    status = {}
    for t in ["prices", "weather", "generation", "nuclear_gen", "forecasts"]:
        r = conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM {t}").fetchone()
        mn, mx, n = r
        status[t] = {"rows": n,
                     "from": str(pd.Timestamp(mn).date()) if n else None,
                     "to":   str(pd.Timestamp(mx).date()) if n else None}
    conn.close()
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    if args.status:
        import json
        print(json.dumps(cache_status(), indent=2))
    else:
        sync_all(args.api_key)
        import json
        print(json.dumps(cache_status(), indent=2))
