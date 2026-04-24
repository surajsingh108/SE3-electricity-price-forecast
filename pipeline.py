"""
pipeline.py — SE3 data pipeline (Module 1)

Fetches and caches all data needed by the ML module:
  - Day-ahead prices       (ENTSO-E)
  - Weather                (Open-Meteo archive + forecast)
  - Wind / load / gen      (ENTSO-E)
  - Nuclear generation     (ENTSO-E query_generation)
  - Cross-border flows     (ENTSO-E crossborder)

Everything lands in a local DuckDB file (data/se3_cache.duckdb).
Each sync is incremental — only the missing tail is fetched.

Usage
-----
  python pipeline.py               # one-time sync to today
  python pipeline.py --schedule    # keep running, refresh hourly
  python pipeline.py --status      # print cache status and exit
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("pipeline")
from dotenv import load_dotenv
load_dotenv()   # reads .env in the current folder before os.environ.get()
# ── Config (override via environment variables) ───────────────────────────────
ENTSO_E_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
SE3             = "10Y1001A1001A46L"
WEATHER_LAT     = float(os.environ.get("SE3_LAT", "59.33"))
WEATHER_LON     = float(os.environ.get("SE3_LON", "18.07"))
TRAIN_START     = os.environ.get("SE3_TRAIN_START", "2020-01-01")
DB_PATH         = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
REFRESH_SECONDS = int(os.environ.get("SE3_REFRESH_SECONDS", "3600"))  # 1 hour

BORDERS = {
    "SE4": "10Y1001A1001A47J",
    "NO1": "10YNO-1--------2",
    "DK1": "10YDK-1--------W",
    "SE2": "10Y1001A1001A45N",
}

# ── Database ──────────────────────────────────────────────────────────────────

def get_conn() -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB cache and ensure all tables exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    conn.executemany("", [])  # warm up
    _create_tables(conn)
    return conn


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            timestamp     TIMESTAMPTZ PRIMARY KEY,
            price_eur_mwh DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            timestamp       TIMESTAMPTZ PRIMARY KEY,
            temperature     DOUBLE,
            windspeed_10m   DOUBLE,
            windspeed_100m  DOUBLE,
            solar_radiation DOUBLE,
            cloudcover      DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation (
            timestamp      TIMESTAMPTZ PRIMARY KEY,
            wind_gen_mw    DOUBLE,
            nuclear_cap_mw DOUBLE,
            load_mw        DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nuclear_gen (
            timestamp      TIMESTAMPTZ PRIMARY KEY,
            nuclear_gen_mw DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cross_border_flows (
            timestamp TIMESTAMPTZ,
            border    VARCHAR,
            flow_mw   DOUBLE,
            PRIMARY KEY (timestamp, border)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            timestamp    TIMESTAMPTZ,
            generated_at TIMESTAMPTZ,
            p05          DOUBLE,
            p50          DOUBLE,
            p95          DOUBLE,
            PRIMARY KEY (timestamp, generated_at)
        )
    """)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cached_range(conn, table: str, border: str | None = None) -> tuple:
    """Return (min_ts, max_ts, row_count) for the given table / border."""
    if border:
        r = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) "
            "FROM cross_border_flows WHERE border = ?",
            [border],
        ).fetchone()
    else:
        r = conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM {table}"
        ).fetchone()
    mn, mx, n = r
    if n:
        mn = pd.Timestamp(mn).tz_convert("Europe/Stockholm")
        mx = pd.Timestamp(mx).tz_convert("Europe/Stockholm")
    return mn, mx, n or 0


def _gap(conn, table: str, start: pd.Timestamp, end: pd.Timestamp,
         border: str | None = None) -> tuple | None:
    """
    Return (fetch_start, fetch_end) for the missing tail, or None if fully cached.
    """
    _, cached_max, n = _cached_range(conn, table, border)
    label = f"{table}{'/' + border if border else ''}"
    if n == 0:
        log.info("  %s: nothing cached, fetching full range", label)
        return start, end
    if cached_max >= end:
        log.info("  ✓ %s: fully cached (%s rows)", label, f"{n:,}")
        return None
    fetch_start = cached_max + pd.Timedelta(hours=1)
    log.info("  ↻ %s: cached to %s, fetching %s → %s",
             label, cached_max.date(), fetch_start.date(), end.date())
    return fetch_start, end


def _upsert(conn, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn.register("_tmp", df)
    conn.execute(f"INSERT OR IGNORE INTO {table} SELECT * FROM _tmp")
    conn.unregister("_tmp")
    log.info("    → wrote %d rows to %s", len(df), table)


def _load(conn, table: str, start: pd.Timestamp, end: pd.Timestamp,
          cols: str = "*") -> pd.DataFrame:
    df = conn.execute(
        f"SELECT {cols} FROM {table} WHERE timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        [start, end],
    ).df()
    if "timestamp" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Stockholm")
        )
        df = df.set_index("timestamp")
    return df


# ── Price sync ────────────────────────────────────────────────────────────────

def sync_prices(
    client,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    conn = get_conn()
    gap  = _gap(conn, "prices", start, end)
    if gap:
        fs, fe = gap
        log.info("  Fetching day-ahead prices %s → %s", fs.date(), fe.date())
        raw = client.query_day_ahead_prices(SE3, start=fs, end=fe)
        raw = raw.rename("price_eur_mwh").to_frame()
        raw.index = raw.index.tz_convert("Europe/Stockholm")
        raw.index.name = "timestamp"
        _upsert(conn, "prices", raw.reset_index())
    df = _load(conn, "prices", start, end)
    conn.close()
    return df


# ── Weather sync ──────────────────────────────────────────────────────────────

def sync_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    lat: float = WEATHER_LAT,
    lon: float = WEATHER_LON,
) -> pd.DataFrame:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    conn = get_conn()
    gap  = _gap(conn, "weather", start, end)
    if gap:
        fs, fe = gap
        log.info("  Fetching weather %s → %s", fs.date(), fe.date())
        sess = requests_cache.CachedSession(".weather_cache", expire_after=-1)
        om   = openmeteo_requests.Client(session=retry(sess, retries=5))
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": fs.strftime("%Y-%m-%d"),
            "end_date":   fe.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m", "windspeed_10m", "windspeed_100m",
                "direct_radiation", "cloudcover",
            ],
            "timezone": "Europe/Stockholm",
        }
        resp   = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
        hourly = resp.Hourly()
        raw = pd.DataFrame({
            "timestamp":      pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ).tz_convert("Europe/Stockholm"),
            "temperature":     hourly.Variables(0).ValuesAsNumpy(),
            "windspeed_10m":   hourly.Variables(1).ValuesAsNumpy(),
            "windspeed_100m":  hourly.Variables(2).ValuesAsNumpy(),
            "solar_radiation": hourly.Variables(3).ValuesAsNumpy(),
            "cloudcover":      hourly.Variables(4).ValuesAsNumpy(),
        })
        _upsert(conn, "weather", raw)
    df = _load(conn, "weather", start, end)
    conn.close()
    return df


def fetch_weather_forecast(
    lat: float = WEATHER_LAT,
    lon: float = WEATHER_LON,
) -> pd.DataFrame:
    """
    Fetch 48h weather FORECAST from Open-Meteo.
    Used for live prediction only — not stored in the cache.
    Cached in memory for 1 hour to avoid hammering the API.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    sess = requests_cache.CachedSession(".weather_forecast_cache", expire_after=3600)
    om   = openmeteo_requests.Client(session=retry(sess, retries=3))
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly": [
            "temperature_2m", "windspeed_10m", "windspeed_100m",
            "direct_radiation", "cloudcover",
        ],
        "forecast_days": 2,
        "timezone":      "Europe/Stockholm",
    }
    resp   = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    hourly = resp.Hourly()
    df = pd.DataFrame({
        "timestamp":      pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert("Europe/Stockholm"),
        "temperature":     hourly.Variables(0).ValuesAsNumpy(),
        "windspeed_10m":   hourly.Variables(1).ValuesAsNumpy(),
        "windspeed_100m":  hourly.Variables(2).ValuesAsNumpy(),
        "solar_radiation": hourly.Variables(3).ValuesAsNumpy(),
        "cloudcover":      hourly.Variables(4).ValuesAsNumpy(),
    }).set_index("timestamp")
    return df


# ── Generation sync ───────────────────────────────────────────────────────────

def sync_generation(
    client,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    conn = get_conn()
    gap  = _gap(conn, "generation", start, end)
    if gap:
        fs, fe = gap
        log.info("  Fetching wind/load/generation %s → %s", fs.date(), fe.date())

        def _fetch(fn, label):
            try:
                raw = fn()
                s   = raw.sum(axis=1) if isinstance(raw, pd.DataFrame) else raw
                s   = s.tz_convert("Europe/Stockholm").resample("h").mean()
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
        _upsert(conn, "generation", raw.reset_index().rename(columns={"index": "timestamp"}))
    df = _load(conn, "generation", start, end)
    conn.close()
    return df


# ── Nuclear generation sync (actual output) ───────────────────────────────────

def sync_nuclear(
    client,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Fetch actual nuclear generation using query_generation (hourly output).
    Fetches in 1-year chunks — ENTSO-E rejects large generation queries.
    Returns a named Series indexed by timestamp.
    """
    conn = get_conn()
    gap  = _gap(conn, "nuclear_gen", start, end)
    if gap:
        fs, fe = gap
        chunk = fs
        while chunk < fe:
            c_end = min(chunk + pd.DateOffset(years=1), fe)
            log.info("    nuclear chunk %s → %s", chunk.date(), c_end.date())
            try:
                raw     = client.query_generation(SE3, start=chunk, end=c_end)
                nuc_col = next(
                    (c for c in raw.columns if "nuclear" in str(c).lower()), None
                )
                if nuc_col is None:
                    log.warning("    no nuclear column — available: %s",
                                list(raw.columns)[:5])
                    chunk = c_end
                    continue
                nuc = raw[nuc_col]
                if isinstance(nuc, pd.DataFrame):
                    nuc = nuc.get("Actual Aggregated", nuc.iloc[:, 0])
                nuc = (nuc.tz_convert("Europe/Stockholm")
                           .resample("h").mean()
                           .ffill(limit=3))
                nuc.name = "nuclear_gen_mw"
                df_c = nuc.reset_index()
                df_c.columns = ["timestamp", "nuclear_gen_mw"]
                _upsert(conn, "nuclear_gen", df_c)
            except Exception as exc:
                log.warning("    nuclear chunk failed: %s", exc)
            chunk = c_end

    df = _load(conn, "nuclear_gen", start, end)
    conn.close()
    return df["nuclear_gen_mw"] if "nuclear_gen_mw" in df.columns else pd.Series(
        dtype=float, name="nuclear_gen_mw"
    )


# ── Cross-border flows sync ───────────────────────────────────────────────────

def sync_flows(
    client,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch net cross-border flows for each SE3 border.
    Net = export − import (positive = SE3 exporting).
    Fetches in 6-month chunks. Returns wide DataFrame with one col per border
    plus net_position_mw.
    """
    conn = get_conn()
    series = {}

    for name, code in BORDERS.items():
        gap = _gap(conn, "cross_border_flows", start, end, border=name)
        if gap:
            fs, fe = gap
            chunk  = fs
            while chunk < fe:
                c_end = min(chunk + pd.DateOffset(months=6), fe)
                log.info("    flow SE3↔%s %s → %s", name, chunk.date(), c_end.date())
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
                        "timestamp": net.index,
                        "border":    name,
                        "flow_mw":   net.values,
                    })
                    _upsert(conn, "cross_border_flows", df_c)
                except Exception as exc:
                    log.warning("    flow SE3↔%s chunk failed: %s", name, exc)
                chunk = c_end

        # Load this border from cache
        df_b = conn.execute(
            "SELECT timestamp, flow_mw FROM cross_border_flows "
            "WHERE border = ? AND timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp",
            [name, start, end],
        ).df()
        df_b["timestamp"] = pd.to_datetime(df_b["timestamp"]).dt.tz_convert("Europe/Stockholm")
        s = df_b.set_index("timestamp")["flow_mw"]
        s.name = f"flow_{name.lower()}_mw"
        series[name] = s

    conn.close()
    if not series:
        return pd.DataFrame()

    flows = pd.concat(series.values(), axis=1)
    flows.columns = list(series.keys())
    flows.columns = [f"flow_{n.lower()}_mw" for n in series]
    flows["net_position_mw"] = flows.sum(axis=1)
    return flows.ffill(limit=3)


# ── Forecast persistence ──────────────────────────────────────────────────────

def save_forecast(forecast_df: pd.DataFrame) -> None:
    """
    Persist a forecast DataFrame (index=future timestamps, cols=p05/p50/p95)
    to the forecasts table so the dashboard can show historical accuracy.
    """
    conn = get_conn()
    now  = pd.Timestamp.now("UTC")
    df   = forecast_df.copy()
    df.index.name = "timestamp"
    df = df.reset_index()
    df["generated_at"] = now
    _upsert(conn, "forecasts", df[["timestamp", "generated_at", "p05", "p50", "p95"]])
    conn.close()


def load_forecasts(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load all saved forecasts for a date range (for backtesting)."""
    conn = get_conn()
    df = conn.execute(
        "SELECT timestamp, generated_at, p05, p50, p95 FROM forecasts "
        "WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp, generated_at",
        [start, end],
    ).df()
    conn.close()
    for col in ["timestamp", "generated_at"]:
        df[col] = pd.to_datetime(df[col]).dt.tz_convert("Europe/Stockholm")
    return df


# ── Cache status ──────────────────────────────────────────────────────────────

def cache_status() -> dict:
    """Return a dict summarising what is in the cache."""
    conn   = get_conn()
    tables = ["prices", "weather", "generation", "nuclear_gen", "forecasts"]
    status = {}
    for t in tables:
        mn, mx, n = _cached_range(conn, t)
        status[t] = {"rows": n, "from": str(mn.date()) if n else None,
                     "to": str(mx.date()) if n else None}
    for border in BORDERS:
        mn, mx, n = _cached_range(conn, "cross_border_flows", border=border)
        status[f"flow_{border.lower()}"] = {
            "rows": n,
            "from": str(mn.date()) if n else None,
            "to":   str(mx.date()) if n else None,
        }
    conn.close()
    return status


def print_status() -> None:
    s = cache_status()
    print(f"\n{'='*55}")
    print(f"  Cache: {DB_PATH}")
    print(f"{'='*55}")
    for name, info in s.items():
        if info["rows"]:
            print(f"  {name:<22} {info['rows']:>8,} rows   "
                  f"{info['from']} → {info['to']}")
        else:
            print(f"  {name:<22}       empty")
    print(f"{'='*55}\n")


# ── Full sync ─────────────────────────────────────────────────────────────────

def sync_all(api_key: str | None = None) -> dict:
    """
    Run a full incremental sync of all data sources.
    Returns a dict with all loaded DataFrames for immediate use.
    """
    from entsoe import EntsoePandasClient

    key    = api_key or ENTSO_E_API_KEY
    client = EntsoePandasClient(api_key=key)
    start  = pd.Timestamp(TRAIN_START, tz="Europe/Stockholm")
    end    = pd.Timestamp.now("Europe/Stockholm").floor("h")

    log.info("Starting full sync %s → %s", start.date(), end.date())

    prices      = sync_prices(client, start, end)
    weather     = sync_weather(start, end)
    gen         = sync_generation(client, start, end)
    nuclear_gen = sync_nuclear(client, start, end)
    flows_df    = sync_flows(client, start, end)

    log.info(
        "Sync complete — prices:%d  weather:%d  gen:%d  nuclear:%d  flows:%s",
        len(prices), len(weather), len(gen), len(nuclear_gen),
        "x".join(str(v) for v in flows_df.shape) if not flows_df.empty else "empty",
    )

    return {
        "prices":      prices,
        "weather":     weather,
        "gen":         gen,
        "nuclear_gen": nuclear_gen,
        "flows_df":    flows_df,
        "start":       start,
        "end":         end,
    }


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduler(api_key: str | None = None) -> None:
    """
    Run sync_all() immediately then repeat every REFRESH_SECONDS.
    Runs forever — suitable for a long-running process or Cloud Run job.
    """
    log.info("Scheduler starting — refresh every %ds", REFRESH_SECONDS)
    while True:
        try:
            sync_all(api_key)
        except Exception as exc:
            log.error("Sync failed: %s", exc)
        log.info("Next sync in %ds", REFRESH_SECONDS)
        time.sleep(REFRESH_SECONDS)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SE3 data pipeline")
    parser.add_argument("--schedule", action="store_true",
                        help="Run continuously, refreshing every hour")
    parser.add_argument("--status",   action="store_true",
                        help="Print cache status and exit")
    parser.add_argument("--api-key",  default=None,
                        help="ENTSO-E API key (overrides ENTSOE_API_KEY env var)")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.schedule:
        run_scheduler(args.api_key)
    else:
        sync_all(args.api_key)
        print_status()
