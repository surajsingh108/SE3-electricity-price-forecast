"""
data_sources.py -- SE3 shared data fetching module

All fetch functions for the three SE3 electricity market notebooks.
Run standalone to verify all sources:
  python notebooks/data_sources.py

Functions
---------
fetch_imbalance(start, end)                          -> DataFrame (15-min, eSett)
fetch_openmeteo_archive(start, end)                  -> DataFrame (15-min, Open-Meteo actuals)
fetch_openmeteo_forecast_hist(start, end)            -> DataFrame (15-min, Open-Meteo historical forecast)
fetch_smhi_forecast()                                -> DataFrame (15-min, SMHI snow1g ~83h, fcst_* cols)
fetch_smhi_analysis()                                -> DataFrame (15-min, SMHI mesan2g ~24h, mesan_* cols)
fetch_nuclear_unavailability(client, start, end)     -> DataFrame (15-min, ENTSO-E A80)
fetch_generation_forecast(client, start, end)        -> DataFrame (15-min, ENTSO-E A69)
fetch_cross_border_flows(client, start, end)         -> DataFrame (15-min, ENTSO-E)
fetch_spot_prices(client, start, end)                -> DataFrame (15-min, ENTSO-E)
merge_all_sources(start, end, entsoe_key="")         -> DataFrame (merged, 15-min)

Forecast column naming convention
----------------------------------
Training window : Open-Meteo Historical Forecast API gives fcst_wind_100m,
  fcst_wind_10m, fcst_cloud_cover, fcst_temperature (~100% coverage).
Inference window: SMHI snow1g is mapped to the same fcst_* names and overlaid
  for future rows where historical forecast is NaN.
This ensures identical feature names at train and inference time.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

SE3_MBA     = "10Y1001A1001A46L"
WEATHER_LAT = 59.33
WEATHER_LON = 18.07

SMHI_FORECAST_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/snow1g"
    "/version/1/geotype/point/lon/{lon}/lat/{lat}/data.json"
)
SMHI_ANALYSIS_URL = (
    "https://opendata-download-metanalys.smhi.se/api/category/mesan2g"
    "/version/2/geotype/point/lon/{lon}/lat/{lat}/data.json"
)

BORDERS = {
    "SE4": "10Y1001A1001A47J",
    "NO1": "10YNO-1--------2",
    "DK1": "10YDK-1--------W",
    "SE2": "10Y1001A1001A45N",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, (pd.Timestamp, datetime)):
        return d.date()
    return date.fromisoformat(str(d)[:10])


def _to_ts(d, tz: str = "Europe/Stockholm") -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return d if d.tz is not None else d.tz_localize(tz)
    if isinstance(d, (date, datetime)):
        return pd.Timestamp(str(d)[:10], tz=tz)
    return pd.Timestamp(d, tz=tz)


def _upsample_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex hourly (or sparser) DataFrame to 15-min with ffill."""
    if df.empty or len(df) < 2:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq="15min",
                        tz="Europe/Stockholm")
    return df.reindex(idx).ffill()


def _ok(name: str, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    lo = df.index.min().date() if n else "?"
    hi = df.index.max().date() if n else "?"
    print(f"  OK  {name:<28}: {n:>7,} rows  {lo} -> {hi}")
    return df


def _fail(name: str, exc: Exception) -> pd.DataFrame:
    print(f"  FAIL {name:<27}: {type(exc).__name__}: {exc} (skipped)")
    return pd.DataFrame()


# ── 1. eSett imbalance ────────────────────────────────────────────────────────

def _encode_direction(raw: pd.Series) -> pd.Series:
    if raw.dtype == object:
        return raw.str.upper().map({"LONG": -1, "NEUTRAL": 0, "SHORT": 1}).fillna(0).astype(int)
    return np.sign(pd.to_numeric(raw, errors="coerce").fillna(0)).astype(int)


def _fetch_imbalance_chunk(start: date, end: date) -> pd.DataFrame:
    s_dt = datetime(start.year, start.month, start.day, 0,  0,  0,  tzinfo=timezone.utc)
    e_dt = datetime(end.year,   end.month,   end.day,   23, 59, 59, tzinfo=timezone.utc)
    r = requests.get(
        "https://api.opendata.esett.com/EXP14/Prices",
        params={
            "start": s_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "end":   e_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "mba":   SE3_MBA,
        },
        timeout=90,
    )
    r.raise_for_status()
    data    = r.json()
    records = data if isinstance(data, list) else data.get("data", [])
    if not records:
        return pd.DataFrame()
    df     = pd.DataFrame(records)
    ts_col = next((c for c in ["timestampUTC", "timestamp"] if c in df.columns), None)
    if ts_col is None:
        return pd.DataFrame()
    raw_ts = pd.to_datetime(df[ts_col])
    if ts_col == "timestampUTC":
        df["timestamp"] = (
            (raw_ts.dt.tz_localize("UTC") if raw_ts.dt.tz is None else raw_ts.dt.tz_convert("UTC"))
            .dt.tz_convert("Europe/Stockholm")
        )
    else:
        df["timestamp"] = (
            raw_ts.dt.tz_localize("Europe/Stockholm", ambiguous="infer", nonexistent="shift_forward")
            if raw_ts.dt.tz is None else raw_ts.dt.tz_convert("Europe/Stockholm")
        )
    df = df.rename(columns={
        "imblSalesPrice":          "imbl_price",
        "mainDirRegPowerPerMBA":   "direction_raw",
        "imblSpotDifferencePrice": "imbl_spot_diff",
        "upRegPrice":              "up_reg_price",
        "downRegPrice":            "down_reg_price",
    }).set_index("timestamp").sort_index()
    df["direction"] = _encode_direction(df["direction_raw"]) if "direction_raw" in df.columns else 0
    keep = [c for c in ["imbl_price", "direction", "imbl_spot_diff",
                         "up_reg_price", "down_reg_price"] if c in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def fetch_imbalance(start, end, chunk_months: int = 2) -> pd.DataFrame:
    """
    Fetch SE3 imbalance prices from eSett Open Data EXP14/Prices.

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: imbl_price, direction, imbl_spot_diff, up_reg_price, down_reg_price.
    """
    try:
        s = _to_date(start)
        e = _to_date(end)
        chunks, cur = [], s
        while cur < e:
            m2 = cur.month + chunk_months
            y2 = cur.year + (m2 - 1) // 12
            m2 = (m2 - 1) % 12 + 1
            cend = min(date(y2, m2, 1) - timedelta(days=1), e)
            chunk = _fetch_imbalance_chunk(cur, cend)
            if not chunk.empty:
                chunks.append(chunk)
            cur = cend + timedelta(days=1)
        if not chunks:
            raise RuntimeError("No data returned")
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return _ok("eSett imbalance", df)
    except Exception as exc:
        return _fail("eSett imbalance", exc)


# ── 2. Open-Meteo archive ─────────────────────────────────────────────────────

def fetch_openmeteo_archive(start, end) -> pd.DataFrame:
    """
    Fetch historical weather from Open-Meteo archive API.

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: temperature, windspeed_10m, windspeed_100m,
                 solar_radiation, cloudcover.
    """
    try:
        s = _to_date(start)
        e = min(_to_date(end), date.today() - timedelta(days=2))
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   WEATHER_LAT,
                "longitude":  WEATHER_LON,
                "start_date": s.isoformat(),
                "end_date":   e.isoformat(),
                "hourly":     "temperature_2m,windspeed_10m,windspeed_100m,direct_radiation,cloudcover",
                "timezone":   "UTC",
            },
            timeout=120,
        )
        r.raise_for_status()
        h  = r.json().get("hourly", {})
        df = pd.DataFrame({
            "timestamp":       pd.to_datetime(h["time"]),
            "temperature":     h.get("temperature_2m"),
            "windspeed_10m":   h.get("windspeed_10m"),
            "windspeed_100m":  h.get("windspeed_100m"),
            "solar_radiation": h.get("direct_radiation"),
            "cloudcover":      h.get("cloudcover"),
        })
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Stockholm")
        df = df.set_index("timestamp").sort_index()
        df = _upsample_15min(df)
        return _ok("Open-Meteo archive", df)
    except Exception as exc:
        return _fail("Open-Meteo archive", exc)


# ── 3. Open-Meteo historical forecast ────────────────────────────────────────

def fetch_openmeteo_forecast_hist(start, end) -> pd.DataFrame:
    """
    Fetch Open-Meteo Historical Forecast API for SE3 training window.

    Returns what the ECMWF/best_match forecast model predicted for each hour --
    genuine forecast features for ML training, not reanalysis actuals.
    Covers from 2022-01-01 onward at ~100% completeness.

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: fcst_wind_100m, fcst_wind_10m, fcst_cloud_cover, fcst_temperature.
    """
    try:
        s = _to_date(start)
        e = min(_to_date(end), date.today())
        chunks, cur = [], s
        # chunk by 6 months to stay within API limits
        while cur < e:
            m6 = cur.month + 6
            y6 = cur.year + (m6 - 1) // 12
            m6 = (m6 - 1) % 12 + 1
            cend = min(date(y6, m6, 1) - timedelta(days=1), e)
            r = requests.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params={
                    "latitude":        WEATHER_LAT,
                    "longitude":       WEATHER_LON,
                    "start_date":      cur.isoformat(),
                    "end_date":        cend.isoformat(),
                    "hourly":          "wind_speed_10m,wind_speed_100m,cloud_cover,temperature_2m",
                    "models":          "best_match",
                    "timezone":        "UTC",
                    "wind_speed_unit": "ms",
                },
                timeout=120,
            )
            r.raise_for_status()
            h = r.json().get("hourly", {})
            if h.get("time"):
                df_c = pd.DataFrame({
                    "timestamp":        pd.to_datetime(h["time"]),
                    "fcst_wind_100m":   h.get("wind_speed_100m"),
                    "fcst_wind_10m":    h.get("wind_speed_10m"),
                    "fcst_cloud_cover": h.get("cloud_cover"),
                    "fcst_temperature": h.get("temperature_2m"),
                })
                df_c["timestamp"] = df_c["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Stockholm")
                df_c = df_c.set_index("timestamp").sort_index()
                chunks.append(df_c)
            cur = cend + timedelta(days=1)
        if not chunks:
            raise RuntimeError("No data returned")
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = _upsample_15min(df)
        return _ok("Open-Meteo hist. forecast", df)
    except Exception as exc:
        return _fail("Open-Meteo hist. forecast", exc)


# ── 4. SMHI snow1g forecast ───────────────────────────────────────────────────

def _parse_smhi_ts(ts_list: list, field_map: dict) -> pd.DataFrame:
    rows = []
    for entry in ts_list:
        ts = pd.Timestamp(entry["time"])
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        ts = ts.tz_convert("Europe/Stockholm")
        d  = entry.get("data", {})
        row = {"timestamp": ts}
        for json_key, col in field_map.items():
            row[col] = d.get(json_key)
        rows.append(row)
    df  = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return _upsample_15min(df)


def fetch_smhi_forecast() -> pd.DataFrame:
    """
    Fetch SMHI snow1g weather forecast for SE3 (Stockholm, ~83h ahead).

    Columns mapped to unified fcst_* names so the model sees identical
    feature names at inference time as during training (Open-Meteo hist.).

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: fcst_wind_10m, fcst_temperature, fcst_cloud_cover,
                 fcst_wind_gust, fcst_precip_prob, fcst_thunderstorm_prob.
    """
    try:
        url = SMHI_FORECAST_URL.format(lat=WEATHER_LAT, lon=WEATHER_LON)
        r   = requests.get(url, timeout=30)
        r.raise_for_status()
        ts_list = r.json().get("timeSeries", [])
        if not ts_list:
            raise ValueError("empty timeSeries")
        df = _parse_smhi_ts(ts_list, {
            "wind_speed":                   "fcst_wind_10m",
            "air_temperature":              "fcst_temperature",
            "cloud_area_fraction":          "fcst_cloud_cover",
            "wind_speed_of_gust":           "fcst_wind_gust",
            "probability_of_precipitation": "fcst_precip_prob",
            "thunderstorm_probability":     "fcst_thunderstorm_prob",
        })
        return _ok("SMHI snow1g forecast", df)
    except Exception as exc:
        return _fail("SMHI snow1g forecast", exc)


# ── 5. SMHI mesan2g analysis ──────────────────────────────────────────────────

def fetch_smhi_analysis() -> pd.DataFrame:
    """
    Fetch SMHI mesan2g weather analysis for SE3 (last ~24h).

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: mesan_wind_speed, mesan_temperature, mesan_cloud_fraction,
                 mesan_wind_gust.
    """
    try:
        url = SMHI_ANALYSIS_URL.format(lat=WEATHER_LAT, lon=WEATHER_LON)
        r   = requests.get(url, timeout=30)
        r.raise_for_status()
        ts_list = r.json().get("timeSeries", [])
        if not ts_list:
            raise ValueError("empty timeSeries")
        df = _parse_smhi_ts(ts_list, {
            "wind_speed":          "mesan_wind_speed",
            "air_temperature":     "mesan_temperature",
            "cloud_area_fraction": "mesan_cloud_fraction",
            "wind_speed_of_gust":  "mesan_wind_gust",
        })
        return _ok("SMHI mesan2g analysis", df)
    except Exception as exc:
        return _fail("SMHI mesan2g analysis", exc)


# ── 6. Nuclear unavailability ─────────────────────────────────────────────────

def fetch_nuclear_unavailability(client, start, end) -> pd.DataFrame:
    """
    Fetch ENTSO-E production unavailability (A80) for SE3, nuclear units only.

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: nuclear_unavail_mw   (total nuclear MW offline),
                 nuclear_unplanned_mw (unplanned/forced outages only),
                 nuclear_outage_flag  (1 if any unplanned outage active).
    """
    try:
        s = _to_ts(start)
        e = _to_ts(end)
        chunks, cs = [], s
        while cs < e:
            ce = min(cs + pd.DateOffset(months=6), e)
            try:
                df_c = client.query_unavailability_of_production_units(
                    SE3_MBA, start=cs, end=ce)
                chunks.append(df_c)
            except Exception as exc2:
                log.warning("Nuclear chunk %s->%s: %s", cs.date(), ce.date(), exc2)
            cs = ce
        if not chunks:
            raise RuntimeError("no data from any chunk")

        df  = pd.concat(chunks)
        df  = df[~df.index.duplicated(keep="last")]

        # Filter to nuclear
        nuc = df[df["plant_type"].astype(str).str.lower().str.contains("nuclear", na=False)].copy()
        if nuc.empty:
            raise RuntimeError("no nuclear units in response")

        # Keep only latest revision per mrid
        latest = nuc.groupby("mrid")["revision"].transform("max")
        nuc    = nuc[nuc["revision"] == latest].copy()

        nuc["unavail_mw"] = (
            pd.to_numeric(nuc["nominal_power"], errors="coerce").fillna(0)
            - pd.to_numeric(nuc["avail_qty"],   errors="coerce").fillna(0)
        ).clip(lower=0)
        nuc["is_unplanned"] = ~nuc["businesstype"].astype(str).str.lower().str.contains(
            "planned", na=False)

        # Build 15-min time series
        idx      = pd.date_range(s, e, freq="15min", tz="Europe/Stockholm")
        unavail  = pd.Series(0.0, index=idx)
        unplanned = pd.Series(0.0, index=idx)

        for _, row in nuc.iterrows():
            try:
                rs = pd.Timestamp(row["start"])
                re = pd.Timestamp(row["end"])
                rs = (rs.tz_localize("UTC") if rs.tz is None else rs).tz_convert("Europe/Stockholm")
                re = (re.tz_localize("UTC") if re.tz is None else re).tz_convert("Europe/Stockholm")
                mw   = float(row["unavail_mw"])
                mask = (idx >= rs) & (idx < re)
                unavail[mask]  += mw
                if row["is_unplanned"]:
                    unplanned[mask] += mw
            except Exception:
                continue

        result = pd.DataFrame({
            "nuclear_unavail_mw":   unavail,
            "nuclear_unplanned_mw": unplanned,
        })
        result["nuclear_outage_flag"] = (result["nuclear_unplanned_mw"] > 0).astype(int)
        return _ok("Nuclear unavailability", result)
    except Exception as exc:
        return _fail("Nuclear unavailability", exc)


# ── 7. Generation forecast ────────────────────────────────────────────────────

def fetch_generation_forecast(client, start, end) -> pd.DataFrame:
    """
    Fetch ENTSO-E day-ahead wind and solar generation forecast (A69) for SE3.

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: wind_gen_forecast_mw, solar_gen_forecast_mw.
    """
    try:
        s = _to_ts(start)
        e = _to_ts(end)
        chunks, cs = [], s
        while cs < e:
            ce = min(cs + pd.DateOffset(months=3), e)
            try:
                df_c = client.query_wind_and_solar_forecast(SE3_MBA, start=cs, end=ce)
                chunks.append(df_c)
            except Exception as exc2:
                log.warning("GenFcst chunk %s->%s: %s", cs.date(), ce.date(), exc2)
            cs = ce
        if not chunks:
            raise RuntimeError("no data")
        raw = pd.concat(chunks)
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()
        raw = raw.tz_convert("Europe/Stockholm")
        df  = pd.DataFrame(index=raw.index)
        df["wind_gen_forecast_mw"]  = raw["Wind Onshore"] if "Wind Onshore" in raw.columns else np.nan
        df["solar_gen_forecast_mw"] = raw["Solar"]        if "Solar"        in raw.columns else np.nan
        # Ensure 15-min index spanning full range
        idx = pd.date_range(s, e, freq="15min", tz="Europe/Stockholm")
        df  = df.reindex(idx).ffill(limit=4)
        return _ok("Generation forecast", df)
    except Exception as exc:
        return _fail("Generation forecast", exc)


# ── 8. Cross-border flows ─────────────────────────────────────────────────────

def fetch_cross_border_flows(client, start, end) -> pd.DataFrame:
    """
    Fetch ENTSO-E cross-border flows for SE3 borders (SE4, NO1, DK1, SE2).

    Net = export - import (positive = SE3 exporting).

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).
        Columns: flow_se4_mw, flow_no1_mw, flow_dk1_mw, flow_se2_mw,
                 net_position_mw.
    """
    try:
        s = _to_ts(start)
        e = _to_ts(end)
        series = {}
        for name, code in BORDERS.items():
            try:
                all_c, cs = [], s
                while cs < e:
                    ce = min(cs + pd.DateOffset(months=6), e)
                    exp = client.query_crossborder_flows(SE3_MBA, code, start=cs, end=ce)
                    imp = client.query_crossborder_flows(code, SE3_MBA, start=cs, end=ce)
                    exp = exp.tz_convert("Europe/Stockholm")
                    imp = imp.tz_convert("Europe/Stockholm")
                    all_c.append(exp.subtract(imp, fill_value=0))
                    cs = ce
                net = pd.concat(all_c)
                net = net[~net.index.duplicated(keep="first")].sort_index()
                net.name = f"flow_{name.lower()}_mw"
                series[name] = net
            except Exception as exc2:
                log.warning("Flow SE3->%s failed: %s", name, exc2)
        if not series:
            raise RuntimeError("no border data")
        df = pd.concat(series.values(), axis=1)
        df.columns = [f"flow_{n.lower()}_mw" for n in series]
        df["net_position_mw"] = df.sum(axis=1)
        df = df.ffill(limit=12).fillna(0)  # covers up to 3h gaps; zero-fill remainder
        return _ok("Cross-border flows", df)
    except Exception as exc:
        return _fail("Cross-border flows", exc)


# ── 9. Spot prices ────────────────────────────────────────────────────────────

def fetch_spot_prices(client, start, end) -> pd.DataFrame:
    """
    Fetch ENTSO-E SE3 day-ahead spot prices.

    Returns
    -------
    pd.DataFrame
        15-min index (Europe/Stockholm).  Column: spot_price (EUR/MWh).
    """
    try:
        s   = _to_ts(start)
        e   = _to_ts(end)
        raw = client.query_day_ahead_prices(SE3_MBA, start=s, end=e)
        raw = raw.rename("spot_price").to_frame()
        raw.index = raw.index.tz_convert("Europe/Stockholm")
        idx = pd.date_range(raw.index.min(), raw.index.max(),
                            freq="15min", tz="Europe/Stockholm")
        df  = raw.reindex(idx).ffill()
        return _ok("Spot prices", df)
    except Exception as exc:
        return _fail("Spot prices", exc)


# ── merge_all_sources ─────────────────────────────────────────────────────────

def merge_all_sources(start, end, entsoe_key: str = "") -> pd.DataFrame:
    """
    Fetch and merge all data sources into one 15-min DataFrame.

    Left-joins all sources on the eSett imbalance 15-min index.
    Prints a merge report with row counts and null percentages.

    Parameters
    ----------
    start, end  : date | str | pd.Timestamp
    entsoe_key  : str  ENTSO-E API key (ENTSO-E sources skipped if empty)

    Returns
    -------
    pd.DataFrame
        Master 15-min DataFrame with all available columns.
    """
    print("=" * 60)
    print("DATA SOURCE STATUS")
    print("=" * 60)

    # --- mandatory ---
    df_imbl = fetch_imbalance(start, end)
    if df_imbl.empty:
        raise RuntimeError("eSett fetch failed -- cannot continue without imbalance data")

    df_weather = fetch_openmeteo_archive(start, end)

    # --- Open-Meteo historical forecast (full training coverage, fcst_* cols) ---
    df_fcst_hist  = fetch_openmeteo_forecast_hist(start, end)

    # --- SMHI (free, no key): forecast (~83h ahead) and analysis (~24h actual) ---
    df_smhi_fcst  = fetch_smhi_forecast()   # outputs fcst_* column names
    df_smhi_mesan = fetch_smhi_analysis()   # outputs mesan_* column names

    # --- ENTSO-E (require key) ---
    df_spot    = pd.DataFrame()
    df_nuclear = pd.DataFrame()
    df_genfcst = pd.DataFrame()
    df_flows   = pd.DataFrame()

    if entsoe_key:
        try:
            from entsoe import EntsoePandasClient
            client  = EntsoePandasClient(api_key=entsoe_key)
            df_spot    = fetch_spot_prices(client, start, end)
            df_nuclear = fetch_nuclear_unavailability(client, start, end)
            df_genfcst = fetch_generation_forecast(client, start, end)
            df_flows   = fetch_cross_border_flows(client, start, end)
        except ImportError:
            print("  FAIL entsoe-py not installed -- ENTSO-E sources skipped")
    else:
        print("  INFO no ENTSOE_API_KEY -- spot/nuclear/gen-forecast/flows skipped")

    # --- left-join everything on imbalance index ---
    df = df_imbl.copy()
    for src in [df_weather, df_spot, df_smhi_mesan, df_nuclear, df_genfcst, df_flows]:
        if not src.empty:
            df = df.join(src, how="left")

    # Join historical forecast (primary source for fcst_* columns)
    if not df_fcst_hist.empty:
        df = df.join(df_fcst_hist, how="left")

    # Overlay SMHI forecast for rows where historical forecast is NaN
    # (covers the recent ~83h gap and future inference rows)
    if not df_smhi_fcst.empty:
        smhi_aligned = df_smhi_fcst.reindex(df.index)
        for col in df_smhi_fcst.columns:
            if col in df.columns:
                df[col] = df[col].combine_first(smhi_aligned[col])
            else:
                df[col] = smhi_aligned[col]

    # --- merge report ---
    print()
    print("MERGE REPORT")
    print(f"  Rows   : {len(df):,}")
    print(f"  Cols   : {len(df.columns)}")
    print(f"  Period : {df.index.min().date()} -> {df.index.max().date()}")
    print()

    groups = {
        "imbalance":       ["imbl_price", "direction", "imbl_spot_diff"],
        "weather":         ["temperature", "windspeed_100m", "cloudcover"],
        "spot":            ["spot_price"],
        "fcst_hist":       ["fcst_wind_100m", "fcst_wind_10m"],
        "smhi_fcst_ovlay": ["fcst_wind_gust", "fcst_precip_prob"],
        "smhi_analysis":   ["mesan_wind_speed", "mesan_cloud_fraction"],
        "nuclear":         ["nuclear_unavail_mw", "nuclear_outage_flag"],
        "gen_forecast":    ["wind_gen_forecast_mw", "solar_gen_forecast_mw"],
        "flows":           ["net_position_mw"],
    }
    for grp, cols in groups.items():
        avail = [c for c in cols if c in df.columns]
        if avail:
            null_pct = df[avail[0]].isna().mean() * 100
            print(f"  {grp:<22}: {len(avail)} col(s), {null_pct:5.1f}% null")
        else:
            print(f"  {grp:<22}: not available")

    print("=" * 60)
    return df


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key = os.environ.get("ENTSOE_API_KEY", "")
    end_d   = date.today()
    start_d = end_d - timedelta(days=int(18 * 30.5))

    df = merge_all_sources(start_d, end_d, entsoe_key=key)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\ndata_sources.py standalone test PASSED")
