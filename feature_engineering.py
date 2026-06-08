"""
feature_engineering.py -- SE3 shared feature engineering module

Run standalone to verify:
  python notebooks/feature_engineering.py

Functions
---------
build_features(df, horizon=1) -> DataFrame
    Builds the full feature matrix. Combines all features from both
    existing notebooks plus new SMHI/nuclear/gen-forecast features.
    All leakage rules enforced (price/direction features use shift >= 1).

make_feature_cols(df, mode) -> list[str]
    mode: "regression" | "spike" | "regime"
    Returns ordered feature list present in df with notna().any().

SPIKE_CLASS / IS_SPIKE / ASSIGN_NORMAL_CLASS helpers also exported.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

SE_HOLIDAYS = holidays.Sweden()

# Spike thresholds (EUR/MWh)
STRESS_HIGH  = 200.0
STRESS_LOW   = -100.0
EXTREME_HIGH = 1000.0
EXTREME_LOW  = -500.0


# ── label helpers ─────────────────────────────────────────────────────────────

def assign_spike_class(price: pd.Series) -> pd.Series:
    """
    Assign multiclass spike label.

    0 = Normal   price in [STRESS_LOW, STRESS_HIGH]
    1 = Stress   price in (STRESS_HIGH, EXTREME_HIGH] or [EXTREME_LOW, STRESS_LOW)
    2 = Extreme  price > EXTREME_HIGH or < EXTREME_LOW
    """
    cls = pd.Series(0, index=price.index, dtype=int)
    cls[(price > STRESS_HIGH)  & (price <= EXTREME_HIGH)] = 1
    cls[(price < STRESS_LOW)   & (price >= EXTREME_LOW)]  = 1
    cls[price > EXTREME_HIGH] = 2
    cls[price < EXTREME_LOW]  = 2
    return cls


def assign_normal_class(price: pd.Series, p33: float, p67: float) -> pd.Series:
    """
    Assign 3-class label for normal (non-spike) periods only.

    0 = Cheap   price < p33    (long / surplus)
    1 = Normal  price in [p33, p67]
    2 = Expensive price > p67  (short / deficit)
    """
    cls = pd.Series(1, index=price.index, dtype=int)
    cls[price <  p33] = 0
    cls[price >= p67] = 2
    return cls


# ── direction streak ──────────────────────────────────────────────────────────

def _direction_streak(d: pd.Series) -> pd.Series:
    """
    Count consecutive periods in same direction (vectorised).
    Uses d.shift(1) to avoid leakage.
    """
    d_lag   = d.shift(1)
    changed = (d_lag != d_lag.shift(1)).fillna(True)
    grp     = changed.cumsum()
    return d_lag.groupby(grp).cumcount().add(1).astype(float)


# ── main feature builder ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Build full feature matrix from a merged 15-min DataFrame.

    Leakage rules
    -------------
    - price / direction features  : shift(1) minimum on ALL derived features
    - Open-Meteo weather (actual) : no lag needed (observed at t)
    - SMHI forecast               : no lag (these ARE forward-looking)
    - Nuclear availability        : shift(1) used as conservative guard
    - Target columns              : shift(-horizon)

    Parameters
    ----------
    df      : pd.DataFrame  Output of merge_all_sources().
    horizon : int           Forecast horizon in 15-min periods (default 1).

    Returns
    -------
    pd.DataFrame
        All features plus target columns:
          target_price   (float, regression)
          target_binary  (0/1, spike binary)
          target_class   (0/1/2, spike multiclass)
    """
    X = df.copy()
    X.index = pd.to_datetime(X.index)
    if X.index.tz is None:
        X.index = X.index.tz_localize(
            "Europe/Stockholm", ambiguous="infer", nonexistent="shift_forward")

    p = X["imbl_price"] if "imbl_price" in X.columns else pd.Series(0.0, index=X.index)
    d = X["direction"].astype(float) if "direction" in X.columns else pd.Series(0.0, index=X.index)

    # ── A. Imbalance price lags ────────────────────────────────────────────────
    for lag in [1, 2, 4, 8, 16]:
        X[f"imbl_lag_{lag}"] = p.shift(lag)
    X["imbl_lag_1h"] = p.shift(4)
    X["imbl_lag_2h"] = p.shift(8)
    X["imbl_lag_4h"] = p.shift(16)

    # ── B. Direction features ──────────────────────────────────────────────────
    for lag in [1, 2, 4, 8]:
        X[f"dir_lag_{lag}"] = d.shift(lag)
    X["dir_streak"] = _direction_streak(d)

    # ── C. Rolling stats on price (shift(1) before rolling) ──────────────────
    p1 = p.shift(1)
    X["imbl_roll_1h_mean"] = p1.rolling(4).mean()
    X["imbl_roll_1h_std"]  = p1.rolling(4).std()
    X["imbl_roll_4h_mean"] = p1.rolling(16).mean()
    X["imbl_roll_4h_std"]  = p1.rolling(16).std()
    X["imbl_roll_1d_mean"] = p1.rolling(96).mean()
    X["imbl_ewa"]          = p1.ewm(span=8, adjust=False).mean()

    # ── D. Regulation / spot spreads ──────────────────────────────────────────
    if ("up_reg_price" in X.columns and "down_reg_price" in X.columns
            and X["up_reg_price"].notna().any()):
        X["reg_spread"] = (X["up_reg_price"] - X["down_reg_price"]).shift(1)
    if "imbl_spot_diff" in X.columns and X["imbl_spot_diff"].notna().any():
        X["imbl_spot_diff_lag1"] = X["imbl_spot_diff"].shift(1)
    if "spot_price" in X.columns and X["spot_price"].notna().any():
        X["spot_price_lag1"] = X["spot_price"].shift(1)

    # ── E. Spike-specific features ────────────────────────────────────────────
    X["imbl_abs_lag1"]    = p.shift(1).abs()
    X["imbl_roll_1h_max"] = p1.rolling(4).max()
    X["imbl_roll_4h_max"] = p1.rolling(16).max()
    X["imbl_roll_1h_min"] = p1.rolling(4).min()
    X["imbl_roll_4h_min"] = p1.rolling(16).min()

    eps = 1e-6
    X["price_zscore_1h"] = (p.shift(1) - X["imbl_roll_1h_mean"]) / (X["imbl_roll_1h_std"] + eps)
    X["price_zscore_4h"] = (p.shift(1) - X["imbl_roll_4h_mean"]) / (X["imbl_roll_4h_std"] + eps)

    X["stress_proximity_up"]   = (STRESS_HIGH  - p.shift(1)).clip(upper=STRESS_HIGH)
    X["stress_proximity_down"] = (p.shift(1)   - STRESS_LOW).clip(upper=abs(STRESS_LOW))

    above    = (p.shift(1).abs() > STRESS_HIGH).astype(int)
    chg      = (above != above.shift(1)).cumsum()
    X["consecutive_stress"] = above.groupby(chg).cumcount() * above

    X["vol_acceleration"]  = X["imbl_roll_1h_std"] - X["imbl_roll_4h_std"]
    X["price_momentum_1h"] = p.shift(1) - p.shift(5)
    X["price_momentum_4h"] = p.shift(1) - p.shift(17)

    d1       = d.shift(1)
    dir_flip = (d1 != d1.shift(1)).astype(int)
    X["dir_flip_rate_1h"] = dir_flip.rolling(4).sum()
    X["dir_flip_rate_4h"] = dir_flip.rolling(16).sum()
    X["dir_flip_rate_1d"] = dir_flip.rolling(96).sum()

    if "reg_spread" in X.columns:
        rs = X["reg_spread"]
        X["reg_spread_roll_1h"] = rs.rolling(4).mean()
        X["reg_spread_roll_4h"] = rs.rolling(16).mean()
        X["reg_spread_accel"]   = rs.rolling(4).mean() - rs.rolling(16).mean()

    d_lag2 = d.shift(2)
    X["regime_flip_to_short"] = ((d1 == 1)  & (d_lag2 == -1)).astype(int)
    X["regime_flip_to_long"]  = ((d1 == -1) & (d_lag2 == 1)).astype(int)

    in_stress_lag = (p.shift(1).abs() > STRESS_HIGH).astype(int)
    ctr, since = 0, []
    for v in in_stress_lag:
        if v == 1:
            ctr = 0
        else:
            ctr += 1
        since.append(ctr)
    X["periods_since_stress"] = since

    # ── F. Calendar ───────────────────────────────────────────────────────────
    X["hour"]      = X.index.hour
    X["minute"]    = X.index.minute
    X["dayofweek"] = X.index.dayofweek
    X["month"]     = X.index.month
    X["slot"]      = X["hour"] * 4 + X["minute"] // 15

    X["hour_sin"]  = np.sin(2 * np.pi * X["hour"]      / 24)
    X["hour_cos"]  = np.cos(2 * np.pi * X["hour"]      / 24)
    X["dow_sin"]   = np.sin(2 * np.pi * X["dayofweek"] / 7)
    X["dow_cos"]   = np.cos(2 * np.pi * X["dayofweek"] / 7)
    X["month_sin"] = np.sin(2 * np.pi * X["month"]     / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"]     / 12)
    X["slot_sin"]  = np.sin(2 * np.pi * X["slot"]      / 96)
    X["slot_cos"]  = np.cos(2 * np.pi * X["slot"]      / 96)

    how = X["dayofweek"] * 96 + X["slot"]
    hoy = (X.index.dayofyear - 1) * 96 + X["slot"]
    for k in [1, 2, 3]:
        X[f"daily_sin_k{k}"]  = np.sin(2 * np.pi * k * X["slot"] / 96)
        X[f"daily_cos_k{k}"]  = np.cos(2 * np.pi * k * X["slot"] / 96)
        X[f"weekly_sin_k{k}"] = np.sin(2 * np.pi * k * how / (7 * 96))
        X[f"weekly_cos_k{k}"] = np.cos(2 * np.pi * k * how / (7 * 96))
    for k in [1, 2]:
        X[f"annual_sin_k{k}"] = np.sin(2 * np.pi * k * hoy / (365 * 96))
        X[f"annual_cos_k{k}"] = np.cos(2 * np.pi * k * hoy / (365 * 96))

    X["season"]     = X["month"].map({12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4})
    X["is_weekend"] = (X.index.dayofweek >= 5).astype(int)
    X["is_holiday"] = X.index.normalize().map(lambda d_: int(d_ in SE_HOLIDAYS))
    is_peak         = (X["hour"].isin(range(7, 10)) | X["hour"].isin(range(17, 21))).astype(int)
    X["is_peak"]    = is_peak
    X["is_night"]   = X["hour"].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    is_winter       = X["month"].isin([12, 1, 2]).astype(int)
    X["is_winter"]  = is_winter

    # ── G. Weather features ───────────────────────────────────────────────────
    wind_col = "windspeed_100m" if "windspeed_100m" in X.columns else "windspeed_10m"
    if wind_col in X.columns:
        w = X[wind_col]
        X["wind_7d_mean"]  = w.rolling(672, min_periods=48).mean()
        X["wind_surprise"] = w - X["wind_7d_mean"]
        X["wind_x_night"]  = w * X["is_night"]
        X["wind_x_peak"]   = w * X["is_peak"]
        X["wind_squared"]  = w ** 2
        if "stress_proximity_up" in X.columns:
            X["wind_surprise_x_stress"] = X["wind_surprise"] * (
                STRESS_HIGH / (X["stress_proximity_up"] + 1))

    if "temperature" in X.columns:
        X["heating_degree"] = (15 - X["temperature"]).clip(lower=0)
        X["temp_x_peak"]    = X["temperature"] * is_peak
        X["temp_x_winter"]  = X["temperature"] * is_winter
        X["demand_stress"]  = X["heating_degree"] * is_peak * is_winter

    if "cloudcover" in X.columns:
        X["cloudcover_lag1"] = X["cloudcover"].shift(1)

    # ── H. Interaction features ───────────────────────────────────────────────
    if all(c in X.columns for c in ["demand_stress", "imbl_roll_1h_std"]):
        X["demand_stress_x_vol"] = X["demand_stress"] * X["imbl_roll_1h_std"].shift(1)
    if "dir_streak" in X.columns:
        X["short_streak_x_peak"] = X["dir_streak"] * (d1 == 1).astype(int) * is_peak

    # ── I. Weather forecast features (fcst_* unified columns) ────────────────
    # Training: Open-Meteo Historical Forecast API (full 18-month coverage ~100%).
    # Inference: SMHI snow1g mapped to identical fcst_* names and overlaid.
    # wind_forecast_surprise: positive = windier than predicted (lower price pressure);
    #   negative = less wind than forecast (higher spike risk).
    if "fcst_wind_10m" in X.columns and X["fcst_wind_10m"].notna().any():
        print(f"  fcst_wind_10m coverage: {X['fcst_wind_10m'].notna().mean():.1%}")
        X["fcst_wind_10m_lag1"] = X["fcst_wind_10m"].shift(1)
        if "fcst_temperature" in X.columns:
            print(f"  fcst_temperature coverage: {X['fcst_temperature'].notna().mean():.1%}")
            X["fcst_temperature_lag1"] = X["fcst_temperature"].shift(1)
        if "fcst_cloud_cover" in X.columns:
            X["fcst_cloud_cover_lag1"] = X["fcst_cloud_cover"].shift(1)
        if "fcst_wind_gust" in X.columns and X["fcst_wind_gust"].notna().any():
            X["fcst_wind_gust_lag1"] = X["fcst_wind_gust"].shift(1)
        if "fcst_precip_prob" in X.columns and X["fcst_precip_prob"].notna().any():
            X["fcst_precip_prob_lag1"] = X["fcst_precip_prob"].shift(1)
        if "fcst_thunderstorm_prob" in X.columns and X["fcst_thunderstorm_prob"].notna().any():
            X["fcst_thunderstorm_lag1"] = X["fcst_thunderstorm_prob"].shift(1)
        # Prefer 100m forecast (closer to wind-generation hub height)
        fcst_ref = "fcst_wind_100m" if (
            "fcst_wind_100m" in X.columns and X["fcst_wind_100m"].notna().any()
        ) else "fcst_wind_10m"
        if wind_col in X.columns:
            X["wind_forecast_surprise"] = X[wind_col].shift(1) - X[fcst_ref].shift(1)

    if "mesan_wind_speed" in X.columns and X["mesan_wind_speed"].notna().any():
        X["mesan_wind_lag1"]  = X["mesan_wind_speed"].shift(1)
        X["mesan_cloud_lag1"] = X["mesan_cloud_fraction"].shift(1) if "mesan_cloud_fraction" in X.columns else np.nan

    # ── J. Nuclear features (guarded, shift(1)) ───────────────────────────────
    if "nuclear_unavail_mw" in X.columns and X["nuclear_unavail_mw"].notna().any():
        X["nuclear_unavail_lag1"]     = X["nuclear_unavail_mw"].shift(1)
        X["nuclear_unplanned_lag1"]   = X["nuclear_unplanned_mw"].shift(1)  if "nuclear_unplanned_mw"  in X.columns else np.nan
        X["nuclear_outage_flag_lag1"] = X["nuclear_outage_flag"].shift(1)   if "nuclear_outage_flag"   in X.columns else np.nan
        X["nuclear_sudden_drop"]      = X["nuclear_unavail_mw"].diff(4).shift(1)  # 1h change, lagged
        X["nuclear_x_peak"]           = X["nuclear_unavail_lag1"] * is_peak
        X["nuclear_x_winter"]         = X["nuclear_unavail_lag1"] * is_winter
        if "stress_proximity_up" in X.columns:
            X["nuclear_x_stress_proximity"] = (
                X["nuclear_unavail_lag1"] * (STRESS_HIGH / (X["stress_proximity_up"] + 1))
            )

    # ── K. Generation forecast features (guarded) ────────────────────────────
    if "wind_gen_forecast_mw" in X.columns and X["wind_gen_forecast_mw"].notna().any():
        X["wind_gen_forecast_lag4"] = X["wind_gen_forecast_mw"].shift(4)  # 1-hour lag
        if wind_col in X.columns:
            X["wind_forecast_vs_actual"] = X[wind_col].shift(96) - X["wind_gen_forecast_mw"].shift(96)
    if "solar_gen_forecast_mw" in X.columns and X["solar_gen_forecast_mw"].notna().any():
        X["solar_gen_forecast_lag4"] = X["solar_gen_forecast_mw"].shift(4)

    # ── L. Cross-border flow features (guarded) ───────────────────────────────
    flow_cols = [c for c in X.columns
                 if c.startswith("flow_") and c.endswith("_mw") and c != "net_position_mw"]
    for fc in flow_cols:
        X[f"{fc}_lag4"] = X[fc].shift(4)
    if "net_position_mw" in X.columns and X["net_position_mw"].notna().any():
        X["net_pos_lag4"]   = X["net_position_mw"].shift(4)
        X["net_pos_roll1d"] = X["net_position_mw"].shift(4).rolling(96, min_periods=12).mean()

    # ── M. Target columns ─────────────────────────────────────────────────────
    if "imbl_price" in X.columns:
        X["target_price"] = p.shift(-horizon)
        spike_cls = assign_spike_class(p)
        X["spike_class"]    = spike_cls
        X["is_spike"]       = (spike_cls > 0).astype(int)
        X["target_binary"]  = X["is_spike"].shift(-horizon)
        X["target_class"]   = X["spike_class"].shift(-horizon)

    return X


# ── feature column selector ───────────────────────────────────────────────────

def make_feature_cols(df: pd.DataFrame, mode: str = "regression") -> list[str]:
    """
    Return ordered feature column list for the given modeling mode.

    Parameters
    ----------
    df   : pd.DataFrame  Output of build_features().
    mode : str
        "regression" -- imbalance price quantile regression
        "spike"      -- binary spike detection classifier
        "regime"     -- 3-class normal-regime classifier

    Returns
    -------
    list[str]
        Only columns present in df with notna().any(), deduplicated.
    """
    # ── base lags (all modes) ─────────────────────────────────────────────────
    base = (
        [f"imbl_lag_{l}" for l in [1, 2, 4, 8, 16]]
        + ["imbl_lag_1h", "imbl_lag_2h", "imbl_lag_4h"]
        + ["imbl_roll_1h_mean", "imbl_roll_1h_std",
           "imbl_roll_4h_mean", "imbl_roll_4h_std",
           "imbl_roll_1d_mean", "imbl_ewa"]
        + [f"dir_lag_{l}" for l in [1, 2, 4, 8]]
        + ["dir_streak"]
    )

    # ── spike-specific (spike mode only) ──────────────────────────────────────
    spike = [
        "imbl_abs_lag1",
        "imbl_roll_1h_max", "imbl_roll_4h_max", "imbl_roll_1h_min", "imbl_roll_4h_min",
        "price_zscore_1h", "price_zscore_4h",
        "stress_proximity_up", "stress_proximity_down", "consecutive_stress",
        "vol_acceleration", "price_momentum_1h", "price_momentum_4h",
        "dir_flip_rate_1h", "dir_flip_rate_4h", "dir_flip_rate_1d",
        "reg_spread", "reg_spread_roll_1h", "reg_spread_roll_4h", "reg_spread_accel",
        "regime_flip_to_short", "regime_flip_to_long", "periods_since_stress",
        "imbl_spot_diff_lag1", "spot_price_lag1",
        "demand_stress_x_vol", "short_streak_x_peak",
    ]

    # ── calendar (all modes) ──────────────────────────────────────────────────
    cal = (
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
         "month_sin", "month_cos", "slot_sin", "slot_cos",
         "season", "is_night", "is_peak", "is_weekend", "is_holiday", "is_winter"]
        + [f"daily_{t}_k{k}"  for k in [1, 2, 3] for t in ["sin", "cos"]]
        + [f"weekly_{t}_k{k}" for k in [1, 2, 3] for t in ["sin", "cos"]]
        + [f"annual_{t}_k{k}" for k in [1, 2]    for t in ["sin", "cos"]]
    )

    # ── weather (all modes) ───────────────────────────────────────────────────
    weather = [
        "windspeed_100m", "windspeed_10m",
        "wind_7d_mean", "wind_surprise",
        "wind_x_night", "wind_x_peak", "wind_squared", "wind_surprise_x_stress",
        "temperature", "heating_degree", "temp_x_peak", "temp_x_winter",
        "demand_stress", "cloudcover", "cloudcover_lag1",
    ]

    # ── interactions (all modes) ──────────────────────────────────────────────
    interactions = ["demand_stress_x_vol", "short_streak_x_peak"]

    # ── forecast features (all modes) ────────────────────────────────────────
    # fcst_* from Open-Meteo Historical Forecast (training) or SMHI (inference)
    fcst = [
        "fcst_wind_10m_lag1", "fcst_wind_100m", "fcst_temperature_lag1",
        "fcst_cloud_cover_lag1", "fcst_wind_gust_lag1",
        "fcst_precip_prob_lag1", "fcst_thunderstorm_lag1",
        "wind_forecast_surprise",
        "mesan_wind_lag1", "mesan_cloud_lag1",
    ]
    nuclear = [
        "nuclear_unavail_lag1", "nuclear_unplanned_lag1", "nuclear_outage_flag_lag1",
        "nuclear_sudden_drop", "nuclear_x_peak", "nuclear_x_winter",
        "nuclear_x_stress_proximity",
    ]
    gen_fcst = [
        "wind_gen_forecast_lag4", "wind_forecast_vs_actual", "solar_gen_forecast_lag4",
    ]
    flows = (
        [c for c in df.columns if c.startswith("flow_") and c.endswith("_lag4")]
        + ["net_pos_lag4", "net_pos_roll1d"]
    )
    reg_extras = ["reg_spread", "imbl_spot_diff_lag1", "spot_price_lag1"]

    if mode == "regression":
        all_feats = base + cal + weather + interactions + reg_extras + fcst + nuclear + gen_fcst + flows
    elif mode == "spike":
        all_feats = base + spike + cal + weather + fcst + nuclear + gen_fcst + flows
    elif mode == "regime":
        # Normal-regime: exclude extreme spike features (near-zero in normal periods)
        all_feats = base + cal + weather + interactions + reg_extras + fcst + nuclear + gen_fcst + flows
    else:
        all_feats = base + spike + cal + weather + fcst + nuclear + gen_fcst + flows

    seen: set[str] = set()
    result: list[str] = []
    for c in all_feats:
        if c in df.columns and c not in seen and df[c].notna().any():
            seen.add(c)
            result.append(c)
    return result


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from data_sources import merge_all_sources
    from datetime import date, timedelta

    key     = os.environ.get("ENTSOE_API_KEY", "")
    end_d   = date.today()
    start_d = end_d - timedelta(days=60)

    print("Building 60-day test feature matrix...")
    df      = merge_all_sources(start_d, end_d, entsoe_key=key)
    df_feat = build_features(df, horizon=1)

    print()
    for mode in ["regression", "spike", "regime"]:
        cols = make_feature_cols(df_feat, mode)
        print(f"Mode '{mode}': {len(cols)} features total")

        groups = {
            "base_lags":    [c for c in cols if c.startswith("imbl_lag") or c.startswith("imbl_roll") or c == "imbl_ewa"],
            "direction":    [c for c in cols if c.startswith("dir_")],
            "spike":        [c for c in cols if any(k in c for k in ["zscore","stress_prox","consec","momentum","flip_rate","regime_flip","periods_since"])],
            "calendar":     [c for c in cols if any(k in c for k in ["_sin","_cos","season","is_","holiday"])],
            "weather":      [c for c in cols if any(k in c for k in ["wind","temp","cloud","heat","demand"])],
            "fcst":         [c for c in cols if c.startswith("fcst_") or c.startswith("mesan_") or c == "wind_forecast_surprise"],
            "nuclear":      [c for c in cols if c.startswith("nuclear_")],
            "gen_forecast": [c for c in cols if "forecast" in c],
            "flows":        [c for c in cols if c.startswith("flow_") or "net_pos" in c],
            "regulation":   [c for c in cols if "reg_spread" in c or "spot_price" in c or "spot_diff" in c],
        }
        for grp, gcols in groups.items():
            if gcols:
                print(f"  {grp:<16}: {len(gcols)}")
        print()

    print("feature_engineering.py standalone test PASSED")
