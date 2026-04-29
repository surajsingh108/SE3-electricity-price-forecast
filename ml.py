"""
ml.py — SE3 ML module (Module 2)

End-to-end machine learning pipeline for SE3 day-ahead price forecasting.
Imports data from pipeline.py, produces forecasts, saves model artifacts.

Public API
----------
  train(data)          → artifacts dict (models + config + neutralizers)
  predict(artifacts, data, weather_fc) → 24h forecast DataFrame
  evaluate(artifacts, data) → metrics dict
  save_artifacts(artifacts, path)
  load_artifacts(path) → artifacts dict

Usage
-----
  python ml.py --train          # retrain and save to model/
  python ml.py --forecast       # load model, print next 24h
  python ml.py --evaluate       # print test set metrics
"""

from __future__ import annotations

import argparse
import holidays
import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

log = logging.getLogger("ml")

SE_HOLIDAYS = holidays.Sweden()
MODEL_DIR   = Path("model")

# ── Feature groups ─────────────────────────────────────────────────────────────

WIND_COL = "windspeed_100m"   # preferred; falls back to windspeed_10m at runtime

TRIVIAL_COLS = [               # absorbed by the Ridge linear baseline (V3)
    "price_ewa",
    "price_roll_720h_mean",
    "price_roll_168h_mean",
] + [f"daily_{t}_k{k}"  for k in [1, 2, 3] for t in ["sin", "cos"]] \
  + [f"weekly_{t}_k{k}" for k in [1, 2, 3] for t in ["sin", "cos"]] \
  + [f"annual_{t}_k{k}" for k in [1, 2]   for t in ["sin", "cos"]]

# Weather / generation features neutralized against Fourier time basis (V2)
NEUTRALIZE_FEATS = [
    "windspeed_100m", "windspeed_10m",
    "wind_surprise", "wind_7d_mean",
    "temperature", "heating_degree",
    "load_lag24", "load_residual",
    "wind_gen_lag24", "wind_load_ratio",
    "cloudcover",
]

LGBM_PARAMS = {
    "n_estimators":      2000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "verbose":          -1,
}

# ── Pretrain / posttrain constants ────────────────────────────────────────────

PRETRAIN_CUTOFF_DAYS  = 180   # pretrain on data older than this many days
POSTTRAIN_WINDOW_DAYS = 90    # posttrain uses this many recent days
POSTTRAIN_TREES       = 300   # extra trees added on top of pretrained model
POSTTRAIN_LR          = 0.02  # lower LR to avoid overfitting recent noise

POSTTRAIN_LGBM_PARAMS = {
    **LGBM_PARAMS,
    "n_estimators":  POSTTRAIN_TREES,
    "learning_rate": POSTTRAIN_LR,
}


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from the merged raw DataFrame.
    All features are lagged ≥ 24h — zero data leakage.

    Parameters
    ----------
    df : merged DataFrame with columns:
         price_eur_mwh, temperature, windspeed_10m, windspeed_100m,
         solar_radiation, cloudcover, wind_gen_mw, load_mw,
         nuclear_gen_mw (optional),
         flow_se4_mw, flow_no1_mw, flow_dk1_mw, flow_se2_mw,
         net_position_mw (optional)
    """
    X = df.copy()

    # DST-safe DatetimeIndex
    X.index = pd.to_datetime(X.index)
    if X.index.tz is None:
        X.index = X.index.tz_localize(
            "Europe/Stockholm", ambiguous="infer", nonexistent="shift_forward"
        )

    p = X["price_eur_mwh"]

    # ── Price lags (all ≥ 24h) ────────────────────────────────────────────────
    X["price_lag_24h"]  = p.shift(24)
    X["price_lag_48h"]  = p.shift(48)
    X["price_lag_168h"] = p.shift(168)

    # EWA same-hour anchor: heavy weight on recent days
    # Adapts to regime shifts faster than a simple 7-day mean
    X["price_ewa"] = (
        p.shift(24)*0.35 + p.shift(48)*0.25 + p.shift(72)*0.17 +
        p.shift(96)*0.11 + p.shift(120)*0.06 + p.shift(144)*0.04 + p.shift(168)*0.02
    )
    X["price_lag24_vs_ewa"]  = X["price_lag_24h"] - X["price_ewa"]
    X["price_3d_mean"]       = (p.shift(24) + p.shift(48) + p.shift(72)) / 3
    X["price_lag24_vs_3d"]   = X["price_lag_24h"] - X["price_3d_mean"]

    X["price_roll_168h_mean"] = p.shift(24).rolling(168).mean()
    X["price_roll_168h_std"]  = p.shift(24).rolling(168).std()
    X["price_roll_720h_mean"] = p.shift(24).rolling(720).mean()
    X["regime_x_vol"]         = X["price_roll_720h_mean"] * X["price_roll_168h_std"]

    # ── Calendar ──────────────────────────────────────────────────────────────
    X["hour"]      = X.index.hour
    X["dayofweek"] = X.index.dayofweek
    X["month"]     = X.index.month

    X["hour_sin"]  = np.sin(2*np.pi*X["hour"]/24)
    X["hour_cos"]  = np.cos(2*np.pi*X["hour"]/24)
    X["month_sin"] = np.sin(2*np.pi*X["month"]/12)
    X["month_cos"] = np.cos(2*np.pi*X["month"]/12)
    X["dow_sin"]   = np.sin(2*np.pi*X["dayofweek"]/7)
    X["dow_cos"]   = np.cos(2*np.pi*X["dayofweek"]/7)

    # Fourier harmonics — k2 captures double morning+evening peak
    how = X["dayofweek"]*24 + X["hour"]
    hoy = (X.index.dayofyear - 1)*24 + X["hour"]
    for k in [1, 2, 3]:
        X[f"daily_sin_k{k}"]  = np.sin(2*np.pi*k*X["hour"]/24)
        X[f"daily_cos_k{k}"]  = np.cos(2*np.pi*k*X["hour"]/24)
        X[f"weekly_sin_k{k}"] = np.sin(2*np.pi*k*how/168)
        X[f"weekly_cos_k{k}"] = np.cos(2*np.pi*k*how/168)
    for k in [1, 2]:
        X[f"annual_sin_k{k}"] = np.sin(2*np.pi*k*hoy/8760)
        X[f"annual_cos_k{k}"] = np.cos(2*np.pi*k*hoy/8760)

    X["season"]     = X["month"].map({12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4})
    X["is_weekend"] = (X.index.dayofweek >= 5).astype(int)
    X["is_holiday"] = X.index.normalize().map(lambda d: int(d in SE_HOLIDAYS))
    X["is_peak"]    = (X["hour"].isin(range(7,10))|X["hour"].isin(range(17,21))).astype(int)
    X["is_night"]   = X["hour"].isin([23,0,1,2,3,4,5]).astype(int)

    X["lag24_x_hour_sin"] = X["price_lag_24h"] * X["hour_sin"]
    X["lag24_x_hour_cos"] = X["price_lag_24h"] * X["hour_cos"]

    # ── Weather ───────────────────────────────────────────────────────────────
    wind_col = WIND_COL if WIND_COL in X.columns else "windspeed_10m"
    w = X[wind_col]

    X["heating_degree"] = (15 - X["temperature"]).clip(lower=0)
    if "cloudcover" in X.columns and "solar_radiation" in X.columns:
        X["cloud_x_solar"] = X["cloudcover"] * X["solar_radiation"]

    # Wind surprise: how much windier than typical at this hour?
    wc = X[wind_col]
    X["wind_7d_mean"] = (
        wc.shift(24)+wc.shift(48)+wc.shift(72)+wc.shift(96)+
        wc.shift(120)+wc.shift(144)+wc.shift(168)
    ) / 7
    X["wind_surprise"]          = w - X["wind_7d_mean"]
    X["wind_surprise_x_night"]  = X["wind_surprise"] * X["is_night"]
    X["wind_vs_ewa"]            = X["wind_surprise"] * X["price_ewa"]

    X["wind_x_night"]   = w * X["is_night"]
    X["wind_x_peak"]    = w * X["is_peak"]
    X["wind_x_weekend"] = w * X["is_weekend"]
    X["wind_x_winter"]  = w * (X["season"]==1).astype(int)
    X["wind_x_summer"]  = w * (X["season"]==3).astype(int)
    X["wind_squared"]   = w ** 2

    X["temp_x_night"]  = X["temperature"] * X["is_night"]
    X["temp_x_peak"]   = X["temperature"] * X["is_peak"]
    X["temp_x_winter"] = X["temperature"] * (X["season"]==1).astype(int)

    # ── Generation (all lagged ≥ 24h) ─────────────────────────────────────────
    if "wind_gen_mw" in X.columns:
        X["wind_gen_lag24"]     = X["wind_gen_mw"].shift(24)
        X["load_lag24"]         = X["load_mw"].shift(24)
        X["wind_load_ratio"]    = X["wind_gen_lag24"] / (X["load_lag24"] + 1)
        X["wind_gen_x_night"]   = X["wind_gen_lag24"] * X["is_night"]
        X["wind_gen_x_weekend"] = X["wind_gen_lag24"] * X["is_weekend"]
        X["load_residual"]      = (X["load_lag24"] - X["wind_gen_lag24"]).clip(lower=0)

    # ── Nuclear (lagged only — no same-hour leakage) ──────────────────────────
    if "nuclear_gen_mw" in X.columns and X["nuclear_gen_mw"].notna().any():
        X["nuclear_gen_lag24"]  = X["nuclear_gen_mw"].shift(24)
        X["nuclear_shortfall"]  = (
            X["nuclear_gen_lag24"].rolling(168).max() - X["nuclear_gen_lag24"]
        )
        X["nuclear_x_peak"]     = X["nuclear_gen_lag24"] * X["is_peak"]
        X["nuclear_ramp_24h"]   = X["nuclear_gen_lag24"].diff(24)

    # ── Cross-border flows (lagged only) ──────────────────────────────────────
    raw_flow_cols = [c for c in X.columns
                     if c.startswith("flow_") and c.endswith("_mw")]
    for fc in raw_flow_cols:
        X[f"{fc}_lag24"]  = X[fc].shift(24)
        X[f"{fc}_lag168"] = X[fc].shift(168)

    if "net_position_mw" in X.columns:
        X["net_pos_lag24"]  = X["net_position_mw"].shift(24)
        X["net_pos_roll7d"] = X["net_position_mw"].shift(24).rolling(168).mean()
        if "flow_se4_mw_lag24" in X.columns:
            X["se4_x_peak"] = X["flow_se4_mw_lag24"] * X["is_peak"]

    # Drop raw nuclear and flow columns — keep only lags
    drop = raw_flow_cols + ["net_position_mw"]
    if "nuclear_gen_mw" in X.columns:
        drop.append("nuclear_gen_mw")
    X = X.drop(columns=[c for c in drop if c in X.columns])

    return X


def make_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered feature column list from a feature-engineered DataFrame.
    Excludes the target column and any intermediate raw columns.
    """
    wind_col = WIND_COL if WIND_COL in df.columns else "windspeed_10m"

    price_f = [
        "price_ewa", "price_lag24_vs_ewa",
        "price_3d_mean", "price_lag24_vs_3d",
        "price_lag_48h", "price_lag_168h",
        "price_roll_168h_mean", "price_roll_168h_std",
        "price_roll_720h_mean", "regime_x_vol",
    ]
    cal_f = [
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "season", "is_night", "is_peak", "is_weekend", "is_holiday",
        "lag24_x_hour_sin", "lag24_x_hour_cos",
    ]
    cal_f += [f"daily_{t}_k{k}"  for k in [1,2,3] for t in ["sin","cos"]]
    cal_f += [f"weekly_{t}_k{k}" for k in [1,2,3] for t in ["sin","cos"]]
    cal_f += [f"annual_{t}_k{k}" for k in [1,2]   for t in ["sin","cos"]]

    weather_f = [
        "temperature", wind_col, "cloudcover", "heating_degree",
        "wind_7d_mean", "wind_surprise", "wind_surprise_x_night", "wind_vs_ewa",
        "wind_x_night", "wind_x_peak", "wind_x_weekend", "wind_x_winter", "wind_x_summer",
        "wind_squared",
        "temp_x_night", "temp_x_peak", "temp_x_winter",
    ]
    if "cloud_x_solar" in df.columns:
        weather_f.append("cloud_x_solar")

    gen_f = [
        "wind_gen_lag24", "load_lag24", "wind_load_ratio",
        "wind_gen_x_night", "wind_gen_x_weekend", "load_residual",
    ]
    nuclear_f = [c for c in df.columns if c.startswith("nuclear")]
    flow_f    = [c for c in df.columns
                 if ("flow_" in c and ("lag24" in c or "lag168" in c))
                 or c in ["net_pos_lag24", "net_pos_roll7d", "se4_x_peak"]]

    all_f = price_f + cal_f + weather_f + gen_f + nuclear_f + flow_f
    seen  = set()
    result = []
    for c in all_f:
        if c in df.columns and c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ── Feature neutralization (V2) ────────────────────────────────────────────────

def fit_neutralizers(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    time_basis: list[str],
) -> dict[str, Ridge]:
    """
    Fit one Ridge regression per feature to neutralize.
    Must be fitted on training data ONLY to avoid leakage.
    """
    neutralizers = {}
    feats_to_neut = [f for f in NEUTRALIZE_FEATS if f in feature_cols]
    basis = [c for c in time_basis if c in df_train.columns]

    for feat in feats_to_neut:
        if feat not in df_train.columns:
            continue
        X_basis = df_train[basis].fillna(0)
        y_feat  = df_train[feat].fillna(0)
        lr = Ridge(alpha=1.0).fit(X_basis, y_feat)
        neutralizers[feat] = lr

    log.info("Fitted %d feature neutralizers", len(neutralizers))
    return neutralizers


def apply_neutralizers(
    df: pd.DataFrame,
    neutralizers: dict[str, Ridge],
    time_basis: list[str],
) -> pd.DataFrame:
    """Apply pre-fitted neutralizers to any DataFrame (train or test)."""
    df = df.copy()
    basis = [c for c in time_basis if c in df.columns]

    for feat, lr in neutralizers.items():
        if feat not in df.columns:
            continue
        explained = lr.predict(df[basis].fillna(0))
        df[feat]  = df[feat] - explained

    # Rebuild interaction features that depend on neutralized inputs
    wind_col = WIND_COL if WIND_COL in df.columns else "windspeed_10m"
    if wind_col in df.columns:
        w = df[wind_col]
        if "is_night" in df.columns:
            df["wind_x_night"]    = w * df["is_night"]
            df["wind_x_peak"]     = w * df["is_peak"]
            df["wind_x_weekend"]  = w * df["is_weekend"]
        if "season" in df.columns:
            df["wind_x_winter"]   = w * (df["season"]==1).astype(int)
            df["wind_x_summer"]   = w * (df["season"]==3).astype(int)
        df["wind_squared"] = w ** 2

    if "wind_surprise" in df.columns and "is_night" in df.columns:
        df["wind_surprise_x_night"] = df["wind_surprise"] * df["is_night"]
    if "wind_surprise" in df.columns and "price_ewa" in df.columns:
        df["wind_vs_ewa"] = df["wind_surprise"] * df["price_ewa"]
    if "temperature" in df.columns and "is_night" in df.columns:
        df["temp_x_night"]  = df["temperature"] * df["is_night"]
        df["temp_x_peak"]   = df["temperature"] * df["is_peak"]
        if "season" in df.columns:
            df["temp_x_winter"] = df["temperature"] * (df["season"]==1).astype(int)

    return df


# ── Train ──────────────────────────────────────────────────────────────────────

def train(data: dict) -> dict:
    """
    Full training pipeline:
      1. Merge raw data
      2. Build features
      3. V2 feature neutralization
      4. V3 target neutralization (Ridge linear baseline)
      5. Train three LightGBM quantile models (p05, p50, p95)
      6. Return artifacts dict

    Parameters
    ----------
    data : dict with keys prices, weather, gen, nuclear_gen, flows_df
    """
    prices      = data["prices"]
    weather     = data["weather"]
    gen         = data["gen"]
    nuclear_gen = data.get("nuclear_gen", pd.Series(dtype=float))
    flows_df    = data.get("flows_df", pd.DataFrame())

    # ── Merge ────────────────────────────────────────────────────────────────
    df = prices.join(weather, how="inner").join(gen, how="left")
    if not nuclear_gen.empty:
        df["nuclear_gen_mw"] = nuclear_gen.reindex(df.index).ffill(limit=3)
    if not flows_df.empty:
        df = df.join(flows_df.reindex(df.index).ffill(limit=3))

    for col in ["wind_gen_mw", "load_mw"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            "Europe/Stockholm", ambiguous="infer", nonexistent="shift_forward"
        )
    df = df.dropna(subset=["price_eur_mwh", "temperature", "windspeed_10m"])

    # ── Build features ───────────────────────────────────────────────────────
    log.info("Building features...")
    df = build_features(df)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["price_roll_720h_mean", "price_roll_168h_mean", "price_lag_168h"])
    log.info("Dataset: %d rows, %d columns", len(df), len(df.columns))

    feature_cols = make_feature_cols(df)
    target_col   = "price_eur_mwh"

    # ── Train / test split ───────────────────────────────────────────────────
    split_date = df.index.max() - pd.Timedelta(days=90)
    train_df   = df[df.index <= split_date]
    test_df    = df[df.index >  split_date]

    trivial = [c for c in TRIVIAL_COLS if c in feature_cols]
    time_basis = [c for c in TRIVIAL_COLS if c in df.columns]

    # ── V2: feature neutralization ───────────────────────────────────────────
    log.info("V2: fitting feature neutralizers on training data...")
    neutralizers = fit_neutralizers(train_df, feature_cols, time_basis)
    df           = apply_neutralizers(df, neutralizers, time_basis)
    train_df     = df[df.index <= split_date]
    test_df      = df[df.index >  split_date]
    feature_cols = make_feature_cols(df)   # rebuild after neutralization

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]
    y_test  = test_df[target_col]

    # ── V3: linear baseline on trivial features ───────────────────────────────
    trivial = [c for c in TRIVIAL_COLS if c in feature_cols]
    log.info("V3: fitting Ridge linear baseline on %d trivial features...", len(trivial))
    linear_baseline = Ridge(alpha=1.0)
    linear_baseline.fit(X_train[trivial], y_train)

    bl_train = linear_baseline.predict(X_train[trivial])
    bl_test  = linear_baseline.predict(X_test[trivial])
    bl_r2    = 1 - np.var(y_train - bl_train) / np.var(y_train)
    log.info("  Linear baseline R²=%.3f  test MAE=%.2f EUR/MWh",
             bl_r2, mean_absolute_error(y_test, bl_test))

    y_train_resid = y_train - bl_train

    # ── LightGBM on residuals ────────────────────────────────────────────────
    val_n   = int(len(X_train) * 0.1)
    X_tr, X_val   = X_train.iloc[:-val_n], X_train.iloc[-val_n:]
    y_tr_r, y_val_r = y_train_resid.iloc[:-val_n], y_train_resid.iloc[-val_n:]
    cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    models = {}
    for alpha, name in [(0.05, "p05"), (0.50, "p50"), (0.95, "p95")]:
        log.info("Training LightGBM quantile=%.2f...", alpha)
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGBM_PARAMS)
        m.fit(X_tr, y_tr_r, eval_set=[(X_val, y_val_r)], callbacks=cbs)
        models[name] = m
        log.info("  %s: %d trees", name, m.best_iteration_)

    artifacts = {
        "models":          models,
        "linear_baseline": linear_baseline,
        "neutralizers":    neutralizers,
        "feature_cols":    feature_cols,
        "trivial_cols":    trivial,
        "time_basis":      time_basis,
        "target_col":      target_col,
        "wind_col":        WIND_COL if WIND_COL in df.columns else "windspeed_10m",
        "split_date":      str(split_date.date()),
        # Keep test data for evaluation without re-loading
        "_X_test":         X_test,
        "_y_test":         y_test,
        "_bl_test":        bl_test,
    }
    return artifacts


# ── Pretrain ──────────────────────────────────────────────────────────────────

def pretrain(data: dict, cutoff_days: int = PRETRAIN_CUTOFF_DAYS) -> dict:
    """
    Phase 1 — train on all data older than `cutoff_days`.
    Run once or on a slow monthly schedule.

    Produces the same artifacts dict as train(), plus 'pretrain_cutoff'.
    Save with save_pretrained() so posttrain() can load it.
    """
    prices      = data["prices"]
    weather     = data["weather"]
    gen         = data["gen"]
    nuclear_gen = data.get("nuclear_gen", pd.Series(dtype=float))
    flows_df    = data.get("flows_df", pd.DataFrame())

    df = prices.join(weather, how="inner").join(gen, how="left")
    if not nuclear_gen.empty:
        df["nuclear_gen_mw"] = nuclear_gen.reindex(df.index).ffill(limit=3)
    if not flows_df.empty:
        df = df.join(flows_df.reindex(df.index).ffill(limit=3))
    for col in ["wind_gen_mw", "load_mw"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            "Europe/Stockholm", ambiguous="infer", nonexistent="shift_forward"
        )
    df = df.dropna(subset=["price_eur_mwh", "temperature", "windspeed_10m"])

    cutoff = df.index.max() - pd.Timedelta(days=cutoff_days)
    df = df[df.index <= cutoff]
    log.info("Pretrain: using data up to %s (%d rows)", cutoff.date(), len(df))

    df = build_features(df)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["price_roll_720h_mean", "price_roll_168h_mean", "price_lag_168h"])
    feature_cols = make_feature_cols(df)
    target_col   = "price_eur_mwh"

    split_date = df.index.max() - pd.Timedelta(days=90)
    train_df   = df[df.index <= split_date]

    time_basis   = [c for c in TRIVIAL_COLS if c in df.columns]
    neutralizers = fit_neutralizers(train_df, feature_cols, time_basis)
    df           = apply_neutralizers(df, neutralizers, time_basis)
    train_df     = df[df.index <= split_date]
    feature_cols = make_feature_cols(df)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    trivial         = [c for c in TRIVIAL_COLS if c in feature_cols]
    linear_baseline = Ridge(alpha=1.0).fit(X_train[trivial], y_train)
    y_train_resid   = y_train - linear_baseline.predict(X_train[trivial])

    val_n           = int(len(X_train) * 0.1)
    X_tr, X_val     = X_train.iloc[:-val_n], X_train.iloc[-val_n:]
    y_tr_r, y_val_r = y_train_resid.iloc[:-val_n], y_train_resid.iloc[-val_n:]
    cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    models = {}
    for alpha, name in [(0.05, "p05"), (0.50, "p50"), (0.95, "p95")]:
        log.info("Pretraining LightGBM quantile=%.2f ...", alpha)
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGBM_PARAMS)
        m.fit(X_tr, y_tr_r, eval_set=[(X_val, y_val_r)], callbacks=cbs)
        models[name] = m
        log.info("  %s: best iteration %d", name, m.best_iteration_)

    return {
        "models":           models,
        "linear_baseline":  linear_baseline,
        "neutralizers":     neutralizers,
        "feature_cols":     feature_cols,
        "trivial_cols":     trivial,
        "time_basis":       time_basis,
        "target_col":       target_col,
        "wind_col":         WIND_COL if WIND_COL in df.columns else "windspeed_10m",
        "pretrain_cutoff":  str(cutoff.date()),
        "_X_test":  df[df.index > split_date][feature_cols],
        "_y_test":  df[df.index > split_date][target_col],
        "_bl_test": linear_baseline.predict(
            df[df.index > split_date][trivial]
        ),
    }


# ── Posttrain ─────────────────────────────────────────────────────────────────

def posttrain(
    data: dict,
    pretrained_artifacts: dict,
    window_days: int = POSTTRAIN_WINDOW_DAYS,
) -> dict:
    """
    Phase 2 — extend the pretrained model with recent data via init_model.
    Run daily. Completes in 2-5 minutes.

    What changes vs pretrained_artifacts:
      - LightGBM models: extended with POSTTRAIN_TREES new trees
      - Ridge baseline:  re-fit on the recent window (tracks regime shifts)
      - Neutralizers:    frozen (stable, derived from long historical data)
    """
    prices      = data["prices"]
    weather     = data["weather"]
    gen         = data["gen"]
    nuclear_gen = data.get("nuclear_gen", pd.Series(dtype=float))
    flows_df    = data.get("flows_df", pd.DataFrame())

    df = prices.join(weather, how="inner").join(gen, how="left")
    if not nuclear_gen.empty:
        df["nuclear_gen_mw"] = nuclear_gen.reindex(df.index).ffill(limit=3)
    if not flows_df.empty:
        df = df.join(flows_df.reindex(df.index).ffill(limit=3))
    for col in ["wind_gen_mw", "load_mw"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            "Europe/Stockholm", ambiguous="infer", nonexistent="shift_forward"
        )
    df = df.dropna(subset=["price_eur_mwh", "temperature", "windspeed_10m"])
    df = build_features(df)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["price_roll_720h_mean", "price_roll_168h_mean", "price_lag_168h"])

    neutralizers = pretrained_artifacts["neutralizers"]
    time_basis   = pretrained_artifacts["time_basis"]
    df           = apply_neutralizers(df, neutralizers, time_basis)

    feature_cols  = pretrained_artifacts["feature_cols"]
    available     = [c for c in feature_cols if c in df.columns]
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        log.warning("Posttrain: %d pretrain features missing: %s",
                    len(missing_feats), missing_feats[:5])

    window_start = df.index.max() - pd.Timedelta(days=window_days)
    recent_df    = df[df.index >= window_start].copy()
    log.info("Posttrain window: %s — %s (%d rows)",
             recent_df.index.min().date(), recent_df.index.max().date(), len(recent_df))

    if len(recent_df) < 48:
        raise ValueError(
            f"Posttrain window too small ({len(recent_df)} rows). "
            "Check that pipeline data sync ran successfully."
        )

    X_recent = recent_df[available]
    y_recent = recent_df["price_eur_mwh"]

    trivial       = pretrained_artifacts["trivial_cols"]
    trivial_avail = [c for c in trivial if c in X_recent.columns]

    linear_baseline = Ridge(alpha=1.0).fit(X_recent[trivial_avail], y_recent)
    bl_recent       = linear_baseline.predict(X_recent[trivial_avail])
    y_resid_recent  = y_recent - bl_recent

    val_n           = max(24, int(len(X_recent) * 0.15))
    X_tr, X_val     = X_recent.iloc[:-val_n], X_recent.iloc[-val_n:]
    y_tr_r, y_val_r = y_resid_recent.iloc[:-val_n], y_resid_recent.iloc[-val_n:]
    cbs = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]

    pretrained_models = pretrained_artifacts["models"]
    adapted_models    = {}

    for name in ["p05", "p50", "p95"]:
        alpha = {"p05": 0.05, "p50": 0.50, "p95": 0.95}[name]
        base_trees = pretrained_models[name].best_iteration_
        log.info("Posttraining %s (extending %d base trees) ...", name, base_trees)

        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **POSTTRAIN_LGBM_PARAMS,
        )
        m.fit(
            X_tr, y_tr_r,
            eval_set   = [(X_val, y_val_r)],
            callbacks  = cbs,
            init_model = pretrained_models[name],
        )
        adapted_models[name] = m
        log.info("  %s: added %d trees (total ~%d)",
                 name, m.best_iteration_, base_trees + m.best_iteration_)

    test_n    = min(val_n, len(recent_df) // 4)
    X_test_pt = X_recent.iloc[-test_n:]
    y_test_pt = y_recent.iloc[-test_n:]
    bl_test   = linear_baseline.predict(X_test_pt[trivial_avail])

    return {
        **pretrained_artifacts,
        "models":               adapted_models,
        "linear_baseline":      linear_baseline,
        "posttrain_date":       str(df.index.max().date()),
        "posttrain_window_days": window_days,
        "_X_test":  X_test_pt,
        "_y_test":  y_test_pt,
        "_bl_test": bl_test,
    }


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(artifacts: dict, X_test=None, y_test=None, bl_test=None) -> dict:
    """
    Compute test set metrics.
    Uses data embedded in artifacts if X_test / y_test not provided.
    """
    models = artifacts["models"]
    X_test  = X_test  if X_test  is not None else artifacts["_X_test"]
    y_test  = y_test  if y_test  is not None else artifacts["_y_test"]
    bl_test = bl_test if bl_test is not None else artifacts["_bl_test"]

    pred  = bl_test + models["p50"].predict(X_test)
    p05   = bl_test + models["p05"].predict(X_test)
    p95   = bl_test + models["p95"].predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mask = y_test.abs() > 10
    mape = (((y_test[mask] - pred[mask]).abs() / y_test[mask].abs()).mean()) * 100
    cov  = ((y_test >= p05) & (y_test <= p95)).mean() * 100

    night_m = y_test.index.hour.isin([23,0,1,2,3,4,5])
    peak_m  = y_test.index.hour.isin(list(range(7,10))+list(range(17,21)))
    spike_m = y_test > 100

    metrics = {
        "mae":          round(float(mae),  2),
        "rmse":         round(float(rmse), 2),
        "mape":         round(float(mape), 1),
        "coverage_q5_q95": round(float(cov), 1),
        "night_mae":    round(float(mean_absolute_error(y_test[night_m], pred[night_m])), 2),
        "peak_mae":     round(float(mean_absolute_error(y_test[peak_m],  pred[peak_m])),  2),
        "n_spikes":     int(spike_m.sum()),
        "spike_mae":    round(float(mean_absolute_error(
                            y_test[spike_m], pred[spike_m])), 2) if spike_m.sum() else None,
        "test_from":    str(y_test.index.min().date()),
        "test_to":      str(y_test.index.max().date()),
    }

    # MAE by hour (for dashboard chart)
    err = pd.Series(np.abs(pred - y_test.values), index=y_test.index)
    metrics["mae_by_hour"] = err.groupby(err.index.hour).mean().round(2).to_dict()

    log.info(
        "MAE=%.2f  RMSE=%.2f  MAPE=%.1f%%  Coverage=%.1f%%  "
        "Night=%.2f  Peak=%.2f",
        mae, rmse, mape, cov,
        metrics["night_mae"], metrics["peak_mae"],
    )
    return metrics


# ── Predict (live 24h forecast) ────────────────────────────────────────────────

def predict(
    artifacts: dict,
    prices: pd.DataFrame,
    weather_hist: pd.DataFrame,
    weather_fc: pd.DataFrame,
    gen: pd.DataFrame,
    nuclear_gen: pd.Series | None = None,
    flows_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a 24h forecast starting from the hour after the last known price.

    All features for future timestamps are constructed from:
    - Historical prices  → lag / rolling features
    - Historical weather → wind_7d_mean baseline
    - FORECAST weather   → actual wind/temp for the prediction period
    - Historical gen     → wind_gen / load lags
    - Historical nuclear → nuclear_gen_lag24
    - Historical flows   → flow lags

    Returns a DataFrame indexed by future timestamps with columns p05, p50, p95.
    """
    feature_cols    = artifacts["feature_cols"]
    trivial_cols    = artifacts["trivial_cols"]
    neutralizers    = artifacts["neutralizers"]
    time_basis      = artifacts["time_basis"]
    linear_baseline = artifacts["linear_baseline"]
    models          = artifacts["models"]
    wind_col        = artifacts["wind_col"]

    last_price_ts = prices.index.max()
    future_ts = pd.date_range(
        start   = last_price_ts + pd.Timedelta(hours=1),
        periods = 24,
        freq    = "h",
        tz      = "Europe/Stockholm",
    )
    log.info("Forecasting %s → %s", future_ts[0], future_ts[-1])

    price_s   = prices["price_eur_mwh"]
    wind_hist = weather_hist[wind_col] if wind_col in weather_hist.columns \
                else weather_hist.get("windspeed_10m", pd.Series(dtype=float))

    def _getp(ts):
        try:    return float(price_s.loc[ts])
        except: return float(price_s.iloc[price_s.index.get_indexer([ts], method="nearest")[0]])

    def _getw(ts):
        try:    return float(wind_hist.loc[ts])
        except: return float(wind_hist.iloc[wind_hist.index.get_indexer([ts], method="nearest")[0]])

    def _get_series(series, ts):
        if series is None or series.empty: return np.nan
        try:    return float(series.loc[ts])
        except: return float(series.iloc[series.index.get_indexer([ts], method="nearest")[0]])

    def _get_df_col(df, col, ts):
        if df is None or df.empty or col not in df.columns: return np.nan
        try:    return float(df.loc[ts, col])
        except: return float(df[col].iloc[df.index.get_indexer([ts], method="nearest")[0]])

    rows = []
    for ts in future_ts:
        r = {}

        # Calendar
        r["hour"]       = ts.hour
        r["dayofweek"]  = ts.dayofweek
        r["month"]      = ts.month
        r["hour_sin"]   = np.sin(2*np.pi*ts.hour/24)
        r["hour_cos"]   = np.cos(2*np.pi*ts.hour/24)
        r["month_sin"]  = np.sin(2*np.pi*ts.month/12)
        r["month_cos"]  = np.cos(2*np.pi*ts.month/12)
        r["dow_sin"]    = np.sin(2*np.pi*ts.dayofweek/7)
        r["dow_cos"]    = np.cos(2*np.pi*ts.dayofweek/7)
        r["season"]     = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[ts.month]
        r["is_weekend"] = int(ts.dayofweek >= 5)
        r["is_holiday"] = int(ts.normalize() in SE_HOLIDAYS)
        r["is_peak"]    = int(ts.hour in list(range(7,10))+list(range(17,21)))
        r["is_night"]   = int(ts.hour in [23,0,1,2,3,4,5])

        # Fourier
        _how = ts.dayofweek*24+ts.hour
        _hoy = (ts.timetuple().tm_yday-1)*24+ts.hour
        for k in [1,2,3]:
            r[f"daily_sin_k{k}"]  = np.sin(2*np.pi*k*ts.hour/24)
            r[f"daily_cos_k{k}"]  = np.cos(2*np.pi*k*ts.hour/24)
            r[f"weekly_sin_k{k}"] = np.sin(2*np.pi*k*_how/168)
            r[f"weekly_cos_k{k}"] = np.cos(2*np.pi*k*_how/168)
        for k in [1,2]:
            r[f"annual_sin_k{k}"] = np.sin(2*np.pi*k*_hoy/8760)
            r[f"annual_cos_k{k}"] = np.cos(2*np.pi*k*_hoy/8760)

        # Price lags from known history
        lags = {h: _getp(ts-pd.Timedelta(hours=h)) for h in [24,48,72,96,120,144,168]}
        r["price_lag_24h"]       = lags[24]
        r["price_lag_48h"]       = lags[48]
        r["price_lag_168h"]      = lags[168]
        r["price_ewa"]           = (lags[24]*0.35+lags[48]*0.25+lags[72]*0.17+
                                    lags[96]*0.11+lags[120]*0.06+lags[144]*0.04+lags[168]*0.02)
        r["price_lag24_vs_ewa"]  = lags[24] - r["price_ewa"]
        r["price_3d_mean"]       = np.nanmean([lags[24],lags[48],lags[72]])
        r["price_lag24_vs_3d"]   = lags[24] - r["price_3d_mean"]
        r["lag24_x_hour_sin"]    = lags[24] * r["hour_sin"]
        r["lag24_x_hour_cos"]    = lags[24] * r["hour_cos"]

        h168 = price_s[price_s.index < ts].tail(168)
        h720 = price_s[price_s.index < ts].tail(720)
        r["price_roll_168h_mean"] = float(h168.mean())
        r["price_roll_168h_std"]  = float(h168.std())
        r["price_roll_720h_mean"] = float(h720.mean())
        r["regime_x_vol"]         = r["price_roll_720h_mean"] * r["price_roll_168h_std"]

        # Weather from FORECAST API
        wfc_idx = weather_fc.index.get_indexer([ts], method="nearest")[0]
        wfc     = weather_fc.iloc[wfc_idx]
        r["temperature"]     = float(wfc["temperature"])
        r["windspeed_10m"]   = float(wfc["windspeed_10m"])
        r["windspeed_100m"]  = float(wfc.get("windspeed_100m", wfc["windspeed_10m"]))
        r["solar_radiation"] = float(wfc.get("solar_radiation", 0))
        r["cloudcover"]      = float(wfc.get("cloudcover", 0))
        r["heating_degree"]  = max(0, 15-r["temperature"])
        r["cloud_x_solar"]   = r["cloudcover"] * r["solar_radiation"]

        w = r[wind_col]
        wind_hist_vals = [_getw(ts-pd.Timedelta(hours=h)) for h in [24,48,72,96,120,144,168]]
        r["wind_7d_mean"]          = np.nanmean(wind_hist_vals)
        r["wind_surprise"]         = w - r["wind_7d_mean"]
        r["wind_surprise_x_night"] = r["wind_surprise"] * r["is_night"]
        r["wind_vs_ewa"]           = r["wind_surprise"] * r["price_ewa"]
        r["wind_x_night"]          = w * r["is_night"]
        r["wind_x_peak"]           = w * r["is_peak"]
        r["wind_x_weekend"]        = w * r["is_weekend"]
        r["wind_x_winter"]         = w * int(r["season"]==1)
        r["wind_x_summer"]         = w * int(r["season"]==3)
        r["wind_squared"]          = w**2
        r["temp_x_night"]          = r["temperature"] * r["is_night"]
        r["temp_x_peak"]           = r["temperature"] * r["is_peak"]
        r["temp_x_winter"]         = r["temperature"] * int(r["season"]==1)

        # Generation lags
        lag_ts = ts - pd.Timedelta(hours=24)
        wg = _get_df_col(gen, "wind_gen_mw", lag_ts)
        ld = _get_df_col(gen, "load_mw",     lag_ts)
        r["wind_gen_lag24"]     = wg
        r["load_lag24"]         = ld
        r["wind_load_ratio"]    = wg/(ld+1)     if not np.isnan(wg) else np.nan
        r["wind_gen_x_night"]   = wg*r["is_night"]   if not np.isnan(wg) else np.nan
        r["wind_gen_x_weekend"] = wg*r["is_weekend"] if not np.isnan(wg) else np.nan
        r["load_residual"]      = max(0,ld-wg) if not (np.isnan(ld) or np.isnan(wg)) else np.nan

        # Nuclear lag
        if nuclear_gen is not None and not nuclear_gen.empty:
            nuc = _get_series(nuclear_gen, lag_ts)
            r["nuclear_gen_lag24"] = nuc
            recent_nuc = nuclear_gen[nuclear_gen.index < ts].tail(168)
            r["nuclear_shortfall"] = float(recent_nuc.max() - nuc) if len(recent_nuc) else np.nan
            r["nuclear_x_peak"]    = nuc * r["is_peak"]
            nuc_48 = _get_series(nuclear_gen, ts-pd.Timedelta(hours=48))
            r["nuclear_ramp_24h"]  = nuc - nuc_48

        # Flow lags
        if flows_df is not None and not flows_df.empty:
            for col in flows_df.columns:
                if col == "net_position_mw": continue
                r[f"{col}_lag24"]  = _get_df_col(flows_df, col, lag_ts)
                r[f"{col}_lag168"] = _get_df_col(flows_df, col, ts-pd.Timedelta(hours=168))
            if "net_position_mw" in flows_df.columns:
                r["net_pos_lag24"] = _get_df_col(flows_df, "net_position_mw", lag_ts)
                recent_net = flows_df["net_position_mw"][flows_df.index < ts].tail(168)
                r["net_pos_roll7d"] = float(recent_net.mean()) if len(recent_net) else np.nan
            if "flow_se4_mw_lag24" in feature_cols and "flow_se4_mw_lag24" in r:
                r["se4_x_peak"] = r["flow_se4_mw_lag24"] * r["is_peak"]

        rows.append(r)

    fc_df = pd.DataFrame(rows, index=future_ts)

    # Apply feature neutralization (same shifts as training)
    fc_df = apply_neutralizers(fc_df, neutralizers, time_basis)

    fc_cols = [c for c in feature_cols if c in fc_df.columns]
    missing = [c for c in feature_cols if c not in fc_df.columns]
    if missing:
        log.warning("Missing %d features from forecast input: %s", len(missing), missing[:5])

    trivial_fc = [c for c in trivial_cols if c in fc_df.columns]
    bl_fc      = linear_baseline.predict(fc_df[trivial_fc])

    forecast = pd.DataFrame({
        "p05": bl_fc + models["p05"].predict(fc_df[fc_cols]),
        "p50": bl_fc + models["p50"].predict(fc_df[fc_cols]),
        "p95": bl_fc + models["p95"].predict(fc_df[fc_cols]),
    }, index=future_ts)

    return forecast


# ── Save / load pretrained artifacts ─────────────────────────────────────────

def save_pretrained(artifacts: dict, path: str | Path = MODEL_DIR) -> None:
    """
    Save Phase 1 base model to pretrained_models.pkl.
    Kept separate from models.pkl so posttrain() can always find the frozen base.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in artifacts.items() if not k.startswith("_")}

    with open(path / "pretrained_models.pkl",  "wb") as f:
        pickle.dump(save["models"], f)
    with open(path / "linear_baseline.pkl", "wb") as f:
        pickle.dump(save["linear_baseline"], f)
    with open(path / "neutralizers.pkl",    "wb") as f:
        pickle.dump(save["neutralizers"], f)

    config = {k: v for k, v in save.items()
              if k not in ("models", "linear_baseline", "neutralizers")}
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Pretrained artifacts saved to %s (%d features)",
             path, len(save["feature_cols"]))


def load_pretrained(path: str | Path = MODEL_DIR) -> dict:
    """Load Phase 1 frozen base for use in posttrain()."""
    path = Path(path)
    pretrained_pkl = path / "pretrained_models.pkl"
    if not pretrained_pkl.exists():
        raise FileNotFoundError(
            f"No pretrained model found at {pretrained_pkl}. "
            "Run POST /pretrain or `python ml.py --pretrain` first."
        )
    with open(pretrained_pkl,               "rb") as f: models          = pickle.load(f)
    with open(path / "linear_baseline.pkl", "rb") as f: linear_baseline = pickle.load(f)
    with open(path / "neutralizers.pkl",    "rb") as f: neutralizers    = pickle.load(f)
    with open(path / "config.json")              as f: config           = json.load(f)
    return {
        "models":          models,
        "linear_baseline": linear_baseline,
        "neutralizers":    neutralizers,
        **config,
    }


# ── Save / load artifacts ──────────────────────────────────────────────────────

def save_artifacts(artifacts: dict, path: str | Path = MODEL_DIR) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Strip large test data before saving
    save = {k: v for k, v in artifacts.items() if not k.startswith("_")}

    with open(path / "models.pkl",          "wb") as f: pickle.dump(save["models"], f)
    with open(path / "linear_baseline.pkl", "wb") as f: pickle.dump(save["linear_baseline"], f)
    with open(path / "neutralizers.pkl",    "wb") as f: pickle.dump(save["neutralizers"], f)

    config = {k: v for k, v in save.items()
              if k not in ("models", "linear_baseline", "neutralizers")}
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Artifacts saved to %s  (%d features)", path, len(save["feature_cols"]))


def load_artifacts(path: str | Path = MODEL_DIR) -> dict:
    path = Path(path)
    with open(path / "models.pkl",          "rb") as f: models          = pickle.load(f)
    with open(path / "linear_baseline.pkl", "rb") as f: linear_baseline = pickle.load(f)
    with open(path / "neutralizers.pkl",    "rb") as f: neutralizers    = pickle.load(f)
    with open(path / "config.json")                as f: config          = json.load(f)

    artifacts = {
        "models":          models,
        "linear_baseline": linear_baseline,
        "neutralizers":    neutralizers,
        **config,
    }
    log.info("Artifacts loaded from %s  (%d features)", path, len(artifacts["feature_cols"]))
    return artifacts


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="SE3 ML module")
    parser.add_argument("--train",    action="store_true", help="Retrain and save model")
    parser.add_argument("--pretrain",  action="store_true",
                        help="Phase 1: train on historical data, save pretrained_models.pkl")
    parser.add_argument("--posttrain", action="store_true",
                        help="Phase 2: extend pretrained model on recent 90 days")
    parser.add_argument("--forecast", action="store_true", help="Run live 24h forecast")
    parser.add_argument("--evaluate", action="store_true", help="Print test set metrics")
    parser.add_argument("--api-key",  default=None)
    args = parser.parse_args()

    from pipeline import sync_all, fetch_weather_forecast

    api_key = args.api_key or os.environ.get("ENTSOE_API_KEY", "")

    if args.train:
        data      = sync_all(api_key)
        artifacts = train(data)
        metrics   = evaluate(artifacts)
        print("\nTest metrics:")
        for k, v in metrics.items():
            if k != "mae_by_hour":
                print(f"  {k:<22}: {v}")
        save_artifacts(artifacts)
        # Save metrics so api.py /metrics endpoint can serve them
        import json
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Metrics saved to %s/metrics.json", MODEL_DIR)

        # Save test set predictions to DB so backtesting page has 90 days of data
        log.info("Saving test set predictions to forecasts table...")
        from pipeline import save_forecast
        X_test_  = artifacts["_X_test"]
        bl_test_ = artifacts["_bl_test"]
        test_fc  = pd.DataFrame({
            "p05": bl_test_ + artifacts["models"]["p05"].predict(X_test_),
            "p50": bl_test_ + artifacts["models"]["p50"].predict(X_test_),
            "p95": bl_test_ + artifacts["models"]["p95"].predict(X_test_),
        }, index=X_test_.index)
        save_forecast(test_fc)
        log.info("Saved %d test set forecast hours to DB", len(test_fc))

    elif args.forecast:
        from pipeline import save_forecast
        data       = sync_all(api_key)
        artifacts  = load_artifacts()
        weather_fc = fetch_weather_forecast()
        forecast   = predict(
            artifacts,
            prices      = data["prices"],
            weather_hist= data["weather"],
            weather_fc  = weather_fc,
            gen         = data["gen"],
            nuclear_gen = data.get("nuclear_gen"),
            flows_df    = data.get("flows_df"),
        )
        save_forecast(forecast)
        log.info("Forecast saved to DB (%d hours)", len(forecast))
        print("\nNext 24h SE3 price forecast (EUR/MWh):")
        print(forecast.round(1).to_string())

    elif args.evaluate:
        data      = sync_all(api_key)
        artifacts = train(data)
        metrics   = evaluate(artifacts)
        for k, v in metrics.items():
            if k != "mae_by_hour":
                print(f"  {k:<22}: {v}")

    elif args.pretrain:
        data      = sync_all(api_key)
        artifacts = pretrain(data)
        metrics   = evaluate(artifacts)
        print("\nPretrain test metrics:")
        for k, v in metrics.items():
            if k != "mae_by_hour":
                print(f"  {k:<22}: {v}")
        save_pretrained(artifacts)
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Pretrain complete. Cutoff: %s", artifacts["pretrain_cutoff"])

    elif args.posttrain:
        data           = sync_all(api_key)
        pretrained     = load_pretrained()
        artifacts      = posttrain(data, pretrained)
        metrics        = evaluate(artifacts)
        print("\nPosttrain test metrics (recent window):")
        for k, v in metrics.items():
            if k != "mae_by_hour":
                print(f"  {k:<22}: {v}")
        save_artifacts(artifacts, MODEL_DIR)
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Posttrain complete. Date: %s", artifacts["posttrain_date"])

    else:
        parser.print_help()
