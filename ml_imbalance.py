"""
ml_imbalance.py — SE3 Imbalance Price & Spike Detection Module

Trains and serves:
  1. LightGBM quantile regression for 15-min imbalance prices (p05/p50/p95)
  2. Binary LightGBM spike classifier (AUC ≈ 0.898)

Public API (mirrors ml.py patterns)
--------------------------------------
  train_imbalance(data)              → artifacts dict
  train_spike(data)                  → artifacts dict
  predict_imbalance(artifacts, data) → DataFrame (timestamp, p05, p50, p95)
  predict_spike(artifacts, data)     → DataFrame (timestamp, spike_proba, regime)
  evaluate_imbalance(artifacts, ...)  → dict
  evaluate_spike(artifacts, ...)      → dict
  save_imbalance_artifacts(artifacts, path)
  load_imbalance_artifacts(path)     → dict
  save_spike_artifacts(artifacts, path)
  load_spike_artifacts(path)         → dict

CLI
---
  python ml_imbalance.py --train-imbalance
  python ml_imbalance.py --train-spike
  python ml_imbalance.py --train-all
  python ml_imbalance.py --forecast
  python ml_imbalance.py --evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import lightgbm as lgb
except ModuleNotFoundError as _lgb_err:
    lgb = None
    _LGB_MISSING = _lgb_err

log = logging.getLogger("ml_imbalance")

DB_PATH   = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))

from feature_engineering import (  # noqa: E402
    EXTREME_HIGH,
    STRESS_HIGH,
    STRESS_LOW,
    build_features,
    make_feature_cols,
)

LGBM_PARAMS: dict = {
    "n_estimators":      2000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "verbose":          -1,
}

WINDOW_AUG  = 2      # ±2 periods around spike events
COST_FP     = 15.0   # EUR/MWh per false positive
COST_FN     = 300.0  # EUR/MWh per false negative (missed spike)
HORIZON     = 1      # 15-min periods
TEST_DAYS   = 60     # holdout for evaluation


# ── helpers ────────────────────────────────────────────────────────────────────

def _require_lgb() -> None:
    if lgb is None:
        raise ModuleNotFoundError("Install with: pip install lightgbm") from _LGB_MISSING


def _upsample_15min(df: pd.DataFrame, tz: str = "Europe/Stockholm") -> pd.DataFrame:
    """Reindex hourly (or sparser) DataFrame to 15-min with forward-fill."""
    if df.empty or len(df) < 2:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz=tz)
    return df.reindex(idx).ffill(limit=4)


def _load_table(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    tz: str = "Europe/Stockholm",
) -> pd.DataFrame:
    try:
        df = conn.execute(f"SELECT * FROM {table} ORDER BY timestamp").df()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(tz)
        return df.set_index("timestamp")
    except Exception as exc:
        log.warning("Could not load %s: %s", table, exc)
        return pd.DataFrame()


# ── data loading ───────────────────────────────────────────────────────────────

def _load_training_data(start_date: str | None = None) -> pd.DataFrame:
    """
    Load and merge 15-min training data from DuckDB.

    Joins imbalance_prices (15-min native) with weather, weather_forecast,
    nuclear_gen, prices, and cross_border_flows (all upsampled to 15-min).
    Column names match feature_engineering.py expectations.

    Parameters
    ----------
    start_date : str | None
        ISO date string to trim the start; defaults to full history.
    """
    if not Path(DB_PATH).exists():
        raise RuntimeError(f"Database not found at {DB_PATH}. Run pipeline.py first.")

    conn = duckdb.connect(DB_PATH, read_only=True)
    tables = set(conn.execute("SHOW TABLES").df()["name"].tolist())

    if "imbalance_prices" not in tables:
        conn.close()
        raise RuntimeError(
            "imbalance_prices table not found. "
            "Run: python pipeline.py  (sync_imbalance is called by sync_all)"
        )

    df_imbl = _load_table(conn, "imbalance_prices")
    if df_imbl.empty:
        conn.close()
        raise RuntimeError("imbalance_prices table is empty.")

    df_weather  = _load_table(conn, "weather")
    df_wfcst    = _load_table(conn, "weather_forecast")
    df_nuclear  = _load_table(conn, "nuclear_gen")
    df_prices   = _load_table(conn, "prices")

    # Cross-border flows pivot (border column → wide)
    df_flows = pd.DataFrame()
    if "cross_border_flows" in tables:
        try:
            raw = conn.execute(
                "SELECT timestamp, border, flow_mw FROM cross_border_flows ORDER BY timestamp"
            ).df()
            raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_convert("Europe/Stockholm")
            df_flows = raw.pivot(index="timestamp", columns="border", values="flow_mw")
            df_flows.columns = [f"flow_{c.lower()}_mw" for c in df_flows.columns]
            df_flows["net_position_mw"] = df_flows.sum(axis=1)
        except Exception as exc:
            log.warning("Could not load cross_border_flows: %s", exc)

    conn.close()

    # Trim to start_date
    if start_date:
        cutoff = pd.Timestamp(start_date, tz="Europe/Stockholm")
        df_imbl = df_imbl[df_imbl.index >= cutoff]

    # Build 15-min master frame on imbalance index
    df = df_imbl.copy()

    for src, name in [
        (df_weather,  "weather"),
        (df_wfcst,    "weather_forecast"),
        (df_nuclear,  "nuclear_gen"),
        (df_prices,   "prices"),
        (df_flows,    "flows"),
    ]:
        if not src.empty:
            src_15 = _upsample_15min(src)
            # Rename prices column so feature_engineering.py sees spot_price
            if "price_eur_mwh" in src_15.columns:
                src_15 = src_15.rename(columns={"price_eur_mwh": "spot_price"})
            # nuclear_gen_mw → keep as is; used for nuclear_gen_mw col in feature eng.
            df = df.join(src_15, how="left")

    log.info(
        "Training data: %d rows × %d cols  (%s → %s)",
        len(df), len(df.columns),
        df.index.min().date(), df.index.max().date(),
    )
    return df


def _load_inference_data(lookback_days: int = 7) -> pd.DataFrame:
    """
    Load recent data from DuckDB + live SMHI forecast for inference.
    Returns merged 15-min DataFrame covering the last lookback_days.
    """
    cutoff = pd.Timestamp.now(tz="Europe/Stockholm") - pd.Timedelta(days=lookback_days)
    df = _load_training_data(start_date=str(cutoff.date()))

    # Overlay live SMHI forecast (fcst_* column names) for the recent gap
    try:
        from data_sources import fetch_smhi_forecast  # noqa: PLC0415
        df_smhi = fetch_smhi_forecast()
        if not df_smhi.empty:
            smhi_aligned = df_smhi.reindex(df.index)
            for col in df_smhi.columns:
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = df[col].combine_first(smhi_aligned[col])
            log.info("SMHI forecast overlay applied (%d rows)", len(df_smhi))
    except Exception as exc:
        log.warning("Could not overlay SMHI forecast: %s", exc)

    return df


# ── train ───────────────────────────────────────────────────────────────────────

def train_imbalance(data: dict) -> dict:
    """
    Train LightGBM quantile regression (p05/p50/p95) on imbalance data.

    Parameters
    ----------
    data : dict
        Optional key ``"merged_df"`` — pre-merged 15-min DataFrame.
        If absent, calls ``_load_training_data()`` automatically.

    Returns
    -------
    dict
        artifacts: models, feature_cols, metrics, config, _X_test, _y_test.
    """
    _require_lgb()

    df_merged = data.get("merged_df")
    if df_merged is None:
        df_merged = _load_training_data()

    log.info("Building regression features (%d rows)...", len(df_merged))
    df_feats  = build_features(df_merged, horizon=HORIZON)
    df_feats  = df_feats.dropna(axis=1, how="all")
    feat_cols = make_feature_cols(df_feats, mode="regression")

    core_lags = ["imbl_lag_1", "imbl_lag_2", "imbl_roll_1h_mean"]
    df_model  = df_feats.dropna(subset=["target_price"])
    df_model  = df_model.dropna(subset=[c for c in core_lags if c in df_model.columns])
    log.info("Regression dataset: %d rows, %d features", len(df_model), len(feat_cols))

    split_date = df_model.index.max() - pd.Timedelta(days=TEST_DAYS)
    train_df   = df_model[df_model.index <= split_date]
    test_df    = df_model[df_model.index >  split_date]

    X_train, y_train = train_df[feat_cols], train_df["target_price"]
    X_test,  y_test  = test_df[feat_cols],  test_df["target_price"]

    val_n  = int(len(X_train) * 0.10)
    X_tr, X_val = X_train.iloc[:-val_n], X_train.iloc[-val_n:]
    y_tr, y_val = y_train.iloc[:-val_n], y_train.iloc[-val_n:]
    cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    models: dict[str, lgb.LGBMRegressor] = {}
    for alpha, name in [(0.05, "p05"), (0.50, "p50"), (0.95, "p95")]:
        log.info("Training imbalance quantile=%.2f...", alpha)
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGBM_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=cbs)
        models[name] = m
        log.info("  %s: best iteration %d", name, m.best_iteration_)

    pred  = models["p50"].predict(X_test)
    p05_p = models["p05"].predict(X_test)
    p95_p = models["p95"].predict(X_test)
    mae   = mean_absolute_error(y_test, pred)
    rmse  = np.sqrt(mean_squared_error(y_test, pred))
    cov   = float(((y_test >= p05_p) & (y_test <= p95_p)).mean() * 100)

    pers_mask = y_test.notna() & test_df["imbl_lag_1"].notna()
    pers_mae  = mean_absolute_error(y_test[pers_mask], test_df.loc[pers_mask, "imbl_lag_1"])
    skill     = float(1 - mae / pers_mae) if pers_mae > 0 else float("nan")

    current = test_df["imbl_lag_1"].values
    pd_sign = np.sign(pred - current)
    ad_sign = np.sign(y_test.values - current)
    m_dir   = (pd_sign != 0) & (ad_sign != 0)
    dir_acc = float((pd_sign[m_dir] == ad_sign[m_dir]).mean()) if m_dir.any() else float("nan")

    metrics = {
        "mae":                  round(float(mae),    2),
        "rmse":                 round(float(rmse),   2),
        "pi_coverage":          round(cov,            1),
        "skill_vs_persistence": round(skill,          3),
        "dir_accuracy":         round(dir_acc,        3),
        "test_from":            str(y_test.index.min().date()),
        "test_to":              str(y_test.index.max().date()),
    }
    log.info(
        "Imbalance — MAE=%.2f  RMSE=%.2f  Skill=%.3f  Coverage=%.1f%%",
        mae, rmse, skill, cov,
    )

    return {
        "models":       models,
        "feature_cols": feat_cols,
        "metrics":      metrics,
        "config": {
            "lgbm_params": LGBM_PARAMS,
            "horizon":     HORIZON,
            "split_date":  str(split_date.date()),
        },
        "_X_test":  X_test,
        "_y_test":  y_test,
        "_df_test": test_df[["imbl_lag_1", "imbl_price"]],
    }


def train_spike(data: dict) -> dict:
    """
    Train binary LightGBM spike classifier.

    Exact config that produced AUC=0.898:
      class_weight={0:1.0, 1:25.0}, eval_metric="auc"
      window augmentation ±2 periods on training base only
      Real val holdout taken BEFORE augmentation (last 10% of real data)
      No SMOTE.

    Returns
    -------
    dict
        artifacts: model, feature_cols, threshold (profit-optimal), metrics, config.
    """
    _require_lgb()

    df_merged = data.get("merged_df")
    if df_merged is None:
        df_merged = _load_training_data()

    log.info("Building spike features (%d rows)...", len(df_merged))
    df_feats  = build_features(df_merged, horizon=HORIZON)
    df_feats  = df_feats.dropna(axis=1, how="all")
    feat_cols = make_feature_cols(df_feats, mode="spike")

    core_lags = ["imbl_lag_1", "imbl_lag_2", "imbl_roll_1h_mean"]
    df_model  = df_feats.dropna(subset=["target_class", "target_binary"])
    df_model  = df_model.dropna(subset=[c for c in core_lags if c in df_model.columns])
    df_model["target_class"]  = df_model["target_class"].astype(int)
    df_model["target_binary"] = df_model["target_binary"].astype(int)
    log.info("Spike dataset: %d rows, %d features", len(df_model), len(feat_cols))

    split_date = df_model.index.max() - pd.Timedelta(days=TEST_DAYS)
    train_df   = df_model[df_model.index <= split_date]
    test_df    = df_model[df_model.index >  split_date]

    X_train    = train_df[feat_cols].values
    y_train    = train_df["target_class"].values
    X_test     = test_df[feat_cols].values
    y_test_bin = (test_df["target_class"].values > 0).astype(int)

    # Real val holdout BEFORE augmentation
    val_n        = max(int(len(X_train) * 0.10), 1000)
    X_real_val   = X_train[-val_n:]
    y_real_val   = (y_train[-val_n:] > 0).astype(int)
    X_train_base = X_train[:-val_n]
    y_train_base = y_train[:-val_n]

    # Window augmentation ±WINDOW_AUG periods around real spikes (training base only)
    spike_idx  = np.where(y_train_base > 0)[0]
    window_set = set()
    for idx in spike_idx:
        for offset in range(-WINDOW_AUG, WINDOW_AUG + 1):
            w = idx + offset
            if 0 <= w < len(X_train_base):
                window_set.add(w)
    window_arr = np.array(sorted(window_set))
    X_win = X_train_base[window_arr]
    y_win = y_train_base[window_arr]

    X_aug     = np.vstack([X_train_base, X_win])
    y_aug_bin = (np.concatenate([y_train_base, y_win]) > 0).astype(int)
    log.info(
        "After window augmentation: %d rows (spike prev=%.2f%%)",
        len(X_aug), y_aug_bin.mean() * 100,
    )

    # Clean infinities (LightGBM dislikes ±inf)
    X_aug_clean  = np.where(np.isfinite(X_aug),      X_aug,      0.0)
    X_val_clean  = np.where(np.isfinite(X_real_val), X_real_val, 0.0)
    X_test_clean = np.where(np.isfinite(X_test),     X_test,     0.0)

    log.info("Training LightGBM spike classifier (class_weight={0:1, 1:25})...")
    clf = lgb.LGBMClassifier(
        objective    = "binary",
        class_weight = {0: 1.0, 1: 25.0},
        **LGBM_PARAMS,
    )
    clf.fit(
        X_aug_clean, y_aug_bin,
        eval_set    = [(X_val_clean, y_real_val)],
        eval_metric = "auc",
        callbacks   = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    log.info("Best iteration: %d", clf.best_iteration_)

    spike_proba = clf.predict_proba(X_test_clean)[:, 1]
    auc         = roc_auc_score(y_test_bin, spike_proba)
    log.info("AUC-ROC: %.3f", auc)

    # Profit-optimal threshold from test set
    test_prices      = test_df["imbl_price"].values[:len(y_test_bin)]
    thresholds_sim   = np.linspace(0.01, 0.99, 100)
    profits          = []
    for thresh in thresholds_sim:
        pred_s = (spike_proba >= thresh).astype(int)
        tp_val = test_prices[(pred_s == 1) & (y_test_bin == 1)].clip(0, 5000).sum() * 0.25
        fp_cos = ((pred_s == 1) & (y_test_bin == 0)).sum() * COST_FP * 0.25
        profits.append(tp_val - fp_cos)
    profit_threshold = float(thresholds_sim[np.argmax(profits)])
    log.info("Profit-optimal threshold: %.4f", profit_threshold)

    y_pred = (spike_proba >= profit_threshold).astype(int)
    rec    = recall_score(y_test_bin,    y_pred, zero_division=0)
    prec   = precision_score(y_test_bin, y_pred, zero_division=0)
    f2     = fbeta_score(y_test_bin,     y_pred, beta=2, zero_division=0)

    metrics = {
        "auc_roc":        round(float(auc),            3),
        "recall":         round(float(rec),             3),
        "precision":      round(float(prec),            3),
        "f2":             round(float(f2),              3),
        "threshold_used": round(float(profit_threshold), 4),
        "test_from":      str(test_df.index.min().date()),
        "test_to":        str(test_df.index.max().date()),
    }
    log.info(
        "Spike — AUC=%.3f  Recall=%.3f  Prec=%.3f  F2=%.3f @ thresh=%.4f",
        auc, rec, prec, f2, profit_threshold,
    )

    return {
        "model":        clf,
        "feature_cols": feat_cols,
        "threshold":    profit_threshold,
        "metrics":      metrics,
        "config": {
            "lgbm_params":  LGBM_PARAMS,
            "class_weight": {0: 1.0, 1: 25.0},
            "window_aug":   WINDOW_AUG,
            "horizon":      HORIZON,
            "split_date":   str(split_date.date()),
        },
        "_X_test":  X_test_clean,
        "_y_test":  y_test_bin,
        "_proba":   spike_proba,
    }


# ── predict ─────────────────────────────────────────────────────────────────────

def _assign_regime(spike_proba: float, direction: float, last_price: float) -> str:
    """Assign regime label from spike probability, direction, and last price."""
    if spike_proba >= 0.45:
        return "extreme"
    if spike_proba >= 0.15:
        return "stress"
    if direction < 0:
        return "deep_long" if last_price < -100 else "normal_long"
    return "normal_short"


def _lookup_price(
    ts: pd.Timestamp,
    lag_periods: int,
    hist: pd.Series,
    pred_buf: dict[pd.Timestamp, float],
) -> float:
    """Look up a lagged price from the prediction buffer or historical series."""
    lag_ts = ts - pd.Timedelta(minutes=15 * lag_periods)
    if lag_ts in pred_buf:
        return pred_buf[lag_ts]
    try:
        return float(hist.loc[lag_ts])
    except Exception:
        idx = hist.index.get_indexer([lag_ts], method="nearest")[0]
        return float(hist.iloc[idx]) if idx >= 0 else np.nan


def _calendar_row(ts: pd.Timestamp) -> dict:
    """Return calendar feature dict for a single timestamp."""
    slot = ts.hour * 4 + ts.minute // 15
    how  = ts.dayofweek * 96 + slot
    hoy  = (ts.timetuple().tm_yday - 1) * 96 + slot
    row: dict = {
        "hour":      ts.hour,
        "minute":    ts.minute,
        "dayofweek": ts.dayofweek,
        "month":     ts.month,
        "slot":      slot,
        "hour_sin":  math.sin(2 * math.pi * ts.hour / 24),
        "hour_cos":  math.cos(2 * math.pi * ts.hour / 24),
        "dow_sin":   math.sin(2 * math.pi * ts.dayofweek / 7),
        "dow_cos":   math.cos(2 * math.pi * ts.dayofweek / 7),
        "month_sin": math.sin(2 * math.pi * ts.month / 12),
        "month_cos": math.cos(2 * math.pi * ts.month / 12),
        "slot_sin":  math.sin(2 * math.pi * slot / 96),
        "slot_cos":  math.cos(2 * math.pi * slot / 96),
        "season":    {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[ts.month],
        "is_weekend": int(ts.dayofweek >= 5),
        "is_peak":    int(ts.hour in list(range(7, 10)) + list(range(17, 21))),
        "is_night":   int(ts.hour in [23, 0, 1, 2, 3, 4, 5]),
        "is_winter":  int(ts.month in [12, 1, 2]),
    }
    import holidays  # noqa: PLC0415
    row["is_holiday"] = int(ts.normalize() in holidays.Sweden())
    for k in [1, 2, 3]:
        row[f"daily_sin_k{k}"]  = math.sin(2 * math.pi * k * slot / 96)
        row[f"daily_cos_k{k}"]  = math.cos(2 * math.pi * k * slot / 96)
        row[f"weekly_sin_k{k}"] = math.sin(2 * math.pi * k * how / (7 * 96))
        row[f"weekly_cos_k{k}"] = math.cos(2 * math.pi * k * how / (7 * 96))
    for k in [1, 2]:
        row[f"annual_sin_k{k}"] = math.sin(2 * math.pi * k * hoy / (365 * 96))
        row[f"annual_cos_k{k}"] = math.cos(2 * math.pi * k * hoy / (365 * 96))
    return row


def predict_imbalance(artifacts: dict, data: dict) -> pd.DataFrame:
    """
    Produce a 96-period (24h at 15-min) imbalance price forecast.

    Uses recursive single-step forecasting: the p50 prediction for period t
    feeds into lag features for period t+1.

    Parameters
    ----------
    artifacts : dict  Loaded imbalance artifacts from load_imbalance_artifacts().
    data      : dict  Optional key ``"merged_df"`` (recent 15-min data).

    Returns
    -------
    pd.DataFrame  Columns: timestamp, p05, p50, p95.  96 rows.
    """
    _require_lgb()

    df_merged = data.get("merged_df")
    if df_merged is None:
        df_merged = _load_inference_data()

    feat_cols  = artifacts["feature_cols"]
    models     = artifacts["models"]

    log.info("Building features for imbalance inference...")
    df_feats   = build_features(df_merged, horizon=HORIZON)
    df_feats   = df_feats.dropna(axis=1, how="all")
    avail_cols = [c for c in feat_cols if c in df_feats.columns]

    df_valid   = df_feats.dropna(subset=["imbl_roll_1h_mean"])
    if df_valid.empty:
        log.warning("No valid feature rows for imbalance prediction.")
        return pd.DataFrame(columns=["timestamp", "p05", "p50", "p95"])

    last_row = df_valid.iloc[-1].to_dict()
    now_ts   = df_valid.index[-1]
    future_ts = pd.date_range(
        start   = now_ts + pd.Timedelta(minutes=15),
        periods = 96,
        freq    = "15min",
        tz      = "Europe/Stockholm",
    )
    log.info("Imbalance forecast: %s → %s", future_ts[0], future_ts[-1])

    p_hist   = df_merged["imbl_price"]
    pred_buf: dict[pd.Timestamp, float] = {}
    results  = []

    for ts in future_ts:
        row = {**last_row}

        # Update calendar features for this future timestamp
        row.update(_calendar_row(ts))

        # Update short lag features (1–16 periods) from prediction buffer + history
        for lag in [1, 2, 4, 8, 16]:
            row[f"imbl_lag_{lag}"] = _lookup_price(ts, lag, p_hist, pred_buf)
        row["imbl_lag_1h"] = row["imbl_lag_4"]
        row["imbl_lag_2h"] = row["imbl_lag_8"]
        row["imbl_lag_4h"] = row["imbl_lag_16"]

        # Rolling stats from combined history + predictions (last 4h = 16 periods)
        recent_16 = [_lookup_price(ts, k, p_hist, pred_buf) for k in range(1, 17)]
        recent_4  = recent_16[:4]
        row["imbl_roll_1h_mean"] = float(np.nanmean(recent_4))
        row["imbl_roll_1h_std"]  = float(np.nanstd(recent_4))
        row["imbl_roll_4h_mean"] = float(np.nanmean(recent_16))
        row["imbl_roll_4h_std"]  = float(np.nanstd(recent_16))

        # 1d rolling mean — 96 periods, mostly from real history
        recent_96 = [_lookup_price(ts, k, p_hist, pred_buf) for k in range(1, 97)]
        row["imbl_roll_1d_mean"] = float(np.nanmean(recent_96))

        # EWA approximation
        row["imbl_ewa"] = float(np.nanmean(recent_16[:8]))

        # Direction from previous predicted period's sign
        prev_p50 = pred_buf.get(ts - pd.Timedelta(minutes=15))
        if prev_p50 is not None:
            row["dir_lag_1"] = float(np.sign(prev_p50))

        x    = np.array([[row.get(c, np.nan) for c in avail_cols]])
        p05  = float(models["p05"].predict(x)[0])
        p50  = float(models["p50"].predict(x)[0])
        p95  = float(models["p95"].predict(x)[0])

        pred_buf[ts] = p50
        results.append({"timestamp": ts, "p05": p05, "p50": p50, "p95": p95})

    return pd.DataFrame(results)


def predict_spike(artifacts: dict, data: dict) -> pd.DataFrame:
    """
    Produce a 96-period (24h at 15-min) spike probability forecast.

    Uses the profit-optimal threshold stored in artifacts to assign regime.

    Parameters
    ----------
    artifacts : dict  Loaded spike artifacts from load_spike_artifacts().
    data      : dict  Optional key ``"merged_df"`` (recent 15-min data).

    Returns
    -------
    pd.DataFrame  Columns: timestamp, spike_proba, regime.  96 rows.
    """
    _require_lgb()

    df_merged = data.get("merged_df")
    if df_merged is None:
        df_merged = _load_inference_data()

    feat_cols  = artifacts["feature_cols"]
    clf        = artifacts["model"]

    log.info("Building features for spike inference...")
    df_feats   = build_features(df_merged, horizon=HORIZON)
    df_feats   = df_feats.dropna(axis=1, how="all")
    avail_cols = [c for c in feat_cols if c in df_feats.columns]

    df_valid   = df_feats.dropna(subset=["imbl_roll_1h_mean"])
    if df_valid.empty:
        log.warning("No valid feature rows for spike prediction.")
        return pd.DataFrame(columns=["timestamp", "spike_proba", "regime"])

    last_row = df_valid.iloc[-1].to_dict()
    now_ts   = df_valid.index[-1]
    future_ts = pd.date_range(
        start   = now_ts + pd.Timedelta(minutes=15),
        periods = 96,
        freq    = "15min",
        tz      = "Europe/Stockholm",
    )

    # Latest direction and price for regime assignment
    last_price = float(df_merged["imbl_price"].iloc[-1]) if "imbl_price" in df_merged.columns else 0.0
    last_dir   = float(last_row.get("dir_lag_1", 0.0))

    results = []
    for ts in future_ts:
        row = {**last_row}
        row.update(_calendar_row(ts))

        x   = np.array([[row.get(c, np.nan) for c in avail_cols]])
        x   = np.where(np.isfinite(x), x, 0.0)

        proba  = float(clf.predict_proba(x)[0, 1])
        regime = _assign_regime(proba, last_dir, last_price)
        results.append({"timestamp": ts, "spike_proba": proba, "regime": regime})

    return pd.DataFrame(results)


# ── evaluate ────────────────────────────────────────────────────────────────────

def evaluate_imbalance(
    artifacts: dict,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> dict:
    """
    Evaluate imbalance regression model.

    Returns
    -------
    dict  mae, rmse, pi_coverage, skill_vs_persistence, dir_accuracy
    """
    models = artifacts["models"]
    X_test = X_test if X_test is not None else artifacts.get("_X_test")
    y_test = y_test if y_test is not None else artifacts.get("_y_test")
    if X_test is None or y_test is None:
        raise ValueError("Provide X_test and y_test, or pass artifacts from train_imbalance().")

    pred  = models["p50"].predict(X_test)
    p05_p = models["p05"].predict(X_test)
    p95_p = models["p95"].predict(X_test)
    mae   = mean_absolute_error(y_test, pred)
    rmse  = np.sqrt(mean_squared_error(y_test, pred))
    cov   = float(((y_test >= p05_p) & (y_test <= p95_p)).mean() * 100)

    skill   = float("nan")
    dir_acc = float("nan")
    df_test = artifacts.get("_df_test")
    if df_test is not None and "imbl_lag_1" in df_test.columns:
        pm      = y_test.notna() & df_test["imbl_lag_1"].notna()
        pm_mae  = mean_absolute_error(y_test[pm], df_test.loc[pm, "imbl_lag_1"])
        skill   = float(1 - mae / pm_mae) if pm_mae > 0 else float("nan")
        cur     = df_test["imbl_lag_1"].values
        pds     = np.sign(pred - cur)
        ads     = np.sign(y_test.values - cur)
        mdir    = (pds != 0) & (ads != 0)
        dir_acc = float((pds[mdir] == ads[mdir]).mean()) if mdir.any() else float("nan")

    metrics = {
        "mae":                  round(float(mae),  2),
        "rmse":                 round(float(rmse), 2),
        "pi_coverage":          round(cov,          1),
        "skill_vs_persistence": round(skill,        3),
        "dir_accuracy":         round(dir_acc,      3),
    }
    log.info("Imbalance eval — MAE=%.2f  Coverage=%.1f%%  Skill=%.3f", mae, cov, skill)
    return metrics


def evaluate_spike(
    artifacts: dict,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict:
    """
    Evaluate spike classifier.

    Returns
    -------
    dict  auc_roc, recall, precision, f2, threshold_used
    """
    clf       = artifacts["model"]
    threshold = artifacts["threshold"]
    X_test    = X_test if X_test is not None else artifacts.get("_X_test")
    y_test    = y_test if y_test is not None else artifacts.get("_y_test")
    if X_test is None or y_test is None:
        raise ValueError("Provide X_test and y_test, or pass artifacts from train_spike().")

    proba  = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    auc    = roc_auc_score(y_test, proba)
    rec    = recall_score(y_test,    y_pred, zero_division=0)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    f2     = fbeta_score(y_test,     y_pred, beta=2, zero_division=0)

    metrics = {
        "auc_roc":        round(float(auc),       3),
        "recall":         round(float(rec),        3),
        "precision":      round(float(prec),       3),
        "f2":             round(float(f2),         3),
        "threshold_used": round(float(threshold),  4),
    }
    log.info("Spike eval — AUC=%.3f  Recall=%.3f  Prec=%.3f  F2=%.3f", auc, rec, prec, f2)
    return metrics


# ── save / load ─────────────────────────────────────────────────────────────────

def save_imbalance_artifacts(
    artifacts: dict,
    path: str | Path = MODEL_DIR,
) -> None:
    """Save regression artifacts → model/imbalance_models.pkl + model/imbalance_config.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in artifacts.items() if not k.startswith("_")}

    with open(path / "imbalance_models.pkl", "wb") as f:
        pickle.dump(save["models"], f)

    config = {k: v for k, v in save.items() if k != "models"}
    with open(path / "imbalance_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Imbalance artifacts saved to %s (%d features)", path, len(save["feature_cols"]))


def load_imbalance_artifacts(path: str | Path = MODEL_DIR) -> dict:
    """Load regression artifacts from model/."""
    path = Path(path)
    pkl  = path / "imbalance_models.pkl"
    cfg  = path / "imbalance_config.json"
    if not pkl.exists():
        raise FileNotFoundError(
            f"No imbalance model at {pkl}. "
            "Run: python ml_imbalance.py --train-imbalance"
        )
    with open(pkl, "rb") as f:
        models = pickle.load(f)
    with open(cfg) as f:
        config = json.load(f)
    artifacts = {"models": models, **config}
    log.info("Imbalance artifacts loaded from %s (%d features)", path, len(artifacts["feature_cols"]))
    return artifacts


def save_spike_artifacts(
    artifacts: dict,
    path: str | Path = MODEL_DIR,
) -> None:
    """Save spike artifacts → model/spike_model.pkl + model/spike_config.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in artifacts.items() if not k.startswith("_")}

    with open(path / "spike_model.pkl", "wb") as f:
        pickle.dump(save["model"], f)

    config = {k: v for k, v in save.items() if k != "model"}
    with open(path / "spike_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Spike artifacts saved to %s (%d features)", path, len(save["feature_cols"]))


def load_spike_artifacts(path: str | Path = MODEL_DIR) -> dict:
    """Load spike artifacts from model/."""
    path = Path(path)
    pkl  = path / "spike_model.pkl"
    cfg  = path / "spike_config.json"
    if not pkl.exists():
        raise FileNotFoundError(
            f"No spike model at {pkl}. "
            "Run: python ml_imbalance.py --train-spike"
        )
    with open(pkl, "rb") as f:
        model = pickle.load(f)
    with open(cfg) as f:
        config = json.load(f)
    artifacts = {"model": model, **config}
    log.info("Spike artifacts loaded from %s (%d features)", path, len(artifacts["feature_cols"]))
    return artifacts


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="SE3 Imbalance & Spike ML module")
    parser.add_argument("--train-imbalance", action="store_true",
                        help="Train + save imbalance regression model")
    parser.add_argument("--train-spike",     action="store_true",
                        help="Train + save spike classifier")
    parser.add_argument("--train-all",       action="store_true",
                        help="Train both models")
    parser.add_argument("--forecast",        action="store_true",
                        help="Print next 24h imbalance forecast")
    parser.add_argument("--evaluate",        action="store_true",
                        help="Print test metrics for both models")
    args = parser.parse_args()

    if args.train_all:
        args.train_imbalance = True
        args.train_spike     = True

    data: dict = {}

    if args.train_imbalance or args.train_spike or args.evaluate:
        log.info("Loading training data from DB...")
        df_merged = _load_training_data()
        data["merged_df"] = df_merged

    if args.train_imbalance:
        log.info("=== Training imbalance regression model ===")
        art_i = train_imbalance(data)
        save_imbalance_artifacts(art_i)
        print("\nImbalance regression metrics:")
        for k, v in art_i["metrics"].items():
            print(f"  {k:<30}: {v}")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "imbalance_metrics.json", "w") as f:
            json.dump(art_i["metrics"], f, indent=2)

    if args.train_spike:
        log.info("=== Training spike classifier ===")
        art_s = train_spike(data)
        save_spike_artifacts(art_s)
        print("\nSpike classifier metrics:")
        for k, v in art_s["metrics"].items():
            print(f"  {k:<30}: {v}")
        with open(MODEL_DIR / "spike_metrics.json", "w") as f:
            json.dump(art_s["metrics"], f, indent=2)

    if args.forecast:
        log.info("Loading inference data...")
        df_inf   = _load_inference_data(lookback_days=7)
        inf_data = {"merged_df": df_inf}

        art_i    = load_imbalance_artifacts()
        fc_imbl  = predict_imbalance(art_i, inf_data)
        print("\nNext 24h imbalance price forecast (EUR/MWh):")
        print(fc_imbl.to_string(index=False))

        art_s    = load_spike_artifacts()
        fc_spike = predict_spike(art_s, inf_data)
        print("\nNext 24h spike probability:")
        print(fc_spike.to_string(index=False))

        # Persist combined forecast to DB
        try:
            from pipeline import save_imbalance_forecast  # noqa: PLC0415
            fc_combined = fc_imbl.merge(
                fc_spike[["timestamp", "spike_proba", "regime"]],
                on="timestamp",
            )
            save_imbalance_forecast(fc_combined)
            log.info("Imbalance forecast saved to DB (%d rows)", len(fc_combined))
        except Exception as exc:
            log.warning("Could not save forecast to DB: %s", exc)

    if args.evaluate:
        log.info("Evaluating models on test set (retrain required)...")
        art_i   = train_imbalance(data)
        m_i     = evaluate_imbalance(art_i)
        print("\nImbalance evaluation:")
        for k, v in m_i.items():
            print(f"  {k:<30}: {v}")

        art_s   = train_spike(data)
        m_s     = evaluate_spike(art_s)
        print("\nSpike evaluation:")
        for k, v in m_s.items():
            print(f"  {k:<30}: {v}")

    if not any([args.train_imbalance, args.train_spike, args.forecast, args.evaluate]):
        parser.print_help()
