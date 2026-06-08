"""
hindcast_imbalance.py — Historical backtesting of the imbalance forecast model.

For each day in the specified range, generates what the model WOULD HAVE
predicted using only data available at that time (strict as-of hindcast).

Writes results to imbalance_forecasts with historical generated_at values.
Safe to re-run: INSERT OR IGNORE preserves existing rows.

CLI:
  python hindcast_imbalance.py --start 2026-06-01
  python hindcast_imbalance.py --start 2026-06-01 --end 2026-06-07
  python hindcast_imbalance.py --last-days 7
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

DB_PATH = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("hindcast")


def _predict_spike_recursive(
    art_s: dict,
    df_day: pd.DataFrame,
    fc_imbl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute spike probability for each forecast period.

    Updates imbalance price lag features from the imbalance model's
    predicted p50 values so that spike_proba varies across the 96
    periods (instead of being constant as with the stock predict_spike).
    """
    from ml_imbalance import (  # noqa: PLC0415
        _calendar_row, _assign_regime, _lookup_price, HORIZON,
    )
    from feature_engineering import build_features  # noqa: PLC0415

    feat_cols  = art_s["feature_cols"]
    clf        = art_s["model"]

    df_feats   = build_features(df_day, horizon=HORIZON)
    df_feats   = df_feats.dropna(axis=1, how="all")
    avail_cols = [c for c in feat_cols if c in df_feats.columns]

    df_valid   = df_feats.dropna(subset=["imbl_roll_1h_mean"])
    if df_valid.empty:
        return pd.DataFrame(columns=["timestamp", "spike_proba", "regime"])

    last_row   = df_valid.iloc[-1].to_dict()
    p_hist     = df_day["imbl_price"]

    # Seed the pred_buf with p50 predictions from the imbalance model
    pred_buf: dict[pd.Timestamp, float] = {
        pd.Timestamp(r["timestamp"]): r["p50"]
        for _, r in fc_imbl.iterrows()
    }

    last_price = float(p_hist.dropna().iloc[-1]) if not p_hist.dropna().empty else 0.0
    last_dir   = float(last_row.get("dir_lag_1", 0.0))

    results = []
    for _, fc_row in fc_imbl.iterrows():
        ts  = pd.Timestamp(fc_row["timestamp"])
        row = {**last_row}
        row.update(_calendar_row(ts))

        # Update price-derived lag features using predicted p50 values
        for lag in [1, 2, 4, 8, 16]:
            row[f"imbl_lag_{lag}"] = _lookup_price(ts, lag, p_hist, pred_buf)
        row["imbl_lag_1h"] = row["imbl_lag_4"]
        row["imbl_lag_2h"] = row["imbl_lag_8"]
        row["imbl_lag_4h"] = row["imbl_lag_16"]

        recent_16 = [_lookup_price(ts, k, p_hist, pred_buf) for k in range(1, 17)]
        recent_4  = recent_16[:4]
        row["imbl_roll_1h_mean"] = float(np.nanmean(recent_4))
        row["imbl_roll_1h_std"]  = float(np.nanstd(recent_4))
        row["imbl_roll_4h_mean"] = float(np.nanmean(recent_16))
        row["imbl_roll_4h_std"]  = float(np.nanstd(recent_16))
        recent_96 = [_lookup_price(ts, k, p_hist, pred_buf) for k in range(1, 97)]
        row["imbl_roll_1d_mean"] = float(np.nanmean(recent_96))
        row["imbl_ewa"]          = float(np.nanmean(recent_16[:8]))

        x     = np.array([[row.get(c, np.nan) for c in avail_cols]])
        x     = np.where(np.isfinite(x), x, 0.0)
        proba = float(clf.predict_proba(x)[0, 1])
        results.append({
            "timestamp":   ts,
            "spike_proba": proba,
            "regime":      _assign_regime(proba, last_dir, last_price),
        })

    return pd.DataFrame(results)


def run_hindcast(start: date, end: date) -> int:
    """
    Generate hindcast forecasts for each day in [start, end].

    Returns total number of rows written.
    """
    from ml_imbalance import (  # noqa: PLC0415
        _load_training_data,
        load_imbalance_artifacts,
        load_spike_artifacts,
        predict_imbalance,
    )
    from pipeline import _upsert, get_conn  # noqa: PLC0415

    log.info("Loading model artifacts...")
    art_i = load_imbalance_artifacts()
    art_s = load_spike_artifacts()

    # Load generous lookback so every day in range has enough context
    lookback_start = str((pd.Timestamp(start) - pd.Timedelta(days=21)).date())
    log.info("Loading training data from %s onward...", lookback_start)
    df_all = _load_training_data(start_date=lookback_start)

    if df_all.empty:
        log.error("No training data loaded — check DB connection.")
        sys.exit(1)

    log.info("Loaded %d rows (%s → %s)",
             len(df_all), df_all.index.min().date(), df_all.index.max().date())

    written_total = 0
    current = start

    while current <= end:
        # Strict as-of cutoff: data through 23:45 Stockholm on this day
        cutoff_local = pd.Timestamp(current, tz="Europe/Stockholm") + \
                       pd.Timedelta(hours=23, minutes=45)

        df_day = df_all[df_all.index <= cutoff_local].copy()

        if len(df_day) < 96:
            log.warning("Only %d rows available for %s — skipping.", len(df_day), current)
            current += timedelta(days=1)
            continue

        data = {"merged_df": df_day}

        try:
            fc_imbl = predict_imbalance(art_i, data)

            if fc_imbl.empty:
                log.warning("Empty imbalance forecast for %s — skipping.", current)
                current += timedelta(days=1)
                continue

            # Use recursive spike predictor so spike_proba varies across periods
            fc_spike = _predict_spike_recursive(art_s, df_day, fc_imbl)

            fc = fc_imbl.merge(
                fc_spike[["timestamp", "spike_proba", "regime"]],
                on="timestamp", how="left",
            )

            # Historical generated_at: cutoff time converted to UTC
            fc["generated_at"] = cutoff_local.tz_convert("UTC")

            cols = ["timestamp", "generated_at", "p05", "p50", "p95",
                    "spike_proba", "regime"]
            for col in cols:
                if col not in fc.columns:
                    fc[col] = np.nan

            conn = get_conn()
            _upsert(conn, "imbalance_forecasts", fc[cols])
            conn.close()

            n = len(fc)
            written_total += n
            log.info(
                "✓ %s → %d rows  p50=[%.1f, %.1f]  spike_p=[%.3f, %.3f]",
                current, n,
                fc["p50"].min(), fc["p50"].max(),
                fc["spike_proba"].min(), fc["spike_proba"].max(),
            )

        except Exception as exc:
            log.error("Failed for %s: %s", current, exc)

        current += timedelta(days=1)

    log.info("Hindcast complete. Total rows written: %d", written_total)
    return written_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate historical as-of imbalance forecasts for backtesting."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start",     type=str,
                       help="Start date YYYY-MM-DD (inclusive)")
    group.add_argument("--last-days", type=int,
                       help="Hindcast last N days ending yesterday")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    yesterday = date.today() - timedelta(days=1)

    if args.last_days:
        start = date.today() - timedelta(days=args.last_days)
        end   = yesterday
    else:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end) if args.end else yesterday

    if start > end:
        log.error("start (%s) must be ≤ end (%s)", start, end)
        sys.exit(1)

    n_days = (end - start).days + 1
    log.info("Hindcast: %s → %s (%d days)", start, end, n_days)
    run_hindcast(start, end)


if __name__ == "__main__":
    main()
