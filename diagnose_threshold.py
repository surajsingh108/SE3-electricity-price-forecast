from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

log = logging.getLogger("diagnose_threshold")

DB_PATH      = os.environ.get("SE3_DB_PATH", "data/se3_cache.duckdb")
MODEL_DIR    = os.environ.get("MODEL_DIR", "model")
CONFIG_PATH  = Path(MODEL_DIR) / "spike_config.json"
HISTORY_PATH = Path(MODEL_DIR) / "threshold_history.json"

# Safety bounds — LLM cannot recommend outside these
THRESHOLD_MIN       = 0.10
THRESHOLD_MAX       = 0.95
MAX_SINGLE_CHANGE   = 0.15   # max change per run


def load_spike_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_spike_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def load_threshold_history() -> list:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_threshold_history(history: list) -> None:
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)


def compute_recent_metrics(days: int = 30) -> dict | None:
    """
    Compute spike detection metrics from the last N days of hindcast data.

    Parameters
    ----------
    days : int
        Number of days of hindcast to evaluate.

    Returns
    -------
    dict | None
        Metrics dict, or None if there is insufficient data.
    """
    conn   = duckdb.connect(DB_PATH, read_only=True)
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = conn.execute(f"""
        SELECT
            ip.timestamp,
            ip.imbl_price,
            f.p05, f.p50, f.p95, f.spike_proba
        FROM imbalance_prices ip
        JOIN imbalance_forecasts f
            ON f.timestamp = ip.timestamp
        WHERE ip.timestamp >= TIMESTAMP '{cutoff} 00:00:00'
          AND ip.imbl_price IS NOT NULL
          AND f.p50 IS NOT NULL
        ORDER BY ip.timestamp
    """).df()
    conn.close()

    if df.empty or len(df) < 10:
        return None

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Europe/Stockholm")
    )
    df["timestamp"] = df["timestamp"].dt.round("15min")
    df = df.drop_duplicates(subset=["timestamp"])

    cfg       = load_spike_config()
    threshold = cfg.get("threshold", 0.45)

    actual_spike = df["imbl_price"] > 200
    pred_spike   = df["spike_proba"] > threshold

    tp = (actual_spike & pred_spike).sum()
    fp = (~actual_spike & pred_spike).sum()
    fn = (actual_spike & ~pred_spike).sum()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f2        = (5 * precision * recall) / max(4 * precision + recall, 1e-9)
    fpr       = fp / max((~actual_spike).sum(), 1)

    in_band = (df["imbl_price"] >= df["p05"]) & (df["imbl_price"] <= df["p95"])
    pi_cov  = in_band.mean()
    mae     = (df["imbl_price"] - df["p50"]).abs().mean()

    thresh_sweep: dict[float, dict] = {}
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ps   = df["spike_proba"] > t
        tp_t = (actual_spike & ps).sum()
        fn_t = (actual_spike & ~ps).sum()
        fp_t = (~actual_spike & ps).sum()
        r_t  = tp_t / max(tp_t + fn_t, 1)
        p_t  = tp_t / max(tp_t + fp_t, 1)
        f2_t = (5 * p_t * r_t) / max(4 * p_t + r_t, 1e-9)
        thresh_sweep[t] = {
            "recall":      round(float(r_t), 3),
            "precision":   round(float(p_t), 3),
            "f2":          round(float(f2_t), 3),
            "n_predicted": int(ps.sum()),
        }

    best_thresh = max(thresh_sweep.items(), key=lambda x: x[1]["f2"])

    now    = datetime.now()
    month  = now.month
    season = (
        "winter" if month in [12, 1, 2] else
        "spring" if month in [3, 4, 5]  else
        "summer" if month in [6, 7, 8]  else "autumn"
    )

    return {
        "n_periods":           len(df),
        "n_days":              days,
        "n_actual_spikes":     int(actual_spike.sum()),
        "n_predicted":         int(pred_spike.sum()),
        "n_tp":                int(tp),
        "recall":              round(float(recall), 3),
        "precision":           round(float(precision), 3),
        "f2":                  round(float(f2), 3),
        "fpr":                 round(float(fpr), 3),
        "pi_coverage":         round(float(pi_cov), 3),
        "mae":                 round(float(mae), 1),
        "spike_prevalence":    round(float(actual_spike.mean()), 4),
        "mean_price":          round(float(df["imbl_price"].mean()), 1),
        "season":              season,
        "current_threshold":   threshold,
        "threshold_sweep":     thresh_sweep,
        "best_f2_threshold":   best_thresh[0],
        "best_f2_value":       best_thresh[1]["f2"],
    }


def call_gemini(prompt: str, system: str) -> str:
    """
    Call Gemini using the same pattern as agent/nodes.py.

    Parameters
    ----------
    prompt : str
        User-facing prompt text.
    system : str
        System instruction for the model.

    Returns
    -------
    str
        Raw text response from the model.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return response.text.strip()


def parse_llm_response(response: str) -> dict:
    """
    Parse JSON from LLM response. Handles markdown code fences.

    Parameters
    ----------
    response : str
        Raw model output.

    Returns
    -------
    dict
        Parsed recommendation with recommended_threshold and reasoning.
    """
    clean = re.sub(r"```json|```", "", response).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from: {clean[:200]}")


def validate_recommendation(
    current: float,
    recommended: float,
    metrics: dict,
) -> tuple[bool, str]:
    """
    Validate LLM recommendation before applying it.

    Parameters
    ----------
    current : float
        Currently deployed threshold.
    recommended : float
        LLM-recommended threshold.
    metrics : dict
        Recent performance metrics including threshold_sweep.

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason)
    """
    if not (THRESHOLD_MIN <= recommended <= THRESHOLD_MAX):
        return False, (
            f"Recommended {recommended} outside bounds "
            f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]"
        )

    change = abs(recommended - current)
    if change > MAX_SINGLE_CHANGE + 1e-9:
        return False, f"Change {change:.3f} exceeds max {MAX_SINGLE_CHANGE}"

    sweep = metrics.get("threshold_sweep", {})
    current_f2 = sweep.get(
        min(sweep.keys(), key=lambda x: abs(x - current)), {}
    ).get("f2", 0)
    new_f2 = sweep.get(
        min(sweep.keys(), key=lambda x: abs(x - recommended)), {}
    ).get("f2", 0)

    if new_f2 < current_f2 - 0.05:
        return False, (
            f"New threshold would reduce F2 from {current_f2:.3f} to {new_f2:.3f}"
        )

    return True, "OK"


def run_diagnosis(days: int = 30, dry_run: bool = False) -> dict:
    """
    Main diagnosis loop.

    Parameters
    ----------
    days : int
        Days of hindcast to evaluate.
    dry_run : bool
        If True, print recommendation without writing changes.

    Returns
    -------
    dict
        Decision record.
    """
    log.info("Computing metrics for last %d days...", days)
    metrics = compute_recent_metrics(days)

    if metrics is None:
        log.warning("Insufficient hindcast data for diagnosis")
        return {
            "timestamp": datetime.now().isoformat(),
            "decision":  "skip",
            "reason":    "insufficient data",
            "metrics":   None,
        }

    cfg               = load_spike_config()
    current_threshold = metrics["current_threshold"]

    log.info(
        "Current metrics: recall=%.3f precision=%.3f F2=%.3f",
        metrics["recall"], metrics["precision"], metrics["f2"],
    )
    log.info("Current threshold: %.4f", current_threshold)
    log.info(
        "Best F2 threshold in sweep: %.2f (F2=%.3f)",
        metrics["best_f2_threshold"], metrics["best_f2_value"],
    )

    SYSTEM_PROMPT = """
You are a quantitative analyst managing a spike detection model
for SE3 Swedish electricity imbalance prices (Stockholm/Mälardalen).

Your job is to recommend an operational threshold adjustment based
on recent model performance. You are NOT retraining the model —
only adjusting the decision boundary (threshold) applied to the
model's output probability.

The model outputs P(spike) for each 15-min period.
A spike is defined as imbalance price > 200 EUR/MWh.
The threshold determines when we act on the signal.

Tradeoffs:
  Lower threshold → more alarms → higher recall, lower precision
  Higher threshold → fewer alarms → lower recall, higher precision
  F2 score weights recall 2x over precision (missing a spike is
  more costly than a false alarm for battery dispatch)

Rules:
  - Recommend threshold between 0.10 and 0.95
  - Change no more than 0.15 from current in one step
  - Optimise for F2 score on recent data
  - If current metrics are acceptable (F2 > 0.30, recall > 0.40),
    recommend NO CHANGE — stability is valuable
  - Consider seasonal context — summer has fewer spikes than winter
  - A threshold sweep is provided showing F2 at each level

Output valid JSON only, no markdown:
{
  "current_threshold": float,
  "recommended_threshold": float,
  "change": float,
  "action": "increase" | "decrease" | "no_change",
  "reasoning": "2-3 sentence explanation",
  "expected_f2_direction": "up" | "down" | "unchanged",
  "confidence": "high" | "medium" | "low"
}
"""

    USER_PROMPT = f"""
Current spike detector configuration:
  threshold: {current_threshold}

Last {metrics['n_days']} days of hindcast performance
({metrics['n_periods']} periods, {metrics['n_actual_spikes']} actual spikes):
  recall:              {metrics['recall']}    (target > 0.40)
  precision:           {metrics['precision']}
  F2 score:            {metrics['f2']}        (target > 0.30)
  false positive rate: {metrics['fpr']}
  PI coverage:         {metrics['pi_coverage']}
  MAE:                 {metrics['mae']} EUR/MWh

Market context:
  season:           {metrics['season']}
  spike prevalence: {metrics['spike_prevalence']:.2%}
  mean price:       {metrics['mean_price']} EUR/MWh

Threshold sweep (F2 at each threshold level):
{json.dumps(metrics['threshold_sweep'], indent=2)}

Best F2 in sweep: threshold={metrics['best_f2_threshold']}, F2={metrics['best_f2_value']}

Reference (notebook validation on April test set):
  AUC-ROC: 0.910
  walk-forward recall: 0.422
  walk-forward F2: 0.436

Recommend a threshold adjustment or confirm no change needed.
"""

    log.info("Calling LLM for threshold recommendation...")
    try:
        raw_response   = call_gemini(USER_PROMPT, SYSTEM_PROMPT)
        recommendation = parse_llm_response(raw_response)
    except Exception as e:
        log.error("LLM call failed: %s", e)
        return {
            "timestamp": datetime.now().isoformat(),
            "decision":  "error",
            "reason":    str(e),
            "metrics":   metrics,
        }

    recommended = float(
        recommendation.get("recommended_threshold", current_threshold)
    )
    action    = recommendation.get("action", "no_change")
    reasoning = recommendation.get("reasoning", "")

    log.info("LLM recommendation: threshold=%.4f action=%s", recommended, action)
    log.info("Reasoning: %s", reasoning)

    is_valid, validation_msg = validate_recommendation(
        current_threshold, recommended, metrics
    )

    record: dict = {
        "timestamp":              datetime.now().isoformat(),
        "current_threshold":      current_threshold,
        "recommended_threshold":  recommended,
        "action":                 action,
        "reasoning":              reasoning,
        "confidence":             recommendation.get("confidence", "low"),
        "metrics_snapshot":       {
            k: v for k, v in metrics.items() if k != "threshold_sweep"
        },
        "validation":             validation_msg,
        "applied":                False,
    }

    if not is_valid:
        log.warning("Recommendation rejected: %s", validation_msg)
        record["decision"] = "rejected"
    elif action == "no_change" or abs(recommended - current_threshold) < 0.01:
        log.info("No change recommended — threshold unchanged")
        record["decision"] = "no_change"
        record["applied"]  = True
    elif dry_run:
        log.info(
            "DRY RUN — would apply threshold %.4f → %.4f",
            current_threshold, recommended,
        )
        record["decision"] = "dry_run"
    else:
        cfg["threshold"]    = recommended
        cfg["last_updated"] = datetime.now().isoformat()
        cfg["update_reason"] = reasoning
        save_spike_config(cfg)
        record["decision"] = "applied"
        record["applied"]  = True
        log.info(
            "✓ Threshold updated: %.4f → %.4f",
            current_threshold, recommended,
        )

        gcs_path = "gs://se3-cache/spike_config.json"
        try:
            subprocess.run(
                ["gsutil", "cp", str(CONFIG_PATH), gcs_path],
                check=True, capture_output=True,
            )
            log.info("Config synced to GCS")
        except Exception as e:
            log.warning("GCS sync failed (local update still applied): %s", e)

    history = load_threshold_history()
    history.append(record)
    save_threshold_history(history)

    _write_decision_to_db(record)

    return record


def _write_decision_to_db(record: dict) -> None:
    """Write decision record to the threshold_decisions table in DuckDB."""
    try:
        conn = duckdb.connect(DB_PATH, read_only=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threshold_decisions (
                timestamp              TIMESTAMPTZ,
                current_threshold      DOUBLE,
                recommended_threshold  DOUBLE,
                action                 VARCHAR,
                decision               VARCHAR,
                reasoning              VARCHAR,
                confidence             VARCHAR,
                recall                 DOUBLE,
                precision_val          DOUBLE,
                f2                     DOUBLE,
                applied                BOOLEAN
            )
        """)
        m = record.get("metrics_snapshot", {})
        conn.execute("""
            INSERT INTO threshold_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            record["timestamp"],
            record["current_threshold"],
            record["recommended_threshold"],
            record["action"],
            record["decision"],
            record["reasoning"],
            record["confidence"],
            m.get("recall"),
            m.get("precision"),
            m.get("f2"),
            record["applied"],
        ])
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("DB write failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="LLM-assisted spike-threshold self-diagnosis"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Days of hindcast to evaluate (default 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print recommendation without applying",
    )
    args = parser.parse_args()

    result = run_diagnosis(days=args.days, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
