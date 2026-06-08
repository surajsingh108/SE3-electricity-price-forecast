"""
retrain.py — Forecast refresh script called by the /retrain API endpoint.

Sequence:
  1. Sync all data sources (pipeline.py via ml.py --forecast)
  2. Generate next 24h spot price forecast (ml.py --forecast)
  3. Generate next 24h imbalance + spike forecast (ml_imbalance.py --forecast)

This is the FAST path (inference only, no model retraining).
Run on a schedule (e.g., hourly via Cloud Scheduler) or via POST /retrain.

Full model retrain: use retrain_imbalance.py
"""
from __future__ import annotations

import logging
import subprocess
import sys

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("retrain")


def run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


if __name__ == "__main__":
    log.info("Step 1: syncing all data sources...")
    run([sys.executable, "pipeline.py"])

    log.info("Step 2: generating spot price forecast...")
    try:
        run([sys.executable, "ml.py", "--forecast"])
    except Exception as exc:
        log.warning("Spot price forecast failed (non-fatal): %s", exc)

    log.info("Step 3: generating imbalance + spike forecast...")
    run([sys.executable, "ml_imbalance.py", "--forecast"])

    log.info("Step 4: backfilling hindcast (last 7 days)...")
    result = subprocess.run(
        [sys.executable, "hindcast_imbalance.py", "--last-days", "7"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        log.info("Hindcast complete: %s", result.stdout.strip()[-200:])
    else:
        log.warning("Hindcast failed (non-fatal): %s", result.stderr[-500:])

    log.info("Running threshold diagnosis...")
    result = subprocess.run(
        [sys.executable, "diagnose_threshold.py", "--days", "30"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        log.info("Threshold diagnosis complete: %s",
                 result.stdout.strip()[-200:])
    else:
        log.warning("Threshold diagnosis failed: %s",
                    result.stderr[-300:])
        # Never block the main pipeline on diagnosis failure

    log.info("Forecast refresh complete.")
