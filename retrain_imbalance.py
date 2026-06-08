"""
retrain_imbalance.py — One-shot script to sync data and retrain both imbalance models.

Can be called from a Cloud Run Job, a cron job, or the /retrain endpoint in api.py.

Usage
-----
  python retrain_imbalance.py
  python retrain_imbalance.py --no-sync   # skip data sync, train from existing DB
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("retrain_imbalance")


def run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip pipeline data sync (use existing DB)")
    args = parser.parse_args()

    if not args.no_sync:
        log.info("Step 1: syncing all data sources (including imbalance)...")
        run([sys.executable, "pipeline.py"])

    log.info("Step 2: training imbalance regression + spike classifier...")
    run([sys.executable, "ml_imbalance.py", "--train-all"])

    log.info("Step 3: generating fresh 24h imbalance forecast...")
    run([sys.executable, "ml_imbalance.py", "--forecast"])

    log.info("Retrain complete.")
