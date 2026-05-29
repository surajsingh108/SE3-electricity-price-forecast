"""
Syncs latest data and model artifacts from GCS to local,
then retrains the SE3 LightGBM model on the latest data.

Usage:
    python sync_and_retrain.py              # sync + retrain
    python sync_and_retrain.py --sync-only  # just pull from GCS
    python sync_and_retrain.py --train-only # just retrain (no sync)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

GCS_BUCKET  = "gs://se3-cache"
LOCAL_DATA  = Path("data")
LOCAL_MODEL = Path("model")

GCS_FILES = {
    f"{GCS_BUCKET}/se3_cache.duckdb":        LOCAL_DATA / "se3_cache.duckdb",
    f"{GCS_BUCKET}/models.pkl":              LOCAL_MODEL / "models.pkl",
    f"{GCS_BUCKET}/pretrained_models.pkl":   LOCAL_MODEL / "pretrained_models.pkl",
    f"{GCS_BUCKET}/linear_baseline.pkl":     LOCAL_MODEL / "linear_baseline.pkl",
    f"{GCS_BUCKET}/neutralizers.pkl":        LOCAL_MODEL / "neutralizers.pkl",
    f"{GCS_BUCKET}/metrics.json":            LOCAL_MODEL / "metrics.json",
    f"{GCS_BUCKET}/config.json":             LOCAL_MODEL / "config.json",
}


def sync_from_gcs():
    log.info("=== Syncing from GCS ===")
    LOCAL_DATA.mkdir(exist_ok=True)
    LOCAL_MODEL.mkdir(exist_ok=True)

    for gcs_path, local_path in GCS_FILES.items():
        log.info(f"  {gcs_path} → {local_path}")
        result = subprocess.run(
            ["gcloud", "storage", "cp", gcs_path, str(local_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            size = local_path.stat().st_size / 1024 / 1024
            log.info(f"  ✓ {local_path.name} ({size:.1f} MB)")
        else:
            log.warning(f"  ✗ {local_path.name} – {result.stderr.strip()}")

    log.info("Sync complete.")


def retrain():
    log.info("=== Retraining model ===")
    result = subprocess.run(
        [sys.executable, "ml.py", "--train"],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        log.error("Training failed.")
        sys.exit(1)
    log.info("Training complete.")

    log.info("=== Pushing updated artifacts to GCS ===")
    for local_path in LOCAL_MODEL.glob("*.pkl"):
        gcs_dest = f"{GCS_BUCKET}/{local_path.name}"
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), gcs_dest],
            capture_output=True
        )
        log.info(f"  ✓ pushed {local_path.name}")
    for fname in ["metrics.json", "config.json"]:
        p = LOCAL_MODEL / fname
        if p.exists():
            subprocess.run(
                ["gcloud", "storage", "cp", str(p), f"{GCS_BUCKET}/{fname}"],
                capture_output=True
            )
            log.info(f"  ✓ pushed {fname}")
    log.info("Artifacts pushed to GCS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-only",  action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    if args.sync_only:
        sync_from_gcs()
    elif args.train_only:
        retrain()
    else:
        sync_from_gcs()
        retrain()
