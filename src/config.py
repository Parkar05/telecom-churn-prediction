"""Central configuration: paths, dates and the churn definition.

Keeping these in one place means the notebook, the training script, the
scoring job and the app all use exactly the same rules.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"

# ---- Data period -----------------------------------------------------------
DATA_START = "2023-01-01"
DATA_END = "2026-10-31"          # last day of available data ("today" for the demo)

# ---- Churn definition --------------------------------------------------------
# A business account is ACTIVE at a snapshot date if its contract has started,
# it has not terminated, and it was billed that month. An active account has
# CHURNED if it terminates all services in the PREDICTION_WINDOW_DAYS after
# the snapshot (moves to a competitor or cancels).
PREDICTION_WINDOW_DAYS = 90

# ---- Snapshots ----------------------------------------------------------------
# Month-end snapshots used for training. The last labelled snapshot must leave a
# full prediction window before DATA_END.
TRAIN_SNAPSHOT_START = "2024-01-31"
TRAIN_SNAPSHOT_END = "2026-07-31"
TEST_SNAPSHOT_START = "2026-01-31"   # time-based split: snapshots >= this are test
SCORING_SNAPSHOT = DATA_END          # unlabelled snapshot we score for the business

# ---- Risk bands (on predicted probability) -------------------------------------
RISK_BANDS = [(0.0, 0.15, "Low"), (0.15, 0.35, "Medium"), (0.35, 1.01, "High")]

CATEGORICAL_FEATURES = ["segment", "industry", "region", "contract_term"]

RANDOM_STATE = 42
