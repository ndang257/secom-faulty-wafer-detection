"""Project-wide constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Business cost of each error type (arbitrary currency units).
# A missed defective wafer (false negative) is assumed 4x as costly as a false alarm.
COST_FP = 100
COST_FN = 400

# Rough feature filtering
NAN_THRESHOLD = 0.6  # drop sensors with more than 60% missing values
SIGMA = 3.0  # outlier boundary for the 3-sigma rule
KNN_NEIGHBORS = 5
