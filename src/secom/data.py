"""Load the UCI SECOM dataset and create the train/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, RANDOM_STATE, TEST_SIZE


def load_secom(data_dir: Path | str = DATA_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return sensor features, binary target (1 = fail) and test timestamps.

    The raw labels file encodes pass as -1 and fail as 1; we map this to 0/1.
    """
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"s{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        data_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
    )
    timestamp = pd.to_datetime(labels["timestamp"], dayfirst=True)
    y = (labels["label"] == 1).astype(int).rename("fail")
    return X, y, timestamp


def split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split, done once before any preprocessing."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
