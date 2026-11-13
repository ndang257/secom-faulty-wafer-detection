"""Scikit-learn compatible preprocessing steps.

Every step learns its parameters in ``fit`` (training folds only) and applies them
in ``transform``, so the steps can sit inside a cross-validated pipeline without
leaking information from validation or test data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from .config import KNN_NEIGHBORS, NAN_THRESHOLD, SIGMA

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
}
OUTLIER_METHODS = ("none", "clip_nan", "clip_bound")


class FeatureFilter(TransformerMixin, BaseEstimator):
    """Drop constant sensors and sensors with too many missing values."""

    def __init__(self, nan_threshold: float = NAN_THRESHOLD):
        self.nan_threshold = nan_threshold

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.constant_ = X.columns[X.nunique(dropna=True) <= 1]
        nan_ratio = X.drop(columns=self.constant_).isna().mean()
        self.sparse_ = nan_ratio.index[nan_ratio > self.nan_threshold]
        self.keep_ = X.columns.difference(self.constant_.union(self.sparse_), sort=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


class SigmaClipper(TransformerMixin, BaseEstimator):
    """Handle outliers with the k-sigma rule, using mean/std learned on training data.

    method:
        "none"       - leave values unchanged
        "clip_nan"   - replace outliers with NaN (later filled by the imputer)
        "clip_bound" - cap outliers at the k-sigma boundary
    """

    def __init__(self, method: str = "clip_bound", k: float = SIGMA):
        self.method = method
        self.k = k

    def fit(self, X: pd.DataFrame, y=None):
        if self.method not in OUTLIER_METHODS:
            raise ValueError(f"method must be one of {OUTLIER_METHODS}, got {self.method!r}")
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        mean, std = X.mean(), X.std()
        self.lower_ = mean - self.k * std
        self.upper_ = mean + self.k * std
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        if self.method == "clip_bound":
            X = X.clip(lower=self.lower_, upper=self.upper_, axis=1)
        elif self.method == "clip_nan":
            outside = X.lt(self.lower_, axis=1) | X.gt(self.upper_, axis=1)
            X = X.mask(outside)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def preprocessing_steps(outlier: str = "clip_bound", scaler: str = "standard") -> list[tuple]:
    """Filter -> outlier handling -> scaling -> kNN imputation.

    Scaling comes before kNN imputation because kNN is distance based.
    """
    if scaler not in SCALERS:
        raise ValueError(f"scaler must be one of {list(SCALERS)}, got {scaler!r}")
    return [
        ("filter", FeatureFilter()),
        ("outliers", SigmaClipper(method=outlier)),
        ("scaler", SCALERS[scaler]().set_output(transform="pandas")),
        ("imputer", KNNImputer(n_neighbors=KNN_NEIGHBORS).set_output(transform="pandas")),
    ]
