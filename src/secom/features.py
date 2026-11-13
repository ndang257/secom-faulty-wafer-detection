"""Boruta feature selection wrapped as a scikit-learn selector."""

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectorMixin

from .config import RANDOM_STATE


class BorutaSelector(SelectorMixin, BaseEstimator):
    """Keep the features Boruta confirms as relevant.

    Boruta compares each feature's Random Forest importance with randomly
    shuffled "shadow" copies; only features that beat the best shadow are kept.
    If no feature is confirmed (possible on small folds), tentative features are
    used, and failing that the ``min_features`` best-ranked ones.
    """

    def __init__(self, max_iter: int = 100, perc: int = 100, min_features: int = 5, random_state: int = RANDOM_STATE):
        self.max_iter = max_iter
        self.perc = perc
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=self.random_state)
        self.boruta_ = BorutaPy(
            rf,
            n_estimators="auto",
            max_iter=self.max_iter,
            perc=self.perc,
            random_state=self.random_state,
            verbose=0,
        )
        self.boruta_.fit(X_arr, np.asarray(y))

        mask = self.boruta_.support_.copy()
        if not mask.any():
            mask = self.boruta_.support_ | self.boruta_.support_weak_
        if not mask.any():
            best = np.argsort(self.boruta_.ranking_)[: self.min_features]
            mask[best] = True
        self.support_ = mask
        self.tentative_ = self.boruta_.support_weak_
        return self

    def _get_support_mask(self):
        return self.support_
