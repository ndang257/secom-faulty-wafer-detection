"""Oversampling strategies for the minority (fail) class.

All samplers are imbalanced-learn objects, so inside an ``imblearn.pipeline.Pipeline``
they only run during ``fit`` - validation and test data are never resampled.
"""

import numpy as np
from imblearn import FunctionSampler
from imblearn.over_sampling import ADASYN, SMOTE

from .config import RANDOM_STATE


def rose(X, y, minority_class: int = 1, shrink: float = 1.0, random_state: int = RANDOM_STATE):
    """ROSE-style smoothed bootstrap oversampling of the minority class.

    New minority samples are bootstrap draws plus Gaussian noise. The noise width
    per feature follows the ROSE kernel bandwidth (Menardi & Torelli, 2014):
    ``h_j = shrink * (4 / ((d + 2) * n)) ** (1 / (d + 4)) * std_j``.
    The minority class is grown until both classes are the same size.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_min = X[y == minority_class]
    n_min, d = X_min.shape
    n_new = int((y != minority_class).sum() - n_min)
    if n_new <= 0:
        return X, y

    bandwidth = shrink * (4 / ((d + 2) * n_min)) ** (1 / (d + 4)) * X_min.std(axis=0, ddof=1)
    seeds = X_min[rng.integers(0, n_min, size=n_new)]
    synthetic = seeds + rng.normal(size=(n_new, d)) * bandwidth

    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.full(n_new, minority_class)])
    return X_res, y_res


def make_sampler(name: str, random_state: int = RANDOM_STATE):
    """Return an imblearn sampler by name (``"none"`` returns ``"passthrough"``)."""
    samplers = {
        "none": lambda: "passthrough",
        "rose": lambda: FunctionSampler(func=rose, kw_args={"random_state": random_state}, validate=False),
        "smote": lambda: SMOTE(random_state=random_state),
        "adasyn": lambda: ADASYN(random_state=random_state),
    }
    if name not in samplers:
        raise ValueError(f"sampler must be one of {list(samplers)}, got {name!r}")
    return samplers[name]()


RESAMPLERS = ("none", "rose", "smote", "adasyn")
