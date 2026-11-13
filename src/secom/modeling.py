"""End-to-end model pipeline.

Every step (filtering, outlier handling, scaling, imputation, Boruta, resampling,
Random Forest) lives in one imblearn ``Pipeline``. Cross-validation therefore
refits all of them on each training fold, and resampling never touches the
validation fold - the two leakage problems of the original notebooks.
"""

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_STATE
from .features import BorutaSelector
from .preprocessing import preprocessing_steps
from .resampling import make_sampler

# Model used while comparing preprocessing pipelines (same as the original project).
BASE_RF_PARAMS = {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced"}

# Hyperparameter grid searched for the selected pipeline.
RF_PARAM_GRID = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [5, 10, None],
    "model__min_samples_leaf": [1, 4],
    "model__class_weight": ["balanced", None],
}


def build_pipeline(
    outlier: str = "clip_bound",
    scaler: str = "standard",
    resampler: str = "rose",
    rf_params: dict | None = None,
    boruta_max_iter: int = 100,
    memory=None,
) -> Pipeline:
    params = {**BASE_RF_PARAMS, **(rf_params or {})}
    return Pipeline(
        [
            *preprocessing_steps(outlier, scaler),
            ("boruta", BorutaSelector(max_iter=boruta_max_iter)),
            ("resample", make_sampler(resampler)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)),
        ],
        memory=memory,
    )
