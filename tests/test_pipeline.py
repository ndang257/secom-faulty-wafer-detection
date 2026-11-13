import numpy as np
import pandas as pd
import pytest

from secom.evaluation import misclassification_cost
from secom.modeling import build_pipeline
from secom.preprocessing import FeatureFilter, SigmaClipper
from secom.resampling import make_sampler, rose


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"s{i}" for i in range(8)])
    X["const"] = 1.0
    X["sparse"] = np.where(rng.random(200) < 0.8, np.nan, rng.normal(size=200))
    X.iloc[::17, 0] = np.nan
    y = pd.Series((X["s1"] + rng.normal(scale=0.5, size=200) > 1.2).astype(int))
    return X, y


def test_feature_filter_drops_constant_and_sparse(toy):
    X, _ = toy
    f = FeatureFilter(nan_threshold=0.6).fit(X)
    assert "const" in f.constant_ and "sparse" in f.sparse_
    assert list(f.transform(X).columns) == [f"s{i}" for i in range(8)]


def test_sigma_clipper_uses_training_bounds_only(toy):
    X, _ = toy
    train, test = X.iloc[:150], X.iloc[150:].copy()
    test.iloc[0, 1] = 1e6
    clip = SigmaClipper("clip_bound").fit(train)
    assert clip.transform(test).iloc[0, 1] == pytest.approx(clip.upper_["s1"])
    assert np.isnan(SigmaClipper("clip_nan").fit(train).transform(test).iloc[0, 1])
    with pytest.raises(ValueError):
        SigmaClipper("bogus").fit(train)


def test_rose_balances_classes_without_touching_originals(toy):
    X, y = toy
    X_res, y_res = rose(X.iloc[:, :8].fillna(0), y)
    assert (y_res == 1).sum() == (y_res == 0).sum()
    np.testing.assert_array_equal(X_res[: len(X)], X.iloc[:, :8].fillna(0).to_numpy())


@pytest.mark.parametrize("name", ["none", "rose", "smote", "adasyn"])
def test_make_sampler(name):
    assert make_sampler(name) is not None


def test_cost_function():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [1, 0, 0, 0, 1]  # 1 FP, 2 FN
    assert misclassification_cost(y_true, y_pred) == 100 + 2 * 400


def test_pipeline_fits_and_predicts(toy):
    X, y = toy
    pipe = build_pipeline("clip_bound", "standard", "rose", boruta_max_iter=10)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and np.all((proba >= 0) & (proba <= 1))
    assert pipe.named_steps["boruta"].support_.sum() >= 1
