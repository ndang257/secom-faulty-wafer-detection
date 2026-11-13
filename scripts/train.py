"""Train and evaluate the SECOM fault-detection model end to end.

Steps
1. Stratified 80/20 split. The test set is used once, at the very end.
2. Pipeline selection: 5-fold CV on the training set over
   outlier handling x scaler x resampler, scored by misclassification cost.
3. Hyperparameter tuning of the Random Forest for the selected pipeline (same CV).
4. Decision threshold: pick the cost-minimising threshold from out-of-fold
   predictions on the training set.
5. Fit on the full training set and evaluate once on the test set.

Usage:
    python scripts/train.py            # full run (~1 h on 2 CPU cores)
    python scripts/train.py --quick    # small grid, 3 folds - for a fast check
"""

import argparse
import itertools
import json
import tempfile
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate  # noqa: E402

from secom.config import COST_FN, COST_FP, CV_FOLDS, FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, ROOT  # noqa: E402
from secom.data import load_secom, split  # noqa: E402
from secom.evaluation import (  # noqa: E402
    CV_SCORING,
    bootstrap_ci,
    classification_metrics,
    misclassification_cost,
    plot_confusion_and_roc,
    plot_cost_breakdown,
)
from secom.modeling import RF_PARAM_GRID, build_pipeline  # noqa: E402
from secom.preprocessing import OUTLIER_METHODS, SCALERS  # noqa: E402
from secom.resampling import RESAMPLERS  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compare_pipelines(X, y, cv, memory, outliers, scalers, resamplers, boruta_iter) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(outliers, scalers, resamplers))
    for i, (outlier, scaler, resampler) in enumerate(combos, 1):
        pipe = build_pipeline(outlier, scaler, resampler, boruta_max_iter=boruta_iter, memory=memory)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING, error_score="raise")
        row = {"outlier": outlier, "scaler": scaler, "resampler": resampler}
        for name in CV_SCORING:
            vals = scores[f"test_{name}"] * (-1 if name == "cost" else 1)
            row[f"{name}_mean"], row[f"{name}_std"] = vals.mean(), vals.std()
        rows.append(row)
        log(
            f"[{i}/{len(combos)}] {outlier:10s} {scaler:8s} {resampler:6s} "
            f"cost={row['cost_mean']:7.0f}  auc={row['roc_auc_mean']:.3f}  recall={row['recall_mean']:.3f}"
        )
    return pd.DataFrame(rows).sort_values(["cost_mean", "roc_auc_mean"], ascending=[True, False])


def best_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [misclassification_cost(y_true, (proba >= t).astype(int)) for t in grid]
    curve = pd.DataFrame({"threshold": grid, "cost": costs})
    return float(curve.loc[curve["cost"].idxmin(), "threshold"]), curve


def evaluate(name, model, X_test, y_test, threshold=0.5) -> tuple[dict, np.ndarray, np.ndarray]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {"model": name, "threshold": threshold, **classification_metrics(y_test, pred, proba)}, pred, proba


def main(quick: bool = False) -> None:
    # quick runs write to reports/quick/ so they never overwrite the real results
    out_dir = REPORTS_DIR / "quick" if quick else REPORTS_DIR
    fig_dir = out_dir / "figures" if quick else FIGURES_DIR
    model_dir = ROOT / "models" / ("quick" if quick else "")
    for d in (out_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    folds = 3 if quick else CV_FOLDS
    boruta_iter = 30 if quick else 100
    outliers = ("clip_bound",) if quick else OUTLIER_METHODS
    scalers = ("standard",) if quick else tuple(SCALERS)
    resamplers = ("none", "rose") if quick else RESAMPLERS
    param_grid = {"model__max_depth": [5, None]} if quick else RF_PARAM_GRID

    X, y, _ = load_secom()
    X_train, X_test, y_train, y_test = split(X, y)
    log(f"train {X_train.shape}, {y_train.sum()} fails | test {X_test.shape}, {y_test.sum()} fails")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    cache = tempfile.mkdtemp(prefix="secom_cache_")

    # 1) pipeline selection
    log("Stage 1: comparing preprocessing pipelines with CV")
    comparison = compare_pipelines(X_train, y_train, cv, cache, outliers, scalers, resamplers, boruta_iter)
    comparison.to_csv(out_dir / "cv_pipeline_comparison.csv", index=False)
    best = comparison.iloc[0][["outlier", "scaler", "resampler"]].to_dict()
    log(f"Selected pipeline: {best}")

    # 2) hyperparameter tuning
    log("Stage 2: tuning Random Forest hyperparameters")
    search = GridSearchCV(
        build_pipeline(**best, boruta_max_iter=boruta_iter, memory=cache),
        param_grid,
        scoring=CV_SCORING,
        refit="cost",
        cv=cv,
    )
    search.fit(X_train, y_train)
    tuning = pd.DataFrame(search.cv_results_)
    tuning.to_csv(out_dir / "cv_tuning.csv", index=False)
    best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    log(f"Best params: {best_params} | CV cost {-search.best_score_:.0f}")

    # 3) cost-optimal decision threshold from out-of-fold probabilities
    log("Stage 3: choosing the decision threshold")
    oof = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, curve = best_threshold(y_train.to_numpy(), oof)
    curve.to_csv(out_dir / "cv_threshold_curve.csv", index=False)
    pd.DataFrame({"fail": y_train.to_numpy(), "proba": oof}, index=X_train.index).to_csv(
        out_dir / "oof_predictions.csv", index_label="wafer"
    )
    log(f"Cost-optimal threshold: {threshold}")

    # 4) final evaluation on the untouched test set
    log("Stage 4: fitting final models and evaluating on the test set")
    baseline = build_pipeline(best["outlier"], best["scaler"], "none", boruta_max_iter=boruta_iter, memory=cache)
    selected = build_pipeline(**best, boruta_max_iter=boruta_iter, memory=cache)
    tuned = search.best_estimator_  # already refit on the full training set
    baseline.fit(X_train, y_train)
    if best["resampler"] != "none":
        selected.fit(X_train, y_train)

    candidates = [("Baseline (no resampling)", baseline, 0.5)]
    if best["resampler"] != "none":  # otherwise identical to the baseline
        candidates.append((f"Selected pipeline ({best['resampler'].upper()})", selected, 0.5))
    candidates += [("Tuned", tuned, 0.5), ("Tuned + cost threshold", tuned, threshold)]
    # reference: never flag a wafer (cost = number of failures x COST_FN)
    results = [
        {
            "model": "Flag nothing",
            "threshold": None,
            **classification_metrics(y_test, np.zeros(len(y_test), int), np.zeros(len(y_test))),
        }
    ]
    preds = {}
    for name, model, thr in candidates:
        res, pred, proba = evaluate(name, model, X_test, y_test, thr)
        results.append(res)
        preds[name] = (pred, proba)
    test_metrics = pd.DataFrame(results)
    test_metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    print(test_metrics.round(3).to_string(index=False))

    pd.DataFrame({"fail": y_test.to_numpy(), "proba": preds["Tuned"][1]}, index=X_test.index).to_csv(
        out_dir / "test_predictions.csv", index_label="wafer"
    )
    final_name = "Tuned + cost threshold"
    ci = bootstrap_ci(y_test, *preds[final_name])
    ci.to_csv(out_dir / "test_bootstrap_ci.csv")

    # selected sensors and their importance in the final model
    boruta = tuned.named_steps["boruta"]
    names_in = tuned[:-3].get_feature_names_out()  # features entering Boruta
    features = pd.DataFrame(
        {
            "sensor": names_in[boruta.support_],
            "importance": tuned.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    features["boruta_status"] = "confirmed"
    tentative = pd.DataFrame({"sensor": names_in[boruta.tentative_ & ~boruta.support_], "boruta_status": "tentative"})
    pd.concat([features, tentative]).to_csv(out_dir / "selected_sensors.csv", index=False)

    # figures
    fig = plot_confusion_and_roc(y_test, *preds[final_name], title=final_name)
    fig.savefig(fig_dir / "final_confusion_roc.png", dpi=150)
    short_names = test_metrics["model"].str.replace(" (", "\n(").str.replace(" + ", "\n+ ")
    plot_cost_breakdown(test_metrics.assign(model=short_names)).savefig(fig_dir / "test_cost_breakdown.png", dpi=150)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["cost"])
    ax.axvline(threshold, color="#C44E52", ls="--", label=f"chosen = {threshold:.2f}")
    ax.axhline(y_train.sum() * COST_FN, color="grey", ls=":", label="flag nothing")
    ax.set(xlabel="Decision threshold", ylabel="Out-of-fold cost (training set)", title="Cost vs. decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "threshold_cost_curve.png", dpi=150)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(features["sensor"][::-1], features["importance"][::-1], color="#4C72B0")
    ax.set(xlabel="Random Forest importance", title="Sensors selected by Boruta")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)

    summary = {
        "quick_run": quick,
        "cv_folds": folds,
        "selected_pipeline": best,
        "rf_params": {k: (None if v is None else v) for k, v in best_params.items()},
        "threshold": threshold,
        "cost_fp": COST_FP,
        "cost_fn": COST_FN,
        "n_selected_sensors": int(len(features)),
        "test": test_metrics.set_index("model").round(4).to_dict(orient="index"),
        "test_ci_95": ci.round(4).to_dict(orient="index"),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    joblib.dump({"model": tuned, "threshold": threshold}, model_dir / "final_model.joblib")
    log(f"Done. Results in {out_dir.relative_to(ROOT)}/, model in {model_dir.relative_to(ROOT)}/final_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small grid and 3 folds for a fast check")
    main(parser.parse_args().quick)
