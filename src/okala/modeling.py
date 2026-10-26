"""Reproducible baseline, CatBoost, and development-only feature-family ablations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from okala.features import KEYS, split_for_origin


FAMILY_DESCRIPTIONS = {
    "main": ("Main features", "Product, brand and hierarchy identity; last-known usable measurements"),
    "aggregated": ("Aggregated history", "Observed/complete sums, means, spread, extrema, medians and calendar-decayed means"),
    "deviation": ("Difference and deviation", "Sales z-score and short/long rate differences; price and discount deviations"),
    "domain": ("Handcrafted domain", "Sales per stocked store, inventory-cover and intermittency proxies, named calendar dates"),
    "temporal": ("Temporal windows and lags", "Exact calendar lags and positive/zero/discounted sales-day counts and rates"),
    "missingness_metadata": ("Missingness and metadata", "Coverage, gap runs, measurement ages, zero-price and inventory-quality flags"),
    "velocity_acceleration": ("Velocity and acceleration", "Disjoint-week fractional trends, first/second differences and price/store changes"),
    "higher_order": ("Higher-order statistics", "Skewness, excess kurtosis, IQR, tail spread, CV, entropy and concentration"),
    "cyclical": ("Cyclical transformations", "Weekday/Jalali sine-cosine encodings and future horizon calendar composition"),
    "competition": ("Competition", "Peer prices, discount gaps, availability/reporting counts, shares and sales trends"),
}
OBJECTIVES = ("raw_rmse", "raw_mae", "log_rmse")
METRICS = ["wape_pct", "mae", "rmse", "smape_pct", "bias_pct"]


@dataclass(frozen=True)
class ExperimentConfig:
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.07
    l2_leaf_reg: float = 5.0
    thread_count: int = 8
    random_seed: int = 42

    def __post_init__(self):
        if min(self.iterations, self.depth, self.thread_count) < 1:
            raise ValueError("Iterations, depth and thread count must be positive")
        if not 0 < self.learning_rate <= 1 or self.depth > 16 or self.l2_leaf_reg < 0:
            raise ValueError("Invalid CatBoost configuration")


def feature_summary(catalog):
    """Exactly one primary family per model column; keys/eligibility are excluded."""
    rows = []
    for family, (title, description) in FAMILY_DESCRIPTIONS.items():
        columns = catalog.loc[catalog.role.eq("feature") & catalog.family.eq(family), "name"]
        rows.append(dict(family=family, category=title, count=len(columns),
                         description=description, examples=", ".join(columns.head(3))))
    result = pd.DataFrame(rows)
    if result["count"].sum() != catalog.role.eq("feature").sum():
        raise ValueError("Feature catalog contains an unrecognized family")
    return result


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Metrics require matching, nonempty one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual < 0).any():
        raise ValueError("Metrics require finite values and nonnegative actual sales")
    absolute = np.abs(predicted - actual)
    volume = actual.sum()
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(2 * absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    return dict(n=len(actual), actual_volume=float(volume),
                wape_pct=float(100 * absolute.sum() / volume) if volume > 0 else np.nan,
                mae=float(absolute.mean()), rmse=float(np.sqrt(np.mean((predicted - actual) ** 2))),
                smape_pct=float(100 * smape.mean()),
                bias_pct=float(100 * (predicted - actual).sum() / volume) if volume > 0 else np.nan)


def baseline_predictions(features):
    """30 times the observed-day mean of the preceding 30 calendar days."""
    predictions = 30 * features.sales_mean_30d.to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("The baseline comparison cohort requires an observed trailing sales rate")
    return np.maximum(predictions, 0)


def select_development_candidate(metrics):
    """Rank by equal-fold WAPE, then sMAPE, then name; reject test contamination."""
    if metrics.role.eq("final_test").any() or not metrics.role.str.startswith("development_").all():
        raise ValueError("Model selection accepts development folds only")
    expected = set(metrics.role.unique())
    for _, group in metrics.groupby("candidate"):
        if set(group.role) != expected or group.role.duplicated().any():
            raise ValueError("Every candidate must have exactly the same development folds")
    table = metrics.groupby(["candidate", "objective"], as_index=False)[METRICS].mean()
    if table.wape_pct.isna().any():
        raise ValueError("Undefined WAPE cannot select a candidate")
    return table.sort_values(["wape_pct", "smape_pct", "candidate"]).reset_index(drop=True)


def _prepare_model_frame(features, manifest):
    columns = manifest["model_feature_columns"]
    forbidden = set(manifest["target_columns"]) | {"forecast_origin", "history_eligible"}
    if set(columns) & forbidden:
        raise ValueError("Feature whitelist contains target or split metadata")
    result = features[columns].copy()
    categories = [name for name in manifest["categorical_feature_columns"] if name in columns]
    for name in columns:
        result[name] = result[name].fillna("__UNKNOWN__").astype(str) if name in categories else result[name].astype(np.float32)
    return result, categories


def fit_predict(model_frame, targets, train_positions, predict_positions, columns,
                categorical_columns, objective, config, *, model_path=None):
    """Fixed-budget fit: no evaluation labels, eval_set, or early stopping."""
    from catboost import CatBoostRegressor

    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if not len(train_positions) or not len(predict_positions):
        raise ValueError("Training and prediction cohorts must be nonempty")
    train = model_frame.iloc[train_positions][columns]
    # Constant/all-missing removal uses training observations only.
    used = train.columns[train.nunique(dropna=False).gt(1)].tolist()
    if not used:
        raise ValueError("No varying training features")
    train = train[used]
    predict = model_frame.iloc[predict_positions][used]
    cats = [name for name in categorical_columns if name in used]
    y = np.asarray(targets, dtype=float)[train_positions]
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training targets must be complete, finite and nonnegative")
    if objective == "log_rmse":
        y = np.log1p(y)
    # Native one-hot splits disable label-derived categorical statistics entirely.
    # This avoids encoding overlapping training labels into earlier-origin rows.
    one_hot_limit = max([2] + [train[name].nunique() for name in cats])
    model = CatBoostRegressor(
        iterations=config.iterations, depth=config.depth, learning_rate=config.learning_rate,
        l2_leaf_reg=config.l2_leaf_reg, loss_function="MAE" if objective == "raw_mae" else "RMSE",
        thread_count=config.thread_count, random_seed=config.random_seed, has_time=True,
        one_hot_max_size=one_hot_limit, nan_mode="Min", allow_writing_files=False, verbose=False,
    )
    start = time.monotonic()
    model.fit(train, y, cat_features=cats)
    predictions = np.asarray(model.predict(predict), dtype=float)
    if objective == "log_rmse":
        predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced nonfinite forecasts")
    if model_path is not None:
        model.save_model(str(model_path))
    importance = pd.DataFrame({"feature": used, "importance": model.feature_importances_})
    return predictions, dict(fit_seconds=time.monotonic() - start, used_features=len(used),
                             used_feature_columns=used, native_one_hot_max_size=one_hot_limit), importance


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_experiments(data_dir="data/processed", output_dir="data/modeling", *,
                    config=None, progress=print):
    """Select using development only, freeze selection, then score final test once."""
    config = config or ExperimentConfig()
    import catboost
    versions = {"pandas": pd.__version__, "numpy": np.__version__, "catboost": catboost.__version__}
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    features = pd.read_parquet(data_dir / "features.parquet")
    labels = pd.read_parquet(data_dir / "targets.parquet")
    catalog = pd.read_csv(data_dir / "feature_catalog.csv")
    plan = pd.read_csv(data_dir / "backtest_plan.csv", parse_dates=["forecast_origin", "target_start", "target_end"])
    features = features.sort_values(KEYS).reset_index(drop=True)
    labels = features[KEYS].merge(labels, on=KEYS, how="left", sort=False, validate="one_to_one")
    if labels.label_complete.isna().any():
        raise ValueError("Missing label-status rows")
    if labels.label_complete.ne(labels.target_sales_30d.notna()).any():
        raise ValueError("Target completeness and target values disagree")
    if not features[KEYS].equals(labels[KEYS]):
        raise ValueError("Misaligned feature/target keys")
    development = plan.loc[plan.role.str.startswith("development_")].copy()
    test = plan.loc[plan.role.eq("final_test")]
    if len(development) != 3 or len(test) != 1:
        raise ValueError("This experiment requires three development origins and one final test")
    test = test.iloc[0]
    model_frame, categories = _prepare_model_frame(features, manifest)
    y = labels.target_sales_30d.to_numpy(dtype=float)
    feature_summary(catalog).to_csv(output_dir / "feature_summary.csv", index=False)
    fingerprint = hashlib.sha256(json.dumps({
        "config": asdict(config), "source": manifest["source_sha256"],
        "features": _digest(data_dir / "features.parquet"),
        "targets": _digest(data_dir / "targets.parquet"),
        "catalog": _digest(data_dir / "feature_catalog.csv"),
        "manifest": _digest(data_dir / "manifest.json"),
        "plan": _digest(data_dir / "backtest_plan.csv"), "versions": versions,
        "implementation": _digest(__file__),
    }, sort_keys=True).encode()).hexdigest()
    checkpoint_dir = output_dir / "checkpoints" / fingerprint[:16]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(config=asdict(config), fingerprint=fingerprint,
                    development_origins=[str(t.date()) for t in development.forecast_origin],
                    final_test_origin=str(test.forecast_origin.date()),
                    objectives=list(OBJECTIVES), selection_metric="Equal-fold mean development WAPE",
                    baseline="30 * observed trailing-30-day mean; an explicit rate forecast where history is incomplete",
                    target="Direct next-30-calendar-day observed sales total",
                    early_stopping=False, target_categorical_statistics=False,
                    preprocessing="Training-only constant removal; no external scaling or numeric imputation",
                    ablation="Drop each catalog family separately; other families may retain related derived information",
                    iterations_fixed_before_scoring=True)
    (output_dir / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    metrics, predictions = [], []
    columns_all = manifest["model_feature_columns"]
    candidate_columns = {}
    candidate_objective = {}

    def score_development(candidate, objective, columns):
        candidate_columns[candidate] = columns
        candidate_objective[candidate] = objective
        for row in development.itertuples():
            split = split_for_origin(features, labels, row.forecast_origin)
            train, evaluation = split["train"], split["evaluation"]
            if not labels.iloc[train].target_end.le(row.forecast_origin).all():
                raise AssertionError("Training outcomes cross the evaluation boundary")
            path = checkpoint_dir / f"{candidate}__{row.role}.parquet"
            info_path = path.with_suffix(".json")
            if path.exists() and info_path.exists():
                batch = pd.read_parquet(path)
                info = json.loads(info_path.read_text())
                if not features.iloc[evaluation][KEYS].reset_index(drop=True).equals(batch[KEYS]):
                    raise ValueError("Cached prediction keys mismatch")
                predicted = batch.predicted.to_numpy()
            else:
                if progress:
                    progress(f"Fit {candidate}, {row.role}: train={len(train):,}, evaluate={len(evaluation):,}, columns={len(columns)}")
                predicted, info, _ = fit_predict(model_frame, y, train, evaluation, columns, categories, objective, config)
                batch = features.iloc[evaluation][KEYS].reset_index(drop=True)
                batch["predicted"] = predicted
                batch.to_parquet(path, index=False)
                info_path.write_text(json.dumps(info, indent=2) + "\n")
            metric = dict(candidate=candidate, objective=objective, role=row.role,
                          forecast_origin=row.forecast_origin, train_samples=len(train),
                          declared_features=len(columns), used_features=info["used_features"],
                          fit_seconds=info["fit_seconds"], **regression_metrics(y[evaluation], predicted))
            metrics.append(metric)
            batch = batch.copy()
            batch["actual"] = y[evaluation]
            batch["candidate"], batch["role"] = candidate, row.role
            batch["historical_30d_rate_forecast"] = baseline_predictions(features.iloc[evaluation])
            batch["historical_zero_rate_90d"] = features.iloc[evaluation].zero_sales_rate_90d.to_numpy()
            training_rates = baseline_predictions(features.iloc[train])
            low, high = np.quantile(training_rates, [1 / 3, 2 / 3])
            batch["volume_segment"] = np.select(
                [batch.historical_30d_rate_forecast.le(low), batch.historical_30d_rate_forecast.le(high)],
                ["low_volume", "mid_volume"], default="high_volume")
            predictions.append(batch)
            pd.DataFrame(metrics).to_csv(output_dir / "development_metrics.csv", index=False)
            if progress:
                progress(f"  WAPE={metric['wape_pct']:.3f}%, MAE={metric['mae']:.2f}, sMAPE={metric['smape_pct']:.3f}%, fit={info['fit_seconds']:.1f}s")

    baseline_metrics = []
    for row in development.itertuples():
        positions = split_for_origin(features, labels, row.forecast_origin)["evaluation"]
        baseline_metrics.append(dict(candidate="baseline", objective="none", role=row.role,
                                     forecast_origin=row.forecast_origin,
                                     **regression_metrics(y[positions], baseline_predictions(features.iloc[positions]))))
    for objective in OBJECTIVES:
        score_development(f"full_{objective}", objective, columns_all)
    objective_comparison = select_development_candidate(pd.DataFrame(metrics))
    objective_comparison.to_csv(output_dir / "objective_comparison.csv", index=False)
    winning_full = objective_comparison.iloc[0]
    objective = winning_full.objective
    if progress:
        progress(f"Objective chosen on development folds: {objective}, mean WAPE={winning_full.wape_pct:.3f}%")

    families = catalog[catalog.role.eq("feature")].groupby("family").name.apply(list).to_dict()
    for family, removed in families.items():
        score_development(f"drop_{family}", objective, [name for name in columns_all if name not in removed])
    dev_metrics = pd.DataFrame(metrics)
    ablation_metrics = dev_metrics.loc[dev_metrics.candidate.eq(winning_full.candidate) | dev_metrics.candidate.str.startswith("drop_")]
    ablation = select_development_candidate(ablation_metrics)
    full_by_fold = ablation_metrics.loc[ablation_metrics.candidate.eq(winning_full.candidate)].set_index("role").wape_pct
    ablation["wape_change_vs_full_pp"] = ablation.wape_pct - winning_full.wape_pct
    ablation["declared_features"] = ablation.candidate.map(lambda name: len(candidate_columns[name]))
    ablation["folds_better_than_full"] = ablation.candidate.map(
        lambda name: int((ablation_metrics[ablation_metrics.candidate.eq(name)].set_index("role").wape_pct < full_by_fold).sum()))
    ablation.to_csv(output_dir / "ablation_summary.csv", index=False)
    selected = ablation.iloc[0].candidate
    selected_objective = candidate_objective[selected]
    selected_columns = candidate_columns[selected]
    selection = dict(candidate=selected, objective=selected_objective,
                     declared_feature_columns=selected_columns,
                     mean_development_wape_pct=float(ablation.iloc[0].wape_pct),
                     fingerprint=fingerprint, selected_using="Development outcomes only",
                     final_test_used_for_selection=False)
    # This decision is persisted before any final-test predictions or scores.
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    pd.concat([pd.DataFrame(baseline_metrics), dev_metrics], ignore_index=True).to_csv(output_dir / "development_metrics.csv", index=False)
    all_dev_predictions = pd.concat(predictions, ignore_index=True)
    all_dev_predictions.to_parquet(output_dir / "development_predictions.parquet", index=False)

    if progress:
        progress(f"Selection frozen: {selected}. Evaluating the reserved final horizon.")
    split = split_for_origin(features, labels, test.forecast_origin)
    train, evaluation = split["train"], split["evaluation"]
    final_cache = checkpoint_dir / "final_predictions.parquet"
    final_info = checkpoint_dir / "final_fit.json"
    model_cache = checkpoint_dir / "test_model.cbm"
    importance_cache = checkpoint_dir / "final_feature_importance.csv"
    if all(path.exists() for path in [final_cache, final_info, model_cache, importance_cache]):
        final_predictions = pd.read_parquet(final_cache)
        info = json.loads(final_info.read_text())
        importance = pd.read_csv(importance_cache)
    else:
        predicted, info, importance = fit_predict(model_frame, y, train, evaluation, selected_columns,
                                                categories, selected_objective, config,
                                                model_path=model_cache)
        final_predictions = features.iloc[evaluation][KEYS].reset_index(drop=True)
        final_predictions["actual"] = y[evaluation]
        final_predictions["main_prediction"] = predicted
        final_predictions["baseline_prediction"] = baseline_predictions(features.iloc[evaluation])
        final_predictions["historical_zero_rate_90d"] = features.iloc[evaluation].zero_sales_rate_90d.to_numpy()
        thresholds = np.quantile(baseline_predictions(features.iloc[train]), [1 / 3, 2 / 3])
        final_predictions["volume_segment"] = np.select(
            [final_predictions.baseline_prediction.le(thresholds[0]), final_predictions.baseline_prediction.le(thresholds[1])],
            ["low_volume", "mid_volume"], default="high_volume")
        final_predictions.to_parquet(final_cache, index=False)
        final_info.write_text(json.dumps(info, indent=2) + "\n")
        importance = importance.merge(catalog[["name", "family"]], left_on="feature", right_on="name", how="left").drop(columns="name")
        importance.sort_values("importance", ascending=False).to_csv(importance_cache, index=False)
    shutil.copy2(model_cache, output_dir / "test_model.cbm")
    shutil.copy2(importance_cache, output_dir / "final_feature_importance.csv")
    final_predictions.to_parquet(output_dir / "final_predictions.parquet", index=False)
    final_metrics = pd.DataFrame([
        dict(candidate=name, role="final_test", forecast_origin=test.forecast_origin,
             train_samples=len(train) if name != "baseline" else 0,
             **regression_metrics(final_predictions.actual, final_predictions[column]))
        for name, column in [("baseline", "baseline_prediction"), (selected, "main_prediction")]
    ])
    final_metrics.to_csv(output_dir / "final_metrics.csv", index=False)
    (output_dir / "final_fit.json").write_text(json.dumps(info, indent=2) + "\n")

    # Segment membership is based on history, never on future target quantiles.
    segment_rows = []
    selected_dev = all_dev_predictions[all_dev_predictions.candidate.eq(selected)]
    for role in [*development.role, "final_test"]:
        if role == "final_test":
            batch = final_predictions.rename(columns={"main_prediction": "predicted", "baseline_prediction": "historical_30d_rate_forecast"})
        else:
            batch = selected_dev[selected_dev.role.eq(role)]
        segments = {name: batch.volume_segment.eq(name) for name in ["low_volume", "mid_volume", "high_volume"]}
        segments.update(sparse=batch.historical_zero_rate_90d.ge(.2), regular=batch.historical_zero_rate_90d.lt(.2))
        for segment, mask in segments.items():
            part = batch[mask]
            if part.empty:
                continue
            for name, column in [("baseline", "historical_30d_rate_forecast"), (selected, "predicted")]:
                segment_rows.append(dict(role=role, segment=segment, candidate=name,
                                         **regression_metrics(part.actual, part[column])))
    pd.DataFrame(segment_rows).to_csv(output_dir / "segment_metrics.csv", index=False)

    comparison = pd.DataFrame(baseline_metrics + [row for row in metrics if row["candidate"] == selected])
    comparison.to_csv(output_dir / "selected_development_comparison.csv", index=False)
    result = dict(selection=selection, protocol=protocol,
                  baseline_mean_development_wape_pct=float(pd.DataFrame(baseline_metrics).wape_pct.mean()),
                  main_mean_development_wape_pct=float(ablation.iloc[0].wape_pct),
                  final_metrics=final_metrics.assign(forecast_origin=lambda frame: frame.forecast_origin.astype(str)).to_dict("records"),
                  feature_count=len(columns_all), selected_feature_count=len(selected_columns),
                  fitted_feature_count=info["used_features"],
                  limitations=["Complete-label cohort excludes unknown outcomes; see backtest coverage",
                               "Ablations measure conditional value with correlated/derived information retained elsewhere",
                               "Three development horizons and one final horizon do not establish significance or year-round performance",
                               "No uncertainty intervals, verified holiday calendar, future commercial plans or lost-sales correction",
                               "Full-dataset EDA preceded holdout designation; this is a reserved model-selection holdout, not a blinded external test"],
                  library_versions=versions)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if progress:
        progress(final_metrics.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/modeling"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run_experiments(args.data_dir, args.output_dir,
                    config=ExperimentConfig(iterations=args.iterations, thread_count=args.threads),
                    progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
