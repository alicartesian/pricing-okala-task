"""Development-only model comparison, rotating origins, routing and blending.

This module has no final-test evaluation branch. It reads source observations
only through the last development target end and protects the old result files.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import time

import jdatetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from okala import features as feature_module
from okala import modeling as model_module
from okala.features import FeatureConfig, KEYS, RetailFeaturePipeline, split_for_origin
from okala.modeling import (
    ExperimentConfig, METRICS, OBJECTIVES, _digest, _prepare_model_frame,
    baseline_predictions, fit_predict, regression_metrics,
)


@dataclass(frozen=True)
class RefinementConfig:
    model: ExperimentConfig = field(default_factory=ExperimentConfig)
    rotation_step_days: int = 5
    sparse_zero_rate: float = 0.2
    guardrail_factor: float = 1.05
    blend_weights: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def __post_init__(self):
        if self.rotation_step_days < 1 or self.rotation_step_days % 7 == 0:
            raise ValueError("Rotating origin spacing must be positive and not a multiple of seven")
        if not 0 <= self.sparse_zero_rate <= 1 or self.guardrail_factor < 1:
            raise ValueError("Invalid sparsity threshold or guardrail factor")
        if not self.blend_weights or any(not 0 < w <= 1 for w in self.blend_weights):
            raise ValueError("Blend weights must be in (0, 1]; baseline is added separately")


def development_plan(plan):
    dev = plan[plan.role.str.startswith("development_")].sort_values("forecast_origin").copy()
    test = plan[plan.role.eq("final_test")]
    if len(dev) != 3 or len(test) != 1:
        raise ValueError("Expected three development windows and one protected final window")
    ceiling = dev.target_end.max()
    if ceiling >= test.target_start.min():
        raise ValueError("Development outcomes overlap the protected final horizon")
    if not (dev.target_end - dev.forecast_origin).dt.days.eq(30).all():
        raise ValueError("This refinement implements the agreed 30-day horizon")
    calibration = pd.DatetimeIndex([dev.forecast_origin.min() - pd.Timedelta(days=n) for n in [90, 60, 30]])
    return dev, calibration, ceiling


def read_development_source(path, ceiling):
    """Push a date predicate into Parquet: protected future rows never enter pandas."""
    path, ceiling = Path(path), pd.Timestamp(ceiling)
    parquet = pq.ParquetFile(path)
    date_type = parquet.schema_arrow.field("date").type
    if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
        date_column = parquet.schema_arrow.get_field_index("date")
        stats = parquet.metadata.row_group(0).column(date_column).statistics
        if stats is None:
            raise ValueError("Date-string schema needs date statistics to identify its calendar encoding")
        first = str(stats.min)
        if re.fullmatch(r"1[34]\d{6}", first):
            bound = jdatetime.date.fromgregorian(date=ceiling.date()).strftime("%Y%m%d")
        else:
            bound = ceiling.strftime("%Y%m%d" if re.fullmatch(r"\d{8}", first) else "%Y-%m-%d")
    else:
        bound = ceiling
    result = pd.read_parquet(path, filters=[("date", "<=", bound)])
    if result.empty:
        raise ValueError("No observations before the development boundary")
    return result


def origin_grids(first_date, last_origin, development_origins, config):
    start = pd.Timestamp(first_date) + pd.Timedelta(days=89)
    common = pd.DatetimeIndex(development_origins)
    return {
        "weekly": pd.date_range(start, last_origin, freq="7D").union(common),
        "rotating": pd.date_range(start, last_origin, freq=f"{config.rotation_step_days}D").union(common),
    }


def fit_xgboost(model_frame, targets, train_positions, predict_positions, columns,
                categorical_columns, objective, config, *, model_path=None):
    """Native CPU hist trees; category vocabulary and constants use training only."""
    import xgboost as xgb
    if objective not in OBJECTIVES:
        raise ValueError("Unknown regression objective")
    train = model_frame.iloc[train_positions][columns].copy()
    used = train.columns[train.nunique(dropna=False).gt(1)].tolist()
    if not used:
        raise ValueError("No varying training features")
    train = train[used]
    predict = model_frame.iloc[predict_positions][used].copy()
    levels = {}
    for name in categorical_columns:
        if name not in used:
            continue
        levels[name] = sorted(train[name].unique().tolist())
        train[name] = pd.Categorical(train[name], categories=levels[name])
        known_prediction = predict[name].where(predict[name].isin(levels[name]))
        predict[name] = pd.Categorical(known_prediction, categories=levels[name])
    y = np.asarray(targets, float)[train_positions]
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training targets must be finite and nonnegative")
    if objective == "log_rmse":
        y = np.log1p(y)
    start = time.monotonic()
    training = xgb.DMatrix(train, label=y, enable_categorical=True, nthread=config.thread_count)
    prediction = xgb.DMatrix(predict, enable_categorical=True, nthread=config.thread_count)
    params = dict(tree_method="hist", device="cpu", max_depth=config.depth,
                  eta=config.learning_rate, reg_lambda=config.l2_leaf_reg,
                  objective="reg:absoluteerror" if objective == "raw_mae" else "reg:squarederror",
                  nthread=config.thread_count, seed=config.random_seed,
                  max_cat_to_onehot=4, max_cat_threshold=64, verbosity=0)
    model = xgb.train(params, training, num_boost_round=config.iterations)
    predicted = model.predict(prediction).astype(float)
    if objective == "log_rmse":
        predicted = np.expm1(predicted)
    predicted = np.maximum(predicted, 0)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite XGBoost predictions")
    if model_path is not None:
        model.save_model(model_path)
    info = dict(fit_seconds=time.monotonic() - start, used_features=len(used),
                used_feature_columns=used, category_levels=levels, params=params)
    return predicted, info


def macro_metrics(frame, predicted):
    predicted = np.asarray(predicted, float)
    rows = [regression_metrics(frame.actual.to_numpy()[mask], predicted[mask])
            for origin in frame.forecast_origin.unique()
            for mask in [frame.forecast_origin.eq(origin).to_numpy()]]
    return {name: float(pd.DataFrame(rows)[name].mean()) for name in METRICS}


def apply_policy(frame, policy, config):
    """Predictions depend on history and base predictions, never on actual outcomes."""
    base = frame.baseline.to_numpy()
    result = policy["weight"] * frame[policy["main"]].to_numpy() + (1 - policy["weight"]) * base
    sparse = frame.historical_zero_rate_90d.ge(config.sparse_zero_rate).to_numpy()
    result[sparse] = frame[policy["sparse"]].to_numpy()[sparse]
    return np.maximum(result, 0)


def choose_policy(calibration, origin, candidates, config):
    """Learn routing/blend choices only from fully available, earlier outcomes."""
    origin = pd.Timestamp(origin)
    if calibration.empty or not calibration.forecast_origin.lt(origin).all() or not calibration.target_end.le(origin).all():
        raise ValueError("Policy calibration must precede the origin and have fully known outcomes")
    base = macro_metrics(calibration, calibration.baseline)
    sparse_frame = calibration[calibration.historical_zero_rate_90d.ge(config.sparse_zero_rate)]
    routes = []
    sparse_choice = "baseline"
    if not sparse_frame.empty:
        sparse_baseline = macro_metrics(sparse_frame, sparse_frame.baseline)
        for candidate in ["baseline", *candidates]:
            score = macro_metrics(sparse_frame, sparse_frame[candidate])
            eligible = candidate == "baseline" or (
                np.isfinite(score["wape_pct"]) and np.isfinite(sparse_baseline["wape_pct"])
                and score["wape_pct"] <= sparse_baseline["wape_pct"] * config.guardrail_factor + 1e-12
            )
            routes.append(dict(candidate=candidate, eligible=bool(eligible), **score))
        sparse_choice = pd.DataFrame(routes).query("eligible").sort_values(["smape_pct", "wape_pct", "candidate"]).iloc[0].candidate
    policies = [dict(main="baseline", weight=0.0, sparse="baseline")]
    if sparse_choice != "baseline":
        policies.append(dict(main="baseline", weight=0.0, sparse=sparse_choice))
    policies.extend(dict(main=name, weight=float(weight), sparse=sparse_choice)
                    for name in candidates for weight in config.blend_weights)
    scores = []
    for policy in policies:
        metrics = macro_metrics(calibration, apply_policy(calibration, policy, config))
        eligible = (metrics["rmse"] <= base["rmse"] * config.guardrail_factor + 1e-12
                    and metrics["smape_pct"] <= base["smape_pct"] * config.guardrail_factor + 1e-12)
        scores.append(dict(**policy, eligible=bool(eligible), **metrics))
    table = pd.DataFrame(scores)
    allowed = table[table.eligible & table.wape_pct.notna()]
    if allowed.empty:
        raise ValueError("No measurable calibration policy; cannot select using undefined WAPE")
    selected = allowed.sort_values(["wape_pct", "smape_pct", "weight", "main"]).iloc[0]
    decision = dict(main=selected["main"], weight=float(selected["weight"]), sparse=selected["sparse"],
                    calibrated_through=str(calibration.target_end.max().date()),
                    calibration_origins=sorted(str(pd.Timestamp(t).date()) for t in calibration.forecast_origin.unique()),
                    calibration_metrics={name: float(selected[name]) for name in METRICS})
    return decision, table, pd.DataFrame(routes)


def _protected_hashes(directory):
    return {str(path): _digest(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def run_refinements(source="data/candidate_dataset.parquet", data_dir="data/processed",
                    output_dir="data/refinement", *, config=None, progress=print):
    import catboost
    import xgboost
    config = config or RefinementConfig()
    source, data_dir, output_dir = Path(source), Path(data_dir), Path(output_dir)
    protected_dir = data_dir.parent / "modeling"
    if any(output_dir.resolve() == directory.resolve() or directory.resolve() in output_dir.resolve().parents
           for directory in [protected_dir, data_dir]):
        raise ValueError("Refinement outputs must not overwrite the original modeling results")
    output_dir.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes(protected_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    original_plan = pd.read_csv(data_dir / "backtest_plan.csv", parse_dates=["forecast_origin", "target_start", "target_end"])
    dev, calibration_origins, ceiling = development_plan(original_plan)
    feature_config = FeatureConfig(**{**manifest["config"], "windows": tuple(manifest["config"]["windows"])})
    if feature_config.history_days != 90 or feature_config.horizon_days != 30:
        raise ValueError("Refinement origin schedule implements the agreed 90-day history and 30-day horizon")
    raw = read_development_source(source, ceiling)
    pipeline = RetailFeaturePipeline(raw, feature_config)
    assert pipeline.dates[-1] <= ceiling
    grids = origin_grids(pipeline.dates[0], dev.forecast_origin.max(), dev.forecast_origin, config)
    all_origins = grids["weekly"].union(grids["rotating"]).union(calibration_origins).sort_values()
    if (all_origins + pd.Timedelta(days=30) > ceiling).any():
        raise AssertionError("A refinement target would enter the protected final horizon")
    fingerprint_payload = dict(config=asdict(config), source=_digest(source),
        manifest=_digest(data_dir / "manifest.json"), plan=_digest(data_dir / "backtest_plan.csv"),
        ceiling=str(ceiling.date()), source_rows=len(raw), origins=[str(t.date()) for t in all_origins],
        features_implementation=_digest(feature_module.__file__), modeling_implementation=_digest(model_module.__file__),
        refinement_implementation=_digest(__file__), catalog=_digest(data_dir / "feature_catalog.csv"),
        versions=dict(pandas=pd.__version__, numpy=np.__version__, catboost=catboost.__version__, xgboost=xgboost.__version__))
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    cache = output_dir / "checkpoints" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "features.parquet").exists() and (cache / "targets.parquet").exists():
        features, labels = pd.read_parquet(cache / "features.parquet"), pd.read_parquet(cache / "targets.parquet")
    else:
        def report(i, total, origin):
            if progress and (i == 1 or i % 20 == 0 or i == total):
                progress(f"Development features: {i}/{total} origins through {origin.date()}")
        features, labels = pipeline.build_dataset(all_origins, progress=report)
        features.to_parquet(cache / "features.parquet", index=False)
        labels.to_parquet(cache / "targets.parquet", index=False)
    features = features.sort_values(KEYS).reset_index(drop=True)
    labels = features[KEYS].merge(labels, on=KEYS, how="left", sort=False, validate="one_to_one")
    # Build definitions even when loading snapshots from cache.
    pipeline.features_at(dev.forecast_origin.min())
    catalog = pipeline.catalog
    catalog.to_csv(output_dir / "feature_catalog.csv", index=False)
    columns_all = pipeline.model_feature_columns
    columns_reduced = catalog.loc[catalog.role.eq("feature") & catalog.family.ne("missingness_metadata"), "name"].tolist()
    matrix, categories = _prepare_model_frame(features, {**manifest, "model_feature_columns": columns_all})
    del pipeline, raw
    y = labels.target_sales_30d.to_numpy()
    specs = {"weekly_catboost_raw_mae_no_metadata": dict(cadence="weekly", algorithm="catboost", objective="raw_mae", feature_set="no_metadata")}
    for algorithm in ["catboost", "xgboost"]:
        for objective in OBJECTIVES:
            for feature_set in ["full", "no_metadata"]:
                name = f"rotating_{algorithm}_{objective}_{feature_set}"
                specs[name] = dict(cadence="rotating", algorithm=algorithm, objective=objective, feature_set=feature_set)
    protocol = {**fingerprint_payload, "fingerprint": fingerprint, "final_test_scored": False,
        "source_read_predicate_end": str(ceiling.date()), "development_only": True,
        "calibration_origins": [str(t.date()) for t in calibration_origins],
        "development_origins": [str(t.date()) for t in dev.forecast_origin],
        "candidates": specs, "categorical_handling": "Train-only vocabulary/native categorical splits for XGBoost; native one-hot for CatBoost",
        "policy_selection": "Earlier completed horizons only; sparse sMAPE with sparse WAPE guardrail, then overall WAPE with RMSE/sMAPE guardrails",
        "guardrail_interpretation": "5% calibration tolerance is a declared experiment setting, not a business-validated cost bound",
        "training_grid": "Cadence grid plus common development origins; calibration-only origins are prediction-only unless naturally on that grid",
        "feature_rows": len(features), "features": len(columns_all), "reduced_features": len(columns_reduced)}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    origin_roles = {origin: f"calibration_{i}" for i, origin in enumerate(calibration_origins, 1)}
    origin_roles.update(dict(zip(dev.forecast_origin, dev.role)))
    prediction_frames, scores, weekday_rows, fitting_rows = [], [], [], []
    decisions, policy_candidate_rows, route_rows = [], [], []
    for origin, role in sorted(origin_roles.items()):
        split = split_for_origin(features, labels, origin)
        evaluation = split["evaluation"]
        frame = features.iloc[evaluation][KEYS].reset_index(drop=True)
        frame["target_end"] = labels.iloc[evaluation].target_end.to_numpy()
        frame["actual"] = y[evaluation]
        frame["baseline"] = baseline_predictions(features.iloc[evaluation])
        frame["historical_zero_rate_90d"] = features.iloc[evaluation].zero_sales_rate_90d.to_numpy()
        frame["role"] = role
        baseline_train = None
        for name, spec in specs.items():
            is_grid = features.forecast_origin.isin(grids[spec["cadence"]]).to_numpy()
            train = split["train"][is_grid[split["train"]]]
            if not labels.iloc[train].target_end.le(origin).all():
                raise AssertionError("Training labels cross the forecast cutoff")
            cols = columns_all if spec["feature_set"] == "full" else columns_reduced
            pred_path, info_path = cache / f"{name}__{origin.date()}.parquet", cache / f"{name}__{origin.date()}.json"
            if pred_path.exists() and info_path.exists():
                saved = pd.read_parquet(pred_path)
                if not saved[KEYS].equals(frame[KEYS]):
                    raise ValueError("Checkpoint keys do not match the evaluation cohort")
                predicted, info = saved.predicted.to_numpy(), json.loads(info_path.read_text())
            else:
                if progress:
                    progress(f"Fit {name}, {role}: train={len(train):,}, score={len(evaluation):,}")
                model_path = None
                if origin == dev.forecast_origin.max():
                    model_path = cache / f"{name}.{'cbm' if spec['algorithm'] == 'catboost' else 'ubj'}"
                if spec["algorithm"] == "catboost":
                    predicted, info, _ = fit_predict(matrix, y, train, evaluation, cols, categories, spec["objective"], config.model, model_path=model_path)
                else:
                    predicted, info = fit_xgboost(matrix, y, train, evaluation, cols, categories, spec["objective"], config.model, model_path=model_path)
                saved = frame[KEYS].copy(); saved["predicted"] = predicted
                saved.to_parquet(pred_path, index=False)
                info_path.write_text(json.dumps(info, indent=2) + "\n")
            frame[name] = predicted
            fitting_rows.append(dict(candidate=name, role=role, forecast_origin=origin, train_samples=len(train),
                                     used_features=info["used_features"], declared_features=len(cols), fit_seconds=info["fit_seconds"], **spec))
            train_dates = features.iloc[train].forecast_origin.drop_duplicates()
            if name in ["weekly_catboost_raw_mae_no_metadata", "rotating_catboost_raw_mae_no_metadata"]:
                for weekday in range(7):
                    weekday_rows.append(dict(role=role, candidate=name, weekday=weekday,
                                             training_origins=int(train_dates.dt.day_of_week.eq(weekday).sum()),
                                             training_samples=len(train)))
            if spec["cadence"] == "rotating":
                baseline_train = train
            if progress:
                score = regression_metrics(frame.actual, predicted)
                progress(f"  WAPE={score['wape_pct']:.3f}%, RMSE={score['rmse']:.2f}, sMAPE={score['smape_pct']:.3f}%, fit={info['fit_seconds']:.1f}s")
        threshold = np.quantile(baseline_predictions(features.iloc[baseline_train]), [1 / 3, 2 / 3])
        frame["volume_segment"] = np.select([frame.baseline.le(threshold[0]), frame.baseline.le(threshold[1])], ["low_volume", "mid_volume"], default="high_volume")
        # Current outcomes are never passed to policy selection.
        if role.startswith("development_"):
            calibration = pd.concat(prediction_frames, ignore_index=True)
            calibration = calibration[calibration.forecast_origin.lt(origin) & calibration.target_end.le(origin)]
            decision, policy_scores, sparse_scores = choose_policy(calibration, origin, list(specs), config)
            decisions.append(dict(forecast_origin=str(origin.date()), **decision))
            (output_dir / "selected_policies.json").write_text(json.dumps(decisions, indent=2) + "\n")
            frame["refined_policy"] = apply_policy(frame, decision, config)
            policy_candidate_rows.append(policy_scores.assign(forecast_origin=origin))
            route_rows.append(sparse_scores.assign(forecast_origin=origin))
            if progress:
                progress(f"Policy frozen at {origin.date()}: main={decision['main']}, weight={decision['weight']}, sparse={decision['sparse']}; calibrated through {decision['calibrated_through']}")
        prediction_frames.append(frame)
        for name in ["baseline", *specs, *(["refined_policy"] if "refined_policy" in frame else [])]:
            scores.append(dict(candidate=name, role=role, forecast_origin=origin, **regression_metrics(frame.actual, frame[name])))
        pd.DataFrame(scores).to_csv(output_dir / "all_window_metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    pd.DataFrame(fitting_rows).to_csv(output_dir / "fit_audit.csv", index=False)
    pd.DataFrame(weekday_rows).to_csv(output_dir / "training_weekdays.csv", index=False)
    pd.concat(policy_candidate_rows, ignore_index=True).to_csv(output_dir / "policy_calibration_scores.csv", index=False)
    pd.concat(route_rows, ignore_index=True).to_csv(output_dir / "sparse_route_calibration_scores.csv", index=False)
    metrics = pd.DataFrame(scores)
    reported = metrics[metrics.role.str.startswith("development_")]
    reported.to_csv(output_dir / "development_metrics.csv", index=False)
    leaderboard = reported.groupby("candidate", as_index=False)[METRICS].mean().sort_values("wape_pct")
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    old_development = protected_dir / "development_metrics.csv"
    if old_development.exists():
        old_scores = pd.read_csv(old_development)
        old_scores = old_scores[old_scores.candidate.eq("drop_missingness_metadata")][["role", "wape_pct"]]
        control = reported[reported.candidate.eq("weekly_catboost_raw_mae_no_metadata")][["role", "wape_pct"]]
        parity = old_scores.merge(control, on="role", suffixes=("_original", "_control"), validate="one_to_one")
        parity["difference_pp"] = parity.wape_pct_control - parity.wape_pct_original
        parity.to_csv(output_dir / "weekly_control_parity.csv", index=False)
    segments = []
    for role, part in predictions[predictions.role.str.startswith("development_")].groupby("role"):
        groups = {name: part.volume_segment.eq(name) for name in ["low_volume", "mid_volume", "high_volume"]}
        groups.update(sparse=part.historical_zero_rate_90d.ge(config.sparse_zero_rate), regular=part.historical_zero_rate_90d.lt(config.sparse_zero_rate))
        for segment, mask in groups.items():
            batch = part[mask]
            if batch.empty:
                continue
            for name in ["baseline", *specs, "refined_policy"]:
                segments.append(dict(role=role, segment=segment, candidate=name, **regression_metrics(batch.actual, batch[name])))
    pd.DataFrame(segments).to_csv(output_dir / "segment_metrics.csv", index=False)
    original_unchanged = protected_before == _protected_hashes(protected_dir)
    if not original_unchanged:
        raise AssertionError("Original modeling artifacts changed")
    preservation = dict(original_artifacts_unchanged=True, final_test_scored=False,
                        source_read_through=str(ceiling.date()), protected_files=protected_before)
    (output_dir / "holdout_preservation.json").write_text(json.dumps(preservation, indent=2) + "\n")
    result = dict(protocol=protocol, leaderboard=leaderboard.to_dict("records"), policies=decisions,
                  original_artifacts_unchanged=original_unchanged, final_test_scored=False,
                  checkpoint_directory=str(cache), limitations=[
                      "These are development results; no new external or untouched test is claimed",
                      "Guardrails constrain calibration performance, not guaranteed future performance",
                      "Equal tree budgets do not mean identical algorithms or identical wall-clock costs",
                      "The reduced feature set was previously selected on development data",
                  ])
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if progress:
        progress(leaderboard.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/candidate_dataset.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/refinement"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run_refinements(args.source, args.data_dir, args.output_dir,
                    config=RefinementConfig(model=ExperimentConfig(iterations=args.iterations, thread_count=args.threads)),
                    progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
