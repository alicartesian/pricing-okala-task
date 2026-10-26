"""Read-only statistical and explanatory analysis of saved retail forecasts.

Regression diagnostics are primary. Confusion matrices describe explicitly
defined alerts derived from forecasts, not a classifier or stockout detector.
No model is trained, selected, calibrated, or threshold-optimized here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from okala.features import KEYS, split_for_origin
from okala.modeling import _digest, _prepare_model_frame, regression_metrics
from okala.refinement import _protected_hashes


REFERENCE = "weekly_catboost_raw_mae_no_metadata"


@dataclass(frozen=True)
class DiagnosticConfig:
    bootstrap_repetitions: int = 2000
    permutation_repetitions: int = 5000
    seed: int = 42
    cluster_column: str = "Level4"
    min_profile_group: int = 5
    high_demand_quantiles: tuple[float, ...] = (0.5, 0.75, 0.9)
    growth_factors: tuple[float, ...] = (1.1, 1.25, 1.5)

    def __post_init__(self):
        if min(self.bootstrap_repetitions, self.permutation_repetitions) < 100:
            raise ValueError("Use at least 100 resamples")
        if self.min_profile_group < 2 or any(not 0 < q < 1 for q in self.high_demand_quantiles):
            raise ValueError("Invalid profile size or quantiles")


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def ratio(a, b):
    return float(a / b) if np.isfinite(b) and b > 0 else np.nan


def standardized_difference(a, b):
    """Signed marginal SMD: (mean(a)-mean(b))/sqrt((var(a)+var(b))/2)."""
    a, b = _finite(a), _finite(b)
    # Repeated non-binary floats can have tiny spurious variance/mean shifts.
    # Use exact constant values rather than dividing floating-point roundoff.
    constant_a, constant_b = bool(len(a) and np.ptp(a) == 0), bool(len(b) and np.ptp(b) == 0)
    ma = float(a[0] if constant_a else a.mean()) if len(a) else np.nan
    mb = float(b[0] if constant_b else b.mean()) if len(b) else np.nan
    va = 0.0 if constant_a else a.var(ddof=1) if len(a) >= 2 else np.nan
    vb = 0.0 if constant_b else b.var(ddof=1) if len(b) >= 2 else np.nan
    pooled = np.sqrt((va + vb) / 2) if min(len(a), len(b)) >= 2 else np.nan
    delta = ma - mb
    smd = delta / pooled if np.isfinite(pooled) and pooled > 0 else (0.0 if pooled == 0 and delta == 0 else np.nan)
    return dict(n_a=len(a), n_b=len(b), mean_a=ma, mean_b=mb, mean_difference=delta,
                pooled_sd=float(pooled), smd=float(smd),
                constant_shift=bool(pooled == 0 and np.isfinite(delta) and delta != 0))


def ks_distance(a, b):
    """Exact empirical CDF distance with ties; no independence-based p-value."""
    a, b = np.sort(_finite(a)), np.sort(_finite(b))
    if not len(a) or not len(b):
        return dict(ks_d=np.nan, ks_location=np.nan, ks_sign=np.nan)
    grid = np.unique(np.r_[a, b])
    difference = np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b)
    i = int(np.argmax(np.abs(difference)))
    return dict(ks_d=float(abs(difference[i])), ks_location=float(grid[i]), ks_sign=float(np.sign(difference[i])))


def ranking_metrics(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite and aligned")
    positive, negative = int(labels.sum()), int((~labels).sum())
    auc = (stats.rankdata(scores)[labels].sum() - positive * (positive + 1) / 2) / (positive * negative) if positive and negative else np.nan
    average_precision = np.nan
    if positive:
        order = np.argsort(-scores, kind="stable")
        sorted_scores, sorted_labels = scores[order], labels[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
        tp = np.cumsum(sorted_labels)[ends]
        recall, precision = tp / positive, tp / (ends + 1)
        average_precision = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    separation = ks_distance(scores[labels], scores[~labels])
    directional = np.nan
    if positive and negative:
        grid = np.unique(scores)
        pos_cdf = np.searchsorted(np.sort(scores[labels]), grid, side="right") / positive
        neg_cdf = np.searchsorted(np.sort(scores[~labels]), grid, side="right") / negative
        directional = float(np.max(neg_cdf - pos_cdf))
    return dict(roc_auc=float(auc), average_precision=average_precision,
                score_ks_d=separation["ks_d"], score_ks_location=separation["ks_location"],
                directional_ks=directional)


def confusion_indices(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Confusion labels must be nonempty and aligned")
    tp, tn = int((actual & predicted).sum()), int((~actual & ~predicted).sum())
    fp, fn = int((~actual & predicted).sum()), int((actual & ~predicted).sum())
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(n=len(actual), TP=tp, TN=tn, FP=fp, FN=fn,
                prevalence=ratio(tp + fn, len(actual)), accuracy=ratio(tp + tn, len(actual)),
                recall=sensitivity, specificity=specificity, precision=ratio(tp, tp + fp),
                npv=ratio(tn, tn + fn), fpr=ratio(fp, fp + tn), fnr=ratio(fn, fn + tp),
                fdr=ratio(fp, fp + tp), false_omission_rate=ratio(fn, fn + tn),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(sensitivity + specificity) / 2,
                mcc=ratio(tp * tn - fp * fn, denominator))


def category_paired_uncertainty(actual, predicted, baseline, categories, config):
    """Paired category-cluster bootstrap and category-level sign randomization.

    Intervals concern the fixed horizon's category composition. They do not
    estimate temporal generalization, or establish that categories are independent.
    """
    actual, predicted, baseline = map(lambda a: np.asarray(a, float), (actual, predicted, baseline))
    data = pd.DataFrame(dict(category=np.asarray(categories), actual=actual,
        absolute_base=np.abs(baseline - actual), absolute_model=np.abs(predicted - actual),
        square_base=(baseline - actual) ** 2, square_model=(predicted - actual) ** 2,
        win=(np.abs(predicted - actual) < np.abs(baseline - actual)).astype(int), n=1))
    if data.category.isna().any():
        raise ValueError("Every comparison needs a known category cluster")
    grouped = data.groupby("category", observed=True).sum(numeric_only=True)
    k = len(grouped)
    if k < 2:
        raise ValueError("Cluster inference requires at least two categories")
    rng = np.random.default_rng(config.seed)
    weights = rng.multinomial(k, np.repeat(1 / k, k), size=config.bootstrap_repetitions)
    totals = weights @ grouped.to_numpy()
    total = {name: totals[:, i] for i, name in enumerate(grouped.columns)}
    def summarize(values):
        values = values[np.isfinite(values)]
        return (float(np.quantile(values, .025)), float(np.quantile(values, .975))) if len(values) else (np.nan, np.nan)
    gains = grouped.absolute_base.to_numpy() - grouped.absolute_model.to_numpy()
    gain = np.abs(baseline - actual) - np.abs(predicted - actual)
    boot_gain = total["absolute_base"] - total["absolute_model"]
    volume = total["actual"]
    boot_wape = np.divide(100 * boot_gain, volume, out=np.full_like(volume, np.nan), where=volume > 0)
    mae_ci = summarize(boot_gain / total["n"])
    wape_ci = summarize(boot_wape)
    rmse_ci = summarize(np.sqrt(total["square_model"] / total["n"]) - np.sqrt(total["square_base"] / total["n"]))
    signs = rng.choice([-1., 1.], size=(config.permutation_repetitions, k))
    p = (1 + (np.abs(signs @ gains) >= abs(gains.sum()) - 1e-10).sum()) / (config.permutation_repetitions + 1)
    return dict(clusters=k, paired_mae_gain=float(gain.mean()),
                paired_wape_gain_pp=ratio(100 * gain.sum(), actual.sum()),
                paired_gain_dz=ratio(gain.mean(), gain.std(ddof=1)),
                product_win_fraction=float((gain > 0).mean()), product_tie_fraction=float((gain == 0).mean()),
                mae_gain_ci_low=mae_ci[0], mae_gain_ci_high=mae_ci[1],
                wape_gain_ci_low=wape_ci[0], wape_gain_ci_high=wape_ci[1],
                rmse_change_ci_low=rmse_ci[0], rmse_change_ci_high=rmse_ci[1],
                cluster_signflip_p=float(p))


def holm_adjust(values):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    return result


def feature_profile(frame, group_a, group_b, numeric_features, catalog, used_features, config):
    rows = []
    for name in numeric_features:
        a = frame.loc[group_a, name].to_numpy(dtype=float, na_value=np.nan)
        b = frame.loc[group_b, name].to_numpy(dtype=float, na_value=np.nan)
        effect = standardized_difference(a, b)
        enough = min(effect["n_a"], effect["n_b"]) >= config.min_profile_group
        row = dict(feature=name, family=catalog.loc[name, "family"], used_by_model=name in used_features,
                   enough_observations=enough, missing_rate_a=float(np.isnan(a).mean()) if len(a) else np.nan,
                   missing_rate_b=float(np.isnan(b).mean()) if len(b) else np.nan,
                   **effect, **ks_distance(a, b))
        row["missing_rate_difference"] = row["missing_rate_a"] - row["missing_rate_b"]
        if not enough:
            row["smd"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _event(frame, predicted, name, value):
    baseline, actual = frame.baseline.to_numpy(), frame.actual.to_numpy()
    predicted = np.asarray(predicted, float)
    if name.startswith("high_demand"):
        valid = np.ones(len(frame), bool)
        return valid, actual >= value, predicted, value
    valid = np.isfinite(baseline) & (baseline > 0)
    actual_ratio = actual[valid] / baseline[valid]
    scores = predicted[valid] / baseline[valid]
    if name.startswith("decline"):
        return valid, actual_ratio <= value, -scores, -value
    return valid, actual_ratio >= value, scores, value


def _native_shap(frame, model_path, info, algorithm, objective):
    columns = info["used_feature_columns"]
    x = frame[columns].copy()
    if algorithm == "catboost":
        from catboost import CatBoostRegressor, Pool
        categories = [name for name in ["product_id", "brand_id", "Level1", "Level2", "Level3", "Level4", "Level5"] if name in columns]
        for name in columns:
            x[name] = x[name].fillna("__UNKNOWN__").astype(str) if name in categories else x[name].astype(np.float32)
        model = CatBoostRegressor(); model.load_model(str(model_path))
        pool = Pool(x, cat_features=categories)
        raw = model.predict(pool)
        contributions = model.get_feature_importance(pool, type="ShapValues", thread_count=8)
    else:
        import xgboost as xgb
        for name in columns:
            if name in info["category_levels"]:
                levels = info["category_levels"][name]
                x[name] = pd.Categorical(x[name].where(x[name].isin(levels)), categories=levels)
            else:
                x[name] = x[name].astype(np.float32)
        model = xgb.Booster(); model.load_model(model_path)
        matrix = xgb.DMatrix(x, enable_categorical=True, nthread=8)
        raw = model.predict(matrix, output_margin=True).astype(float)
        contributions = model.predict(matrix, pred_contribs=True).astype(float)
    values, expected = contributions[:, :-1], contributions[:, -1]
    reconstructed = values.sum(axis=1) + expected
    np.testing.assert_allclose(reconstructed, raw, rtol=2e-5, atol=.1)
    transformed = np.expm1(raw) if objective == "log_rmse" else np.asarray(raw, float)
    return values, expected, raw, np.maximum(transformed, 0), columns


def _export_explanations(frame, model_name, role, path, info, objective, algorithm, catalog, out):
    values, expected, raw, forecast, columns = _native_shap(frame, path, info, algorithm, objective)
    np.testing.assert_allclose(forecast, frame[model_name], rtol=2e-6, atol=.1)
    family = catalog.loc[columns, "family"]
    global_rows = []
    for i, name in enumerate(columns):
        feature = frame[name]
        rho = np.nan
        if pd.api.types.is_numeric_dtype(feature):
            valid = feature.notna().to_numpy()
            if valid.sum() >= 5 and feature[valid].nunique() > 1 and np.std(values[valid, i]) > 0:
                rho = float(stats.spearmanr(feature[valid].astype(float), values[valid, i]).statistic)
        global_rows.append(dict(explained_model=model_name, role=role, feature=name, family=family.loc[name],
                                mean_abs_shap=float(np.abs(values[:, i]).mean()),
                                mean_shap=float(values[:, i].mean()), feature_shap_spearman=rho))
    family_rows = []
    for name in family.unique():
        positions = np.flatnonzero(family.eq(name).to_numpy())
        summed = values[:, positions].sum(axis=1)
        family_rows.append(dict(explained_model=model_name, role=role, family=name, features=len(positions),
                                mean_abs_family_sum=float(np.abs(summed).mean()),
                                sum_mean_abs_features=float(np.abs(values[:, positions]).mean(axis=0).sum())))
    rows = frame[KEYS].copy()
    rows["actual"], rows["forecast"], rows["baseline"] = frame.actual, forecast, frame.baseline
    rows["native_expected_value"], rows["native_output"] = expected, raw
    rows["postprocessing_adjustment"] = forecast - raw if objective != "log_rmse" else np.nan
    rows["shap_reconstruction_error"] = values.sum(axis=1) + expected - raw
    rows["explained_model"], rows["role"], rows["output_scale"] = model_name, role, "log1p_sales" if objective == "log_rmse" else "sales_units"
    wide = pd.concat([rows[KEYS], pd.DataFrame(values, columns=columns).rename(columns=lambda c: "shap__" + c)], axis=1)
    wide.to_parquet(out / f"shap_values_{role}_{algorithm}.parquet", index=False)
    # Selection is retrospective diagnosis, not a new training or decision rule.
    residual = forecast - frame.actual.to_numpy()
    gain = np.abs(frame.baseline - frame.actual).to_numpy() - np.abs(residual)
    selections = {}
    for reason, order in [
        ("largest_underforecast", np.argsort(residual)[:2]),
        ("largest_overforecast", np.argsort(-residual)[:2]),
        ("largest_regression_vs_baseline", np.argsort(gain)[:2]),
        ("largest_improvement_vs_baseline", np.argsort(-gain)[:2]),
        ("zero_forecast_positive_actual", np.flatnonzero((forecast == 0) & frame.actual.gt(0).to_numpy())[:2]),
    ]:
        for i in order:
            selections.setdefault(int(i), []).append(reason)
    local = []
    for i, reasons in selections.items():
        top = np.argsort(-np.abs(values[i]))[:8]
        for j in top:
            local.append(dict(explained_model=model_name, role=role, product_id=frame.product_id.iloc[i],
                              reason="; ".join(reasons), feature=columns[j], family=family.iloc[j],
                              feature_value=str(frame[columns[j]].iloc[i]), contribution=float(values[i, j])))
        local.append(dict(explained_model=model_name, role=role, product_id=frame.product_id.iloc[i],
                          reason="; ".join(reasons), feature="OTHER_FEATURES", family="other",
                          feature_value="", contribution=float(values[i].sum() - values[i, top].sum())))
    return pd.DataFrame(global_rows), pd.DataFrame(family_rows), rows, pd.DataFrame(local)


def run_diagnostics(data_dir="data/processed", model_dir="data/modeling", refinement_dir="data/refinement",
                    output_dir="data/analysis", *, config=None, progress=print):
    config = config or DiagnosticConfig()
    data_dir, model_dir, refinement_dir, out = [Path(p).resolve() for p in [data_dir, model_dir, refinement_dir, output_dir]]
    if any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents for p in [data_dir, model_dir, refinement_dir]):
        raise ValueError("Analysis outputs must be separate from model and input artifacts")
    out.mkdir(parents=True, exist_ok=True)
    protected = {str(p): _protected_hashes(p) for p in [model_dir, refinement_dir]}
    features, labels = pd.read_parquet(data_dir / "features.parquet"), pd.read_parquet(data_dir / "targets.parquet")
    features = features.sort_values(KEYS).reset_index(drop=True)
    labels = features[KEYS].merge(labels, on=KEYS, how="left", sort=False, validate="one_to_one")
    if labels.label_complete.isna().any() or not labels.label_complete.eq(labels.target_sales_30d.notna()).all():
        raise ValueError("Feature keys and complete-target status must align")
    catalog = pd.read_csv(data_dir / "feature_catalog.csv").set_index("name")
    model_columns = catalog.index[catalog.role.eq("feature")].tolist()
    numeric = [name for name in model_columns if pd.api.types.is_numeric_dtype(features[name])]
    final_info = json.loads((model_dir / "final_fit.json").read_text())
    selection = json.loads((model_dir / "selection.json").read_text())
    refinement = json.loads((refinement_dir / "results.json").read_text())
    cache = Path(refinement["checkpoint_directory"])
    if not cache.exists():
        # Notebook runs may save absolute paths; support CLI-relative manifests too.
        cache = data_dir.parent.parent / cache
    leaderboard = pd.read_csv(refinement_dir / "leaderboard.csv")
    best_xgb = leaderboard[leaderboard.candidate.str.startswith("rotating_xgboost_")].iloc[0].candidate
    candidates = ["baseline", *refinement["protocol"]["candidates"], "refined_policy"]
    reported = pd.read_parquet(refinement_dir / "predictions.parquet")
    contexts = []
    for role, frame in reported[reported.role.str.startswith("development_")].groupby("role", sort=True):
        # Avoid carrying any columns that could collide with the feature dictionary.
        base = frame[KEYS + ["actual", "baseline", "volume_segment", *[c for c in candidates if c != "baseline"]]].copy()
        contexts.append((role, base.merge(features, on=KEYS, how="left", validate="one_to_one"), candidates))
    final = pd.read_parquet(model_dir / "final_predictions.parquet").rename(columns={"main_prediction": REFERENCE, "baseline_prediction": "baseline"})
    final = final[KEYS + ["actual", "baseline", REFERENCE, "volume_segment"]].merge(features, on=KEYS, how="left", validate="one_to_one")
    contexts.append(("final_test", final, ["baseline", REFERENCE]))
    for _, frame, _ in contexts:
        if frame.Level4.isna().any():
            raise ValueError("Missing cutoff-aligned explanatory features")
    protocol = dict(config=asdict(config), analysis_only=True, retraining=False, model_selection_changed=False,
        final_scope="Retrospective diagnostics of already saved CatBoost/baseline final predictions only",
        xgboost_scope="Development predictions and saved development models only; no XGBoost final-test score",
        smd="Signed (mean A - mean B)/sqrt((sample variance A + sample variance B)/2); paired dz is reported separately",
        ks="Empirical two-sided CDF distance; no independent-samples p-value for paired or category-dependent observations",
        uncertainty="Level4 paired cluster bootstrap and category sign flips, conditional on each fixed horizon",
        independence_limit="Cross-category common shocks and temporal uncertainty are not resolved by cluster resampling",
        alerts="High demand at historical completed-target q50/q75/q90; growth >=10/25/50%; decline >=25%",
        thresholds="Historical latest-completed-target snapshot for high demand; fixed ratios for growth/decline; no threshold optimization",
        attribution="Native TreeSHAP explains model output, not causal effects; clipping is a separate postprocessing step",
        sources={str(p): _digest(p) for p in [data_dir / "manifest.json", data_dir / "features.parquet",
            data_dir / "targets.parquet", data_dir / "feature_catalog.csv", model_dir / "final_predictions.parquet",
            refinement_dir / "predictions.parquet"]})
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    regression_rows, paired_rows, confusion_rows, case_tables, category_rows = [], [], [], [], []
    feature_profiles, drift_rows, reliability_rows, error_tables, thresholds = [], [], [], [], []
    for role, frame, models in contexts:
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        origin = pd.Timestamp(frame.forecast_origin.iloc[0])
        if progress:
            progress(f"Diagnostics: {role}, {origin.date()}, products={len(frame)}, models={len(models)}")
        train_positions = split_for_origin(features, labels, origin)["train"]
        train = features.iloc[train_positions].copy()
        train["completed_target"] = labels.iloc[train_positions].target_sales_30d.to_numpy()
        train["completed_target_end"] = labels.iloc[train_positions].target_end.to_numpy()
        train = train.sort_values(KEYS).drop_duplicates("product_id", keep="last")
        assert train.completed_target_end.le(origin).all()
        event_specs = [(f"high_demand_q{int(q * 100)}", float(train.completed_target.quantile(q))) for q in config.high_demand_quantiles]
        event_specs += [(f"growth_{int((factor - 1) * 100 + .01)}pct", factor) for factor in config.growth_factors]
        event_specs += [("decline_25pct", .75)]
        for name, value in event_specs:
            thresholds.append(dict(role=role, forecast_origin=origin, event=name, value=value,
                                   historical_products=len(train), latest_historical_target_end=train.completed_target_end.max()))
        # Match the same products when measuring changes in historical covariates.
        matched = frame[["product_id"]].merge(train, on="product_id", validate="one_to_one")
        now = frame.set_index("product_id").loc[matched.product_id]
        for name in numeric:
            old_values, new_values = matched[name].to_numpy(dtype=float), now[name].to_numpy(dtype=float)
            drift_rows.append(dict(role=role, forecast_origin=origin, feature=name, family=catalog.loc[name, "family"],
                                   availability=catalog.loc[name, "availability"],
                                   missing_rate_difference=float(np.isnan(new_values).mean() - np.isnan(old_values).mean()),
                                   **standardized_difference(new_values, old_values), **ks_distance(new_values, old_values)))
        drift_rows.append(dict(role=role, forecast_origin=origin, feature="OUTCOME_completed_30d_total", family="outcome",
                               availability="retrospective_outcome", missing_rate_difference=0.,
                               **standardized_difference(now.actual, matched.completed_target),
                               **ks_distance(now.actual, matched.completed_target)))
        for model_name in models:
            predicted, actual, baseline = frame[model_name].to_numpy(float), frame.actual.to_numpy(float), frame.baseline.to_numpy(float)
            residual = predicted - actual
            absolute = np.abs(residual)
            distribution = standardized_difference(predicted, actual)
            regression_rows.append(dict(role=role, forecast_origin=origin, model=model_name,
                **regression_metrics(actual, predicted), mean_residual=float(residual.mean()), median_residual=float(np.median(residual)),
                residual_q05=float(np.quantile(residual, .05)), residual_q95=float(np.quantile(residual, .95)),
                median_absolute_error=float(np.median(absolute)), p90_absolute_error=float(np.quantile(absolute, .9)),
                p95_absolute_error=float(np.quantile(absolute, .95)),
                overforecast_fraction=float((residual > 0).mean()), underforecast_fraction=float((residual < 0).mean()),
                residual_skewness=float(stats.skew(residual, bias=False)), residual_excess_kurtosis=float(stats.kurtosis(residual, bias=False)),
                residual_dz=ratio(residual.mean(), residual.std(ddof=1)),
                predicted_actual_smd=distribution["smd"], **ks_distance(predicted, actual),
                spearman=float(stats.spearmanr(actual, predicted).statistic),
                zero_prediction_positive_actual=int(((predicted == 0) & (actual > 0)).sum()),
                top_1pct_absolute_error_share=ratio(np.sort(absolute)[-max(1, int(np.ceil(.01 * len(actual)))):].sum(), absolute.sum()),
                top_1pct_squared_error_share=ratio(np.sort(residual ** 2)[-max(1, int(np.ceil(.01 * len(actual)))):].sum(), (residual ** 2).sum())))
            if model_name != "baseline":
                paired_rows.append(dict(role=role, forecast_origin=origin, model=model_name,
                    **category_paired_uncertainty(actual, predicted, baseline, frame[config.cluster_column], config)))
            errors = frame[KEYS + ["actual", "baseline", "Level4", "volume_segment", "zero_sales_rate_90d"]].copy()
            errors["model"], errors["role"], errors["predicted"] = model_name, role, predicted
            errors["residual"], errors["absolute_error"] = residual, absolute
            errors["baseline_absolute_error"] = np.abs(baseline - actual)
            errors["absolute_error_gain"] = errors.baseline_absolute_error - absolute
            errors["extra_squared_error"] = residual ** 2 - (baseline - actual) ** 2
            errors["symmetric_percentage_error"] = np.divide(200 * residual, np.abs(predicted) + actual,
                out=np.zeros_like(residual), where=(np.abs(predicted) + actual) > 0)
            error_tables.append(errors)
            for category, positions in frame.groupby("Level4", observed=True).indices.items():
                category_rows.append(dict(role=role, model=model_name, Level4=category,
                    baseline_absolute_error=float(np.abs(baseline[positions] - actual[positions]).sum()),
                    model_absolute_error=float(absolute[positions].sum()),
                    **regression_metrics(actual[positions], predicted[positions])))
            edges = np.unique(np.r_[-np.inf, train.completed_target.quantile(np.arange(.1, 1, .1)).to_numpy(), np.inf])
            bins = pd.cut(predicted, edges, labels=False)
            for bin_id in np.unique(bins):
                mask = bins == bin_id
                reliability_rows.append(dict(role=role, model=model_name, bin=int(bin_id), n=int(mask.sum()),
                    mean_actual=float(actual[mask].mean()), mean_predicted=float(predicted[mask].mean()),
                    observed_forecast_ratio=ratio(actual[mask].mean(), predicted[mask].mean()),
                    **{k: v for k, v in regression_metrics(actual[mask], predicted[mask]).items() if k != "n"}))
            primary_cases = {}
            for event, threshold in event_specs:
                valid, true, scores, decision_threshold = _event(frame, predicted, event, threshold)
                predicted_class = scores >= decision_threshold
                if not len(true):
                    continue
                confusion_rows.append(dict(role=role, forecast_origin=origin, model=model_name, event=event,
                    threshold=threshold, excluded_undefined_baseline=int((~valid).sum()),
                    **confusion_indices(true, predicted_class), **ranking_metrics(true, scores)))
                if event in ["high_demand_q75", "growth_25pct", "decline_25pct"]:
                    cases = frame.loc[valid, KEYS].reset_index(drop=True)
                    cases["role"], cases["model"], cases["event"] = role, model_name, event
                    cases["actual"], cases["predicted"] = actual[valid], predicted[valid]
                    cases["actual_positive"], cases["predicted_positive"] = true, predicted_class
                    cases["score"], cases["decision_score_threshold"] = scores, decision_threshold
                    cases["outcome"] = np.select([true & predicted_class, ~true & predicted_class, true & ~predicted_class], ["TP", "FP", "FN"], default="TN")
                    case_tables.append(cases)
                    primary_cases[event] = (valid, cases)
            detail = (role == "final_test" and model_name == REFERENCE) or (role == "development_3" and model_name in [REFERENCE, best_xgb])
            if detail:
                if role == "final_test":
                    used = final_info["used_feature_columns"]
                else:
                    info = json.loads((cache / f"{model_name}__{origin.date()}.json").read_text())
                    used = info["used_feature_columns"]
                for event, (valid, cases) in primary_cases.items():
                    if event == "decline_25pct":
                        continue
                    subset = frame.loc[valid].reset_index(drop=True)
                    for a, b in [("FP", "TN"), ("FN", "TP")]:
                        profile = feature_profile(subset, cases.outcome.eq(a), cases.outcome.eq(b), numeric, catalog, used, config)
                        feature_profiles.append(profile.assign(role=role, model=model_name, event=event, comparison=f"{a}_minus_{b}"))
                gain = np.abs(baseline - actual) - absolute
                profile = feature_profile(frame, gain < 0, gain > 0, numeric, catalog, used, config)
                feature_profiles.append(profile.assign(role=role, model=model_name, event="regression_error", comparison="worse_minus_better_than_baseline"))
    pd.DataFrame(regression_rows).to_csv(out / "regression_statistics.csv", index=False)
    paired = pd.DataFrame(paired_rows)
    paired["cluster_signflip_p_holm"] = paired.groupby("role").cluster_signflip_p.transform(lambda s: holm_adjust(s.to_numpy()))
    paired.to_csv(out / "paired_cluster_inference.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "alert_statistics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "historical_alert_thresholds.csv", index=False)
    pd.concat(case_tables, ignore_index=True).to_parquet(out / "alert_cases.parquet", index=False)
    pd.concat(feature_profiles, ignore_index=True).to_csv(out / "error_feature_profiles.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(out / "matched_product_drift.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(out / "forecast_reliability.csv", index=False)
    pd.DataFrame(category_rows).to_csv(out / "category_errors.csv", index=False)
    pd.concat(error_tables, ignore_index=True).to_parquet(out / "prediction_errors.parquet", index=False)
    global_shap, family_shap, explained_rows, local_shap = [], [], [], []
    for role, frame, _ in contexts:
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        origin = pd.Timestamp(frame.forecast_origin.iloc[0])
        models = [(REFERENCE, "catboost")] if role == "final_test" else ([(REFERENCE, "catboost"), (best_xgb, "xgboost")] if role == "development_3" else [])
        for name, algorithm in models:
            if progress:
                progress(f"Native SHAP: {algorithm}, {role}, products={len(frame)}")
            if role == "final_test":
                info, objective, model_path = final_info, selection["objective"], model_dir / "test_model.cbm"
            else:
                info = json.loads((cache / f"{name}__{origin.date()}.json").read_text())
                objective = refinement["protocol"]["candidates"][name]["objective"]
                model_path = cache / f"{name}.{'cbm' if algorithm == 'catboost' else 'ubj'}"
            g, f, rows, local = _export_explanations(frame, name, role, model_path, info, objective, algorithm, catalog, out)
            global_shap.append(g); family_shap.append(f); explained_rows.append(rows); local_shap.append(local)
    pd.concat(global_shap, ignore_index=True).to_csv(out / "shap_global.csv", index=False)
    pd.concat(family_shap, ignore_index=True).to_csv(out / "shap_families.csv", index=False)
    pd.concat(explained_rows, ignore_index=True).to_parquet(out / "shap_reconstruction.parquet", index=False)
    pd.concat(local_shap, ignore_index=True).to_csv(out / "shap_local_cases.csv", index=False)
    unchanged = all(protected[str(p)] == _protected_hashes(p) for p in [model_dir, refinement_dir])
    if not unchanged:
        raise AssertionError("Analysis modified upstream model artifacts")
    result = dict(protocol=protocol, upstream_artifacts_unchanged=unchanged,
                  regression_comparisons=len(regression_rows), alert_comparisons=len(confusion_rows),
                  native_models_explained=3, explained_rows=sum(len(rows) for rows in explained_rows),
                  final_reference=REFERENCE, development_xgboost_reference=best_xgb,
                  final_paired=paired[(paired.role.eq("final_test")) & paired.model.eq(REFERENCE)].to_dict("records"),
                  warnings=["Effect sizes and SHAP associations are not causal effects",
                            "Illustrative alerts are not classifiers, stockout labels, or optimized business policies",
                            "KS distances are descriptive; no iid KS significance claim is made",
                            "Cluster intervals condition on a fixed horizon and do not measure future temporal uncertainty",
                            "The final report was already inspected; this is retrospective diagnosis and not a fresh test"],
                  protected_artifact_hashes=protected)
    (out / "results.json").write_text(json.dumps(result, default=str, indent=2) + "\n")
    if progress:
        progress(f"Analysis complete: {len(regression_rows)} regression comparisons, {len(confusion_rows)} alert comparisons, three native SHAP explanations; upstream artifacts unchanged.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/modeling"))
    parser.add_argument("--refinement-dir", type=Path, default=Path("data/refinement"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    args = parser.parse_args()
    run_diagnostics(args.data_dir, args.model_dir, args.refinement_dir, args.output_dir,
                    progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
