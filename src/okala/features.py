"""Point-in-time retail features. Features and future targets are separate artifacts.

Run from the project root: PYTHONPATH=src .venv/bin/python -m okala.features
No model, fitted transform, future imputation, or external calendar is used here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import jdatetime
import numpy as np
import pandas as pd


KEYS = ["forecast_origin", "product_id"]
CATEGORIES = ["brand_id", "Level1", "Level2", "Level3", "Level4", "Level5"]
MEASURES = ["sale_qty", "stock_qty", "num_stores_with_stock", "discount_pct", "price"]
REQUIRED = ["date", "product_id", *CATEGORIES, *MEASURES]


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...] = (7, 14, 30, 60, 90, 180)
    horizon_days: int = 30
    history_days: int = 90
    min_history_coverage: float = 0.95
    recency_days: int = 7
    competition_level: str = "Level4"
    peer_price_max_age_days: int = 30
    min_moment_observations: int = 8

    def __post_init__(self):
        if not self.windows or any(w <= 0 for w in self.windows):
            raise ValueError("History windows must be positive calendar-day counts")
        if min(self.horizon_days, self.history_days, self.recency_days) <= 0:
            raise ValueError("Horizon, history and recency must be positive")
        if not 0 < self.min_history_coverage <= 1:
            raise ValueError("min_history_coverage must be in (0, 1]")
        if self.competition_level not in CATEGORIES[1:]:
            raise ValueError("competition_level must be a category hierarchy field")
        if self.peer_price_max_age_days < 0 or self.min_moment_observations < 4:
            raise ValueError("Price age must be nonnegative; moments require at least 4 observations")


def safe_ratio(numerator, denominator):
    """A ratio is unknown for nonfinite inputs or nonpositive denominators."""
    a, b = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(a.shape, np.nan)
    np.divide(a, b, out=result, where=np.isfinite(a) & np.isfinite(b) & (b > 0))
    return result


def prepare_observations(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate raw measurements and add row-local flags without altering raw."""
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = raw[REQUIRED].copy()
    if data.empty or data[REQUIRED].isna().any().any():
        raise ValueError("Source observations must be nonempty and contain no null measurements/IDs")
    if not pd.api.types.is_datetime64_any_dtype(data.date):
        strings = data.date.astype(str)
        is_jalali = strings.str.fullmatch(r"1[34]\d{6}")
        if is_jalali.all():
            lookup = {
                s: jdatetime.date(int(s[:4]), int(s[4:6]), int(s[6:])).togregorian()
                for s in strings.unique()
            }
            data["date"] = pd.to_datetime(strings.map(lookup), errors="raise")
        elif is_jalali.any():
            raise ValueError("Mixed Jalali and Gregorian date encodings")
        else:
            data["date"] = pd.to_datetime(strings, errors="raise")
    if data.date.dt.tz is not None or not data.date.eq(data.date.dt.normalize()).all():
        raise ValueError("Expected timezone-naive, midnight business dates")
    data["date"] = data.date.astype("datetime64[ns]")
    for name in ["product_id", *CATEGORIES]:
        data[name] = data[name].astype(str)
    for name in MEASURES:
        values = pd.to_numeric(data[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} contains nonfinite or negative observations")
        data[name] = values
    if data.discount_pct.gt(100).any():
        raise ValueError("Discount must use the 0-100 percentage-point scale")
    if data.num_stores_with_stock.mod(1).ne(0).any():
        raise ValueError("Store counts must be integers")
    if data.duplicated(["date", "product_id"]).any():
        raise ValueError("Duplicate product-day observations require reconciliation")
    data["price_zero_flag"] = data.price.eq(0)
    data["stock_equal_sales_flag"] = data.stock_qty.gt(0) & np.isclose(
        data.stock_qty, data.sale_qty, rtol=0, atol=1e-8
    )
    data["stock_store_disagreement_flag"] = (
        data.stock_qty.gt(0) & data.num_stores_with_stock.eq(0)
    ) | (data.stock_qty.eq(0) & data.num_stores_with_stock.gt(0))
    data["price_positive"] = data.price.where(data.price.gt(0))
    data["stock_usable"] = data.stock_qty.mask(data.stock_equal_sales_flag)
    return data.sort_values(["date", "product_id"]).reset_index(drop=True)


class RetailFeaturePipeline:
    """Calendar-aligned matrices with causal snapshots and a feature dictionary.

    Missing rows remain NaN. Forward-filled caches describe last-known historical
    measurements only; rolling sales and targets always use original observations.
    Category membership is taken from each product's latest record at the cutoff.
    """

    def __init__(
        self,
        raw: pd.DataFrame,
        config: FeatureConfig | None = None,
        *,
        verified_holidays: Iterable | None = None,
    ):
        self.config = config or FeatureConfig()
        self.observations = prepare_observations(raw)
        data = self.observations
        self.dates = pd.date_range(data.date.min(), data.date.max(), freq="D")
        self.products = pd.Index(sorted(data.product_id.unique()), name="product_id")
        self.first_seen = data.groupby("product_id").date.min().reindex(self.products)
        self.matrices = {}
        for name in [*MEASURES, "price_positive", "stock_usable", "price_zero_flag",
                     "stock_equal_sales_flag", "stock_store_disagreement_flag"]:
            matrix = data.pivot(index="date", columns="product_id", values=name)
            self.matrices[name] = matrix.reindex(index=self.dates, columns=self.products).astype(float)
        self.present = self.matrices["sale_qty"].notna()
        self.last_values = {}
        self.last_positions = {}
        positions = np.broadcast_to(np.arange(len(self.dates))[:, None], self.present.shape)
        for name in ["sale_qty", "price_positive", "stock_usable", "discount_pct", "num_stores_with_stock"]:
            matrix = self.matrices[name]
            self.last_values[name] = matrix.ffill()
            observed_positions = pd.DataFrame(
                np.where(matrix.notna(), positions, np.nan), index=self.dates, columns=self.products
            )
            self.last_positions[name] = observed_positions.ffill()
        positive = self.matrices["sale_qty"].gt(0)
        self.last_positive_sale_position = pd.DataFrame(
            np.where(positive, positions, np.nan), index=self.dates, columns=self.products
        ).ffill()
        self.verified_holidays = None if verified_holidays is None else {
            pd.Timestamp(day).normalize() for day in verified_holidays
        }
        self._catalog: dict[str, dict] = {}

    @property
    def catalog(self) -> pd.DataFrame:
        return pd.DataFrame(self._catalog.values())

    @property
    def model_feature_columns(self) -> list[str]:
        return [name for name, entry in self._catalog.items() if entry["role"] == "feature"]

    def _cutoff(self, origin) -> pd.Timestamp:
        origin = pd.Timestamp(origin)
        if origin.tz is not None or origin != origin.normalize() or origin not in self.dates:
            raise ValueError("Origin must be a midnight business date inside the source date range")
        return origin

    def _known_products(self, origin) -> pd.Index:
        return self.products[self.first_seen.le(origin)]

    def _window(self, name, origin, products, days, offset=0):
        end = origin - pd.Timedelta(days=offset)
        dates = pd.date_range(end - pd.Timedelta(days=days - 1), end)
        return self.matrices[name].reindex(index=dates, columns=products)

    def features_at(self, origin) -> pd.DataFrame:
        """Build one snapshot. No value dated after origin is consulted."""
        t = self._cutoff(origin)
        cfg = self.config
        products = self._known_products(t)
        count = len(products)
        position = self.dates.get_loc(t)
        values: dict[str, object] = {}

        def add(name, value, family, definition, days=0, availability="historical", role="feature"):
            array = np.asarray(value)
            if array.ndim == 0:
                array = np.repeat(array, count)
            if len(array) != count:
                raise ValueError(f"Feature length mismatch: {name}")
            if np.issubdtype(array.dtype, np.number) and np.isinf(array).any():
                raise ValueError(f"Infinite feature: {name}")
            values[name] = array
            entry = dict(name=name, family=family, role=role, availability=availability,
                         lookback_days=days, definition=definition,
                         missing_policy="Unknown/insufficient observations or nonpositive ratio denominator remain NaN")
            if name in self._catalog and self._catalog[name] != entry:
                raise ValueError(f"Inconsistent catalog entry: {name}")
            self._catalog[name] = entry

        def last(name):
            return self.last_values[name].loc[t, products].to_numpy(dtype=float)

        def age(name):
            return position - self.last_positions[name].loc[t, products].to_numpy(dtype=float)

        def window(name, days, offset=0):
            return self._window(name, t, products, days, offset)

        add("forecast_origin", t, "keys", "Information cutoff, not a model input", role="key")
        add("product_id", products.to_numpy(), "main", "Product identity known by the cutoff", availability="static_as_of_cutoff")
        past = self.observations.loc[self.observations.date.le(t)]
        metadata = past.drop_duplicates("product_id", keep="last").set_index("product_id").reindex(products)
        for name in CATEGORIES:
            add(name, metadata[name].to_numpy(), "main", f"Latest observed {name} at or before cutoff", availability="static_as_of_cutoff")
        life = (t - self.first_seen.loc[products]).dt.days.to_numpy() + 1
        add("observed_life_days", life, "missingness_metadata", "Days since first recorded observation, not confirmed product age")
        add("observed_today", self.present.loc[t, products].to_numpy(dtype=np.int8), "missingness_metadata", "Whether a source row exists on the cutoff day")
        add("days_since_last_record", age("sale_qty"), "missingness_metadata", "Calendar age of the latest actual source record")
        add("days_since_last_positive_sale", position - self.last_positive_sale_position.loc[t, products].to_numpy(dtype=float), "missingness_metadata", "Age of latest strictly positive recorded sales; unknown if none")
        for name in ["sale_qty", "price_positive", "stock_usable", "discount_pct", "num_stores_with_stock"]:
            add(f"{name}_last_observed", last(name), "main", f"Last known historical {name}; inspect its age alongside its value")
            add(f"{name}_age_days", age(name), "missingness_metadata", f"Age of last usable {name} measurement")
        add("log1p_last_positive_price", np.log1p(last("price_positive")), "domain", "log1p of last positive recorded price, in unspecified source currency units")

        statistics = {}
        windows = sorted(set(cfg.windows) | {7, 14, 30, 90, cfg.history_days})
        for days in windows:
            frame = window("sale_qty", days)
            n = frame.count().to_numpy()
            mean = frame.mean().to_numpy()
            std = frame.std(ddof=1).to_numpy()
            total = frame.sum(min_count=1).to_numpy()
            statistics[days] = dict(frame=frame, n=n, mean=mean, std=std, total=total)
            for suffix, value, definition in [
                ("observed_days", n, "Count of recorded days, including recorded zero sales"),
                ("sum_observed", total, "Sum over recorded days only; not an imputed calendar total"),
                ("sum_complete", np.where(n == days, total, np.nan), "Calendar total only when every day is recorded"),
                ("mean", mean, "Observed-day sales mean"),
                ("std", std, "Sample standard deviation across observed days"),
                ("min", frame.min().to_numpy(), "Minimum observed daily sales"),
                ("max", frame.max().to_numpy(), "Maximum observed daily sales"),
                ("median", frame.median().to_numpy(), "Median observed daily sales"),
            ]:
                add(f"sales_{suffix}_{days}d", value, "aggregated", definition, days)
            available_life = np.minimum(life, days)
            add(f"missing_days_within_life_{days}d", available_life - n, "missingness_metadata", "Absent days since first observation within this trailing window", days)
            add(f"coverage_fraction_{days}d", n / days, "missingness_metadata", "Observed days divided by the full calendar window length", days)
            add(f"zero_sales_rate_{days}d", safe_ratio(frame.eq(0).sum(), n), "temporal", "Recorded zero-sales days divided by observed days", days)
            add(f"positive_sales_days_{days}d", frame.gt(0).sum().to_numpy(), "temporal", "Number of recorded strictly positive sales days", days)

        for name in ["discount_pct", "num_stores_with_stock", "stock_usable", "price_positive"]:
            for days in [7, 30, 90]:
                frame = window(name, days)
                add(f"{name}_mean_{days}d", frame.mean().to_numpy(), "aggregated", f"Mean of actual usable {name} measurements; no gap filling", days)
                add(f"{name}_observed_days_{days}d", frame.count().to_numpy(), "missingness_metadata", f"Number of usable {name} measurements", days)
        for days in [30, 90]:
            for name in ["price_zero_flag", "stock_equal_sales_flag", "stock_store_disagreement_flag"]:
                add(f"{name}_count_{days}d", window(name, days).sum(min_count=1).to_numpy(), "missingness_metadata", f"Count of observed {name} events; absent rows are not events", days)
            discount = window("discount_pct", days)
            add(f"discounted_days_rate_{days}d", safe_ratio(discount.gt(0).sum(), discount.count()), "temporal", "Share of recorded days with positive discount", days)

        # Calendar lags deliberately do not step across missing records.
        for lag in [0, 1, 7, 14, 28, 30, 60, 90, 364]:
            date = t - pd.Timedelta(days=lag)
            observed = self.matrices["sale_qty"].reindex(index=[date], columns=products).iloc[0]
            add(f"sales_lag_{lag}d", observed.to_numpy(), "temporal", f"Sales on exact calendar date T-{lag}; absent row stays unknown", lag + 1)

        for half_life in [7, 30]:
            days = max(cfg.windows)
            frame = window("sale_qty", days)
            weights = 2.0 ** (-np.arange(days - 1, -1, -1) / half_life)
            denominator = frame.notna().mul(weights, axis=0).sum().to_numpy()
            numerator = frame.mul(weights, axis=0).sum(min_count=1).to_numpy()
            add(f"sales_ewmean_half_life_{half_life}d", safe_ratio(numerator, denominator), "aggregated", f"Calendar-decayed observed sales mean, half-life {half_life} days, truncated at {days} days", days)

        s30 = statistics[30]
        add("sales_deviation_from_mean_30d", values["sales_lag_0d"] - s30["mean"], "deviation", "Observed cutoff-day sales minus trailing observed-day mean", 30)
        add("sales_zscore_30d", safe_ratio(values["sales_lag_0d"] - s30["mean"], s30["std"]), "deviation", "Cutoff-day sales deviation divided by sample standard deviation", 30)
        add("sales_mean_7d_vs_30d_fraction", safe_ratio(statistics[7]["mean"], s30["mean"]) - 1, "deviation", "Recent 7-day observed mean relative to 30-day observed mean", 30)
        price_median = window("price_positive", 30).median().to_numpy()
        add("price_deviation_from_median_30d_fraction", safe_ratio(last("price_positive"), price_median) - 1, "deviation", "Last positive recorded price relative to recent positive-price median", 30)
        add("discount_deviation_from_mean_30d_pp", last("discount_pct") - values["discount_pct_mean_30d"], "deviation", "Last recorded discount minus 30-day mean, in percentage points", 30)

        weeks = [window("sale_qty", 7, offset).mean().to_numpy() for offset in [0, 7, 14]]
        velocity = safe_ratio(weeks[0], weeks[1]) - 1
        prior_velocity = safe_ratio(weeks[1], weeks[2]) - 1
        add("sales_weekly_fractional_velocity", velocity, "velocity_acceleration", "Mean of latest 7 calendar days / preceding disjoint 7-day mean minus 1", 14)
        add("sales_weekly_fractional_acceleration", velocity - prior_velocity, "velocity_acceleration", "Current fractional weekly velocity minus preceding fractional velocity", 21)
        add("sales_daily_velocity_7d", (weeks[0] - weeks[1]) / 7, "velocity_acceleration", "Difference between adjacent weekly means divided by 7", 14)
        add("sales_daily_acceleration_7d", (weeks[0] - 2 * weeks[1] + weeks[2]) / 49, "velocity_acceleration", "Second difference of disjoint weekly means divided by 7 squared", 21)
        add("sales_previous_week_mean_zero", np.isfinite(weeks[1]) & (weeks[1] == 0), "missingness_metadata", "Indicates a known zero growth denominator, rather than inventing infinite growth", 14)
        for name in ["price_positive", "num_stores_with_stock"]:
            previous_t = t - pd.Timedelta(days=7)
            previous = self.last_values[name].reindex(index=[previous_t], columns=products).iloc[0].to_numpy()
            add(f"{name}_fractional_change_7d", safe_ratio(last(name), previous) - 1, "velocity_acceleration", "Last-known measurement relative to its last-known value as of T-7", 7)

        for days in [30, 90]:
            frame = statistics[days]["frame"]
            n = statistics[days]["n"]
            enough = (n >= cfg.min_moment_observations) & (statistics[days]["std"] > 0)
            add(f"sales_skewness_{days}d", frame.skew().where(enough).to_numpy(), "higher_order", "Unbiased sample skewness; requires enough observations and positive variance", days)
            add(f"sales_excess_kurtosis_{days}d", frame.kurt().where(enough).to_numpy(), "higher_order", "Unbiased Fisher excess kurtosis; requires enough observations and positive variance", days)
            quantiles = frame.quantile([.1, .25, .75, .9])
            add(f"sales_iqr_{days}d", (quantiles.loc[.75] - quantiles.loc[.25]).to_numpy(), "higher_order", "75th minus 25th percentile of observed sales", days)
            add(f"sales_p90_minus_p10_{days}d", (quantiles.loc[.9] - quantiles.loc[.1]).to_numpy(), "higher_order", "90th minus 10th percentile of observed sales", days)
            add(f"sales_cv_{days}d", safe_ratio(statistics[days]["std"], statistics[days]["mean"]), "higher_order", "Observed daily sample standard deviation / observed mean", days)
            positive = frame.where(frame.gt(0))
            add(f"positive_sales_cv2_{days}d", safe_ratio(positive.std(), positive.mean()) ** 2, "higher_order", "Squared coefficient of variation on positive-sales days only", days)
            total = statistics[days]["total"]
            xlogx = (positive * np.log(positive)).sum(min_count=1).to_numpy()
            entropy = np.log(np.where(total > 0, total, np.nan)) - safe_ratio(xlogx, total)
            positive_count = positive.count().to_numpy()
            normalizer = np.log(np.where(positive_count > 1, positive_count, np.nan))
            add(f"sales_normalized_entropy_{days}d", safe_ratio(entropy, normalizer), "higher_order", "Entropy of positive daily sales shares / log(number of positive days); unknown for <=1 positive day", days)
            add(f"sales_top_day_share_{days}d", safe_ratio(frame.max(), total), "higher_order", "Maximum daily sales / sum of observed sales", days)

        stores = window("num_stores_with_stock", 30)
        sales_per_store = window("sale_qty", 30) / stores.where(stores.gt(0))
        add("sales_per_stocked_store_mean_30d", sales_per_store.mean().to_numpy(), "domain", "Mean daily sales / recorded positive store count; excludes zero denominators, not a causal demand estimate", 30)
        add("stock_cover_days_at_recent_sales_rate", safe_ratio(last("stock_usable"), s30["mean"]), "domain", "Last usable closing stock / observed daily sales mean; inspect stock age, not a replenishment forecast", 30)
        add("nonzero_interval_proxy_90d", safe_ratio(statistics[90]["n"], window("sale_qty", 90).gt(0).sum()), "domain", "Observed days / positive-sales days; descriptive proxy, not calendar ADI", 90)
        recorded = window("sale_qty", 90).notna().to_numpy()
        in_life = (pd.date_range(t - pd.Timedelta(days=89), t).to_numpy()[:, None]
                   >= self.first_seen.loc[products].to_numpy()[None, :])
        missing = (~recorded) & in_life
        previous_missing = np.vstack([np.zeros((1, count), dtype=bool), missing[:-1]])
        add("missing_run_count_90d", (missing & ~previous_missing).sum(axis=0), "missingness_metadata", "Missing-run starts inside the trailing 90-day window, excluding pre-observation life", 90)
        run = np.zeros(count, dtype=int)
        longest = run.copy()
        for row in missing:
            run = np.where(row, run + 1, 0)
            longest = np.maximum(longest, run)
        add("longest_missing_run_90d", longest, "missingness_metadata", "Longest internal or trailing unknown run inside the last 90 calendar days", 90)

        self._competition_features(products, metadata, values, add)
        self._calendar_features(t, add)
        history_count = window("sale_qty", cfg.history_days).count().to_numpy()
        recent = window("sale_qty", cfg.recency_days).notna().any().to_numpy()
        eligible = (life >= cfg.history_days) & (history_count >= np.ceil(cfg.history_days * cfg.min_history_coverage)) & recent
        add("history_eligible", eligible, "keys", "Past-only pilot eligibility; never depends on target completeness", cfg.history_days, role="eligibility")
        return pd.DataFrame(values)

    def _competition_features(self, products, metadata, values, add):
        cfg = self.config
        group = metadata[cfg.competition_level].to_numpy()
        n = len(products)
        result = {name: np.full(n, np.nan) for name in [
            "peer_count_known", "peer_observed_count_today", "peer_reporting_fraction_today",
            "peer_available_count_today", "peer_available_rate_among_observed_today",
            "peer_price_median", "relative_price_vs_peer_median", "fraction_peers_cheaper",
            "peer_discount_mean", "discount_gap_vs_peers_pp", "peer_complete_30d_count",
            "sales_share_30d_among_complete_products", "sales_vs_peer_mean_30d",
            "peer_sales_fractional_trend_7d_vs_30d",
        ]}
        price = np.asarray(values["price_positive_last_observed"], float).copy()
        price[np.asarray(values["price_positive_age_days"]) > cfg.peer_price_max_age_days] = np.nan
        discount = np.asarray(values["discount_pct_last_observed"], float).copy()
        discount[np.asarray(values["discount_pct_age_days"]) >= cfg.recency_days] = np.nan
        observed = np.asarray(values["observed_today"], bool)
        available = observed & (np.asarray(values["num_stores_with_stock_last_observed"]) > 0)
        complete = np.isfinite(values["sales_sum_complete_30d"])
        mean7 = np.asarray(values["sales_mean_7d"], float)
        mean30 = np.asarray(values["sales_mean_30d"], float)
        reliable7 = np.asarray(values["coverage_fraction_7d"]) >= cfg.min_history_coverage
        reliable30 = np.asarray(values["coverage_fraction_30d"]) >= cfg.min_history_coverage
        for category in pd.unique(group):
            members = np.flatnonzero(group == category)
            for i in members:
                peers = members[members != i]
                result["peer_count_known"][i] = len(peers)
                result["peer_observed_count_today"][i] = observed[peers].sum()
                result["peer_reporting_fraction_today"][i] = safe_ratio(observed[peers].sum(), len(peers))
                result["peer_available_count_today"][i] = available[peers].sum()
                result["peer_available_rate_among_observed_today"][i] = safe_ratio(available[peers].sum(), observed[peers].sum())
                priced = price[peers][np.isfinite(price[peers])]
                if len(priced):
                    result["peer_price_median"][i] = np.median(priced)
                    result["relative_price_vs_peer_median"][i] = safe_ratio(price[i], np.median(priced))
                    if np.isfinite(price[i]):
                        result["fraction_peers_cheaper"][i] = ((priced < price[i]).sum() + .5 * (priced == price[i]).sum()) / len(priced)
                discounted = discount[peers][np.isfinite(discount[peers])]
                if len(discounted):
                    result["peer_discount_mean"][i] = discounted.mean()
                    result["discount_gap_vs_peers_pp"][i] = discount[i] - discounted.mean()
                result["peer_complete_30d_count"][i] = complete[peers].sum()
                complete_members = members[complete[members]]
                if complete[i]:
                    group_total = np.asarray(values["sales_sum_complete_30d"])[complete_members].sum()
                    result["sales_share_30d_among_complete_products"][i] = safe_ratio(values["sales_sum_complete_30d"][i], group_total)
                reliable_peers = peers[reliable30[peers]]
                if len(reliable_peers):
                    result["sales_vs_peer_mean_30d"][i] = safe_ratio(mean30[i], mean30[reliable_peers].mean())
                trend_peers = peers[reliable7[peers] & reliable30[peers]]
                if len(trend_peers):
                    result["peer_sales_fractional_trend_7d_vs_30d"][i] = safe_ratio(mean7[trend_peers].mean(), mean30[trend_peers].mean()) - 1
        definitions = {
            "peer_count_known": "Other products whose latest known category matches at cutoff; no future entrants",
            "peer_observed_count_today": "Other known category products with a record on cutoff day",
            "peer_reporting_fraction_today": "Observed peer count / known peer count; zero peers gives unknown",
            "peer_available_count_today": "Observed peers with positive recorded stores today; unknown peers are not asserted unavailable",
            "peer_available_rate_among_observed_today": "Known available peers / observed peers today",
            "peer_price_median": f"Median last positive peer price, maximum age {cfg.peer_price_max_age_days} days; excludes own product",
            "relative_price_vs_peer_median": "Own age-qualified last positive price / peer price median; not normalized for package size",
            "fraction_peers_cheaper": "Fraction of age-qualified peers with lower price, assigning half weight to ties",
            "peer_discount_mean": f"Mean last recorded peer discount with age less than {cfg.recency_days} days",
            "discount_gap_vs_peers_pp": "Own age-qualified last discount minus peer mean, in percentage points",
            "peer_complete_30d_count": "Other category products with complete historical 30-day sales",
            "sales_share_30d_among_complete_products": "Own complete historical sales / category total restricted to complete products, including own; inspect peer coverage",
            "sales_vs_peer_mean_30d": "Own observed sales mean / mean of peer daily means with sufficient historical coverage",
            "peer_sales_fractional_trend_7d_vs_30d": "Leave-one-out peer mean sales rate in 7 days / rate in 30 days minus 1; both require coverage",
        }
        for name, value in result.items():
            add(name, value, "competition", definitions[name], 30)

    def _calendar_features(self, origin, add):
        cfg = self.config
        jalali = jdatetime.date.fromgregorian(date=origin.date())
        for prefix, phase, period in [
            ("weekday", origin.day_of_week, 7),
            ("jalali_month", jalali.month - 1, 12),
            ("jalali_year_progress", (jalali - jdatetime.date(jalali.year, 1, 1)).days, 366 if jalali.isleap() else 365),
        ]:
            for name, function in [("sin", np.sin), ("cos", np.cos)]:
                add(f"{prefix}_{name}", function(2 * np.pi * phase / period), "cyclical", f"{name} transform of {prefix} with its actual calendar period", availability="future_known")
        add("jalali_month", jalali.month, "cyclical", "Jalali calendar month at cutoff", availability="future_known")
        add("jalali_day_of_month", jalali.day, "cyclical", "Jalali day of month at cutoff", availability="future_known")
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=cfg.horizon_days)
        future_jalali = [jdatetime.date.fromgregorian(date=date.date()) for date in future]
        for weekday in range(7):
            add(f"horizon_weekday_{weekday}_count", int((future.day_of_week == weekday).sum()), "cyclical", "Count of this Gregorian weekday in the known forecast calendar, Monday=0", availability="future_known")
        for month in range(1, 13):
            add(f"horizon_jalali_month_{month}_count", sum(day.month == month for day in future_jalali), "cyclical", "Count of dates in this Jalali month during the forecast horizon", availability="future_known")
        for day in [1, 13]:
            add(f"horizon_jalali_01_{day:02d}_count", sum(j.month == 1 and j.day == day for j in future_jalali), "domain", "Count of the named Jalali calendar date, without assuming store closure or holiday status", availability="future_known")
        supplied = self.verified_holidays is not None
        add("verified_holiday_calendar_supplied", supplied, "missingness_metadata", "Whether a caller supplied a verified holiday-date set", availability="future_known")
        holiday_count = sum(date in self.verified_holidays for date in future) if supplied else np.nan
        add("horizon_verified_holiday_count", holiday_count, "domain", "Known holiday count if a verified complete calendar was supplied; otherwise unknown", availability="future_known")

    def targets_at(self, origin) -> pd.DataFrame:
        """Future outcomes, deliberately not accessible through feature_columns."""
        t = self._cutoff(origin)
        products = self._known_products(t)
        horizon = self.config.horizon_days
        dates = pd.date_range(t + pd.Timedelta(days=1), periods=horizon)
        future = self.matrices["sale_qty"].reindex(index=dates, columns=products)
        n = future.count().to_numpy()
        return pd.DataFrame({
            "forecast_origin": t, "product_id": products,
            "target_end": t + pd.Timedelta(days=horizon),
            "target_observed_days": n, "label_complete": n == horizon,
            "target_sales_30d" if horizon == 30 else f"target_sales_{horizon}d": future.sum(min_count=horizon).to_numpy(),
        })

    def build_dataset(self, origins: Iterable, *, progress=None):
        origins = sorted({self._cutoff(origin) for origin in origins})
        if not origins:
            raise ValueError("At least one forecast origin is required")
        features, targets = [], []
        for i, origin in enumerate(origins, start=1):
            features.append(self.features_at(origin))
            targets.append(self.targets_at(origin))
            if progress is not None:
                progress(i, len(origins), origin)
        return pd.concat(features, ignore_index=True), pd.concat(targets, ignore_index=True)

    def backtest_plan(self, development_folds=3) -> pd.DataFrame:
        if development_folds < 1:
            raise ValueError("At least one development fold is required")
        end = self.dates[-1]
        horizon = self.config.horizon_days
        rows = []
        for i, offset in enumerate(range(development_folds + 1, 0, -1), start=1):
            origin = end - pd.Timedelta(days=offset * horizon)
            if origin < self.dates[0]:
                raise ValueError("Not enough calendar history for this backtest plan")
            rows.append(dict(role=f"development_{i}" if offset > 1 else "final_test",
                             forecast_origin=origin, target_start=origin + pd.Timedelta(days=1),
                             target_end=origin + pd.Timedelta(days=horizon)))
        return pd.DataFrame(rows)

    def default_origins(self, step_days=7) -> pd.DatetimeIndex:
        if step_days < 1:
            raise ValueError("step_days must be positive")
        start = self.dates[0] + pd.Timedelta(days=self.config.history_days - 1)
        stop = self.dates[-1] - pd.Timedelta(days=self.config.horizon_days)
        weekly = pd.date_range(start, stop, freq=f"{step_days}D")
        plan = self.backtest_plan()
        return weekly.union(pd.DatetimeIndex(plan.forecast_origin)).union(self.dates[-1:]).sort_values()

    def quality_report(self) -> pd.DataFrame:
        n_by_product = self.present.sum()
        last_seen = self.observations.groupby("product_id").date.max().reindex(self.products)
        span = (last_seen - self.first_seen).dt.days + 1
        entries = {
            "source_rows": len(self.observations), "products": len(self.products),
            "calendar_days": len(self.dates),
            "null_source_values": int(self.observations[REQUIRED].isna().sum().sum()),
            "globally_absent_dates": int((~self.present.any(axis=1)).sum()),
            "internal_missing_product_days": int((span - n_by_product).sum()),
            "zero_price_measurements": int(self.observations.price_zero_flag.sum()),
            "masked_inventory_measurements": int(self.observations.stock_equal_sales_flag.sum()),
            "stock_store_disagreements": int(self.observations.stock_store_disagreement_flag.sum()),
        }
        return pd.Series(entries, name="count").to_frame()


def split_for_origin(features: pd.DataFrame, targets: pd.DataFrame, origin):
    """Return positional indexes for features.iloc, using label-available training."""
    samples = features[KEYS + ["history_eligible"]].merge(
        targets, on=KEYS, how="left", sort=False, validate="one_to_one"
    )
    if samples[["label_complete", "target_end"]].isna().any().any():
        raise ValueError("Each feature row requires a matching target-status row")
    t = pd.Timestamp(origin)
    measurable = samples.label_complete & samples.history_eligible
    train = measurable & samples.forecast_origin.lt(t) & samples.target_end.le(t)
    evaluation = measurable & samples.forecast_origin.eq(t)
    return {"train": np.flatnonzero(train), "evaluation": np.flatnonzero(evaluation)}


def export_artifacts(pipeline, features, targets, output_dir, *, source_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "features.parquet", index=False)
    targets.to_parquet(output_dir / "targets.parquet", index=False)
    pipeline.catalog.to_csv(output_dir / "feature_catalog.csv", index=False)
    plan = pipeline.backtest_plan()
    samples = features.merge(targets, on=KEYS, validate="one_to_one")
    counts = []
    for row in plan.itertuples():
        batch = samples[samples.forecast_origin.eq(row.forecast_origin)]
        splits = split_for_origin(features, targets, row.forecast_origin)
        eligible = batch.history_eligible
        # Excluded historical volume share exposes complete-case selection bias.
        volume = batch.sales_sum_observed_30d
        excluded = eligible & ~batch.label_complete
        counts.append(dict(known_products=len(batch), history_eligible=int(eligible.sum()),
                           complete_targets=int(batch.label_complete.sum()),
                           evaluation_samples=len(splits["evaluation"]), training_samples=len(splits["train"]),
                           excluded_history_volume_share=float(safe_ratio(volume[excluded].sum(), volume[eligible].sum()))))
    plan = pd.concat([plan, pd.DataFrame(counts)], axis=1)
    plan.to_csv(output_dir / "backtest_plan.csv", index=False)
    source_digest = None
    if source_path is not None:
        with open(source_path, "rb") as source_file:
            source_digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    manifest = {
        "schema_version": 1, "config": asdict(pipeline.config),
        "feature_rows": len(features), "model_feature_count": len(pipeline.model_feature_columns),
        "complete_target_rows": int(targets.label_complete.sum()),
        "model_feature_columns": pipeline.model_feature_columns,
        "categorical_feature_columns": ["product_id", *CATEGORIES],
        "keys": KEYS, "excluded_feature_columns": ["forecast_origin", "history_eligible"],
        "target_columns": [column for column in targets if column not in KEYS],
        "latest_inference_origin": str(features.forecast_origin.max().date()),
        "source_sha256": source_digest,
        "verified_holidays": None if pipeline.verified_holidays is None else sorted(str(day.date()) for day in pipeline.verified_holidays),
        "not_implemented": {"spatial": "No location coordinates or store identities in source",
                            "holiday_inference": "A verified complete holiday calendar must be supplied explicitly",
                            "models": "No model training or tuning performed"},
        "numeric_missingness": "Missing measurements, insufficient history and undefined denominators remain NaN; source zeros stay distinct",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/candidate_dataset.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    pipeline = RetailFeaturePipeline(pd.read_parquet(args.data))
    print(pipeline.quality_report().to_string(), flush=True)
    def report(i, total, origin):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"Features: {i}/{total} origins, through {origin.date()}", flush=True)
    features, targets = pipeline.build_dataset(pipeline.default_origins(args.step_days), progress=report)
    plan = export_artifacts(pipeline, features, targets, args.output_dir, source_path=args.data)
    print(plan.to_string(index=False), flush=True)
    print(f"Wrote {len(features):,} rows and {len(pipeline.model_feature_columns)} model features to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
