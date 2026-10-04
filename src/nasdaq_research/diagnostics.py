"""Stage 7: read-only, offline diagnostics of existing research features.

Reports describe the input; they never produce a cleaned or modeling dataset.
Daily fundamental repetitions and distinct filing snapshots are separate views.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from nasdaq_research.alignment import RESEARCH_DATA_DIR
from nasdaq_research.dates import date_columns, parse_dates
from nasdaq_research.features import WARMUP_ROWS
from nasdaq_research.fundamental_mapping import FINANCIAL_FIELDS
from nasdaq_research.research_features import (
    ANNUAL_GROWTH_FEATURES, ANNUAL_RATIO_FEATURES, BALANCE_FIELDS,
    BALANCE_RATIOS, QUARTERLY_GROWTH_FEATURES, QUARTERLY_RATIO_FEATURES,
)

RESEARCH_GROUPS = ("market", "quarterly", "annual", "balance_sheet")
FEATURE_GROUPS = {
    **{c: "market" for c in WARMUP_ROWS},
    **{c: "quarterly" for c in (*QUARTERLY_RATIO_FEATURES, *QUARTERLY_GROWTH_FEATURES)},
    **{c: "annual" for c in (*ANNUAL_RATIO_FEATURES, *ANNUAL_GROWTH_FEATURES)},
    **{c: "balance_sheet" for c in BALANCE_RATIOS},
}
PREFIXES = {"quarterly": "q", "annual": "fy", "balance_sheet": "bs"}
STATE_KEYS = ("accession", "filing_date", "effective_date", "period_start", "period_end", "data_period_type")
PAIR_COLUMNS = ("view", "feature_group", "feature_a", "feature_b", "correlation",
                "pairwise_count", "distinct_states_a", "distinct_states_b",
                "status", "highly_correlated", "potentially_redundant")
LIMITATIONS = [
    "Results describe this ticker and this observed research window only.",
    "The daily window is about one year; serial dependence reduces independent information.",
    "Distinct fundamental states are few and need not be statistically independent.",
    "Daily repetitions are not independent fundamental observations.",
    "Correlation does not establish causality or predictive power.",
    "Stage 7 has no prediction target or future-return analysis.",
    "Usable means data quality permits further research, not a modeling recommendation.",
    "Historical snapshots use the archived Stage 6.1 vintage; they do not repair source limitations.",
    "Historical effective dates before coverage are sample-truncated; not historical exchange dates.",
    "Historical snapshot views stop strictly before the final market session's date.",
    "No feature selection, data cleaning, imputation, winsorization or scaling is performed.",
]


@dataclass(frozen=True)
class DiagnosticThresholds:
    """All diagnostic cutoffs live here; labels are not selection decisions."""

    high_coverage: float = 0.95
    sparse_coverage: float = 0.50
    near_constant_dominance: float = 0.99
    near_constant_min_observations: int = 20
    outlier_iqr_multiplier: float = 3.0
    outlier_min_observations: int = 8
    correlation_min_pairwise: int = 20
    high_correlation: float = 0.95
    short_daily_history: int = 60
    short_state_history: int = 5
    market_constant_run: int = 20

    def __post_init__(self):
        if not 0 < self.sparse_coverage < self.high_coverage <= 1:
            raise ValueError("Coverage thresholds must be ordered within (0, 1]")
        if not 0 < self.near_constant_dominance <= 1 or not 0 < self.high_correlation <= 1:
            raise ValueError("Dominance and correlation thresholds must be within (0, 1]")
        if self.outlier_iqr_multiplier <= 0 or min(
            self.near_constant_min_observations, self.outlier_min_observations,
            self.short_daily_history, self.short_state_history, self.market_constant_run,
        ) < 2 or self.correlation_min_pairwise < 3:
            raise ValueError("Positive cutoffs and at least three correlation observations required")


def classify_feature(name: str) -> str:
    """Use Stage 3/6 definitions, never numeric dtype or prefix alone.

    Numeric ages/fiscal years/flags remain metadata. Unknown columns stay in the
    inventory as unclassified and cannot silently enter numerical analysis.
    """
    if name in FEATURE_GROUPS:
        return FEATURE_GROUPS[name]
    if name in ("open", "high", "low", "close", "volume"):
        return "market_source"
    if name in FINANCIAL_FIELDS or name in {f"bs_{c}" for c in BALANCE_FIELDS}:
        return "fundamental_source"
    if name in ("date", "ticker", "days_since_filing", "days_since_effective_date") or (
        name.startswith(("q_", "fy_", "bs_", "fundamental_", "yoy_"))
        or name.endswith("_concept")
    ):
        return "metadata"
    return "unclassified"


def _finite(values: pd.Series) -> pd.Series:
    """Finite statistical view only; infinities are counted, source is untouched."""
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise ValueError(f"Research feature must be numeric: {values.name}")
    return values.astype(float).where(np.isfinite(values.astype(float)))


def _iso(value):
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _runs(mask) -> tuple[int, int]:
    lengths, length = [], 0
    for active in mask:
        if active:
            length += 1
        elif length:
            lengths.append(length)
            length = 0
    if length:
        lengths.append(length)
    return max(lengths, default=0), len(lengths)


def _changes(values: pd.Series) -> pd.Series:
    """Adjacent changes including availability changes, excluding initial row."""
    previous = values.shift()
    equal = values.eq(previous).fillna(False) | (values.isna() & previous.isna())
    changed = ~equal
    if len(changed):
        changed.iloc[0] = False
    return changed


def coverage_label(ratio: float, thresholds: DiagnosticThresholds) -> str:
    if ratio == 0:
        return "all_missing"
    if ratio == 1:
        return "full_coverage"
    if ratio >= thresholds.high_coverage:
        return "high_coverage"
    if ratio < thresholds.sparse_coverage:
        return "sparse"
    return "moderate_missingness"


def feature_inventory(data: pd.DataFrame, thresholds=DiagnosticThresholds()) -> pd.DataFrame:
    dates = parse_dates(data.date)
    rows = []
    for name, values in data.items():
        valid = values.notna()
        longest, blocks = _runs(~valid)
        changed = _changes(values) & valid & valid.shift(fill_value=False)
        ratio = float(valid.mean())
        rows.append({
            "feature_name": name, "feature_group": classify_feature(name),
            "is_research_feature": classify_feature(name) in RESEARCH_GROUPS,
            "dtype": str(values.dtype), "non_null_count": int(valid.sum()),
            "null_count": int((~valid).sum()), "missing_ratio": 1 - ratio,
            "coverage_ratio": ratio, "unique_count": int(values.nunique(dropna=True)),
            "first_valid_date": _iso(dates[valid].iloc[0]) if valid.any() else None,
            "last_valid_date": _iso(dates[valid].iloc[-1]) if valid.any() else None,
            "number_of_value_changes": int(changed.sum()),
            "longest_consecutive_missing_run": longest, "number_of_missing_blocks": blocks,
            "coverage_label": coverage_label(ratio, thresholds),
        })
    return pd.DataFrame(rows)


def distribution(values: pd.Series, thresholds=DiagnosticThresholds()) -> dict:
    finite = _finite(values).dropna()
    count = len(finite)
    quantiles = finite.quantile([.01, .05, .25, .5, .75, .95, .99])
    result = {"count": count, "non_finite_count": int((values.notna() & _finite(values).isna()).sum()),
              "mean": finite.mean(), "std": finite.std(ddof=1), "min": finite.min(),
              "p01": quantiles.loc[.01], "p05": quantiles.loc[.05], "p25": quantiles.loc[.25],
              "median": quantiles.loc[.5], "p75": quantiles.loc[.75],
              "p95": quantiles.loc[.95], "p99": quantiles.loc[.99], "max": finite.max(),
              "skewness": finite.skew() if count >= 3 else np.nan,
              "excess_kurtosis": finite.kurt() if count >= 4 else np.nan}
    iqr = result["p75"] - result["p25"]
    status = ("insufficient_observations" if count < thresholds.outlier_min_observations
              else "zero_iqr_not_assessed" if iqr == 0 else "assessed")
    lower = result["p25"] - thresholds.outlier_iqr_multiplier * iqr if status == "assessed" else np.nan
    upper = result["p75"] + thresholds.outlier_iqr_multiplier * iqr if status == "assessed" else np.nan
    outliers = int(((finite < lower) | (finite > upper)).sum()) if status == "assessed" else 0
    result.update({"outlier_method": "outer_IQR_fences", "outlier_status": status,
                   "outlier_lower_fence": lower, "outlier_upper_fence": upper,
                   "outlier_count": outliers, "potential_outlier_issue": outliers > 0})
    return result


def near_constant(values: pd.Series, thresholds=DiagnosticThresholds()) -> dict:
    values = _finite(values).dropna()
    count, unique = len(values), int(values.nunique())
    dominant = float(values.value_counts().iloc[0] / count) if count else 0.0
    flag = (count >= 2 and unique == 1) or (
        count >= thresholds.near_constant_min_observations and dominant >= thresholds.near_constant_dominance)
    return {"near_constant": bool(flag), "diagnostic_observation_count": count,
            "diagnostic_unique_count": unique, "dominant_value_ratio": dominant}


def state_keys(group: str) -> list[str]:
    return [f"{PREFIXES[group]}_{c}" for c in STATE_KEYS]


def state_frame(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """One actual daily selected state, keyed by provenance, not feature values."""
    keys = state_keys(group)
    if not set(keys) <= set(data):
        raise ValueError(f"Missing {group} state metadata: {sorted(set(keys) - set(data))}")
    present = data[keys[0]].notna()
    return data.loc[present].drop_duplicates(keys, keep="first").reset_index(drop=True)


def temporal_diagnostics(data: pd.DataFrame, features: list[str], thresholds=DiagnosticThresholds()) -> pd.DataFrame:
    dates = parse_dates(data.date)
    rows = []
    for name in features:
        group, values = classify_feature(name), data[name]
        valid = _finite(values).notna()
        changes = _changes(values)
        adjacent_changes = changes & valid & valid.shift(fill_value=False)
        constant_run = longest = 0
        for i in range(len(values)):
            constant_run = (constant_run + 1 if i and valid.iloc[i] and valid.iloc[i - 1]
                            and not changes.iloc[i] else int(valid.iloc[i]))
            longest = max(longest, constant_run)
        leading = next((i for i, present in enumerate(valid) if present), len(values))
        notes, issues = [], set()
        state_count, state_change_count, aligned_count = None, None, None
        expected_warmup, unexpected_missing, missing_warmup = None, None, None
        if group == "market":
            expected_warmup = min(WARMUP_ROWS[name], len(data))
            expected = np.arange(len(data)) < expected_warmup
            missing = values.isna().to_numpy()
            unexpected_missing = int((missing & ~expected).sum())
            missing_warmup = int((~missing & expected).sum())
            if unexpected_missing or missing_warmup:
                notes.append("unexpected_market_missingness")
                issues.update(dates.iloc[np.flatnonzero(missing != expected)])
            if longest >= thresholds.market_constant_run:
                notes.append("abnormally_long_market_constant_run")
            diagnostic_values = values
        else:
            prefix, keys = PREFIXES[group], state_keys(group)
            effective = parse_dates(data[f"{prefix}_effective_date"], allow_missing=True)
            filed = parse_dates(data[f"{prefix}_filing_date"], allow_missing=True)
            present = data[f"{prefix}_accession"].notna()
            states_changed = pd.concat([_changes(data[c]) for c in keys], axis=1).any(axis=1)
            # Initial state at row zero is a sample boundary, not a historical change.
            expected_effective = [dates.iloc[pos] if pos < len(dates) else pd.NaT
                                  for pos in pd.DatetimeIndex(dates).searchsorted(filed, side="right")]
            bad_metadata = present & (
                filed.isna() | effective.isna() | filed.ge(dates) | effective.gt(dates)
                | effective.ne(pd.Series(expected_effective, index=data.index))
                | data.loc[:, keys].isna().any(axis=1))
            if group != "balance_sheet":
                bad_metadata |= present & data[f"{prefix}_data_period_type"].ne(
                    "quarterly" if group == "quarterly" else "annual")
            aligned = changes & states_changed & dates.eq(effective)
            # Missing-to-value and value-to-missing transitions matter too.
            bad_changes = changes & ~aligned
            bad_states = states_changed & (~present | ~dates.eq(effective))
            unsupported_value = values.notna() & ~present
            issues.update(dates[bad_metadata | bad_changes | bad_states | unsupported_value])
            if issues:
                notes.append("potential temporal inconsistency")
            states = state_frame(data, group)
            state_count = len(states)
            state_change_count = int(states_changed.sum())
            aligned_count = int(aligned.sum())
            diagnostic_values = states[name]
            notes.append("filing-driven step behavior; daily repetition is expected")
        constant = near_constant(diagnostic_values, thresholds)
        coverage = float(valid.mean())
        behavior = ("sparse" if coverage < thresholds.sparse_coverage else
                    "mostly_constant" if constant["near_constant"] else
                    "daily_varying" if group == "market" else "step_function")
        rows.append({
            "feature_name": name, "feature_group": group, "temporal_behavior": behavior,
            "value_change_count": int(adjacent_changes.sum()), "payload_transition_count": int(changes.sum()),
            "longest_constant_run": longest, "leading_missing_count": leading,
            "expected_warmup_count": expected_warmup, "unexpected_missing_count": unexpected_missing,
            "missing_warmup_count": missing_warmup, "distinct_fundamental_states": state_count,
            "state_change_count": state_change_count, "effective_date_aligned_changes": aligned_count,
            "potential_temporal_inconsistency": bool(issues) or "abnormally_long_market_constant_run" in notes,
            "inconsistency_dates": "|".join(_iso(d) for d in sorted(issues)),
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows)


def correlations(data: pd.DataFrame, features: list[str], *, view="daily_repeated",
                 group="all_research", thresholds=DiagnosticThresholds()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    finite = pd.DataFrame({name: _finite(data[name]) for name in features})
    observed = finite.notna().astype("int64")
    counts = observed.T.dot(observed)
    matrix = finite.corr(method="pearson", min_periods=thresholds.correlation_min_pairwise)
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            overlap = finite[a].notna() & finite[b].notna()
            count = int(counts.loc[a, b])
            coefficient = matrix.loc[a, b]
            status = ("insufficient_pairwise_sample" if count < thresholds.correlation_min_pairwise
                      else "constant_or_undefined" if pd.isna(coefficient) else "assessed")
            distinct = []
            for name in (a, b):
                basis = classify_feature(name)
                keys = state_keys(basis) if basis in PREFIXES else []
                distinct.append(len(data.loc[overlap].drop_duplicates(keys)) if keys and set(keys) <= set(data)
                                else count if basis in PREFIXES and view != "daily_repeated" else None)
            high = bool(pd.notna(coefficient) and abs(coefficient) >= thresholds.high_correlation)
            rows.append({"view": view, "feature_group": group, "feature_a": a, "feature_b": b,
                         "correlation": coefficient, "pairwise_count": count,
                         "distinct_states_a": distinct[0], "distinct_states_b": distinct[1],
                         "status": status, "highly_correlated": high, "potentially_redundant": high})
    return matrix, counts, pd.DataFrame(rows, columns=PAIR_COLUMNS)


def snapshot_views(data: pd.DataFrame, snapshots: pd.DataFrame, features: list[str]) -> tuple[dict, dict]:
    """Window states and available archived history; never join q and fy flows."""
    views, counts = {}, {}
    cutoff = parse_dates(data.date).iloc[-1]
    filed = parse_dates(snapshots.filing_date)
    known = snapshots.loc[filed.lt(cutoff)]
    counts.update({"archived_snapshot_count": len(snapshots), "historical_snapshot_count": len(known),
                   "excluded_at_or_after_window_end": int(filed.ge(cutoff).sum()),
                   "pre_window_historical_snapshot_count": int(filed.lt(parse_dates(data.date).iloc[0]).sum()),
                   "historical_filing_date_range": {"start": _iso(filed[filed.lt(cutoff)].min()),
                                                    "end": _iso(filed[filed.lt(cutoff)].max())}})
    for group, prefix in PREFIXES.items():
        columns = [c for c in features if classify_feature(c) == group]
        window = state_frame(data, group)
        if group == "balance_sheet":
            mask = known.data_period_type.isin(["quarterly", "annual"]) & known.loc[:, BALANCE_FIELDS].notna().any(axis=1)
        else:
            mask = known.data_period_type.eq("quarterly" if group == "quarterly" else "annual")
        history = known.loc[mask].copy().reset_index(drop=True)
        views[("window_snapshot", group)] = (window, columns)
        views[("historical_snapshot", group)] = (history, columns)
        counts[group] = {"distinct_window_states": len(window), "historical_snapshots": len(history),
                         "window_non_missing_observations": {c: int(_finite(window[c]).notna().sum()) for c in columns}}
    return views, counts


def build_diagnostics(data: pd.DataFrame, snapshots: pd.DataFrame,
                      thresholds=DiagnosticThresholds()) -> tuple[dict, dict]:
    """Pure analysis: no filesystem access, no mutation of either dataframe."""
    inventory = feature_inventory(data, thresholds)
    features = inventory.loc[inventory.is_research_feature, "feature_name"].tolist()
    if not features:
        raise ValueError("No recognized research features")
    temporal = temporal_diagnostics(data, features, thresholds)
    matrix, counts, pairs = correlations(data, features, thresholds=thresholds)
    views, snapshot_counts = snapshot_views(data, snapshots, features)
    distributions = []
    for name in features:
        distributions.append({"view": "daily_repeated", "feature_group": classify_feature(name),
                              "feature_name": name, **distribution(data[name], thresholds)})
    snapshot_pairs = []
    for (view, group), (frame, columns) in views.items():
        for name in columns:
            distributions.append({"view": view, "feature_group": group, "feature_name": name,
                                  **distribution(frame[name], thresholds)})
        _, _, snapshot_pair = correlations(frame, columns, view=view, group=group, thresholds=thresholds)
        snapshot_pairs.append(snapshot_pair)
    distribution_frame = pd.DataFrame(distributions)
    quality = []
    redundant = set(pairs.loc[pairs.highly_correlated, ["feature_a", "feature_b"]].to_numpy().ravel())
    for row in inventory.loc[inventory.is_research_feature].to_dict("records"):
        name, group = row["feature_name"], row["feature_group"]
        diagnostic_values = data[name] if group == "market" else views[("window_snapshot", group)][0][name]
        stats, const = distribution(diagnostic_values, thresholds), near_constant(diagnostic_values, thresholds)
        timing = temporal.loc[temporal.feature_name.eq(name)].iloc[0]
        sparse = row["coverage_label"] == "sparse"
        all_missing = row["non_null_count"] == 0
        short = stats["count"] < (thresholds.short_daily_history if group == "market" else thresholds.short_state_history)
        usable = (row["coverage_ratio"] >= thresholds.high_coverage and not const["near_constant"]
                  and const["diagnostic_unique_count"] > 1 and stats["count"] >= 2
                  and not timing.potential_temporal_inconsistency
                  and not distribution(data[name], thresholds)["non_finite_count"])
        flags = {"usable": bool(usable), "sparse": sparse, "all_missing": all_missing,
                 "near_constant": const["near_constant"], "short_history": short,
                 "potential_outlier_issue": stats["potential_outlier_issue"],
                 "high_redundancy_candidate": name in redundant,
                 "potential_temporal_inconsistency": bool(timing.potential_temporal_inconsistency)}
        notes = [f"coverage={row['coverage_label']}", f"outlier_assessment={stats['outlier_status']}"]
        if group != "market":
            notes.append("near-constant/outlier/short-history labels use distinct window states")
            if name in redundant:
                notes.append("redundancy flag comes from daily repetition; check distinct state counts")
        if short:
            notes.append("limited diagnostic observations")
        if usable:
            notes.append("usable for data-quality research only; predictive ability untested")
        quality.append({**row, "std": distribution(data[name], thresholds)["std"],
                        "diagnostic_view": "daily_repeated" if group == "market" else "window_snapshot",
                        "diagnostic_std": stats["std"], **const, **flags,
                        "outlier_status": stats["outlier_status"], "outlier_count": stats["outlier_count"],
                        "non_finite_count": distribution(data[name], thresholds)["non_finite_count"],
                        "quality_flags": "|".join(k for k, v in flags.items() if v),
                        "quality_notes": "; ".join(notes)})
    tables = {
        "feature_inventory": inventory,
        "feature_missingness": inventory.loc[inventory.is_research_feature].reset_index(drop=True),
        "feature_quality": pd.DataFrame(quality), "feature_distributions": distribution_frame,
        "feature_correlations": matrix, "correlation_pairwise_counts": counts,
        "correlation_pairs": pairs, "high_correlation_pairs": pairs.loc[pairs.highly_correlated].sort_values(
            "correlation", key=lambda values: values.abs(), ascending=False, kind="stable").reset_index(drop=True),
        "temporal_diagnostics": temporal,
        "snapshot_correlations": pd.concat(snapshot_pairs, ignore_index=True),
    }
    state_rows = []
    for group in PREFIXES:
        frame = views[("window_snapshot", group)][0]
        for _, row in frame.iterrows():
            keys = state_keys(group)
            mask = pd.Series(True, index=data.index)
            for key in keys:
                mask &= data[key].eq(row[key]).fillna(False)
            state_rows.append({"feature_group": group, **{key.split("_", 1)[1]: row[key] for key in keys},
                               "first_observed_date": data.loc[mask, "date"].iloc[0],
                               "last_observed_date": data.loc[mask, "date"].iloc[-1], "daily_repeat_count": int(mask.sum())})
    tables["fundamental_states"] = pd.DataFrame(state_rows, columns=(
        "feature_group", *STATE_KEYS, "first_observed_date", "last_observed_date", "daily_repeat_count"))
    quality_frame = tables["feature_quality"]
    groups = inventory.feature_group.value_counts().to_dict()
    summary = {
        "ticker": str(data.ticker.iloc[0]), "row_count": len(data), "column_count": len(data.columns),
        "date_range": {"start": _iso(parse_dates(data.date).iloc[0]), "end": _iso(parse_dates(data.date).iloc[-1])},
        "research_feature_count": len(features), "feature_group_counts": groups,
        **{f"{group}_feature_count": int(groups.get(group, 0)) for group in RESEARCH_GROUPS},
        "metadata_count": int(groups.get("metadata", 0)), "thresholds": asdict(thresholds),
        "unclassified_columns": inventory.loc[inventory.feature_group.eq("unclassified"), "feature_name"].tolist(),
        "high_correlation_pair_count": len(tables["high_correlation_pairs"]),
        "snapshot_high_correlation_pair_counts": {
            f"{view}/{group}": int(((tables["snapshot_correlations"].view == view)
                                    & (tables["snapshot_correlations"].feature_group == group)
                                    & tables["snapshot_correlations"].highly_correlated).sum())
            for view, group in views},
        "fundamental_snapshots": snapshot_counts,
        "known_missingness_checks": {
            name: {"non_null_count": int(data[name].notna().sum()), "coverage_ratio": float(data[name].notna().mean()),
                   "all_missing": bool(data[name].isna().all())} if name in data else {"present": False}
            for name in ("q_fcf_margin", "fy_fcf_margin", "q_ocf_margin", "q_operating_cash_flow_growth_yoy")},
        "temporal_behavior_counts": temporal.temporal_behavior.value_counts().to_dict(),
        "fundamental_temporal_inconsistencies": temporal.loc[
            temporal.feature_group.ne("market") & temporal.potential_temporal_inconsistency, "feature_name"].tolist(),
        "market_temporal_inconsistencies": temporal.loc[
            temporal.feature_group.eq("market") & temporal.potential_temporal_inconsistency, "feature_name"].tolist(),
        "limitations": LIMITATIONS,
        "methods": {
            "classification": "Existing Stage 3/6 feature definitions; metadata and source values excluded from statistics",
            "missingness": "Original NaN cells; no imputation; missing blocks count adjacent observed sessions",
            "distributions": "Finite values only; sample std ddof=1; linear quantiles; pandas skew and excess kurtosis",
            "outliers": f"Q1 - {thresholds.outlier_iqr_multiplier}*IQR and Q3 + {thresholds.outlier_iqr_multiplier}*IQR; "
                        f"zero IQR or fewer than {thresholds.outlier_min_observations} values left unassessed",
            "near_constant": f"One distinct value with >=2 observations, or >={thresholds.near_constant_dominance:.0%} "
                             f"dominant value with >={thresholds.near_constant_min_observations} observations",
            "fundamental_quality": "Distinct window filing states, including NaN states, not repeated daily values",
            "correlation": f"Pearson, finite pairwise complete observations; minimum {thresholds.correlation_min_pairwise} for all views",
            "snapshot_correlation": f"Within each basis only; fewer than {thresholds.correlation_min_pairwise} complete snapshots gives no coefficient/flag",
            "temporal": "Adjacent payload changes including NaN transitions must accompany a new state on its exact observed effective date",
        },
    }
    for flag in ("all_missing", "sparse", "near_constant", "short_history", "potential_outlier_issue", "high_redundancy_candidate", "usable"):
        key = "potential_outlier_features" if flag == "potential_outlier_issue" else f"{flag}_features"
        summary[key] = quality_frame.loc[quality_frame[flag], "feature_name"].tolist()
    summary["coverage_label_counts"] = quality_frame.coverage_label.value_counts().to_dict()
    summary["outlier_features_by_view"] = {
        view: distribution_frame.loc[distribution_frame.view.eq(view) & distribution_frame.potential_outlier_issue,
                                     "feature_name"].tolist()
        for view in ("daily_repeated", "window_snapshot", "historical_snapshot")}
    return tables, summary


def validate_diagnostics(data: pd.DataFrame, original: pd.DataFrame, tables: dict, summary: dict,
                         snapshots: pd.DataFrame | None = None) -> dict:
    """Independently audit report identities, count overlaps and change reporting."""
    checks = {}
    try:
        pd.testing.assert_frame_equal(data, original, check_exact=True)
        checks["source_dataframe_unchanged"] = True
    except AssertionError:
        checks["source_dataframe_unchanged"] = False
    inventory = tables["feature_inventory"]
    expected = [c for c in original if classify_feature(c) in RESEARCH_GROUPS]
    checks["inventory_matches_every_source_column"] = inventory.feature_name.tolist() == original.columns.tolist()
    checks["research_features_match_source"] = tables["feature_quality"].feature_name.tolist() == expected
    checks["row_count_unchanged"] = summary["row_count"] == len(original)
    checks["column_count_correct"] = summary["column_count"] == len(original.columns)
    dates = parse_dates(original.date)
    checks["date_range_correct"] = summary["date_range"] == {"start": _iso(dates.iloc[0]), "end": _iso(dates.iloc[-1])}
    checks["ticker_correct"] = original.ticker.nunique() == 1 and summary["ticker"] == original.ticker.iloc[0]
    checks["feature_count_correct"] = summary["research_feature_count"] == len(expected)
    checks["all_missing_correct"] = set(summary["all_missing_features"]) == {c for c in expected if original[c].isna().all()}
    checks["no_future_derived_column_created"] = all(
        not re.search(r"(?:forward|future|target|lead)[_\s].*(?:return|\d)|(?:prediction_target)", str(c), re.I)
        for table in tables.values() for c in table.columns) and all(
            name in original for name in tables["feature_quality"].feature_name)
    overlaps_ok, coefficients_ok = True, True
    counts, matrix = tables["correlation_pairwise_counts"], tables["feature_correlations"]
    minimum = summary["thresholds"]["correlation_min_pairwise"]
    for a in expected:
        for b in expected:
            overlap = (_finite(original[a]).notna() & _finite(original[b]).notna())
            count = int(overlap.sum())
            overlaps_ok &= counts.loc[a, b] == count
            coefficient = matrix.loc[a, b]
            if pd.notna(coefficient):
                coefficients_ok &= count >= minimum and abs(coefficient) <= 1 + 1e-12
            elif count < minimum:
                coefficients_ok &= pd.isna(coefficient)
    checks["correlation_pairwise_counts_exact"] = bool(overlaps_ok)
    checks["correlation_respects_minimum_sample"] = bool(coefficients_ok)
    # Audit the long-form reports too, not just their source matrices.
    checks["daily_pair_report_matches_matrices"] = all(
        row.pairwise_count == counts.loc[row.feature_a, row.feature_b]
        and (pd.isna(row.correlation) and pd.isna(matrix.loc[row.feature_a, row.feature_b])
             or row.correlation == matrix.loc[row.feature_a, row.feature_b])
        and row.highly_correlated == bool(pd.notna(row.correlation)
                                         and abs(row.correlation) >= summary["thresholds"]["high_correlation"])
        for row in tables["correlation_pairs"].itertuples())
    if snapshots is not None:
        views, _ = snapshot_views(original, snapshots, expected)
        counts_ok, coefficients_ok = True, True
        for row in tables["snapshot_correlations"].itertuples():
            frame = views[(row.view, row.feature_group)][0]
            overlap = _finite(frame[row.feature_a]).notna() & _finite(frame[row.feature_b]).notna()
            counts_ok &= row.pairwise_count == int(overlap.sum())
            coefficients_ok &= pd.isna(row.correlation) or row.pairwise_count >= minimum
        checks["snapshot_pairwise_counts_exact"] = bool(counts_ok)
        checks["snapshot_correlation_respects_minimum_sample"] = bool(coefficients_ok)
    checks["fundamental_changes_aligned_or_flagged"] = all(
        row.potential_temporal_inconsistency or row.payload_transition_count == row.effective_date_aligned_changes
        for row in tables["temporal_diagnostics"].itertuples() if row.feature_group != "market")
    checks["no_feature_removed"] = len(tables["feature_missingness"]) == len(expected)
    return {"valid": all(checks.values()), "checks": checks,
            "errors": [name for name, passed in checks.items() if not passed]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    # Explicit string IDs/dates keep accessions and fiscal metadata unambiguous.
    strings = [c for c in columns if c in date_columns(columns) or c == "ticker"
               or c == "accession" or c.endswith("_accession")]
    return pd.read_csv(path, dtype={c: "string" for c in strings}, float_precision="round_trip")


def _check_inputs(data: pd.DataFrame, snapshots: pd.DataFrame):
    if data.empty or not {"date", "ticker"} <= set(data):
        raise ValueError("Nonempty feature matrix with date/ticker required")
    dates = parse_dates(data.date)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Source dates must be unique and ascending; no reordering is performed")
    if data.ticker.isna().any() or data.ticker.nunique() != 1:
        raise ValueError("Exactly one nonmissing ticker required")
    required = {"ticker", "filing_date", "accession", "period_start", "period_end", "data_period_type", *BALANCE_FIELDS}
    if not required <= set(snapshots):
        raise ValueError(f"Missing snapshot columns: {sorted(required - set(snapshots))}")
    if not snapshots.ticker.eq(data.ticker.iloc[0]).all():
        raise ValueError("Snapshot ticker differs from feature matrix")
    keys = ["accession", "period_start", "period_end", "data_period_type"]
    if snapshots.loc[:, keys].isna().any().any() or snapshots.duplicated(keys).any():
        raise ValueError("Snapshot identities must be nonmissing and unique")
    parse_dates(snapshots.filing_date)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(output_dir: Path, tables: dict, summary: dict):
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=name in ("feature_correlations", "correlation_pairwise_counts"),
                     index_label="feature_name" if name in ("feature_correlations", "correlation_pairwise_counts") else None)
    (output_dir / f"{summary['ticker']}_feature_diagnostics.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_pipeline(features_path=RESEARCH_DATA_DIR / "NVDA_features.csv",
                 snapshots_path=RESEARCH_DATA_DIR / "NVDA_fundamental_features.csv",
                 validation_path=RESEARCH_DATA_DIR / "NVDA_feature_validation.json",
                 output_dir=RESEARCH_DATA_DIR / "diagnostics", thresholds=DiagnosticThresholds()) -> dict:
    """Read only Stage 6.1 files; write reports/figures to a separate directory."""
    features_path, snapshots_path, validation_path, output_dir = map(
        lambda p: Path(p).resolve(), (features_path, snapshots_path, validation_path, output_dir))
    inputs = (features_path, snapshots_path, validation_path)
    if any(output_dir == p or output_dir in p.parents for p in inputs):
        raise ValueError("Output directory must not contain source artifacts")
    # Guard all nearby Stage 1–6.1 data, including the Stage 5 checkpoint.
    protected = set(inputs)
    excluded_dirs = (output_dir, (RESEARCH_DATA_DIR / "diagnostics").resolve())
    for root in {RESEARCH_DATA_DIR.parent.resolve(), features_path.parent}:
        if root.exists():
            protected.update(p.resolve() for p in root.rglob("*") if p.is_file()
                             and not any(directory in p.resolve().parents or p.resolve() == directory
                                         for directory in excluded_dirs))
    before = {str(p): _sha256(p) for p in sorted(protected)}
    data, snapshots = _read_csv(features_path), _read_csv(snapshots_path)
    original, original_snapshots = data.copy(deep=True), snapshots.copy(deep=True)
    _check_inputs(data, snapshots)
    upstream = json.loads(validation_path.read_text(encoding="utf-8"))
    if upstream.get("validation", {}).get("valid") is not True:
        raise ValueError("Stage 6.1 validation must pass before diagnostics")
    tables, summary = build_diagnostics(data, snapshots, thresholds)
    validation = validate_diagnostics(data, original, tables, summary, snapshots)
    pd.testing.assert_frame_equal(snapshots, original_snapshots, check_exact=True)
    validation["checks"]["source_snapshots_dataframe_unchanged"] = True
    from nasdaq_research.diagnostics_figures import generate_figures

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = generate_figures(data, tables, output_dir / "figures", summary["ticker"],
                                    thresholds.correlation_min_pairwise)
    summary.update({"input_feature_matrix": str(features_path), "input_fundamental_snapshots": str(snapshots_path),
                    "input_validation": str(validation_path), "source_sha256_before": before,
                    "output_directory": str(output_dir), "figure_directory": str(output_dir / "figures"),
                    "figures": [str(p) for p in figure_paths],
                    "reproducibility": {"randomness": "none", "generated_timestamp": "omitted",
                                        "numpy_version": np.__version__, "pandas_version": pd.__version__,
                                        "matplotlib_version": version("matplotlib")}})
    after = {path: _sha256(Path(path)) for path in before}
    summary["source_sha256_after"] = after
    validation["checks"]["source_artifact_hashes_unchanged"] = before == after
    validation["checks"]["source_rows_unchanged_on_disk"] = len(_read_csv(features_path)) == len(original)
    validation["checks"]["figures_generated"] = bool(figure_paths) and all(p.stat().st_size > 0 for p in figure_paths)
    validation["valid"] = all(validation["checks"].values())
    validation["errors"] = [k for k, passed in validation["checks"].items() if not passed]
    summary["validation"] = validation
    summary["validation_status"] = "passed" if validation["valid"] else "failed"
    summary["artifact_sha256"] = {str(p): _sha256(p) for p in figure_paths}
    _write_report(output_dir, tables, summary)
    # Hash exported CSVs too; the JSON itself is excluded to avoid a recursive hash.
    summary["artifact_sha256"].update({str(p): _sha256(p) for p in sorted(output_dir.glob("*.csv"))})
    (output_dir / f"{summary['ticker']}_feature_diagnostics.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not validation["valid"]:
        raise ValueError(f"Diagnostics validation failed: {validation['errors']}")
    return summary


def terminal_summary(summary: dict) -> str:
    counts, states = summary["feature_group_counts"], summary["fundamental_snapshots"]
    lines = [f"Ticker: {summary['ticker']}", f"Rows: {summary['row_count']}", f"Columns: {summary['column_count']}",
             f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}",
             f"Research features: {summary['research_feature_count']}"]
    lines.extend(f"{group} features: {counts.get(group, 0)}" for group in RESEARCH_GROUPS)
    lines.extend(f"{key}: {len(summary[key])}" for key in (
        "all_missing_features", "sparse_features", "near_constant_features", "potential_outlier_features"))
    lines.append(f"High-correlation daily pairs: {summary['high_correlation_pair_count']}")
    lines.append(f"Historical snapshots: {states['historical_snapshot_count']}")
    lines.extend(f"{group} states / historical snapshots: {states[group]['distinct_window_states']} / {states[group]['historical_snapshots']}"
                 for group in PREFIXES)
    lines.extend([f"Diagnostics: {summary['output_directory']}", f"Figures: {summary['figure_directory']}",
                  f"Validation: {summary['validation_status']}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=RESEARCH_DATA_DIR / "NVDA_features.csv")
    parser.add_argument("--snapshots-path", type=Path, default=RESEARCH_DATA_DIR / "NVDA_fundamental_features.csv")
    parser.add_argument("--validation-path", type=Path, default=RESEARCH_DATA_DIR / "NVDA_feature_validation.json")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DATA_DIR / "diagnostics")
    args = parser.parse_args()
    print(terminal_summary(run_pipeline(args.features_path, args.snapshots_path, args.validation_path, args.output_dir)))


if __name__ == "__main__":
    main()
