"""Stage 9/9.1: chronological splits, label availability and locked protocol.

No model, baseline predictions, target distribution analysis or preprocessing is
run here. Eligibility uses label availability and dates, never target magnitude.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nasdaq_research.config import DATA_DIR
from nasdaq_research.dates import parse_dates
from nasdaq_research.targets import (
    HORIZONS, PRIMARY_TARGET, RETURN_COLUMNS, _read_csv, validate_targets,
)

RESEARCH_DIR = DATA_DIR / "research"
PRIMARY_METRIC = "MAE"
SECONDARY_METRICS = ("RMSE", "R²", "Pearson correlation", "Spearman correlation", "Directional accuracy")
PROTOCOL_VERSION = "9.1"
LEGACY_OVERLAP_POLICY = "Report validation exits reaching the test calendar; keep prescribed blocks unchanged. No evaluation-label purge is applied."
MANIFEST_COLUMNS = (
    "date", "ticker", "target_name", "fold", "partition", "pre_test_gap", "nominal_role", "role", "is_usable",
    "exclusion_reason", "target_entry_date", "target_exit_date", "evaluation_start_date",
    "target_missing", "evaluation_locked", "final_train_candidate", "final_train_purged", "final_train_usable",
)
LIMITATIONS = [
    "The current dataset covers about one year for NVDA only.",
    "The final test has 50 nominal primary-labelable observations and each CV block has 25 under the default protocol.",
    "Overlapping targets reduce independent information; neither row counts nor folds imply independence.",
    "Purging reduces effective training sizes, especially for the 20-session target.",
    "Secondary final-test labels can be missing within the shared primary-defined date block.",
    "The dynamic pre-test gap reduces CV initial training history while preserving complete validation blocks.",
    "Training-label purge prevents training exits at/after an evaluation start; it does not make evaluation blocks independent.",
    "Gap rows are excluded from CV model selection but can return to horizon-specific final training when their labels have matured.",
    "No model, baseline predictions, transaction-cost simulation, feature selection or preprocessing is performed.",
    "The locked final test has not been used for performance evaluation.",
]


@dataclass(frozen=True)
class SplitConfig:
    """Explicit default protocol; never shrink blocks to make a sample fit."""

    final_test_size: int = 50
    validation_size: int = 25
    n_folds: int = 3
    minimum_initial_training_size: int = 60
    embargo_sessions: int = 0

    def __post_init__(self) -> None:
        for name in ("final_test_size", "validation_size", "n_folds", "minimum_initial_training_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.embargo_sessions) is not int or self.embargo_sessions != 0:
            raise ValueError("Only embargo_sessions=0 is supported by this causal protocol")


def _iso(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _checked_data(data: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Check a copy; accept missing secondary labels but require a true primary tail."""
    required = {"date", "ticker", "target_entry_date", *RETURN_COLUMNS,
                *(f"target_exit_date_{h}d" for h in HORIZONS)}
    if data.empty or not required <= set(data):
        raise ValueError(f"Nonempty labeled data missing required columns: {sorted(required - set(data))}")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names")
    if not data.ticker.eq("NVDA").fillna(False).all():
        raise ValueError("Stage 9 supports exactly ticker == NVDA")
    checked = data.loc[:, sorted(required)].copy().reset_index(drop=True)
    checked["date"] = parse_dates(checked.date)
    if checked.date.duplicated().any():
        raise ValueError("Duplicate ticker/trading-date observations")
    if not checked.date.is_monotonic_increasing:
        raise ValueError("Dates must be ascending; input is not sorted or reordered automatically")
    for c in ("target_entry_date", *(f"target_exit_date_{h}d" for h in HORIZONS)):
        checked[c] = parse_dates(checked[c], allow_missing=True)
    for target, h in zip(RETURN_COLUMNS, HORIZONS):
        values = checked[target]
        if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_complex_dtype(values)):
            raise ValueError(f"Target must be real numeric or missing: {target}")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Non-finite target: {target}")
        entry, exit_date = checked.target_entry_date, checked[f"target_exit_date_{h}d"]
        valid = values.notna()
        if (valid & (entry.isna() | exit_date.isna())).any():
            raise ValueError(f"Missing required target metadata: {target}")
        if (entry.notna() & (~entry.isin(checked.date) | entry.le(checked.date))).any():
            raise ValueError("Entry must be a later observed session")
        if (exit_date.notna() & (entry.isna() | ~exit_date.isin(checked.date) | exit_date.lt(entry))).any():
            raise ValueError(f"Invalid observed exit date: {target}")
    primary = checked[PRIMARY_TARGET].notna()
    missing_positions = np.flatnonzero(~primary.to_numpy())
    if len(missing_positions) and primary.iloc[missing_positions[0]:].any():
        raise ValueError("Primary missing labels must form a contiguous unlabeled tail, not interior gaps")
    if checked.loc[~primary, "target_exit_date_5d"].notna().any():
        raise ValueError("Primary unlabeled tail must lack future exit provenance")
    required_count = config.final_test_size + config.n_folds * config.validation_size + config.minimum_initial_training_size
    if int(primary.sum()) < required_count:
        raise ValueError(f"Insufficient primary-labelable observations: {int(primary.sum())}; protocol requires {required_count}")
    return checked


def validation_history_eligibility(data: pd.DataFrame, final_test_start: pd.Timestamp,
                                   registered_horizons: tuple[int, ...] = HORIZONS) -> pd.Series:
    """Require complete, pre-test labels for every registered horizon; never use returns."""
    if (not registered_horizons or len(set(registered_horizons)) != len(registered_horizons)
            or any(type(h) is not int or h not in HORIZONS for h in registered_horizons)):
        raise ValueError("Registered horizons must be a nonempty unique subset of supported horizons")
    eligible = parse_dates(data.target_entry_date, allow_missing=True).notna()
    for h in registered_horizons:
        exits = parse_dates(data[f"target_exit_date_{h}d"], allow_missing=True)
        eligible &= data[f"forward_return_{h}d"].notna() & exits.notna() & exits.lt(final_test_start)
    return eligible


def _layout(data: pd.DataFrame, config: SplitConfig) -> tuple[list[int], list[int], list[int], list[int], list[tuple[str, list[int], list[int]]]]:
    """Keep the test fixed; move full CV blocks into label-available history."""
    labeled = data.index[data[PRIMARY_TARGET].notna()].tolist()
    tail = data.index[data[PRIMARY_TARGET].isna()].tolist()
    development, test = labeled[:-config.final_test_size], labeled[-config.final_test_size:]
    eligible_mask = validation_history_eligibility(data, data.date.iloc[test[0]])
    eligible = [i for i in development if eligible_mask.iloc[i]]
    needed = config.n_folds * config.validation_size
    if len(eligible) < needed or eligible[-needed] < config.minimum_initial_training_size:
        raise ValueError("Insufficient history after dynamic pre_test_gap for full validation blocks and minimum initial training size")
    validation_history = eligible[-needed:]
    gap = [i for i in development if i > eligible[-1]]
    blocks = []
    for i in range(config.n_folds):
        evaluation = validation_history[i * config.validation_size:(i + 1) * config.validation_size]
        blocks.append((f"cv_{i + 1}", development[:evaluation[0]], evaluation))
    blocks.append(("final", development, test))
    return development, test, tail, gap, blocks


def purge_eligibility(target_present: pd.Series, exit_dates: pd.Series,
                      evaluation_start_date: pd.Timestamp) -> pd.Series:
    """A candidate is usable iff its own label exists and exit is strictly earlier."""
    return target_present & exit_dates.notna() & exit_dates.lt(evaluation_start_date)


def _manifest(data: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    development, test, tail, gap, blocks = _layout(data, config)
    dev_set, test_set, tail_set, gap_set = set(development), set(test), set(tail), set(gap)
    records = []
    for fold, train, evaluation in blocks:
        train_set, evaluation_set = set(train), set(evaluation)
        evaluation_start = data.date.iloc[evaluation[0]]
        for target, h in zip(RETURN_COLUMNS, HORIZONS):
            present = data[target].notna()
            exits = data[f"target_exit_date_{h}d"]
            usable_train = purge_eligibility(present, exits, evaluation_start)
            for position in data.index:
                partition = "development" if position in dev_set else "final_test" if position in test_set else "unlabeled_tail"
                nominal = ("unlabeled_tail" if position in tail_set else "pre_test_gap" if fold != "final" and position in gap_set else
                           "train" if position in train_set else
                           ("final_test" if fold == "final" else "validation") if position in evaluation_set else "not_candidate")
                missing = not bool(present.iloc[position])
                reason = ""
                if nominal == "unlabeled_tail":
                    role, reason = "unlabeled_tail", "primary_unlabeled_tail"
                elif nominal == "pre_test_gap":
                    role, reason = "pre_test_gap", "reserved_for_registered_validation_label_maturation"
                elif missing:
                    role, reason = "target_missing", "target_missing"
                elif nominal == "train" and not usable_train.iloc[position]:
                    role, reason = "purged", "target_exit_at_or_after_evaluation_start"
                elif nominal == "not_candidate":
                    role, reason = "not_candidate", "outside_this_training_and_evaluation_block"
                else:
                    role = nominal
                records.append({
                    "date": _iso(data.date.iloc[position]), "ticker": "NVDA", "target_name": target, "fold": fold,
                    "partition": partition, "pre_test_gap": position in gap_set, "nominal_role": nominal, "role": role,
                    "is_usable": role in ("train", "validation", "final_test"), "exclusion_reason": reason,
                    "target_entry_date": _iso(data.target_entry_date.iloc[position]),
                    "target_exit_date": _iso(exits.iloc[position]), "evaluation_start_date": _iso(evaluation_start),
                    "target_missing": missing, "evaluation_locked": partition == "final_test",
                    "final_train_candidate": fold == "final" and nominal == "train",
                    "final_train_purged": fold == "final" and role == "purged",
                    "final_train_usable": fold == "final" and role == "train",
                })
    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def _fold_summaries(manifest: pd.DataFrame, config: SplitConfig, test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts and date geometry only; no target means or performance statistics."""
    rows = []
    for fold in (*[f"cv_{i + 1}" for i in range(config.n_folds)], "final"):
        for target in RETURN_COLUMNS:
            m = manifest.loc[manifest.fold.eq(fold) & manifest.target_name.eq(target)]
            candidates = m.loc[m.nominal_role.eq("train")]
            train = m.loc[m.role.eq("train")]
            evaluation = m.loc[m.nominal_role.eq("final_test" if fold == "final" else "validation")]
            exit_dates = parse_dates(evaluation.target_exit_date, allow_missing=True)
            rows.append({
                "target_name": target, "fold": fold, "train_start_date": candidates.date.min(),
                "train_end_date_nominal": candidates.date.max(), "train_end_date_effective": train.date.max(),
                "train_count_nominal": len(candidates), "train_count_effective": len(train),
                "purged_count": int(candidates.role.eq("purged").sum()),
                "train_missing_target_count": int(candidates.role.eq("target_missing").sum()),
                "validation_start_date": evaluation.date.min(), "validation_end_date": evaluation.date.max(),
                "validation_count": len(evaluation), "validation_count_effective": int(evaluation.is_usable.sum()),
                "evaluation_missing_target_count": int(evaluation.target_missing.sum()), "embargo_sessions": config.embargo_sessions,
                "max_evaluation_target_exit_date": _iso(exit_dates.max()),
                "pre_test_gap_rows": int(m.pre_test_gap.sum()),
                "pre_test_gap_final_train_usable_count": int((train.pre_test_gap).sum()) if fold == "final" else 0,
                "validation_labels_reaching_final_test": int((~evaluation.target_missing & exit_dates.ge(test_start)).sum()) if fold != "final" else 0,
            })
    all_folds = pd.DataFrame(rows)
    cv = all_folds.loc[all_folds.fold.ne("final")].rename(columns={
        "max_evaluation_target_exit_date": "max_validation_target_exit_date",
    }).drop(columns="pre_test_gap_final_train_usable_count").reset_index(drop=True)
    cv["final_test_start_date"] = _iso(test_start)
    final = all_folds.loc[all_folds.fold.eq("final")].rename(columns={
        "train_count_nominal": "final_train_count_nominal", "train_count_effective": "final_train_count_effective",
        "purged_count": "final_purged_count", "validation_start_date": "final_test_start_date",
        "validation_end_date": "final_test_end_date", "validation_count": "final_test_count",
        "validation_count_effective": "final_test_count_effective",
        "max_evaluation_target_exit_date": "max_final_test_target_exit_date",
    }).drop(columns="validation_labels_reaching_final_test").reset_index(drop=True)
    return cv, final


def evaluation_protocol(config: SplitConfig, boundaries: dict[str, Any]) -> dict[str, Any]:
    """Pre-register Stage 10 behavior as definitions only, with no metric results."""
    return {
        "protocol_version": PROTOCOL_VERSION, "registered_targets": list(RETURN_COLUMNS),
        "ticker": "NVDA", "primary_target": PRIMARY_TARGET, "secondary_targets": [RETURN_COLUMNS[0], RETURN_COLUMNS[2]],
        "config": asdict(config), "final_test_size": config.final_test_size, "cv_fold_count": config.n_folds,
        "validation_size": config.validation_size, "split_type": "chronological_expanding_window",
        "purge_rule": "target present AND target_exit_date < evaluation_start_date",
        "exact_boundary_policy": "target_exit_date == evaluation_start_date is purged",
        "secondary_target_boundaries": "Same primary-defined date blocks; independent horizon-specific training purge",
        "embargo_sessions": config.embargo_sessions,
        "embargo_reason": "Training is earlier than evaluation; provenance purge controls training overlap; no extra post-validation embargo",
        "primary_metric": PRIMARY_METRIC, "secondary_metrics": list(SECONDARY_METRICS),
        "metric_definitions": {
            "MAE": "mean(abs(y_true - prediction)); fixed primary metric",
            "RMSE": "sqrt(mean((y_true - prediction)**2))",
            "R²": "1 - SSE/SST; undefined if SST is zero",
            "Pearson correlation": "Linear correlation; undefined for constant vectors or fewer than two complete observations",
            "Spearman correlation": "Correlation of average ranks for ties; undefined for constant ranks or fewer than two complete observations",
            "Directional accuracy": "mean(sign(prediction) == sign(y_true)); zero is its own sign category",
        },
        "cv_metric_aggregation": "Equal-weight mean of fold MAE; report per-fold and per-horizon usable counts; no best-fold selection",
        "primary_metric_changes_after_results_allowed": False,
        "baselines": [
            {"name": "Zero-return baseline", "definition": "prediction = 0", "executed": False},
            {"name": "Historical-mean baseline", "definition": "mean of this horizon's usable, purged training-fold targets only",
             "fit_scope": "training_only", "validation_test_full_sample_mean_allowed": False, "executed": False},
        ],
        "preprocessing": {"fit_scope": "usable_training_only", "validation_test_action": "transform_only",
                          "full_sample_fit_allowed": False, "implemented": False,
                          "operations": ["imputation", "scaling", "normalization", "feature selection", "target-dependent filtering"]},
        "feature_selection": {"scope": "usable_training_only", "final_test_access_allowed": False, "implemented": False},
        "final_test_locked": True, "final_test_policy": "Final test is a one-shot out-of-sample evaluation set.",
        "final_test_use_for_model_feature_preprocessing_target_selection_allowed": False,
        "final_test_performance_evaluated": False, "random_split_allowed": False, "model_training_performed": False,
        "unlabeled_tail_policy": "Retained and excluded for every horizon, even if a secondary target is present",
        "boundaries": boundaries, "limitations": LIMITATIONS,
        "validation_label_availability_rule": "target present AND target_exit_date < final_test_start_date",
        "pre_test_gap_policy": "dynamic", "pre_test_gap_basis": "all_registered_horizons",
        "longest_registered_horizon": f"forward_return_{max(HORIZONS)}d",
        "cv_validation_policy": "Last n_folds * validation_size fully label-available development observations; chronological full blocks",
        "missing_validation_label_policy": "Not safe; exclude from CV validation history for every registered horizon",
        "pre_test_gap_final_training_policy": "Gap membership is not a final-training exclusion; apply each horizon's original target-exit purge",
    }


def build_splits(data: pd.DataFrame, config: SplitConfig = SplitConfig()) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return compact manifest/summary artifacts; input data and targets remain untouched."""
    checked = _checked_data(data, config)
    development, test, tail, gap, blocks = _layout(checked, config)
    manifest = _manifest(checked, config)
    cv, final = _fold_summaries(manifest, config, checked.date.iloc[test[0]])
    if (cv.train_count_effective < 1).any() or (final.final_train_count_effective < 1).any():
        raise ValueError("Protocol leaves an empty effective training pool; configuration is not reduced automatically")
    boundaries = {"development": {"start": _iso(checked.date.iloc[development[0]]), "end": _iso(checked.date.iloc[development[-1]])},
                  "final_test": {"start": _iso(checked.date.iloc[test[0]]), "end": _iso(checked.date.iloc[test[-1]])},
                  "unlabeled_tail": {"start": _iso(checked.date.iloc[tail[0]]) if tail else None,
                                     "end": _iso(checked.date.iloc[tail[-1]]) if tail else None},
                  "pre_test_gap": {"start": _iso(checked.date.iloc[gap[0]]) if gap else None,
                                   "end": _iso(checked.date.iloc[gap[-1]]) if gap else None},
                  "cv": [{"fold": fold, "start": _iso(checked.date.iloc[evaluation[0]]),
                          "end": _iso(checked.date.iloc[evaluation[-1]])} for fold, _, evaluation in blocks if fold != "final"]}
    summary_rows = []
    for target in RETURN_COLUMNS:
        pool = final.loc[final.target_name.eq(target)].iloc[0]
        summary_rows.append({
            "target_name": target, "primary_target": PRIMARY_TARGET, "total_rows": len(checked),
            "primary_labelable_rows": len(development) + len(test), "unlabeled_tail_rows": len(tail),
            "development_rows": len(development), "final_test_rows": len(test), "cv_fold_count": config.n_folds,
            "validation_rows_per_fold": config.validation_size, "embargo_sessions": config.embargo_sessions,
            "pre_test_gap_rows": len(gap), "pre_test_gap_start_date": boundaries["pre_test_gap"]["start"],
            "pre_test_gap_end_date": boundaries["pre_test_gap"]["end"],
            "cv_history_rows": len(development) - len(gap), "cv_validation_total_rows": config.n_folds * config.validation_size,
            "initial_train_count_nominal": len(blocks[0][1]),
            "effective_train_count": int(pool.final_train_count_effective), "purged_count": int(pool.final_purged_count),
            "missing_target_count": int(checked[target].isna().sum()),
            "final_train_missing_target_count": int(pool.train_missing_target_count),
            "final_test_usable_count": int(pool.final_test_count_effective),
            "final_test_missing_target_count": int(pool.evaluation_missing_target_count),
            "final_test_locked": True,
        })
    protocol = evaluation_protocol(config, boundaries)
    protocol["pre_test_gap"] = {"row_count": len(gap), "start_date": boundaries["pre_test_gap"]["start"],
                                "end_date": boundaries["pre_test_gap"]["end"]}
    crossing = cv.loc[cv.validation_labels_reaching_final_test.gt(0),
                      ["target_name", "fold", "validation_labels_reaching_final_test"]].to_dict("records")
    protocol["validation_labels_reaching_final_test_calendar"] = crossing
    protocol["strict_pre_test_model_selection_boundary_satisfied"] = not crossing
    protocol["stage10_readiness"] = "NOT READY FOR STAGE 10" if crossing else "READY FOR STAGE 10"
    protocol["readiness_reason"] = ("Validation exits reach the locked test calendar" if crossing else
                                    "All three leakage boundaries pass; final readiness also requires pipeline source/round-trip validation and regression/reproducibility checks")
    return {"split_manifest": manifest, "cv_folds": cv, "final_training_pool": final,
            "split_summary": pd.DataFrame(summary_rows)}, protocol


def validate_splits(data: pd.DataFrame, tables: dict[str, pd.DataFrame], protocol: dict[str, Any],
                    config: SplitConfig = SplitConfig()) -> dict[str, Any]:
    """Audit roles against scalar source chronology, without calling the builder."""
    names = ("chronological_order_violations", "train_validation_overlap_violations", "target_overlap_leakage_violations",
             "exact_boundary_violations", "final_train_test_overlap_violations", "role_conflict_violations",
             "missing_required_target_metadata", "duplicate_date_count", "manifest_provenance_violations",
             "role_assignment_violations", "train_to_validation_violations", "final_training_to_final_test_violations",
             "validation_label_availability_violations")
    report: dict[str, Any] = {name: 0 for name in names}
    report.update({f"validation_label_crosses_final_test_{h}d": 0 for h in HORIZONS})
    report.update({"ticker": "NVDA", "primary_target": PRIMARY_TARGET, "total_rows": len(data),
                   "cv_fold_count": config.n_folds, "validation_size_per_fold": config.validation_size,
                   "embargo_sessions": config.embargo_sessions, "final_test_locked": True})
    checks: dict[str, bool] = {}
    try:
        if "date" in data:
            report["duplicate_date_count"] = int(parse_dates(data.date).duplicated().sum())
        for target, h in zip(RETURN_COLUMNS, HORIZONS):
            for metadata in ("target_entry_date", f"target_exit_date_{h}d"):
                if metadata not in data:
                    report["missing_required_target_metadata"] += 1
                elif target in data:
                    report["missing_required_target_metadata"] += int((data[target].notna() & data[metadata].isna()).sum())
        source = _checked_data(data, config)
    except (ValueError, TypeError) as exc:
        report.update({"valid": False, "validation_status": "failed", "errors": [str(exc)], "checks": {"valid_source": False}})
        return report
    # Independent positional layout: no production _layout or _manifest call.
    count = int(source[PRIMARY_TARGET].notna().sum())
    development_count = count - config.final_test_size
    tail_start = count
    test_start = source.date.iloc[development_count]
    eligible_history = []
    for i in range(development_count):
        complete = all(pd.notna(source[f"forward_return_{h}d"].iloc[i])
                       and pd.notna(source[f"target_exit_date_{h}d"].iloc[i])
                       and source[f"target_exit_date_{h}d"].iloc[i] < test_start for h in HORIZONS)
        if complete and pd.notna(source.target_entry_date.iloc[i]):
            eligible_history.append(i)
    needed = config.n_folds * config.validation_size
    if len(eligible_history) < needed or eligible_history[-needed] < config.minimum_initial_training_size:
        report.update({"valid": False, "validation_status": "failed", "errors": ["Insufficient history after dynamic pre_test_gap"],
                       "checks": {"post_gap_configuration_feasible": False}})
        return report
    validation_history = eligible_history[-needed:]
    gap = list(range(eligible_history[-1] + 1, development_count))
    gap_set = set(gap)
    contexts = {}
    for fold in range(config.n_folds):
        evaluation = validation_history[fold * config.validation_size:(fold + 1) * config.validation_size]
        contexts[f"cv_{fold + 1}"] = (evaluation[0], evaluation)
    contexts["final"] = (development_count, list(range(development_count, count)))
    report.update({"labelable_primary_rows": count, "unlabeled_tail_rows": len(source) - count,
                   "development_rows": development_count, "final_test_rows": config.final_test_size,
                   "pre_test_gap_rows": len(gap), "pre_test_gap_row_count": len(gap),
                   "pre_test_gap_start_date": _iso(source.date.iloc[gap[0]]) if gap else None,
                   "pre_test_gap_end_date": _iso(source.date.iloc[gap[-1]]) if gap else None,
                   "cv_history_rows": development_count - len(gap), "cv_eligible_history_rows": len(eligible_history),
                   "initial_train_count_nominal": validation_history[0],
                   "date_range": {"start": _iso(source.date.iloc[0]), "end": _iso(source.date.iloc[-1])}})
    manifest = tables["split_manifest"]
    checks["manifest_schema_exact"] = list(manifest.columns) == list(MANIFEST_COLUMNS)
    if not checks["manifest_schema_exact"]:
        report.update({"valid": False, "validation_status": "failed", "errors": ["Manifest schema differs"], "checks": checks})
        return report
    keys = ["fold", "target_name", "date"]
    report["role_conflict_violations"] = int(manifest.duplicated(keys).sum())
    expected_keys = [(fold, target, _iso(day)) for fold in contexts for target in RETURN_COLUMNS for day in source.date]
    actual_keys = list(manifest.loc[:, keys].itertuples(index=False, name=None))
    checks["manifest_covers_all_source_dates_targets_and_folds"] = set(actual_keys) == set(expected_keys) and len(actual_keys) == len(expected_keys)
    checks["manifest_order_stable"] = actual_keys == expected_keys
    lookup = {_iso(day): i for i, day in enumerate(source.date)}
    try:
        parse_dates(manifest.date)
        parse_dates(manifest.target_entry_date, allow_missing=True)
        parse_dates(manifest.target_exit_date, allow_missing=True)
        parse_dates(manifest.evaluation_start_date)
    except ValueError as exc:
        report.update({"valid": False, "validation_status": "failed", "errors": [str(exc)], "checks": checks})
        return report
    for row in manifest.itertuples(index=False):
        if row.date not in lookup or row.fold not in contexts or row.target_name not in RETURN_COLUMNS:
            report["role_assignment_violations"] += 1
            continue
        i = lookup[row.date]
        start, evaluation = contexts[row.fold]
        evaluation_start = source.date.iloc[start]
        target_index = RETURN_COLUMNS.index(row.target_name)
        exit_date = source[f"target_exit_date_{HORIZONS[target_index]}d"].iloc[i]
        missing = bool(pd.isna(source[row.target_name].iloc[i]))
        partition = "development" if i < development_count else "final_test" if i < count else "unlabeled_tail"
        nominal = ("unlabeled_tail" if i >= tail_start else "pre_test_gap" if row.fold != "final" and i in gap_set else
                   "train" if i < start else
                   ("final_test" if row.fold == "final" else "validation") if i in evaluation else "not_candidate")
        if i >= tail_start:
            expected_role, reason = "unlabeled_tail", "primary_unlabeled_tail"
        elif nominal == "pre_test_gap":
            expected_role, reason = "pre_test_gap", "reserved_for_registered_validation_label_maturation"
        elif missing:
            expected_role, reason = "target_missing", "target_missing"
        elif i < start and exit_date >= evaluation_start:
            expected_role, reason = "purged", "target_exit_at_or_after_evaluation_start"
        elif nominal == "not_candidate":
            expected_role, reason = "not_candidate", "outside_this_training_and_evaluation_block"
        else:
            expected_role, reason = nominal, ""
        report["role_assignment_violations"] += int(any((
            row.partition != partition, row.pre_test_gap != (i in gap_set), row.nominal_role != nominal, row.role != expected_role,
            row.is_usable != (expected_role in ("train", "validation", "final_test")), row.target_missing != missing,
            row.evaluation_locked != (partition == "final_test"), row.exclusion_reason != reason,
            row.final_train_candidate != (row.fold == "final" and nominal == "train"),
            row.final_train_purged != (row.fold == "final" and expected_role == "purged"),
            row.final_train_usable != (row.fold == "final" and expected_role == "train"))))
        report["manifest_provenance_violations"] += int(any((
            row.ticker != "NVDA", row.evaluation_start_date != _iso(evaluation_start),
            (None if pd.isna(row.target_entry_date) else row.target_entry_date) != _iso(source.target_entry_date.iloc[i]),
            (None if pd.isna(row.target_exit_date) else row.target_exit_date) != _iso(exit_date))))
        if row.role == "train":
            report["chronological_order_violations"] += int(source.date.iloc[i] >= evaluation_start)
            leakage = int(pd.isna(exit_date) or exit_date >= evaluation_start or missing)
            report["target_overlap_leakage_violations"] += leakage
            report["final_training_to_final_test_violations" if row.fold == "final" else "train_to_validation_violations"] += leakage
            report["exact_boundary_violations"] += int(exit_date == evaluation_start)
            report["final_train_test_overlap_violations"] += int(i >= development_count)
        if row.role == "validation":
            crosses = int(pd.notna(exit_date) and exit_date >= test_start)
            report[f"validation_label_crosses_final_test_{HORIZONS[target_index]}d"] += crosses
            report["validation_label_availability_violations"] += int(
                missing or pd.isna(exit_date) or pd.isna(source.target_entry_date.iloc[i]) or crosses)
    for _, group in manifest.groupby(keys, sort=False):
        roles = set(group.role)
        report["train_validation_overlap_violations"] += int({"train", "validation"} <= roles)
        report["final_train_test_overlap_violations"] += int({"train", "final_test"} <= roles)
    for name in names:
        checks[name.replace("_violations", "_valid")] = report[name] == 0
    checks["final_test_is_last_primary_block"] = manifest.loc[
        manifest.fold.eq("final") & manifest.target_name.eq(PRIMARY_TARGET) & manifest.nominal_role.eq("final_test"), "date"].tolist() == [_iso(day) for day in source.date.iloc[development_count:count]]
    checks["tail_never_usable"] = not manifest.loc[manifest.partition.eq("unlabeled_tail"), "is_usable"].any()
    checks["missing_targets_never_usable"] = not manifest.loc[manifest.target_missing, "is_usable"].any()
    checks["gap_never_used_in_cv"] = not manifest.loc[manifest.fold.ne("final") & manifest.pre_test_gap, "is_usable"].any()
    checks["gap_membership_exact"] = manifest.loc[manifest.fold.eq("final") & manifest.target_name.eq(PRIMARY_TARGET)
                                                & manifest.pre_test_gap, "date"].tolist() == [_iso(source.date.iloc[i]) for i in gap]
    validation_sets = []
    for fold, (start, evaluation) in contexts.items():
        if fold == "final":
            continue
        selected = manifest.loc[manifest.fold.eq(fold) & manifest.target_name.eq(PRIMARY_TARGET) & manifest.nominal_role.eq("validation"), "date"].tolist()
        checks[f"{fold}_validation_exact"] = selected == [_iso(day) for day in source.date.iloc[evaluation]] and len(selected) == config.validation_size
        validation_sets.append(set(selected))
    checks["validation_blocks_disjoint"] = sum(map(len, validation_sets)) == len(set().union(*validation_sets))
    expected_cv, expected_final = _fold_summaries(manifest, config, source.date.iloc[development_count])
    for name, expected in (("cv_folds", expected_cv), ("final_training_pool", expected_final)):
        try:
            pd.testing.assert_frame_equal(tables[name], expected, check_dtype=False, check_exact=True)
            checks[f"{name}_counts_and_dates_correct"] = True
        except AssertionError:
            checks[f"{name}_counts_and_dates_correct"] = False
    summary = tables["split_summary"]
    checks["summary_targets_exact"] = summary.target_name.tolist() == list(RETURN_COLUMNS)
    expected_summary = {"primary_target": PRIMARY_TARGET, "total_rows": len(source), "primary_labelable_rows": count,
                        "development_rows": development_count, "final_test_rows": config.final_test_size,
                        "unlabeled_tail_rows": len(source) - count, "cv_fold_count": config.n_folds,
                        "validation_rows_per_fold": config.validation_size, "embargo_sessions": config.embargo_sessions,
                        "pre_test_gap_rows": len(gap), "pre_test_gap_start_date": report["pre_test_gap_start_date"],
                        "pre_test_gap_end_date": report["pre_test_gap_end_date"],
                        "cv_history_rows": development_count - len(gap), "cv_validation_total_rows": needed,
                        "initial_train_count_nominal": validation_history[0],
                        "final_test_locked": True}
    checks["summary_global_counts_correct"] = all(c in summary and summary[c].eq(v).all() for c, v in expected_summary.items())
    target_counts_ok = True
    for row in summary.itertuples(index=False):
        if row.target_name not in RETURN_COLUMNS:
            target_counts_ok = False
            continue
        pool = expected_final.loc[expected_final.target_name.eq(row.target_name)].iloc[0]
        target_counts_ok &= (row.effective_train_count == pool.final_train_count_effective
                             and row.purged_count == pool.final_purged_count
                             and row.missing_target_count == int(source[row.target_name].isna().sum())
                             and row.final_train_missing_target_count == pool.train_missing_target_count
                             and row.final_test_usable_count == pool.final_test_count_effective
                             and row.final_test_missing_target_count == pool.evaluation_missing_target_count)
    checks["summary_horizon_counts_correct"] = bool(target_counts_ok)
    expected_contract = evaluation_protocol(config, {})
    checks["evaluation_protocol_fixed"] = all(
        protocol.get(key) == value for key, value in expected_contract.items() if key != "boundaries")
    checks["protocol_boundaries_correct"] = (
        protocol.get("boundaries", {}).get("final_test") == {"start": _iso(source.date.iloc[development_count]), "end": _iso(source.date.iloc[count - 1])}
        and protocol.get("boundaries", {}).get("development") == {"start": _iso(source.date.iloc[0]), "end": _iso(source.date.iloc[development_count - 1])}
        and protocol.get("boundaries", {}).get("unlabeled_tail") == {"start": _iso(source.date.iloc[count]) if count < len(source) else None,
                                                                  "end": _iso(source.date.iloc[-1]) if count < len(source) else None}
        and protocol.get("boundaries", {}).get("pre_test_gap") == {"start": report["pre_test_gap_start_date"], "end": report["pre_test_gap_end_date"]}
        and protocol.get("boundaries", {}).get("cv") == [
            {"fold": fold, "start": _iso(source.date.iloc[start]), "end": _iso(source.date.iloc[evaluation[-1]])}
            for fold, (start, evaluation) in contexts.items() if fold != "final"])
    checks["protocol_gap_metadata_correct"] = protocol.get("pre_test_gap") == {
        "row_count": len(gap), "start_date": report["pre_test_gap_start_date"], "end_date": report["pre_test_gap_end_date"]}
    report["validation_labels_reaching_final_test_calendar"] = expected_cv.loc[
        expected_cv.validation_labels_reaching_final_test.gt(0),
        ["target_name", "fold", "validation_labels_reaching_final_test"]].to_dict("records")
    checks["validation_test_calendar_overlap_reported"] = protocol.get("validation_labels_reaching_final_test_calendar") == report["validation_labels_reaching_final_test_calendar"]
    report["validation_label_crosses_final_test_total"] = sum(report[f"validation_label_crosses_final_test_{h}d"] for h in HORIZONS)
    checks["validation_labels_available_before_final_test"] = report["validation_label_availability_violations"] == 0
    report["validation_test_boundary_status"] = "passed" if checks["validation_labels_available_before_final_test"] else "failed"
    report["maximum_validation_target_exit_dates"] = {
        target: expected_cv.loc[expected_cv.target_name.eq(target), "max_validation_target_exit_date"].max()
        for target in RETURN_COLUMNS}
    crossing = report["validation_label_availability_violations"] > 0
    checks["readiness_matches_date_audit"] = (
        protocol.get("strict_pre_test_model_selection_boundary_satisfied") is (not crossing)
        and protocol.get("stage10_readiness") == ("NOT READY FOR STAGE 10" if crossing else "READY FOR STAGE 10"))
    report.update({"checks": {k: bool(v) for k, v in checks.items()},
                   "errors": [k for k, passed in checks.items() if not passed]})
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    return report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_audit(labeled_path: Path, targets_path: Path, features_path: Path, validation_path: Path) -> pd.DataFrame:
    """Read-only Stage 8 formula/identity audit, including its checkpoint hashes."""
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    if report.get("valid") is not True or report.get("primary_target") != PRIMARY_TARGET:
        raise ValueError("Successful Stage 8 primary-target validation required")
    for path in (labeled_path, targets_path):
        if report.get("output_artifact_sha256", {}).get(str(path)) != _sha256(path):
            raise ValueError(f"Stage 8 checkpoint hash mismatch: {path.name}")
    if report.get("source_feature_sha256") != _sha256(features_path):
        raise ValueError("Source feature hash differs from the Stage 8 checkpoint")
    labeled = _read_csv(labeled_path, artifact=True)
    audit = validate_targets(_read_csv(targets_path, artifact=True), _read_csv(features_path), labeled)
    if not audit["valid"]:
        raise ValueError(f"Read-only Stage 8 audit failed: {audit['errors']}")
    return labeled


def _check_existing_protocol(locked: dict[str, Any], protocol: dict[str, Any],
                             data: pd.DataFrame, config: SplitConfig) -> None:
    """Allow only the authorized Stage 9→9.1 upgrade; preserve test and metric locks."""
    message = "Existing final-test protocol is locked; refusing to change its source, configuration, dates or evaluation contract"
    contract_keys = set(evaluation_protocol(config, {}))
    if locked.get("protocol_version") == PROTOCOL_VERSION:
        keys = contract_keys | {"source_labeled_sha256", "pre_test_gap"}
        if any(locked.get(key) != protocol[key] for key in keys):
            raise ValueError(message)
        return
    if "protocol_version" in locked or locked.get("validation_test_calendar_overlap_policy") != LEGACY_OVERLAP_POLICY:
        raise ValueError(message)
    new_policy_keys = {
        "protocol_version", "registered_targets", "validation_label_availability_rule", "pre_test_gap_policy",
        "pre_test_gap_basis", "longest_registered_horizon", "cv_validation_policy", "missing_validation_label_policy",
        "pre_test_gap_final_training_policy", "limitations", "boundaries",
    }
    immutable_keys = (contract_keys - new_policy_keys) | {"source_labeled_sha256"}
    if any(locked.get(key) != protocol[key] for key in immutable_keys):
        raise ValueError(message)
    # Recognize the original full Stage 9 layout independently, including its CV.
    dates = parse_dates(data.date)
    development_count = int(data[PRIMARY_TARGET].notna().sum()) - config.final_test_size
    first = development_count - config.n_folds * config.validation_size
    expected = {key: protocol["boundaries"][key] for key in ("development", "final_test", "unlabeled_tail")}
    expected["cv"] = [{"fold": f"cv_{i + 1}", "start": _iso(dates.iloc[first + i * config.validation_size]),
                       "end": _iso(dates.iloc[first + (i + 1) * config.validation_size - 1])} for i in range(config.n_folds)]
    if locked.get("boundaries") != expected:
        raise ValueError(message)


def run_pipeline(
    labeled_path: Path = RESEARCH_DIR / "NVDA_labeled.csv",
    targets_path: Path = RESEARCH_DIR / "targets" / "NVDA_targets.csv",
    features_path: Path = RESEARCH_DIR / "NVDA_features.csv",
    target_validation_path: Path = RESEARCH_DIR / "targets" / "NVDA_target_validation.json",
    output_dir: Path = RESEARCH_DIR / "splits",
    config: SplitConfig = SplitConfig(),
) -> dict[str, Any]:
    """Write new manifests/protocol only; verify earlier files and persistent test lock."""
    labeled_path, targets_path, features_path, target_validation_path, output_dir = (
        Path(p).resolve() for p in (labeled_path, targets_path, features_path, target_validation_path, output_dir))
    inputs = (labeled_path, targets_path, features_path, target_validation_path)
    if any(output_dir == p or output_dir in p.parents for p in inputs):
        raise ValueError("Output directory must not contain source artifacts")
    table_names = ("split_manifest", "cv_folds", "final_training_pool", "split_summary")
    outputs = {name: output_dir / f"NVDA_{name}.csv" for name in table_names}
    validation_path = output_dir / "NVDA_split_validation.json"
    protocol_path = output_dir / "NVDA_evaluation_protocol.json"
    output_paths = {*outputs.values(), validation_path, protocol_path}
    if any(p.is_symlink() for p in output_paths):
        raise ValueError("Output symlinks cannot overwrite source artifacts")
    protected = set(inputs)
    for root in {DATA_DIR.resolve(), labeled_path.parent}:
        if root.exists():
            protected.update(p.resolve() for p in root.rglob("*") if p.is_file() and p.resolve() not in output_paths)
    before = {str(p): _sha256(p) for p in sorted(protected)}
    labeled = _source_audit(labeled_path, targets_path, features_path, target_validation_path)
    original = labeled.copy(deep=True)
    tables, protocol = build_splits(labeled, config)
    protocol["source_labeled_sha256"] = before[str(labeled_path)]
    protocol["source_labeled_file"] = str(labeled_path)
    if protocol_path.exists():
        locked = json.loads(protocol_path.read_text(encoding="utf-8"))
        _check_existing_protocol(locked, protocol, labeled, config)
    report = validate_splits(labeled, tables, protocol, config)
    pd.testing.assert_frame_equal(labeled, original, check_exact=True)
    if not report["valid"]:
        raise ValueError(f"Split validation failed: {report['errors']}")
    csv_text = {name: table.to_csv(index=False) for name, table in tables.items()}
    reloaded = {name: pd.read_csv(StringIO(value), keep_default_na=False) for name, value in csv_text.items()}
    # Empty metadata dates/reasons round-trip to the same missing/empty semantics.
    reloaded["split_manifest"]["target_entry_date"] = reloaded["split_manifest"].target_entry_date.replace("", None)
    reloaded["split_manifest"]["target_exit_date"] = reloaded["split_manifest"].target_exit_date.replace("", None)
    roundtrip = validate_splits(labeled, reloaded, protocol, config)
    if not roundtrip["valid"]:
        raise ValueError(f"Split CSV round-trip validation failed: {roundtrip['errors']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in outputs.items():
        path.write_text(csv_text[name], encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after = {path: _sha256(Path(path)) for path in before}
    report["checks"].update({"source_dataframe_unchanged": labeled.equals(original),
                             "source_artifact_hashes_unchanged": before == after, "csv_roundtrip_validated": roundtrip["valid"]})
    report.update({"source_labeled_file": str(labeled_path), "source_labeled_sha256": before[str(labeled_path)],
                   "source_column_count": len(labeled.columns), "source_sha256_before": before, "source_sha256_after": after,
                   "source_artifacts_unchanged": before == after, "boundaries": protocol["boundaries"],
                   "primary_metric": PRIMARY_METRIC, "secondary_metrics": list(SECONDARY_METRICS),
                   "stage10_readiness": protocol["stage10_readiness"], "readiness_reason": protocol["readiness_reason"],
                   "config": asdict(config), "limitations": LIMITATIONS,
                   "output_paths": {**{k: str(p) for k, p in outputs.items()}, "validation": str(validation_path), "protocol": str(protocol_path)},
                   "output_artifact_sha256": {str(p): _sha256(p) for p in [*outputs.values(), protocol_path]},
                   "reproducibility": {"randomness": "none", "run_timestamp": "omitted", "pandas_version": pd.__version__, "numpy_version": np.__version__}})
    report["errors"].extend(k for k, passed in report["checks"].items() if not passed and k not in report["errors"])
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    validation_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise ValueError(f"Split artifact validation failed: {report['errors']}")
    return report


def terminal_summary(report: dict[str, Any]) -> str:
    """Show actual counts for every fold/horizon; no scores or target statistics."""
    cv = pd.read_csv(report["output_paths"]["cv_folds"])
    final = pd.read_csv(report["output_paths"]["final_training_pool"])
    lines = [f"Ticker: {report['ticker']}", f"Primary target: {PRIMARY_TARGET}", f"Total rows: {report['total_rows']}",
             f"Primary labelable / unlabeled tail: {report['labelable_primary_rows']} / {report['unlabeled_tail_rows']}",
             f"Development / locked final test: {report['development_rows']} / {report['final_test_rows']}",
             f"CV folds / nominal validation rows per fold: {report['cv_fold_count']} / {report['validation_size_per_fold']}"]
    lines.extend([f"Final test start: {report['boundaries']['final_test']['start']}",
                  f"Pre-test gap: rows={report['pre_test_gap_rows']}, {report['pre_test_gap_start_date']} to {report['pre_test_gap_end_date']}",
                  f"Gap basis: all registered horizons; longest={max(HORIZONS)}d"])
    for fold in cv.fold.drop_duplicates():
        block = cv.loc[cv.fold.eq(fold)]
        first = block.iloc[0]
        lines.append(f"{fold}: nominal train={first.train_count_nominal}, validation={first.validation_start_date} to {first.validation_end_date}")
        lines.extend(f"  {r.target_name}: effective train={r.train_count_effective}, purged={r.purged_count}, usable validation={r.validation_count_effective}"
                     for r in block.itertuples())
    lines.append(f"Final training pool: nominal development={report['development_rows']}")
    lines.extend(f"  {r.target_name}: effective train={r.final_train_count_effective}, purged={r.final_purged_count}, usable test={r.final_test_count_effective}"
                 for r in final.itertuples())
    lines.extend(f"  {r.target_name}: gap rows returning to final training={r.pre_test_gap_final_train_usable_count}"
                 for r in final.itertuples())
    lines.extend(f"Max validation target exit {target}: {day}"
                 for target, day in report["maximum_validation_target_exit_dates"].items())
    lines.extend(f"Validation labels crossing final test {h}d: {report[f'validation_label_crosses_final_test_{h}d']}"
                 for h in HORIZONS)
    lines.extend([f"Embargo sessions: {report['embargo_sessions']}", f"Primary metric: {PRIMARY_METRIC}",
                  f"Secondary metrics: {', '.join(SECONDARY_METRICS)}", "Final test locked: True; performance evaluation: not run",
                  f"Validation labels reaching final-test calendar: {report['validation_labels_reaching_final_test_calendar']}",
                  f"Source artifacts unchanged: {report['source_artifacts_unchanged']}",
                  f"Output directory: {Path(report['output_paths']['validation']).parent}", f"Validation: {report['validation_status']}"])
    lines.append(f"Readiness: {report['stage10_readiness']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-path", type=Path, default=RESEARCH_DIR / "NVDA_labeled.csv")
    parser.add_argument("--targets-path", type=Path, default=RESEARCH_DIR / "targets" / "NVDA_targets.csv")
    parser.add_argument("--features-path", type=Path, default=RESEARCH_DIR / "NVDA_features.csv")
    parser.add_argument("--target-validation-path", type=Path, default=RESEARCH_DIR / "targets" / "NVDA_target_validation.json")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "splits")
    args = parser.parse_args()
    print(terminal_summary(run_pipeline(args.labeled_path, args.targets_path, args.features_path, args.target_validation_path, args.output_dir)))


if __name__ == "__main__":
    main()
