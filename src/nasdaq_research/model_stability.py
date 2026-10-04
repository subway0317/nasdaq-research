"""Stage 10.1: descriptive stability diagnostics of the frozen development CV."""

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nasdaq_research import modeling as baseline
from nasdaq_research.dates import parse_dates
from nasdaq_research.research_features import STATE_METADATA_COLUMNS

RESEARCH_DIR = baseline.RESEARCH_DIR
ATOL = 1e-12
FOLDS = ("cv_1", "cv_2", "cv_3")
SPECS = ("market", "all")
STATE_GROUPS = ("q", "fy", "bs")
STATE_ID_FIELDS = ("accession", "filing_date", "period_start", "period_end", "data_period_type")
TABLE_NAMES = ("row_errors", "prediction_summary", "feature_shift", "zscores", "contributions",
               "coefficient_summary", "conditioning", "influence", "state_context")
FIGURE_NAMES = ("actual_vs_prediction_by_fold", "prediction_range_by_fold", "cv3_largest_absolute_errors",
                "max_validation_abs_z_by_feature", "coefficient_drift")
VIOLATIONS = (
    "stage10_prediction_consistency_violations", "stage10_metric_consistency_violations",
    "prediction_reconstruction_violations", "feature_range_validation_violations", "zscore_validation_violations",
    "coefficient_validation_violations", "conditioning_validation_violations", "influence_validation_violations",
    "state_context_validation_violations", "state_metadata_columns_in_model_X", "secondary_target_dependency_violations",
    "gap_mutation_violations", "final_test_feature_mutation_violations", "final_test_target_mutation_violations",
    "final_test_metadata_mutation_violations", "secondary_target_mutation_violations", "tail_mutation_violations",
    "validation_X_training_fit_violations", "validation_Y_prediction_fit_violations", "upstream_artifact_mutations",
)


@dataclass
class Diagnostics:
    """In-memory artifacts and original Stage 10 fits; nothing is fitted on the test."""

    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    fits: dict[tuple[str, str], baseline.FittedLinear]
    fit_records: list[dict[str, Any]]


def protocol() -> dict[str, Any]:
    """Definitions describe diagnostics only and never introduce model treatments."""
    return {
        "stage": "10.1", "ticker": "NVDA", "primary_target": baseline.PRIMARY_TARGET,
        "split_protocol_version": "9.1", "models": list(baseline.MODELS),
        "residual_definition": "actual - prediction", "descriptive_std_ddof": 1,
        "range_definition": "strict raw non-null validation < training min or > training max; equality is inside",
        "range_fraction_denominator": "non-null validation values; undefined training range yields NaN",
        "zscore_definition": "(validation filled with fixed training median - training scaler mean) / training scaler scale",
        "zscore_percentile_method": "linear", "z_thresholds_are_diagnostic_only": [3, 5],
        "prediction_magnitude_thresholds_are_diagnostic_only": [0.10, 0.20, 0.30],
        "coefficient_std_ddof": 1,
        "coefficient_sign_policy": "sign(0)=0; missing folds skipped; flips require opposite nonzero signs in consecutive available folds",
        "svd_rank_tolerance": "max(matrix.shape) * float64 eps * largest singular value",
        "condition_number_policy": "infinity if numerically rank deficient; JSON null plus condition_number_is_infinite flag",
        "influence_design": "post-scaling training X with intercept; SVD Moore-Penrose projection",
        "influence_studentization": "internal: residual / sqrt(MSE * (1 - leverage))",
        "influence_MSE": "training residual SSE / (n - intercept-design numerical rank)",
        "cooks_distance": "residual^2 / (rank * MSE) * leverage / (1-leverage)^2",
        "influence_undefined_policy": "NaN with reason for zero residual variance, nonpositive residual df or unit leverage",
        "state_identity_fields": list(STATE_ID_FIELDS),
        "state_comparison": "current validation row versus its immediately previous observed session; no future state",
        "state_age_semantics": "existing calendar-day metadata; sample-effective age can be truncated at source window start",
        "comparison_tolerance": {"rtol": 0, "atol": ATOL},
        "new_model_ranking": False, "ridge_alpha": 1.0, "new_preprocessing": False,
        "hyperparameter_tuning": False, "feature_selection": False, "final_training_pool_used": False,
        "final_test_locked": True, "final_test_predictions_generated": False, "final_test_metrics_computed": False,
        "figures_feature_ordering": "variability/magnitude ordering is visualization only; fitted features are unchanged",
    }


def _stats(values: pd.Series | np.ndarray, prefix: str) -> dict[str, float]:
    observed = pd.Series(values, dtype=float).dropna()
    result = {f"{prefix}_{name}": float(getattr(observed, name)())
              for name in ("mean", "std", "median", "min", "max")}
    if len(observed) > 1 and observed.eq(observed.iloc[0]).all():
        result[f"{prefix}_std"] = 0.0
    return result


def feature_range(training: pd.Series, validation: pd.Series) -> dict[str, Any]:
    """Raw non-null strict bounds; missing validation values never count as outside."""
    train, valid = training.dropna().astype(float), validation.dropna().astype(float)
    defined = not train.empty
    below = int((valid < train.min()).sum()) if defined else np.nan
    above = int((valid > train.max()).sum()) if defined else np.nan
    outside = below + above
    return {"validation_below_training_min_count": below, "validation_above_training_max_count": above,
            "validation_outside_training_range_count": outside,
            "validation_outside_training_range_fraction": outside / len(valid) if defined and len(valid) else np.nan}


def zscore_summary(values: np.ndarray) -> dict[str, Any]:
    """Absolute transformed validation values, including training-median fills."""
    values = np.abs(np.asarray(values, dtype=float)).ravel()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Complete finite validation z-scores required")
    return {"zscore_value_count": len(values), "max_abs_validation_z": float(values.max()),
            "median_abs_validation_z": float(np.median(values)),
            "p95_abs_validation_z": float(np.percentile(values, 95, method="linear")),
            "count_abs_z_gt_3": int((values > 3).sum()), "count_abs_z_gt_5": int((values > 5).sum()),
            "fraction_abs_z_gt_3": float((values > 3).mean()), "fraction_abs_z_gt_5": float((values > 5).mean())}


def validation_zscores(fitted: baseline.FittedLinear, validation: pd.DataFrame,
                       fold: str, spec: str) -> pd.DataFrame:
    """Expose the exact Stage 10 transform; do not fit new preprocessing."""
    pp = fitted.preprocessing
    z = baseline.transform_features(pp, validation)
    rows = []
    for j, name in enumerate(pp.features):
        raw = validation[name].astype(float)
        filled = raw.fillna(pp.medians[name])
        for i, day in enumerate(validation.index):
            rows.append({"date": str(day), "fold": fold, "feature_spec": spec, "feature_name": name,
                         "raw_validation_value": raw.iloc[i], "was_validation_missing": bool(pd.isna(raw.iloc[i])),
                         "imputed_validation_value": filled.iloc[i], "training_median": pp.medians[name],
                         "training_scaler_mean": pp.scaler.mean_[j], "training_scaler_scale": pp.scaler.scale_[j],
                         "validation_zscore": z[i, j], "abs_validation_zscore": abs(z[i, j])})
    return pd.DataFrame(rows)


def prediction_contributions(zscores: pd.DataFrame, coefficients: pd.Series, intercept: float,
                             predictions: pd.Series, model: str) -> pd.DataFrame:
    """Separate positive, negative and absolute driver ranks without selecting features."""
    result = zscores[["date", "fold", "feature_spec", "feature_name", "validation_zscore"]].copy()
    result = result.rename(columns={"validation_zscore": "z_value"})
    result.insert(2, "model_name", model)
    result["coefficient"] = result.feature_name.map(coefficients)
    if result.coefficient.isna().any():
        raise ValueError("Used features require saved coefficients")
    result["feature_contribution"] = result.coefficient * result.z_value
    result["intercept"] = intercept
    result["prediction"] = result.date.map(predictions)
    if result.prediction.isna().any():
        raise ValueError("Contribution dates must have predictions")
    grouping = ["model_name", "fold", "date"]
    result["contribution_rank_by_absolute_value"] = result.feature_contribution.abs().groupby(
        [result[name] for name in grouping]).rank(method="min", ascending=False)
    result["positive_contribution_rank"] = np.nan
    result["negative_contribution_rank"] = np.nan
    for sign, column, ascending in ((1, "positive_contribution_rank", False), (-1, "negative_contribution_rank", True)):
        subset = result.loc[np.sign(result.feature_contribution).eq(sign)]
        result.loc[subset.index, column] = subset.groupby(grouping).feature_contribution.rank(method="min", ascending=ascending)
    return result


def coefficient_stability(coefficients: pd.DataFrame, specifications: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Missing-fold coefficients remain NaN, with explicit zero-sign semantics."""
    rows = []
    for model in baseline.LINEAR_MODELS:
        spec = model.split("_", 1)[1]
        for name in specifications[spec]:
            available = coefficients.loc[coefficients.model_name.eq(model) & coefficients.feature_name.eq(name)].set_index("fold")
            values = np.array([available.loc[fold, "coefficient"] if fold in available.index else np.nan for fold in FOLDS])
            present = values[np.isfinite(values)]
            signs = np.sign(present)
            nonzero = np.abs(present[present != 0])
            row = {"model_name": model, "feature_spec": spec, "feature_name": name, "folds_present": len(present),
                   "coefficient_mean_available": float(present.mean()) if len(present) else np.nan,
                   "coefficient_std_available": float(present.std(ddof=1)) if len(present) > 1 else np.nan,
                   "max_abs_coefficient": float(np.abs(present).max()) if len(present) else np.nan,
                   "min_abs_nonzero_coefficient": float(nonzero.min()) if len(nonzero) else np.nan,
                   "max_abs_consecutive_available_change": float(np.abs(np.diff(present)).max()) if len(present) > 1 else np.nan,
                   "sign_flip_count_across_consecutive_available_folds": int((signs[:-1] * signs[1:] < 0).sum()),
                   "sign_consistent_across_available_folds": bool(len(set(signs)) <= 1) if len(signs) else np.nan}
            row.update({f"coefficient_{fold}": value for fold, value in zip(FOLDS, values)})
            row.update({f"sign_{fold}": np.sign(value) if np.isfinite(value) else np.nan for fold, value in zip(FOLDS, values)})
            rows.append(row)
    return pd.DataFrame(rows)


def conditioning(matrix: np.ndarray) -> dict[str, Any]:
    """SVD diagnostics of the post-imputation/scaling training X, without intercept."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError("Nonempty finite training matrix required")
    n, p = matrix.shape
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    rank = int((singular > tolerance).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.corrcoef(matrix, rowvar=False) if p > 1 else np.ones((1, 1))
    pairs = np.abs(correlations[np.triu_indices(p, k=1)])
    finite = pairs[np.isfinite(pairs)]
    deficient = rank < p
    return {"training_rows": n, "used_features": p, "n_to_p_ratio": n / p,
            "matrix_rank": rank, "full_column_rank": not deficient, "rank_deficient": deficient,
            "largest_singular_value": float(singular[0]), "smallest_singular_value": float(singular[-1]),
            "rank_tolerance": float(tolerance), "condition_number": np.inf if deficient else float(singular[0] / singular[-1]),
            "condition_number_is_infinite": deficient,
            "max_abs_pairwise_correlation": float(finite.max()) if len(finite) else np.nan,
            "high_corr_pair_count_abs_ge_0_95": int((finite >= .95).sum()),
            "undefined_pairwise_correlation_count": int((~np.isfinite(pairs)).sum())}


def ols_influence(matrix: np.ndarray, actual: np.ndarray, fitted: np.ndarray,
                  dates: list[str], fold: str, spec: str) -> pd.DataFrame:
    """Hat diagonal via SVD Moore-Penrose projection; effective-rank df includes intercept.

    Internally studentized residuals and Cook's distance use rank in place of
    parameter count for rank-deficient designs. No observation is removed/refit.
    """
    matrix, actual, fitted = (np.asarray(value, dtype=float) for value in (matrix, actual, fitted))
    if matrix.ndim != 2 or actual.ndim != 1 or actual.shape != fitted.shape or len(actual) != len(matrix) or len(dates) != len(actual):
        raise ValueError("Aligned training matrix, targets, fitted values and dates required")
    if not all(np.isfinite(value).all() for value in (matrix, actual, fitted)):
        raise ValueError("Influence inputs must be finite")
    design = np.column_stack([np.ones(len(matrix)), matrix])
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular[0]
    retained = singular > tolerance
    rank = int(retained.sum())
    leverage = (left[:, retained] ** 2).sum(axis=1)
    residual = actual - fitted
    df = len(matrix) - rank
    negligible = np.linalg.norm(residual) <= np.finfo(float).eps * max(design.shape) * max(1., np.linalg.norm(actual), np.linalg.norm(fitted))
    mse = float(residual @ residual / df) if df > 0 and not negligible else np.nan
    reasons = []
    studentized, cooks = np.full(len(matrix), np.nan), np.full(len(matrix), np.nan)
    for i, h in enumerate(leverage):
        reason = ("nonpositive_residual_df" if df <= 0 else "zero_residual_variance" if negligible
                  else "unit_leverage" if 1 - h <= 16 * np.finfo(float).eps * max(design.shape) else "defined")
        reasons.append(reason)
        if reason == "defined":
            studentized[i] = residual[i] / np.sqrt(mse * (1 - h))
            cooks[i] = residual[i] ** 2 / (rank * mse) * h / (1 - h) ** 2
    return pd.DataFrame({"date": dates, "fold": fold, "model_name": f"ols_{spec}", "feature_spec": spec,
                         "training_actual": actual, "training_fitted": fitted, "training_residual": residual,
                         "leverage": leverage, "studentized_residual": studentized, "cooks_distance": cooks,
                         "undefined_reason": reasons, "intercept_design_rank": rank, "residual_df": df,
                         "residual_MSE": mse})


def state_context(data: pd.DataFrame, authority: dict[str, tuple[list[str], list[str]]],
                  errors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare only each validation row and its immediate previous observed session.

    The composite identities describe selected source states, not model features.
    Existing filing/effective ages retain Stage 6.1 calendar-day/truncation semantics.
    """
    keyed = data.copy()
    keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
    dates = keyed.index.tolist()
    positions = {day: i for i, day in enumerate(dates)}
    rows, previous_dates = [], []
    availability = {group: all(f"{group}_{field}" in data for field in STATE_ID_FIELDS) for group in STATE_GROUPS}
    for fold, (_, validation_dates) in authority.items():
        for day in validation_dates:
            previous = dates[positions[day] - 1] if positions[day] > 0 else None
            if previous is not None:
                previous_dates.append(previous)
            row = {"date": day, "fold": fold, "previous_observed_date": previous,
                   "actual_return": keyed.loc[day, baseline.PRIMARY_TARGET]}
            for group in STATE_GROUPS:
                identity_columns = [f"{group}_{field}" for field in STATE_ID_FIELDS]
                current_values = keyed.loc[day, identity_columns].tolist() if availability[group] else []
                previous_values = keyed.loc[previous, identity_columns].tolist() if availability[group] and previous is not None else []
                current_known = bool(current_values) and all(pd.notna(value) and str(value) != "" for value in current_values)
                previous_known = bool(previous_values) and all(pd.notna(value) and str(value) != "" for value in previous_values)
                row[f"{group}_state_identifier"] = json.dumps(list(map(str, current_values)), separators=(",", ":")) if current_known else np.nan
                row[f"{group}_state_changed_since_previous_observed_session"] = (
                    list(map(str, current_values)) != list(map(str, previous_values)) if current_known and previous_known else np.nan)
                for field in ("effective_date", "filing_date", "days_since_filing", "days_since_effective_date",
                              "effective_date_is_sample_truncated", "effective_date_basis"):
                    column = f"{group}_{field}"
                    if column in keyed:
                        value = keyed.loc[day, column]
                        if pd.notna(value) and field.startswith("days_since_"):
                            value = float(value)
                        if pd.notna(value) and field == "effective_date_is_sample_truncated" and isinstance(value, str):
                            if value.lower() not in ("true", "false"):
                                raise ValueError("Invalid sample-truncation flag in authorized state context")
                            value = value.lower() == "true"
                        row[column] = value
            selected = errors.loc[errors.date.eq(day) & errors.fold.eq(fold)].set_index("model_name")
            for model in baseline.LINEAR_MODELS:
                row[f"{model}_prediction"] = selected.loc[model, "prediction"]
                row[f"{model}_absolute_error"] = selected.loc[model, "absolute_error"]
            rows.append(row)
    return pd.DataFrame(rows), {"state_identity_available": availability,
                               "state_context_status": "available" if all(availability.values()) else "not reliably computable for all groups from available metadata",
                               "state_previous_observed_dates_used": sorted(set(previous_dates))}


def _coefficient_norms(coefficients: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    norms, shrinkage = [], []
    for fold in FOLDS:
        for model in baseline.LINEAR_MODELS:
            values = coefficients.loc[coefficients.fold.eq(fold) & coefficients.model_name.eq(model), "coefficient"].to_numpy()
            norms.append({"fold": fold, "model_name": model, "feature_spec": model.split("_", 1)[1],
                          "L1_coefficient_norm": float(np.abs(values).sum()), "L2_coefficient_norm": float(np.linalg.norm(values)),
                          "max_abs_coefficient": float(np.abs(values).max()), "median_abs_coefficient": float(np.median(np.abs(values)))})
        for spec in SPECS:
            pair = {row["model_name"]: row for row in norms if row["fold"] == fold and row["feature_spec"] == spec}
            ols, ridge = pair[f"ols_{spec}"], pair[f"ridge_{spec}"]
            shrinkage.append({"fold": fold, "feature_spec": spec,
                              "ridge_L2_to_ols_L2_ratio": ridge["L2_coefficient_norm"] / ols["L2_coefficient_norm"] if ols["L2_coefficient_norm"] else np.nan,
                              "ridge_max_abs_to_ols_max_abs_ratio": ridge["max_abs_coefficient"] / ols["max_abs_coefficient"] if ols["max_abs_coefficient"] else np.nan})
    return norms, shrinkage


def _maximum(rows: pd.DataFrame, column: str, absolute: bool = False) -> tuple[float, Any]:
    values = rows[column].abs() if absolute else rows[column]
    if not values.notna().any():
        return np.nan, None
    position = values.idxmax()
    return float(values.loc[position]), rows.loc[position, "date"]


def _summarize(tables: dict[str, pd.DataFrame], state_info: dict[str, Any],
               baseline_validation: dict[str, Any]) -> dict[str, Any]:
    errors, shift, zscores = (tables[name] for name in ("row_errors", "feature_shift", "zscores"))
    extremes, z_summaries, range_summaries, influence_summaries, top_rows = [], [], [], [], []
    for fold in FOLDS:
        for model in baseline.MODELS:
            rows = errors.loc[errors.fold.eq(fold) & errors.model_name.eq(model)]
            ordered = rows.sort_values(["absolute_error", "date"], ascending=[False, True], kind="stable")
            total = rows.absolute_error.sum()
            extremes.append({"fold": fold, "model_name": model, "largest_absolute_error": float(ordered.absolute_error.iloc[0]),
                             "largest_error_date": ordered.date.iloc[0], "absolute_error_mean": float(rows.absolute_error.mean()),
                             "absolute_error_median": float(rows.absolute_error.median()), "absolute_error_min": float(rows.absolute_error.min()),
                             "top_3_share_of_total_absolute_error": float(ordered.absolute_error.head(3).sum() / total) if total else np.nan,
                             "top_5_share_of_total_absolute_error": float(ordered.absolute_error.head(5).sum() / total) if total else np.nan,
                             "count_absolute_error_gt_0_10": int(rows.absolute_error.gt(.10).sum()),
                             "count_absolute_error_gt_0_20": int(rows.absolute_error.gt(.20).sum()),
                             "top_5_error_rows": ordered.head(5)[["date", "actual_forward_return_5d", "prediction", "absolute_error"]].to_dict("records")})
        for spec in SPECS:
            selected = zscores.loc[zscores.fold.eq(fold) & zscores.feature_spec.eq(spec)]
            z_summaries.append({"fold": fold, "feature_spec": spec, **zscore_summary(selected.validation_zscore.to_numpy())})
            used = shift.loc[shift.fold.eq(fold) & shift.feature_spec.eq(spec) & shift.used_in_stage10_fold.eq(True)]
            count = int(used.validation_outside_training_range_count.sum())
            denominator = int(used.validation_non_null_count.sum())
            range_summaries.append({"fold": fold, "feature_spec": spec, "used_features": len(used),
                                    "validation_non_null_values": denominator, "out_of_training_range_count": count,
                                    "out_of_training_range_fraction": count / denominator if denominator else np.nan})
            influence = tables["influence"].loc[tables["influence"].fold.eq(fold) & tables["influence"].feature_spec.eq(spec)]
            record = {"fold": fold, "feature_spec": spec, "undefined_metric_count": int(influence.undefined_reason.ne("defined").sum())}
            for column, absolute in (("leverage", False), ("studentized_residual", True), ("cooks_distance", False)):
                value, day = _maximum(influence, column, absolute)
                label = f"max_{'abs_' if absolute else ''}{column}"
                record[label], record[f"date_of_{label}"] = value, day
            influence_summaries.append(record)
    for model in baseline.LINEAR_MODELS:
        spec = model.split("_", 1)[1]
        rows = errors.loc[errors.fold.eq("cv_3") & errors.model_name.eq(model)].sort_values(
            ["absolute_error", "date"], ascending=[False, True], kind="stable").head(5)
        for error in rows.to_dict("records"):
            day = error["date"]
            zs = zscores.loc[zscores.fold.eq("cv_3") & zscores.feature_spec.eq(spec) & zscores.date.eq(day)]
            ranges = shift.loc[shift.fold.eq("cv_3") & shift.feature_spec.eq(spec)].set_index("feature_name")
            raw = zs.set_index("feature_name").raw_validation_value
            outside = (raw.lt(ranges.loc[raw.index, "training_min_raw"]) | raw.gt(ranges.loc[raw.index, "training_max_raw"])).sum()
            contributions = tables["contributions"].loc[tables["contributions"].model_name.eq(model) & tables["contributions"].date.eq(day)]
            error.update({"row_max_abs_z": float(zs.abs_validation_zscore.max()), "used_features_outside_training_range_count": int(outside),
                          "intercept": float(contributions.intercept.iloc[0])})
            for kind, rank in (("positive", "positive_contribution_rank"), ("negative", "negative_contribution_rank"),
                               ("absolute", "contribution_rank_by_absolute_value")):
                drivers = contributions.loc[contributions[rank].notna()].sort_values([rank, "feature_name"], kind="stable").head(3)
                error[f"top_3_{kind}_contributions"] = drivers[["feature_name", "coefficient", "z_value", "feature_contribution"]].to_dict("records")
            state = tables["state_context"].loc[tables["state_context"].date.eq(day)].iloc[0]
            for group in STATE_GROUPS:
                for field in ("state_changed_since_previous_observed_session", "days_since_effective_date", "days_since_filing"):
                    column = f"{group}_{field}"
                    if column in state:
                        error[column] = state[column]
            top_rows.append(error)
    norms, shrinkage = _coefficient_norms(tables["coefficients_internal"])
    state_changes = {group: tables["state_context"].loc[
        tables["state_context"][f"{group}_state_changed_since_previous_observed_session"].eq(True), ["date", "fold"]].to_dict("records") for group in STATE_GROUPS}
    return {"ticker": "NVDA", "stage": "10.1", "primary_target": baseline.PRIMARY_TARGET,
            "diagnostic_protocol": protocol(), "stage10_best_cv_candidate_reference": baseline_validation["best_stage10_cv_candidate"],
            "prediction_distributions": tables["prediction_summary"].to_dict("records"), "extreme_errors": extremes,
            "zscore_summary": z_summaries, "out_of_training_range_summary": range_summaries,
            "coefficient_norms": norms, "ridge_to_ols_coefficient_ratios": shrinkage,
            "coefficient_sign_flips": tables["coefficient_summary"].loc[tables["coefficient_summary"].sign_flip_count_across_consecutive_available_folds.gt(0)].to_dict("records"),
            "conditioning": tables["conditioning"].to_dict("records"), "influence_maxima": influence_summaries,
            "cv_3_top_error_rows_and_contributors": top_rows, "state_transitions_on_validation_dates": state_changes,
            **state_info,
            "mechanism_assessment_policy": "Combine errors, raw shift, z-scores, contributions, coefficient drift, conditioning and shrinkage descriptively; no automatic dominance or causality claim.",
            "limitations": baseline.LIMITATIONS + ["Linear diagnostics and state-transition associations are not causal evidence; coefficient drift and regime changes may be mixed."]}


def build_diagnostics(checkpoint: tuple[Any, ...], reference: dict[str, pd.DataFrame],
                      stage10_protocol: dict[str, Any], stage10_validation: dict[str, Any]) -> Diagnostics:
    """Reconstruct only frozen CV fits; mutated in-memory inputs remain auditable.

    The CLI requires consistency validation before publishing. This pure builder
    also permits adversarial tests to inspect validation mutations without writes.
    """
    data, manifest, split_protocol, folds, inventory, quality = checkpoint
    specifications = baseline.feature_specs(data.columns, inventory, quality)
    if any(stage10_protocol.get(key) != value for key, value in baseline.baseline_protocol(specifications).items()):
        raise ValueError("Frozen Stage 10 protocol required; no target/specification/preprocessing/hyperparameter changes")
    if stage10_validation.get("valid") is not True or stage10_validation.get("final_test_locked") is not True:
        raise ValueError("Successful locked Stage 10 validation checkpoint required")
    authority = baseline.cv_authority(data, manifest, split_protocol, folds)
    keyed = data.copy()
    keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
    fit_records, fits, errors, predictions, shifts, z_blocks, contribution_blocks, conditions, influence_blocks, coefficients = [], {}, [], [], [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for fold, (train_dates, valid_dates) in authority.items():
            training, validation = keyed.loc[train_dates], keyed.loc[valid_dates]
            train_y = baseline._target(training[baseline.PRIMARY_TARGET])
            valid_y = baseline._target(validation[baseline.PRIMARY_TARGET])
            fold_predictions = {"zero_return": np.zeros(len(valid_dates)), "historical_mean": np.full(len(valid_dates), train_y.mean())}
            for model in baseline.LINEAR_MODELS:
                spec = model.split("_", 1)[1]
                candidates = specifications[spec]
                fitted = baseline.fit_linear(model, training.loc[:, list(candidates)], training[baseline.PRIMARY_TARGET], candidates)
                fits[fold, model] = fitted
                fold_predictions[model] = baseline.predict_linear(fitted, validation.loc[:, list(candidates)])
                fit_records.append({"fold": fold, "model_name": model, "training_dates": list(fitted.train_dates),
                                    "validation_dates": valid_dates, "preprocessing_fit_dates": list(fitted.preprocessing.fit_dates),
                                    "scaler_sample_count": int(fitted.preprocessing.scaler.n_samples_seen_)})
                for name, value in zip(fitted.preprocessing.features, fitted.estimator.coef_):
                    coefficients.append({"model_name": model, "feature_spec": spec, "fold": fold, "feature_name": name,
                                         "coefficient": float(value), "intercept": float(fitted.estimator.intercept_)})
            for spec in SPECS:
                fitted = fits[fold, f"ols_{spec}"]
                pp = fitted.preprocessing
                zrows = validation_zscores(fitted, validation, fold, spec)
                z_blocks.append(zrows)
                for usage in pp.usage.itertuples():
                    name = usage.feature_name
                    train, valid = training[name].astype(float), validation[name].astype(float)
                    record = {"fold": fold, "feature_spec": spec, "feature_name": name,
                              "used_in_stage10_fold": usage.used, "stage10_drop_reason": usage.drop_reason,
                              "training_non_null_count": int(train.notna().sum()), "training_missing_count": int(train.isna().sum()),
                              "training_coverage": float(train.notna().mean()), "validation_non_null_count": int(valid.notna().sum()),
                              "validation_missing_count": int(valid.isna().sum()), "validation_coverage": float(valid.notna().mean()),
                              **{key + "_raw": value for key, value in _stats(train, "training").items()},
                              **{key + "_raw": value for key, value in _stats(valid, "validation").items()}, **feature_range(train, valid)}
                    zs = zrows.loc[zrows.feature_name.eq(name), "validation_zscore"].to_numpy()
                    record.update(zscore_summary(zs) if len(zs) else {key: np.nan for key in zscore_summary(np.array([0.])).keys()})
                    shifts.append(record)
                matrix = baseline.transform_features(pp, training)
                conditions.append({"fold": fold, "feature_spec": spec, **conditioning(matrix)})
                influence_blocks.append(ols_influence(matrix, train_y, fitted.estimator.predict(matrix), train_dates, fold, spec))
                for family in ("ols", "ridge"):
                    model = f"{family}_{spec}"
                    current = fits[fold, model]
                    model_z = validation_zscores(current, validation, fold, spec)
                    contribution_blocks.append(prediction_contributions(
                        model_z, pd.Series(current.estimator.coef_, index=current.preprocessing.features),
                        float(current.estimator.intercept_), pd.Series(fold_predictions[model], index=valid_dates), model))
            for model in baseline.MODELS:
                values = fold_predictions[model]
                for i, day in enumerate(valid_dates):
                    residual = valid_y[i] - values[i]
                    errors.append({"date": day, "ticker": "NVDA", "fold": fold, "model_name": model,
                                   "feature_spec": "none" if model in baseline.MODELS[:2] else model.split("_", 1)[1],
                                   "actual_forward_return_5d": valid_y[i], "prediction": values[i], "residual": residual,
                                   "absolute_error": abs(residual), "squared_error": residual ** 2})
                actual_stats, prediction_stats = _stats(valid_y, "actual"), _stats(values, "prediction")
                predictions.append({"fold": fold, "model_name": model, "feature_spec": "none" if model in baseline.MODELS[:2] else model.split("_", 1)[1],
                                    "n": len(valid_dates), **actual_stats, **prediction_stats,
                                    "prediction_abs_max": float(np.abs(values).max()),
                                    "prediction_std_to_actual_std_ratio": prediction_stats["prediction_std"] / actual_stats["actual_std"] if actual_stats["actual_std"] else np.nan,
                                    "prediction_range": float(np.ptp(values)), "actual_range": float(np.ptp(valid_y)),
                                    **{f"count_abs_prediction_gt_0_{int(threshold * 100):02d}": int((np.abs(values) > threshold).sum()) for threshold in (.10, .20, .30)}})
    error_table = pd.DataFrame(errors)
    error_table["absolute_error_rank_within_fold_model"] = error_table.groupby(["fold", "model_name"]).absolute_error.rank(method="min", ascending=False)
    state, state_info = state_context(data, authority, error_table)
    tables = {"row_errors": error_table, "prediction_summary": pd.DataFrame(predictions), "feature_shift": pd.DataFrame(shifts),
              "zscores": pd.concat(z_blocks, ignore_index=True), "contributions": pd.concat(contribution_blocks, ignore_index=True),
              "coefficients_internal": pd.DataFrame(coefficients), "conditioning": pd.DataFrame(conditions),
              "influence": pd.concat(influence_blocks, ignore_index=True), "state_context": state}
    tables["coefficient_summary"] = coefficient_stability(tables["coefficients_internal"], specifications)
    summary = _summarize(tables, state_info, stage10_validation)
    del tables["coefficients_internal"]
    return Diagnostics({name: tables[name] for name in TABLE_NAMES}, summary, fits, fit_records)


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=0, atol=ATOL)
        return True
    except AssertionError:
        return False


def _same_json(actual: Any, expected: Any) -> bool:
    return baseline._json_clean(actual) == baseline._json_clean(expected)


def validate_diagnostics(checkpoint: tuple[Any, ...], reference: dict[str, pd.DataFrame],
                         stage10_protocol: dict[str, Any], stage10_validation: dict[str, Any],
                         diagnostics: Diagnostics) -> dict[str, Any]:
    """Audit diagnostic scopes, reference consistency, arithmetic and saved tables."""
    report = {name: 0 for name in VIOLATIONS}
    report.update({"ticker": "NVDA", "stage": "10.1", "primary_target": baseline.PRIMARY_TARGET,
                   "split_protocol_version": "9.1", "fold_count": 3, "oof_row_count": 75,
                   "final_test_locked": True, "final_test_predictions_generated": False,
                   "final_test_metrics_computed": False, "final_training_pool_used": False,
                   "mutation_checks_executed": False})
    checks: dict[str, bool] = {}
    try:
        data, manifest, split_protocol, folds, inventory, quality = checkpoint
        specifications = baseline.feature_specs(data.columns, inventory, quality)
        authority = baseline.cv_authority(data, manifest, split_protocol, folds)
        checks["stage10_protocol_unchanged"] = all(stage10_protocol.get(key) == value for key, value in baseline.baseline_protocol(specifications).items())
        checks["stage10_checkpoint_valid_locked"] = stage10_validation.get("valid") is True and all(
            stage10_validation.get(key) is False for key in ("final_test_predictions_generated", "final_test_metrics_computed", "final_training_pool_used"))
        keyed = data.copy()
        keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
        tables = diagnostics.tables
        checks["artifact_set_exact"] = list(tables) == list(TABLE_NAMES)
        errors, reference_oof = tables["row_errors"], reference["oof_predictions"]
        validation_dates = [day for _, valid in authority.values() for day in valid]
        checks["stage10_oof_date_identity"] = reference_oof.date.tolist() == validation_dates
        checks["row_errors_exact_keys"] = list(errors[["fold", "model_name", "date"]].itertuples(index=False, name=None)) == [
            (fold, model, day) for fold, (_, valid) in authority.items() for model in baseline.MODELS for day in valid]
        report["row_error_count"] = len(errors)
        report["duplicate_model_fold_date_rows"] = int(errors.duplicated(["model_name", "fold", "date"]).sum())
        report["missing_prediction_count"] = int(errors.prediction.isna().sum())
        report["non_finite_prediction_count"] = int((~np.isfinite(errors.prediction.to_numpy())).sum())
        checks["row_errors_finite_complete_unique"] = not (report["duplicate_model_fold_date_rows"] or report["non_finite_prediction_count"])
        checks["row_errors_ticker"] = errors.ticker.eq("NVDA").all()
        residual = errors.actual_forward_return_5d.to_numpy() - errors.prediction.to_numpy()
        checks["row_error_arithmetic"] = all(np.allclose(errors[column], expected, rtol=0, atol=ATOL) for column, expected in (
            ("residual", residual), ("absolute_error", np.abs(residual)), ("squared_error", residual ** 2)))
        expected_ranks = errors.groupby(["fold", "model_name"]).absolute_error.rank(method="min", ascending=False)
        checks["row_error_ranks"] = errors.absolute_error_rank_within_fold_model.equals(expected_ranks)
        referenced = reference_oof.set_index("date")
        checks["stage10_actuals_consistent"] = np.allclose(referenced.loc[validation_dates, f"actual_{baseline.PRIMARY_TARGET}"],
                                                           keyed.loc[validation_dates, baseline.PRIMARY_TARGET], rtol=0, atol=ATOL)
        prediction_rows, shift_rows, coefficient_rows, condition_rows, influence_blocks, expected_z = [], [], [], [], [], []
        used_dates = set(errors.date)
        checks["fit_records_exact"] = [(row["fold"], row["model_name"]) for row in diagnostics.fit_records] == [
            (fold, model) for fold in FOLDS for model in baseline.LINEAR_MODELS]
        records = {(row["fold"], row["model_name"]): row for row in diagnostics.fit_records}
        with threadpool_limits(limits=1):
            for fold, (train_dates, valid_dates) in authority.items():
                training, validation = keyed.loc[train_dates], keyed.loc[valid_dates]
                for model in baseline.MODELS:
                    current = errors.loc[errors.fold.eq(fold) & errors.model_name.eq(model)]
                    saved_predictions = referenced.loc[valid_dates, f"{model}_prediction"].to_numpy()
                    report["stage10_prediction_consistency_violations"] += int((~np.isclose(current.prediction, saved_predictions, rtol=0, atol=ATOL)).sum())
                    actual = current.actual_forward_return_5d.to_numpy()
                    checks[f"{fold}_{model}_actual_source_identity"] = np.array_equal(actual, keyed.loc[valid_dates, baseline.PRIMARY_TARGET].to_numpy())
                    metrics = baseline.regression_metrics(actual, current.prediction.to_numpy())
                    saved_metrics = reference["cv_metrics"].loc[reference["cv_metrics"].fold.eq(fold) & reference["cv_metrics"].model_name.eq(model)]
                    report["stage10_metric_consistency_violations"] += int(len(saved_metrics) != 1 or any(
                        not np.isclose(value, saved_metrics[name].iloc[0], rtol=0, atol=ATOL, equal_nan=True) for name, value in metrics.items()))
                    checks[f"{fold}_{model}_specification"] = current.feature_spec.eq("none" if model in baseline.MODELS[:2] else model.split("_", 1)[1]).all()
                    astats, pstats = _stats(actual, "actual"), _stats(current.prediction, "prediction")
                    prediction_rows.append({"fold": fold, "model_name": model, "feature_spec": current.feature_spec.iloc[0], "n": len(current),
                                            **astats, **pstats, "prediction_abs_max": float(current.prediction.abs().max()),
                                            "prediction_std_to_actual_std_ratio": pstats["prediction_std"] / astats["actual_std"] if astats["actual_std"] else np.nan,
                                            "prediction_range": float(current.prediction.max() - current.prediction.min()),
                                            "actual_range": float(actual.max() - actual.min()),
                                            **{f"count_abs_prediction_gt_0_{int(t * 100):02d}": int(current.prediction.abs().gt(t).sum()) for t in (.10, .20, .30)}})
                    if model in baseline.LINEAR_MODELS:
                        spec = model.split("_", 1)[1]
                        fitted = diagnostics.fits[fold, model]
                        pp = fitted.preprocessing
                        record = records[fold, model]
                        checks[f"{fold}_{model}_training_fit_scope"] = (record["training_dates"] == train_dates
                            and record["preprocessing_fit_dates"] == train_dates and record["validation_dates"] == valid_dates
                            and list(fitted.train_dates) == train_dates and list(pp.fit_dates) == train_dates
                            and record["scaler_sample_count"] == len(train_dates))
                        used_dates.update(record["training_dates"] + record["preprocessing_fit_dates"] + record["validation_dates"])
                        report["state_metadata_columns_in_model_X"] += len(set(pp.features) - set(baseline.FEATURE_GROUPS))
                        checks[f"{fold}_{model}_X_whitelist"] = set(pp.features) <= set(specifications[spec])
                        saved_usage = reference["feature_usage"].loc[reference["feature_usage"].fold.eq(fold) & reference["feature_usage"].model_name.eq(model)]
                        checks[f"{fold}_{model}_training_preprocessing_matches_stage10"] = _same_frame(pp.usage, saved_usage.loc[:, pp.usage.columns])
                        values = pd.DataFrame({"model_name": model, "feature_spec": spec, "fold": fold,
                                               "feature_name": pp.features, "coefficient": fitted.estimator.coef_, "intercept": fitted.estimator.intercept_})
                        saved_coefficients = reference["coefficients"].loc[reference["coefficients"].fold.eq(fold) & reference["coefficients"].model_name.eq(model)]
                        report["coefficient_validation_violations"] += int(not _same_frame(values, saved_coefficients))
                        coefficient_rows.extend(values.to_dict("records"))
                        contributions = tables["contributions"].loc[tables["contributions"].fold.eq(fold) & tables["contributions"].model_name.eq(model)]
                        z = validation_zscores(fitted, validation, fold, spec)
                        expected_contributions = prediction_contributions(z, pd.Series(fitted.estimator.coef_, index=pp.features),
                                                                          float(fitted.estimator.intercept_), pd.Series(current.prediction.to_numpy(), index=valid_dates), model)
                        checks[f"{fold}_{model}_contribution_arithmetic_and_ranks"] = _same_frame(contributions, expected_contributions)
                        for day in valid_dates:
                            selected = contributions.loc[contributions.date.eq(day)]
                            value = float(selected.feature_contribution.sum() + fitted.estimator.intercept_)
                            prediction = float(current.loc[current.date.eq(day), "prediction"].iloc[0])
                            report["prediction_reconstruction_violations"] += int(not np.isclose(value, prediction, rtol=0, atol=ATOL))
                for spec in SPECS:
                    fitted = diagnostics.fits[fold, f"ols_{spec}"]
                    pp = fitted.preprocessing
                    z = validation_zscores(fitted, validation, fold, spec)
                    expected_z.append(z)
                    for usage in pp.usage.itertuples():
                        name = usage.feature_name
                        train, valid = training[name].astype(float), validation[name].astype(float)
                        row = {"fold": fold, "feature_spec": spec, "feature_name": name,
                               "used_in_stage10_fold": usage.used, "stage10_drop_reason": usage.drop_reason,
                               "training_non_null_count": int(train.notna().sum()), "training_missing_count": int(train.isna().sum()),
                               "training_coverage": float(train.notna().mean()), "validation_non_null_count": int(valid.notna().sum()),
                               "validation_missing_count": int(valid.isna().sum()), "validation_coverage": float(valid.notna().mean()),
                               **{key + "_raw": value for key, value in _stats(train, "training").items()},
                               **{key + "_raw": value for key, value in _stats(valid, "validation").items()}, **feature_range(train, valid)}
                        observed = z.loc[z.feature_name.eq(name), "validation_zscore"].to_numpy()
                        row.update(zscore_summary(observed) if len(observed) else {key: np.nan for key in zscore_summary(np.array([0.])).keys()})
                        shift_rows.append(row)
                    matrix = baseline.transform_features(pp, training)
                    condition_rows.append({"fold": fold, "feature_spec": spec, **conditioning(matrix)})
                    influence_blocks.append(ols_influence(matrix, baseline._target(training[baseline.PRIMARY_TARGET]), fitted.estimator.predict(matrix), train_dates, fold, spec))
        checks["prediction_summary_arithmetic"] = _same_frame(tables["prediction_summary"], pd.DataFrame(prediction_rows))
        report["feature_range_validation_violations"] += int(not _same_frame(tables["feature_shift"], pd.DataFrame(shift_rows)))
        report["zscore_validation_violations"] += int(not _same_frame(tables["zscores"], pd.concat(expected_z, ignore_index=True)))
        report["coefficient_validation_violations"] += int(not _same_frame(tables["coefficient_summary"], coefficient_stability(pd.DataFrame(coefficient_rows), specifications)))
        report["conditioning_validation_violations"] += int(not _same_frame(tables["conditioning"], pd.DataFrame(condition_rows)))
        report["influence_validation_violations"] += int(not _same_frame(tables["influence"], pd.concat(influence_blocks, ignore_index=True)))
        state, state_info = state_context(data, authority, errors)
        report["state_context_validation_violations"] += int(not _same_frame(tables["state_context"], state))
        used_dates.update(state_info["state_previous_observed_dates_used"])
        forbidden = manifest.loc[manifest.target_name.eq(baseline.PRIMARY_TARGET) & manifest.fold.eq("final")]
        for label, mask in (("pre_test_gap", forbidden.pre_test_gap), ("final_test", forbidden.partition.eq("final_test")),
                            ("unlabeled_tail", forbidden.partition.eq("unlabeled_tail"))):
            count = len(used_dates & set(forbidden.loc[mask, "date"]))
            report[f"{label}_rows_used"] = count
            report[f"{label}_rows_used_in_diagnostics"] = count
            checks[f"{label}_isolated"] = count == 0
        summary_tables = {**tables, "coefficients_internal": pd.DataFrame(coefficient_rows)}
        checks["summary_arithmetic_and_protocol"] = _same_json(diagnostics.summary, _summarize(summary_tables, state_info, stage10_validation))
        checks["all_expected_diagnostic_rows"] = len(errors) == 450 and len(tables["feature_shift"]) == 144 and len(tables["conditioning"]) == 6
        for name in VIOLATIONS:
            checks[f"{name}_zero"] = report[name] == 0
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
        checks["artifact_audit_completed"] = False
        report["audit_error"] = str(exc)
    report["checks"] = {name: bool(value) for name, value in checks.items()}
    report["errors"] = [name for name, passed in checks.items() if not passed]
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    return report


def _fit_equal(first: baseline.FittedLinear, second: baseline.FittedLinear) -> bool:
    a, b = first.preprocessing, second.preprocessing
    return (a.features == b.features and a.fit_dates == b.fit_dates and first.train_dates == second.train_dates
            and _same_frame(a.usage, b.usage) and a.medians.equals(b.medians)
            and np.array_equal(a.scaler.mean_, b.scaler.mean_) and np.array_equal(a.scaler.scale_, b.scaler.scale_)
            and np.array_equal(first.estimator.coef_, second.estimator.coef_) and first.estimator.intercept_ == second.estimator.intercept_)


def isolation_checks(checkpoint: tuple[Any, ...], reference: dict[str, pd.DataFrame],
                     stage10_protocol: dict[str, Any], stage10_validation: dict[str, Any],
                     original: Diagnostics) -> dict[str, int]:
    """Execute adversarial copies, including state metadata; never write source changes."""
    data, manifest = checkpoint[:2]
    scoped = manifest.loc[manifest.target_name.eq(baseline.PRIMARY_TARGET) & manifest.fold.eq("final")]
    groups = {"gap": set(scoped.loc[scoped.pre_test_gap, "date"]),
              "final_test": set(scoped.loc[scoped.partition.eq("final_test"), "date"]),
              "tail": set(scoped.loc[scoped.partition.eq("unlabeled_tail"), "date"])}
    targets = [name for name in ("forward_return_1d", baseline.PRIMARY_TARGET, "forward_return_20d") if name in data]
    metadata = [name for name in STATE_METADATA_COLUMNS if name in data]
    cases = [("gap_mutation_violations", "gap", [*baseline.FEATURE_GROUPS, *targets, *metadata]),
             ("final_test_feature_mutation_violations", "final_test", list(baseline.FEATURE_GROUPS)),
             ("final_test_target_mutation_violations", "final_test", targets),
             ("final_test_metadata_mutation_violations", "final_test", metadata),
             ("secondary_target_mutation_violations", None, [name for name in targets if name != baseline.PRIMARY_TARGET]),
             ("tail_mutation_violations", "tail", [*baseline.FEATURE_GROUPS, *targets, *metadata])]
    results = {}
    for name, group, columns in cases:
        changed = data.copy(deep=True)
        mask = changed.date.isin(groups[group]) if group else pd.Series(True, index=changed.index)
        for i, column in enumerate(columns):
            if column in metadata:
                changed[column] = changed[column].astype(object)
                changed.loc[mask, column] = f"MUTATED_FUTURE_STATE_{i}"
            else:
                changed.loc[mask, column] = (-1 if i % 2 else 1) * 1e9
        mutated = build_diagnostics((changed, *checkpoint[1:]), reference, stage10_protocol, stage10_validation)
        results[name] = int(not (all(_same_frame(original.tables[key], mutated.tables[key]) for key in TABLE_NAMES)
                                 and _same_json(original.summary, mutated.summary)))
    first_fold = FOLDS[0]
    authority = baseline.cv_authority(data, manifest, checkpoint[2], checkpoint[3])
    day = authority[first_fold][1][0]
    for kind, column in (("X", "simple_return"), ("Y", baseline.PRIMARY_TARGET)):
        changed = data.copy(deep=True)
        changed.loc[changed.date.eq(day), column] += 1000. if kind == "X" else .5
        mutated = build_diagnostics((changed, *checkpoint[1:]), reference, stage10_protocol, stage10_validation)
        good = all(_fit_equal(original.fits[first_fold, model], mutated.fits[first_fold, model]) for model in baseline.LINEAR_MODELS)
        for table in ("conditioning", "influence"):
            good = good and _same_frame(original.tables[table].loc[original.tables[table].fold.eq(first_fold)],
                                       mutated.tables[table].loc[mutated.tables[table].fold.eq(first_fold)])
        if kind == "Y":
            for table, columns in (("zscores", None), ("contributions", None), ("row_errors", ["date", "model_name", "prediction"])):
                first = original.tables[table].loc[original.tables[table].fold.eq(first_fold)]
                second = mutated.tables[table].loc[mutated.tables[table].fold.eq(first_fold)]
                good = good and _same_frame(first if columns is None else first[columns], second if columns is None else second[columns])
        results[f"validation_{kind}_{'training_fit' if kind == 'X' else 'prediction_fit'}_violations"] = int(not good)
    results["secondary_target_dependency_violations"] = results["secondary_target_mutation_violations"]
    return results


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_inputs(labeled_path: Path, split_dir: Path, diagnostic_dir: Path,
                 baseline_dir: Path) -> tuple[tuple[Any, ...], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Read frozen hashes first; metadata is separate from the Stage 10 X whitelist."""
    checkpoint = baseline._read_checkpoint(labeled_path, split_dir, diagnostic_dir)
    columns = pd.read_csv(labeled_path, nrows=0).columns
    metadata_columns = [name for name in STATE_METADATA_COLUMNS if name in columns]
    if metadata_columns:
        # String parsing keeps changes to unused future metadata from affecting
        # dtype inference for authorized rows. Ages/booleans are parsed locally.
        metadata = pd.read_csv(labeled_path, usecols=["date", *metadata_columns], dtype=str)
        data = checkpoint[0].merge(metadata, on="date", how="left", validate="one_to_one")
        checkpoint = (data, *checkpoint[1:])
    validation = json.loads((baseline_dir / "NVDA_baseline_validation.json").read_text())
    stage10_protocol = json.loads((baseline_dir / "NVDA_baseline_protocol.json").read_text())
    for name in (*baseline.TABLE_NAMES, "protocol"):
        suffix = "json" if name == "protocol" else "csv"
        path = baseline_dir / f"NVDA_baseline_{name}.{suffix}"
        if validation.get("output_artifact_sha256", {}).get(str(path)) != _sha256(path):
            raise ValueError(f"Frozen Stage 10 artifact hash mismatch: {path.name}")
    for path, digest in stage10_protocol.get("input_artifact_sha256", {}).items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"Stage 10 upstream checkpoint changed: {Path(path).name}")
    tables = {name: pd.read_csv(baseline_dir / f"NVDA_baseline_{name}.csv", float_precision="round_trip") for name in baseline.TABLE_NAMES}
    return checkpoint, tables, stage10_protocol, validation


def diagnostic_figures(diagnostics: Diagnostics) -> dict[str, bytes]:
    """Five deterministic development-only plots; display ordering cannot alter X."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nasdaq-research-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = {}

    def save(figure: Any, name: str) -> None:
        figure.suptitle(f"{figure._diagnostic_title} — development CV only", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, .96))
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=140, metadata={"Software": "nasdaq-research Stage 10.1"})
        outputs[name] = buffer.getvalue()
        plt.close(figure)

    with plt.rc_context({"font.size": 9, "axes.grid": True, "grid.alpha": .2}):
        figure, axes = plt.subplots(4, 3, figsize=(12, 12))
        figure._diagnostic_title = "Actual vs prediction"
        for i, model in enumerate(baseline.LINEAR_MODELS):
            for j, fold in enumerate(FOLDS):
                rows = diagnostics.tables["row_errors"].loc[lambda f: f.model_name.eq(model) & f.fold.eq(fold)]
                ax = axes[i, j]
                ax.scatter(rows.actual_forward_return_5d, rows.prediction, s=17, alpha=.8)
                low = min(rows.actual_forward_return_5d.min(), rows.prediction.min())
                high = max(rows.actual_forward_return_5d.max(), rows.prediction.max())
                ax.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=1)
                ax.set(title=f"{model} / {fold}", xlabel="Actual return", ylabel="Predicted return")
        save(figure, "actual_vs_prediction_by_fold")

        figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
        figure._diagnostic_title = "Prediction ranges (min, mean, max)"
        for ax, fold in zip(axes, FOLDS):
            rows = diagnostics.tables["prediction_summary"].loc[lambda f: f.fold.eq(fold)].set_index("model_name")
            names = ["actual", *baseline.LINEAR_MODELS]
            means = [rows.actual_mean.iloc[0], *[rows.loc[name, "prediction_mean"] for name in baseline.LINEAR_MODELS]]
            mins = [rows.actual_min.iloc[0], *[rows.loc[name, "prediction_min"] for name in baseline.LINEAR_MODELS]]
            maxs = [rows.actual_max.iloc[0], *[rows.loc[name, "prediction_max"] for name in baseline.LINEAR_MODELS]]
            ax.errorbar(np.arange(len(names)), means, yerr=[np.array(means) - mins, np.array(maxs) - means], fmt="o", capsize=4)
            ax.set(title=fold, ylabel="Return ratio", xticks=np.arange(len(names)), xticklabels=names)
            ax.tick_params(axis="x", rotation=45)
        save(figure, "prediction_range_by_fold")

        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        figure._diagnostic_title = "cv_3 largest absolute errors"
        for ax, model in zip(axes.ravel(), baseline.LINEAR_MODELS):
            rows = diagnostics.tables["row_errors"].loc[lambda f: f.fold.eq("cv_3") & f.model_name.eq(model)].sort_values(
                ["absolute_error", "date"], ascending=[False, True], kind="stable").head(5)
            x = np.arange(len(rows))
            ax.bar(x - .18, rows.actual_forward_return_5d, width=.36, label="actual")
            ax.bar(x + .18, rows.prediction, width=.36, label="prediction")
            ax.set(title=model, ylabel="Return ratio", xticks=x, xticklabels=rows.date.str[5:])
            ax.legend()
        save(figure, "cv3_largest_absolute_errors")

        figure, ax = plt.subplots(figsize=(12, 5))
        figure._diagnostic_title = "Largest validation |z| features"
        shift = diagnostics.tables["feature_shift"].loc[lambda f: f.feature_spec.eq("all")]
        names = shift.loc[shift.fold.eq("cv_3") & shift.used_in_stage10_fold.eq(True)].sort_values(
            ["max_abs_validation_z", "feature_name"], ascending=[False, True], kind="stable").head(10).feature_name.tolist()
        x = np.arange(len(names))
        for j, fold in enumerate(FOLDS):
            rows = shift.loc[shift.fold.eq(fold)].set_index("feature_name")
            ax.bar(x + (j - 1) * .25, rows.loc[names, "max_abs_validation_z"], width=.25, label=fold)
        ax.set(xticks=x, xticklabels=names, ylabel="Maximum absolute z-score")
        ax.tick_params(axis="x", rotation=55)
        ax.legend()
        save(figure, "max_validation_abs_z_by_feature")

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        figure._diagnostic_title = "Coefficient drift (visual ordering only)"
        for ax, model in zip(axes.ravel(), baseline.LINEAR_MODELS):
            rows = diagnostics.tables["coefficient_summary"].loc[lambda f: f.model_name.eq(model)].sort_values(
                ["coefficient_std_available", "max_abs_coefficient", "feature_name"], ascending=[False, False, True], kind="stable").head(4)
            for row in rows.itertuples():
                ax.plot([1, 2, 3], [getattr(row, f"coefficient_{fold}") for fold in FOLDS], marker="o", label=row.feature_name)
            ax.set(title=model, xlabel="CV fold", ylabel="Standardized-feature coefficient", xticks=[1, 2, 3])
            ax.legend(fontsize=7)
        save(figure, "coefficient_drift")
    return outputs


def _atomic_bytes(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Stability output symlinks are forbidden")
    try:
        temporary.write_bytes(contents)
        temporary.replace(path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def run_pipeline(labeled_path: Path = RESEARCH_DIR / "NVDA_labeled.csv", split_dir: Path = RESEARCH_DIR / "splits",
                 diagnostic_dir: Path = RESEARCH_DIR / "diagnostics", baseline_dir: Path = RESEARCH_DIR / "modeling",
                 output_dir: Path = RESEARCH_DIR / "modeling/stability", generate_figures: bool = True) -> dict[str, Any]:
    """Publish stability diagnostics only, after reference/isolation/round-trip audits."""
    labeled_path, split_dir, diagnostic_dir, baseline_dir, output_dir = (Path(path).resolve() for path in (
        labeled_path, split_dir, diagnostic_dir, baseline_dir, output_dir))
    inputs = {labeled_path, *(split_dir / f"NVDA_{name}" for name in ("split_manifest.csv", "cv_folds.csv", "final_training_pool.csv",
        "split_summary.csv", "split_validation.json", "evaluation_protocol.json")),
        diagnostic_dir / "feature_inventory.csv", diagnostic_dir / "feature_quality.csv",
        *(baseline_dir / f"NVDA_baseline_{name}.csv" for name in baseline.TABLE_NAMES),
        baseline_dir / "NVDA_baseline_protocol.json", baseline_dir / "NVDA_baseline_validation.json"}
    if any(output_dir == path or output_dir in path.parents for path in inputs):
        raise ValueError("Stability output directory must not contain Stage 1–10 inputs")
    paths = {name: output_dir / f"NVDA_stability_{name}.csv" for name in TABLE_NAMES}
    summary_path, validation_path = (output_dir / f"NVDA_stability_{name}.json" for name in ("summary", "validation"))
    figure_paths = {name: output_dir / "figures" / f"{name}.png" for name in FIGURE_NAMES} if generate_figures else {}
    if any(path.is_symlink() for path in (*paths.values(), summary_path, validation_path, *figure_paths.values())):
        raise ValueError("Stability output symlinks cannot overwrite sources")
    protected = set(inputs)
    for root in {baseline.DATA_DIR.resolve(), labeled_path.parent, baseline_dir}:
        protected.update(path.resolve() for path in root.rglob("*") if path.is_file() and output_dir not in path.resolve().parents)
    before = {str(path): _sha256(path) for path in sorted(protected)}
    checkpoint, reference, stage10_protocol, stage10_validation = _read_inputs(labeled_path, split_dir, diagnostic_dir, baseline_dir)
    original = checkpoint[0].copy(deep=True)
    diagnostics = build_diagnostics(checkpoint, reference, stage10_protocol, stage10_validation)
    report = validate_diagnostics(checkpoint, reference, stage10_protocol, stage10_validation, diagnostics)
    if not report["valid"]:
        raise ValueError(f"Stability reference/diagnostic validation failed: {report['errors']}; {report.get('audit_error', '')}")
    report.update(isolation_checks(checkpoint, reference, stage10_protocol, stage10_validation, diagnostics))
    report["mutation_checks_executed"] = True
    report["mutation_checks"] = [name for name in VIOLATIONS if "mutation" in name or name.startswith("validation_")]
    for name in VIOLATIONS:
        report["checks"][f"{name}_zero"] = report[name] == 0
    pd.testing.assert_frame_equal(checkpoint[0], original, check_exact=True)
    report["checks"]["source_dataframe_unchanged"] = True
    csv_text = {name: table.to_csv(index=False) for name, table in diagnostics.tables.items()}
    reloaded = {name: pd.read_csv(StringIO(contents), float_precision="round_trip") for name, contents in csv_text.items()}
    roundtrip = validate_diagnostics(checkpoint, reference, stage10_protocol, stage10_validation,
                                     Diagnostics(reloaded, diagnostics.summary, diagnostics.fits, diagnostics.fit_records))
    report["checks"]["csv_roundtrip_validated"] = roundtrip["valid"]
    if not roundtrip["valid"]:
        raise ValueError(f"Stability CSV round-trip validation failed: {roundtrip['errors']}")
    figures = diagnostic_figures(diagnostics) if generate_figures else {}
    after = {path: _sha256(Path(path)) for path in before}
    report["upstream_artifact_mutations"] = sum(before[path] != after[path] for path in before)
    report["checks"]["upstream_artifact_mutations_zero"] = report["upstream_artifact_mutations"] == 0
    report["errors"] = [name for name, passed in report["checks"].items() if not passed]
    if report["errors"]:
        raise ValueError(f"Stability isolation/immutability validation failed: {report['errors']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        _atomic_bytes(path, csv_text[name].encode())
    _atomic_bytes(summary_path, (json.dumps(baseline._json_clean(diagnostics.summary), indent=2, allow_nan=False) + "\n").encode())
    if figures:
        (output_dir / "figures").mkdir(exist_ok=True)
        for name, contents in figures.items():
            _atomic_bytes(figure_paths[name], contents)
    after = {path: _sha256(Path(path)) for path in before}
    report.update({"source_sha256_before": before, "source_sha256_after": after, "source_artifact_count": len(before),
                   "source_artifacts_unchanged": before == after, "stage10_baseline_artifacts_unchanged": all(
                       before[str(path)] == after[str(path)] for path in inputs if path.parent == baseline_dir),
                   "fit_records": diagnostics.fit_records, "comparison_tolerance": {"rtol": 0, "atol": ATOL},
                   "output_paths": {**{name: str(path) for name, path in paths.items()}, "summary": str(summary_path), "validation": str(validation_path)},
                   "figure_paths": {name: str(path) for name, path in figure_paths.items()},
                   "output_artifact_sha256": {str(path): _sha256(path) for path in (*paths.values(), summary_path, *figure_paths.values())},
                   "reproducibility": {**stage10_validation["reproducibility"], "run_timestamp": "omitted"}})
    report["checks"]["all_source_hashes_unchanged_after_publish"] = before == after
    report["errors"] = [name for name, passed in report["checks"].items() if not passed]
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    _atomic_bytes(validation_path, (json.dumps(baseline._json_clean(report), indent=2, allow_nan=False) + "\n").encode())
    if not report["valid"]:
        raise ValueError("Upstream inputs changed during output publication")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-path", type=Path, default=RESEARCH_DIR / "NVDA_labeled.csv")
    parser.add_argument("--split-dir", type=Path, default=RESEARCH_DIR / "splits")
    parser.add_argument("--diagnostic-dir", type=Path, default=RESEARCH_DIR / "diagnostics")
    parser.add_argument("--baseline-dir", type=Path, default=RESEARCH_DIR / "modeling")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "modeling/stability")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    report = run_pipeline(args.labeled_path, args.split_dir, args.diagnostic_dir, args.baseline_dir, args.output_dir, not args.no_figures)
    print("Stage 10.1: Baseline Model Stability Diagnostics — development CV only")
    print(f"OOF rows: {report['oof_row_count']}; row errors: {report['row_error_count']}; folds: {report['fold_count']}")
    print(f"Stage 10 prediction consistency violations: {report['stage10_prediction_consistency_violations']}")
    print(f"Stage 10 metric consistency violations: {report['stage10_metric_consistency_violations']}")
    print("Final test locked: true; final test predictions generated: false; final test metrics computed: false")
    print(f"Gap/test/tail rows used: {report['pre_test_gap_rows_used']}/{report['final_test_rows_used']}/{report['unlabeled_tail_rows_used']}")
    print(f"Validation: {report['validation_status']}; upstream artifacts unchanged: {report['source_artifacts_unchanged']}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
