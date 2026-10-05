"""Stage 11: one pre-registered Market SMA representation ablation, offline.

No upstream definitions or whitelists change. A local, bijective column-name
adapter reuses Stage 10's unmodified preprocessing and estimators for treatment.
Only manifest-authorized CV rows are transformed; test/gap/tail are never fit.
"""

import argparse
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from nasdaq_research import modeling as baseline
from nasdaq_research import model_stability as stability
from nasdaq_research.dates import parse_dates

ATOL = 1e-12
FAMILIES = ("ols", "ridge")
ARMS = ("control", "treatment")
CONTROL = tuple(c for c, g in baseline.FEATURE_GROUPS.items() if g == "market")
REPLACEMENTS = {f"sma_{k}": f"close_to_sma_{k}" for k in (5, 20, 60)}
TREATMENT = tuple(REPLACEMENTS.get(c, c) for c in CONTROL)
SPECS = {"control": CONTROL, "treatment": TREATMENT}
TABLE_NAMES = ("feature_manifest", "oof_predictions", "fold_metrics", "feature_shift", "zscores",
               "prediction_summary", "coefficients", "coefficient_summary", "contributions",
               "ablation_summary", "benchmark_metrics", "stability_summary")
SAFETY = {"final_test_locked": True, "final_test_predictions_generated": False,
          "final_test_metrics_computed": False, "final_training_pool_used": False}
VIOLATIONS = (
    "control_prediction_reproduction_violations", "control_metric_reproduction_violations",
    "control_feature_usage_reproduction_violations", "control_coefficient_reproduction_violations",
    "control_fold_membership_reproduction_violations", "benchmark_reproduction_violations",
    "training_only_coverage_violations", "training_only_imputation_violations",
    "training_only_zero_variance_violations", "training_only_scaling_violations",
    "prediction_reconstruction_violations", "metric_integrity_violations",
    "diagnostic_integrity_violations", "feature_whitelist_violations", "candidate_count_violations",
    "pit_formula_violations", "truncation_invariance_violations", "past_sma_oracle_violations",
    "future_dependency_violations", "forbidden_rows_used_violations",
    "final_test_feature_mutation_dependency_violations", "final_test_target_mutation_dependency_violations",
    "final_test_metadata_mutation_dependency_violations", "pre_test_gap_mutation_dependency_violations",
    "unlabeled_tail_mutation_dependency_violations", "secondary_target_mutation_dependency_violations",
    "validation_feature_fit_dependency_violations", "validation_target_fit_dependency_violations",
    "upstream_mutation_violations")


@dataclass
class Experiment:
    tables: dict[str, pd.DataFrame]
    fits: dict[tuple[str, str, str], baseline.FittedLinear]
    fit_records: list[dict[str, Any]]
    summary: dict[str, Any]


def protocol(split_protocol: dict[str, Any], manifest_hash: str,
             input_hashes: dict[str, str]) -> dict[str, Any]:
    """The contract is saved before any Stage 11 fitting or result inspection."""
    return {
        "stage": "11", "stage_name": "Pre-Registered Extrapolation-Stable Market Feature Representation Ablation",
        "ticker": "NVDA", "experiment_status": "diagnostic-informed pre-registered follow-up development experiment",
        "independent_confirmatory_experiment": False, "primary_target": baseline.PRIMARY_TARGET,
        "target_definition": "Close[t+5] / Open[t+1] - 1", "feature_timestamp_semantics": "trading day t after close",
        "earliest_entry": "Open[t+1]", "split_authority": "frozen Stage 9.1 manifest and evaluation protocol",
        "split_manifest_sha256": manifest_hash, "split_protocol_version": "9.1",
        "cv_boundaries": split_protocol["boundaries"]["cv"],
        "control_features": list(CONTROL), "treatment_features": list(TREATMENT),
        "candidate_count_per_arm": 9,
        "formulas": {v: f"Close / {k} - 1" for k, v in REPLACEMENTS.items()},
        "representation_scope": "local same-row transformation of authorized CV rows; original SMA artifacts unchanged",
        "name_adapter": "bijective treatment-to-control column labels for unchanged Stage 10 fit; restore treatment labels before transform/diagnostics",
        "models": ["C1 Original Market OLS", "C2 Original Market Ridge", "T1 Relative Market OLS", "T2 Relative Market Ridge"],
        "paired_comparisons": ["C1 vs T1", "C2 vs T2"],
        "ols_implementation": "Stage 10 LinearRegression(fit_intercept=True, n_jobs=1)",
        "ridge_alpha": 1.0, "ridge_solver": "svd", "fit_intercept": True,
        "training_coverage_threshold": 0.50, "exact_50_percent_rule": "KEEP",
        "imputation_rule": "effective-training median",
        "zero_variance_rule": "drop exactly constant imputed effective-training feature; Stage 10 semantics",
        "scaling_rule": "effective-training StandardScaler, ddof=0", "target_scaling_rule": "none",
        "primary_metric": "MAE", "aggregation": "equal-weight mean fold MAE; sample fold std ddof=1",
        "secondary_metrics": list(baseline.SECONDARY_METRICS),
        "range_diagnostics": "strict below/above raw training bounds; equality within; denominator non-null validation values",
        "zscore_diagnostics": ["max |z|", "count |z| > 3", "count |z| > 5"],
        "prediction_diagnostics": ["mean", "std ddof=1", "min", "max", "mean absolute", "median absolute",
                                   "count/fraction |prediction| > 0.10", "count/fraction |prediction| > 0.30"],
        "coefficient_diagnostics": ["intercept", "standardized coefficients", "L2", "max absolute", "contributions"],
        "residual_definition": "actual minus prediction",
        "success_interpretation_framework": {
            "categories": ["SUPPORTED", "PARTIALLY_SUPPORTED_OR_INCONCLUSIVE", "NOT_SUPPORTED"],
            "safety_gate": "all leakage, reproduction, determinism, PIT and immutability checks must pass",
            "primary_stability_endpoints": ["SMA-related raw out-of-range fraction", "SMA-related max absolute training z",
                                            "mean absolute prediction for each model family"],
            "material_improvement_ratio": 0.75,
            "supported_rule": "all three endpoints reduce at least 25% in cv_3 for both OLS and Ridge; pooled endpoints strictly improve; no new extreme instability in cv_1/cv_2",
            "new_instability_rule": "outside cv_3: treatment all-feature max |z| > 5 and > control, or treatment count |prediction| > 0.10 or > 0.30 exceeds control",
            "partial_rule": "at least one cv_3 stability endpoint reduces at least 25%, but supported rule fails; mixed folds/families remain inconclusive",
            "not_supported_rule": "no cv_3 stability endpoint reduces at least 25%",
            "MAE_role": "report unchanged primary predictive metric; MAE cannot determine mechanism category",
            "claim_limit": "descriptive support only; no proven causal mechanism, better generalization or alpha",
        },
        "benchmark_anchors": {"zero_return": "0", "historical_mean": "mean current effective training primary Y",
                              "part_of_representation_ablation": False},
        "forbidden_interventions": ["alternative SMA formulas", "formula search", "fundamental interventions",
            "All specification", "target changes", "new splits", "alpha tuning", "new model families",
            "predictive feature selection", "PCA", "clipping", "winsorization", "outlier removal",
            "alternative scaler/imputation", "final model training", "backtest", "trading strategy"],
        "final_test_lock_rules": "no predictions, metrics, distribution diagnostics or conclusion inputs; in-memory isolation tests only",
        "pre_test_gap_allowed_in_cv": False, "unlabeled_tail_allowed": False, **SAFETY,
        "comparison_tolerance": {"rtol": 0, "atol": ATOL},
        "input_artifact_sha256": input_hashes,
        "limitations": ["Development CV was previously examined in Stage 10/10.1.",
                        "75 overlapping-label OOF rows, 25 per fold; observations are not independent.",
                        "Locked Final Test remains the unused confirmatory dataset."]}


def representation(rows: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Exact nine-column X; Close is a local formula input, never a predictor."""
    if arm not in SPECS or rows.columns.duplicated().any():
        raise ValueError("Registered arm and unique source columns required")
    result = baseline._numeric(rows, CONTROL)
    if arm == "treatment":
        close = baseline._numeric(rows, ("close",))["close"]
        if close.isna().any() or (close <= 0).any():
            raise ValueError("Positive observed same-day Close required")
        for old, new in REPLACEMENTS.items():
            denominator = result[old]
            if (denominator.dropna() <= 0).any():
                raise ValueError("Positive observed SMA denominator required")
            result[old] = close / denominator - 1
        result = result.rename(columns=REPLACEMENTS)
    if tuple(result.columns) != SPECS[arm] or len(result.columns) != 9:
        raise ValueError("Exact nine-feature representation required")
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Non-finite relative representation")
    return result


def fit_arm(family: str, arm: str, training_x: pd.DataFrame, target: pd.Series) -> baseline.FittedLinear:
    """Reuse Stage 10 exactly, without extending or mutating its global whitelist."""
    if family not in FAMILIES or arm not in SPECS or tuple(training_x.columns) != SPECS[arm]:
        raise ValueError("Registered family/arm and exact X whitelist required")
    reverse = {v: k for k, v in REPLACEMENTS.items()} if arm == "treatment" else {}
    fitted = baseline.fit_linear(f"{family}_market", training_x.rename(columns=reverse), target, CONTROL)
    if arm == "treatment":
        pp = fitted.preprocessing
        pp.features = tuple(REPLACEMENTS.get(c, c) for c in pp.features)
        pp.medians = pp.medians.rename(index=REPLACEMENTS)
        pp.usage["feature_name"] = pp.usage.feature_name.replace(REPLACEMENTS)
        pp.scaler.feature_names_in_ = np.array(pp.features, dtype=object)
    return fitted


def prediction_summary(predictions: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(predictions)
    result = {**stability._stats(predictions, "prediction"),
              "mean_abs_prediction": float(absolute.mean()), "median_abs_prediction": float(np.median(absolute)),
              "max_abs_prediction": float(absolute.max())}
    for threshold, tag in ((.10, "0_10"), (.30, "0_30")):
        result[f"count_abs_prediction_gt_{tag}"] = int((absolute > threshold).sum())
        result[f"fraction_abs_prediction_gt_{tag}"] = float((absolute > threshold).mean())
    return result


def cv_authority(data: pd.DataFrame, manifest: pd.DataFrame, split_protocol: dict[str, Any],
                 folds: pd.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    """Delegate all split rules to Stage 10, scope payload metadata to CV roles.

    Dates remain immutable join identities. Unused rows' ticker payload cannot
    veto development outputs; ticker is still checked on every authorized row.
    """
    relevant = manifest.loc[manifest.target_name.eq(baseline.PRIMARY_TARGET)
        & manifest.fold.isin(stability.FOLDS) & manifest.role.isin(["train", "validation"]), "date"]
    scoped = data.copy(deep=True)
    scoped.loc[~scoped.date.isin(relevant), "ticker"] = "NVDA"
    return baseline.cv_authority(scoped, manifest, split_protocol, folds)


def mechanism_inputs(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fold and pooled endpoints, without duplicated OLS/Ridge feature exposure."""
    rows = []
    for fold in (*stability.FOLDS, "pooled"):
        for arm in ARMS:
            for family in FAMILIES:
                shift = tables["feature_shift"]
                zs = tables["zscores"]
                oof = tables["oof_predictions"]
                select = lambda frame: frame.loc[frame.arm.eq(arm) & frame.model.eq(family)
                                                   & (True if fold == "pooled" else frame.fold.eq(fold))]
                shift, zs, oof = map(select, (shift, zs, oof))
                sma_names = tuple(REPLACEMENTS) if arm == "control" else tuple(REPLACEMENTS.values())
                sma_shift = shift.loc[shift.feature_name.isin(sma_names)]
                sma_z = zs.loc[zs.feature_name.isin(sma_names)]
                row = {"fold": fold, "arm": arm, "model": family}
                for subset, prefix in ((shift, "all_features"), (sma_shift, "sma_related")):
                    n = int(subset.validation_non_null_count.sum())
                    count = int(subset.validation_outside_training_range_count.sum())
                    row[f"{prefix}_range_value_count"] = n
                    row[f"{prefix}_outside_count"] = count
                    row[f"{prefix}_outside_fraction"] = count / n if n else np.nan
                for subset, prefix in ((zs, "all_features"), (sma_z, "sma_related")):
                    info = stability.zscore_summary(subset.validation_zscore.to_numpy()) if len(subset) else {
                        key: np.nan for key in stability.zscore_summary(np.array([0.])).keys()}
                    row.update({f"{prefix}_{key}": value for key, value in info.items()})
                row.update(prediction_summary(oof.prediction.to_numpy()))
                rows.append(row)
    return pd.DataFrame(rows)


def assess_mechanism(endpoints: pd.DataFrame) -> dict[str, Any]:
    """Apply the pre-registered rule; no metric ranking or model selection."""
    endpoint_names = ("sma_related_outside_fraction", "sma_related_max_abs_validation_z", "mean_abs_prediction")
    ratios, material, pooled, new_instability = [], [], [], []
    for family in FAMILIES:
        for fold in (*stability.FOLDS, "pooled"):
            pair = endpoints.loc[endpoints.model.eq(family) & endpoints.fold.eq(fold)].set_index("arm")
            control, treatment = pair.loc["control"], pair.loc["treatment"]
            for name in endpoint_names:
                c, t = float(control[name]), float(treatment[name])
                ratio = t / c if c > 0 else (1.0 if t == 0 else np.inf)
                ratios.append({"model": family, "fold": fold, "endpoint": name, "control": c, "treatment": t,
                               "treatment_to_control_ratio": ratio})
                if fold == "cv_3":
                    material.append(bool(np.isfinite(ratio) and ratio <= .75))
                if fold == "pooled":
                    pooled.append(bool(np.isfinite(ratio) and ratio < 1))
            if fold in ("cv_1", "cv_2"):
                new = (treatment.all_features_max_abs_validation_z > 5
                       and treatment.all_features_max_abs_validation_z > control.all_features_max_abs_validation_z)
                new |= any(treatment[name] > control[name] for name in (
                    "count_abs_prediction_gt_0_10", "count_abs_prediction_gt_0_30"))
                if new:
                    new_instability.append({"fold": fold, "model": family})
    category = ("SUPPORTED" if all(material) and all(pooled) and not new_instability else
                "PARTIALLY_SUPPORTED_OR_INCONCLUSIVE" if any(material) else "NOT_SUPPORTED")
    return {"category": category, "cv3_material_endpoint_improvement_count": sum(material),
            "cv3_endpoint_count": len(material), "pooled_endpoints_all_improve": all(pooled),
            "new_extreme_instability_outside_cv3": new_instability, "endpoint_comparisons": ratios,
            "interpretation": "Development-CV descriptive evidence only; no causal proof or generalization claim.",
            "MAE_used_to_determine_category": False, "model_selected": None}


def build_experiment(data: pd.DataFrame, manifest: pd.DataFrame, split_protocol: dict[str, Any],
                     folds: pd.DataFrame) -> Experiment:
    """Run only four learned arms plus fixed benchmark anchors on existing roles."""
    authority = cv_authority(data, manifest, split_protocol, folds)
    keyed = data.copy(deep=True)
    keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
    blocks: dict[str, list[Any]] = {name: [] for name in TABLE_NAMES}
    fits, records = {}, []
    with threadpool_limits(limits=1):
        for fold, (train_dates, valid_dates) in authority.items():
            training, validation = keyed.loc[train_dates], keyed.loc[valid_dates]
            y_train, y_valid = baseline._target(training[baseline.PRIMARY_TARGET]), baseline._target(validation[baseline.PRIMARY_TARGET])
            for anchor, prediction in (("zero_return", np.zeros(len(valid_dates))),
                                       ("historical_mean", np.full(len(valid_dates), y_train.mean()))):
                blocks["benchmark_metrics"].append({"fold": fold, "model": anchor, "train_count": len(train_dates),
                    "validation_count": len(valid_dates), **baseline.regression_metrics(y_valid, prediction)})
            for arm in ARMS:
                train_x, valid_x = representation(training, arm), representation(validation, arm)
                for family in FAMILIES:
                    tags = {"fold": fold, "arm": arm, "model": family}
                    fitted = fit_arm(family, arm, train_x, training[baseline.PRIMARY_TARGET])
                    fits[fold, arm, family] = fitted
                    pp = fitted.preprocessing
                    predictions = baseline.predict_linear(fitted, valid_x)
                    blocks["oof_predictions"].append(pd.DataFrame({"date": valid_dates, "ticker": "NVDA", **tags,
                        "actual": y_valid, "prediction": predictions, "error": y_valid - predictions,
                        "absolute_error": np.abs(y_valid - predictions)}))
                    blocks["fold_metrics"].append({**tags, "train_count": len(train_dates),
                        "validation_count": len(valid_dates), **baseline.regression_metrics(y_valid, predictions)})
                    blocks["feature_manifest"].append(pp.usage.assign(**tags))
                    records.append({**tags, "training_dates": list(fitted.train_dates), "validation_dates": valid_dates,
                        "preprocessing_fit_dates": list(pp.fit_dates), "scaler_sample_count": int(pp.scaler.n_samples_seen_),
                        "fit_target": baseline.PRIMARY_TARGET, "ridge_alpha": 1.0 if family == "ridge" else None})
                    zrows = stability.validation_zscores(fitted, valid_x, fold, arm)
                    zrows = zrows.drop(columns="feature_spec").assign(arm=arm, model=family)
                    blocks["zscores"].append(zrows)
                    for name in pp.features:
                        train, valid = train_x[name], valid_x[name]
                        range_info = stability.feature_range(train, valid)
                        below = range_info["validation_below_training_min_count"]
                        above = range_info["validation_above_training_max_count"]
                        blocks["feature_shift"].append({**tags, "feature_name": name,
                            "training_min_raw": float(train.min()), "training_max_raw": float(train.max()),
                            "training_non_null_count": int(train.notna().sum()), "validation_non_null_count": int(valid.notna().sum()),
                            "validation_missing_count": int(valid.isna().sum()), **range_info,
                            "validation_within_training_range_count": int(valid.notna().sum() - below - above),
                            **stability.zscore_summary(zrows.loc[zrows.feature_name.eq(name), "validation_zscore"].to_numpy())})
                    coef = pd.Series(fitted.estimator.coef_, index=pp.features)
                    intercept = float(fitted.estimator.intercept_)
                    blocks["coefficients"].append(pd.DataFrame({**tags, "feature_name": pp.features,
                        "coefficient": coef.to_numpy(), "intercept": intercept}))
                    blocks["coefficient_summary"].append({**tags, "intercept": intercept,
                        "coefficient_L2": float(np.linalg.norm(coef)), "max_abs_coefficient": float(coef.abs().max())})
                    contribution = stability.prediction_contributions(zrows.assign(feature_spec=arm), coef, intercept,
                        pd.Series(predictions, index=valid_dates), family)
                    contribution = contribution.drop(columns=["feature_spec", "model_name"]).assign(arm=arm, model=family)
                    blocks["contributions"].append(contribution)
                    blocks["prediction_summary"].append({**tags, "n": len(valid_dates), **prediction_summary(predictions)})
    tables = {name: (pd.concat(values, ignore_index=True) if isinstance(values[0], pd.DataFrame) else pd.DataFrame(values))
              for name, values in blocks.items() if values}
    summary_rows = []
    for arm in ARMS:
        for family in FAMILIES:
            rows = tables["fold_metrics"].loc[lambda f: f.arm.eq(arm) & f.model.eq(family)]
            oof = tables["oof_predictions"].loc[lambda f: f.arm.eq(arm) & f.model.eq(family)]
            summary_rows.append({"arm": arm, "model": family,
                **{f"mean_fold_{metric}": float(rows[metric].mean()) for metric in baseline.METRICS},
                "std_fold_MAE": float(rows.MAE.std(ddof=1)),
                **{f"pooled_oof_{metric}": value for metric, value in baseline.regression_metrics(oof.actual, oof.prediction).items()}})
    tables["ablation_summary"] = pd.DataFrame(summary_rows)
    tables["stability_summary"] = mechanism_inputs(tables)
    assessment = assess_mechanism(tables["stability_summary"])
    summary = {"stage": "11", "ticker": "NVDA", "primary_target": baseline.PRIMARY_TARGET, **SAFETY,
        "oof_unique_dates": 75, "learned_prediction_rows": len(tables["oof_predictions"]),
        "candidate_feature_counts": {arm: len(names) for arm, names in SPECS.items()},
        "fold_train_counts": {fold: len(train) for fold, (train, _) in authority.items()},
        "fold_validation_counts": {fold: len(valid) for fold, (_, valid) in authority.items()},
        "predictive_summary": tables["ablation_summary"].to_dict("records"),
        "stability_endpoints": tables["stability_summary"].to_dict("records"), "mechanism_assessment": assessment}
    return Experiment({name: tables[name] for name in TABLE_NAMES}, fits, records, summary)


def _same(actual: pd.DataFrame, expected: pd.DataFrame, *, exact: bool = False) -> bool:
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=exact, rtol=0, atol=ATOL)
        return True
    except (AssertionError, ValueError):
        return False


def control_reproduction(experiment: Experiment, reference: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compare original rows, metrics, complete filter statistics and coefficients."""
    report = {name: 0 for name in VIOLATIONS if "reproduction" in name}
    serialized = {}
    oof_reference = reference["oof_predictions"]
    for family in FAMILIES:
        model = f"{family}_market"
        oof = experiment.tables["oof_predictions"].loc[lambda f: f.arm.eq("control") & f.model.eq(family)]
        membership = oof[["date", "ticker", "fold"]]
        report["control_fold_membership_reproduction_violations"] += int(not _same(membership, oof_reference[["date", "ticker", "fold"]], exact=True))
        prediction = oof[["date", "ticker", "fold", "actual", "prediction"]].rename(columns={
            "actual": f"actual_{baseline.PRIMARY_TARGET}", "prediction": f"{model}_prediction"})
        expected = oof_reference[prediction.columns]
        report["control_prediction_reproduction_violations"] += int(not _same(prediction, expected))
        projected = {"predictions": (prediction, expected)}
        for name, source, violation in (
                ("fold_metrics", "cv_metrics", "control_metric_reproduction_violations"),
                ("feature_manifest", "feature_usage", "control_feature_usage_reproduction_violations"),
                ("coefficients", "coefficients", "control_coefficient_reproduction_violations")):
            current = experiment.tables[name].loc[lambda f: f.arm.eq("control") & f.model.eq(family)].copy()
            current["model_name"], current["feature_spec"] = model, "market"
            expected = reference[source].loc[lambda f: f.model_name.eq(model)]
            current = current.loc[:, expected.columns]
            report[violation] += int(not _same(current, expected))
            projected[name] = (current, expected)
        serialized[model] = {name: a.to_csv(index=False) == b.to_csv(index=False) for name, (a, b) in projected.items()}
    for anchor in ("zero_return", "historical_mean"):
        current = experiment.tables["benchmark_metrics"].loc[lambda f: f.model.eq(anchor)].copy()
        current["model_name"], current["feature_spec"] = anchor, "none"
        expected = reference["cv_metrics"].loc[lambda f: f.model_name.eq(anchor)]
        report["benchmark_reproduction_violations"] += int(not _same(current.loc[:, expected.columns], expected))
        for fold in stability.FOLDS:
            rows = oof_reference.loc[oof_reference.fold.eq(fold)]
            if anchor == "zero_return":
                correct = rows.zero_return_prediction.eq(0).all()
            else:
                mean = experiment.fits[fold, "control", "ols"].estimator.intercept_
                correct = np.allclose(rows.historical_mean_prediction, mean, rtol=0, atol=ATOL)
            report["benchmark_reproduction_violations"] += int(not correct)
    report["control_projected_csv_byte_equality"] = serialized
    report["control_reproduction_tolerance"] = {"rtol": 0, "atol": ATOL}
    return report


def pit_checks(data: pd.DataFrame, authority: dict[str, tuple[list[str], list[str]]],
               *, check_sma: bool = False) -> dict[str, int]:
    """Independent scalar formula, prefix oracle and archived trailing-mean audit."""
    dates = sorted(set(day for train, valid in authority.values() for day in (*train, *valid)))
    authorized = data.set_index("date").loc[dates]
    relative = representation(authorized, "treatment")
    formula = prefix = sma = 0
    # All authorized rows, not just favorable examples; missing warmups are paired.
    for day, row in authorized.iterrows():
        for old, new in REPLACEMENTS.items():
            expected = np.nan if pd.isna(row[old]) else float(row["close"]) / float(row[old]) - 1.0
            formula += int(not np.isclose(relative.loc[day, new], expected, rtol=0, atol=ATOL, equal_nan=True))
    sample = sorted(set([dates[0], dates[-1], *[days[i] for _, days in authority.values() for i in (0, 12, 24)]]))
    for day in sample:
        prefix_rows = authorized.loc[authorized.index <= day]
        prefix += int(not _same(representation(prefix_rows, "treatment").tail(1), relative.loc[[day]], exact=True))
    if check_sma:
        # The actual Stage 3 artifact was retained. Recompute its causal mean
        # from the entire authorized prefix, without loading future Close.
        source = data.loc[data.date.le(dates[-1])].set_index("date")
        for day in dates:
            past = source.loc[source.index <= day, "close"].to_numpy(dtype=float)
            for old in REPLACEMENTS:
                k = int(old.split("_")[1])
                expected = float(sum(map(float, past[-k:])) / k) if len(past) >= k else np.nan
                sma += int(not np.isclose(source.loc[day, old], expected, rtol=0, atol=ATOL, equal_nan=True))
    return {"pit_formula_violations": formula, "truncation_invariance_violations": prefix,
            "past_sma_oracle_violations": sma, "future_dependency_violations": formula + prefix + sma}


def validate_experiment(data: pd.DataFrame, manifest: pd.DataFrame, split_protocol: dict[str, Any],
                        folds: pd.DataFrame, experiment: Experiment,
                        reference: dict[str, pd.DataFrame], *, check_sma: bool = False) -> dict[str, Any]:
    """Audit independently from fitted source statistics and serialized row records."""
    report: dict[str, Any] = {name: 0 for name in VIOLATIONS}
    report.update({"stage": "11", **SAFETY, "checks": {}})
    authority = cv_authority(data, manifest, split_protocol, folds)
    keyed = data.copy()
    keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
    checks = report["checks"]
    report.update(control_reproduction(experiment, reference))
    report.update(pit_checks(data, authority, check_sma=check_sma))
    report["past_sma_oracle_executed"] = check_sma
    used_dates = set()
    checks["table_names_exact"] = tuple(experiment.tables) == TABLE_NAMES
    checks["fit_keys_exact"] = set(experiment.fits) == {(fold, arm, family) for fold in authority for arm in ARMS for family in FAMILIES}
    checks["fit_records_exact"] = len(experiment.fit_records) == 12
    for record in experiment.fit_records:
        train, valid = authority[record["fold"]]
        used_dates.update(record["training_dates"] + record["validation_dates"] + record["preprocessing_fit_dates"])
        checks[f"{record['fold']}_{record['arm']}_{record['model']}_fit_scope"] = (
            record["training_dates"] == train and record["preprocessing_fit_dates"] == train
            and record["validation_dates"] == valid and record["scaler_sample_count"] == len(train)
            and record["fit_target"] == baseline.PRIMARY_TARGET
            and record["ridge_alpha"] == (1.0 if record["model"] == "ridge" else None))
    forbidden = manifest.loc[manifest.fold.eq("final") & manifest.target_name.eq(baseline.PRIMARY_TARGET)
        & (manifest.pre_test_gap | manifest.partition.isin(["final_test", "unlabeled_tail"]))]
    report["forbidden_rows_used_violations"] = len(used_dates & set(forbidden.date))
    for fold, (train_dates, valid_dates) in authority.items():
        for arm in ARMS:
            train_x = representation(keyed.loc[train_dates], arm)
            valid_x = representation(keyed.loc[valid_dates], arm)
            report["feature_whitelist_violations"] += int(tuple(train_x) != SPECS[arm] or tuple(valid_x) != SPECS[arm])
            report["candidate_count_violations"] += int(len(train_x.columns) != 9 or len(valid_x.columns) != 9)
            for family in FAMILIES:
                select = lambda f: f.loc[f.fold.eq(fold) & f.arm.eq(arm) & f.model.eq(family)]
                usage = select(experiment.tables["feature_manifest"])
                coeff = select(experiment.tables["coefficients"])
                oof = select(experiment.tables["oof_predictions"])
                zrows = select(experiment.tables["zscores"])
                shifts = select(experiment.tables["feature_shift"])
                key = f"{fold}_{arm}_{family}"
                checks[key + "_usage_schema"] = (usage.feature_name.tolist() == list(SPECS[arm])
                    and not usage.feature_name.duplicated().any() and usage.candidate_feature.eq(True).all()
                    and usage.training_row_count.eq(len(train_dates)).all())
                used = usage.loc[usage.used.eq(True)].set_index("feature_name")
                checks[key + "_coefficient_schema"] = (not used.empty and coeff.feature_name.tolist() == used.index.tolist()
                    and coeff.intercept.nunique() == 1 and np.isfinite(coeff[["coefficient", "intercept"]].to_numpy()).all())
                checks[key + "_oof_membership"] = (oof.date.tolist() == valid_dates and oof.ticker.eq("NVDA").all())
                actual = baseline._target(keyed.loc[valid_dates, baseline.PRIMARY_TARGET])
                checks[key + "_actuals"] = np.array_equal(oof.actual.to_numpy(), actual)
                checks[key + "_errors"] = np.array_equal(oof.error, oof.actual - oof.prediction) and np.array_equal(oof.absolute_error, np.abs(oof.error))
                for row in usage.itertuples():
                    expected = baseline._audit_training_stats(train_x[row.feature_name])
                    close = lambda a, b: np.isclose(a, b, rtol=0, atol=ATOL, equal_nan=True)
                    report["training_only_coverage_violations"] += int(row.training_non_null_count != expected["training_non_null_count"] or not close(row.training_coverage, expected["training_coverage"]))
                    report["training_only_imputation_violations"] += int(not close(row.training_median, expected["training_median"]))
                    report["training_only_zero_variance_violations"] += int(row.drop_reason != expected["drop_reason"] or row.used != (expected["drop_reason"] == "used"))
                    report["training_only_scaling_violations"] += int(any(not close(getattr(row, field), expected[field]) for field in (
                        "training_mean_after_imputation", "training_std_after_imputation", "scaler_scale")))
                manual_z = (valid_x.loc[:, used.index].fillna(used.training_median) - used.training_mean_after_imputation) / used.scaler_scale
                contribution = select(experiment.tables["contributions"])
                rebuilt = contribution.groupby("date", sort=False).feature_contribution.sum() + float(coeff.intercept.iloc[0])
                report["prediction_reconstruction_violations"] += int((~np.isclose(rebuilt.loc[valid_dates], oof.prediction, rtol=0, atol=ATOL)).sum())
                for name in used.index:
                    train, valid = train_x[name], valid_x[name]
                    observed = valid.dropna()
                    outside = (observed < train.min()) | (observed > train.max())
                    summary = shifts.loc[shifts.feature_name.eq(name)]
                    expected_range = {"training_min_raw": train.min(), "training_max_raw": train.max(),
                        "training_non_null_count": int(train.notna().sum()), "validation_non_null_count": len(observed),
                        "validation_missing_count": int(valid.isna().sum()),
                        "validation_below_training_min_count": int((observed < train.min()).sum()),
                        "validation_above_training_max_count": int((observed > train.max()).sum()),
                        "validation_within_training_range_count": int((~outside).sum()),
                        "validation_outside_training_range_count": int(outside.sum()),
                        "validation_outside_training_range_fraction": float(outside.mean()) if len(observed) else np.nan,
                        **stability.zscore_summary(manual_z[name].to_numpy())}
                    report["diagnostic_integrity_violations"] += int(len(summary) != 1 or any(not close(summary.iloc[0][field], value) for field, value in expected_range.items()))
                    z = zrows.loc[zrows.feature_name.eq(name)]
                    report["diagnostic_integrity_violations"] += int(z.date.tolist() != valid_dates or not np.allclose(z.validation_zscore, manual_z[name], rtol=0, atol=ATOL))
                checks[key + "_diagnostic_features"] = shifts.feature_name.tolist() == list(used.index) and set(zrows.feature_name) == set(used.index)
                expected_metrics = baseline.regression_metrics(actual, oof.prediction.to_numpy())
                metrics = select(experiment.tables["fold_metrics"])
                report["metric_integrity_violations"] += int(len(metrics) != 1 or any(not close(metrics.iloc[0][name], value) for name, value in expected_metrics.items())
                    or metrics.iloc[0].train_count != len(train_dates) or metrics.iloc[0].validation_count != len(valid_dates))
                pred_summary = select(experiment.tables["prediction_summary"])
                report["diagnostic_integrity_violations"] += int(len(pred_summary) != 1 or any(not close(pred_summary.iloc[0][name], value)
                    for name, value in prediction_summary(oof.prediction.to_numpy()).items()))
                coef_summary = select(experiment.tables["coefficient_summary"])
                report["diagnostic_integrity_violations"] += int(len(coef_summary) != 1 or any(not close(coef_summary.iloc[0][name], value) for name, value in {
                    "intercept": coeff.intercept.iloc[0], "coefficient_L2": np.linalg.norm(coeff.coefficient), "max_abs_coefficient": coeff.coefficient.abs().max()}.items()))
                # Check each serialized feature contribution, not only its sum.
                for name in used.index:
                    c = contribution.loc[contribution.feature_name.eq(name)]
                    coefficient = float(coeff.set_index("feature_name").loc[name, "coefficient"])
                    checks[key + "_contribution_" + name] = (c.date.tolist() == valid_dates and np.allclose(c.feature_contribution,
                        manual_z[name] * coefficient, rtol=0, atol=ATOL) and np.allclose(c.coefficient, coefficient, rtol=0, atol=ATOL))
    expected_endpoints = mechanism_inputs(experiment.tables)
    checks["stability_summary_integrity"] = _same(experiment.tables["stability_summary"], expected_endpoints)
    checks["mechanism_inputs_integrity"] = stability._same_json(experiment.summary["mechanism_assessment"], assess_mechanism(expected_endpoints))
    for arm in ARMS:
        for family in FAMILIES:
            rows = experiment.tables["fold_metrics"].loc[lambda f: f.arm.eq(arm) & f.model.eq(family)]
            summary = experiment.tables["ablation_summary"].loc[lambda f: f.arm.eq(arm) & f.model.eq(family)]
            oof = experiment.tables["oof_predictions"].loc[lambda f: f.arm.eq(arm) & f.model.eq(family)]
            expected = {**{f"mean_fold_{name}": float(rows[name].mean()) for name in baseline.METRICS},
                "std_fold_MAE": float(rows.MAE.std(ddof=1)),
                **{f"pooled_oof_{name}": value for name, value in baseline.regression_metrics(oof.actual, oof.prediction).items()}}
            checks[f"{arm}_{family}_ablation_summary"] = len(summary) == 1 and all(np.isclose(summary.iloc[0][name], value, rtol=0, atol=ATOL, equal_nan=True) for name, value in expected.items())
    checks["prediction_rows_complete"] = len(experiment.tables["oof_predictions"]) == 300 and not experiment.tables["oof_predictions"].duplicated(["date", "arm", "model"]).any()
    report["prediction_reconstruction_rows_checked"] = len(experiment.tables["oof_predictions"])
    for name in VIOLATIONS:
        checks[name + "_zero"] = report[name] == 0
    report["errors"] = [name for name, passed in checks.items() if not passed]
    report["valid"] = not report["errors"]
    return report


def isolation_checks(data: pd.DataFrame, manifest: pd.DataFrame, split_protocol: dict[str, Any],
                     folds: pd.DataFrame, original: Experiment) -> dict[str, int]:
    """Mutate only in-memory copies, including unusable values and provenance."""
    report = {name: 0 for name in VIOLATIONS if "dependency" in name}
    primary = manifest.loc[manifest.fold.eq("final") & manifest.target_name.eq(baseline.PRIMARY_TARGET)]
    features = [*CONTROL, "close"]
    targets = [c for c in data if c.startswith("forward_return_") or c.startswith("target_")]
    metadata = [c for c in data if c not in features and c not in targets and c != "date"]
    scenarios = [
        ("final_test_feature", set(primary.loc[primary.partition.eq("final_test"), "date"]), features),
        ("final_test_target", set(primary.loc[primary.partition.eq("final_test"), "date"]), targets),
        ("final_test_metadata", set(primary.loc[primary.partition.eq("final_test"), "date"]), metadata),
        ("pre_test_gap", set(primary.loc[primary.pre_test_gap, "date"]), [*features, *targets, *metadata]),
        ("unlabeled_tail", set(primary.loc[primary.partition.eq("unlabeled_tail"), "date"]), [*features, *targets, *metadata]),
        ("secondary_target", set(data.date), [c for c in data if c.endswith("_1d") or c.endswith("_20d")])]
    for label, dates, columns in scenarios:
        changed = data.copy(deep=True)
        mask = changed.date.isin(dates)
        if not mask.any() or not columns:
            raise ValueError(f"Isolation scenario was not executed: {label}")
        for name in columns:
            if pd.api.types.is_numeric_dtype(changed[name]) and not pd.api.types.is_bool_dtype(changed[name]):
                changed.loc[mask, name] = 1e9 + np.arange(int(mask.sum()))
            else:
                changed[name] = changed[name].astype(object)
                changed.loc[mask, name] = "MUTATED_UNUSED_VALUE"
        rebuilt = build_experiment(changed, manifest, split_protocol, folds)
        equal = all(_same(rebuilt.tables[name], original.tables[name], exact=True) for name in TABLE_NAMES)
        equal &= stability._same_json(rebuilt.summary, original.summary) and rebuilt.fit_records == original.fit_records
        report[label + "_mutation_dependency_violations"] = int(not equal)
    authority = cv_authority(data, manifest, split_protocol, folds)
    # Earlier validations enter later training legitimately. Compare same fold only.
    for fold, (_, valid_dates) in authority.items():
        for label, columns in (("validation_feature", features), ("validation_target", [baseline.PRIMARY_TARGET])):
            changed = data.copy(deep=True)
            for name in columns:
                changed.loc[changed.date.isin(valid_dates), name] = 1e8
            rebuilt = build_experiment(changed, manifest, split_protocol, folds)
            for arm in ARMS:
                for family in FAMILIES:
                    a, b = original.fits[fold, arm, family], rebuilt.fits[fold, arm, family]
                    report[label + "_fit_dependency_violations"] += int(not stability._fit_equal(a, b))
    return report


def _read_inputs(labeled_path: Path, split_dir: Path, diagnostic_dir: Path,
                 baseline_dir: Path, stability_dir: Path) -> tuple[Any, ...]:
    """Verify all frozen checkpoints before loading any development inputs."""
    checkpoint = baseline._read_checkpoint(labeled_path, split_dir, diagnostic_dir)
    data, manifest, split_protocol, folds, inventory, quality = checkpoint
    specs = baseline.feature_specs(data.columns, inventory, quality)
    stage10_protocol = json.loads((baseline_dir / "NVDA_baseline_protocol.json").read_text())
    validation = json.loads((baseline_dir / "NVDA_baseline_validation.json").read_text())
    if validation.get("valid") is not True or any(stage10_protocol.get(k) != v for k, v in baseline.baseline_protocol(specs).items()):
        raise ValueError("Successful frozen Stage 10 protocol/validation required")
    for name in (*baseline.TABLE_NAMES, "protocol"):
        suffix = "json" if name == "protocol" else "csv"
        path = baseline_dir / f"NVDA_baseline_{name}.{suffix}"
        if validation.get("output_artifact_sha256", {}).get(str(path)) != baseline._sha256(path):
            raise ValueError(f"Stage 10 artifact hash mismatch: {path.name}")
    for path, digest in stage10_protocol["input_artifact_sha256"].items():
        if baseline._sha256(Path(path)) != digest:
            raise ValueError("Stage 10 upstream checkpoint changed")
    stage101 = json.loads((stability_dir / "NVDA_stability_validation.json").read_text())
    if stage101.get("valid") is not True or stage101.get("final_test_locked") is not True:
        raise ValueError("Successful locked Stage 10.1 checkpoint required")
    for path, digest in stage101.get("output_artifact_sha256", {}).items():
        if baseline._sha256(Path(path)) != digest:
            raise ValueError("Stage 10.1 artifact hash mismatch")
    # Secondary targets and metadata are loaded only to exercise isolation tests.
    # Close stays a formula input. Model selection always uses explicit columns.
    columns = pd.read_csv(labeled_path, nrows=0).columns
    extra = [c for c in columns if c not in data.columns]
    additional = pd.read_csv(labeled_path, usecols=["date", *extra], dtype={c: str for c in extra if not c.startswith("forward_return_") and c != "close"})
    data = data.merge(additional, on="date", validate="one_to_one")
    reference = {name: pd.read_csv(baseline_dir / f"NVDA_baseline_{name}.csv", float_precision="round_trip") for name in baseline.TABLE_NAMES}
    return data, manifest, split_protocol, folds, reference


def _protected_files(output_dir: Path, roots: set[Path]) -> dict[str, str]:
    paths = {p.resolve() for root in roots for p in root.rglob("*") if p.is_file()
             and output_dir not in p.resolve().parents and p.resolve() != output_dir}
    return {str(p): baseline._sha256(p) for p in sorted(paths)}


def register_protocol(output_dir: Path, contract: dict[str, Any], before: dict[str, str]) -> None:
    """Refuse revised protocols rather than adapting rules after results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("protocol", contract), ("upstream_sha256", before)):
        path = output_dir / f"NVDA_stage11_{name}.json"
        contents = json.dumps(baseline._json_clean(value), indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text() != contents:
            raise ValueError(f"Existing Stage 11 {name} is locked; changes require a separate stage")
        if path.is_symlink():
            raise ValueError("Output symlinks forbidden")
        if not path.exists():
            baseline._atomic_write(path, contents)


def run_pipeline(labeled_path: Path = baseline.RESEARCH_DIR / "NVDA_labeled.csv",
                 split_dir: Path = baseline.RESEARCH_DIR / "splits",
                 diagnostic_dir: Path = baseline.RESEARCH_DIR / "diagnostics",
                 baseline_dir: Path = baseline.RESEARCH_DIR / "modeling",
                 stability_dir: Path = baseline.RESEARCH_DIR / "modeling/stability",
                 output_dir: Path = baseline.RESEARCH_DIR / "modeling/stage11_market_representation",
                 register_only: bool = False) -> dict[str, Any]:
    """Pre-register, fit, audit and publish independent deterministic outputs."""
    labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir, output_dir = (Path(p).resolve() for p in (
        labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir, output_dir))
    inputs = {labeled_path, *(split_dir / f"NVDA_{name}" for name in (
        "split_manifest.csv", "cv_folds.csv", "final_training_pool.csv", "split_summary.csv", "split_validation.json", "evaluation_protocol.json")),
        diagnostic_dir / "feature_inventory.csv", diagnostic_dir / "feature_quality.csv",
        *(baseline_dir / f"NVDA_baseline_{name}.csv" for name in baseline.TABLE_NAMES),
        baseline_dir / "NVDA_baseline_protocol.json", baseline_dir / "NVDA_baseline_validation.json",
        *(p for p in stability_dir.rglob("*") if p.is_file())}
    if any(output_dir == p or output_dir in p.parents for p in inputs):
        raise ValueError("Stage 11 output directory must not contain upstream inputs")
    if output_dir.exists() and any(p.is_symlink() for p in output_dir.rglob("*")):
        raise ValueError("Output symlinks forbidden")
    roots = {baseline.DATA_DIR.resolve(), labeled_path.parent, baseline_dir, stability_dir, split_dir, diagnostic_dir}
    before = _protected_files(output_dir, roots)
    data, manifest, split_protocol, folds, reference = _read_inputs(labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir)
    contract = protocol(split_protocol, before[str(split_dir / "NVDA_split_manifest.csv")], {str(p): before[str(p)] for p in sorted(inputs)})
    register_protocol(output_dir, contract, before)
    if register_only:
        return {"protocol_registered_before_fit": True, "upstream_file_count": len(before), **SAFETY}
    original = data.copy(deep=True)
    experiment = build_experiment(data, manifest, split_protocol, folds)
    report = validate_experiment(data, manifest, split_protocol, folds, experiment, reference, check_sma=True)
    if not report["valid"]:
        raise ValueError(f"Stage 11 validation FAIL: {report['errors']}")
    report.update(isolation_checks(data, manifest, split_protocol, folds, experiment))
    report["mutation_checks_executed"] = True
    report["protocol_registered_before_fit"] = True
    contents = {name: table.to_csv(index=False) for name, table in experiment.tables.items()}
    reloaded = {name: pd.read_csv(StringIO(text), float_precision="round_trip") for name, text in contents.items()}
    reloaded_experiment = Experiment(reloaded, experiment.fits, experiment.fit_records, experiment.summary)
    roundtrip = validate_experiment(data, manifest, split_protocol, folds, reloaded_experiment, reference, check_sma=True)
    report["checks"]["csv_roundtrip_validated"] = roundtrip["valid"]
    report["checks"]["source_dataframe_unchanged"] = data.equals(original)
    after = _protected_files(output_dir, roots)
    report["upstream_mutation_violations"] = sum(before.get(p) != after.get(p) for p in set(before) | set(after))
    report["checks"].update({name + "_zero": report[name] == 0 for name in VIOLATIONS})
    report["errors"] = [name for name, passed in report["checks"].items() if not passed]
    if report["errors"]:
        raise ValueError(f"Stage 11 audit FAIL: {report['errors']}")
    paths = {name: output_dir / f"NVDA_stage11_{name}.csv" for name in TABLE_NAMES}
    summary_path, validation_path = (output_dir / f"NVDA_stage11_{name}.json" for name in ("summary", "validation"))
    experiment.summary.update({"protocol_registered_before_fit": True, "validation_passed": True,
        "control_reproduction_violations": sum(report[name] for name in VIOLATIONS if "control_" in name),
        "upstream_mutation_violations": report["upstream_mutation_violations"],
        **{name: report[name] for name in VIOLATIONS if "dependency" in name}})
    for name, path in paths.items():
        baseline._atomic_write(path, contents[name])
    baseline._atomic_write(summary_path, json.dumps(baseline._json_clean(experiment.summary), indent=2, allow_nan=False) + "\n")
    after = _protected_files(output_dir, roots)
    report.update({"source_sha256_before": before, "source_sha256_after": after,
        "source_artifact_count": len(before), "source_artifacts_unchanged": before == after,
        "fit_records": experiment.fit_records, "comparison_tolerance": {"rtol": 0, "atol": ATOL},
        "output_paths": {**{name: str(path) for name, path in paths.items()}, "summary": str(summary_path), "validation": str(validation_path)},
        "output_artifact_sha256": {str(p): baseline._sha256(p) for p in (*paths.values(), summary_path,
            output_dir / "NVDA_stage11_protocol.json", output_dir / "NVDA_stage11_upstream_sha256.json")},
        "reproducibility": {"randomness": "none", "BLAS_threads": 1, "run_timestamp": "omitted",
            "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__, "numpy_version": np.__version__}})
    report["checks"]["upstream_unchanged_after_publish"] = before == after
    report["errors"] = [name for name, passed in report["checks"].items() if not passed]
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    baseline._atomic_write(validation_path, json.dumps(baseline._json_clean(report), indent=2, allow_nan=False) + "\n")
    if not report["valid"]:
        raise ValueError("Upstream mutation during publication")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("labeled-path", baseline.RESEARCH_DIR / "NVDA_labeled.csv"),
            ("split-dir", baseline.RESEARCH_DIR / "splits"), ("diagnostic-dir", baseline.RESEARCH_DIR / "diagnostics"),
            ("baseline-dir", baseline.RESEARCH_DIR / "modeling"), ("stability-dir", baseline.RESEARCH_DIR / "modeling/stability"),
            ("output-dir", baseline.RESEARCH_DIR / "modeling/stage11_market_representation")):
        parser.add_argument("--" + name, type=Path, default=default)
    parser.add_argument("--register-only", action="store_true", help="Save locked protocol/hash manifest before fitting")
    args = vars(parser.parse_args())
    report = run_pipeline(**args)
    if args["register_only"]:
        print(f"Stage 11 protocol registered before fit; {report['upstream_file_count']} upstream files protected.")
    else:
        summary = json.loads((args["output_dir"] / "NVDA_stage11_summary.json").read_text())
        print(f"Stage 11 validation: {report['validation_status']}; mechanism: {summary['mechanism_assessment']['category']}")
        print("Final Test locked; predictions=false; metrics=false; final_training_pool_used=false.")


if __name__ == "__main__":
    main()
