"""Stage 12: terminal diagnostics of frozen Stage 11 predictions, never a fit."""

import argparse
from dataclasses import dataclass
from io import StringIO
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from nasdaq_research import modeling as baseline
from nasdaq_research import model_stability as stability
from nasdaq_research import market_representation as stage11
from nasdaq_research.config import PROJECT_ROOT

ATOL = 1e-12
SIGN_TOL = TIE_TOL = 1e-15
FOLDS = stability.FOLDS
TABLE_NAMES = ("row_baseline_comparison", "baseline_advantage_summary", "prediction_calibration",
               "prediction_dispersion", "coefficient_stability", "coefficient_similarity",
               "feature_target_associations", "feature_target_stability", "target_windows",
               "target_overlap", "target_dependence", "fold_diagnostics", "extrapolation_context")
SAFETY = {**stage11.SAFETY, "calibrated_predictions_generated": False,
          "calibrated_metrics_computed": False, "new_learned_model_predictions_generated": False}
VIOLATIONS = ("stage11_prediction_consistency_violations", "stage11_coefficient_consistency_violations",
    "new_learned_prediction_rows", "modified_learned_prediction_rows", "baseline_consistency_violations",
    "training_preprocessing_authority_violations", "validation_zscore_reconstruction_violations",
    "diagnostic_integrity_violations", "forbidden_rows_used_violations", "pit_formula_violations",
    "truncation_invariance_violations", "past_sma_oracle_violations", "future_dependency_violations",
    "final_test_feature_mutation_dependency_violations", "final_test_target_mutation_dependency_violations",
    "final_test_metadata_mutation_dependency_violations", "pre_test_gap_mutation_dependency_violations",
    "unlabeled_tail_mutation_dependency_violations", "secondary_target_mutation_dependency_violations",
    "validation_y_directional_dependency_violations", "validation_x_frozen_prediction_violations",
    "calibration_feedback_violations", "upstream_mutation_violations")
PREDICTION_KEYS = ["date", "ticker", "fold", "arm", "model", "prediction", "actual"]


@dataclass
class Inputs:
    data: pd.DataFrame
    manifest: pd.DataFrame
    split_protocol: dict[str, Any]
    folds: pd.DataFrame
    stage11_tables: dict[str, pd.DataFrame]
    stage10_tables: dict[str, pd.DataFrame]
    stage11_protocol: dict[str, Any]


@dataclass
class Diagnostics:
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]


def protocol(inputs: Inputs, hashes: dict[str, str], manifest_hash: str) -> dict[str, Any]:
    """Save this contract before computing any Stage 12 diagnostics."""
    return {"stage": "12", "stage_name": "Post-Representation Linear Signal and Calibration Diagnostics",
        "ticker": "NVDA", "diagnostic_only": True, "primary_specification": "Stage 11 Relative Market",
        "reference_specification": "Stage 11 Original Market; historical reference only",
        "features": list(stage11.TREATMENT), "formula": "Close(t) / SMA_k(t) - 1; k=5/20/60; unchanged",
        "prediction_authority": "NVDA_stage11_oof_predictions.csv; no estimator fitting or prediction calls",
        "coefficient_authority": "NVDA_stage11_coefficients.csv; dropped coefficients remain missing",
        "preprocessing_authority": "NVDA_stage11_feature_manifest.csv; frozen training medians, means and scaler scales",
        "split_authority": "frozen Stage 9.1 manifest/protocol", "split_manifest_sha256": manifest_hash,
        "cv_boundaries": inputs.split_protocol["boundaries"]["cv"],
        "primary_target": baseline.PRIMARY_TARGET, "target_definition": "Close[t+5] / Open[t+1] - 1",
        "feature_timestamp_semantics": "trading day t after close", "ridge_alpha": 1.0,
        "baseline_authority": "Stage 10 row-level OOF predictions and metrics, checked against effective training Y",
        "baseline_definitions": {"zero_return": "prediction=0", "historical_mean": "mean(current effective training primary Y)"},
        "row_loss_definitions": {"AE_model": "abs(actual-prediction)", "AE_zero": "abs(actual)",
            "AE_hist": "abs(actual-historical_mean_prediction)", "delta_AE": "AE_model-AE_baseline"},
        "tie_tolerance": TIE_TOL, "sign_zero_tolerance": SIGN_TOL,
        "bias_definition": "error=prediction-actual; positive mean bias means overprediction; opposite of Stage 11 signed error",
        "dispersion_definition": "sample std ddof=1; std_prediction/std_actual; zero actual std => NaN",
        "calibration_regression_definition": "actual=a+b*prediction; b=sample_cov(prediction,actual)/sample_var(prediction), ddof=1; a=mean(actual)-b*mean(prediction)",
        "undefined_calibration_rule": "n<2 or constant prediction => slope/intercept NaN with reason; no infinity",
        "calibration_usage_restriction": "terminal diagnostic coefficients only; no calibrated prediction or metric may be constructed",
        "coefficient_similarity_definition": "cosine on pairwise actually-used feature intersection; zero norm => NaN; report names/count; never zero-fill dropped features",
        "coefficient_sign_stability": "only all-three-fold used features get sign-consistency counts; missing stays missing",
        "feature_target_association_definition": "Pearson and average-rank Spearman on frozen training-median/scaler transforms for each fold; training and validation separately",
        "undefined_association_rule": "n<2, constant feature/target => NaN with explicit reason; no zero substitution",
        "association_sign_match": "three-category sign at fixed zero tolerance; strict opposite signs are flips; denominator defined pairs only",
        "target_overlap_definition": "inclusive observed-session [entry_date,exit_date]; intersects iff max(entry)<=min(exit); no calendar-day horizon assumption",
        "overlap_denominator": "all unordered distinct within-fold pairs; pooled includes cross-fold pairs, each OOF date once; adjacent pairs reported separately",
        "target_dependence": "Pearson of validation target with its lagged sequence for lags 1..5, n-lag pairs, separately by fold; no significance tests",
        "extrapolation_context": "frozen Stage 11 row max |z| >5 vs <=5; summarize original prediction bias/loss only, including empty strata",
        "primary_predictive_metric": "MAE; retain Stage 11 fixed metrics; no model ranking/selection",
        "conclusion_categories": ["SIGNAL_WITH_CALIBRATION_INSTABILITY", "WEAK_OR_TEMPORALLY_UNSTABLE_SIGNAL", "MIXED_OR_INCONCLUSIVE"],
        "conclusion_framework": {
            "scope": "both Relative OLS and Ridge must agree; descriptive classification only",
            "positive_direction": "all three slopes and Pearson > zero tolerance; at least two positive Spearman folds",
            "majority_advantage": "at least two folds model-beats-zero >0.50; counted folds within one observation (1/25) of 0.50 are inconclusive",
            "calibration_problem": "at least two folds: abs(mean bias)/std_actual>0.25, or dispersion ratio outside [0.75,1.25], or slope outside [0.75,1.25]",
            "signal_category": "positive direction + robust majority advantage + calibration problem for both families; coefficient common-feature sign consistency >=0.50 and all pairwise cosines positive; pooled feature-target sign matches >=0.50 for Pearson and Spearman",
            "weak_category": "at least two robust non-majority folds (beat fraction <=0.50 and distance >1/25), plus slope sign reversal or at least two nonpositive slopes, plus corroborating coefficient or feature-target instability, for both families",
            "coefficient_instability": "common-feature sign consistency <0.50 or at least two nonpositive pairwise cosines",
            "association_instability": "at least two folds have <=0.50 sign matches for both Pearson and Spearman",
            "mixed_category": "all other cases, undefined diagnostics, conflicting families or borderline majority evidence",
            "claim_limit": "no profitable alpha, final model, causal proof, statistically significant calibration or proven generalization"},
        "forbidden_interventions": ["new fit", "new prediction specification", "model/feature selection", "tuning",
            "new representation/formula", "calibration correction", "calibrated metrics", "clipping", "winsorization",
            "alternative scaler/imputation", "target change", "fundamentals", "PCA", "nonlinear model",
            "backtest", "transaction costs", "trading strategy", "final training", "Final Test evaluation"],
        "final_test_lock_rules": "test/gap/tail content excluded; in-memory isolation tests only; date remains manifest join identity",
        **SAFETY, "comparison_tolerance": {"rtol": 0, "atol": ATOL}, "input_artifact_sha256": hashes,
        "limitations": ["75 OOF rows and three 25-row folds with overlapping labels, not independent observations.",
                        "Development CV previously examined; no independent confirmatory evaluation.",
                        "Standardized coefficients across folds use different frozen scales; cosine is descriptive."]}


def signed(value: float) -> float:
    return np.nan if not np.isfinite(value) else 0 if abs(value) <= SIGN_TOL else 1 if value > 0 else -1


def correlation(feature: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    x, y = np.asarray(feature, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite association vectors required")
    reason = "too_few_pairs" if len(x) < 2 else "constant_feature" if np.ptp(x) == 0 else "constant_target" if np.ptp(y) == 0 else "defined"
    if reason != "defined":
        return {"pearson": np.nan, "spearman": np.nan, "undefined_reason": reason}
    return {"pearson": baseline._correlation(y, x),
            "spearman": baseline._correlation(pd.Series(y).rank(method="average").to_numpy(), pd.Series(x).rank(method="average").to_numpy()),
            "undefined_reason": "defined"}


def calibration(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Return a,b only. Never calculate the validation-fitted transformed vector."""
    y, p = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isfinite([y, p]).all():
        raise ValueError("Aligned nonempty finite calibration vectors required")
    error = p - y
    reason = "too_few_rows" if len(p) < 2 else "constant_prediction" if np.ptp(p) == 0 else "defined"
    slope = intercept = np.nan
    if reason == "defined":
        centered_p, centered_y = p - p.mean(), y - y.mean()
        variance = float(centered_p @ centered_p / (len(p) - 1))
        covariance = float(centered_p @ centered_y / (len(p) - 1))
        slope = covariance / variance
        intercept = float(y.mean() - slope * p.mean())
    associations = correlation(p, y)
    return {"n": len(y), "mean_actual": float(y.mean()), "mean_prediction": float(p.mean()),
        "mean_bias": float(error.mean()), "median_error": float(np.median(error)),
        "median_absolute_error": float(np.median(np.abs(error))), "calibration_intercept": intercept,
        "calibration_slope": slope, "calibration_undefined_reason": reason,
        "Pearson": associations["pearson"], "Spearman": associations["spearman"]}


def dispersion(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y, p = np.asarray(actual), np.asarray(prediction)
    actual_std, pred_std = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan, float(np.std(p, ddof=1)) if len(p) > 1 else np.nan
    if len(y) > 1 and np.ptp(y) == 0:
        actual_std = 0.
    if len(p) > 1 and np.ptp(p) == 0:
        pred_std = 0.
    return {"std_actual": actual_std, "std_prediction": pred_std,
        "dispersion_ratio": pred_std / actual_std if actual_std > 0 else np.nan,
        "dispersion_undefined_reason": "defined" if actual_std > 0 else "constant_actual" if actual_std == 0 else "too_few_rows",
        "actual_min": float(y.min()), "actual_max": float(y.max()), **stage11.prediction_summary(p)}


def advantage(delta: np.ndarray, anchor: str) -> dict[str, Any]:
    values = np.asarray(delta, dtype=float)
    ties, beats, losses = int((np.abs(values) <= TIE_TOL).sum()), int((values < -TIE_TOL).sum()), int((values > TIE_TOL).sum())
    return {f"mean_delta_AE_{anchor}": float(values.mean()), f"median_delta_AE_{anchor}": float(np.median(values)),
        f"model_beats_{anchor}_count": beats, f"model_beats_{anchor}_fraction": beats / len(values),
        f"ties_{anchor}_count": ties, f"ties_{anchor}_fraction": ties / len(values),
        f"{anchor}_beats_model_count": losses, f"{anchor}_beats_model_fraction": losses / len(values)}


def frozen_feature_matrices(inputs: Inputs, fold: str, train: list[str], valid: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply saved values directly, without fitting any transformer."""
    keyed = inputs.data.set_index("date")
    usage = inputs.stage11_tables["feature_manifest"].loc[lambda f: f.fold.eq(fold) & f.arm.eq("treatment") & f.model.eq("ols") & f.used.eq(True)].set_index("feature_name")
    result = []
    for days in (train, valid):
        raw = stage11.representation(keyed.loc[days], "treatment").loc[:, usage.index]
        transformed = (raw.fillna(usage.training_median) - usage.training_mean_after_imputation) / usage.scaler_scale
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError("Invalid frozen-state feature reconstruction")
        result.append(transformed)
    return tuple(result)


def coefficient_diagnostics(usage: pd.DataFrame, coefficients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, similarities = [], []
    for model in stage11.FAMILIES:
        scoped = coefficients.loc[coefficients.arm.eq("treatment") & coefficients.model.eq(model)]
        vectors = {fold: scoped.loc[scoped.fold.eq(fold)].set_index("feature_name").coefficient for fold in FOLDS}
        common = set.intersection(*(set(v.index) for v in vectors.values()))
        for name in stage11.TREATMENT:
            all_signs = [signed(vectors[fold].get(name, np.nan)) for fold in FOLDS]
            is_common = name in common
            counts = {f"{label}_fold_count": int(sum(value == sign for value in all_signs)) if is_common else np.nan
                      for label, sign in (("positive", 1), ("negative", -1), ("zero", 0))}
            for fold, sign in zip(FOLDS, all_signs):
                record = usage.loc[usage.arm.eq("treatment") & usage.model.eq(model) & usage.fold.eq(fold) & usage.feature_name.eq(name)].iloc[0]
                value = float(vectors[fold].get(name, np.nan))
                intercept = float(scoped.loc[scoped.fold.eq(fold), "intercept"].iloc[0])
                rows.append({"arm": "treatment", "model": model, "fold": fold, "feature_name": name,
                    "used": bool(record.used), "drop_reason": record.drop_reason, "coefficient": value,
                    "intercept": intercept if record.used else np.nan, "sign": sign, "absolute_coefficient": abs(value),
                    "common_all_three_folds": is_common, **counts,
                    "sign_consistent_all_folds": len(set(all_signs)) == 1 if is_common else None})
        for a, b in itertools.combinations(FOLDS, 2):
            names = [name for name in stage11.TREATMENT if name in vectors[a].index and name in vectors[b].index]
            left, right = vectors[a].loc[names].to_numpy(), vectors[b].loc[names].to_numpy()
            norms = float(np.linalg.norm(left)), float(np.linalg.norm(right))
            cosine = float(left @ right / (norms[0] * norms[1])) if all(norm > 0 for norm in norms) else np.nan
            similarities.append({"arm": "treatment", "model": model, "fold_a": a, "fold_b": b,
                "n_common_features": len(names), "common_feature_names": json.dumps(names, separators=(",", ":")),
                "cosine_similarity": cosine, "undefined_reason": "defined" if np.isfinite(cosine) else "zero_norm_or_no_common_features"})
    return pd.DataFrame(rows), pd.DataFrame(similarities)


def association_diagnostics(inputs: Inputs, authority: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = inputs.data.set_index("date")
    rows = []
    for fold, (train, valid) in authority.items():
        matrices = frozen_feature_matrices(inputs, fold, train, valid)
        for name in matrices[0]:
            train_info = correlation(matrices[0][name].to_numpy(), baseline._target(keyed.loc[train, baseline.PRIMARY_TARGET]))
            valid_info = correlation(matrices[1][name].to_numpy(), baseline._target(keyed.loc[valid, baseline.PRIMARY_TARGET]))
            row = {"fold": fold, "feature_name": name, "n_train": len(train), "n_validation": len(valid),
                   "train_undefined_reason": train_info["undefined_reason"], "validation_undefined_reason": valid_info["undefined_reason"]}
            for metric in ("pearson", "spearman"):
                a, b = train_info[metric], valid_info[metric]
                defined = np.isfinite([a, b]).all()
                row.update({f"train_{metric}": a, f"validation_{metric}": b,
                    f"{metric}_sign_match": signed(a) == signed(b) if defined else None,
                    f"{metric}_sign_flip": signed(a) * signed(b) == -1 if defined else None,
                    f"absolute_change_{metric}": abs(a-b) if defined else np.nan})
            rows.append(row)
    associations = pd.DataFrame(rows)
    stability_rows = []
    for fold in (*FOLDS, "pooled"):
        selected = associations if fold == "pooled" else associations.loc[associations.fold.eq(fold)]
        row = {"fold": fold, "n_used_feature_fold_pairs": len(selected)}
        for metric in ("pearson", "spearman"):
            flags = selected[f"{metric}_sign_match"].dropna().astype(bool)
            flips = selected[f"{metric}_sign_flip"].dropna().astype(bool)
            row.update({f"{metric}_defined_pair_count": len(flags), f"{metric}_sign_match_count": int(flags.sum()),
                f"{metric}_sign_mismatch_count": int((~flags).sum()), f"{metric}_sign_flip_count": int(flips.sum()),
                f"{metric}_sign_match_fraction": float(flags.mean()) if len(flags) else np.nan,
                f"mean_absolute_change_{metric}": float(selected[f"absolute_change_{metric}"].mean())})
        stability_rows.append(row)
    return associations, pd.DataFrame(stability_rows)


def overlap_summary(windows: pd.DataFrame) -> dict[str, Any]:
    """Closed date intervals; all unordered distinct windows are the denominator."""
    items = list(windows[["entry_date", "exit_date"]].itertuples(index=False, name=None))
    overlap = lambda a, b: max(a[0], b[0]) <= min(a[1], b[1])
    pairs = len(items) * (len(items)-1) // 2
    count = sum(overlap(a, b) for a, b in itertools.combinations(items, 2))
    adjacent_count = sum(overlap(a, b) for a, b in zip(items, items[1:]))
    return {"n_windows": len(items), "pair_denominator": pairs, "overlapping_pair_count": count,
        "overlapping_pair_fraction": count / pairs if pairs else np.nan,
        "adjacent_pair_denominator": max(0, len(items)-1), "adjacent_overlapping_pair_count": adjacent_count,
        "adjacent_overlapping_fraction": adjacent_count / (len(items)-1) if len(items) > 1 else np.nan}


def target_diagnostics(inputs: Inputs, authority: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, dependence = [], []
    keyed = inputs.data.set_index("date")
    for fold, (_, dates) in authority.items():
        manifest = inputs.manifest.loc[inputs.manifest.fold.eq(fold) & inputs.manifest.target_name.eq(baseline.PRIMARY_TARGET)].set_index("date")
        for i, day in enumerate(dates):
            entry, exit_date = manifest.loc[day, ["target_entry_date", "target_exit_date"]]
            next_day = dates[i+1] if i+1 < len(dates) else None
            next_overlap = (max(entry, manifest.loc[next_day, "target_entry_date"]) <= min(exit_date, manifest.loc[next_day, "target_exit_date"])) if next_day else None
            rows.append({"date": day, "fold": fold, "entry_date": entry, "exit_date": exit_date,
                         "next_validation_date": next_day, "overlaps_next_window": next_overlap})
        y = baseline._target(keyed.loc[dates, baseline.PRIMARY_TARGET])
        for lag in range(1, 6):
            info = correlation(y[:-lag], y[lag:])
            count = sum(max(manifest.loc[a, "target_entry_date"], manifest.loc[b, "target_entry_date"])
                        <= min(manifest.loc[a, "target_exit_date"], manifest.loc[b, "target_exit_date"])
                        for a, b in zip(dates[:-lag], dates[lag:]))
            dependence.append({"fold": fold, "lag_observations": lag, "pair_count": len(y)-lag,
                "target_autocorrelation": info["pearson"], "undefined_reason": info["undefined_reason"],
                "overlapping_lag_pair_count": count, "overlapping_lag_pair_fraction": count / (len(y)-lag)})
    windows = pd.DataFrame(rows)
    summaries = [{"fold": fold, **overlap_summary(windows if fold == "pooled" else windows.loc[windows.fold.eq(fold)])}
                 for fold in (*FOLDS, "pooled")]
    return windows, pd.DataFrame(summaries), pd.DataFrame(dependence)


def assess(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Conservative pre-registered synthesis; borderline/conflicting cases stay mixed."""
    fold_rows = tables["fold_diagnostics"]
    signs = tables["coefficient_stability"].drop_duplicates(["model", "feature_name"])
    similarity = tables["coefficient_similarity"]
    assoc = tables["feature_target_stability"].set_index("fold")
    association_support = all(assoc.loc["pooled", f"{metric}_sign_match_fraction"] >= .5 for metric in ("pearson", "spearman"))
    association_instability = sum(all(assoc.loc[fold, f"{metric}_sign_match_fraction"] <= .5 for metric in ("pearson", "spearman")) for fold in FOLDS) >= 2
    evidence = []
    for model in stage11.FAMILIES:
        rows = fold_rows.loc[fold_rows.model.eq(model)]
        slopes, pearson, spearman = rows.calibration_slope.to_numpy(), rows.Pearson.to_numpy(), rows.Spearman.to_numpy()
        beat = rows.model_beats_zero_fraction.to_numpy()
        near = np.abs(beat-.5) <= 1/25 + SIGN_TOL
        robust_majority = int(((beat > .5) & ~near).sum())
        robust_nonmajority = int(((beat <= .5) & ~near).sum())
        positive_direction = bool(np.all(slopes > SIGN_TOL) and np.all(pearson > SIGN_TOL) and (spearman > SIGN_TOL).sum() >= 2)
        slope_signs = [signed(value) for value in slopes]
        direction_unstable = (1 in slope_signs and -1 in slope_signs) or sum(value <= 0 for value in slope_signs) >= 2
        calibration_problem_count = int(((rows.mean_bias.abs() / rows.std_actual > .25)
            | ~rows.dispersion_ratio.between(.75, 1.25) | ~rows.calibration_slope.between(.75, 1.25)).sum())
        common = signs.loc[signs.model.eq(model) & signs.common_all_three_folds.eq(True)]
        sign_fraction = float(common.sign_consistent_all_folds.astype(bool).mean()) if len(common) else np.nan
        cosines = similarity.loc[similarity.model.eq(model), "cosine_similarity"]
        coefficient_support = sign_fraction >= .5 and cosines.gt(0).all()
        coefficient_unstable = sign_fraction < .5 or cosines.le(0).sum() >= 2
        defined = np.isfinite(rows[["calibration_slope", "Pearson", "Spearman", "dispersion_ratio"]].to_numpy()).all()
        signal = bool(defined and positive_direction and robust_majority >= 2 and calibration_problem_count >= 2
                      and coefficient_support and association_support)
        weak = bool(defined and robust_nonmajority >= 2 and direction_unstable and (coefficient_unstable or association_instability))
        evidence.append({"model": model, "positive_validation_association_direction": positive_direction,
            "robust_majority_advantage_fold_count": robust_majority, "robust_nonmajority_fold_count": robust_nonmajority,
            "near_majority_boundary_folds": rows.loc[near, "fold"].tolist(),
            "calibration_problem_fold_count": calibration_problem_count, "slope_direction_unstable": bool(direction_unstable),
            "common_feature_sign_consistent_fraction": sign_fraction, "coefficient_direction_support": bool(coefficient_support),
            "coefficient_instability": bool(coefficient_unstable), "signal_with_calibration_gate": signal, "weak_signal_gate": weak})
    category = "SIGNAL_WITH_CALIBRATION_INSTABILITY" if all(r["signal_with_calibration_gate"] for r in evidence) else "WEAK_OR_TEMPORALLY_UNSTABLE_SIGNAL" if all(r["weak_signal_gate"] for r in evidence) else "MIXED_OR_INCONCLUSIVE"
    return {"category": category, "family_evidence": evidence, "feature_association_support": bool(association_support),
            "feature_association_instability": bool(association_instability), "model_selected": None,
            "interpretation": "Descriptive development-CV synthesis only; not proof of alpha, causality or generalization."}


def build_diagnostics(inputs: Inputs) -> Diagnostics:
    """Read fixed predictions; X/Y mutations may change diagnostics, never predictions."""
    authority = stage11.cv_authority(inputs.data, inputs.manifest, inputs.split_protocol, inputs.folds)
    keyed = inputs.data.set_index("date")
    source = inputs.stage11_tables["oof_predictions"]
    rows = source[PREDICTION_KEYS].copy(deep=True)
    anchors = inputs.stage10_tables["oof_predictions"][["date", "fold", "zero_return_prediction", "historical_mean_prediction"]]
    rows = rows.merge(anchors, on=["date", "fold"], how="left", validate="many_to_one", sort=False)
    rows["actual"] = baseline._target(keyed.loc[rows.date, baseline.PRIMARY_TARGET])
    rows["error"] = rows.prediction - rows.actual
    rows["AE_model"] = rows.error.abs()
    rows["AE_zero"] = (rows.actual - rows.zero_return_prediction).abs()
    rows["AE_hist"] = (rows.actual - rows.historical_mean_prediction).abs()
    for anchor in ("zero", "hist"):
        delta = rows.AE_model - rows[f"AE_{anchor}"]
        rows[f"delta_AE_{anchor}"] = delta
        rows[f"comparison_{anchor}"] = np.where(delta < -TIE_TOL, "model_beats_baseline", np.where(delta > TIE_TOL, "baseline_beats_model", "tie"))
    advantage_rows, calibration_rows, dispersion_rows, context_rows = [], [], [], []
    for fold in (*FOLDS, "pooled"):
        for arm in stage11.ARMS:
            for model in stage11.FAMILIES:
                selected = rows.loc[rows.arm.eq(arm) & rows.model.eq(model) & (True if fold == "pooled" else rows.fold.eq(fold))]
                tags = {"fold": fold, "arm": arm, "model": model}
                advantage_rows.append({**tags, "n": len(selected), **advantage(selected.delta_AE_zero, "zero"), **advantage(selected.delta_AE_hist, "hist")})
                if fold != "pooled":
                    calibration_rows.append({**tags, **calibration(selected.actual.to_numpy(), selected.prediction.to_numpy())})
                    dispersion_rows.append({**tags, **dispersion(selected.actual.to_numpy(), selected.prediction.to_numpy())})
    coeff, similarity = coefficient_diagnostics(inputs.stage11_tables["feature_manifest"], inputs.stage11_tables["coefficients"])
    associations, assoc_stability = association_diagnostics(inputs, authority)
    windows, overlap, dependence = target_diagnostics(inputs, authority)
    tables = {"row_baseline_comparison": rows, "baseline_advantage_summary": pd.DataFrame(advantage_rows),
        "prediction_calibration": pd.DataFrame(calibration_rows), "prediction_dispersion": pd.DataFrame(dispersion_rows),
        "coefficient_stability": coeff, "coefficient_similarity": similarity,
        "feature_target_associations": associations, "feature_target_stability": assoc_stability,
        "target_windows": windows, "target_overlap": overlap, "target_dependence": dependence}
    fold_rows = tables["prediction_calibration"].loc[lambda f: f.arm.eq("treatment")].merge(
        tables["prediction_dispersion"], on=["fold", "arm", "model"], validate="one_to_one")
    fold_rows = fold_rows.merge(tables["baseline_advantage_summary"], on=["fold", "arm", "model", "n"], validate="one_to_one")
    metrics = inputs.stage11_tables["fold_metrics"].loc[lambda f: f.arm.eq("treatment")]
    fold_rows = fold_rows.merge(metrics[["fold", "arm", "model", "MAE", "RMSE", "R2", "directional_accuracy"]], on=["fold", "arm", "model"], validate="one_to_one")
    for fold, (_, valid) in authority.items():
        for model in stage11.FAMILIES:
            selected = rows.loc[rows.arm.eq("treatment") & rows.model.eq(model) & rows.fold.eq(fold)].set_index("date")
            z = inputs.stage11_tables["zscores"].loc[lambda f: f.arm.eq("treatment") & f.model.eq(model) & f.fold.eq(fold)]
            maxima = z.groupby("date").abs_validation_zscore.max().loc[valid]
            for label, mask in (("abs_z_gt_5", maxima > 5), ("abs_z_le_5", maxima <= 5)):
                subset = selected.loc[maxima.index[mask]]
                context_rows.append({"fold": fold, "arm": "treatment", "model": model, "frozen_z_stratum": label,
                    "n": len(subset), "mean_bias": float(subset.error.mean()), "MAE": float(subset.AE_model.mean()),
                    "mean_delta_AE_zero": float(subset.delta_AE_zero.mean()), "mean_abs_prediction": float(subset.prediction.abs().mean())})
    tables["fold_diagnostics"] = fold_rows
    tables["extrapolation_context"] = pd.DataFrame(context_rows)
    tables = {name: tables[name] for name in TABLE_NAMES}
    summary = {"stage": "12", "ticker": "NVDA", "diagnostic_only": True, **SAFETY,
        "primary_target": baseline.PRIMARY_TARGET, "unique_oof_dates": int(rows.date.nunique()),
        "new_learned_prediction_rows": 0, "modified_learned_prediction_rows": 0,
        "relative_fold_diagnostics": fold_rows.to_dict("records"),
        "baseline_advantage": tables["baseline_advantage_summary"].loc[lambda f: f.arm.eq("treatment")].to_dict("records"),
        "coefficient_similarity": similarity.to_dict("records"), "feature_target_stability": assoc_stability.to_dict("records"),
        "target_overlap": overlap.to_dict("records"), "target_dependence": dependence.to_dict("records"),
        "integrated_mechanism_assessment": assess(tables)}
    return Diagnostics(tables, summary)


def _same(a: pd.DataFrame, b: pd.DataFrame, exact: bool = False) -> bool:
    # CSV uses NaN for terminal undefined flags; memory may use None/pd.NA.
    # Normalize sentinels only, preserving missing positions and every value.
    a, b = a.copy().reset_index(drop=True), b.copy().reset_index(drop=True)
    if list(a.columns) != list(b.columns) or not a.isna().equals(b.isna()):
        return False
    for name in a:
        if a[name].isna().any():
            a[name] = a[name].astype(object).mask(a[name].isna(), np.nan)
            b[name] = b[name].astype(object).mask(b[name].isna(), np.nan)
    return stage11._same(a, b, exact=exact)


def validate_diagnostics(inputs: Inputs, diagnostics: Diagnostics, *, check_sma: bool = False) -> dict[str, Any]:
    """Consistency audit, independent source/statistic oracles and terminal output integrity."""
    report = {name: 0 for name in VIOLATIONS}
    report.update({"stage": "12", **SAFETY, "checks": {}})
    checks = report["checks"]
    authority = stage11.cv_authority(inputs.data, inputs.manifest, inputs.split_protocol, inputs.folds)
    keyed = inputs.data.set_index("date")
    rows = diagnostics.tables["row_baseline_comparison"]
    expected = inputs.stage11_tables["oof_predictions"][PREDICTION_KEYS]
    projected = rows[PREDICTION_KEYS]
    checks["table_names_exact"] = tuple(diagnostics.tables) == TABLE_NAMES
    report["stage11_prediction_consistency_violations"] = int(not _same(projected, expected, exact=True))
    report["prediction_projected_csv_byte_equality"] = projected.to_csv(index=False) == expected.to_csv(index=False)
    report["new_learned_prediction_rows"] = len(set(map(tuple, projected[["date", "fold", "arm", "model"]].to_numpy())) - set(map(tuple, expected[["date", "fold", "arm", "model"]].to_numpy())))
    report["modified_learned_prediction_rows"] = int((~np.isclose(projected.prediction, expected.prediction, rtol=0, atol=ATOL)).sum()) if len(projected) == len(expected) else len(projected)
    coeff = diagnostics.tables["coefficient_stability"].loc[lambda f: f.used.eq(True), ["arm", "model", "fold", "feature_name", "coefficient", "intercept"]]
    reference = inputs.stage11_tables["coefficients"].loc[lambda f: f.arm.eq("treatment"), coeff.columns]
    order = ["model", "feature_name", "fold"]
    coeff, reference = (f.sort_values(order, kind="stable").reset_index(drop=True) for f in (coeff, reference))
    report["stage11_coefficient_consistency_violations"] = int(not _same(coeff, reference, exact=True))
    report["coefficient_projected_csv_byte_equality"] = coeff.to_csv(index=False) == reference.to_csv(index=False)
    report.update(stage11.pit_checks(inputs.data, authority, check_sma=check_sma))
    checks["prediction_schema_membership"] = (len(rows) == 300 and not rows.duplicated(["date", "arm", "model"]).any()
        and rows.date.nunique() == 75 and np.isfinite(rows[["actual", "prediction", "historical_mean_prediction"]].to_numpy()).all())
    checks["terminal_safety_state"] = all(diagnostics.summary.get(k) == v for k, v in SAFETY.items())
    allowed = {day for train, valid in authority.values() for day in (*train, *valid)}
    used_dates = set(rows.date) | set(diagnostics.tables["target_windows"].date)
    report["forbidden_rows_used_violations"] = len(used_dates - allowed)
    for fold, (train, valid) in authority.items():
        raw = stage11.representation(keyed.loc[train], "treatment")
        usage = inputs.stage11_tables["feature_manifest"].loc[lambda f: f.arm.eq("treatment") & f.fold.eq(fold)]
        for row in usage.itertuples():
            stats = baseline._audit_training_stats(raw[row.feature_name])
            report["training_preprocessing_authority_violations"] += int(row.training_row_count != len(train)
                or row.drop_reason != stats["drop_reason"] or row.used != (stats["drop_reason"] == "used")
                or any(not np.isclose(getattr(row, name), value, rtol=0, atol=ATOL, equal_nan=True) for name, value in stats.items() if name != "drop_reason"))
        matrices = frozen_feature_matrices(inputs, fold, train, valid)
        for model in stage11.FAMILIES:
            model_usage = usage.loc[usage.model.eq(model)]
            ols_usage = usage.loc[usage.model.eq("ols")]
            checks[f"{fold}_{model}_same_frozen_preprocessing"] = _same(model_usage.drop(columns="model"), ols_usage.drop(columns="model"), exact=True)
            for name in matrices[1]:
                z = inputs.stage11_tables["zscores"].loc[lambda f: f.fold.eq(fold) & f.arm.eq("treatment") & f.model.eq(model) & f.feature_name.eq(name)]
                report["validation_zscore_reconstruction_violations"] += int(z.date.tolist() != valid or not np.allclose(z.validation_zscore, matrices[1][name], rtol=0, atol=ATOL))
        expected_mean = baseline._target(keyed.loc[train, baseline.PRIMARY_TARGET]).mean()
        anchors = inputs.stage10_tables["oof_predictions"].loc[lambda f: f.fold.eq(fold)]
        report["baseline_consistency_violations"] += int(anchors.date.tolist() != valid or not anchors.zero_return_prediction.eq(0).all()
            or not np.allclose(anchors.historical_mean_prediction, expected_mean, rtol=0, atol=ATOL))
        actual = baseline._target(keyed.loc[valid, baseline.PRIMARY_TARGET])
        for anchor in ("zero_return", "historical_mean"):
            metrics = baseline.regression_metrics(actual, anchors[f"{anchor}_prediction"].to_numpy())
            saved = inputs.stage10_tables["cv_metrics"].loc[lambda f: f.fold.eq(fold) & f.model_name.eq(anchor)].iloc[0]
            report["baseline_consistency_violations"] += int(any(not np.isclose(saved[name], value, rtol=0, atol=ATOL, equal_nan=True) for name, value in metrics.items()))
        # Independent scalar closed-form calibration oracle, not sklearn.
        for arm in stage11.ARMS:
            for model in stage11.FAMILIES:
                selected = rows.loc[rows.fold.eq(fold) & rows.arm.eq(arm) & rows.model.eq(model)]
                p, y = selected.prediction.to_numpy(), selected.actual.to_numpy()
                mean_p, mean_y = sum(map(float, p))/len(p), sum(map(float, y))/len(y)
                denominator = sum((float(value)-mean_p)**2 for value in p)
                slope = sum((float(a)-mean_p)*(float(b)-mean_y) for a, b in zip(p, y))/denominator if np.ptp(p) else np.nan
                saved = diagnostics.tables["prediction_calibration"].loc[lambda f: f.fold.eq(fold) & f.arm.eq(arm) & f.model.eq(model)].iloc[0]
                report["diagnostic_integrity_violations"] += int(not np.isclose(saved.calibration_slope, slope, rtol=0, atol=ATOL, equal_nan=True)
                    or not np.isclose(saved.calibration_intercept, mean_y-slope*mean_p, rtol=0, atol=ATOL, equal_nan=True))
    # Pure builder repeats no fitting or prediction. Numeric oracles above and
    # tests below are independent; this also checks every serialized output.
    rebuilt = build_diagnostics(inputs)
    for name in TABLE_NAMES:
        checks[name + "_integrity"] = _same(diagnostics.tables[name], rebuilt.tables[name])
    checks["summary_integrity"] = stability._same_json(diagnostics.summary, rebuilt.summary)
    for name in VIOLATIONS:
        checks[name + "_zero"] = report[name] == 0
    report["errors"] = [name for name, passed in checks.items() if not passed]
    report["valid"] = not report["errors"]
    return report


def _changed_inputs(inputs: Inputs, data: pd.DataFrame) -> Inputs:
    return Inputs(data, inputs.manifest, inputs.split_protocol, inputs.folds, inputs.stage11_tables, inputs.stage10_tables, inputs.stage11_protocol)


def isolation_checks(inputs: Inputs, original: Diagnostics) -> dict[str, Any]:
    report = {name: 0 for name in VIOLATIONS if "dependency" in name or "feedback" in name or "validation_x" in name}
    data = inputs.data
    primary = inputs.manifest.loc[inputs.manifest.fold.eq("final") & inputs.manifest.target_name.eq(baseline.PRIMARY_TARGET)]
    features = [*stage11.CONTROL, "close"]
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
            raise ValueError("Isolation mutation was not executed")
        for name in columns:
            if pd.api.types.is_numeric_dtype(changed[name]) and not pd.api.types.is_bool_dtype(changed[name]):
                changed.loc[mask, name] = 1000000000 + np.arange(int(mask.sum()))
            else:
                changed[name] = changed[name].astype(object)
                changed.loc[mask, name] = "MUTATED_UNUSED_PAYLOAD"
        rebuilt = build_diagnostics(_changed_inputs(inputs, changed))
        same = all(_same(original.tables[name], rebuilt.tables[name], exact=True) for name in TABLE_NAMES)
        report[label + "_mutation_dependency_violations"] = int(not same or not stability._same_json(original.summary, rebuilt.summary))
    directional = []
    authority = stage11.cv_authority(data, inputs.manifest, inputs.split_protocol, inputs.folds)
    saved_preprocessing = inputs.stage11_tables["feature_manifest"].copy(deep=True)
    for fold, (train, valid) in authority.items():
        for axis in ("y", "x"):
            changed = data.copy(deep=True)
            mask = changed.date.isin(valid)
            if axis == "y":
                changed.loc[mask, baseline.PRIMARY_TARGET] = 10 + np.linspace(-1, 1, len(valid))**3
            else:
                changed.loc[mask, "close"] = 1000000 + np.arange(len(valid))**2
            rebuilt = build_diagnostics(_changed_inputs(inputs, changed))
            frozen = all(_same(original.tables[name], rebuilt.tables[name], exact=True) for name in ("coefficient_stability", "coefficient_similarity"))
            pred_fields = [c for c in PREDICTION_KEYS if c != "actual"]
            frozen &= _same(original.tables["row_baseline_comparison"][pred_fields], rebuilt.tables["row_baseline_comparison"][pred_fields], exact=True)
            preprocessing_unchanged = _same(saved_preprocessing, inputs.stage11_tables["feature_manifest"], exact=True)
            before_x = frozen_feature_matrices(inputs, fold, train, valid)
            after_x = frozen_feature_matrices(_changed_inputs(inputs, changed), fold, train, valid)
            feature_values_unchanged = _same(before_x[0], after_x[0], exact=True)
            if axis == "y":
                feature_values_unchanged &= _same(before_x[1], after_x[1], exact=True)
            frozen &= preprocessing_unchanged and feature_values_unchanged
            changed_association = not _same(original.tables["feature_target_associations"].loc[lambda f: f.fold.eq(fold)], rebuilt.tables["feature_target_associations"].loc[lambda f: f.fold.eq(fold)], exact=True)
            if axis == "y":
                changed_calibration = not _same(original.tables["prediction_calibration"].loc[lambda f: f.fold.eq(fold)], rebuilt.tables["prediction_calibration"].loc[lambda f: f.fold.eq(fold)], exact=True)
                changed_errors = not _same(original.tables["row_baseline_comparison"].loc[lambda f: f.fold.eq(fold), ["AE_model"]], rebuilt.tables["row_baseline_comparison"].loc[lambda f: f.fold.eq(fold), ["AE_model"]], exact=True)
                passed = frozen and changed_association and changed_calibration and changed_errors
                report["validation_y_directional_dependency_violations"] += int(not passed)
            else:
                passed = frozen and changed_association and _same(original.tables["row_baseline_comparison"], rebuilt.tables["row_baseline_comparison"], exact=True)
                report["validation_x_frozen_prediction_violations"] += int(not passed)
            directional.append({"fold": fold, "mutation_axis": axis, "diagnostics_changed_as_expected": bool(passed),
                "frozen_prediction_coefficient_state_unchanged": bool(frozen),
                "frozen_preprocessing_unchanged": bool(preprocessing_unchanged),
                "same_fold_training_features_unchanged": bool(feature_values_unchanged)})
    # Replace the terminal calibration table with grossly different a,b via
    # the function boundary. This never feeds row losses or original metrics.
    from unittest.mock import patch
    original_calibration = calibration
    def adversarial_calibration(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        return {**original_calibration(y, p), "calibration_intercept": 1e9, "calibration_slope": -1e9}
    with patch(__name__ + ".calibration", side_effect=adversarial_calibration):
        rebuilt = build_diagnostics(inputs)
    preserved = all(_same(original.tables[name], rebuilt.tables[name], exact=True) for name in (
        "row_baseline_comparison", "baseline_advantage_summary", "coefficient_stability", "prediction_dispersion"))
    preserved &= _same(original.tables["fold_diagnostics"][["fold", "model", "MAE", "RMSE"]], rebuilt.tables["fold_diagnostics"][["fold", "model", "MAE", "RMSE"]], exact=True)
    report["calibration_feedback_violations"] = int(not preserved)
    report["directional_mutation_checks"] = directional
    report["calibration_non_feedback_checked"] = True
    return report


def read_inputs(labeled_path: Path, split_dir: Path, diagnostic_dir: Path, baseline_dir: Path,
                stability_dir: Path, stage11_dir: Path) -> Inputs:
    data, manifest, split_protocol, folds, reference = stage11._read_inputs(labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir)
    validation = json.loads((stage11_dir / "NVDA_stage11_validation.json").read_text())
    contract = json.loads((stage11_dir / "NVDA_stage11_protocol.json").read_text())
    if validation.get("valid") is not True or any(validation.get(k) != v for k, v in stage11.SAFETY.items()):
        raise ValueError("Successful locked Stage 11 validation required")
    expected = stage11.protocol(split_protocol, baseline._sha256(split_dir / "NVDA_split_manifest.csv"), contract.get("input_artifact_sha256", {}))
    if contract != expected:
        raise ValueError("Frozen Stage 11 protocol changed")
    for path, digest in contract["input_artifact_sha256"].items():
        if baseline._sha256(Path(path)) != digest:
            raise ValueError("Stage 11 input checkpoint changed")
    for name in (*stage11.TABLE_NAMES, "protocol", "summary", "upstream_sha256"):
        suffix = "csv" if name in stage11.TABLE_NAMES else "json"
        path = stage11_dir / f"NVDA_stage11_{name}.{suffix}"
        if validation.get("output_artifact_sha256", {}).get(str(path)) != baseline._sha256(path):
            raise ValueError(f"Stage 11 artifact hash mismatch: {path.name}")
    tables = {name: pd.read_csv(stage11_dir / f"NVDA_stage11_{name}.csv", float_precision="round_trip") for name in stage11.TABLE_NAMES}
    return Inputs(data, manifest, split_protocol, folds, tables, reference, contract)


def protected_files(output_dir: Path, roots: set[Path]) -> dict[str, str]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT_ROOT).decode().split("\0")
    paths = {PROJECT_ROOT / name for name in tracked if name and (PROJECT_ROOT / name).is_file()}
    paths.update(p.resolve() for root in roots for p in root.rglob("*") if p.is_file())
    stage12_sources = {Path(__file__).resolve(), PROJECT_ROOT / "tests/test_signal_diagnostics.py", PROJECT_ROOT / "docs/stage12_execution_report.md"}
    paths = {p for p in paths if output_dir not in p.parents and p != output_dir and p not in stage12_sources
             and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    return {str(p): baseline._sha256(p) for p in sorted(paths)}


def register(output_dir: Path, contract: dict[str, Any], upstream: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("protocol", contract), ("upstream_sha256", upstream)):
        path = output_dir / f"NVDA_stage12_{name}.json"
        contents = json.dumps(baseline._json_clean(payload), indent=2, allow_nan=False) + "\n"
        if path.is_symlink():
            raise ValueError("Output symlinks forbidden")
        if path.exists() and path.read_text() != contents:
            raise ValueError(f"Existing Stage 12 {name} is locked")
        if not path.exists():
            baseline._atomic_write(path, contents)


def run_pipeline(labeled_path: Path = baseline.RESEARCH_DIR / "NVDA_labeled.csv", split_dir: Path = baseline.RESEARCH_DIR / "splits",
                 diagnostic_dir: Path = baseline.RESEARCH_DIR / "diagnostics", baseline_dir: Path = baseline.RESEARCH_DIR / "modeling",
                 stability_dir: Path = baseline.RESEARCH_DIR / "modeling/stability",
                 stage11_dir: Path = baseline.RESEARCH_DIR / "modeling/stage11_market_representation",
                 output_dir: Path = baseline.RESEARCH_DIR / "modeling/stage12_signal_diagnostics", register_only: bool = False) -> dict[str, Any]:
    paths = [Path(p).resolve() for p in (labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir, stage11_dir, output_dir)]
    labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir, stage11_dir, output_dir = paths
    inputs_paths = {labeled_path, *(p for root in (split_dir, diagnostic_dir, stability_dir, stage11_dir) for p in root.rglob("*") if p.is_file()),
        *(baseline_dir / f"NVDA_baseline_{name}.csv" for name in baseline.TABLE_NAMES),
        baseline_dir / "NVDA_baseline_protocol.json", baseline_dir / "NVDA_baseline_validation.json"}
    if any(output_dir == p or output_dir in p.parents for p in inputs_paths):
        raise ValueError("Stage 12 output must not contain upstream inputs")
    if output_dir.exists() and any(p.is_symlink() for p in output_dir.rglob("*")):
        raise ValueError("Output symlinks forbidden")
    roots = {baseline.DATA_DIR.resolve(), labeled_path.parent, split_dir, diagnostic_dir, baseline_dir, stability_dir, stage11_dir}
    before = protected_files(output_dir, roots)
    inputs = read_inputs(labeled_path, split_dir, diagnostic_dir, baseline_dir, stability_dir, stage11_dir)
    contract = protocol(inputs, {str(p): before[str(p)] for p in sorted(inputs_paths)}, before[str(split_dir / "NVDA_split_manifest.csv")])
    register(output_dir, contract, before)
    if register_only:
        return {"protocol_registered_before_diagnostics": True, "upstream_file_count": len(before), **SAFETY}
    original_data = inputs.data.copy(deep=True)
    source_tables = {name: table.copy(deep=True) for name, table in inputs.stage11_tables.items()}
    diagnostics = build_diagnostics(inputs)
    report = validate_diagnostics(inputs, diagnostics, check_sma=True)
    if not report["valid"]:
        raise ValueError(f"Stage 12 consistency validation FAIL: {report['errors']}")
    report.update(isolation_checks(inputs, diagnostics))
    text = {name: table.to_csv(index=False) for name, table in diagnostics.tables.items()}
    reloaded = {name: pd.read_csv(StringIO(value), float_precision="round_trip") for name, value in text.items()}
    roundtrip = validate_diagnostics(inputs, Diagnostics(reloaded, diagnostics.summary), check_sma=True)
    report["checks"].update({"csv_roundtrip_validated": roundtrip["valid"], "source_dataframe_unchanged": inputs.data.equals(original_data),
        "stage11_input_tables_unchanged": all(_same(source_tables[name], inputs.stage11_tables[name], exact=True) for name in source_tables)})
    report["csv_roundtrip_validation_errors"] = roundtrip["errors"]
    after = protected_files(output_dir, roots)
    report["upstream_mutation_violations"] = sum(before.get(p) != after.get(p) for p in set(before) | set(after))
    report["checks"].update({name + "_zero": report[name] == 0 for name in VIOLATIONS})
    report["errors"] = [name for name, value in report["checks"].items() if not value]
    if report["errors"]:
        raise ValueError(f"Stage 12 isolation/roundtrip audit FAIL: {report['errors']}")
    outputs = {name: output_dir / f"NVDA_stage12_{name}.csv" for name in TABLE_NAMES}
    summary_path, validation_path = (output_dir / f"NVDA_stage12_{name}.json" for name in ("summary", "validation"))
    for name, path in outputs.items():
        baseline._atomic_write(path, text[name])
    diagnostics.summary.update({"validation_passed": True, "protocol_registered_before_diagnostics": True,
        **{name: report[name] for name in VIOLATIONS}})
    baseline._atomic_write(summary_path, json.dumps(baseline._json_clean(diagnostics.summary), indent=2, allow_nan=False) + "\n")
    after = protected_files(output_dir, roots)
    report.update({"source_sha256_before": before, "source_sha256_after": after, "source_artifact_count": len(before),
        "source_artifacts_unchanged": before == after, "protocol_registered_before_diagnostics": True, "mutation_checks_executed": True,
        "output_paths": {**{name: str(path) for name, path in outputs.items()}, "summary": str(summary_path), "validation": str(validation_path)},
        "output_artifact_sha256": {str(p): baseline._sha256(p) for p in (*outputs.values(), summary_path,
            output_dir / "NVDA_stage12_protocol.json", output_dir / "NVDA_stage12_upstream_sha256.json")},
        "reproducibility": {"randomness": "none", "run_timestamp": "omitted", "numpy_version": np.__version__, "pandas_version": pd.__version__}})
    report["checks"]["upstream_unchanged_after_publish"] = before == after
    report["errors"] = [name for name, value in report["checks"].items() if not value]
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    baseline._atomic_write(validation_path, json.dumps(baseline._json_clean(report), indent=2, allow_nan=False) + "\n")
    if not report["valid"]:
        raise ValueError("Upstream mutation during publication")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {"labeled-path": baseline.RESEARCH_DIR / "NVDA_labeled.csv", "split-dir": baseline.RESEARCH_DIR / "splits",
        "diagnostic-dir": baseline.RESEARCH_DIR / "diagnostics", "baseline-dir": baseline.RESEARCH_DIR / "modeling",
        "stability-dir": baseline.RESEARCH_DIR / "modeling/stability", "stage11-dir": baseline.RESEARCH_DIR / "modeling/stage11_market_representation",
        "output-dir": baseline.RESEARCH_DIR / "modeling/stage12_signal_diagnostics"}
    for name, default in defaults.items():
        parser.add_argument("--" + name, type=Path, default=default)
    parser.add_argument("--register-only", action="store_true")
    args = vars(parser.parse_args())
    report = run_pipeline(**args)
    if args["register_only"]:
        print(f"Stage 12 protocol registered before diagnostics; {report['upstream_file_count']} upstream files protected.")
    else:
        summary = json.loads((args["output_dir"] / "NVDA_stage12_summary.json").read_text())
        print(f"Stage 12 validation: {report['validation_status']}; assessment: {summary['integrated_mechanism_assessment']['category']}")
        print("No new learned/calibrated predictions or calibrated metrics. Final Test locked and unused.")


if __name__ == "__main__":
    main()
