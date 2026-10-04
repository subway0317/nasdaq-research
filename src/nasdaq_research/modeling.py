"""Stage 10: primary-target development CV only, with the final test closed."""

import argparse
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from nasdaq_research.config import DATA_DIR
from nasdaq_research.dates import parse_dates
from nasdaq_research.diagnostics import FEATURE_GROUPS, RESEARCH_GROUPS, classify_feature
from nasdaq_research.splits import MANIFEST_COLUMNS, PRIMARY_METRIC, SECONDARY_METRICS
from nasdaq_research.targets import PRIMARY_TARGET

RESEARCH_DIR = DATA_DIR / "research"
MODELS = ("zero_return", "historical_mean", "ols_market", "ols_all", "ridge_market", "ridge_all")
LINEAR_MODELS = MODELS[2:]
TRAINING_COVERAGE_THRESHOLD = 0.50
RIDGE_ALPHA = 1.0
METRICS = ("MAE", "RMSE", "R2", "Pearson", "Spearman", "directional_accuracy")
PREDICTION_COLUMNS = tuple(f"{name}_prediction" for name in MODELS)
TABLE_NAMES = ("oof_predictions", "cv_metrics", "cv_summary", "feature_usage", "coefficients")
LIMITATIONS = [
    "About one year of NVDA data; only the primary 5-session target is modeled.",
    "Only 75 OOF observations, with 25 per fold under the registered protocol.",
    "Overlapping 5-session targets and repeated fundamentals mean observations are not independent.",
    "Results are a descriptive development-CV benchmark, not evidence of profitability or generalization.",
    "No hyperparameter tuning, predictive feature selection or nonlinear models.",
    "No final model fit, final-test prediction or final-test metric calculation.",
    "No backtest, statistical significance test, transaction costs or portfolio construction.",
    "Standardized-feature coefficients are diagnostics and cannot drive feature/model changes in this stage.",
]


@dataclass
class Preprocessor:
    """A fitted, training-only feature transform and its auditable parameters."""

    features: tuple[str, ...]
    medians: pd.Series
    scaler: StandardScaler
    usage: pd.DataFrame
    fit_dates: tuple[str, ...]


@dataclass
class FittedLinear:
    """Training result; no validation labels are accepted by the fit interface."""

    preprocessing: Preprocessor
    estimator: LinearRegression | Ridge
    train_dates: tuple[str, ...]


def _check_candidates(candidates: tuple[str, ...]) -> None:
    if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(FEATURE_GROUPS):
        raise ValueError("X candidates must be unique Stage 7 research features; targets/raw/provenance are forbidden")


def feature_specs(columns: pd.Index, inventory: pd.DataFrame, quality: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Strictly verify Stage 7 classification; ignore its overall quality/coverage scores."""
    for name, frame in (("inventory", inventory), ("quality", quality)):
        required = {"feature_name", "feature_group", "is_research_feature"}
        if not required <= set(frame) or frame.feature_name.duplicated().any():
            raise ValueError(f"Invalid Stage 7 {name} schema or duplicate features")
        research = frame.loc[frame.is_research_feature.eq(True)]
        mapping = dict(zip(research.feature_name, research.feature_group))
        if mapping != FEATURE_GROUPS:
            raise ValueError(f"Stage 7 {name} research whitelist differs from registered definitions")
    if not set(FEATURE_GROUPS) <= set(columns):
        raise ValueError("Labeled dataset lacks registered research features")
    counts = {g: sum(value == g for value in FEATURE_GROUPS.values()) for g in RESEARCH_GROUPS}
    if counts != {"market": 9, "quarterly": 13, "annual": 13, "balance_sheet": 4}:
        raise ValueError("Stage 7 registered feature specification changed")
    return {"market": tuple(c for c, g in FEATURE_GROUPS.items() if g == "market"), "all": tuple(FEATURE_GROUPS)}


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Validate only the supplied rows/columns; unused future values are never inspected."""
    result = frame.loc[:, list(columns)].copy()
    for name in columns:
        values = result[name]
        if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_complex_dtype(values)):
            raise ValueError(f"Invalid numeric research feature: {name}")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Infinite research feature in authorized rows: {name}")
    return result.astype(float)


def fit_preprocessor(training: pd.DataFrame, candidates: tuple[str, ...]) -> Preprocessor:
    """Coverage ≥50%, median fill, exact zero-variance removal, then StandardScaler."""
    _check_candidates(candidates)
    if training.empty:
        raise ValueError("Nonempty effective training rows required")
    numeric = _numeric(training, candidates)
    rows, imputed, medians = [], {}, {}
    for name in candidates:
        values = numeric[name]
        count = int(values.notna().sum())
        coverage = count / len(training)
        median = mean = std = np.nan
        reason = "low_training_coverage"
        if coverage >= TRAINING_COVERAGE_THRESHOLD:
            median = float(values.median())
            if not np.isfinite(median):
                raise ValueError(f"Undefined training median: {name}")
            filled = values.fillna(median).to_numpy()
            mean, variance = float(filled.mean()), float(filled.var(ddof=0))
            # An exactly repeated float can have tiny numerical np.var residue.
            constant = bool(np.all(filled == filled[0])) or variance == 0
            std = 0.0 if constant else float(np.sqrt(variance))
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError(f"Invalid training moments: {name}")
            reason = "zero_training_variance" if constant else "used"
            if reason == "used":
                imputed[name], medians[name] = filled, median
        rows.append({"feature_name": name, "candidate_feature": True, "training_row_count": len(training),
                     "training_non_null_count": count, "training_coverage": coverage,
                     "used": reason == "used", "drop_reason": reason, "training_median": median,
                     "training_mean_after_imputation": mean, "training_std_after_imputation": std,
                     "scaler_scale": np.nan})
    if not imputed:
        raise ValueError("No valid features after training-only coverage/variance filters")
    matrix = pd.DataFrame(imputed, index=training.index)
    scaler = StandardScaler().fit(matrix)
    usage = pd.DataFrame(rows)
    for i, name in enumerate(matrix.columns):
        usage.loc[usage.feature_name.eq(name), "training_mean_after_imputation"] = scaler.mean_[i]
        usage.loc[usage.feature_name.eq(name), "scaler_scale"] = scaler.scale_[i]
    return Preprocessor(tuple(matrix.columns), pd.Series(medians, dtype=float), scaler, usage,
                        tuple(map(str, training.index)))


def transform_features(preprocessing: Preprocessor, values: pd.DataFrame) -> np.ndarray:
    """Use fixed training medians and scaler; no fit occurs during prediction."""
    matrix = _numeric(values, preprocessing.features).fillna(preprocessing.medians)
    transformed = preprocessing.scaler.transform(matrix)
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite transformed model input")
    return transformed


def _target(values: pd.Series) -> np.ndarray:
    if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)):
        raise ValueError("Primary target must be real numeric")
    result = values.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(result).all():
        raise ValueError("Missing/non-finite primary target in authorized CV rows")
    return result


def fit_linear(model_name: str, training: pd.DataFrame, target: pd.Series,
               candidates: tuple[str, ...]) -> FittedLinear:
    """Fit one OLS/Ridge specification on effective training data, with raw Y."""
    if model_name not in LINEAR_MODELS or not training.index.equals(target.index):
        raise ValueError("Registered linear model and exactly aligned training X/Y required")
    preprocessing = fit_preprocessor(training, candidates)
    matrix = transform_features(preprocessing, training)
    estimator = (LinearRegression(fit_intercept=True, n_jobs=1) if model_name.startswith("ols_")
                 else Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd"))
    estimator.fit(matrix, _target(target))
    if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_):
        raise ValueError("Non-finite fitted coefficients")
    return FittedLinear(preprocessing, estimator, tuple(map(str, training.index)))


def predict_linear(fitted: FittedLinear, validation: pd.DataFrame) -> np.ndarray:
    predictions = fitted.estimator.predict(transform_features(fitted.preprocessing, validation))
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite validation predictions")
    return predictions


def _correlation(actual: np.ndarray, prediction: np.ndarray) -> float:
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(prediction) == 0:
        return np.nan
    return float(np.corrcoef(actual, prediction)[0, 1])


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Registered metrics; undefined correlations/R² stay NaN, negative R² is valid."""
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if actual.ndim != 1 or actual.shape != prediction.shape or not len(actual):
        raise ValueError("Aligned, nonempty metric vectors required")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite complete actuals and predictions")
    errors = actual - prediction
    result = {"MAE": float(np.abs(errors).mean()), "RMSE": float(np.sqrt(np.square(errors).mean())),
              "R2": float(r2_score(actual, prediction)) if len(actual) >= 2 and np.ptp(actual) else np.nan,
              "Pearson": _correlation(actual, prediction),
              "Spearman": _correlation(pd.Series(actual).rank(method="average").to_numpy(),
                                       pd.Series(prediction).rank(method="average").to_numpy()),
              "directional_accuracy": float((np.sign(actual) == np.sign(prediction)).mean())}
    if not np.isfinite([result["MAE"], result["RMSE"]]).all():
        raise ValueError("Non-finite primary error metrics")
    return result


def cv_authority(data: pd.DataFrame, manifest: pd.DataFrame, protocol: dict[str, Any],
                 folds: pd.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    """Read existing roles only; audit dates/counts without generating any new split."""
    if (protocol.get("protocol_version") != "9.1" or protocol.get("primary_target") != PRIMARY_TARGET
            or protocol.get("primary_metric") != PRIMARY_METRIC or protocol.get("secondary_metrics") != list(SECONDARY_METRICS)
            or protocol.get("final_test_locked") is not True or protocol.get("random_split_allowed") is not False
            or protocol.get("embargo_sessions") != 0 or protocol.get("cv_fold_count") != 3
            or protocol.get("validation_size") != 25 or protocol.get("final_test_size") != 50
            or protocol.get("strict_pre_test_model_selection_boundary_satisfied") is not True):
        raise ValueError("Stage 9.1 locked primary-target evaluation protocol required")
    if list(manifest.columns) != list(MANIFEST_COLUMNS) or manifest.duplicated(["fold", "target_name", "date"]).any():
        raise ValueError("Stage 9.1 manifest schema/unique roles required")
    if data.columns.duplicated().any() or not data.ticker.eq("NVDA").all():
        raise ValueError("Unique columns and single NVDA ticker required")
    dates = parse_dates(data.date)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Unique ascending source dates required")
    source_dates = dates.dt.strftime("%Y-%m-%d").tolist()
    keyed = data.copy()
    keyed.index = source_dates
    final_start = protocol["boundaries"]["final_test"]["start"]
    result = {}
    for boundary in protocol["boundaries"]["cv"]:
        fold = boundary["fold"]
        scoped = manifest.loc[manifest.target_name.eq(PRIMARY_TARGET) & manifest.fold.eq(fold)]
        if scoped.date.tolist() != source_dates or not scoped.ticker.eq("NVDA").all():
            raise ValueError("Manifest source row/date identity mismatch")
        train = scoped.loc[scoped.role.eq("train")]
        valid = scoped.loc[scoped.role.eq("validation")]
        summary = folds.loc[folds.target_name.eq(PRIMARY_TARGET) & folds.fold.eq(fold)]
        if len(summary) != 1 or train.empty or len(valid) != protocol["validation_size"]:
            raise ValueError("Missing/invalid manifest fold counts")
        expected = summary.iloc[0]
        if (len(train) != expected.train_count_effective or len(valid) != expected.validation_count_effective
                or valid.date.iloc[0] != boundary["start"] or valid.date.iloc[-1] != boundary["end"]
                or expected.validation_start_date != boundary["start"] or expected.validation_end_date != boundary["end"]):
            raise ValueError("Manifest differs from authoritative CV count/date summary")
        for role, selected in (("train", train), ("validation", valid)):
            if (not selected.nominal_role.eq(role).all() or not selected.is_usable.eq(True).all()
                    or not selected.partition.eq("development").all() or selected.pre_test_gap.any()
                    or selected.evaluation_locked.any() or selected.target_missing.any()
                    or not selected.evaluation_start_date.eq(boundary["start"]).all()):
                raise ValueError("Unauthorized gap/test/tail/purged rows in CV role")
            source = keyed.loc[selected.date]
            entry = parse_dates(source.target_entry_date)
            exits = parse_dates(source.target_exit_date_5d)
            if (not entry.gt(parse_dates(source.date)).all() or not exits.ge(entry).all()
                    or exits.dt.strftime("%Y-%m-%d").tolist() != selected.target_exit_date.tolist()
                    or entry.dt.strftime("%Y-%m-%d").tolist() != selected.target_entry_date.tolist()):
                raise ValueError("CV target provenance differs from Stage 9.1 manifest")
            _target(source[PRIMARY_TARGET])
            if role == "train" and (not selected.date.lt(boundary["start"]).all() or not selected.target_exit_date.lt(boundary["start"]).all()):
                raise ValueError("Train → Validation label overlap")
            if role == "validation" and not selected.target_exit_date.lt(final_start).all():
                raise ValueError("Validation → Final Test label overlap")
        result[fold] = (train.date.tolist(), valid.date.tolist())
    validation_dates = [day for _, days in result.values() for day in days]
    if (len(result) != 3 or list(result) != ["cv_1", "cv_2", "cv_3"] or len(set(validation_dates)) != 75
            or validation_dates != sorted(validation_dates)):
        raise ValueError("Three ordered, disjoint, complete manifest validation blocks required")
    return result


def baseline_protocol(specifications: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Fixed Stage 10 contract, separate from immutable Stage 9.1 artifacts."""
    return {"stage": "10", "ticker": "NVDA", "primary_target": PRIMARY_TARGET, "split_protocol_version": "9.1",
            "models": list(MODELS), "feature_specs": {k: list(v) for k, v in specifications.items()},
            "training_coverage_threshold": TRAINING_COVERAGE_THRESHOLD, "coverage_rule": ">= 0.50 on effective training only",
            "imputation_strategy": "effective-training median", "zero_variance_scope": "imputed effective training only",
            "scaling_strategy": "training-only StandardScaler; ddof=0", "target_transformation": "none",
            "ridge_alpha": RIDGE_ALPHA, "ridge_solver": "svd", "fit_intercept": True,
            "primary_metric": PRIMARY_METRIC, "secondary_metrics": list(SECONDARY_METRICS),
            "model_ranking_rule": "lowest equal-weight mean fold MAE; exact ties reported together",
            "fold_metric_std_ddof": 1, "constant_prediction_correlation": "undefined; CSV NaN/JSON null",
            "final_test_locked": True, "final_test_predictions_allowed": False, "final_test_metrics_allowed": False,
            "final_test_predictions_generated": False, "final_test_metrics_computed": False,
            "pre_test_gap_allowed_in_cv": False, "final_training_pool_used": False,
            "random_split_allowed": False, "hyperparameter_tuning_allowed": False, "predictive_feature_selection_allowed": False,
            "limitations": LIMITATIONS}


def summarize_cv(metrics: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight fold summaries plus pooled OOF metrics; no test selection."""
    rows = []
    actual = oof[f"actual_{PRIMARY_TARGET}"].to_numpy()
    for model in MODELS:
        per_fold = metrics.loc[metrics.model_name.eq(model)]
        row = {"model_name": model, "feature_spec": per_fold.feature_spec.iloc[0]}
        row.update({f"fold_{i + 1}_MAE": float(per_fold.MAE.iloc[i]) for i in range(len(per_fold))})
        row.update({f"mean_fold_{name}": float(per_fold[name].mean()) for name in METRICS})
        row["std_fold_MAE"] = float(per_fold.MAE.std(ddof=1))
        row.update({f"pooled_oof_{name}": value for name, value in regression_metrics(actual, oof[f"{model}_prediction"].to_numpy()).items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["rank_by_mean_fold_MAE"] = summary.mean_fold_MAE.rank(method="min").astype(int)
    return summary.sort_values(["mean_fold_MAE", "model_name"], kind="stable").reset_index(drop=True)


def build_cv(data: pd.DataFrame, manifest: pd.DataFrame, split_protocol: dict[str, Any],
             folds: pd.DataFrame, inventory: pd.DataFrame, quality: pd.DataFrame
             ) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[dict[str, Any]]]:
    """Fit the six fixed specifications and return development validation artifacts."""
    specifications = feature_specs(data.columns, inventory, quality)
    authority = cv_authority(data, manifest, split_protocol, folds)
    keyed = data.copy()
    keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
    oof_blocks, metric_rows, usage_blocks, coefficient_rows, fit_records = [], [], [], [], []
    with threadpool_limits(limits=1):
        for fold, (train_dates, valid_dates) in authority.items():
            training, validation = keyed.loc[train_dates], keyed.loc[valid_dates]
            y_train, y_valid = _target(training[PRIMARY_TARGET]), _target(validation[PRIMARY_TARGET])
            block = pd.DataFrame({"date": valid_dates, "ticker": "NVDA", "fold": fold, f"actual_{PRIMARY_TARGET}": y_valid})
            for model in MODELS:
                spec = "none" if model in MODELS[:2] else model.split("_", 1)[1]
                record = {"model_name": model, "fold": fold, "fit_target": PRIMARY_TARGET,
                          "training_dates": [] if model == "zero_return" else train_dates,
                          "validation_dates": valid_dates, "preprocessing_fit_dates": [], "scaler_sample_count": 0,
                          "ridge_alpha": RIDGE_ALPHA if model.startswith("ridge_") else None}
                if model == "zero_return":
                    predictions = np.zeros(len(valid_dates))
                elif model == "historical_mean":
                    predictions = np.full(len(valid_dates), y_train.mean())
                else:
                    candidates = specifications[spec]
                    fitted = fit_linear(model, training.loc[:, list(candidates)], training[PRIMARY_TARGET], candidates)
                    predictions = predict_linear(fitted, validation.loc[:, list(candidates)])
                    usage = fitted.preprocessing.usage.copy()
                    usage.insert(0, "fold", fold)
                    usage.insert(0, "feature_spec", spec)
                    usage.insert(0, "model_name", model)
                    usage_blocks.append(usage)
                    record["training_dates"] = list(fitted.train_dates)
                    record["preprocessing_fit_dates"] = list(fitted.preprocessing.fit_dates)
                    record["scaler_sample_count"] = int(fitted.preprocessing.scaler.n_samples_seen_)
                    for feature, coefficient in zip(fitted.preprocessing.features, fitted.estimator.coef_):
                        coefficient_rows.append({"model_name": model, "feature_spec": spec, "fold": fold,
                                                 "feature_name": feature, "coefficient": float(coefficient),
                                                 "intercept": float(fitted.estimator.intercept_)})
                fit_records.append(record)
                block[f"{model}_prediction"] = predictions
                metric_rows.append({"model_name": model, "feature_spec": spec, "fold": fold,
                                    "train_count": len(train_dates), "validation_count": len(valid_dates),
                                    **regression_metrics(y_valid, predictions)})
            oof_blocks.append(block)
    oof, metrics = pd.concat(oof_blocks, ignore_index=True), pd.DataFrame(metric_rows)
    tables = {"oof_predictions": oof, "cv_metrics": metrics, "cv_summary": summarize_cv(metrics, oof),
              "feature_usage": pd.concat(usage_blocks, ignore_index=True), "coefficients": pd.DataFrame(coefficient_rows)}
    return tables, baseline_protocol(specifications), fit_records


def _close(actual: Any, expected: Any) -> bool:
    return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True))


def _audit_training_stats(values: pd.Series) -> dict[str, Any]:
    """Independent scalar moments for the saved preprocessing audit, without fitting."""
    observed = [float(value) for value in values if pd.notna(value)]
    coverage = len(observed) / len(values)
    result = {"training_non_null_count": len(observed), "training_coverage": coverage,
              "training_median": np.nan, "training_mean_after_imputation": np.nan,
              "training_std_after_imputation": np.nan, "scaler_scale": np.nan,
              "drop_reason": "low_training_coverage"}
    if coverage < TRAINING_COVERAGE_THRESHOLD:
        return result
    median = float(np.median(observed))
    filled = np.asarray([median if pd.isna(value) else float(value) for value in values])
    mean, variance = float(filled.mean()), float(np.mean((filled - filled.mean()) ** 2))
    constant = bool(np.all(filled == filled[0])) or variance == 0
    std = 0.0 if constant else float(np.sqrt(variance))
    # StandardScaler's floating-point bound for near-constant features.
    eps = np.finfo(float).eps
    bound = len(filled) * eps * variance + (len(filled) * mean * eps) ** 2
    scale = np.nan if constant else 1.0 if variance <= bound else std
    result.update({"training_median": median, "training_mean_after_imputation": mean,
                   "training_std_after_imputation": std, "scaler_scale": scale,
                   "drop_reason": "zero_training_variance" if constant else "used"})
    return result


def _audit_preprocessing(keyed: pd.DataFrame, authority: dict[str, tuple[list[str], list[str]]],
                         specifications: dict[str, tuple[str, ...]], tables: dict[str, pd.DataFrame],
                         report: dict[str, Any], checks: dict[str, bool]) -> None:
    """Compare coverage, fills, moments and coefficient-based predictions to source training."""
    usage, coefficients, oof = tables["feature_usage"], tables["coefficients"], tables["oof_predictions"]
    checks["feature_usage_unique"] = not usage.duplicated(["model_name", "fold", "feature_name"]).any()
    checks["coefficient_rows_unique"] = not coefficients.duplicated(["model_name", "fold", "feature_name"]).any()
    checks["feature_usage_models_exact"] = set(usage.model_name) == set(LINEAR_MODELS)
    checks["coefficient_models_exact"] = set(coefficients.model_name) == set(LINEAR_MODELS)
    for fold, (train_dates, valid_dates) in authority.items():
        for model in LINEAR_MODELS:
            spec = model.split("_", 1)[1]
            rows = usage.loc[usage.model_name.eq(model) & usage.fold.eq(fold)]
            candidates = specifications[spec]
            checks[f"{model}_{fold}_candidate_schema"] = (
                rows.feature_name.tolist() == list(candidates) and rows.feature_spec.eq(spec).all()
                and rows.candidate_feature.eq(True).all() and rows.training_row_count.eq(len(train_dates)).all())
            for row in rows.itertuples():
                if row.feature_name not in candidates:
                    continue
                expected = _audit_training_stats(keyed.loc[train_dates, row.feature_name])
                report["training_only_coverage_violations"] += int(
                    row.training_non_null_count != expected["training_non_null_count"]
                    or not _close(row.training_coverage, expected["training_coverage"])
                    or row.drop_reason != expected["drop_reason"] or row.used != (expected["drop_reason"] == "used"))
                report["training_only_imputation_violations"] += int(not _close(row.training_median, expected["training_median"]))
                report["training_only_scaling_violations"] += int(any(not _close(getattr(row, field), expected[field]) for field in (
                    "training_mean_after_imputation", "training_std_after_imputation", "scaler_scale")))
            retained = rows.loc[rows.used.eq(True)].set_index("feature_name")
            coefficient_rows = coefficients.loc[coefficients.model_name.eq(model) & coefficients.fold.eq(fold)]
            checks[f"{model}_{fold}_coefficient_schema"] = (
                not retained.empty and coefficient_rows.feature_name.tolist() == retained.index.tolist()
                and coefficient_rows.feature_spec.eq(spec).all() and coefficient_rows.intercept.nunique() == 1
                and np.isfinite(coefficient_rows[["coefficient", "intercept"]].to_numpy()).all())
            if not checks[f"{model}_{fold}_coefficient_schema"]:
                continue
            validation = keyed.loc[valid_dates, retained.index].astype(float)
            filled = validation.fillna(retained.training_median)
            transformed = (filled - retained.training_mean_after_imputation) / retained.scaler_scale
            expected_prediction = transformed.to_numpy() @ coefficient_rows.coefficient.to_numpy() + coefficient_rows.intercept.iloc[0]
            actual_prediction = oof.loc[oof.fold.eq(fold), f"{model}_prediction"].to_numpy()
            checks[f"{model}_{fold}_prediction_reconstructs"] = (
                expected_prediction.shape == actual_prediction.shape
                and np.allclose(actual_prediction, expected_prediction, rtol=1e-10, atol=1e-12))


def _audit_fit_records(authority: dict[str, tuple[list[str], list[str]]], records: list[dict[str, Any]],
                       manifest: pd.DataFrame, prediction_dates: list[str],
                       report: dict[str, Any], checks: dict[str, bool]) -> None:
    keys = [(record["fold"], record["model_name"]) for record in records]
    checks["fit_records_exact"] = keys == [(fold, model) for fold in authority for model in MODELS]
    forbidden = manifest.loc[manifest.target_name.eq(PRIMARY_TARGET) & manifest.fold.eq("final")]
    gap_dates = set(forbidden.loc[forbidden.pre_test_gap, "date"])
    test_dates = set(forbidden.loc[forbidden.partition.eq("final_test"), "date"])
    tail_dates = set(forbidden.loc[forbidden.partition.eq("unlabeled_tail"), "date"])
    used_dates = set(prediction_dates)
    for record in records:
        fold, model = record["fold"], record["model_name"]
        if fold not in authority or model not in MODELS:
            checks["fit_records_exact"] = False
            continue
        train_dates, valid_dates = authority[fold]
        expected_train = [] if model == "zero_return" else train_dates
        expected_preprocessing = train_dates if model in LINEAR_MODELS else []
        checks[f"{model}_{fold}_fit_rows"] = (
            record["training_dates"] == expected_train and record["validation_dates"] == valid_dates
            and record["preprocessing_fit_dates"] == expected_preprocessing
            and record["scaler_sample_count"] == (len(train_dates) if model in LINEAR_MODELS else 0)
            and record["fit_target"] == PRIMARY_TARGET
            and record["ridge_alpha"] == (RIDGE_ALPHA if model.startswith("ridge_") else None))
        used_dates.update(record["training_dates"] + record["validation_dates"] + record["preprocessing_fit_dates"])
    report.update({"pre_test_gap_rows_used": len(used_dates & gap_dates), "final_test_rows_used": len(used_dates & test_dates),
                   "unlabeled_tail_rows_used": len(used_dates & tail_dates)})
    checks["no_forbidden_rows_used"] = not (used_dates & (gap_dates | test_dates | tail_dates))


def _audit_metrics(keyed: pd.DataFrame, authority: dict[str, tuple[list[str], list[str]]],
                   tables: dict[str, pd.DataFrame], checks: dict[str, bool]) -> None:
    metrics, oof = tables["cv_metrics"], tables["oof_predictions"]
    checks["metric_rows_exact"] = list(metrics[["fold", "model_name"]].itertuples(index=False, name=None)) == [
        (fold, model) for fold in authority for model in MODELS]
    for fold, (train_dates, valid_dates) in authority.items():
        predictions = oof.loc[oof.fold.eq(fold)]
        for model in MODELS:
            row = metrics.loc[metrics.fold.eq(fold) & metrics.model_name.eq(model)]
            if len(row) != 1:
                checks["metric_rows_exact"] = False
                continue
            expected = regression_metrics(keyed.loc[valid_dates, PRIMARY_TARGET].to_numpy(), predictions[f"{model}_prediction"].to_numpy())
            checks[f"{model}_{fold}_metrics_correct"] = (
                row.train_count.iloc[0] == len(train_dates) and row.validation_count.iloc[0] == len(valid_dates)
                and row.feature_spec.iloc[0] == ("none" if model in MODELS[:2] else model.split("_", 1)[1])
                and all(_close(row[name].iloc[0], value) for name, value in expected.items()))
    try:
        pd.testing.assert_frame_equal(tables["cv_summary"], summarize_cv(metrics, oof), check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-12)
        checks["cv_summary_correct"] = True
    except AssertionError:
        checks["cv_summary_correct"] = False


def validate_modeling(data: pd.DataFrame, manifest: pd.DataFrame, split_protocol: dict[str, Any],
                      folds: pd.DataFrame, inventory: pd.DataFrame, quality: pd.DataFrame,
                      tables: dict[str, pd.DataFrame], protocol: dict[str, Any],
                      fit_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit saved CV outputs and fitting scopes; never fit or predict on the final test."""
    report: dict[str, Any] = {name: 0 for name in (
        "target_columns_in_X", "target_provenance_columns_in_X", "metadata_columns_in_X",
        "raw_non_research_columns_in_X", "pre_test_gap_rows_used", "final_test_rows_used", "unlabeled_tail_rows_used",
        "training_only_coverage_violations", "training_only_imputation_violations", "training_only_scaling_violations",
        "historical_mean_training_only_violations", "missing_prediction_count", "non_finite_prediction_count")}
    report.update({"ticker": "NVDA", "primary_target": PRIMARY_TARGET, "final_test_locked": True,
                   "final_test_predictions_generated": False, "final_test_metrics_computed": False,
                   "final_training_pool_used": False})
    checks: dict[str, bool] = {}
    try:
        specifications = feature_specs(data.columns, inventory, quality)
        authority = cv_authority(data, manifest, split_protocol, folds)
        keyed = data.copy()
        keyed.index = parse_dates(data.date).dt.strftime("%Y-%m-%d")
        oof, usage, coefficients = tables["oof_predictions"], tables["feature_usage"], tables["coefficients"]
        expected_dates = [day for _, valid_dates in authority.values() for day in valid_dates]
        expected_folds = [fold for fold, (_, valid_dates) in authority.items() for _ in valid_dates]
        checks["oof_schema_exact"] = oof.columns.tolist() == ["date", "ticker", "fold", f"actual_{PRIMARY_TARGET}", *PREDICTION_COLUMNS]
        checks["oof_rows_exact"] = oof.date.tolist() == expected_dates and oof.fold.tolist() == expected_folds and oof.ticker.eq("NVDA").all()
        checks["oof_actuals_unchanged"] = np.array_equal(oof[f"actual_{PRIMARY_TARGET}"].to_numpy(), keyed.loc[expected_dates, PRIMARY_TARGET].to_numpy())
        report.update({"fold_count": len(authority), "oof_row_count": len(oof), "oof_duplicate_count": int(oof.date.duplicated().sum()),
                       "oof_date_range": {"start": oof.date.min(), "end": oof.date.max()},
                       "fold_train_counts": {fold: len(train) for fold, (train, _) in authority.items()},
                       "fold_validation_counts": {fold: len(valid) for fold, (_, valid) in authority.items()},
                       "feature_whitelist_count": len(specifications["all"]),
                       "feature_candidate_counts": {name: len(features) for name, features in specifications.items()}})
        prediction_values = oof.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=float)
        report["missing_prediction_count"] = int(np.isnan(prediction_values).sum())
        report["non_finite_prediction_count"] = int((~np.isfinite(prediction_values)).sum())
        checks["predictions_complete_finite"] = report["non_finite_prediction_count"] == 0
        checks["oof_dates_unique"] = report["oof_duplicate_count"] == 0
        names = set(usage.loc[usage.used.eq(True), "feature_name"]) | set(coefficients.feature_name)
        report["target_columns_in_X"] = sum(name.startswith("forward_return_") for name in names)
        report["target_provenance_columns_in_X"] = sum(name.startswith("target_") for name in names)
        report["metadata_columns_in_X"] = sum(classify_feature(name) == "metadata" for name in names)
        report["raw_non_research_columns_in_X"] = len(names - set(FEATURE_GROUPS))
        checks["feature_whitelist_enforced"] = not (names - set(FEATURE_GROUPS))
        checks["zero_baseline_correct"] = oof.zero_return_prediction.eq(0).all()
        for fold, (train_dates, _) in authority.items():
            expected_mean = keyed.loc[train_dates, PRIMARY_TARGET].to_numpy(dtype=float).mean()
            actual = oof.loc[oof.fold.eq(fold), "historical_mean_prediction"].to_numpy()
            report["historical_mean_training_only_violations"] += int(not np.allclose(actual, expected_mean, rtol=1e-10, atol=1e-12))
        _audit_fit_records(authority, fit_records, manifest, oof.date.tolist(), report, checks)
        _audit_preprocessing(keyed, authority, specifications, tables, report, checks)
        if checks["predictions_complete_finite"] and checks["oof_rows_exact"]:
            _audit_metrics(keyed, authority, tables, checks)
        checks["stage10_protocol_fixed"] = all(protocol.get(key) == value for key, value in baseline_protocol(specifications).items())
        for key, value in report.items():
            if key.endswith("_violations") or key.endswith("_in_X") or key.endswith("_rows_used"):
                checks[f"{key}_zero"] = value == 0
        summary = tables["cv_summary"]
        winners = sorted(summary.loc[summary.mean_fold_MAE.eq(summary.mean_fold_MAE.min()), "model_name"].tolist())
        report["best_stage10_cv_candidate"] = winners[0] if len(winners) == 1 else None
        report["best_stage10_cv_candidates"] = winners
        report["fit_records"] = fit_records
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
        checks["artifact_audit_completed"] = False
        report["audit_error"] = str(exc)
    report["checks"] = {key: bool(value) for key, value in checks.items()}
    report["errors"] = [key for key, passed in checks.items() if not passed]
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    return report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_checkpoint(labeled_path: Path, split_dir: Path, diagnostic_dir: Path) -> tuple[Any, ...]:
    """Check frozen source hashes; ingest only the primary target and whitelisted X."""
    split_validation = json.loads((split_dir / "NVDA_split_validation.json").read_text())
    split_protocol = json.loads((split_dir / "NVDA_evaluation_protocol.json").read_text())
    if split_validation.get("valid") is not True or split_validation.get("stage10_readiness") != "READY FOR STAGE 10":
        raise ValueError("Successful Stage 9.1 validation/readiness checkpoint required")
    source_hash = _sha256(labeled_path)
    if source_hash != split_validation.get("source_labeled_sha256") or source_hash != split_protocol.get("source_labeled_sha256"):
        raise ValueError("Labeled dataset hash differs from the frozen Stage 9.1 checkpoint")
    for name in ("split_manifest.csv", "cv_folds.csv", "final_training_pool.csv", "split_summary.csv", "evaluation_protocol.json"):
        path = split_dir / f"NVDA_{name}"
        if split_validation.get("output_artifact_sha256", {}).get(str(path)) != _sha256(path):
            raise ValueError(f"Stage 9.1 artifact hash mismatch: {path.name}")
    columns = ["date", "ticker", "target_entry_date", "target_exit_date_5d", PRIMARY_TARGET, *FEATURE_GROUPS]
    labeled = pd.read_csv(labeled_path, usecols=columns,
                          dtype={name: str for name in ("date", "ticker", "target_entry_date", "target_exit_date_5d")},
                          converters={PRIMARY_TARGET: lambda value: float(value) if value else np.nan})
    manifest = pd.read_csv(split_dir / "NVDA_split_manifest.csv")
    folds = pd.read_csv(split_dir / "NVDA_cv_folds.csv")
    inventory = pd.read_csv(diagnostic_dir / "feature_inventory.csv")
    quality = pd.read_csv(diagnostic_dir / "feature_quality.csv")
    return labeled, manifest, split_protocol, folds, inventory, quality


def _json_clean(value: Any) -> Any:
    """Standard JSON uses null for undefined metrics, never NaN/Infinity tokens."""
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_write(path: Path, contents: str) -> None:
    """Replace each output with a complete file, leaving no partial final artifact."""
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError("Modeling output symlinks are forbidden")
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def run_pipeline(
    labeled_path: Path = RESEARCH_DIR / "NVDA_labeled.csv",
    split_dir: Path = RESEARCH_DIR / "splits",
    diagnostic_dir: Path = RESEARCH_DIR / "diagnostics",
    output_dir: Path = RESEARCH_DIR / "modeling",
) -> dict[str, Any]:
    """Write seven Stage 10 artifacts; immutable upstream inputs and closed test remain intact."""
    labeled_path, split_dir, diagnostic_dir, output_dir = (
        Path(path).resolve() for path in (labeled_path, split_dir, diagnostic_dir, output_dir))
    inputs = {labeled_path, *(split_dir / f"NVDA_{name}" for name in (
        "split_manifest.csv", "cv_folds.csv", "final_training_pool.csv", "split_summary.csv",
        "split_validation.json", "evaluation_protocol.json")),
        diagnostic_dir / "feature_inventory.csv", diagnostic_dir / "feature_quality.csv"}
    if any(output_dir == path or output_dir in path.parents for path in inputs):
        raise ValueError("Output directory must not contain upstream inputs")
    outputs = {name: output_dir / f"NVDA_baseline_{name}.csv" for name in TABLE_NAMES}
    protocol_path = output_dir / "NVDA_baseline_protocol.json"
    validation_path = output_dir / "NVDA_baseline_validation.json"
    output_paths = {*outputs.values(), protocol_path, validation_path}
    if any(path.is_symlink() for path in output_paths):
        raise ValueError("Modeling output symlinks cannot overwrite upstream files")
    protected = set(inputs)
    for root in {DATA_DIR.resolve(), labeled_path.parent}:
        protected.update(path.resolve() for path in root.rglob("*") if path.is_file() and path.resolve() not in output_paths)
    before = {str(path): _sha256(path) for path in sorted(protected)}
    checkpoint = _read_checkpoint(labeled_path, split_dir, diagnostic_dir)
    original = checkpoint[0].copy(deep=True)
    tables, protocol, records = build_cv(*checkpoint)
    protocol["input_artifact_sha256"] = {str(path): before[str(path)] for path in sorted(inputs)}
    if protocol_path.exists():
        locked = json.loads(protocol_path.read_text())
        if locked != protocol:
            raise ValueError("Existing Stage 10 contract is locked; source/specification/metric/tuning changes require a new stage")
    report = validate_modeling(*checkpoint, tables, protocol, records)
    pd.testing.assert_frame_equal(checkpoint[0], original, check_exact=True)
    if not report["valid"]:
        raise ValueError(f"Stage 10 validation failed: {report['errors']}; {report.get('audit_error', '')}")
    csv_text = {name: table.to_csv(index=False) for name, table in tables.items()}
    reloaded = {name: pd.read_csv(StringIO(contents), float_precision="round_trip") for name, contents in csv_text.items()}
    roundtrip = validate_modeling(*checkpoint, reloaded, protocol, records)
    if not roundtrip["valid"]:
        raise ValueError(f"Stage 10 CSV round-trip validation failed: {roundtrip['errors']}")
    after = {path: _sha256(Path(path)) for path in before}
    if before != after:
        raise ValueError("Upstream artifacts changed during modeling; refusing to publish outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in outputs.items():
        _atomic_write(path, csv_text[name])
    _atomic_write(protocol_path, json.dumps(_json_clean(protocol), indent=2, allow_nan=False) + "\n")
    after = {path: _sha256(Path(path)) for path in before}
    report["checks"].update({"source_dataframe_unchanged": checkpoint[0].equals(original),
                             "source_artifact_hashes_unchanged": before == after,
                             "csv_roundtrip_validated": roundtrip["valid"]})
    report.update({"source_labeled_file": str(labeled_path), "split_manifest_file": str(split_dir / "NVDA_split_manifest.csv"),
                   "split_protocol_version": checkpoint[2]["protocol_version"],
                   "source_sha256_before": before, "source_sha256_after": after, "source_artifacts_unchanged": before == after,
                   "source_artifact_count": len(before), "primary_metric": PRIMARY_METRIC,
                   "secondary_metrics": list(SECONDARY_METRICS), "preprocessing_scope": "effective training only",
                   "audit_numeric_tolerance": {"rtol": 1e-10, "atol": 1e-12}, "limitations": LIMITATIONS,
                   "output_paths": {**{name: str(path) for name, path in outputs.items()},
                                    "protocol": str(protocol_path), "validation": str(validation_path)},
                   "output_artifact_sha256": {str(path): _sha256(path) for path in [*outputs.values(), protocol_path]},
                   "reproducibility": {"randomness": "none", "BLAS_threads": 1, "run_timestamp": "omitted",
                                       "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__, "numpy_version": np.__version__}})
    report["errors"] = [key for key, passed in report["checks"].items() if not passed]
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    _atomic_write(validation_path, json.dumps(_json_clean(report), indent=2, allow_nan=False) + "\n")
    if not report["valid"]:
        raise ValueError(f"Stage 10 source validation failed: {report['errors']}")
    return report


def terminal_summary(report: dict[str, Any]) -> str:
    summary = pd.read_csv(report["output_paths"]["cv_summary"])
    lines = ["Ticker: NVDA", f"Primary target: {PRIMARY_TARGET}", f"Split protocol version: {report['split_protocol_version']}",
             f"Fold count: {report['fold_count']}"]
    lines.extend(f"{fold}: effective train={count}, validation={report['fold_validation_counts'][fold]}"
                 for fold, count in report["fold_train_counts"].items())
    lines.extend([f"OOF rows: {report['oof_row_count']}", f"Feature candidates: {report['feature_candidate_counts']}"])
    lines.extend(f"{row.model_name}: mean fold MAE={row.mean_fold_MAE:.10g}, std={row.std_fold_MAE:.10g}, pooled OOF MAE={row.pooled_oof_MAE:.10g}"
                 for row in summary.itertuples())
    lines.extend([f"Best Stage-10 CV candidates: {', '.join(report['best_stage10_cv_candidates'])}",
                  "Final test locked: true", "Final test predictions generated: false", "Final test metrics computed: false",
                  f"Pre-test gap rows used: {report['pre_test_gap_rows_used']}", f"Validation: {report['validation_status']}",
                  f"Output directory: {Path(report['output_paths']['validation']).parent}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-path", type=Path, default=RESEARCH_DIR / "NVDA_labeled.csv")
    parser.add_argument("--split-dir", type=Path, default=RESEARCH_DIR / "splits")
    parser.add_argument("--diagnostic-dir", type=Path, default=RESEARCH_DIR / "diagnostics")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "modeling")
    args = parser.parse_args()
    print(terminal_summary(run_pipeline(args.labeled_path, args.split_dir, args.diagnostic_dir, args.output_dir)))


if __name__ == "__main__":
    main()
