"""Offline Stage 10 contracts, independent modeling oracle and leakage mutations."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research.diagnostics import FEATURE_GROUPS
from nasdaq_research.modeling import (
    MODELS, PRIMARY_TARGET, build_cv, cv_authority, feature_specs, fit_linear,
    fit_preprocessor, predict_linear, regression_metrics, run_pipeline,
    summarize_cv, transform_features, validate_modeling,
)
from nasdaq_research.splits import build_splits, validate_splits
from test_splits import fixture as timeline_fixture


def fixture():
    """Freeze a temporary Stage 9.1 authority over independently generated features."""
    data = timeline_fixture(251)
    positions = np.arange(len(data), dtype=float)
    for i, name in enumerate(FEATURE_GROUPS):
        data[name] = np.sin(positions / (i + 3)) + positions / (i + 7)
    data.loc[:58, "sma_60"] = np.nan
    data.loc[:59, "rolling_volatility_60"] = np.nan
    data["q_fcf_margin"] = np.nan
    data["fy_fcf_margin"] = np.nan
    data["q_gross_margin"] = 0.7
    # Dangerous numeric columns must never become X candidates.
    for name in ("close", "open", "volume", "revenue", "q_fiscal_year", "target_entry_open", "target_exit_close_5d"):
        data[name] = positions + 1e6
    inventory = pd.DataFrame([{"feature_name": name, "feature_group": group, "is_research_feature": True,
                               "coverage_ratio": 1.0, "usable": True} for name, group in FEATURE_GROUPS.items()])
    quality = inventory.copy(deep=True)
    tables, protocol = build_splits(data)
    return data, tables["split_manifest"], protocol, tables["cv_folds"], inventory, quality


def checkpoint_files(root):
    checkpoint = fixture()
    data, manifest, protocol, folds, inventory, quality = checkpoint
    split_dir, diagnostic_dir = root / "splits", root / "diagnostics"
    split_dir.mkdir()
    diagnostic_dir.mkdir()
    labeled = root / "NVDA_labeled.csv"
    data.to_csv(labeled, index=False)
    split_tables, _ = build_splits(data)
    for name, table in split_tables.items():
        table.to_csv(split_dir / f"NVDA_{name}.csv", index=False)
    protocol = copy.deepcopy(protocol)
    protocol["source_labeled_sha256"] = hashlib.sha256(labeled.read_bytes()).hexdigest()
    protocol["source_labeled_file"] = str(labeled)
    protocol_path = split_dir / "NVDA_evaluation_protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    report = validate_splits(data, split_tables, protocol)
    report.update({"source_labeled_sha256": protocol["source_labeled_sha256"], "stage10_readiness": "READY FOR STAGE 10",
                   "output_artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                for path in split_dir.iterdir() if path.is_file()}})
    (split_dir / "NVDA_split_validation.json").write_text(json.dumps(report))
    inventory.to_csv(diagnostic_dir / "feature_inventory.csv", index=False)
    quality.to_csv(diagnostic_dir / "feature_quality.csv", index=False)
    return {"labeled_path": labeled, "split_dir": split_dir, "diagnostic_dir": diagnostic_dir, "output_dir": root / "modeling"}


class FeatureContractTests(unittest.TestCase):
    def test_registered_groups_and_specs(self):
        data, _, _, _, inventory, quality = fixture()
        specs = feature_specs(data.columns, inventory, quality)
        self.assertEqual(len(specs["market"]), 9)
        self.assertEqual(len(specs["all"]), 39)
        self.assertEqual({group: sum(FEATURE_GROUPS[name] == group for name in specs["all"])
                          for group in ("market", "quarterly", "annual", "balance_sheet")},
                         {"market": 9, "quarterly": 13, "annual": 13, "balance_sheet": 4})

    def test_target_provenance_metadata_raw_numeric_candidates_rejected(self):
        for name in ("forward_return_1d", PRIMARY_TARGET, "forward_return_20d", "target_entry_open",
                     "target_exit_close_5d", "q_fiscal_year", "close", "volume", "revenue"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Stage 7 research"):
                fit_preprocessor(pd.DataFrame({name: [1., 2., 3.]}), (name,))

    def test_inventory_classification_tampering_rejected(self):
        data, _, _, _, inventory, quality = fixture()
        inventory.loc[0, "feature_group"] = "metadata"
        with self.assertRaisesRegex(ValueError, "whitelist"):
            feature_specs(data.columns, inventory, quality)

    def test_missing_and_duplicate_research_features_rejected(self):
        data, _, _, _, inventory, quality = fixture()
        with self.assertRaisesRegex(ValueError, "lacks"):
            feature_specs(data.drop(columns="q_gross_margin").columns, inventory, quality)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            feature_specs(data.columns, pd.concat([inventory, inventory.iloc[[0]]]), quality)

    def test_overall_stage7_quality_scores_do_not_filter_candidates(self):
        data, _, _, _, inventory, quality = fixture()
        expected = feature_specs(data.columns, inventory, quality)
        inventory["coverage_ratio"] = 0.0
        quality["usable"] = False
        self.assertEqual(feature_specs(data.columns, inventory, quality), expected)


class PreprocessingTests(unittest.TestCase):
    def test_exact_50_percent_coverage_kept(self):
        training = pd.DataFrame({"simple_return": [1., 2., 3., 4.], "log_return": [1., np.nan, np.nan, 3.]})
        preprocessing = fit_preprocessor(training, tuple(training))
        self.assertIn("log_return", preprocessing.features)
        self.assertEqual(preprocessing.usage.set_index("feature_name").loc["log_return", "training_coverage"], 0.5)

    def test_low_training_coverage_dropped_despite_full_validation(self):
        training = pd.DataFrame({"simple_return": [1., 2., 3., 4.], "log_return": [1., np.nan, np.nan, np.nan]})
        preprocessing = fit_preprocessor(training, tuple(training))
        transform_features(preprocessing, pd.DataFrame({"simple_return": [8., 9.], "log_return": [8., 9.]}))
        row = preprocessing.usage.set_index("feature_name").loc["log_return"]
        self.assertFalse(row.used)
        self.assertEqual(row.drop_reason, "low_training_coverage")

    def test_sparse_validation_does_not_drop_training_qualified_feature(self):
        training = pd.DataFrame({"simple_return": [1., 2., 3., 4.]})
        preprocessing = fit_preprocessor(training, tuple(training))
        actual = transform_features(preprocessing, pd.DataFrame({"simple_return": [np.nan, np.nan]}))
        self.assertEqual(preprocessing.features, ("simple_return",))
        self.assertTrue(np.isfinite(actual).all())

    def test_training_median_ignores_validation(self):
        training = pd.DataFrame({"simple_return": [1., 2., np.nan, 100.]})
        preprocessing = fit_preprocessor(training, tuple(training))
        self.assertEqual(preprocessing.medians["simple_return"], 2.)
        transformed = transform_features(preprocessing, pd.DataFrame({"simple_return": [1e9, np.nan]}))
        self.assertAlmostEqual(transformed[1, 0], (2 - 26.25) / np.std([1., 2., 2., 100.], ddof=0))
        self.assertEqual(preprocessing.medians["simple_return"], 2.)

    def test_scaler_population_moments_only_training(self):
        training = pd.DataFrame({"simple_return": [1., 2., np.nan, 100.]})
        preprocessing = fit_preprocessor(training, tuple(training))
        mean, scale = preprocessing.scaler.mean_.copy(), preprocessing.scaler.scale_.copy()
        np.testing.assert_allclose(mean, [26.25], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scale, [np.std([1., 2., 2., 100.], ddof=0)], rtol=0, atol=1e-12)
        transform_features(preprocessing, pd.DataFrame({"simple_return": [1e9, -1e9]}))
        np.testing.assert_array_equal(preprocessing.scaler.mean_, mean)
        np.testing.assert_array_equal(preprocessing.scaler.scale_, scale)
        self.assertEqual(preprocessing.scaler.n_samples_seen_, 4)

    def test_exact_constant_float_dropped_even_with_variation_in_validation(self):
        training = pd.DataFrame({"simple_return": [1., 2., 3., 4.], "log_return": [0.7] * 4})
        preprocessing = fit_preprocessor(training, tuple(training))
        transform_features(preprocessing, pd.DataFrame({"simple_return": [5., 6.], "log_return": [-1e9, 1e9]}))
        row = preprocessing.usage.set_index("feature_name").loc["log_return"]
        self.assertEqual(row.drop_reason, "zero_training_variance")
        self.assertFalse(row.used)
        self.assertEqual(row.training_std_after_imputation, 0)

    def test_all_missing_features_drop_by_coverage(self):
        training = pd.DataFrame({"simple_return": [1., 2., 3., 4.], "q_fcf_margin": [np.nan] * 4})
        preprocessing = fit_preprocessor(training, tuple(training))
        row = preprocessing.usage.set_index("feature_name").loc["q_fcf_margin"]
        self.assertEqual(row.drop_reason, "low_training_coverage")
        self.assertEqual(row.training_coverage, 0)
        self.assertTrue(np.isnan(row.training_median))

    def test_empty_effective_feature_set_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "No valid features"):
            fit_preprocessor(pd.DataFrame({"simple_return": [1., 1., 1.]}), ("simple_return",))

    def test_invalid_numeric_or_infinite_training_feature_fails(self):
        for values in ([True, False], ["1", "2"], [1., np.inf], [1 + 1j, 2 + 1j]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_preprocessor(pd.DataFrame({"simple_return": values}), ("simple_return",))

    def test_validation_missing_uses_fixed_median_without_neighbor_fill(self):
        preprocessing = fit_preprocessor(pd.DataFrame({"simple_return": [1., 2., 3., 4.]}), ("simple_return",))
        result = transform_features(preprocessing, pd.DataFrame({"simple_return": [np.nan, 100., np.nan]}))
        self.assertEqual(result[0, 0], result[2, 0])
        self.assertAlmostEqual(result[0, 0], 0.0)

    def test_fit_and_transform_preserve_source_frames(self):
        training = pd.DataFrame({"simple_return": [1., 2., np.nan, 100.]})
        validation = pd.DataFrame({"simple_return": [3., np.nan]})
        train_before, valid_before = training.copy(deep=True), validation.copy(deep=True)
        preprocessing = fit_preprocessor(training, tuple(training))
        transform_features(preprocessing, validation)
        pd.testing.assert_frame_equal(training, train_before, check_exact=True)
        pd.testing.assert_frame_equal(validation, valid_before, check_exact=True)


class MetricTests(unittest.TestCase):
    def test_mae_known_values(self):
        self.assertAlmostEqual(regression_metrics([1., 2., 3.], [1., 3., 2.])["MAE"], 2 / 3)

    def test_rmse_known_values(self):
        self.assertAlmostEqual(regression_metrics([1., 2., 3.], [1., 3., 2.])["RMSE"], np.sqrt(2 / 3))

    def test_r2_known_and_negative_values(self):
        self.assertEqual(regression_metrics([1., 2., 3.], [1., 3., 2.])["R2"], 0.)
        self.assertEqual(regression_metrics([-1., 0., 1.], [3., 3., 3.])["R2"], -13.5)

    def test_pearson_known_values(self):
        self.assertAlmostEqual(regression_metrics([1., 2., 3.], [1., 3., 2.])["Pearson"], 0.5)

    def test_spearman_known_values_and_average_ties(self):
        self.assertAlmostEqual(regression_metrics([1., 2., 3.], [1., 3., 2.])["Spearman"], 0.5)
        a, b = np.array([1.5, 1.5, 3., 4.]), np.array([1., 2.5, 2.5, 4.])
        expected = sum((a - a.mean()) * (b - b.mean())) / np.sqrt(sum((a - a.mean()) ** 2) * sum((b - b.mean()) ** 2))
        self.assertAlmostEqual(regression_metrics([1., 1., 3., 4.], [1., 2., 2., 4.])["Spearman"], expected)

    def test_directional_accuracy_zero_is_own_category(self):
        self.assertEqual(regression_metrics([-2., 0., 3., 0.], [-1., 1., 0., 0.])["directional_accuracy"], 0.5)

    def test_constant_prediction_correlations_are_nan(self):
        for prediction in ([0., 0., 0.], [0.1, 0.1, 0.1]):
            result = regression_metrics([-.1, .2, .3], prediction)
            self.assertTrue(np.isnan(result["Pearson"]))
            self.assertTrue(np.isnan(result["Spearman"]))
            self.assertTrue(np.isfinite(result["MAE"]))

    def test_constant_actual_r2_and_correlations_undefined(self):
        result = regression_metrics([1., 1., 1.], [0., 1., 2.])
        for name in ("R2", "Pearson", "Spearman"):
            self.assertTrue(np.isnan(result[name]))

    def test_metrics_reject_missing_nonfinite_or_misaligned(self):
        for actual, prediction in (([1., np.nan], [1., 2.]), ([1., 2.], [1., np.inf]), ([1.], [1., 2.]), ([], [])):
            with self.assertRaises(ValueError):
                regression_metrics(actual, prediction)


class IndependentModelingOracleTests(unittest.TestCase):
    def test_one_feature_ols_manual_median_scale_and_closed_form(self):
        # Hand-specified train X becomes [1,2,2,100], median=2, mean=26.25.
        # Y=2*imputed_X+1, so raw predictions are exactly 2*x+1.
        training = pd.DataFrame({"simple_return": [1., 2., np.nan, 100.]}, index=["a", "b", "c", "d"])
        target = pd.Series([3., 5., 5., 201.], index=training.index)
        validation = pd.DataFrame({"simple_return": [3., np.nan, 4.]})
        fitted = fit_linear("ols_market", training, target, ("simple_return",))
        scale = np.sqrt(((1 - 26.25) ** 2 + 2 * (2 - 26.25) ** 2 + (100 - 26.25) ** 2) / 4)
        self.assertEqual(fitted.preprocessing.medians["simple_return"], 2.)
        self.assertAlmostEqual(fitted.preprocessing.scaler.scale_[0], scale)
        self.assertAlmostEqual(fitted.estimator.coef_[0], 2 * scale)
        self.assertAlmostEqual(fitted.estimator.intercept_, 53.5)
        np.testing.assert_allclose(predict_linear(fitted, validation), [7., 5., 9.], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(target.to_numpy(), [3., 5., 5., 201.])

    def test_one_feature_ridge_fixed_alpha_manual_closed_form(self):
        training = pd.DataFrame({"simple_return": [1., 2., np.nan, 100.]})
        target = pd.Series([3., 5., 5., 201.])
        validation = pd.DataFrame({"simple_return": [3., np.nan, 4.]})
        fitted = fit_linear("ridge_market", training, target, ("simple_return",))
        # Standardized sum(z²)=4; beta shrinkage is 4/(4+alpha)=4/5.
        expected = 53.5 + 1.6 * (np.array([3., 2., 4.]) - 26.25)
        np.testing.assert_allclose(predict_linear(fitted, validation), expected, rtol=0, atol=1e-12)
        self.assertEqual(fitted.estimator.alpha, 1.0)
        self.assertTrue(fitted.estimator.fit_intercept)


class CVIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = fixture()
        cls.tables, cls.protocol, cls.records = build_cv(*cls.checkpoint)

    def assert_same_outputs(self, changed_data):
        changed = (changed_data, *self.checkpoint[1:])
        tables, protocol, records = build_cv(*changed)
        for name in self.tables:
            pd.testing.assert_frame_equal(tables[name], self.tables[name], check_exact=True)
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(records, self.records)
        self.assertTrue(validate_modeling(*changed, tables, protocol, records)["valid"])

    def test_gap_feature_and_all_target_mutations_do_not_change_cv(self):
        data = self.checkpoint[0].copy(deep=True)
        manifest = self.checkpoint[1]
        gap = set(manifest.loc[manifest.pre_test_gap, "date"])
        data.loc[data.date.isin(gap), [*FEATURE_GROUPS, "forward_return_1d", PRIMARY_TARGET, "forward_return_20d"]] = 1e9
        self.assert_same_outputs(data)

    def test_final_test_feature_extremes_do_not_change_cv(self):
        data = self.checkpoint[0].copy(deep=True)
        final = set(self.checkpoint[1].loc[self.checkpoint[1].partition.eq("final_test"), "date"])
        for i, name in enumerate(FEATURE_GROUPS):
            data.loc[data.date.isin(final), name] = 1e9 if i % 2 else -1e9
        self.assert_same_outputs(data)

    def test_final_test_target_mutation_does_not_change_cv(self):
        data = self.checkpoint[0].copy(deep=True)
        final = set(self.checkpoint[1].loc[self.checkpoint[1].partition.eq("final_test"), "date"])
        data.loc[data.date.isin(final), PRIMARY_TARGET] = -1e9
        self.assert_same_outputs(data)

    def test_secondary_targets_never_affect_outputs(self):
        data = self.checkpoint[0].copy(deep=True)
        data["forward_return_1d"] = 1e9
        data["forward_return_20d"] = np.nan
        self.assert_same_outputs(data)

    def test_unlabeled_tail_is_ignored_even_if_mutated_into_fake_labels(self):
        data = self.checkpoint[0].copy(deep=True)
        tail = set(self.checkpoint[1].loc[self.checkpoint[1].partition.eq("unlabeled_tail"), "date"])
        data.loc[data.date.isin(tail), [*FEATURE_GROUPS, PRIMARY_TARGET]] = 1e9
        self.assert_same_outputs(data)

    def test_validation_features_do_not_change_same_fold_fit(self):
        data = self.checkpoint[0].copy(deep=True)
        dates = self.tables["oof_predictions"].query("fold == 'cv_1'").date
        data.loc[data.date.isin(dates), list(FEATURE_GROUPS)] = 1000.
        altered, _, _ = build_cv(data, *self.checkpoint[1:])
        for name in ("feature_usage", "coefficients"):
            pd.testing.assert_frame_equal(altered[name].query("fold == 'cv_1'").reset_index(drop=True),
                                          self.tables[name].query("fold == 'cv_1'").reset_index(drop=True), check_exact=True)
        self.assertFalse(altered["oof_predictions"].query("fold == 'cv_1'").ols_market_prediction.equals(
            self.tables["oof_predictions"].query("fold == 'cv_1'").ols_market_prediction))

    def test_validation_y_changes_metrics_but_not_same_fold_fit_or_predictions(self):
        data = self.checkpoint[0].copy(deep=True)
        dates = self.tables["oof_predictions"].query("fold == 'cv_1'").date
        data.loc[data.date.isin(dates), PRIMARY_TARGET] += 0.5
        altered, _, _ = build_cv(data, *self.checkpoint[1:])
        for name in ("feature_usage", "coefficients"):
            pd.testing.assert_frame_equal(altered[name].query("fold == 'cv_1'").reset_index(drop=True),
                                          self.tables[name].query("fold == 'cv_1'").reset_index(drop=True), check_exact=True)
        predictions = [f"{name}_prediction" for name in MODELS]
        pd.testing.assert_frame_equal(altered["oof_predictions"].query("fold == 'cv_1'")[predictions],
                                      self.tables["oof_predictions"].query("fold == 'cv_1'")[predictions], check_exact=True)
        self.assertFalse(altered["cv_metrics"].query("fold == 'cv_1'").MAE.equals(self.tables["cv_metrics"].query("fold == 'cv_1'").MAE))

    def test_historical_mean_excludes_purged_validation_gap_and_test_targets(self):
        data = self.checkpoint[0].copy(deep=True)
        authority = cv_authority(*self.checkpoint[:4])
        train_dates = authority["cv_1"][0]
        data.loc[~data.date.isin(train_dates), PRIMARY_TARGET] = -1e9
        altered, _, _ = build_cv(data, *self.checkpoint[1:])
        first = altered["oof_predictions"].query("fold == 'cv_1'").historical_mean_prediction
        pd.testing.assert_series_equal(first, self.tables["oof_predictions"].query("fold == 'cv_1'").historical_mean_prediction, check_exact=True)
        data.loc[data.date.isin(train_dates), PRIMARY_TARGET] += 0.25
        changed, _, _ = build_cv(data, *self.checkpoint[1:])
        np.testing.assert_allclose(changed["oof_predictions"].query("fold == 'cv_1'").historical_mean_prediction, first + 0.25)

    def test_actual_fit_calls_receive_only_authorized_training_dates(self):
        authority = cv_authority(*self.checkpoint[:4])
        expected = [tuple(train) for train, _ in authority.values() for _ in range(4)]
        actual = []
        original = fit_linear
        def spy(model, training, target, candidates):
            actual.append(tuple(training.index))
            self.assertTrue(training.index.equals(target.index))
            return original(model, training, target, candidates)
        with patch("nasdaq_research.modeling.fit_linear", side_effect=spy):
            build_cv(*self.checkpoint)
        self.assertEqual(actual, expected)

    def test_oof_is_75_unique_chronological_validation_rows(self):
        oof = self.tables["oof_predictions"]
        self.assertEqual(len(oof), 75)
        self.assertEqual(oof.groupby("fold").size().tolist(), [25, 25, 25])
        self.assertFalse(oof.date.duplicated().any())
        self.assertEqual(oof.date.tolist(), sorted(oof.date))
        self.assertTrue(np.isfinite(oof[[f"{model}_prediction" for model in MODELS]].to_numpy()).all())

    def test_cv_outputs_preserve_source_and_authority(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        before = copy.deepcopy(checkpoint)
        build_cv(*checkpoint)
        for actual, expected in zip(checkpoint, before):
            if isinstance(actual, pd.DataFrame):
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            else:
                self.assertEqual(actual, expected)

    def test_sample_std_and_pooled_metrics(self):
        row = self.tables["cv_summary"].query("model_name == 'historical_mean'").iloc[0]
        maes = self.tables["cv_metrics"].query("model_name == 'historical_mean'").MAE.to_numpy()
        expected_std = np.sqrt(sum((maes - sum(maes) / 3) ** 2) / 2)
        self.assertAlmostEqual(row.std_fold_MAE, expected_std)
        self.assertAlmostEqual(row.pooled_oof_MAE, row.mean_fold_MAE)


class ArtifactAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = fixture()
        cls.tables, cls.protocol, cls.records = build_cv(*cls.checkpoint)

    def audit(self, tables=None, protocol=None, records=None):
        return validate_modeling(*self.checkpoint, self.tables if tables is None else tables,
                                 self.protocol if protocol is None else protocol, self.records if records is None else records)

    def test_valid_outputs_and_closed_test_contract(self):
        report = self.audit()
        self.assertTrue(report["valid"], report.get("errors"))
        for name in ("target_columns_in_X", "target_provenance_columns_in_X", "metadata_columns_in_X",
                     "raw_non_research_columns_in_X", "pre_test_gap_rows_used", "final_test_rows_used", "unlabeled_tail_rows_used",
                     "training_only_coverage_violations", "training_only_imputation_violations", "training_only_scaling_violations",
                     "historical_mean_training_only_violations"):
            self.assertEqual(report[name], 0)
        self.assertFalse(report["final_test_predictions_generated"])
        self.assertFalse(report["final_test_metrics_computed"])

    def test_preprocessing_tampering_detected(self):
        for column, counter in (("training_coverage", "training_only_coverage_violations"),
                                ("training_median", "training_only_imputation_violations"),
                                ("scaler_scale", "training_only_scaling_violations")):
            tables = copy.deepcopy(self.tables)
            tables["feature_usage"].loc[0, column] += 0.2
            report = self.audit(tables=tables)
            self.assertFalse(report["valid"])
            self.assertGreater(report[counter], 0)

    def test_foreign_feature_in_x_is_detected(self):
        tables = copy.deepcopy(self.tables)
        tables["coefficients"].loc[0, "feature_name"] = "target_entry_open"
        report = self.audit(tables=tables)
        self.assertFalse(report["valid"])
        self.assertEqual(report["target_provenance_columns_in_X"], 1)

    def test_forbidden_fit_date_is_detected(self):
        records = copy.deepcopy(self.records)
        gap_date = self.checkpoint[1].loc[self.checkpoint[1].pre_test_gap, "date"].iloc[0]
        records[2]["training_dates"].append(gap_date)
        report = self.audit(records=records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["pre_test_gap_rows_used"], 1)

    def test_prediction_tampering_and_nonfinite_detected(self):
        for value in (0.5, np.nan, np.inf):
            tables = copy.deepcopy(self.tables)
            tables["oof_predictions"].loc[0, "ols_market_prediction"] = value
            self.assertFalse(self.audit(tables=tables)["valid"])

    def test_metric_and_summary_tampering_detected(self):
        for name, column in (("cv_metrics", "MAE"), ("cv_summary", "mean_fold_MAE")):
            tables = copy.deepcopy(self.tables)
            tables[name].loc[0, column] += 0.5
            self.assertFalse(self.audit(tables=tables)["valid"])

    def test_protocol_changes_to_target_alpha_metric_or_test_rejected(self):
        for key, value in (("primary_target", "forward_return_1d"), ("ridge_alpha", 10.),
                           ("primary_metric", "RMSE"), ("final_test_predictions_allowed", True)):
            protocol = copy.deepcopy(self.protocol)
            protocol[key] = value
            self.assertFalse(self.audit(protocol=protocol)["valid"])

    def test_exact_ties_reported_together(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint[0].loc[checkpoint[0][PRIMARY_TARGET].notna(), PRIMARY_TARGET] = 0.0
        tables, protocol, records = build_cv(*checkpoint)
        report = validate_modeling(*checkpoint, tables, protocol, records)
        self.assertTrue(report["valid"], report.get("errors"))
        self.assertIsNone(report["best_stage10_cv_candidate"])
        self.assertEqual(report["best_stage10_cv_candidates"], sorted(MODELS))
        self.assertTrue(tables["cv_summary"].rank_by_mean_fold_MAE.eq(1).all())


class AuthorityTests(unittest.TestCase):
    def test_manifest_summary_count_mismatch_fails(self):
        checkpoint = fixture()
        checkpoint[3].loc[checkpoint[3].target_name.eq(PRIMARY_TARGET) & checkpoint[3].fold.eq("cv_1"), "train_count_effective"] += 1
        with self.assertRaisesRegex(ValueError, "authoritative"):
            build_cv(*checkpoint)

    def test_gap_test_or_tail_cannot_be_injected_into_train(self):
        for selection in ("pre_test_gap", "final_test", "unlabeled_tail"):
            checkpoint = fixture()
            manifest = checkpoint[1]
            mask = manifest.target_name.eq(PRIMARY_TARGET) & manifest.fold.eq("cv_1")
            selected = mask & (manifest.pre_test_gap if selection == "pre_test_gap" else manifest.partition.eq(selection))
            index = manifest.index[selected][0]
            manifest.loc[index, "role"] = "train"
            with self.assertRaises(ValueError):
                build_cv(*checkpoint)

    def test_unlocked_test_or_changed_primary_protocol_fails(self):
        for key, value in (("final_test_locked", False), ("primary_target", "forward_return_1d"), ("primary_metric", "RMSE")):
            checkpoint = fixture()
            checkpoint[2][key] = value
            with self.assertRaisesRegex(ValueError, "locked"):
                build_cv(*checkpoint)

    def test_duplicate_manifest_roles_and_unsorted_dates_fail(self):
        checkpoint = fixture()
        duplicated = pd.concat([checkpoint[1], checkpoint[1].iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "unique roles"):
            build_cv(checkpoint[0], duplicated, *checkpoint[2:])
        with self.assertRaisesRegex(ValueError, "ascending"):
            build_cv(checkpoint[0].iloc[::-1], *checkpoint[1:])


class PipelineTests(unittest.TestCase):
    def test_offline_byte_reproducibility_upstream_hashes_and_standard_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = checkpoint_files(Path(tmp))
            initial = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(tmp).rglob("*") if path.is_file()}
            with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
                first = run_pipeline(**args)
                contents = {path: path.read_bytes() for path in args["output_dir"].iterdir()}
                second = run_pipeline(**args)
            self.assertEqual(len(contents), 7)
            self.assertEqual(contents, {path: path.read_bytes() for path in args["output_dir"].iterdir()})
            self.assertEqual(first["output_artifact_sha256"], second["output_artifact_sha256"])
            self.assertTrue(first["valid"])
            for path, digest in initial.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            report = json.loads((args["output_dir"] / "NVDA_baseline_validation.json").read_text(),
                                parse_constant=lambda token: self.fail(f"Invalid JSON token {token}"))
            self.assertFalse(report["final_test_predictions_generated"])
            self.assertFalse(report["final_test_metrics_computed"])
            self.assertFalse(any(path.suffix == ".tmp" for path in args["output_dir"].iterdir()))

    def test_changed_labeled_or_manifest_checkpoint_rejected_before_output(self):
        for selection in ("labeled", "manifest"):
            with tempfile.TemporaryDirectory() as tmp:
                args = checkpoint_files(Path(tmp))
                path = args["labeled_path"] if selection == "labeled" else args["split_dir"] / "NVDA_split_manifest.csv"
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "hash"):
                    run_pipeline(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_existing_stage10_contract_cannot_change_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = checkpoint_files(Path(tmp))
            run_pipeline(**args)
            path = args["output_dir"] / "NVDA_baseline_protocol.json"
            protocol = json.loads(path.read_text())
            protocol["ridge_alpha"] = 100.
            path.write_text(json.dumps(protocol))
            contents = {p: p.read_bytes() for p in args["output_dir"].iterdir()}
            with self.assertRaisesRegex(ValueError, "contract is locked"):
                run_pipeline(**args)
            self.assertEqual(contents, {p: p.read_bytes() for p in args["output_dir"].iterdir()})

    def test_output_symlink_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = checkpoint_files(Path(tmp))
            args["output_dir"].mkdir()
            (args["output_dir"] / "NVDA_baseline_oof_predictions.csv").symlink_to(args["labeled_path"])
            before = args["labeled_path"].read_bytes()
            with self.assertRaisesRegex(ValueError, "symlinks"):
                run_pipeline(**args)
            self.assertEqual(args["labeled_path"].read_bytes(), before)

    def test_cli_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = checkpoint_files(Path(tmp))
            command = [sys.executable, "-m", "nasdaq_research.modeling"]
            for key, value in args.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("OOF rows: 75", result.stdout)
            self.assertIn("Final test predictions generated: false", result.stdout)
            self.assertIn("Validation: passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
