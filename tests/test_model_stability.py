"""Independent stability oracles, frozen-reference audits and scope mutations."""

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

from nasdaq_research import modeling as baseline
from nasdaq_research import model_stability as stability
from test_modeling import fixture as modeling_fixture, checkpoint_files


def add_state_metadata(data):
    data = data.copy(deep=True)
    dates = pd.to_datetime(data.date)
    for group, starts in (("q", [0, 100, 165]), ("fy", [0, 100]), ("bs", [0, 100, 165])):
        state_start = np.array([max(start for start in starts if start <= i) for i in range(len(data))])
        effective = pd.Series([dates.iloc[start] for start in state_start], index=data.index)
        data[f"{group}_accession"] = [f"{group}-source-{start}" for start in state_start]
        data[f"{group}_filing_date"] = (effective - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        data[f"{group}_effective_date"] = effective.dt.strftime("%Y-%m-%d")
        data[f"{group}_period_start"] = "2024-01-01"
        data[f"{group}_period_end"] = "2024-12-31"
        data[f"{group}_data_period_type"] = "annual" if group == "fy" else "quarterly"
        data[f"{group}_days_since_effective_date"] = (dates - effective).dt.days
        data[f"{group}_days_since_filing"] = data[f"{group}_days_since_effective_date"] + 1
        data[f"{group}_effective_date_is_sample_truncated"] = state_start == 0
        data[f"{group}_effective_date_basis"] = "sample-start" if group == "fy" else "observed-session"
    return data


def fixture():
    checkpoint = modeling_fixture()
    checkpoint = (add_state_metadata(checkpoint[0]), *checkpoint[1:])
    tables, protocol, records = baseline.build_cv(*checkpoint)
    validation = baseline.validate_modeling(*checkpoint, tables, protocol, records)
    return checkpoint, tables, protocol, validation


def files(root):
    args = checkpoint_files(root)
    data = pd.read_csv(args["labeled_path"], converters={baseline.PRIMARY_TARGET: lambda value: float(value) if value else np.nan})
    add_state_metadata(data).to_csv(args["labeled_path"], index=False)
    digest = hashlib.sha256(args["labeled_path"].read_bytes()).hexdigest()
    protocol_path = args["split_dir"] / "NVDA_evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["source_labeled_sha256"] = digest
    protocol_path.write_text(json.dumps(protocol))
    validation_path = args["split_dir"] / "NVDA_split_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["source_labeled_sha256"] = digest
    validation["output_artifact_sha256"][str(protocol_path)] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    validation_path.write_text(json.dumps(validation))
    baseline.run_pipeline(**args)
    return {**args, "baseline_dir": args["output_dir"], "output_dir": root / "stability", "generate_figures": False}


class RangeOracleTests(unittest.TestCase):
    def test_constant_sample_std_exact_zero_and_singleton_undefined(self):
        self.assertEqual(stability._stats(np.full(25, .7), "actual")["actual_std"], 0.)
        self.assertTrue(np.isnan(stability._stats(np.array([.7]), "actual")["actual_std"]))

    def test_independent_strict_boundary_oracle(self):
        result = stability.feature_range(pd.Series([0., 10.]), pd.Series([-1., 0., 5., 10., 11.]))
        self.assertEqual(result["validation_below_training_min_count"], 1)
        self.assertEqual(result["validation_above_training_max_count"], 1)
        self.assertEqual(result["validation_outside_training_range_count"], 2)
        self.assertEqual(result["validation_outside_training_range_fraction"], .4)

    def test_missing_excluded_from_range_denominator(self):
        result = stability.feature_range(pd.Series([np.nan, 0., 10.]), pd.Series([-1., np.nan, 11.]))
        self.assertEqual(result["validation_outside_training_range_count"], 2)
        self.assertEqual(result["validation_outside_training_range_fraction"], 1.)

    def test_undefined_training_range_is_nan(self):
        result = stability.feature_range(pd.Series([np.nan]), pd.Series([3.]))
        self.assertTrue(all(np.isnan(value) for value in result.values()))

    def test_all_missing_validation_has_zero_counts_nan_fraction(self):
        result = stability.feature_range(pd.Series([0., 10.]), pd.Series([np.nan]))
        self.assertEqual(result["validation_outside_training_range_count"], 0)
        self.assertTrue(np.isnan(result["validation_outside_training_range_fraction"]))


class ZscoreContributionOracleTests(unittest.TestCase):
    def test_independent_two_feature_ols_z_coefficients_predictions(self):
        # Orthogonal, centered columns: each population std is one.
        training = pd.DataFrame({"simple_return": [-1., -1., 1., 1.], "log_return": [-1., 1., -1., 1.]})
        target = pd.Series([.02, -.04, .06, 0.], index=training.index)
        validation = pd.DataFrame({"simple_return": [2., np.nan], "log_return": [1., -2.]}, index=["2025-01-10", "2025-01-13"])
        fitted = baseline.fit_linear("ols_market", training, target, tuple(training))
        # Direct normal equation is independent of production diagnostics/sklearn.
        design = np.column_stack([np.ones(4), training.to_numpy()])
        expected_coefficients = np.linalg.inv(design.T @ design) @ design.T @ target.to_numpy()
        np.testing.assert_allclose(expected_coefficients, [.01, .02, -.03], rtol=0, atol=1e-15)
        np.testing.assert_allclose(fitted.estimator.coef_, expected_coefficients[1:], rtol=0, atol=1e-15)
        z = stability.validation_zscores(fitted, validation, "cv_1", "market")
        expected_z = {("2025-01-10", "simple_return"): 2., ("2025-01-10", "log_return"): 1.,
                      ("2025-01-13", "simple_return"): 0., ("2025-01-13", "log_return"): -2.}
        for row in z.itertuples():
            self.assertEqual(row.validation_zscore, expected_z[row.date, row.feature_name])
        predictions = pd.Series([.02, .07], index=validation.index)
        contributions = stability.prediction_contributions(z, pd.Series([.02, -.03], index=training.columns), .01, predictions, "ols_market")
        for day, expected in predictions.items():
            rows = contributions.loc[contributions.date.eq(day)]
            self.assertAlmostEqual(.01 + rows.feature_contribution.sum(), expected)
        np.testing.assert_allclose(contributions.feature_contribution, [.04, 0., -.03, .06], rtol=0, atol=1e-15)

    def test_fixed_median_scale_ignore_validation_extremes(self):
        training = pd.DataFrame({"simple_return": [1., 2., np.nan, 100.]}, index=list("abcd"))
        target = pd.Series([3., 5., 5., 201.], index=training.index)
        fitted = baseline.fit_linear("ols_market", training, target, ("simple_return",))
        before = copy.deepcopy(fitted)
        validation = pd.DataFrame({"simple_return": [1e9, np.nan]}, index=["2025-01-10", "2025-01-13"])
        z = stability.validation_zscores(fitted, validation, "cv_1", "market")
        filled = np.array([1., 2., 2., 100.])
        expected = (np.array([1e9, 2.]) - 26.25) / np.sqrt(sum((filled - 26.25) ** 2) / 4)
        np.testing.assert_allclose(z.validation_zscore, expected, rtol=0, atol=1e-12)
        self.assertTrue(stability._fit_equal(fitted, before))
        self.assertTrue(z.was_validation_missing.iloc[1])

    def test_zscore_strict_thresholds_and_linear_p95(self):
        result = stability.zscore_summary(np.array([-5., -3., 0., 3.01, 5.01]))
        self.assertEqual(result["count_abs_z_gt_3"], 3)
        self.assertEqual(result["count_abs_z_gt_5"], 1)
        self.assertEqual(result["max_abs_validation_z"], 5.01)
        self.assertAlmostEqual(result["p95_abs_validation_z"], 5.008)

    def test_zscore_empty_and_nonfinite_rejected(self):
        for values in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                stability.zscore_summary(np.array(values))

    def test_contribution_positive_negative_and_absolute_ranks(self):
        z = pd.DataFrame({"date": ["2025-01-01"] * 3, "fold": "cv_1", "feature_spec": "market",
                          "feature_name": ["a", "b", "c"], "validation_zscore": [2., 1., 0.]})
        result = stability.prediction_contributions(z, pd.Series({"a": .02, "b": -.03, "c": 1.}), .01,
                                                   pd.Series({"2025-01-01": .02}), "ols_market")
        self.assertEqual(result.contribution_rank_by_absolute_value.tolist(), [1., 2., 3.])
        self.assertEqual(result.positive_contribution_rank.iloc[0], 1.)
        self.assertEqual(result.negative_contribution_rank.iloc[1], 1.)
        self.assertTrue(result.positive_contribution_rank.iloc[2:].isna().all())

    def test_missing_used_coefficient_rejected(self):
        z = pd.DataFrame({"date": ["2025-01-01"], "fold": "cv_1", "feature_spec": "market", "feature_name": ["a"], "validation_zscore": [1.]})
        with self.assertRaisesRegex(ValueError, "coefficients"):
            stability.prediction_contributions(z, pd.Series({"b": 1.}), 0., pd.Series({"2025-01-01": 1.}), "ols_market")


class CoefficientOracleTests(unittest.TestCase):
    def summary(self, values):
        rows = [{"model_name": "ols_market", "feature_name": "simple_return", "fold": fold, "coefficient": value}
                for fold, value in zip(stability.FOLDS, values) if value is not None]
        coefficients = pd.DataFrame(rows, columns=["model_name", "feature_name", "fold", "coefficient"])
        result = stability.coefficient_stability(coefficients, {"market": ("simple_return",), "all": ("simple_return",)})
        return result.loc[result.model_name.eq("ols_market")].iloc[0]

    def test_positive_to_positive_no_flip(self):
        row = self.summary([1., 2., 3.])
        self.assertEqual(row.sign_flip_count_across_consecutive_available_folds, 0)
        self.assertTrue(row.sign_consistent_across_available_folds)
        self.assertEqual(row.coefficient_std_available, 1.)

    def test_positive_negative_flip(self):
        self.assertEqual(self.summary([1., -2., -3.]).sign_flip_count_across_consecutive_available_folds, 1)

    def test_missing_fold_stays_nan_and_is_skipped(self):
        row = self.summary([1., None, -1.])
        self.assertTrue(np.isnan(row.coefficient_cv_2))
        self.assertTrue(np.isnan(row.sign_cv_2))
        self.assertEqual(row.folds_present, 2)
        self.assertEqual(row.sign_flip_count_across_consecutive_available_folds, 1)

    def test_exact_zero_policy_does_not_create_opposite_sign_flip(self):
        row = self.summary([1., 0., -1.])
        self.assertEqual(row.sign_cv_2, 0.)
        self.assertEqual(row.sign_flip_count_across_consecutive_available_folds, 0)
        self.assertFalse(row.sign_consistent_across_available_folds)

    def test_never_used_feature_has_no_fake_zero_coefficient(self):
        row = self.summary([None, None, None])
        self.assertEqual(row.folds_present, 0)
        self.assertTrue(np.isnan(row.max_abs_coefficient))
        self.assertTrue(np.isnan(row.sign_consistent_across_available_folds))


class ConditioningInfluenceOracleTests(unittest.TestCase):
    def test_full_rank_known_singular_values_condition(self):
        result = stability.conditioning(np.diag([3., 2., 1.]))
        self.assertEqual(result["matrix_rank"], 3)
        self.assertFalse(result["rank_deficient"])
        self.assertEqual(result["largest_singular_value"], 3.)
        self.assertEqual(result["smallest_singular_value"], 1.)
        self.assertEqual(result["condition_number"], 3.)
        self.assertEqual(result["n_to_p_ratio"], 1.)

    def test_perfect_collinearity_rank_and_infinity(self):
        result = stability.conditioning(np.array([[-1., -2.], [0., 0.], [1., 2.]]))
        self.assertEqual(result["matrix_rank"], 1)
        self.assertTrue(result["rank_deficient"])
        self.assertTrue(np.isinf(result["condition_number"]))
        self.assertAlmostEqual(result["largest_singular_value"], np.sqrt(10.))
        self.assertAlmostEqual(result["max_abs_pairwise_correlation"], 1.)
        self.assertEqual(result["high_corr_pair_count_abs_ge_0_95"], 1)

    def test_high_correlation_pairs_count_once(self):
        matrix = np.array([[-2., -4., -6.], [-1., -2., -3.], [1., 2., 3.], [2., 4., 6.]])
        self.assertEqual(stability.conditioning(matrix)["high_corr_pair_count_abs_ge_0_95"], 3)

    def test_independent_leverage_cooks_and_studentization(self):
        x = np.array([0., 1., 2., 3.])
        actual = np.array([1., 2., 1., 4.])
        fitted = .8 + .8 * x  # Hand-computed OLS intercept/slope.
        residual = actual - fitted
        h = .25 + (x - 1.5) ** 2 / 5
        mse = 1.4  # SSE 2.8 / residual df 2.
        expected_cooks = residual ** 2 / (2 * mse) * h / (1 - h) ** 2
        result = stability.ols_influence(x[:, None], actual, fitted, list("abcd"), "cv_1", "market")
        np.testing.assert_allclose(result.leverage, [.7, .3, .3, .7], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.cooks_distance, expected_cooks, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.studentized_residual, residual / np.sqrt(mse * (1 - h)), rtol=0, atol=1e-14)
        self.assertTrue(result.undefined_reason.eq("defined").all())
        self.assertTrue(result.residual_df.eq(2).all())

    def test_rank_deficient_influence_uses_projection_and_rank_df(self):
        x = np.array([0., 1., 2., 3.])
        result = stability.ols_influence(np.column_stack([x, 2 * x]), np.array([1., 2., 1., 4.]), .8 + .8 * x, list("abcd"), "cv_1", "all")
        np.testing.assert_allclose(result.leverage, [.7, .3, .3, .7], rtol=0, atol=1e-14)
        self.assertTrue(result.intercept_design_rank.eq(2).all())
        self.assertTrue(result.residual_df.eq(2).all())

    def test_zero_residual_variance_has_explicit_nan_reason(self):
        x = np.arange(4.)[:, None]
        result = stability.ols_influence(x, np.arange(4.), np.arange(4.), list("abcd"), "cv_1", "market")
        self.assertTrue(result.studentized_residual.isna().all())
        self.assertTrue(result.cooks_distance.isna().all())
        self.assertTrue(result.undefined_reason.eq("zero_residual_variance").all())

    def test_nonpositive_df_has_explicit_nan_reason(self):
        result = stability.ols_influence(np.eye(2), np.array([0., 1.]), np.array([0., 1.]), list("ab"), "cv_1", "market")
        self.assertTrue(result.undefined_reason.eq("nonpositive_residual_df").all())
        self.assertTrue(result.cooks_distance.isna().all())

    def test_unit_leverage_has_explicit_nan_reason(self):
        x = np.array([[1.], [0.], [0.]])
        result = stability.ols_influence(x, np.array([1., -1., 1.]), np.array([1., 0., 0.]), list("abc"), "cv_1", "market")
        self.assertEqual(result.undefined_reason.iloc[0], "unit_leverage")
        self.assertTrue(np.isnan(result.cooks_distance.iloc[0]))


class DiagnosticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixture()
        cls.diagnostics = stability.build_diagnostics(*cls.inputs)

    def audit(self, diagnostics=None):
        return stability.validate_diagnostics(*self.inputs, self.diagnostics if diagnostics is None else diagnostics)

    def test_complete_scope_and_reference_consistency(self):
        report = self.audit()
        self.assertTrue(report["valid"], report.get("audit_error", report["errors"]))
        self.assertEqual(report["row_error_count"], 450)
        for name in stability.VIOLATIONS:
            self.assertEqual(report[name], 0, name)
        self.assertEqual(report["pre_test_gap_rows_used"], 0)
        self.assertEqual(report["final_test_rows_used"], 0)
        self.assertEqual(report["unlabeled_tail_rows_used"], 0)
        self.assertFalse(report["final_test_predictions_generated"])
        self.assertFalse(report["final_test_metrics_computed"])

    def test_baselines_have_no_feature_diagnostics(self):
        for name in ("contributions", "coefficient_summary", "influence"):
            self.assertFalse(self.diagnostics.tables[name].model_name.isin(baseline.MODELS[:2]).any())
        self.assertEqual(len(self.diagnostics.tables["prediction_summary"]), 18)

    def test_dropped_features_have_no_zscore_or_contribution(self):
        shift = self.diagnostics.tables["feature_shift"]
        for row in shift.loc[shift.used_in_stage10_fold.eq(False)].itertuples():
            z = self.diagnostics.tables["zscores"]
            self.assertFalse((z.fold.eq(row.fold) & z.feature_spec.eq(row.feature_spec) & z.feature_name.eq(row.feature_name)).any())
            self.assertTrue(np.isnan(row.max_abs_validation_z))

    def test_saved_predictions_and_metrics_tampering_fail(self):
        for name, column in (("oof_predictions", "ols_market_prediction"), ("cv_metrics", "MAE")):
            inputs = copy.deepcopy(self.inputs)
            inputs[1][name].loc[0, column] += .01
            self.assertFalse(stability.validate_diagnostics(*inputs, self.diagnostics)["valid"])

    def test_diagnostic_artifact_tampering_detected(self):
        for table, column in (("row_errors", "residual"), ("prediction_summary", "prediction_mean"),
                              ("feature_shift", "validation_outside_training_range_count"), ("zscores", "validation_zscore"),
                              ("contributions", "feature_contribution"), ("coefficient_summary", "coefficient_cv_1"),
                              ("conditioning", "matrix_rank"), ("influence", "leverage")):
            changed = copy.deepcopy(self.diagnostics)
            changed.tables[table][column] = changed.tables[table][column].astype(float)
            changed.tables[table].loc[0, column] += .5
            with self.subTest(table=table):
                self.assertFalse(self.audit(changed)["valid"])

    def test_summary_tampering_detected(self):
        changed = copy.deepcopy(self.diagnostics)
        changed.summary["coefficient_norms"][0]["L2_coefficient_norm"] += 1
        self.assertFalse(self.audit(changed)["valid"])

    def test_fit_date_and_foreign_state_feature_detected(self):
        changed = copy.deepcopy(self.diagnostics)
        manifest = self.inputs[0][1]
        gap = manifest.loc[manifest.pre_test_gap, "date"].iloc[0]
        changed.fit_records[0]["training_dates"].append(gap)
        report = self.audit(changed)
        self.assertFalse(report["valid"])
        self.assertEqual(report["pre_test_gap_rows_used"], 1)

    def test_whitelist_never_contains_state_metadata(self):
        for fitted in self.diagnostics.fits.values():
            self.assertFalse(set(fitted.preprocessing.features) & set(stability.STATE_METADATA_COLUMNS))
            self.assertLessEqual(set(fitted.preprocessing.features), set(baseline.FEATURE_GROUPS))

    def test_changed_protocol_rejected(self):
        for key, value in (("ridge_alpha", 10.), ("primary_target", "forward_return_1d"), ("training_coverage_threshold", .8),
                           ("final_test_predictions_allowed", True)):
            inputs = copy.deepcopy(self.inputs)
            inputs[2][key] = value
            with self.assertRaisesRegex(ValueError, "Frozen Stage 10"):
                stability.build_diagnostics(*inputs)

    def test_source_frames_and_reference_tables_unchanged(self):
        before = copy.deepcopy(self.inputs)
        stability.build_diagnostics(*self.inputs)
        for actual, expected in zip(self.inputs[0], before[0]):
            if isinstance(actual, pd.DataFrame):
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            else:
                self.assertEqual(actual, expected)
        for name in self.inputs[1]:
            pd.testing.assert_frame_equal(self.inputs[1][name], before[1][name], check_exact=True)

    def test_all_forbidden_scope_and_validation_mutations(self):
        result = stability.isolation_checks(*self.inputs, self.diagnostics)
        self.assertEqual(len(result), 9)
        self.assertTrue(all(value == 0 for value in result.values()), result)

    def test_validation_x_changes_z_prediction_but_not_same_fold_training(self):
        checkpoint, tables, protocol, validation = copy.deepcopy(self.inputs)
        day = tables["oof_predictions"].loc[tables["oof_predictions"].fold.eq("cv_1"), "date"].iloc[0]
        checkpoint[0].loc[checkpoint[0].date.eq(day), "simple_return"] += 1000.
        changed = stability.build_diagnostics(checkpoint, tables, protocol, validation)
        for model in baseline.LINEAR_MODELS:
            self.assertTrue(stability._fit_equal(self.diagnostics.fits["cv_1", model], changed.fits["cv_1", model]))
        original_z = self.diagnostics.tables["zscores"].loc[lambda f: f.fold.eq("cv_1") & f.date.eq(day)]
        changed_z = changed.tables["zscores"].loc[lambda f: f.fold.eq("cv_1") & f.date.eq(day)]
        self.assertFalse(stability._same_frame(original_z, changed_z))
        original_predictions = self.diagnostics.tables["row_errors"].loc[lambda f: f.fold.eq("cv_1") & f.date.eq(day)].prediction
        changed_predictions = changed.tables["row_errors"].loc[lambda f: f.fold.eq("cv_1") & f.date.eq(day)].prediction
        self.assertFalse(original_predictions.equals(changed_predictions))

    def test_validation_y_changes_only_same_fold_error_context(self):
        checkpoint, tables, protocol, validation = copy.deepcopy(self.inputs)
        day = tables["oof_predictions"].loc[tables["oof_predictions"].fold.eq("cv_1"), "date"].iloc[0]
        checkpoint[0].loc[checkpoint[0].date.eq(day), baseline.PRIMARY_TARGET] += .5
        changed = stability.build_diagnostics(checkpoint, tables, protocol, validation)
        for name in ("zscores", "contributions", "conditioning", "influence"):
            self.assertTrue(stability._same_frame(self.diagnostics.tables[name].loc[lambda f: f.fold.eq("cv_1")], changed.tables[name].loc[lambda f: f.fold.eq("cv_1")]))
        first = self.diagnostics.tables["row_errors"].loc[lambda f: f.fold.eq("cv_1") & f.date.eq(day)]
        second = changed.tables["row_errors"].loc[lambda f: f.fold.eq("cv_1") & f.date.eq(day)]
        np.testing.assert_array_equal(first.prediction, second.prediction)
        np.testing.assert_allclose(second.residual.to_numpy() - first.residual.to_numpy(), .5, rtol=0, atol=1e-12)

    def test_state_context_immediate_previous_session_including_purged(self):
        checkpoint, tables, _, _ = self.inputs
        state = self.diagnostics.tables["state_context"]
        dates = checkpoint[0].date.tolist()
        for row in state.itertuples():
            previous = dates[dates.index(row.date) - 1]
            self.assertEqual(row.previous_observed_date, previous)
        first = state.iloc[0]
        data = checkpoint[0].set_index("date")
        expected = data.loc[first.date, "fy_accession"] != data.loc[first.previous_observed_date, "fy_accession"]
        self.assertEqual(first.fy_state_changed_since_previous_observed_session, expected)

    def test_unavailable_metadata_explicit_unknown_not_fake_false(self):
        checkpoint, tables, protocol, validation = copy.deepcopy(self.inputs)
        checkpoint = (checkpoint[0].drop(columns=[c for c in stability.STATE_METADATA_COLUMNS if c in checkpoint[0]]), *checkpoint[1:])
        result = stability.build_diagnostics(checkpoint, tables, protocol, validation)
        self.assertIn("not reliably computable", result.summary["state_context_status"])
        self.assertTrue(result.tables["state_context"].q_state_identifier.isna().all())
        self.assertTrue(result.tables["state_context"].q_state_changed_since_previous_observed_session.isna().all())

    def test_independent_state_transition_uses_only_current_and_previous(self):
        data = pd.DataFrame({"date": ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"],
                             baseline.PRIMARY_TARGET: [0., 0., 0., 0.]})
        for group in stability.STATE_GROUPS:
            data[f"{group}_accession"] = ["A", "A", "B", "FUTURE"]
            for field in stability.STATE_ID_FIELDS[1:]:
                data[f"{group}_{field}"] = "same-source-field"
        errors = pd.DataFrame([{"date": "2025-01-08", "fold": "cv_1", "model_name": model,
                                "prediction": .01, "absolute_error": .01} for model in baseline.LINEAR_MODELS])
        authority = {"cv_1": ([], ["2025-01-08"])}
        result, _ = stability.state_context(data, authority, errors)
        self.assertEqual(result.previous_observed_date.iloc[0], "2025-01-07")
        self.assertTrue(result.q_state_changed_since_previous_observed_session.iloc[0])
        changed = data.copy()
        changed.loc[3, "q_accession"] = "UNKNOWN_FUTURE"
        again, _ = stability.state_context(changed, authority, errors)
        pd.testing.assert_frame_equal(result, again, check_exact=True)


class PipelineTests(unittest.TestCase):
    def test_offline_repeated_bytes_upstream_immutability_and_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            initial = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tmp).rglob("*") if p.is_file()}
            with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
                first = stability.run_pipeline(**args)
                contents = {p.name: p.read_bytes() for p in args["output_dir"].iterdir()}
                second = stability.run_pipeline(**args)
            self.assertEqual(len(contents), 11)
            self.assertEqual(contents, {p.name: p.read_bytes() for p in args["output_dir"].iterdir()})
            self.assertTrue(first["valid"])
            self.assertTrue(first["mutation_checks_executed"])
            self.assertTrue(second["stage10_baseline_artifacts_unchanged"])
            for path, digest in initial.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            for path in args["output_dir"].glob("*.json"):
                json.loads(path.read_text(), parse_constant=lambda token: self.fail(f"Invalid JSON {token}"))
            self.assertFalse(list(args["output_dir"].glob(".*.tmp")))

    def test_changed_stage10_baseline_rejected_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            path = args["baseline_dir"] / "NVDA_baseline_oof_predictions.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                stability.run_pipeline(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_outputs_cannot_contain_baseline_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            args["output_dir"] = args["baseline_dir"]
            with self.assertRaisesRegex(ValueError, "must not contain"):
                stability.run_pipeline(**args)

    def test_symlink_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            args["output_dir"].mkdir()
            (args["output_dir"] / "NVDA_stability_row_errors.csv").symlink_to(args["labeled_path"])
            initial = args["labeled_path"].read_bytes()
            with self.assertRaisesRegex(ValueError, "symlinks"):
                stability.run_pipeline(**args)
            self.assertEqual(initial, args["labeled_path"].read_bytes())

    def test_cli_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = files(Path(tmp))
            command = [sys.executable, "-m", "nasdaq_research.model_stability", "--no-figures"]
            for key, value in args.items():
                if key != "generate_figures":
                    command.extend(["--" + key.replace("_", "-"), str(value)])
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("OOF rows: 75; row errors: 450", result.stdout)
            self.assertIn("final test predictions generated: false", result.stdout)


if __name__ == "__main__":
    unittest.main()
