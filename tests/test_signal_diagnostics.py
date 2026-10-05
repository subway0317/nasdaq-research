"""Stage 12 scalar oracles, fixed-authority and directional-dependency tests."""

import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import modeling as baseline
from nasdaq_research import market_representation as stage11
from nasdaq_research import signal_diagnostics as signal
from test_market_representation import fixture as stage11_fixture, pipeline_files as stage11_files


def fixture():
    checkpoint = stage11_fixture()
    old, _, _ = baseline.build_cv(*checkpoint)
    experiment = stage11.build_experiment(*checkpoint[:4])
    contract = stage11.protocol(checkpoint[2], "test-manifest-hash", {})
    return signal.Inputs(*checkpoint[:4], experiment.tables, old, contract)


class ScalarOracleTests(unittest.TestCase):
    def test_calibration_closed_form_affine_exact_relation(self):
        p = np.array([-.2, -.1, .1, .2])
        y = .03 + 2*p
        result = signal.calibration(y, p)
        self.assertAlmostEqual(result["calibration_intercept"], .03, places=12)
        self.assertAlmostEqual(result["calibration_slope"], 2., places=12)
        self.assertAlmostEqual(result["mean_bias"], -.03, places=12)
        self.assertEqual(result["calibration_undefined_reason"], "defined")

    def test_calibration_negative_slope_and_correct_bias_sign(self):
        p, y = np.array([1., 2., 3.]), np.array([2., 0., -2.])
        result = signal.calibration(y, p)
        self.assertEqual(result["calibration_slope"], -2.)
        self.assertEqual(result["calibration_intercept"], 4.)
        self.assertEqual(result["mean_bias"], 2.)
        self.assertEqual(result["median_error"], 2.)
        self.assertEqual(result["median_absolute_error"], 2.)

    def test_constant_prediction_and_singleton_undefined_no_infinity(self):
        for y, p, reason in (([1., 2., 3.], [.1, .1, .1], "constant_prediction"), ([1.], [.1], "too_few_rows")):
            result = signal.calibration(np.array(y), np.array(p))
            self.assertTrue(np.isnan(result["calibration_slope"]))
            self.assertTrue(np.isnan(result["calibration_intercept"]))
            self.assertEqual(result["calibration_undefined_reason"], reason)

    def test_constant_actual_calibration_slope_zero_dispersion_undefined(self):
        result = signal.calibration(np.array([.7]*3), np.array([.1, .2, .3]))
        self.assertAlmostEqual(result["calibration_slope"], 0., places=12)
        self.assertAlmostEqual(result["calibration_intercept"], .7, places=12)
        dispersion = signal.dispersion(np.array([.7]*3), np.array([.1, .2, .3]))
        self.assertEqual(dispersion["std_actual"], 0.)
        self.assertTrue(np.isnan(dispersion["dispersion_ratio"]))
        self.assertEqual(dispersion["dispersion_undefined_reason"], "constant_actual")

    def test_sample_dispersion_and_original_fixed_thresholds(self):
        y, p = np.array([-.2, 0., .2]), np.array([-.3, .1, .5])
        result = signal.dispersion(y, p)
        self.assertAlmostEqual(result["std_actual"], .2, places=12)
        self.assertAlmostEqual(result["std_prediction"], .4, places=12)
        self.assertAlmostEqual(result["dispersion_ratio"], 2., places=12)
        self.assertEqual(result["count_abs_prediction_gt_0_10"], 2)
        self.assertEqual(result["count_abs_prediction_gt_0_30"], 1)

    def test_delta_sign_counts_ties_and_exact_boundary(self):
        delta = np.array([-.1, -1e-15, 0., 1e-15, .3])
        result = signal.advantage(delta, "zero")
        self.assertEqual(result["model_beats_zero_count"], 1)
        self.assertEqual(result["ties_zero_count"], 3)
        self.assertEqual(result["zero_beats_model_count"], 1)
        self.assertEqual(result["model_beats_zero_fraction"], .2)
        self.assertAlmostEqual(result["mean_delta_AE_zero"], .04, places=12)
        self.assertEqual(result["median_delta_AE_zero"], 0.)

    def test_pearson_and_tied_rank_spearman_independent_oracle(self):
        x, y = np.array([1., 2., 2., 4.]), np.array([3., 1., 2., 4.])
        def pearson(a, b):
            ma, mb = sum(a)/len(a), sum(b)/len(b)
            return sum((aa-ma)*(bb-mb) for aa, bb in zip(a, b))/np.sqrt(sum((aa-ma)**2 for aa in a)*sum((bb-mb)**2 for bb in b))
        info = signal.correlation(x, y)
        self.assertAlmostEqual(info["pearson"], pearson(x, y), places=12)
        self.assertAlmostEqual(info["spearman"], pearson([1., 2.5, 2.5, 4.], [3., 1., 2., 4.]), places=12)

    def test_undefined_associations_preserve_nan_and_reasons(self):
        for x, y, reason in (([1.], [2.], "too_few_pairs"), ([1., 1.], [1., 2.], "constant_feature"), ([1., 2.], [1., 1.], "constant_target")):
            info = signal.correlation(np.array(x), np.array(y))
            self.assertTrue(np.isnan(info["pearson"]))
            self.assertTrue(np.isnan(info["spearman"]))
            self.assertEqual(info["undefined_reason"], reason)

    def test_numeric_zero_sign_convention_and_nan(self):
        self.assertEqual(signal.signed(-1e-15), 0)
        self.assertEqual(signal.signed(1e-15), 0)
        self.assertEqual(signal.signed(-2e-15), -1)
        self.assertEqual(signal.signed(2e-15), 1)
        self.assertTrue(np.isnan(signal.signed(np.nan)))

    def test_overlap_inclusive_touching_and_irregular_dates_oracle(self):
        # A Friday/Tuesday shared date is a real session, not five calendar days.
        windows = pd.DataFrame({"entry_date": ["2026-01-02", "2026-01-06", "2026-01-12", "2026-01-15"],
                                "exit_date": ["2026-01-06", "2026-01-09", "2026-01-14", "2026-01-16"]})
        info = signal.overlap_summary(windows)
        self.assertEqual(info["pair_denominator"], 6)
        self.assertEqual(info["overlapping_pair_count"], 1)
        self.assertEqual(info["overlapping_pair_fraction"], 1/6)
        self.assertEqual(info["adjacent_pair_denominator"], 3)
        self.assertEqual(info["adjacent_overlapping_pair_count"], 1)

    def test_overlap_singleton_is_undefined_not_zero(self):
        windows = pd.DataFrame({"entry_date": ["2026-01-02"], "exit_date": ["2026-01-06"]})
        info = signal.overlap_summary(windows)
        self.assertEqual(info["pair_denominator"], 0)
        self.assertTrue(np.isnan(info["overlapping_pair_fraction"]))


class FrozenAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixture()
        cls.diagnostics = signal.build_diagnostics(cls.inputs)

    def test_no_fit_transformer_fit_or_estimator_prediction_calls(self):
        with patch.object(baseline, "fit_linear", side_effect=AssertionError("fit forbidden")), \
             patch.object(stage11, "fit_arm", side_effect=AssertionError("fit forbidden")), \
             patch.object(baseline, "predict_linear", side_effect=AssertionError("predict forbidden")), \
             patch.object(baseline.LinearRegression, "fit", side_effect=AssertionError("fit forbidden")), \
             patch.object(baseline.Ridge, "fit", side_effect=AssertionError("fit forbidden")), \
             patch.object(baseline.StandardScaler, "fit", side_effect=AssertionError("preprocessing fit forbidden")):
            diagnostics = signal.build_diagnostics(self.inputs)
            report = signal.validate_diagnostics(self.inputs, diagnostics, check_sma=True)
        self.assertTrue(report["valid"], report["errors"])

    def test_predictions_and_coefficients_exact_projected_byte_equality(self):
        report = signal.validate_diagnostics(self.inputs, self.diagnostics, check_sma=True)
        self.assertTrue(report["valid"], report["errors"])
        for field in ("stage11_prediction_consistency_violations", "stage11_coefficient_consistency_violations", "new_learned_prediction_rows", "modified_learned_prediction_rows"):
            self.assertEqual(report[field], 0)
        self.assertTrue(report["prediction_projected_csv_byte_equality"])
        self.assertTrue(report["coefficient_projected_csv_byte_equality"])

    def test_row_error_delta_and_baseline_prediction_scalar_oracle(self):
        data = self.inputs.data.set_index("date")
        authority = stage11.cv_authority(self.inputs.data, self.inputs.manifest, self.inputs.split_protocol, self.inputs.folds)
        rows = self.diagnostics.tables["row_baseline_comparison"]
        for row in rows.itertuples():
            y = float(data.loc[row.date, "forward_return_5d"])
            hist = sum(map(float, data.loc[authority[row.fold][0], "forward_return_5d"]))/len(authority[row.fold][0])
            self.assertEqual(row.error, row.prediction-y)
            self.assertEqual(row.AE_model, abs(row.prediction-y))
            self.assertEqual(row.AE_zero, abs(y))
            self.assertAlmostEqual(row.historical_mean_prediction, hist, places=12)
            self.assertAlmostEqual(row.delta_AE_hist, abs(row.prediction-y)-abs(y-hist), places=12)

    def test_saved_training_medians_and_scales_used_without_validation_fit(self):
        authority = stage11.cv_authority(self.inputs.data, self.inputs.manifest, self.inputs.split_protocol, self.inputs.folds)
        for fold, (train, valid) in authority.items():
            train_x, valid_x = signal.frozen_feature_matrices(self.inputs, fold, train, valid)
            usage = self.inputs.stage11_tables["feature_manifest"].loc[lambda f: f.arm.eq("treatment") & f.model.eq("ols") & f.fold.eq(fold) & f.used].set_index("feature_name")
            raw = stage11.representation(self.inputs.data.set_index("date").loc[valid], "treatment")
            for name in valid_x:
                expected = [(float(usage.loc[name, "training_median"]) if pd.isna(value) else float(value)) for value in raw[name]]
                expected = (np.array(expected)-float(usage.loc[name, "training_mean_after_imputation"]))/float(usage.loc[name, "scaler_scale"])
                np.testing.assert_allclose(valid_x[name], expected, rtol=0, atol=1e-12)

    def test_dropped_features_stay_missing_and_common_counts_correct(self):
        rows = self.diagnostics.tables["coefficient_stability"]
        dropped = rows.loc[~rows.used]
        self.assertEqual(len(dropped), 4)
        self.assertTrue(dropped.coefficient.isna().all())
        self.assertTrue(dropped.sign.isna().all())
        self.assertTrue(rows.loc[~rows.common_all_three_folds, "positive_fold_count"].isna().all())
        common = rows.loc[rows.common_all_three_folds].drop_duplicates(["model", "feature_name"])
        self.assertEqual(len(common), 14)
        self.assertTrue((common.positive_fold_count+common.negative_fold_count+common.zero_fold_count).eq(3).all())
        similarities = self.diagnostics.tables["coefficient_similarity"]
        self.assertEqual(similarities.n_common_features.tolist(), [7, 7, 9, 7, 7, 9])

    def test_coefficient_cosine_independent_intersection_oracle(self):
        source = self.inputs.stage11_tables["coefficients"]
        for row in self.diagnostics.tables["coefficient_similarity"].itertuples():
            names = json.loads(row.common_feature_names)
            a = source.loc[source.arm.eq("treatment") & source.model.eq(row.model) & source.fold.eq(row.fold_a)].set_index("feature_name")
            b = source.loc[source.arm.eq("treatment") & source.model.eq(row.model) & source.fold.eq(row.fold_b)].set_index("feature_name")
            self.assertEqual(set(names), set(a.index) & set(b.index))
            numerator = sum(float(a.loc[n, "coefficient"])*float(b.loc[n, "coefficient"]) for n in names)
            denominator = np.sqrt(sum(float(a.loc[n, "coefficient"])**2 for n in names)*sum(float(b.loc[n, "coefficient"])**2 for n in names))
            self.assertAlmostEqual(row.cosine_similarity, numerator/denominator, places=12)

    def test_target_overlap_counts_and_lag_autocorrelation_oracle(self):
        overlap = self.diagnostics.tables["target_overlap"].set_index("fold")
        for fold in signal.FOLDS:
            self.assertEqual(overlap.loc[fold, "pair_denominator"], 300)
            self.assertEqual(overlap.loc[fold, "overlapping_pair_count"], 90)
            self.assertEqual(overlap.loc[fold, "adjacent_overlapping_pair_count"], 24)
        self.assertEqual(overlap.loc["pooled", "overlapping_pair_count"], 290)
        self.assertEqual(overlap.loc["pooled", "pair_denominator"], 2775)
        keyed = self.inputs.data.set_index("date")
        for row in self.diagnostics.tables["target_dependence"].itertuples():
            dates = self.diagnostics.tables["target_windows"].loc[lambda f: f.fold.eq(row.fold), "date"]
            y = keyed.loc[dates, "forward_return_5d"].to_numpy()
            a, b = y[:-row.lag_observations], y[row.lag_observations:]
            self.assertAlmostEqual(row.target_autocorrelation, np.corrcoef(a, b)[0, 1], places=12)
            self.assertEqual(row.overlapping_lag_pair_count, 25-row.lag_observations if row.lag_observations < 5 else 0)

    def test_all_unused_and_directional_mutations_pass(self):
        original = copy.deepcopy(self.inputs)
        report = signal.isolation_checks(self.inputs, self.diagnostics)
        for name, value in report.items():
            if name.endswith("_violations"):
                self.assertEqual(value, 0, name)
        self.assertEqual(len(report["directional_mutation_checks"]), 6)
        self.assertTrue(report["calibration_non_feedback_checked"])
        pd.testing.assert_frame_equal(self.inputs.data, original.data, check_exact=True)
        for name in self.inputs.stage11_tables:
            pd.testing.assert_frame_equal(self.inputs.stage11_tables[name], original.stage11_tables[name], check_exact=True)

    def test_validation_y_changes_diagnostics_but_frozen_state_stays_exact(self):
        changed = self.inputs.data.copy(deep=True)
        dates = self.diagnostics.tables["target_windows"].loc[lambda f: f.fold.eq("cv_3"), "date"]
        changed.loc[changed.date.isin(dates), "forward_return_5d"] = np.linspace(1., 3., 25)**2
        result = signal.build_diagnostics(signal._changed_inputs(self.inputs, changed))
        keys = [c for c in signal.PREDICTION_KEYS if c != "actual"]
        pd.testing.assert_frame_equal(result.tables["row_baseline_comparison"][keys], self.diagnostics.tables["row_baseline_comparison"][keys], check_exact=True)
        pd.testing.assert_frame_equal(result.tables["coefficient_stability"], self.diagnostics.tables["coefficient_stability"], check_exact=True)
        self.assertFalse(signal._same(result.tables["prediction_calibration"], self.diagnostics.tables["prediction_calibration"], exact=True))
        self.assertFalse(signal._same(result.tables["feature_target_associations"], self.diagnostics.tables["feature_target_associations"], exact=True))

    def test_validation_x_changes_association_only_not_frozen_predictions(self):
        changed = self.inputs.data.copy(deep=True)
        dates = self.diagnostics.tables["target_windows"].loc[lambda f: f.fold.eq("cv_3"), "date"]
        changed.loc[changed.date.isin(dates), "close"] = 1000000 + np.arange(25)**2
        result = signal.build_diagnostics(signal._changed_inputs(self.inputs, changed))
        for name in ("row_baseline_comparison", "prediction_calibration", "coefficient_stability"):
            pd.testing.assert_frame_equal(result.tables[name], self.diagnostics.tables[name], check_exact=True)
        self.assertFalse(signal._same(result.tables["feature_target_associations"], self.diagnostics.tables["feature_target_associations"], exact=True))

    def test_validation_y_cannot_change_feature_values_or_preprocessing(self):
        changed = self.inputs.data.copy(deep=True)
        authority = stage11.cv_authority(self.inputs.data, self.inputs.manifest, self.inputs.split_protocol, self.inputs.folds)
        changed.loc[changed.date.isin(authority["cv_3"][1]), "forward_return_5d"] = np.arange(25) * 1e8
        mutated = signal._changed_inputs(self.inputs, changed)
        for fold, (train, valid) in authority.items():
            before = signal.frozen_feature_matrices(self.inputs, fold, train, valid)
            after = signal.frozen_feature_matrices(mutated, fold, train, valid)
            for a, b in zip(before, after):
                pd.testing.assert_frame_equal(a, b, check_exact=True)
        pd.testing.assert_frame_equal(self.inputs.stage11_tables["feature_manifest"], mutated.stage11_tables["feature_manifest"], check_exact=True)

    def test_undefined_validation_association_has_nan_reason_and_roundtrip(self):
        changed = self.inputs.data.copy(deep=True)
        dates = self.diagnostics.tables["target_windows"].loc[lambda f: f.fold.eq("cv_3"), "date"]
        changed.loc[changed.date.isin(dates), "simple_return"] = 0.
        diagnostics = signal.build_diagnostics(signal._changed_inputs(self.inputs, changed))
        associations = diagnostics.tables["feature_target_associations"]
        row = associations.loc[associations.fold.eq("cv_3") & associations.feature_name.eq("simple_return")].iloc[0]
        self.assertTrue(np.isnan(row.validation_pearson))
        self.assertTrue(np.isnan(row.validation_spearman))
        self.assertEqual(row.validation_undefined_reason, "constant_feature")
        self.assertTrue(pd.isna(row.pearson_sign_match))
        from io import StringIO
        reloaded = pd.read_csv(StringIO(associations.to_csv(index=False)), float_precision="round_trip")
        self.assertTrue(signal._same(reloaded, associations))
        state = diagnostics.tables["feature_target_stability"].set_index("fold")
        self.assertEqual(state.loc["cv_3", "pearson_defined_pair_count"], 8)

    def test_calibration_adversary_never_changes_original_error_or_mae(self):
        original_calibration = signal.calibration
        def mutated(y, p):
            return {**original_calibration(y, p), "calibration_slope": -1e12, "calibration_intercept": 1e12}
        with patch.object(signal, "calibration", side_effect=mutated):
            result = signal.build_diagnostics(self.inputs)
        for name in ("row_baseline_comparison", "baseline_advantage_summary", "prediction_dispersion"):
            pd.testing.assert_frame_equal(result.tables[name], self.diagnostics.tables[name], check_exact=True)
        self.assertFalse(result.tables["prediction_calibration"].equals(self.diagnostics.tables["prediction_calibration"]))
        columns = ["fold", "model", "MAE", "RMSE"]
        pd.testing.assert_frame_equal(result.tables["fold_diagnostics"][columns], self.diagnostics.tables["fold_diagnostics"][columns], check_exact=True)
        self.assertFalse(any("calibrated_prediction" == c or "calibrated_MAE" == c for table in result.tables.values() for c in table))

    def test_prediction_coefficient_and_diagnostic_tampering_fail(self):
        for table, field in (("row_baseline_comparison", "prediction"), ("coefficient_stability", "coefficient"),
                             ("prediction_calibration", "calibration_slope"), ("baseline_advantage_summary", "mean_delta_AE_zero"),
                             ("target_overlap", "overlapping_pair_count"), ("target_dependence", "target_autocorrelation"),
                             ("feature_target_associations", "train_pearson")):
            with self.subTest(table=table):
                changed = copy.deepcopy(self.diagnostics)
                changed.tables[table].loc[0, field] += 1 if pd.api.types.is_integer_dtype(changed.tables[table][field]) else .1
                report = signal.validate_diagnostics(self.inputs, changed)
                self.assertFalse(report["valid"])

    def test_baseline_and_frozen_scaler_tampering_fail(self):
        for table, field in (("historical", "historical_mean_prediction"), ("scaler", "scaler_scale")):
            changed = copy.deepcopy(self.inputs)
            if table == "historical":
                changed.stage10_tables["oof_predictions"].loc[0, field] += .1
            else:
                mask = changed.stage11_tables["feature_manifest"].arm.eq("treatment")
                changed.stage11_tables["feature_manifest"].loc[mask, field] *= 2.
            result = signal.build_diagnostics(changed)
            report = signal.validate_diagnostics(changed, result)
            self.assertFalse(report["valid"])

    def test_summary_safety_tampering_and_extra_rows_fail(self):
        changed = copy.deepcopy(self.diagnostics)
        changed.summary["calibrated_predictions_generated"] = True
        self.assertFalse(signal.validate_diagnostics(self.inputs, changed)["valid"])
        changed = copy.deepcopy(self.diagnostics)
        changed.tables["row_baseline_comparison"] = pd.concat([changed.tables["row_baseline_comparison"], changed.tables["row_baseline_comparison"].iloc[[0]]], ignore_index=True)
        self.assertFalse(signal.validate_diagnostics(self.inputs, changed)["valid"])

    def test_historical_control_not_used_for_mechanism_classification(self):
        tables = copy.deepcopy(self.diagnostics.tables)
        mask = tables["baseline_advantage_summary"].arm.eq("control")
        tables["baseline_advantage_summary"].loc[mask, "model_beats_zero_fraction"] = 1.
        self.assertEqual(signal.assess(tables), signal.assess(self.diagnostics.tables))

    def test_pit_formula_prefix_and_future_dependency_oracles(self):
        authority = stage11.cv_authority(self.inputs.data, self.inputs.manifest, self.inputs.split_protocol, self.inputs.folds)
        report = stage11.pit_checks(self.inputs.data, authority, check_sma=True)
        self.assertTrue(all(value == 0 for value in report.values()))


class CoefficientZeroOracleTests(unittest.TestCase):
    def synthetic(self):
        usage, coefficients = [], []
        for model in stage11.FAMILIES:
            for fold, value in zip(signal.FOLDS, [2., -4., 0.]):
                for name in stage11.TREATMENT:
                    used = name == "simple_return"
                    usage.append({"arm": "treatment", "model": model, "fold": fold, "feature_name": name,
                                  "used": used, "drop_reason": "used" if used else "zero_training_variance"})
                    if used:
                        coefficients.append({"arm": "treatment", "model": model, "fold": fold, "feature_name": name, "coefficient": value, "intercept": .1})
        return pd.DataFrame(usage), pd.DataFrame(coefficients)

    def test_zero_norm_cosine_nan_and_actual_zero_sign_separate(self):
        coeff, similarity = signal.coefficient_diagnostics(*self.synthetic())
        common = coeff.loc[coeff.common_all_three_folds]
        self.assertTrue(common.positive_fold_count.eq(1).all())
        self.assertTrue(common.negative_fold_count.eq(1).all())
        self.assertTrue(common.zero_fold_count.eq(1).all())
        self.assertTrue(common.sign_consistent_all_folds.eq(False).all())
        self.assertEqual(similarity.cosine_similarity.dropna().tolist(), [-1., -1.])
        self.assertEqual(int(similarity.cosine_similarity.isna().sum()), 4)
        self.assertTrue(similarity.n_common_features.eq(1).all())


class ConclusionFrameworkTests(unittest.TestCase):
    def tables(self, weak=False):
        rows = []
        for model in stage11.FAMILIES:
            for fold in signal.FOLDS:
                rows.append({"model": model, "fold": fold, "calibration_slope": -1. if weak else 2.,
                    "Pearson": -.5 if weak else .5, "Spearman": -.5 if weak else .5,
                    "model_beats_zero_fraction": .2 if weak else .8, "mean_bias": .1, "std_actual": .1, "dispersion_ratio": 2.})
        signs = pd.DataFrame([{"model": model, "feature_name": name, "common_all_three_folds": True,
            "sign_consistent_all_folds": not weak} for model in stage11.FAMILIES for name in ("a", "b")])
        sim = pd.DataFrame([{"model": model, "cosine_similarity": -1. if weak else 1.} for model in stage11.FAMILIES for pair in range(3)])
        assoc = pd.DataFrame([{"fold": fold, "pearson_sign_match_fraction": .2 if weak else .8,
                              "spearman_sign_match_fraction": .2 if weak else .8} for fold in (*signal.FOLDS, "pooled")])
        return {"fold_diagnostics": pd.DataFrame(rows), "coefficient_stability": signs,
                "coefficient_similarity": sim, "feature_target_stability": assoc}

    def test_signal_requires_joint_axes_and_both_families(self):
        self.assertEqual(signal.assess(self.tables())["category"], "SIGNAL_WITH_CALIBRATION_INSTABILITY")

    def test_weak_requires_directional_and_auxiliary_evidence(self):
        self.assertEqual(signal.assess(self.tables(weak=True))["category"], "WEAK_OR_TEMPORALLY_UNSTABLE_SIGNAL")

    def test_low_row_advantage_with_positive_slopes_is_mixed(self):
        tables = self.tables()
        tables["fold_diagnostics"]["model_beats_zero_fraction"] = .2
        self.assertEqual(signal.assess(tables)["category"], "MIXED_OR_INCONCLUSIVE")

    def test_borderline_majority_and_disagreeing_families_are_mixed(self):
        tables = self.tables()
        tables["fold_diagnostics"]["model_beats_zero_fraction"] = .52
        self.assertEqual(signal.assess(tables)["category"], "MIXED_OR_INCONCLUSIVE")
        tables = self.tables()
        tables["fold_diagnostics"].loc[lambda f: f.model.eq("ridge"), "calibration_slope"] = -1.
        self.assertEqual(signal.assess(tables)["category"], "MIXED_OR_INCONCLUSIVE")

    def test_undefined_diagnostics_are_inconclusive(self):
        tables = self.tables()
        tables["fold_diagnostics"].loc[0, "calibration_slope"] = np.nan
        self.assertEqual(signal.assess(tables)["category"], "MIXED_OR_INCONCLUSIVE")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        old = stage11_files(Path(cls.temporary.name))
        stage11.run_pipeline(**old)
        cls.args = {**old, "stage11_dir": old["output_dir"], "output_dir": Path(cls.temporary.name) / "stage12"}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_protocol_before_diagnostics_and_unchanged_contract_lock(self):
        with patch.object(signal, "build_diagnostics", side_effect=AssertionError("must not diagnose yet")):
            report = signal.run_pipeline(**self.args, register_only=True)
        self.assertTrue(report["protocol_registered_before_diagnostics"])
        path = self.args["output_dir"] / "NVDA_stage12_protocol.json"
        original = path.read_bytes()
        contract = json.loads(original)
        self.assertTrue(contract["diagnostic_only"])
        contract["tie_tolerance"] = .01
        path.write_text(json.dumps(contract))
        try:
            with self.assertRaisesRegex(ValueError, "locked"):
                signal.run_pipeline(**self.args, register_only=True)
        finally:
            path.write_bytes(original)

    def test_repeated_byte_equality_no_fit_no_feedback_upstream_immutability(self):
        with patch.object(baseline, "fit_linear", side_effect=AssertionError("fit forbidden")), \
             patch.object(baseline, "predict_linear", side_effect=AssertionError("prediction forbidden")):
            report = signal.run_pipeline(**self.args)
            first = {p.name: p.read_bytes() for p in self.args["output_dir"].iterdir()}
            second_report = signal.run_pipeline(**self.args)
            second = {p.name: p.read_bytes() for p in self.args["output_dir"].iterdir()}
        self.assertEqual(first, second)
        self.assertEqual(len(first), 17)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["source_sha256_before"], second_report["source_sha256_after"])
        self.assertFalse(list(self.args["output_dir"].rglob("*.tmp")))
        for field, value in signal.SAFETY.items():
            self.assertEqual(report[field], value)

    def test_frozen_prediction_and_coefficient_artifact_tampering_rejected(self):
        for name in ("oof_predictions", "coefficients", "feature_manifest"):
            path = self.args["stage11_dir"] / f"NVDA_stage11_{name}.csv"
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    signal.run_pipeline(**self.args, register_only=True)
            finally:
                path.write_bytes(original)

    def test_upstream_output_directory_and_symlinks_rejected(self):
        with self.assertRaisesRegex(ValueError, "upstream"):
            signal.run_pipeline(**{**self.args, "output_dir": self.args["stage11_dir"]})
        self.args["output_dir"].mkdir(exist_ok=True)
        path = self.args["output_dir"] / "unsafe.csv"
        path.symlink_to(self.args["labeled_path"])
        try:
            with self.assertRaisesRegex(ValueError, "symlink"):
                signal.run_pipeline(**self.args, register_only=True)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
