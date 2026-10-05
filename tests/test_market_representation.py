"""Independent Stage 11 formula/fit oracles, frozen-control audits and mutations."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import modeling as baseline
from nasdaq_research import model_stability as stability
from nasdaq_research import market_representation as stage11
from test_modeling import fixture as original_fixture, checkpoint_files
from test_model_stability import add_state_metadata


def fixture():
    data, manifest, protocol, folds, inventory, quality = original_fixture()
    position = np.arange(len(data), dtype=float)
    data["close"] = 100 + .15 * position + 3 * np.sin(position / 7)
    for k in (5, 20, 60):
        data[f"sma_{k}"] = data.close.rolling(k, min_periods=k).mean()
    return add_state_metadata(data), manifest, protocol, folds, inventory, quality


def small_x():
    result = pd.DataFrame(0., index=["a", "b", "c", "d"], columns=stage11.TREATMENT)
    result["close_to_sma_5"] = [-.03, .01, .02, .08]
    return result


class RepresentationOracleTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()[0]

    def test_independent_scalar_formula_every_representative_date(self):
        rows = self.data.set_index("date")
        actual = stage11.representation(rows, "treatment")
        for i in (0, 4, 19, 59, 60, 95, 125, 160, 180):
            for k in (5, 20, 60):
                sma = rows.iloc[i][f"sma_{k}"]
                expected = np.nan if pd.isna(sma) else float(rows.iloc[i]["close"]) / float(sma) - 1
                np.testing.assert_allclose(actual.iloc[i][f"close_to_sma_{k}"], expected, rtol=0, atol=1e-12, equal_nan=True)

    def test_exact_known_values_and_no_inverse_or_log_formula(self):
        rows = self.data.iloc[[70]].copy()
        rows["close"] = 120.
        rows[["sma_5", "sma_20", "sma_60"]] = [100., 150., 80.]
        values = stage11.representation(rows, "treatment").iloc[0]
        np.testing.assert_allclose(values[["close_to_sma_5", "close_to_sma_20", "close_to_sma_60"]], [.2, -.2, .5], rtol=0, atol=1e-12)

    def test_control_preserves_original_nine_columns(self):
        actual = stage11.representation(self.data, "control")
        pd.testing.assert_frame_equal(actual, self.data[list(stage11.CONTROL)], check_dtype=False, check_exact=True)

    def test_treatment_whitelist_and_exactly_nine(self):
        actual = stage11.representation(self.data, "treatment")
        self.assertEqual(tuple(actual), stage11.TREATMENT)
        self.assertEqual(len(actual.columns), 9)
        self.assertFalse(set(actual) & {"close", "sma_5", "sma_20", "sma_60", "forward_return_5d", "revenue", "q_accession"})

    def test_six_unchanged_features_are_exact(self):
        actual = stage11.representation(self.data, "treatment")
        unchanged = [c for c in stage11.CONTROL if c not in stage11.REPLACEMENTS]
        pd.testing.assert_frame_equal(actual[unchanged], self.data[unchanged], check_exact=True)

    def test_preserves_warmup_nan_instead_of_filling_future(self):
        actual = stage11.representation(self.data, "treatment")
        for k in (5, 20, 60):
            self.assertTrue(actual[f"close_to_sma_{k}"].iloc[:k-1].isna().all())
            self.assertFalse(pd.isna(actual[f"close_to_sma_{k}"].iloc[k-1]))

    def test_source_and_upstream_feature_whitelist_unchanged(self):
        original, whitelist = self.data.copy(deep=True), dict(baseline.FEATURE_GROUPS)
        stage11.representation(self.data, "treatment")
        pd.testing.assert_frame_equal(original, self.data, check_exact=True)
        self.assertEqual(whitelist, baseline.FEATURE_GROUPS)
        self.assertFalse(set(stage11.REPLACEMENTS.values()) & set(baseline.FEATURE_GROUPS))

    def test_prefix_and_independent_trailing_mean_oracle(self):
        # Rebuild causal SMAs from raw prices in independent Python loops.
        for end in (60, 95, 125, 180):
            prefix = self.data.iloc[:end+1].copy()
            for k in (5, 20, 60):
                price = list(map(float, prefix.close))
                prefix[f"sma_{k}"] = [sum(price[max(0, i-k+1):i+1]) / k if i >= k-1 else np.nan for i in range(len(price))]
            actual = stage11.representation(prefix, "treatment").iloc[-1]
            full = stage11.representation(self.data, "treatment").iloc[end]
            np.testing.assert_allclose(actual, full, rtol=0, atol=1e-12, equal_nan=True)

    def test_future_price_mutation_does_not_rewrite_prefix(self):
        original = stage11.representation(self.data.iloc[:126], "treatment")
        changed = self.data.copy(deep=True)
        changed.loc[126:, ["close", "sma_5", "sma_20", "sma_60"]] = 1e12
        pd.testing.assert_frame_equal(stage11.representation(changed.iloc[:126], "treatment"), original, check_exact=True)

    def test_nonpositive_denominator_or_close_rejected(self):
        for name in ("close", "sma_5", "sma_20", "sma_60"):
            for bad in (0., -1., np.inf):
                with self.subTest(name=name, bad=bad):
                    data = self.data.iloc[[70]].copy()
                    data[name] = bad
                    with self.assertRaises(ValueError):
                        stage11.representation(data, "treatment")

    def test_unknown_arm_and_duplicate_columns_rejected(self):
        with self.assertRaises(ValueError):
            stage11.representation(self.data, "log_relative")
        with self.assertRaises(ValueError):
            stage11.representation(pd.concat([self.data, self.data[["close"]]], axis=1), "treatment")


class FitOracleTests(unittest.TestCase):
    def test_ols_one_retained_feature_closed_form(self):
        x = small_x()
        y = pd.Series([.05, -.01, .02, .08], index=x.index)
        fitted = stage11.fit_arm("ols", "treatment", x, y)
        raw = x.close_to_sma_5.to_numpy()
        z = (raw - raw.mean()) / np.sqrt(np.mean((raw - raw.mean())**2))
        coefficient = float(sum(z * (y.to_numpy() - y.mean())) / sum(z**2))
        self.assertEqual(fitted.preprocessing.features, ("close_to_sma_5",))
        self.assertAlmostEqual(fitted.estimator.coef_[0], coefficient, places=12)
        self.assertAlmostEqual(fitted.estimator.intercept_, y.mean(), places=12)
        np.testing.assert_allclose(baseline.predict_linear(fitted, x), y.mean() + z * coefficient, rtol=0, atol=1e-12)

    def test_ridge_one_retained_feature_fixed_alpha_closed_form(self):
        x = small_x()
        y = pd.Series([.05, -.01, .02, .08], index=x.index)
        fitted = stage11.fit_arm("ridge", "treatment", x, y)
        raw = x.close_to_sma_5.to_numpy()
        z = (raw - raw.mean()) / raw.std(ddof=0)
        expected = sum(z * (y.to_numpy() - y.mean())) / (sum(z**2) + 1.)
        self.assertEqual(fitted.estimator.alpha, 1.)
        self.assertEqual(fitted.estimator.solver, "svd")
        self.assertAlmostEqual(fitted.estimator.coef_[0], expected, places=12)

    def test_exact_half_coverage_keep_and_lower_drop(self):
        x = small_x()
        x["close_to_sma_20"] = [1., np.nan, np.nan, 3.]
        x["close_to_sma_60"] = [1., np.nan, np.nan, np.nan]
        fitted = stage11.fit_arm("ols", "treatment", x, pd.Series([0., .1, .2, .3], index=x.index))
        usage = fitted.preprocessing.usage.set_index("feature_name")
        self.assertTrue(usage.loc["close_to_sma_20", "used"])
        self.assertEqual(usage.loc["close_to_sma_20", "training_coverage"], .5)
        self.assertEqual(usage.loc["close_to_sma_60", "drop_reason"], "low_training_coverage")
        self.assertEqual(usage.loc["simple_return", "drop_reason"], "zero_training_variance")

    def test_independent_median_population_scale_and_validation_missing_fill(self):
        x = small_x()
        x["close_to_sma_5"] = [-.03, .01, np.nan, .08]
        fitted = stage11.fit_arm("ols", "treatment", x, pd.Series([0., .1, .2, .3], index=x.index))
        filled = np.array([-.03, .01, .01, .08])
        pp = fitted.preprocessing
        self.assertEqual(pp.medians["close_to_sma_5"], .01)
        np.testing.assert_allclose(pp.scaler.mean_, [sum(filled)/4], rtol=0, atol=1e-12)
        np.testing.assert_allclose(pp.scaler.scale_, [np.sqrt(sum((filled-filled.mean())**2)/4)], rtol=0, atol=1e-12)
        validation = x.iloc[[0, 1]].copy()
        validation["close_to_sma_5"] = [np.nan, 1e6]
        original = copy.deepcopy(pp.usage)
        z = baseline.transform_features(pp, validation)
        np.testing.assert_allclose(z[:, 0], (np.array([.01, 1e6])-filled.mean())/filled.std(ddof=0), rtol=0, atol=1e-8)
        pd.testing.assert_frame_equal(original, pp.usage, check_exact=True)

    def test_raw_target_is_not_standardized(self):
        x = small_x()
        y = pd.Series([10., 20., 30., 40.], index=x.index)
        fitted = stage11.fit_arm("ols", "treatment", x, y)
        self.assertAlmostEqual(fitted.estimator.intercept_, 25., places=12)

    def test_rejects_append_sma_raw_close_wrong_model_and_alpha_search(self):
        x = small_x()
        y = pd.Series([0., .1, .2, .3], index=x.index)
        for name in ("close", "sma_5", "forward_return_1d", "q_fiscal_year"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                stage11.fit_arm("ols", "treatment", x.assign(**{name: 1.}), y)
        for family in ("ridge_cv", "lasso", "ols_all"):
            with self.subTest(family=family), self.assertRaises(ValueError):
                stage11.fit_arm(family, "treatment", x, y)


class ExperimentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = fixture()
        cls.data, cls.manifest, cls.protocol, cls.folds = cls.checkpoint[:4]
        cls.reference, _, _ = baseline.build_cv(*cls.checkpoint)
        cls.experiment = stage11.build_experiment(*cls.checkpoint[:4])

    def audit(self, experiment=None, reference=None):
        return stage11.validate_experiment(*self.checkpoint[:4], experiment or self.experiment,
            reference or self.reference, check_sma=True)

    def test_control_all_channels_reproduce(self):
        report = self.audit()
        self.assertTrue(report["valid"], report["errors"])
        for name in stage11.VIOLATIONS:
            self.assertEqual(report[name], 0, name)
        for channels in report["control_projected_csv_byte_equality"].values():
            self.assertTrue(all(channels.values()))

    def test_manifest_only_dates_counts_four_arms_and_no_all(self):
        table = self.experiment.tables["oof_predictions"]
        self.assertEqual(len(table), 300)
        self.assertEqual(table.date.nunique(), 75)
        self.assertEqual(set(zip(table.arm, table.model)), {("control", "ols"), ("control", "ridge"), ("treatment", "ols"), ("treatment", "ridge")})
        self.assertEqual(self.experiment.summary["fold_train_counts"], {"cv_1": 96, "cv_2": 121, "cv_3": 146})
        self.assertEqual({r["scaler_sample_count"] for r in self.experiment.fit_records}, {96, 121, 146})

    def test_all_isolation_mutations_and_same_fold_training_invariance(self):
        original = self.data.copy(deep=True)
        report = stage11.isolation_checks(*self.checkpoint[:4], self.experiment)
        self.assertTrue(all(value == 0 for value in report.values()), report)
        self.assertIn("final_test_metadata_mutation_dependency_violations", report)
        self.assertIn("validation_target_fit_dependency_violations", report)
        pd.testing.assert_frame_equal(self.data, original, check_exact=True)

    def test_final_test_and_gap_invalid_payloads_never_transformed(self):
        changed = self.data.copy(deep=True)
        final = self.manifest.loc[self.manifest.fold.eq("final") & self.manifest.target_name.eq("forward_return_5d")]
        excluded = set(final.loc[final.pre_test_gap | final.partition.isin(["final_test", "unlabeled_tail"]), "date"])
        changed.loc[changed.date.isin(excluded), [*stage11.CONTROL, "close", "forward_return_5d"]] = np.nan
        changed.loc[changed.date.isin(excluded), "ticker"] = "CORRUPT_UNUSED_TICKER"
        result = stage11.build_experiment(changed, self.manifest, self.protocol, self.folds)
        for name in stage11.TABLE_NAMES:
            pd.testing.assert_frame_equal(result.tables[name], self.experiment.tables[name], check_exact=True)

    def test_secondary_targets_can_be_removed_from_fit_input(self):
        names = [c for c in self.data if c.endswith("_1d") or c.endswith("_20d")]
        result = stage11.build_experiment(self.data.drop(columns=names), self.manifest, self.protocol, self.folds)
        for name in stage11.TABLE_NAMES:
            pd.testing.assert_frame_equal(result.tables[name], self.experiment.tables[name], check_exact=True)

    def test_control_prediction_tamper_fails(self):
        reference = copy.deepcopy(self.reference)
        reference["oof_predictions"].loc[0, "ols_market_prediction"] += .01
        report = self.audit(reference=reference)
        self.assertFalse(report["valid"])
        self.assertEqual(report["control_prediction_reproduction_violations"], 1)

    def test_control_fold_metric_tamper_fails(self):
        reference = copy.deepcopy(self.reference)
        reference["cv_metrics"].loc[lambda f: f.model_name.eq("ridge_market"), "MAE"] += .01
        report = self.audit(reference=reference)
        self.assertFalse(report["valid"])
        self.assertEqual(report["control_metric_reproduction_violations"], 1)

    def test_control_feature_filter_or_scaler_tamper_fails(self):
        for field, value in (("scaler_scale", 100.), ("used", False), ("drop_reason", "low_training_coverage")):
            reference = copy.deepcopy(self.reference)
            reference["feature_usage"].loc[lambda f: f.model_name.eq("ols_market"), field] = value
            report = stage11.control_reproduction(self.experiment, reference)
            self.assertEqual(report["control_feature_usage_reproduction_violations"], 1)

    def test_control_coefficient_and_membership_tamper_fails(self):
        reference = copy.deepcopy(self.reference)
        reference["coefficients"].loc[lambda f: f.model_name.eq("ols_market"), "coefficient"] += .01
        reference["oof_predictions"].loc[0, "fold"] = "cv_2"
        report = stage11.control_reproduction(self.experiment, reference)
        self.assertEqual(report["control_coefficient_reproduction_violations"], 1)
        self.assertEqual(report["control_fold_membership_reproduction_violations"], 2)

    def test_zscore_range_contribution_and_summary_tamper_detected(self):
        for table, field in (("zscores", "validation_zscore"), ("feature_shift", "validation_outside_training_range_count"),
                             ("contributions", "feature_contribution"), ("prediction_summary", "mean_abs_prediction"),
                             ("coefficient_summary", "coefficient_L2"), ("ablation_summary", "mean_fold_MAE")):
            with self.subTest(table=table):
                changed = copy.deepcopy(self.experiment)
                changed.tables[table].loc[0, field] += 1 if pd.api.types.is_integer_dtype(changed.tables[table][field]) else .01
                report = self.audit(experiment=changed)
                self.assertFalse(report["valid"])

    def test_contributions_reconstruct_all_300_rows_independently(self):
        c = self.experiment.tables["contributions"]
        sums = c.groupby(["date", "fold", "arm", "model"]).agg(total=("feature_contribution", "sum"), intercept=("intercept", "first"))
        oof = self.experiment.tables["oof_predictions"].set_index(["date", "fold", "arm", "model"])
        np.testing.assert_allclose((sums.total+sums.intercept).loc[oof.index], oof.prediction, rtol=0, atol=1e-12)

    def test_mean_fold_mae_and_fixed_benchmark_definition(self):
        keyed = self.data.set_index("date")
        authority = baseline.cv_authority(*self.checkpoint[:4])
        for fold, (train, valid) in authority.items():
            anchors = self.experiment.tables["benchmark_metrics"].loc[lambda f: f.fold.eq(fold)].set_index("model")
            y, mean = keyed.loc[valid, "forward_return_5d"].to_numpy(), keyed.loc[train, "forward_return_5d"].mean()
            self.assertAlmostEqual(anchors.loc["zero_return", "MAE"], sum(np.abs(y))/25, places=12)
            self.assertAlmostEqual(anchors.loc["historical_mean", "MAE"], sum(np.abs(y-mean))/25, places=12)
        for row in self.experiment.tables["ablation_summary"].itertuples():
            folds = self.experiment.tables["fold_metrics"].loc[lambda f: f.arm.eq(row.arm) & f.model.eq(row.model)]
            self.assertAlmostEqual(row.mean_fold_MAE, sum(folds.MAE)/3, places=12)

    def test_raw_support_equal_bounds_missing_and_training_z_oracle(self):
        training, validation = pd.Series([1., 2., 3., np.nan]), pd.Series([0., 1., 3., 4., np.nan])
        info = stability.feature_range(training, validation)
        self.assertEqual(info["validation_outside_training_range_count"], 2)
        self.assertEqual(info["validation_outside_training_range_fraction"], .5)
        info = stability.zscore_summary(np.array([-6., -3., 0., 3., 5.]))
        self.assertEqual(info["max_abs_validation_z"], 6.)
        self.assertEqual(info["count_abs_z_gt_3"], 2)
        self.assertEqual(info["count_abs_z_gt_5"], 1)

    def test_prediction_thresholds_strict_and_std_sample(self):
        values = np.array([-.31, -.30, -.10, 0., .10, .30, .31])
        info = stage11.prediction_summary(values)
        self.assertEqual(info["count_abs_prediction_gt_0_10"], 4)
        self.assertEqual(info["count_abs_prediction_gt_0_30"], 2)
        self.assertAlmostEqual(info["prediction_std"], np.sqrt(sum((values-values.mean())**2)/6), places=12)
        self.assertAlmostEqual(info["mean_abs_prediction"], sum(np.abs(values))/7, places=12)

    def test_authority_and_original_feature_corruption_rejected(self):
        changed = self.manifest.copy(deep=True)
        mask = changed.fold.eq("cv_1") & changed.target_name.eq("forward_return_5d") & changed.role.eq("validation")
        changed.loc[mask, "pre_test_gap"] = True
        with self.assertRaises(ValueError):
            stage11.build_experiment(self.data, changed, self.protocol, self.folds)
        data = self.data.copy(deep=True)
        data.loc[0, "ticker"] = "OTHER"
        with self.assertRaises(ValueError):
            stage11.build_experiment(data, self.manifest, self.protocol, self.folds)


class MechanismRuleTests(unittest.TestCase):
    def endpoints(self, ratio=.5):
        rows = []
        for fold in ("cv_1", "cv_2", "cv_3", "pooled"):
            for arm in ("control", "treatment"):
                for family in ("ols", "ridge"):
                    r = 1. if arm == "control" else ratio
                    rows.append({"fold": fold, "arm": arm, "model": family,
                        "sma_related_outside_fraction": r, "sma_related_max_abs_validation_z": 20*r,
                        "mean_abs_prediction": .3*r, "all_features_max_abs_validation_z": 20*r,
                        "count_abs_prediction_gt_0_10": 0, "count_abs_prediction_gt_0_30": 0})
        return pd.DataFrame(rows)

    def test_supported_requires_all_families_and_pooled_consistency(self):
        assessment = stage11.assess_mechanism(self.endpoints())
        self.assertEqual(assessment["category"], "SUPPORTED")
        self.assertEqual(assessment["cv3_material_endpoint_improvement_count"], 6)
        self.assertFalse(assessment["MAE_used_to_determine_category"])

    def test_partial_when_ridge_not_improved(self):
        endpoints = self.endpoints()
        endpoints.loc[endpoints.arm.eq("treatment") & endpoints.model.eq("ridge"), "mean_abs_prediction"] = .3
        self.assertEqual(stage11.assess_mechanism(endpoints)["category"], "PARTIALLY_SUPPORTED_OR_INCONCLUSIVE")

    def test_partial_when_new_other_fold_instability(self):
        endpoints = self.endpoints()
        endpoints.loc[endpoints.arm.eq("treatment") & endpoints.fold.eq("cv_1"), "count_abs_prediction_gt_0_10"] = 1
        self.assertEqual(stage11.assess_mechanism(endpoints)["category"], "PARTIALLY_SUPPORTED_OR_INCONCLUSIVE")

    def test_not_supported_if_no_material_improvement(self):
        self.assertEqual(stage11.assess_mechanism(self.endpoints(.9))["category"], "NOT_SUPPORTED")

    def test_material_boundary_inclusive_and_mae_irrelevant(self):
        endpoints = self.endpoints(.75)
        endpoints["MAE"] = np.arange(len(endpoints)) * 1e6
        self.assertEqual(stage11.assess_mechanism(endpoints)["category"], "SUPPORTED")


def pipeline_files(root):
    args = checkpoint_files(root)
    data = fixture()[0]
    data.to_csv(args["labeled_path"], index=False)
    digest = baseline._sha256(args["labeled_path"])
    protocol_path = args["split_dir"] / "NVDA_evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["source_labeled_sha256"] = digest
    protocol_path.write_text(json.dumps(protocol))
    validation_path = args["split_dir"] / "NVDA_split_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["source_labeled_sha256"] = digest
    validation["output_artifact_sha256"][str(protocol_path)] = baseline._sha256(protocol_path)
    validation_path.write_text(json.dumps(validation))
    baseline.run_pipeline(**args)
    stability_dir = root / "stability"
    stability.run_pipeline(**{**args, "baseline_dir": args["output_dir"], "output_dir": stability_dir, "generate_figures": False})
    return {**args, "baseline_dir": args["output_dir"], "stability_dir": stability_dir, "output_dir": root / "stage11"}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.args = pipeline_files(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_protocol_is_saved_before_any_fit_and_locks_formulas(self):
        with patch.object(stage11, "build_experiment", side_effect=AssertionError("must not fit")):
            result = stage11.run_pipeline(**self.args, register_only=True)
        self.assertTrue(result["protocol_registered_before_fit"])
        path = self.args["output_dir"] / "NVDA_stage11_protocol.json"
        protocol = json.loads(path.read_text())
        self.assertEqual(protocol["formulas"], {"close_to_sma_5": "Close / sma_5 - 1", "close_to_sma_20": "Close / sma_20 - 1", "close_to_sma_60": "Close / sma_60 - 1"})
        self.assertEqual(protocol["candidate_count_per_arm"], 9)
        original = path.read_bytes()
        protocol["ridge_alpha"] = 2.
        path.write_text(json.dumps(protocol))
        try:
            with self.assertRaisesRegex(ValueError, "locked"):
                stage11.run_pipeline(**self.args, register_only=True)
        finally:
            path.write_bytes(original)

    def test_repeated_pipeline_byte_equality_and_upstream_immutability(self):
        report = stage11.run_pipeline(**self.args)
        self.assertTrue(report["valid"], report["errors"])
        paths = list(self.args["output_dir"].iterdir())
        first = {p.name: p.read_bytes() for p in paths}
        second = stage11.run_pipeline(**self.args)
        self.assertEqual(first, {p.name: p.read_bytes() for p in paths})
        self.assertEqual(report["source_sha256_before"], second["source_sha256_after"])
        self.assertEqual(report["upstream_mutation_violations"], 0)
        self.assertEqual(report["prediction_reconstruction_rows_checked"], 300)
        self.assertEqual(len(paths), 16)
        self.assertFalse(list(self.args["output_dir"].rglob("*.tmp")))
        for field, expected in stage11.SAFETY.items():
            self.assertEqual(report[field], expected)

    def test_tampered_frozen_stage10_and_stage101_artifacts_rejected(self):
        for path in (self.args["baseline_dir"] / "NVDA_baseline_oof_predictions.csv",
                     self.args["stability_dir"] / "NVDA_stability_zscores.csv"):
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    stage11.run_pipeline(**self.args, register_only=True)
            finally:
                path.write_bytes(original)

    def test_upstream_output_directory_and_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, "upstream"):
            stage11.run_pipeline(**{**self.args, "output_dir": self.args["baseline_dir"]})
        symlink = self.args["output_dir"] / "unsafe.csv"
        symlink.symlink_to(self.args["labeled_path"])
        try:
            with self.assertRaisesRegex(ValueError, "symlink"):
                stage11.run_pipeline(**self.args, register_only=True)
        finally:
            symlink.unlink()


if __name__ == "__main__":
    unittest.main()
