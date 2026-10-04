"""Stage 7 offline fixtures, independent checks and artifact reproducibility."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research.diagnostics import (
    DiagnosticThresholds, build_diagnostics, classify_feature, correlations,
    distribution, feature_inventory, near_constant, run_pipeline,
    temporal_diagnostics, validate_diagnostics,
)
from nasdaq_research.research_features import BALANCE_FIELDS


def fixture(size=80):
    """Independent Stage 6 shaped fixture; no production feature calculation."""
    dates = pd.bdate_range("2026-01-05", periods=size)
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": "NVDA",
                          "simple_return": np.sin(np.arange(size)) / 100,
                          "log_return": np.sin(np.arange(size)) / 100,
                          "rolling_volatility_20": np.arange(size, dtype=float) / 1000})
    frame.loc[0, ["simple_return", "log_return"]] = np.nan
    frame.loc[:19, "rolling_volatility_20"] = np.nan
    snapshots = []
    for prefix, basis, name in (("q", "quarterly", "q_gross_margin"),
                                ("fy", "annual", "fy_gross_margin"),
                                ("bs", "quarterly", "current_ratio")):
        mid = size // 2
        filed = [(dates[0] - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                 (dates[mid] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
        for key, a, b in (
            ("accession", f"{prefix}-a", f"{prefix}-b"),
            ("filing_date", *filed), ("effective_date", frame.date.iloc[0], frame.date.iloc[mid]),
            ("period_start", "2025-01-01", "2025-04-01"),
            ("period_end", "2025-03-31", "2025-06-30"), ("data_period_type", basis, basis),
        ):
            frame[f"{prefix}_{key}"] = [a] * mid + [b] * (size - mid)
        frame[name] = [.4] * mid + [.6] * (size - mid)
        for i, accession in enumerate((f"{prefix}-a", f"{prefix}-b")):
            snapshots.append({"ticker": "NVDA", "accession": accession, "filing_date": filed[i],
                              "effective_date": frame.date.iloc[0 if i == 0 else mid],
                              "period_start": "2025-01-01" if i == 0 else "2025-04-01",
                              "period_end": "2025-03-31" if i == 0 else "2025-06-30",
                              "data_period_type": basis, "q_gross_margin": .4 + i * .2 if basis == "quarterly" else np.nan,
                              "fy_gross_margin": .4 + i * .2 if basis == "annual" else np.nan,
                              "current_ratio": .4 + i * .2,
                              **{c: 1. for c in BALANCE_FIELDS}})
    frame["q_fcf_margin"] = np.nan
    snapshots = pd.DataFrame(snapshots).assign(q_fcf_margin=np.nan)
    return frame, snapshots


def values_frame(values, name="simple_return"):
    return pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=len(values)), name: values})


class MissingnessTests(unittest.TestCase):
    def test_full_coverage(self):
        row = feature_inventory(values_frame([1., 2., 3.])).iloc[-1]
        self.assertEqual(row.coverage_label, "full_coverage")
        self.assertEqual(row.coverage_ratio, 1)
        self.assertEqual(row.longest_consecutive_missing_run, 0)
        self.assertEqual(row.number_of_missing_blocks, 0)

    def test_partial_missingness_and_blocks(self):
        row = feature_inventory(values_frame([None, None, 1., None, 2., None, None, None])).iloc[-1]
        self.assertEqual(row.non_null_count, 2)
        self.assertEqual(row.null_count, 6)
        self.assertEqual(row.longest_consecutive_missing_run, 3)
        self.assertEqual(row.number_of_missing_blocks, 3)
        self.assertEqual(row.coverage_label, "sparse")
        self.assertEqual(row.first_valid_date, "2026-01-07")
        self.assertEqual(row.last_valid_date, "2026-01-09")

    def test_all_missing(self):
        row = feature_inventory(values_frame([np.nan] * 6)).iloc[-1]
        self.assertEqual(row.coverage_label, "all_missing")
        self.assertEqual(row.longest_consecutive_missing_run, 6)
        self.assertEqual(row.number_of_missing_blocks, 1)
        self.assertTrue(pd.isna(row.first_valid_date))
        self.assertEqual(row.unique_count, 0)

    def test_threshold_boundaries(self):
        for count, label in ((0, "all_missing"), (49, "sparse"), (50, "moderate_missingness"),
                             (94, "moderate_missingness"), (95, "high_coverage"), (100, "full_coverage")):
            with self.subTest(count=count):
                row = feature_inventory(values_frame([1.] * count + [np.nan] * (100 - count))).iloc[-1]
                self.assertEqual(row.coverage_label, label)

    def test_value_changes_do_not_bridge_missing_blocks(self):
        row = feature_inventory(values_frame([1., 2., None, 4., 4.])).iloc[-1]
        self.assertEqual(row.number_of_value_changes, 1)


class DistributionTests(unittest.TestCase):
    def test_normal_numeric_statistics(self):
        result = distribution(pd.Series([1., 2., 3., 4., 5.]))
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["mean"], 3)
        self.assertAlmostEqual(result["std"], np.sqrt(2.5))
        self.assertEqual(result["median"], 3)
        self.assertAlmostEqual(result["p01"], 1.04)
        self.assertAlmostEqual(result["p99"], 4.96)

    def test_constant(self):
        result = distribution(pd.Series([3.] * 20))
        self.assertEqual(result["std"], 0)
        self.assertEqual(result["outlier_status"], "zero_iqr_not_assessed")
        self.assertFalse(result["potential_outlier_issue"])

    def test_all_nan(self):
        result = distribution(pd.Series([np.nan] * 10))
        self.assertEqual(result["count"], 0)
        self.assertTrue(np.isnan(result["std"]))
        self.assertTrue(np.isnan(result["median"]))
        self.assertEqual(result["outlier_status"], "insufficient_observations")

    def test_extreme_value_not_modified(self):
        values = pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10000.])
        original = values.copy()
        result = distribution(values)
        self.assertTrue(result["potential_outlier_issue"])
        self.assertEqual(result["outlier_count"], 1)
        self.assertEqual(result["max"], 10000.)
        pd.testing.assert_series_equal(values, original)

    def test_infinity_is_counted_separately(self):
        values = pd.Series([1., 2., np.inf, -np.inf, np.nan])
        result = distribution(values)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["non_finite_count"], 2)
        self.assertEqual(result["max"], 2)
        self.assertTrue(np.isinf(values.iloc[2]))


class NearConstantTests(unittest.TestCase):
    def test_fully_constant(self):
        self.assertTrue(near_constant(pd.Series([1.] * 10))["near_constant"])

    def test_almost_constant(self):
        self.assertTrue(near_constant(pd.Series([1.] * 99 + [2.]))["near_constant"])

    def test_varying(self):
        self.assertFalse(near_constant(pd.Series(np.arange(100, dtype=float)))["near_constant"])

    def test_no_data_and_single_observation(self):
        self.assertFalse(near_constant(pd.Series([np.nan]))["near_constant"])
        self.assertFalse(near_constant(pd.Series([1.]))["near_constant"])

    def test_long_daily_step_is_not_near_constant(self):
        frame, snapshots = fixture(240)
        frame.loc[:237, "fy_gross_margin"] = .4
        # Move the second annual state to the last two days, leaving 238 repeated rows.
        for key in ("accession", "filing_date", "effective_date", "period_start", "period_end", "data_period_type"):
            frame.loc[:237, f"fy_{key}"] = frame.loc[0, f"fy_{key}"]
        frame.loc[238:, "fy_filing_date"] = (pd.Timestamp(frame.date.iloc[238]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frame.loc[238:, "fy_effective_date"] = frame.date.iloc[238]
        tables, summary = build_diagnostics(frame, snapshots)
        row = tables["feature_quality"].set_index("feature_name").loc["fy_gross_margin"]
        self.assertFalse(row.near_constant)
        self.assertEqual(row.diagnostic_observation_count, 2)
        self.assertEqual(summary["fundamental_snapshots"]["annual"]["distinct_window_states"], 2)


class CorrelationTests(unittest.TestCase):
    def test_positive_negative_and_high_flags(self):
        data = pd.DataFrame({"a": np.arange(30, dtype=float), "b": np.arange(30, dtype=float) * 2,
                             "c": -np.arange(30, dtype=float)})
        matrix, counts, pairs = correlations(data, list(data))
        self.assertAlmostEqual(matrix.loc["a", "b"], 1)
        self.assertAlmostEqual(matrix.loc["a", "c"], -1)
        self.assertEqual(counts.loc["a", "b"], 30)
        self.assertTrue(pairs.highly_correlated.all())

    def test_uncorrelated(self):
        data = pd.DataFrame({"a": [-1., -1., 1., 1.] * 8, "b": [-1., 1., -1., 1.] * 8})
        matrix, _, pairs = correlations(data, list(data))
        self.assertAlmostEqual(matrix.loc["a", "b"], 0)
        self.assertFalse(pairs.highly_correlated.iloc[0])

    def test_insufficient_pairwise_sample(self):
        data = pd.DataFrame({"a": [1., 2., 3., 4., 5.], "b": [2., 4., 6., 8., 10.]})
        matrix, counts, pairs = correlations(data, list(data))
        self.assertTrue(np.isnan(matrix.loc["a", "b"]))
        self.assertEqual(counts.loc["a", "b"], 5)
        self.assertEqual(pairs.status.iloc[0], "insufficient_pairwise_sample")
        self.assertFalse(pairs.highly_correlated.iloc[0])

    def test_missing_overlap(self):
        data = pd.DataFrame({"a": np.arange(50, dtype=float), "b": np.arange(50, dtype=float)})
        data.loc[:9, "a"] = np.nan
        data.loc[30:, "b"] = np.nan
        matrix, counts, pairs = correlations(data, list(data))
        self.assertEqual(counts.loc["a", "b"], 20)
        self.assertEqual(pairs.pairwise_count.iloc[0], 20)
        self.assertAlmostEqual(matrix.loc["a", "b"], 1)

    def test_zero_overlap_and_constant(self):
        data = pd.DataFrame({"a": [1.] * 25 + [np.nan] * 25, "b": [np.nan] * 25 + [2.] * 25,
                             "c": [3.] * 50})
        _, counts, pairs = correlations(data, list(data))
        self.assertEqual(counts.loc["a", "b"], 0)
        self.assertFalse(pairs.highly_correlated.any())
        self.assertIn("constant_or_undefined", pairs.status.tolist())

    def test_pairwise_distinct_states(self):
        frame, snapshots = fixture()
        tables, _ = build_diagnostics(frame, snapshots)
        pair = tables["correlation_pairs"].query("feature_a == 'q_gross_margin' and feature_b == 'fy_gross_margin'").iloc[0]
        self.assertEqual(pair.pairwise_count, 80)
        self.assertEqual(pair.distinct_states_a, 2)
        self.assertEqual(pair.distinct_states_b, 2)
        annual = tables["snapshot_correlations"].query("view == 'window_snapshot' and feature_group == 'annual'")
        self.assertTrue(annual.correlation.isna().all())


class TemporalTests(unittest.TestCase):
    def test_daily_varying_and_warmup(self):
        frame, _ = fixture()
        result = temporal_diagnostics(frame, ["simple_return", "rolling_volatility_20"]).set_index("feature_name")
        self.assertEqual(result.loc["simple_return", "temporal_behavior"], "daily_varying")
        self.assertEqual(result.loc["rolling_volatility_20", "leading_missing_count"], 20)
        self.assertEqual(result.loc["rolling_volatility_20", "unexpected_missing_count"], 0)
        self.assertFalse(result.potential_temporal_inconsistency.any())

    def test_valid_step_function(self):
        frame, _ = fixture()
        result = temporal_diagnostics(frame, ["q_gross_margin", "fy_gross_margin", "current_ratio"])
        self.assertTrue(result.temporal_behavior.eq("step_function").all())
        self.assertTrue(result.effective_date_aligned_changes.eq(1).all())
        self.assertFalse(result.potential_temporal_inconsistency.any())

    def test_change_outside_effective_date(self):
        frame, _ = fixture()
        frame.loc[10, "q_gross_margin"] += .1
        row = temporal_diagnostics(frame, ["q_gross_margin"]).iloc[0]
        self.assertTrue(row.potential_temporal_inconsistency)
        self.assertIn(frame.date.iloc[10], row.inconsistency_dates)
        self.assertIn(frame.date.iloc[11], row.inconsistency_dates)

    def test_missing_to_value_outside_effective_date(self):
        frame, _ = fixture()
        frame.loc[:5, "q_gross_margin"] = np.nan
        row = temporal_diagnostics(frame, ["q_gross_margin"]).iloc[0]
        self.assertTrue(row.potential_temporal_inconsistency)

    def test_market_unexpected_gap(self):
        frame, _ = fixture()
        frame.loc[30, "rolling_volatility_20"] = np.nan
        row = temporal_diagnostics(frame, ["rolling_volatility_20"]).iloc[0]
        self.assertEqual(row.unexpected_missing_count, 1)
        self.assertTrue(row.potential_temporal_inconsistency)

    def test_market_constant_run(self):
        frame, _ = fixture()
        frame.loc[20:, "rolling_volatility_20"] = .01
        row = temporal_diagnostics(frame, ["rolling_volatility_20"]).iloc[0]
        self.assertEqual(row.longest_constant_run, 60)
        self.assertTrue(row.potential_temporal_inconsistency)

    def test_future_filing_or_wrong_effective_metadata(self):
        for field, value in (("q_filing_date", "2027-01-01"), ("q_effective_date", "2027-01-01"),
                             ("q_data_period_type", "annual")):
            frame, _ = fixture()
            frame.loc[15, field] = value
            row = temporal_diagnostics(frame, ["q_gross_margin"]).iloc[0]
            self.assertTrue(row.potential_temporal_inconsistency, field)


class ClassificationTests(unittest.TestCase):
    def test_groups(self):
        expected = {"simple_return": "market", "sma_60": "market", "q_net_margin": "quarterly",
                    "fy_revenue_growth_yoy": "annual", "current_ratio": "balance_sheet",
                    "q_accession": "metadata", "q_fiscal_year": "metadata", "bs_days_since_filing": "metadata",
                    "q_yoy_reference_filing_date": "metadata", "q_effective_date_is_sample_truncated": "metadata",
                    "close": "market_source", "revenue": "fundamental_source",
                    "bs_total_assets": "fundamental_source", "new_unknown": "unclassified"}
        for name, group in expected.items():
            with self.subTest(name=name):
                self.assertEqual(classify_feature(name), group)

    def test_metadata_not_in_numeric_analysis(self):
        frame, snapshots = fixture()
        frame["q_fiscal_year"] = 2026
        frame["q_days_since_filing"] = np.arange(len(frame))
        tables, summary = build_diagnostics(frame, snapshots)
        self.assertNotIn("q_fiscal_year", tables["feature_correlations"])
        self.assertNotIn("q_days_since_filing", tables["feature_quality"].feature_name.tolist())
        self.assertEqual(len(tables["feature_inventory"]), len(frame.columns))
        self.assertGreater(summary["metadata_count"], 0)


class OutputTests(unittest.TestCase):
    def test_reports_and_unchanged_frames(self):
        frame, snapshots = fixture()
        original, snapshot_original = frame.copy(deep=True), snapshots.copy(deep=True)
        tables, summary = build_diagnostics(frame, snapshots)
        self.assertTrue({"feature_name", "feature_group", "coverage_ratio", "unique_count", "std", "quality_notes", "usable"}
                        <= set(tables["feature_quality"]))
        self.assertEqual(summary["all_missing_features"], ["q_fcf_margin"])
        self.assertTrue(validate_diagnostics(frame, original, tables, summary)["valid"])
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        pd.testing.assert_frame_equal(snapshots, snapshot_original, check_exact=True)

    def test_independent_validation_catches_wrong_counts(self):
        frame, snapshots = fixture()
        tables, summary = build_diagnostics(frame, snapshots)
        tables["correlation_pairwise_counts"].loc["simple_return", "log_return"] += 1
        self.assertFalse(validate_diagnostics(frame, frame.copy(), tables, summary)["valid"])

    def test_future_snapshots_excluded_from_history(self):
        frame, snapshots = fixture()
        future = snapshots.iloc[[0]].copy()
        future["accession"] = "future"
        future["filing_date"] = "2027-01-01"
        future["q_gross_margin"] = 99999.
        before_tables, before = build_diagnostics(frame, snapshots)
        after_tables, after = build_diagnostics(frame, pd.concat([snapshots, future], ignore_index=True))
        pd.testing.assert_frame_equal(before_tables["feature_distributions"], after_tables["feature_distributions"])
        self.assertEqual(after["fundamental_snapshots"]["excluded_at_or_after_window_end"], 1)
        self.assertEqual(before["fundamental_snapshots"]["historical_snapshot_count"],
                         after["fundamental_snapshots"]["historical_snapshot_count"])

    def test_temporal_anomaly_reported_without_repair(self):
        frame, snapshots = fixture()
        frame.loc[10, "q_gross_margin"] = 99.
        tables, summary = build_diagnostics(frame, snapshots)
        self.assertIn("q_gross_margin", summary["fundamental_temporal_inconsistencies"])
        self.assertEqual(frame.q_gross_margin.iloc[10], 99.)
        self.assertTrue(validate_diagnostics(frame, frame.copy(), tables, summary)["valid"])

    def test_pipeline_json_hashes_figures_and_reproducibility(self):
        frame, snapshots = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "NVDA_features.csv", root / "NVDA_fundamental_features.csv", root / "NVDA_feature_validation.json"]
            frame.to_csv(paths[0], index=False)
            snapshots.to_csv(paths[1], index=False)
            paths[2].write_text(json.dumps({"validation": {"valid": True}}))
            stage5 = root / "NVDA_research.csv"
            stage5.write_text("date,ticker\n2026-01-05,NVDA\n")
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths, stage5]}
            output = root / "diagnostics"
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network is forbidden")), \
                 patch("socket.create_connection", side_effect=AssertionError("Network is forbidden")):
                first = run_pipeline(*paths, output)
                first_bytes = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
                second = run_pipeline(*paths, output)
            self.assertEqual(first["validation_status"], "passed")
            self.assertEqual(first["source_sha256_before"], first["source_sha256_after"])
            self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])
            for relative, content in first_bytes.items():
                self.assertEqual(content, (output / relative).read_bytes(), str(relative))
            for p, digest in before.items():
                self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(), digest)
            def reject_nonstandard_constant(value):
                self.fail(f"Nonstandard JSON numeric constant: {value}")
            summary = json.loads((output / "NVDA_feature_diagnostics.json").read_text(),
                                 parse_constant=reject_nonstandard_constant)
            self.assertEqual(summary["row_count"], len(frame))
            self.assertEqual(summary["column_count"], len(frame.columns))
            self.assertTrue(summary["validation"]["valid"])
            self.assertGreaterEqual(len(summary["figures"]), 6)
            for name in summary["figures"]:
                self.assertEqual(Path(name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_output_cannot_contain_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "must not contain"):
                run_pipeline(root / "a.csv", root / "b.csv", root / "c.json", root)

    def test_invalid_input_dates_rejected(self):
        frame, snapshots = fixture()
        from nasdaq_research.diagnostics import _check_inputs
        for bad in (frame.iloc[::-1], frame.assign(ticker=None), pd.concat([frame, frame.iloc[[0]]])):
            with self.assertRaises(ValueError):
                _check_inputs(bad, snapshots)

    def test_threshold_validation(self):
        for changes in ({"sparse_coverage": .99}, {"correlation_min_pairwise": 2}, {"high_correlation": 1.1}):
            with self.assertRaises(ValueError):
                DiagnosticThresholds(**changes)


if __name__ == "__main__":
    unittest.main()
