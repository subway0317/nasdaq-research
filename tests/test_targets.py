"""Stage 8 offline formulas, independent oracle, timing and artifact integrity."""

import csv
from datetime import date
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research.targets import (
    HORIZONS, PRIMARY_TARGET, RETURN_COLUMNS, TARGET_COLUMNS, TARGET_PAYLOAD_COLUMNS,
    build_labeled_dataset, build_targets, invalid_price_count, run_pipeline,
    target_summary, validate_targets,
)


def fixture(size=30):
    """Observed sessions skip weekends and a synthetic missing January session."""
    days = [d for d in pd.bdate_range("2026-01-02", periods=size + 5)
            if d.strftime("%Y-%m-%d") != "2026-01-19"][:size]
    positions = np.arange(size, dtype=float)
    opens = 100 + positions * 1.5
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in days], "ticker": "NVDA",
        "open": opens, "high": opens + 4, "low": opens - 4,
        "close": opens + np.cos(positions) * 2, "volume": 1000 + positions,
        "simple_return": [np.nan, *[.01] * (size - 1)] if size else [],
        "q_fcf_margin": np.nan, "q_accession": "0000000000-26-000001",
    })


def independent_oracle(features):
    """Plain records and scalar positions; no builder, shifts or shared horizon constants."""
    source = features.to_dict("records")
    expected = []
    for i, row in enumerate(source):
        entry = source[i + 1] if i + 1 < len(source) else None
        answer = {"date": date.fromisoformat(row["date"]), "ticker": row["ticker"],
                  "target_entry_date": date.fromisoformat(entry["date"]) if entry else None,
                  "target_entry_open": entry["open"] if entry else np.nan}
        for h in (1, 5, 20):
            exit_row = source[i + h] if i + h < len(source) else None
            answer[f"target_exit_date_{h}d"] = date.fromisoformat(exit_row["date"]) if exit_row else None
            answer[f"target_exit_close_{h}d"] = exit_row["close"] if exit_row else np.nan
            answer[f"forward_return_{h}d"] = exit_row["close"] / entry["open"] - 1 if exit_row else np.nan
        expected.append(answer)
    return expected


def write_inputs(root, features=None):
    """Minimal consistent upstream checkpoint, entirely inside a temporary directory."""
    features = fixture() if features is None else features
    path = root / "NVDA_features.csv"
    features.to_csv(path, index=False)
    stage6 = root / "NVDA_feature_validation.json"
    stage6.write_text(json.dumps({"ticker": "NVDA", "final_feature_rows": len(features),
                                 "final_matrix_columns": len(features.columns), "validation": {"valid": True}}))
    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    stage7 = diagnostics / "NVDA_feature_diagnostics.json"
    stage7.write_text(json.dumps({"ticker": "NVDA", "row_count": len(features), "column_count": len(features.columns),
                                 "date_range": {"start": features.date.iloc[0], "end": features.date.iloc[-1]},
                                 "source_sha256_after": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()},
                                 "validation": {"valid": True}}))
    (diagnostics / "preserved.csv").write_text("feature_name,coverage_ratio\nq_fcf_margin,0\n")
    (root / "NVDA_research.csv").write_text("date,ticker\n2026-01-02,NVDA\n")
    return {"features_path": path, "output_dir": root / "targets", "labeled_path": root / "NVDA_labeled.csv",
            "feature_validation_path": stage6, "diagnostics_path": stage7}


class FormulaTests(unittest.TestCase):
    def test_1d_formula_uses_next_open(self):
        features = fixture()
        targets = build_targets(features)
        expected = features.close.iloc[1] / features.open.iloc[1] - 1
        self.assertAlmostEqual(targets.forward_return_1d.iloc[0], expected, places=14)
        self.assertNotAlmostEqual(expected, features.close.iloc[1] / features.close.iloc[0] - 1, places=5)

    def test_5d_formula(self):
        features = fixture()
        targets = build_targets(features)
        self.assertAlmostEqual(targets.forward_return_5d.iloc[0], features.close.iloc[5] / features.open.iloc[1] - 1, places=14)

    def test_20d_formula(self):
        features = fixture()
        targets = build_targets(features)
        self.assertAlmostEqual(targets.forward_return_20d.iloc[0], features.close.iloc[20] / features.open.iloc[1] - 1, places=14)

    def test_independent_oracle_all_dates_prices_and_returns(self):
        features = fixture(28)
        expected = independent_oracle(features)
        actual = build_targets(features)
        for i, row in enumerate(expected):
            for name, value in row.items():
                with self.subTest(row=i, field=name):
                    observed = actual.iloc[i][name]
                    if value is None or pd.isna(value):
                        self.assertTrue(pd.isna(observed))
                    elif isinstance(value, date):
                        self.assertEqual(date.fromisoformat(observed), value)
                    elif name.startswith("forward_return_"):
                        self.assertAlmostEqual(observed, value, places=14)
                    else:
                        self.assertEqual(observed, value)
        self.assertTrue(validate_targets(actual, features)["valid"])

    def test_zero_return_is_not_missing(self):
        features = fixture()
        features["close"] = features.open
        targets = build_targets(features)
        self.assertEqual(targets.forward_return_1d.iloc[0], 0.)
        self.assertTrue(validate_targets(targets, features)["valid"])

    def test_primary_target_and_no_extra_labels(self):
        targets = build_targets(fixture())
        self.assertEqual(PRIMARY_TARGET, "forward_return_5d")
        self.assertEqual(tuple(targets.columns), TARGET_COLUMNS)
        self.assertEqual(len(targets.columns), 13)
        self.assertEqual(set(c for c in targets if c.startswith("forward_return_")),
                         {"forward_return_1d", "forward_return_5d", "forward_return_20d"})


class TimingTests(unittest.TestCase):
    def test_exact_session_positions_and_boundaries(self):
        features = fixture()
        targets = build_targets(features)
        for i in range(len(features)):
            for h in (1, 5, 20):
                if i + h < len(features):
                    self.assertEqual(targets[f"target_exit_date_{h}d"].iloc[i], features.date.iloc[i + h])
            if i + 1 < len(features):
                self.assertGreater(targets.target_entry_date.iloc[i], features.date.iloc[i])
                self.assertEqual(targets.target_entry_date.iloc[i], features.date.iloc[i + 1])
                self.assertEqual(targets.target_exit_date_1d.iloc[i], targets.target_entry_date.iloc[i])
        report = validate_targets(targets, features)
        self.assertEqual(report["entry_after_feature_date_violations"], 0)
        self.assertEqual(report["horizon_order_violations"], 0)

    def test_friday_to_monday_entry(self):
        targets = build_targets(fixture())
        self.assertEqual(targets.date.iloc[0], "2026-01-02")
        self.assertEqual(targets.target_entry_date.iloc[0], "2026-01-05")

    def test_missing_calendar_session_uses_observed_rows(self):
        features = fixture()
        position = features.index[features.date.eq("2026-01-16")][0]
        targets = build_targets(features)
        self.assertEqual(targets.target_entry_date.iloc[position], "2026-01-20")
        self.assertEqual(targets.target_exit_date_5d.iloc[position], features.date.iloc[position + 5])
        self.assertNotEqual(targets.target_exit_date_5d.iloc[position], "2026-01-21")

    def test_tail_nan_counts_are_computed_for_each_input_length(self):
        for size in (1, 2, 4, 5, 19, 20, 21, 30):
            features = fixture(size)
            targets = build_targets(features)
            report = validate_targets(targets, features)
            self.assertTrue(report["valid"])
            self.assertEqual(len(targets), size)
            for h in (1, 5, 20):
                expected = sum(i + h >= size for i in range(size))
                with self.subTest(size=size, horizon=h):
                    self.assertEqual(targets[f"forward_return_{h}d"].isna().sum(), expected)
                    self.assertEqual(report[f"missing_target_count_{h}d"], expected)
                    self.assertEqual(targets[f"target_exit_date_{h}d"].isna().sum(), expected)

    def test_partial_tail_keeps_available_entry_provenance(self):
        features = fixture()
        targets = build_targets(features)
        self.assertTrue(pd.isna(targets.forward_return_20d.iloc[-2]))
        self.assertEqual(targets.target_entry_date.iloc[-2], features.date.iloc[-1])
        self.assertEqual(targets.target_entry_open.iloc[-2], features.open.iloc[-1])
        self.assertTrue(targets.iloc[-1].loc[list(TARGET_PAYLOAD_COLUMNS)].isna().all())

    def test_unsorted_input_rejected_without_mutation(self):
        features = fixture().iloc[::-1].copy()
        original = features.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "ascending"):
            build_targets(features)
        pd.testing.assert_frame_equal(features, original, check_exact=True)

    def test_duplicate_date_rejected(self):
        features = fixture()
        features.loc[1, "date"] = features.date.iloc[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_targets(features)

    def test_duplicate_date_mixed_iso_representations(self):
        features = fixture()
        features.loc[1, "date"] = "20260102"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_targets(features)

    def test_mixed_tickers_rejected(self):
        features = fixture()
        features.loc[len(features) - 1, "ticker"] = "MSFT"
        with self.assertRaisesRegex(ValueError, "ticker == NVDA"):
            build_targets(features)

    def test_other_single_ticker_rejected(self):
        with self.assertRaisesRegex(ValueError, "ticker == NVDA"):
            build_targets(fixture().assign(ticker="MSFT"))

    def test_malformed_dates_rejected(self):
        for value in (None, "01/02/2026", "2026-02-30", "2026-01-02T16:00:00", 20260102):
            features = fixture()
            features["date"] = features.date.astype(object)
            features.loc[0, "date"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_targets(features)


class PriceContractTests(unittest.TestCase):
    def test_missing_prices_rejected_without_imputation(self):
        for field in ("open", "close"):
            features = fixture()
            features.loc[3, field] = np.nan
            before = features.copy(deep=True)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Invalid open/close"):
                build_targets(features)
            self.assertEqual(invalid_price_count(features), 1)
            pd.testing.assert_frame_equal(features, before, check_exact=True)

    def test_non_numeric_prices_rejected(self):
        features = fixture()
        features["open"] = features.open.astype(object)
        features.loc[1, "open"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Invalid open/close"):
            build_targets(features)

    def test_zero_negative_and_infinite_prices_rejected(self):
        for field in ("open", "close"):
            for value in (0., -1., np.inf, -np.inf):
                features = fixture()
                features.loc[1, field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "Invalid open/close"):
                    build_targets(features)

    def test_boolean_prices_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid open/close"):
            build_targets(fixture().assign(open=True))

    def test_complex_prices_rejected(self):
        features = fixture()
        features["open"] = features.open.astype(complex)
        features.loc[1, "open"] += 2j
        with self.assertRaisesRegex(ValueError, "Invalid open/close"):
            build_targets(features)

    def test_overflow_raises_instead_of_creating_infinite_labels(self):
        features = fixture()
        features.loc[1, "open"] = 1e-300
        features.loc[1, "close"] = 1e300
        with self.assertRaisesRegex(ValueError, "Non-finite return"):
            build_targets(features)


class ValidationTests(unittest.TestCase):
    def test_mutated_entry_price_detected(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[0, "target_entry_open"] += .01
        report = validate_targets(targets, features)
        self.assertFalse(report["valid"])
        self.assertEqual(report["price_provenance_violations"], 1)
        self.assertGreater(report["return_formula_violations"], 0)

    def test_mutated_exit_price_detected(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[0, "target_exit_close_5d"] += .01
        report = validate_targets(targets, features)
        self.assertFalse(report["valid"])
        self.assertEqual(report["return_formula_violations_5d"], 1)

    def test_price_and_formula_mutated_together_still_detected(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[0, "target_entry_open"] += 10
        for h in HORIZONS:
            targets.loc[0, f"forward_return_{h}d"] = targets.loc[0, f"target_exit_close_{h}d"] / targets.loc[0, "target_entry_open"] - 1
        report = validate_targets(targets, features)
        self.assertFalse(report["valid"])
        self.assertGreater(report["price_provenance_violations"], 0)

    def test_one_ulp_price_mutation_is_not_hidden_by_return_tolerance(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[0, "target_entry_open"] = np.nextafter(targets.target_entry_open.iloc[0], np.inf)
        self.assertEqual(validate_targets(targets, features)["price_provenance_violations"], 1)

    def test_return_tolerance_is_explicit_and_strict(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[0, "forward_return_5d"] += 1e-10
        report = validate_targets(targets, features)
        self.assertEqual(report["formula_tolerance"], {"rtol": 0., "atol": 1e-12})
        self.assertEqual(report["return_formula_violations_5d"], 1)

    def test_wrong_entry_date_and_horizon_order_detected(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[0, "target_entry_date"] = targets.date.iloc[0]
        targets.loc[0, "target_exit_date_5d"] = targets.target_exit_date_1d.iloc[0]
        report = validate_targets(targets, features)
        self.assertEqual(report["entry_after_feature_date_violations"], 1)
        self.assertEqual(report["horizon_order_violations"], 1)
        self.assertFalse(report["valid"])

    def test_reordered_target_rows_rejected(self):
        features = fixture()
        targets = build_targets(features).iloc[::-1]
        report = validate_targets(targets, features)
        self.assertFalse(report["valid"])
        self.assertFalse(report["checks"]["feature_date_identity"])
        with self.assertRaisesRegex(ValueError, "Invalid target alignment"):
            build_labeled_dataset(features, targets)

    def test_fabricated_tail_return_rejected(self):
        features = fixture()
        targets = build_targets(features)
        targets.loc[len(targets) - 1, "forward_return_20d"] = 0.
        report = validate_targets(targets, features)
        self.assertFalse(report["valid"])
        self.assertEqual(report["return_formula_violations_20d"], 1)

    def test_nullable_missing_label_or_price_provenance_reported_as_invalid(self):
        for column in ("forward_return_5d", "target_entry_open", "target_exit_close_5d"):
            features = fixture()
            targets = build_targets(features)
            targets[column] = targets[column].astype("Float64")
            targets.loc[0, column] = pd.NA
            report = validate_targets(targets, features)
            self.assertFalse(report["valid"], column)
            self.assertGreater(report["return_formula_violations"], 0)

    def test_row_count_and_extra_schema_rejected(self):
        features = fixture()
        targets = build_targets(features)
        self.assertFalse(validate_targets(targets.iloc[:-1], features)["valid"])
        self.assertFalse(validate_targets(targets.assign(direction_5d=1), features)["valid"])

    def test_labeled_feature_mutation_detected(self):
        features = fixture()
        targets = build_targets(features)
        labeled = build_labeled_dataset(features, targets)
        labeled.loc[0, "q_accession"] = "wrong-source"
        self.assertFalse(validate_targets(targets, features, labeled)["valid"])


class IsolationTests(unittest.TestCase):
    def test_every_feature_cell_index_and_nan_preserved(self):
        features = fixture()
        features.index = [i * 3 + 5 for i in range(len(features))]
        features["research_note"] = "unchanged"
        original = features.copy(deep=True)
        targets = build_targets(features)
        labeled = build_labeled_dataset(features, targets)
        pd.testing.assert_frame_equal(features, original, check_exact=True)
        pd.testing.assert_frame_equal(labeled[features.columns], original, check_exact=True)
        self.assertEqual(len(labeled), len(features))
        self.assertEqual(labeled.columns.tolist(), [*features.columns, *TARGET_PAYLOAD_COLUMNS])
        self.assertTrue(validate_targets(targets, features, labeled)["valid"])

    def test_far_future_close_changes_only_targets_using_that_exit(self):
        features = fixture()
        before = build_targets(features)
        position = 23
        changed = features.copy()
        changed.loc[position, "close"] += 30.
        after = build_targets(changed)
        for h in (1, 5, 20):
            a, b = before[f"forward_return_{h}d"], after[f"forward_return_{h}d"]
            changed_positions = np.flatnonzero(~(a.eq(b) | (a.isna() & b.isna())))
            np.testing.assert_array_equal(changed_positions, [position - h])
        pd.testing.assert_series_equal(before.loc[0, list(RETURN_COLUMNS)], after.loc[0, list(RETURN_COLUMNS)])

    def test_future_open_changes_only_next_session_entry_targets(self):
        features = fixture()
        before = build_targets(features)
        changed = features.copy()
        changed.loc[8, "open"] += 20.
        after = build_targets(changed)
        for name in RETURN_COLUMNS:
            a, b = before[name], after[name]
            np.testing.assert_array_equal(np.flatnonzero(~(a.eq(b) | (a.isna() & b.isna()))), [7])

    def test_existing_target_namespace_cannot_enter_x(self):
        with self.assertRaisesRegex(ValueError, "Source X must not contain"):
            build_targets(fixture().assign(forward_return_5d=0.))


class PipelineTests(unittest.TestCase):
    def test_summary_only_describes_continuous_targets(self):
        targets = build_targets(fixture())
        summary = target_summary(targets)
        self.assertEqual(summary.target_name.tolist(), list(RETURN_COLUMNS))
        self.assertEqual(summary["count"].tolist(), [29, 25, 10])
        self.assertEqual(summary.missing_count.tolist(), [1, 5, 20])
        self.assertEqual(summary.primary_target.tolist(), [False, True, False])
        self.assertTrue({"count", "mean", "std", "min", "median", "max", "p01", "p99"} <= set(summary))

    def test_offline_pipeline_hashes_and_byte_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = write_inputs(root)
            baseline = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
                 patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
                first = run_pipeline(**args)
                artifacts = [Path(first["output_paths"][key]) for key in ("targets", "labeled", "summary", "validation")]
                artifacts.extend(Path(p) for p in first["output_paths"]["figures"])
                original_bytes = {p: p.read_bytes() for p in artifacts}
                second = run_pipeline(**args)
            self.assertEqual(first["output_artifact_sha256"], second["output_artifact_sha256"])
            self.assertEqual(first["validation_status"], "passed")
            self.assertTrue(first["source_artifacts_unchanged"])
            self.assertEqual(first["row_count"], 30)
            self.assertEqual(first["primary_target"], "forward_return_5d")
            self.assertEqual(first["return_formula_violations"], 0)
            for p, content in original_bytes.items():
                self.assertEqual(p.read_bytes(), content, str(p))
            for p, digest in baseline.items():
                self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(), digest)
            report = json.loads(artifacts[3].read_text(), parse_constant=lambda value: self.fail(f"Invalid JSON: {value}"))
            self.assertTrue(report["valid"])
            self.assertEqual(len(report["output_paths"]["figures"]), 4)
            for figure in report["output_paths"]["figures"]:
                self.assertEqual(Path(figure).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_original_decimal_tokens_and_future_price_tokens_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = write_inputs(root)
            source = args["features_path"]
            text = source.read_text().replace("101.5", "184.01800231933595")
            source.write_text(text)
            checkpoint = json.loads(args["diagnostics_path"].read_text())
            checkpoint["source_sha256_after"][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
            args["diagnostics_path"].write_text(json.dumps(checkpoint))
            report = run_pipeline(**args)
            original = list(csv.reader(StringIO(source.read_text())))
            labeled = list(csv.reader(StringIO(Path(report["output_paths"]["labeled"]).read_text())))
            targets = list(csv.reader(StringIO(Path(report["output_paths"]["targets"]).read_text())))
            for a, b in zip(original[1:], labeled[1:]):
                self.assertEqual(a, b[:len(original[0])])
            price_col = targets[0].index("target_entry_open")
            self.assertEqual(targets[1][price_col], "184.01800231933595")
            self.assertEqual(report["price_provenance_violations"], 0)

    def test_stale_checkpoint_hash_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_inputs(Path(tmp))
            args["features_path"].write_text(args["features_path"].read_text().replace("101.5", "102.5"))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                run_pipeline(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_failed_upstream_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_inputs(Path(tmp))
            args["feature_validation_path"].write_text(json.dumps({"validation": {"valid": False}}))
            with self.assertRaisesRegex(ValueError, "reports must pass"):
                run_pipeline(**args)

    def test_output_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_inputs(Path(tmp))
            args["labeled_path"] = args["features_path"]
            with self.assertRaisesRegex(ValueError, "source artifacts"):
                run_pipeline(**args)

    def test_output_symlink_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_inputs(Path(tmp))
            before = args["features_path"].read_bytes()
            args["output_dir"].mkdir()
            (args["output_dir"] / "NVDA_targets.csv").symlink_to(args["features_path"])
            with self.assertRaisesRegex(ValueError, "symlinks"):
                run_pipeline(**args)
            self.assertEqual(args["features_path"].read_bytes(), before)

    def test_cli_runs_with_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_inputs(Path(tmp))
            command = [sys.executable, "-m", "nasdaq_research.targets"]
            for key, value in args.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("Primary target: forward_return_5d", result.stdout)
            self.assertIn("Validation: passed", result.stdout)
            self.assertTrue(args["labeled_path"].exists())


if __name__ == "__main__":
    unittest.main()
