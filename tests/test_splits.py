"""Stage 9 offline temporal protocol, independent oracle and adversarial audits."""

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

from nasdaq_research.splits import (
    SplitConfig, build_splits, run_pipeline, validate_splits, validation_history_eligibility,
)
from nasdaq_research.targets import build_targets, build_labeled_dataset, RETURN_COLUMNS
from test_targets import fixture as market_fixture

SMALL = SplitConfig(final_test_size=10, validation_size=10, n_folds=3, minimum_initial_training_size=20)


def fixture(size=90):
    """Synthetic session metadata; scalar labels are arbitrary, never used for boundaries."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-02", periods=size + 3)
             if d.strftime("%Y-%m-%d") != "2026-01-19"][:size]
    data = pd.DataFrame({"date": dates, "ticker": "NVDA",
                         "target_entry_date": dates[1:] + [None], "feature_note": "preserved"})
    for h in (1, 5, 20):
        data[f"forward_return_{h}d"] = [float(i + h) / 1000 if i + h < size else np.nan for i in range(size)]
        data[f"target_exit_date_{h}d"] = [dates[i + h] if i + h < size else None for i in range(size)]
    return data


def selected(tables, target="forward_return_5d", fold="cv_1"):
    m = tables["split_manifest"]
    return m.loc[m.target_name.eq(target) & m.fold.eq(fold)].reset_index(drop=True)


def independent_oracle(data):
    """Independent 90-session example: primary 0:85, development 0:75, test 75:85.

    Explicit evaluation slices are hand-specified, not derived via production
    constants/layout/eligibility helpers. Provenance decides purge with scalars.
    """
    source = data.to_dict("records")
    expected = []
    blocks = (("cv_1", 25, 35), ("cv_2", 35, 45), ("cv_3", 45, 55), ("final", 75, 85))
    for fold, start, end in blocks:
        for h in (1, 5, 20):
            target = f"forward_return_{h}d"
            for i, row in enumerate(source):
                missing = pd.isna(row[target])
                if i >= 85:
                    role = "unlabeled_tail"
                elif fold != "final" and 55 <= i < 75:
                    role = "pre_test_gap"
                elif missing:
                    role = "target_missing"
                elif i < start:
                    role = "train" if row[f"target_exit_date_{h}d"] < source[start]["date"] else "purged"
                elif i < end:
                    role = "final_test" if fold == "final" else "validation"
                else:
                    role = "not_candidate"
                expected.append((row["date"], target, fold, role, role in ("train", "validation", "final_test")))
    return expected


def write_checkpoint(root):
    """Construct temporary Stage 8 fixtures only; no real upstream output is rebuilt."""
    features = market_fixture(251)
    targets = build_targets(features)
    labeled = build_labeled_dataset(features, targets)
    paths = {"labeled_path": root / "NVDA_labeled.csv", "targets_path": root / "NVDA_targets.csv",
             "features_path": root / "NVDA_features.csv", "target_validation_path": root / "NVDA_target_validation.json",
             "output_dir": root / "splits"}
    features.to_csv(paths["features_path"], index=False)
    targets.to_csv(paths["targets_path"], index=False)
    labeled.to_csv(paths["labeled_path"], index=False)
    report = {"valid": True, "primary_target": "forward_return_5d",
              "source_feature_sha256": hashlib.sha256(paths["features_path"].read_bytes()).hexdigest(),
              "output_artifact_sha256": {str(paths[k].resolve()): hashlib.sha256(paths[k].read_bytes()).hexdigest()
                                          for k in ("labeled_path", "targets_path")}}
    paths["target_validation_path"].write_text(json.dumps(report))
    (root / "preserved_stage7.csv").write_text("feature_name,coverage_ratio\nexample,1\n")
    return paths


class LayoutTests(unittest.TestCase):
    def test_default_config_and_current_size(self):
        config = SplitConfig()
        self.assertEqual((config.final_test_size, config.validation_size, config.n_folds, config.embargo_sessions), (50, 25, 3, 0))
        data = fixture(251)
        tables, protocol = build_splits(data)
        primary = selected(tables, fold="final")
        self.assertEqual(primary.role.eq("final_test").sum(), 50)
        self.assertEqual(primary.partition.eq("development").sum(), 196)
        self.assertEqual(primary.role.eq("unlabeled_tail").sum(), 5)
        self.assertEqual(primary.loc[primary.role.eq("final_test"), "date"].tolist(), data.date.iloc[196:246].tolist())
        self.assertTrue(validate_splits(data, tables, protocol)["valid"])

    def test_independent_oracle_complete_manifest(self):
        data = fixture()
        tables, protocol = build_splits(data, SMALL)
        m = tables["split_manifest"]
        actual = list(m[["date", "target_name", "fold", "role", "is_usable"]].itertuples(index=False, name=None))
        self.assertEqual(actual, independent_oracle(data))
        self.assertTrue(validate_splits(data, tables, protocol, SMALL)["valid"])

    def test_unsorted_input_fails_without_mutation(self):
        data = fixture().iloc[::-1].copy()
        original = data.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "ascending"):
            build_splits(data, SMALL)
        pd.testing.assert_frame_equal(data, original, check_exact=True)

    def test_duplicate_and_missing_date_fail(self):
        for value in (None, "2026-01-02", "20260102"):
            data = fixture()
            data.loc[1, "date"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_splits(data, SMALL)

    def test_mixed_ticker_fails(self):
        data = fixture()
        data.loc[4, "ticker"] = "MSFT"
        with self.assertRaisesRegex(ValueError, "ticker == NVDA"):
            build_splits(data, SMALL)

    def test_insufficient_data_does_not_shrink_configuration(self):
        with self.assertRaisesRegex(ValueError, "protocol requires 185"):
            build_splits(fixture(100))

    def test_interior_primary_missing_fails(self):
        data = fixture()
        data.loc[3, "forward_return_5d"] = np.nan
        with self.assertRaisesRegex(ValueError, "contiguous unlabeled tail"):
            build_splits(data, SMALL)

    def test_tail_never_enters_evaluation_even_when_1d_label_present(self):
        data = fixture()
        tables, _ = build_splits(data, SMALL)
        m = tables["split_manifest"]
        tail = m.loc[m.partition.eq("unlabeled_tail")]
        self.assertTrue(tail.role.eq("unlabeled_tail").all())
        self.assertFalse(tail.is_usable.any())
        self.assertTrue((~tail.loc[tail.target_name.eq("forward_return_1d"), "target_missing"]).any())
        self.assertEqual(len(tail), 5 * 3 * 4)

    def test_shorter_source_preserves_tail_and_missing_secondary_labels(self):
        data = fixture()
        config = SplitConfig(final_test_size=10, validation_size=5, n_folds=3, minimum_initial_training_size=20)
        data = data.iloc[:80].copy()
        for h in (1, 5, 20):
            for i in range(len(data)):
                if i + h >= len(data):
                    data.loc[i, f"forward_return_{h}d"] = np.nan
                    data.loc[i, f"target_exit_date_{h}d"] = None
        data.loc[len(data) - 1, "target_entry_date"] = None
        tables, _ = build_splits(data, config)
        self.assertEqual(tables["split_summary"].unlabeled_tail_rows.iloc[0], 5)


class ExpandingTests(unittest.TestCase):
    def test_folds_expand_and_validation_blocks_are_disjoint(self):
        data = fixture()
        tables, _ = build_splits(data, SMALL)
        cv = tables["cv_folds"].query("target_name == 'forward_return_5d'")
        self.assertEqual(cv.train_count_nominal.tolist(), [25, 35, 45])
        self.assertEqual(cv.validation_count.tolist(), [10, 10, 10])
        dates = []
        for i in range(1, 4):
            block = selected(tables, fold=f"cv_{i}")
            dates.extend(block.loc[block.nominal_role.eq("validation"), "date"])
        self.assertEqual(len(dates), len(set(dates)))
        self.assertEqual(dates, data.date.iloc[25:55].tolist())

    def test_earlier_validation_becomes_later_training_candidate(self):
        tables, _ = build_splits(fixture(), SMALL)
        first = selected(tables, fold="cv_1")
        second = selected(tables, fold="cv_2")
        prior_validation = first.loc[first.role.eq("validation"), "date"]
        self.assertTrue(second.loc[second.date.isin(prior_validation), "nominal_role"].eq("train").all())

    def test_secondary_boundaries_identical_and_final_20d_missing_retained(self):
        data = fixture(251)
        tables, _ = build_splits(data)
        for fold in ("cv_1", "cv_2", "cv_3", "final"):
            primary = selected(tables, fold=fold)
            expected = primary.loc[primary.nominal_role.isin(["validation", "final_test"]), "date"].tolist()
            for target in ("forward_return_1d", "forward_return_20d"):
                secondary = selected(tables, target, fold)
                self.assertEqual(secondary.loc[secondary.nominal_role.isin(["validation", "final_test"]), "date"].tolist(), expected)
        pool = tables["final_training_pool"].set_index("target_name")
        self.assertEqual(pool.loc["forward_return_20d", "final_test_count"], 50)
        self.assertEqual(pool.loc["forward_return_20d", "final_test_count_effective"], 35)
        self.assertEqual(pool.loc["forward_return_20d", "evaluation_missing_target_count"], 15)


class PurgeTests(unittest.TestCase):
    def test_before_equal_after_exit_boundaries(self):
        data = fixture()
        tables, _ = build_splits(data, SMALL)
        m = selected(tables, "forward_return_5d")
        self.assertEqual(m.role.iloc[19], "train")
        self.assertEqual(m.role.iloc[20], "purged")
        self.assertEqual(m.target_exit_date.iloc[20], m.evaluation_start_date.iloc[20])
        self.assertEqual(m.role.iloc[21], "purged")

    def test_independent_horizon_purge_counts(self):
        tables, _ = build_splits(fixture(), SMALL)
        for h, count in ((1, 1), (5, 5), (20, 20)):
            self.assertEqual(selected(tables, f"forward_return_{h}d").role.eq("purged").sum(), count)
        self.assertEqual(selected(tables, "forward_return_1d").role.iloc[10], "train")
        self.assertEqual(selected(tables, "forward_return_20d").role.iloc[10], "purged")

    def test_variable_exit_provenance_prevents_fixed_row_purge(self):
        data = fixture()
        data.loc[5, "target_exit_date_5d"] = data.date.iloc[45]
        tables, _ = build_splits(data, SMALL)
        self.assertEqual(selected(tables).role.iloc[5], "purged")
        self.assertEqual(selected(tables).role.eq("purged").sum(), 6)

    def test_final_training_pool_is_purged_separately(self):
        data = fixture(251)
        tables, _ = build_splits(data)
        pool = tables["final_training_pool"]
        nominal = int(data.forward_return_5d.notna().sum()) - SplitConfig().final_test_size
        effective = [sum(pd.notna(row[f'forward_return_{h}d']) and row[f'target_exit_date_{h}d'] < data.date.iloc[nominal]
                         for row in data.iloc[:nominal].to_dict('records')) for h in (1, 5, 20)]
        self.assertEqual(pool.final_train_count_nominal.tolist(), [nominal] * 3)
        self.assertEqual(pool.final_train_count_effective.tolist(), effective)
        self.assertEqual(pool.final_purged_count.tolist(), [nominal - n for n in effective])
        final = selected(tables, fold="final")
        self.assertEqual(final.final_train_candidate.sum(), nominal)
        self.assertEqual(final.final_train_usable.sum(), effective[1])
        self.assertEqual(final.final_train_purged.sum(), nominal - effective[1])

    def test_missing_secondary_cannot_train_or_evaluate(self):
        data = fixture()
        data.loc[[10, 50], "forward_return_20d"] = np.nan
        tables, _ = build_splits(data, SMALL)
        first = selected(tables, "forward_return_20d")
        for i in (10, 50):
            self.assertEqual(first.role.iloc[i], "target_missing")
            self.assertFalse(first.is_usable.iloc[i])
        summary = tables["cv_folds"].query("target_name == 'forward_return_20d' and fold == 'cv_1'").iloc[0]
        self.assertEqual(summary.train_missing_target_count, 1)
        self.assertEqual(summary.validation_count_effective, SMALL.validation_size)
        self.assertNotIn(data.date.iloc[50], first.loc[first.role.eq("validation"), "date"].tolist())

    def test_missing_exit_metadata_cannot_be_assumed_safe(self):
        data = fixture()
        data.loc[5, "target_exit_date_5d"] = None
        with self.assertRaisesRegex(ValueError, "Missing required target metadata"):
            build_splits(data, SMALL)


class AuditTests(unittest.TestCase):
    def test_role_conflict_detected_with_fold_scope(self):
        data = fixture()
        tables, protocol = build_splits(data, SMALL)
        duplicate = tables["split_manifest"].iloc[[0]].copy()
        duplicate["role"] = "validation"
        tables["split_manifest"] = pd.concat([tables["split_manifest"], duplicate], ignore_index=True)
        report = validate_splits(data, tables, protocol, SMALL)
        self.assertFalse(report["valid"])
        self.assertEqual(report["role_conflict_violations"], 1)
        self.assertEqual(report["train_validation_overlap_violations"], 1)

    def test_exact_boundary_wrongly_marked_train_detected(self):
        data = fixture()
        tables, protocol = build_splits(data, SMALL)
        m = tables["split_manifest"]
        mask = m.fold.eq("cv_1") & m.target_name.eq("forward_return_5d") & m.date.eq(data.date.iloc[20])
        m.loc[mask, "role"] = "train"
        m.loc[mask, "is_usable"] = True
        report = validate_splits(data, tables, protocol, SMALL)
        self.assertFalse(report["valid"])
        self.assertEqual(report["exact_boundary_violations"], 1)
        self.assertEqual(report["target_overlap_leakage_violations"], 1)

    def test_final_test_marked_train_detected(self):
        data = fixture()
        tables, protocol = build_splits(data, SMALL)
        m = tables["split_manifest"]
        mask = m.fold.eq("final") & m.target_name.eq("forward_return_1d") & m.date.eq(data.date.iloc[75])
        m.loc[mask, "role"] = "train"
        report = validate_splits(data, tables, protocol, SMALL)
        self.assertFalse(report["valid"])
        self.assertGreater(report["final_train_test_overlap_violations"], 0)

    def test_metadata_and_summary_tampering_detected(self):
        data = fixture()
        for artifact in ("split_manifest", "cv_folds", "split_summary"):
            tables, protocol = build_splits(data, SMALL)
            if artifact == "split_manifest":
                tables[artifact].loc[0, "target_exit_date"] = data.date.iloc[2]
            elif artifact == "cv_folds":
                tables[artifact].loc[0, "train_count_effective"] += 1
            else:
                tables[artifact].loc[0, "effective_train_count"] += 1
            self.assertFalse(validate_splits(data, tables, protocol, SMALL)["valid"], artifact)

    def test_fold_order_shuffle_detected(self):
        data = fixture()
        tables, protocol = build_splits(data, SMALL)
        tables["split_manifest"] = tables["split_manifest"].iloc[::-1].reset_index(drop=True)
        self.assertFalse(validate_splits(data, tables, protocol, SMALL)["valid"])

    def test_target_magnitude_cannot_change_split(self):
        data = fixture()
        original, protocol = build_splits(data, SMALL)
        changed = data.copy()
        for target in RETURN_COLUMNS:
            changed.loc[changed[target].notna(), target] = -999999.
        altered, altered_protocol = build_splits(changed, SMALL)
        for name in original:
            pd.testing.assert_frame_equal(original[name], altered[name], check_exact=True)
        self.assertEqual(protocol, altered_protocol)

    def test_source_rows_features_and_index_unchanged(self):
        data = fixture()
        data.index = [i * 3 + 2 for i in range(len(data))]
        original = data.copy(deep=True)
        build_splits(data, SMALL)
        pd.testing.assert_frame_equal(data, original, check_exact=True)


class ProtocolTests(unittest.TestCase):
    def test_embargo_zero_and_nonzero_explicitly_unsupported(self):
        self.assertEqual(SplitConfig().embargo_sessions, 0)
        for value in (1, -1, True):
            with self.assertRaisesRegex(ValueError, "embargo_sessions=0"):
                SplitConfig(embargo_sessions=value)

    def test_invalid_config_fails(self):
        for kwargs in ({"final_test_size": 0}, {"validation_size": -1}, {"n_folds": True}, {"minimum_initial_training_size": 0}):
            with self.assertRaises(ValueError):
                SplitConfig(**kwargs)

    def test_metrics_baselines_training_only_and_test_lock_contract(self):
        data = fixture()
        tables, protocol = build_splits(data, SMALL)
        self.assertEqual(protocol["primary_metric"], "MAE")
        self.assertEqual(protocol["secondary_metrics"], ["RMSE", "R²", "Pearson correlation", "Spearman correlation", "Directional accuracy"])
        self.assertTrue(protocol["final_test_locked"])
        self.assertFalse(protocol["random_split_allowed"])
        self.assertFalse(protocol["model_training_performed"])
        self.assertEqual(protocol["baselines"][1]["fit_scope"], "training_only")
        self.assertFalse(protocol["baselines"][1]["executed"])
        self.assertEqual(protocol["preprocessing"]["fit_scope"], "usable_training_only")
        for field, replacement in (("baselines", []), ("preprocessing", {"fit_scope": "full_sample"}),
                                   ("feature_selection", {"final_test_access_allowed": True}),
                                   ("stage10_readiness", "NOT READY FOR STAGE 10")):
            altered = dict(protocol)
            altered[field] = replacement
            with self.subTest(field=field):
                self.assertFalse(validate_splits(data, tables, altered, SMALL)["valid"])

    def test_cross_evaluation_calendar_overlap_is_prevented_with_full_blocks(self):
        tables, protocol = build_splits(fixture(251))
        last = tables["cv_folds"].query("fold == 'cv_3'")
        self.assertEqual(last.validation_count.tolist(), [25, 25, 25])
        self.assertEqual(last.validation_labels_reaching_final_test.tolist(), [0, 0, 0])
        self.assertTrue(protocol["strict_pre_test_model_selection_boundary_satisfied"])
        self.assertEqual(protocol["stage10_readiness"], "READY FOR STAGE 10")


class BoundaryHardeningTests(unittest.TestCase):
    def test_independent_boundary_oracle_with_variable_exits(self):
        data = fixture(120)
        # The named 20d column has deliberately irregular, hand-set provenance.
        # This helper-free oracle knows the final test is 105:115 for this example.
        for i in range(100):
            data.loc[i, "target_exit_date_20d"] = data.date.iloc[i + (17 if i % 7 == 0 else 11)]
        source = data.to_dict("records")
        test_start = source[105]["date"]
        safe = []
        for i in range(105):
            if all(pd.notna(source[i][f"forward_return_{h}d"])
                   and source[i][f"target_exit_date_{h}d"] < test_start for h in (1, 5, 20)):
                safe.append(i)
        self.assertEqual(safe[-1], 93)
        expected = safe[-30:]
        tables, protocol = build_splits(data, SMALL)
        for i in range(3):
            m = selected(tables, fold=f"cv_{i + 1}")
            actual = m.loc[m.role.eq("validation"), "date"].tolist()
            self.assertEqual(actual, [source[j]["date"] for j in expected[i * 10:(i + 1) * 10]])
            self.assertEqual(m.loc[m.nominal_role.eq("train"), "date"].tolist(), data.date.iloc[:expected[i * 10]].tolist())
        self.assertEqual(protocol["pre_test_gap"]["row_count"], 11)
        self.assertEqual(protocol["pre_test_gap"]["start_date"], source[94]["date"])
        self.assertTrue(validate_splits(data, tables, protocol, SMALL)["valid"])

    def test_last_safe_validation_exit_is_allowed(self):
        data = fixture(120)
        tables, _ = build_splits(data, SMALL)
        row = selected(tables, "forward_return_20d", "cv_3").iloc[84]
        self.assertEqual(row.role, "validation")
        self.assertEqual(row.target_exit_date, data.date.iloc[104])
        self.assertLess(row.target_exit_date, data.date.iloc[105])

    def test_exit_equal_to_test_start_is_gap(self):
        data = fixture(120)
        tables, _ = build_splits(data, SMALL)
        row = selected(tables, "forward_return_20d", "cv_3").iloc[85]
        self.assertEqual(row.target_exit_date, data.date.iloc[105])
        self.assertEqual(row.role, "pre_test_gap")
        self.assertFalse(row.is_usable)

    def test_exit_one_session_after_test_start_is_gap(self):
        data = fixture(120)
        tables, _ = build_splits(data, SMALL)
        row = selected(tables, "forward_return_20d", "cv_3").iloc[86]
        self.assertEqual(row.target_exit_date, data.date.iloc[106])
        self.assertEqual(row.role, "pre_test_gap")

    def test_short_horizons_safe_but_20d_unsafe_still_excluded(self):
        data = fixture(120)
        tables, _ = build_splits(data, SMALL)
        self.assertLess(data.target_exit_date_1d.iloc[85], data.date.iloc[105])
        self.assertLess(data.target_exit_date_5d.iloc[85], data.date.iloc[105])
        for target in RETURN_COLUMNS:
            self.assertEqual(selected(tables, target, "cv_3").role.iloc[85], "pre_test_gap")

    def test_irregular_calendar_uses_exit_dates_not_calendar_day_offsets(self):
        data = fixture(120)
        replacement = [(pd.Timestamp("2026-01-02") + pd.Timedelta(days=i * 3 + i // 5)).strftime("%Y-%m-%d")
                       for i in range(120)]
        mapping = dict(zip(data.date, replacement))
        for column in ("date", "target_entry_date", "target_exit_date_1d", "target_exit_date_5d", "target_exit_date_20d"):
            data[column] = data[column].map(mapping)
        tables, protocol = build_splits(data, SMALL)
        self.assertEqual(protocol["pre_test_gap"], {"row_count": 20, "start_date": replacement[85], "end_date": replacement[104]})
        self.assertEqual(protocol["boundaries"]["final_test"]["start"], replacement[105])
        self.assertEqual(tables["cv_folds"].query("fold == 'cv_3' and target_name == 'forward_return_20d'")
                         .max_validation_target_exit_date.iloc[0], replacement[104])

    def test_changed_provenance_changes_gap_row_count(self):
        data = fixture(120)
        for i in range(100):
            data.loc[i, "target_exit_date_20d"] = data.date.iloc[i + 12]
        _, protocol = build_splits(data, SMALL)
        self.assertEqual(protocol["pre_test_gap"]["row_count"], 12)
        self.assertEqual(protocol["pre_test_gap"]["start_date"], data.date.iloc[93])

    def test_helper_protecting_only_1d_5d_has_shorter_gap(self):
        data = fixture(120)
        start = pd.Timestamp(data.date.iloc[105])
        all_registered = validation_history_eligibility(data, start)
        shorter = validation_history_eligibility(data, start, (1, 5))
        self.assertEqual(data.index[all_registered].max(), 84)
        self.assertEqual(data.index[shorter].max(), 99)
        self.assertLess(105 - 1 - data.index[shorter].max(), 105 - 1 - data.index[all_registered].max())
        for horizons in ((), (1, 1), (1, 30)):
            with self.assertRaises(ValueError):
                validation_history_eligibility(data, start, horizons)

    def test_missing_secondary_label_is_not_safe_and_blocks_stay_full(self):
        data = fixture(120)
        data.loc[84, "forward_return_20d"] = np.nan
        tables, protocol = build_splits(data, SMALL)
        self.assertEqual(protocol["pre_test_gap"]["row_count"], 21)
        self.assertFalse(validation_history_eligibility(data, pd.Timestamp(data.date.iloc[105])).iloc[84])
        self.assertTrue(tables["cv_folds"].validation_count_effective.eq(SMALL.validation_size).all())
        self.assertEqual(selected(tables, "forward_return_20d", "final").role.iloc[84], "target_missing")

    def test_missing_exit_is_not_assumed_available_by_helper(self):
        data = fixture(120)
        data.loc[84, "target_exit_date_20d"] = None
        self.assertFalse(validation_history_eligibility(data, pd.Timestamp(data.date.iloc[105])).iloc[84])
        with self.assertRaisesRegex(ValueError, "Missing required target metadata"):
            build_splits(data, SMALL)

    def test_gap_can_return_to_final_training_independently_by_horizon(self):
        data = fixture()
        tables, _ = build_splits(data, SMALL)
        for target in RETURN_COLUMNS:
            for fold in ("cv_1", "cv_2", "cv_3"):
                row = selected(tables, target, fold).iloc[55]
                self.assertEqual(row.role, "pre_test_gap")
                self.assertFalse(row.is_usable)
            row = selected(tables, target, "final").iloc[55]
            self.assertTrue(row.pre_test_gap)
            self.assertEqual(row.final_train_usable, target != "forward_return_20d")
        for row in tables["final_training_pool"].itertuples():
            h = int(row.target_name.removeprefix("forward_return_").removesuffix("d"))
            expected = sum(pd.notna(data[f"forward_return_{h}d"].iloc[i])
                           and data[f"target_exit_date_{h}d"].iloc[i] < data.date.iloc[75] for i in range(55, 75))
            self.assertEqual(row.pre_test_gap_final_train_usable_count, expected)

    def test_gap_does_not_move_test_or_unlabeled_tail(self):
        data = fixture(120)
        tables, protocol = build_splits(data, SMALL)
        final = selected(tables, fold="final")
        self.assertEqual(final.loc[final.role.eq("final_test"), "date"].tolist(), data.date.iloc[105:115].tolist())
        self.assertEqual(final.loc[final.role.eq("unlabeled_tail"), "date"].tolist(), data.date.iloc[115:120].tolist())
        self.assertFalse(final.loc[final.partition.ne("development"), "pre_test_gap"].any())
        self.assertTrue(protocol["final_test_locked"])
        self.assertEqual(protocol["embargo_sessions"], 0)

    def test_insufficient_post_gap_history_fails_without_shrinking(self):
        # Passes the old 185-labelable feasibility check, fails after maturation gap.
        with self.assertRaisesRegex(ValueError, "after dynamic pre_test_gap"):
            build_splits(fixture(190))

    def test_crossing_label_in_any_fold_is_detected(self):
        data = fixture()
        for fold in ("cv_1", "cv_2", "cv_3"):
            tables, protocol = build_splits(data, SMALL)
            m = tables["split_manifest"]
            mask = m.fold.eq(fold) & m.target_name.eq("forward_return_20d") & m.date.eq(data.date.iloc[55])
            m.loc[mask, "nominal_role"] = "validation"
            m.loc[mask, "role"] = "validation"
            m.loc[mask, "is_usable"] = True
            report = validate_splits(data, tables, protocol, SMALL)
            with self.subTest(fold=fold):
                self.assertFalse(report["valid"])
                self.assertEqual(report["validation_label_crosses_final_test_20d"], 1)
                self.assertEqual(report["validation_label_crosses_final_test_total"], 1)
                self.assertEqual(report["validation_test_boundary_status"], "failed")

    def test_missing_label_marked_validation_is_detected(self):
        data = fixture()
        data.loc[30, "forward_return_20d"] = np.nan
        tables, protocol = build_splits(data, SMALL)
        m = tables["split_manifest"]
        mask = m.fold.eq("cv_1") & m.target_name.eq("forward_return_20d") & m.date.eq(data.date.iloc[30])
        m.loc[mask, "role"] = "validation"
        m.loc[mask, "is_usable"] = True
        report = validate_splits(data, tables, protocol, SMALL)
        self.assertFalse(report["valid"])
        self.assertEqual(report["validation_label_availability_violations"], 1)

    def test_gap_metadata_and_policy_tampering_is_detected(self):
        data = fixture()
        for field in ("manifest", "summary", "policy", "gap"):
            tables, protocol = build_splits(data, SMALL)
            if field == "manifest":
                tables["split_manifest"].loc[0, "pre_test_gap"] = True
            elif field == "summary":
                tables["split_summary"].loc[0, "pre_test_gap_rows"] -= 1
            elif field == "policy":
                protocol["pre_test_gap_basis"] = "primary_only"
            else:
                protocol["pre_test_gap"]["row_count"] -= 1
            with self.subTest(field=field):
                self.assertFalse(validate_splits(data, tables, protocol, SMALL)["valid"])

    def test_three_boundary_counters_and_maximum_exits_are_safe(self):
        data = fixture(120)
        tables, protocol = build_splits(data, SMALL)
        report = validate_splits(data, tables, protocol, SMALL)
        self.assertTrue(report["valid"])
        for field in ("train_to_validation_violations", "final_training_to_final_test_violations",
                      "validation_label_availability_violations", "validation_label_crosses_final_test_total"):
            self.assertEqual(report[field], 0)
        for h in (1, 5, 20):
            self.assertEqual(report[f"validation_label_crosses_final_test_{h}d"], 0)
            self.assertLess(report["maximum_validation_target_exit_dates"][f"forward_return_{h}d"], data.date.iloc[105])


class PipelineTests(unittest.TestCase):
    def test_offline_reproducibility_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = write_checkpoint(root)
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
                 patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
                first = run_pipeline(**args)
                contents = {p: p.read_bytes() for p in args["output_dir"].glob("*") if p.is_file()}
                second = run_pipeline(**args)
            self.assertTrue(first["valid"])
            self.assertTrue(first["source_artifacts_unchanged"])
            self.assertEqual(first["output_artifact_sha256"], second["output_artifact_sha256"])
            self.assertEqual(len(contents), 6)
            for path, content in contents.items():
                self.assertEqual(path.read_bytes(), content, str(path))
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            report = json.loads((args["output_dir"] / "NVDA_split_validation.json").read_text(),
                                parse_constant=lambda value: self.fail(f"Nonstandard JSON {value}"))
            self.assertEqual(report["total_rows"], 251)
            self.assertEqual(report["labelable_primary_rows"], 246)
            self.assertEqual(report["validation_status"], "passed")

    def test_stale_stage8_hash_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_checkpoint(Path(tmp))
            args["labeled_path"].write_text(args["labeled_path"].read_text().replace("NVDA", "MSFT", 1))
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                run_pipeline(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_existing_protocol_lock_prevents_date_or_metric_changes(self):
        for key, value in (("primary_metric", "RMSE"), ("final_test_locked", False),
                           ("boundaries", {"final_test": {"start": "2026-01-02", "end": "2026-01-02"}}),
                           ("baselines", []), ("preprocessing", {"fit_scope": "full_sample"})):
            with tempfile.TemporaryDirectory() as tmp:
                args = write_checkpoint(Path(tmp))
                run_pipeline(**args)
                path = args["output_dir"] / "NVDA_evaluation_protocol.json"
                locked = json.loads(path.read_text())
                locked[key] = value
                path.write_text(json.dumps(locked))
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "protocol is locked"):
                    run_pipeline(**args)
                self.assertEqual(path.read_bytes(), before)

    def test_output_symlink_cannot_mutate_labeled_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_checkpoint(Path(tmp))
            args["output_dir"].mkdir()
            (args["output_dir"] / "NVDA_split_manifest.csv").symlink_to(args["labeled_path"])
            before = args["labeled_path"].read_bytes()
            with self.assertRaisesRegex(ValueError, "symlinks"):
                run_pipeline(**args)
            self.assertEqual(args["labeled_path"].read_bytes(), before)

    def test_cli_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_checkpoint(Path(tmp))
            command = [sys.executable, "-m", "nasdaq_research.splits"]
            for key, value in args.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("Primary metric: MAE", result.stdout)
            self.assertIn("Validation: passed", result.stdout)


class ProtocolUpgradeTests(unittest.TestCase):
    @staticmethod
    def install_legacy_protocol(args):
        """Reproduce Stage 9's recorded layout in a temporary checkpoint only."""
        path = args["output_dir"] / "NVDA_evaluation_protocol.json"
        legacy = json.loads(path.read_text())
        for field in ("protocol_version", "registered_targets", "validation_label_availability_rule", "pre_test_gap_policy",
                      "pre_test_gap_basis", "longest_registered_horizon", "cv_validation_policy", "missing_validation_label_policy",
                      "pre_test_gap_final_training_policy", "pre_test_gap"):
            legacy.pop(field)
        legacy["boundaries"].pop("pre_test_gap")
        data = pd.read_csv(args["labeled_path"])
        development = int(data.forward_return_5d.notna().sum()) - 50
        first = development - 75
        legacy["boundaries"]["cv"] = [{"fold": f"cv_{i + 1}", "start": data.date.iloc[first + i * 25],
                                       "end": data.date.iloc[first + (i + 1) * 25 - 1]} for i in range(3)]
        legacy["validation_test_calendar_overlap_policy"] = (
            "Report validation exits reaching the test calendar; keep prescribed blocks unchanged. No evaluation-label purge is applied.")
        legacy["stage10_readiness"] = "NOT READY FOR STAGE 10"
        path.write_text(json.dumps(legacy, indent=2) + "\n")
        return legacy

    def test_authorized_legacy_upgrade_preserves_test_and_evaluation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_checkpoint(Path(tmp))
            run_pipeline(**args)
            legacy = self.install_legacy_protocol(args)
            run_pipeline(**args)
            path = args["output_dir"] / "NVDA_evaluation_protocol.json"
            upgraded = json.loads(path.read_text())
            self.assertEqual(upgraded["protocol_version"], "9.1")
            for field in ("source_labeled_sha256", "config", "primary_target", "primary_metric", "secondary_metrics", "baselines", "preprocessing"):
                self.assertEqual(upgraded[field], legacy[field])
            self.assertEqual(upgraded["boundaries"]["final_test"], legacy["boundaries"]["final_test"])
            self.assertNotEqual(upgraded["boundaries"]["cv"], legacy["boundaries"]["cv"])
            before = {p: p.read_bytes() for p in args["output_dir"].iterdir()}
            run_pipeline(**args)
            self.assertEqual(before, {p: p.read_bytes() for p in args["output_dir"].iterdir()})

    def test_legacy_upgrade_rejects_changed_locked_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_checkpoint(Path(tmp))
            run_pipeline(**args)
            legacy = self.install_legacy_protocol(args)
            path = args["output_dir"] / "NVDA_evaluation_protocol.json"
            legacy["boundaries"]["final_test"]["start"] = "2026-01-02"
            path.write_text(json.dumps(legacy))
            before = {p: p.read_bytes() for p in args["output_dir"].iterdir()}
            with self.assertRaisesRegex(ValueError, "protocol is locked"):
                run_pipeline(**args)
            self.assertEqual(before, {p: p.read_bytes() for p in args["output_dir"].iterdir()})

    def test_removing_modern_version_cannot_bypass_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = write_checkpoint(Path(tmp))
            run_pipeline(**args)
            path = args["output_dir"] / "NVDA_evaluation_protocol.json"
            altered = json.loads(path.read_text())
            altered.pop("protocol_version")
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "protocol is locked"):
                run_pipeline(**args)


if __name__ == "__main__":
    unittest.main()
