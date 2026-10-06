"""Stage15 authority, leakage, adversarial mutations and saved-model oracles."""

import ast
import copy
from functools import lru_cache
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import fundamental_ablation as a


@lru_cache(maxsize=1)
def context():
    with a.authority.offline_only():
        protocol = a.load_protocol()
        data, inventory, purges, members, features, evidence = a.load_fold_context(protocol)
        control = a.build_models(data, inventory, members)
        replay, reproduction = a.control_reproduction(control.tables['oof_predictions'],
            a.h.read_csv(a.w.OUTPUT_DIR / (a.w.PREFIX + 'oof_predictions.csv')), protocol['control_reproduction_tolerance'])
        data, daily, snapshots, facts, sessions = a.load_fundamentals(data, protocol)
        pit_table, pit = a.pit_audit(daily, snapshots, facts, sessions)
        treatment = a.build_models(data, inventory, members, treatment=True)
        result = a.merge_results(control, treatment)
        return protocol, data, inventory, purges, members, features, evidence, control, replay, reproduction, daily, snapshots, facts, sessions, pit_table, pit, result


class Stage15Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.protocol, cls.data, cls.inventory, cls.purges, cls.members, cls.features,
         cls.evidence, cls.control, cls.replay, cls.reproduction, cls.daily, cls.snapshots,
         cls.facts, cls.sessions, cls.pit_table, cls.pit, cls.result) = context()

    def first_fold(self, data=None):
        fold = self.inventory.fold_id.iloc[0]
        return a.build_models(self.data if data is None else data, self.inventory.iloc[:1],
                              {fold: self.members[fold]}, treatment=True)


class AuthorityAndControlTests(Stage15Case):
    def test_registered_protocol_and_hashes_are_locked(self):
        registration = a.authority.load_json(a.output('registration'))
        self.assertEqual(a.h.sha256(a.output('protocol')), registration['protocol_sha256'])
        self.assertEqual(self.protocol, a.contract())

    def test_canonical_feature_scope_and_identity(self):
        manifest = a.authority.require_canonical_vintage(self.protocol['canonical_vintage_id'], required_features=a.TREATMENT)
        self.assertEqual(manifest['approval_status'], a.authority.QUALIFIED)
        self.assertTrue(manifest['expanded_dataset_approved_for_downstream_research'])

    def test_all_22_folds_match_frozen_inventory_exactly(self):
        path = a.w.OUTPUT_DIR / (a.w.PREFIX + 'fold_inventory.csv')
        self.assertEqual(path.read_text(), a.h.csv_text(self.inventory))
        self.assertEqual(len(self.inventory), 22)

    def test_all_training_dates_match_stage14_relative_fit_membership(self):
        historic = a.h.read_csv(a.w.OUTPUT_DIR / (a.w.PREFIX + 'fit_membership.csv'))
        historic = historic.loc[historic.model.isin(a.CONTROLS)].reset_index(drop=True)
        pd.testing.assert_frame_equal(self.control.tables['fit_membership'], historic, check_exact=True)

    def test_training_target_realization_and_purge_match_stage14(self):
        self.assertEqual(a.h.csv_text(self.purges), (a.w.OUTPUT_DIR / (a.w.PREFIX + 'target_purge_audit.csv')).read_text())
        self.assertEqual(int(self.purges.violations.sum()), 0)
        self.assertTrue(self.inventory.target_purged_rows.eq(5).all())

    def test_relative_whitelist_comes_from_frozen_manifest(self):
        rows = self.features.loc[self.features.representation.eq('relative')]
        self.assertEqual(tuple(rows.feature_name), a.CONTROL)
        self.assertEqual(len(a.CONTROL), 9)

    def test_fundamental_whitelist_is_exact_stage6_order(self):
        self.assertEqual(a.FUNDAMENTALS, a.rf.FEATURE_COLUMNS)
        self.assertEqual(len(a.FUNDAMENTALS), 30)
        self.assertEqual(len(a.TREATMENT), 39)
        groups = [a.baseline.FEATURE_GROUPS[c] for c in a.FUNDAMENTALS]
        self.assertEqual({g: groups.count(g) for g in set(groups)}, {'quarterly': 13, 'annual': 13, 'balance_sheet': 4})

    def test_control_OLS_predictions_reproduce_all_1367_dates_exactly(self):
        rows = self.replay.loc[self.replay.model.eq(a.CONTROLS[0])]
        self.assertEqual(len(rows), 1367)
        self.assertTrue(rows.absolute_difference.eq(0).all())

    def test_control_Ridge_predictions_reproduce_all_1367_dates_exactly(self):
        rows = self.replay.loc[self.replay.model.eq(a.CONTROLS[1])]
        self.assertEqual(len(rows), 1367)
        self.assertTrue(rows.absolute_difference.eq(0).all())

    def test_saved_control_preprocessing_and_coefficients_match_stage14(self):
        old = a.h.read_csv(a.w.OUTPUT_DIR / (a.w.PREFIX + 'model_parameters.csv'))
        old = old.loc[old.model.isin(a.CONTROLS)].reset_index(drop=True)
        current = self.control.tables['model_parameters'].loc[:, old.columns]
        pd.testing.assert_frame_equal(current, old, check_exact=True)

    def test_reproduction_tolerance_does_not_hide_prediction_drift(self):
        replay = self.control.tables['oof_predictions'].copy()
        replay.loc[0, a.CONTROLS[0]] += 1e-6
        _, report = a.control_reproduction(replay, self.control.tables['oof_predictions'], self.protocol['control_reproduction_tolerance'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['tolerance_violations'], 1)

    def test_missing_OOF_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            a.control_reproduction(self.control.tables['oof_predictions'].iloc[1:], self.control.tables['oof_predictions'], self.protocol['control_reproduction_tolerance'])

    def test_duplicate_OOF_date_is_rejected(self):
        replay = self.control.tables['oof_predictions'].copy()
        replay.loc[1, 'date'] = replay.loc[0, 'date']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            a.control_reproduction(replay, self.control.tables['oof_predictions'], self.protocol['control_reproduction_tolerance'])

    def test_fold_or_target_drift_is_rejected(self):
        for column, value in [('fold_id', '2020Q4'), ('y_true', 999.)]:
            replay = self.control.tables['oof_predictions'].copy()
            replay.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, 'coverage'):
                a.control_reproduction(replay, self.control.tables['oof_predictions'], self.protocol['control_reproduction_tolerance'])

    def test_control_failure_blocks_all_treatment_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = copy.deepcopy(self.reproduction)
            failed['passed'] = False
            with patch.object(a, 'control_reproduction', return_value=(self.replay, failed)), \
                 patch.object(a, 'fit_treatment', side_effect=AssertionError('Treatment must not run')) as treatment:
                with self.assertRaisesRegex(ValueError, 'Control reproduction failed'):
                    a.run_pipeline(Path(directory))
                treatment.assert_not_called()

    def test_tampered_fold_membership_blocks_fit(self):
        fold = self.inventory.fold_id.iloc[0]
        changed = {fold: copy.deepcopy(self.members[fold])}
        changed[fold][0].pop()
        with self.assertRaisesRegex(ValueError, 'membership'):
            a.build_models(self.data, self.inventory.iloc[:1], changed, treatment=True)


class PITTests(Stage15Case):
    def test_every_development_date_and_feature_is_audited(self):
        self.assertEqual(len(self.pit_table), 1620 * 30)
        self.assertFalse(self.pit_table.duplicated(['feature_date', 'feature']).any())

    def test_all_filing_effective_dates_are_eligible(self):
        present = self.pit_table.loc[self.pit_table.source_filing.notna()]
        self.assertTrue(present.eligible_on_feature_date.all())
        self.assertTrue(present.effective_date.le(present.feature_date).all())
        self.assertTrue(present.filing_date.lt(present.feature_date).all())

    def test_activation_is_next_observed_session_including_presample(self):
        present = self.pit_table.loc[self.pit_table.source_filing.notna()]
        self.assertTrue(present.effective_date.eq(present.expected_effective_date).all())
        self.assertLess(self.sessions[0], '2010-01-01')
        first = present.loc[present.feature_date.eq('2020-01-02')]
        self.assertTrue(first.effective_date.lt('2020-01-02').all())

    def test_independent_source_state_and_growth_oracles_pass(self):
        self.assertEqual(self.pit['independent_state_oracle']['violations'], 0)
        self.assertEqual(self.pit['growth_reference_oracle']['violations'], 0)
        self.assertEqual(self.pit['replay_violations'], 0)

    def test_daily_states_match_frozen_snapshots_exactly(self):
        self.assertTrue(self.pit_table.canonical_state_exact.all())
        self.assertEqual(self.pit['canonical_exact_state_cells'], 48600)

    def test_growth_reference_is_not_a_future_filing(self):
        rows = self.pit_table.loc[self.pit_table.reference_filing_date.notna()]
        self.assertTrue(rows.reference_filing_date.le(rows.filing_date).all())

    def test_future_filing_metadata_injection_fails_audit(self):
        changed = self.daily.iloc[:1].copy()
        changed.loc[changed.index[0], 'q_filing_date'] = '2025-01-01'
        _, report = a.pit_audit(changed, self.snapshots, self.facts, self.sessions)
        self.assertFalse(report['passed'])

    def test_period_end_backfill_is_detected(self):
        changed = self.daily.iloc[:1].copy()
        changed.loc[changed.index[0], 'q_effective_date'] = changed.q_period_end.iloc[0]
        _, report = a.pit_audit(changed, self.snapshots, self.facts, self.sessions)
        self.assertGreater(report['future_information_violations'], 0)

    def test_validation_state_is_checked_against_then_known_filings(self):
        dates = [self.members[f][1][0] for f in self.members]
        sample = self.daily.loc[self.daily.date.isin(dates)]
        _, report = a.pit_audit(sample, self.snapshots, self.facts, self.sessions)
        self.assertTrue(report['passed'])

    def test_future_filing_mutation_preserves_past_states_and_predictions(self):
        fold = '2023Q4'
        train, valid = self.members[fold]
        candidates = self.facts.loc[self.facts.data_period_type.eq('quarterly') &
            self.facts.filing_date.between(valid[0], valid[-1]) & self.facts.research_and_development.gt(0)]
        self.assertFalse(candidates.empty)
        index = candidates.index[0]
        filed = self.facts.loc[index, 'filing_date']
        mutated = self.facts.copy()
        mutated.loc[index, 'research_and_development'] *= 1.5
        ordinary = a.replay_pit_values(self.facts, self.sessions)
        changed = a.replay_pit_values(mutated, self.sessions)
        before = ordinary.date.le(filed)
        pd.testing.assert_frame_equal(ordinary.loc[before], changed.loc[before], check_exact=True)
        effective = next(d for d in self.sessions if d > filed)
        self.assertNotEqual(ordinary.loc[ordinary.date.eq(effective), 'q_rd_to_revenue'].iloc[0],
                            changed.loc[changed.date.eq(effective), 'q_rd_to_revenue'].iloc[0])
        keyed = self.data.set_index('date', drop=False)
        fit = a.fit_treatment('ridge', a.predictor_matrix(keyed.loc[train], True), keyed.loc[train, a.PRIMARY_TARGET])
        vx1 = a.predictor_matrix(keyed.loc[valid], True)
        vx2 = vx1.copy()
        vx2.loc[:, a.FUNDAMENTALS] = changed.set_index('date').loc[valid, a.FUNDAMENTALS]
        prior = np.array(valid) < effective
        np.testing.assert_array_equal(a.baseline.predict_linear(fit, vx1)[prior], a.baseline.predict_linear(fit, vx2)[prior])

    def test_PIT_failure_blocks_treatment(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = copy.deepcopy(self.pit)
            failed['passed'] = False
            with patch.object(a, 'pit_audit', return_value=(self.pit_table, failed)), \
                 patch.object(a, 'fit_treatment', side_effect=AssertionError('Treatment must not run')) as treatment:
                with self.assertRaisesRegex(ValueError, 'PIT source'):
                    a.run_pipeline(Path(directory))
                treatment.assert_not_called()


class PreprocessingAndOOFTests(Stage15Case):
    def test_all_1367_treatment_dates_and_folds_equal_control(self):
        oof = self.result.tables['oof_predictions']
        self.assertEqual(len(oof), 1367)
        self.assertEqual(oof.date.tolist(), self.control.tables['oof_predictions'].date.tolist())
        self.assertEqual(oof.fold_id.tolist(), self.control.tables['oof_predictions'].fold_id.tolist())
        self.assertFalse(oof.date.duplicated().any())
        self.assertTrue(np.isfinite(oof.loc[:, a.TREATMENTS].to_numpy()).all())

    def test_full_saved_parameter_reconstruction_all_88_fits(self):
        saved = a.h.read_csv(a.output('model_parameters', suffix='csv'))
        oof = a.h.read_csv(a.output('oof_predictions', suffix='csv'))
        keyed = self.data.set_index('date', drop=False)
        for (fold, model), rows in saved.groupby(['fold_id', 'model'], sort=False):
            valid = self.members[fold][1]
            source = keyed if model in a.TREATMENTS else keyed.loc[:, a.w.MATRIX_COLUMNS]
            reconstructed = a.w.reconstruction(rows, a.predictor_matrix(source.loc[valid], model in a.TREATMENTS))
            np.testing.assert_allclose(reconstructed, oof.loc[oof.fold_id.eq(fold), model], rtol=1e-10, atol=1e-12)

    def test_saved_training_statistics_match_independent_oracle(self):
        report = a.validate_result(self.data, self.inventory, self.purges, self.members, self.result, self.reproduction, self.pit)
        self.assertTrue(report['valid'], report)
        self.assertTrue(report['checks']['saved_preprocessing_training_only'])

    def test_validation_missingness_cannot_change_selected_features(self):
        changed = self.data.copy()
        train, valid = next(iter(self.members.values()))
        changed.loc[changed.date.isin(valid), a.FUNDAMENTALS] = np.nan
        altered = self.first_fold(changed)
        for model in a.TREATMENTS:
            fold = self.inventory.fold_id.iloc[0]
            pp, other = self.result.fits[fold, model].preprocessing, altered.fits[fold, model].preprocessing
            self.assertEqual(pp.features, other.features)
            pd.testing.assert_frame_equal(pp.usage, other.usage, check_exact=True)
            np.testing.assert_array_equal(pp.scaler.mean_, other.scaler.mean_)

    def test_future_fold_values_cannot_change_current_selection_or_prediction(self):
        changed = self.data.copy()
        _, valid = next(iter(self.members.values()))
        changed.loc[changed.date.gt(valid[-1]), a.FUNDAMENTALS] = 1e8
        altered = self.first_fold(changed)
        fold = self.inventory.fold_id.iloc[0]
        for model in a.TREATMENTS:
            self.assertEqual(altered.fits[fold, model].preprocessing.features, self.result.fits[fold, model].preprocessing.features)
            np.testing.assert_array_equal(altered.tables['oof_predictions'][model],
                self.result.tables['oof_predictions'].loc[self.result.tables['oof_predictions'].fold_id.eq(fold), model])

    def test_training_median_imputes_validation_missing_values(self):
        fit = self.result.fits['2026Q2', a.TREATMENTS[1]]
        pp = fit.preprocessing
        missing = pd.DataFrame({c: [np.nan] for c in a.TREATMENT})
        actual = a.baseline.transform_features(pp, missing)
        expected = (pp.medians.loc[list(pp.features)].to_numpy() - pp.scaler.mean_) / pp.scaler.scale_
        np.testing.assert_array_equal(actual[0], expected)

    def test_exact_50_percent_coverage_is_kept(self):
        frame = pd.DataFrame({c: [1., 2., 3., 4.] for c in a.TREATMENT})
        frame.loc[2:, 'q_gross_margin'] = np.nan
        target = pd.Series([.1, .2, -.1, 0.])
        fit = a.fit_treatment('ridge', frame, target)
        self.assertIn('q_gross_margin', fit.preprocessing.features)
        self.assertEqual(fit.preprocessing.usage.set_index('feature_name').loc['q_gross_margin', 'training_coverage'], .50)

    def test_low_coverage_feature_drops_without_dropping_rows(self):
        frame = pd.DataFrame({c: [1., 2., 3., 4.] for c in a.TREATMENT})
        frame.loc[1:, 'q_gross_margin'] = np.nan
        fit = a.fit_treatment('ridge', frame, pd.Series([.1, .2, -.1, 0.]))
        self.assertNotIn('q_gross_margin', fit.preprocessing.features)
        self.assertEqual(len(fit.train_dates), 4)

    def test_exact_zero_variance_removed_after_imputation(self):
        frame = pd.DataFrame({c: [1., 2., 3., 4.] for c in a.TREATMENT})
        frame['q_gross_margin'] = [1., 1., np.nan, np.nan]
        fit = a.fit_treatment('ols', frame, pd.Series([.1, .2, -.1, 0.]))
        self.assertEqual(fit.preprocessing.usage.set_index('feature_name').loc['q_gross_margin', 'drop_reason'], 'zero_training_variance')

    def test_no_complete_case_filter_even_with_all_validation_fundamentals_missing(self):
        changed = self.data.copy()
        _, valid = next(iter(self.members.values()))
        changed.loc[changed.date.isin(valid), a.FUNDAMENTALS] = np.nan
        result = self.first_fold(changed)
        self.assertEqual(result.tables['oof_predictions'].date.tolist(), valid)
        self.assertTrue(np.isfinite(result.tables['oof_predictions'].loc[:, a.TREATMENTS]).all().all())

    def test_no_global_dropna_in_stage15_pipeline(self):
        tree = ast.parse(Path(a.__file__).read_text())
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr == 'dropna' for node in ast.walk(tree)))

    def test_zero_and_training_historical_mean_match_all_folds(self):
        keyed = self.data.set_index('date')
        oof = self.result.tables['oof_predictions']
        for fold, (train, valid) in self.members.items():
            block = oof.loc[oof.fold_id.eq(fold)]
            np.testing.assert_array_equal(block.zero_return, np.zeros(len(valid)))
            np.testing.assert_array_equal(block.historical_mean, np.full(len(valid), keyed.loc[train, a.PRIMARY_TARGET].mean()))

    def test_validation_targets_cannot_change_fitted_baseline(self):
        fold = self.inventory.fold_id.iloc[0]
        _, valid = self.members[fold]
        changed = self.data.loc[:, a.w.MATRIX_COLUMNS].copy()
        changed.loc[changed.date.isin(valid), a.PRIMARY_TARGET] = 999.
        result = a.build_models(changed, self.inventory.iloc[:1], {fold: self.members[fold]})
        np.testing.assert_array_equal(result.tables['oof_predictions'].historical_mean,
            self.control.tables['oof_predictions'].loc[self.control.tables['oof_predictions'].fold_id.eq(fold), 'historical_mean'])


class SafetyAndAssessmentTests(Stage15Case):
    def test_Volume_excluded_from_every_whitelist_and_dependency(self):
        manifests = a.feature_manifests(self.features)
        for manifest in manifests:
            self.assertFalse(manifest.Volume_dependency.any())
            self.assertFalse(manifest.source_fields.str.lower().str.contains('volume').any())
        self.assertFalse(any('volume' in c.lower() for c in a.TREATMENT))

    def test_Volume_metadata_and_secondary_label_injection_rejected(self):
        for name in ('volume', 'forward_return_1d', 'forward_return_20d', 'days_since_filing', 'feature_missing_flag'):
            changed = self.data.assign(**{name: 1.})
            with self.assertRaisesRegex(ValueError, 'schema'):
                a.predictor_matrix(changed, True)

    def test_no_Volume_or_metadata_in_actual_scalers(self):
        for (_, model), fit in self.result.fits.items():
            self.assertTrue(set(fit.preprocessing.features) <= set(a.TREATMENT if model in a.TREATMENTS else a.CONTROL))
            self.assertEqual(tuple(fit.preprocessing.scaler.feature_names_in_), fit.preprocessing.features)

    def test_no_new_estimator_or_alpha(self):
        self.assertEqual(len(self.result.fits), 88)
        for (_, model), fit in self.result.fits.items():
            self.assertTrue(fit.estimator.fit_intercept)
            if 'ridge' in model:
                self.assertEqual(fit.estimator.alpha, 1.)
                self.assertEqual(fit.estimator.solver, 'svd')
            else:
                self.assertEqual(fit.estimator.n_jobs, 1)

    def test_unknown_family_and_partial_fundamental_block_rejected(self):
        keyed = self.data.set_index('date', drop=False)
        train, _ = next(iter(self.members.values()))
        x = a.predictor_matrix(keyed.loc[train], True)
        for family, values in [('lasso', x), ('ridge', x.drop(columns='q_gross_margin'))]:
            with self.assertRaisesRegex(ValueError, 'Fixed family'):
                a.fit_treatment(family, values, keyed.loc[train, a.PRIMARY_TARGET])

    def test_Final_Test_gap_and_training_pool_are_locked(self):
        lock = a.w.final_lock_audit(self.data, self.inventory, self.result, self.evidence)
        self.assertTrue(lock['valid'])
        for name in ('final_test_predictions_generated', 'final_test_metrics_computed', 'final_training_pool_used'):
            self.assertFalse(lock[name])
        self.assertTrue(self.result.tables['oof_predictions'].date.lt(a.w.GAP_START).all())

    def test_gap_observation_injection_rejected(self):
        changed = self.data.copy()
        changed.loc[changed.index[-1], 'date'] = a.w.GAP_START
        with self.assertRaisesRegex(ValueError, 'Gap/Final'):
            a.predictor_matrix(changed, True)

    def test_offline_guard_blocks_transport(self):
        import socket
        with a.authority.offline_only():
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                socket.create_connection(('example.com', 443))

    def test_upstream_all_protected_hashes_unchanged(self):
        self.assertEqual(a.verify_upstream()['mutation_violations'], 0)
        self.assertEqual(a.verify_upstream()['files_checked'], 299)

    def test_freeze_preserves_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frozen.json'
            a.freeze(path, '{}\n')
            a.freeze(path, '{}\n')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                a.freeze(path, '{"changed":true}\n')
            self.assertEqual(path.read_text(), '{}\n')

    def test_output_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'source.json'
            target.write_text('{}\n')
            link = Path(directory) / 'link.json'
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                a.freeze(link, '{}\n')

    def test_all_fold_MAEs_recompute_from_saved_predictions(self):
        oof, metrics = self.result.tables['oof_predictions'], self.result.tables['fold_metrics']
        for row in metrics.itertuples(index=False):
            block = oof.loc[oof.fold_id.eq(row.fold_id)]
            self.assertEqual(row.MAE, float(np.abs(block.y_true - block[row.model]).mean()))

    def test_incremental_comparisons_are_paired_equal_weight(self):
        _, incremental, _, _ = a.aggregate_results(self.result.tables['oof_predictions'], self.result.tables['fold_metrics'])
        for model in a.TREATMENTS:
            per = incremental.loc[incremental.model.eq(model) & incremental.scope.eq('fold')]
            summary = incremental.loc[incremental.model.eq(model) & incremental.scope.eq('aggregate')].iloc[0]
            self.assertEqual(len(per), 22)
            self.assertEqual(summary.mean_delta_MAE, per.paired_delta_MAE.mean())
            self.assertEqual(summary.win_rate, per.paired_delta_MAE.lt(0).mean())

    def test_robust_incremental_gate_and_ties(self):
        rows = pd.DataFrame([{'scope': 'aggregate', 'model': name, 'mean_delta_MAE': -.01,
            'median_delta_MAE': -.01, 'win_rate': .5} for name in a.TREATMENTS])
        self.assertEqual(a.incremental_assessment(rows)['classification'], 'NO_ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE')
        rows.loc[0, 'win_rate'] = .51
        self.assertEqual(a.incremental_assessment(rows)['classification'], 'ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE')
        rows.loc[0, 'median_delta_MAE'] = 0.
        self.assertEqual(a.incremental_assessment(rows)['classification'], 'MIXED_OR_REGIME_DEPENDENT_INCREMENTAL_VALUE')

    def test_absolute_assessment_is_exact_stage14_six_gate_logic(self):
        _, _, absolute, _ = a.aggregate_results(self.result.tables['oof_predictions'], self.result.tables['fold_metrics'])
        rows = absolute.loc[absolute.scope.eq('aggregate')].rename(columns={'reference': 'baseline'}).copy()
        rows['model'] = rows.model.replace(dict(zip(a.TREATMENTS, a.CONTROLS)))
        self.assertEqual(a.absolute_assessment(absolute)['classification'], a.w.predictive_assessment(rows)['classification'])

    def test_numerical_diagnostics_do_not_filter_rank_deficient_features(self):
        frame = pd.DataFrame({c: [1., 2., 3., 4.] for c in a.TREATMENT})
        fit = a.fit_treatment('ols', frame, pd.Series([.1, .2, -.1, 0.]))
        self.assertEqual(len(fit.preprocessing.features), 39)
        self.assertTrue(np.isfinite(a.baseline.predict_linear(fit, frame)).all())

    def test_prediction_stability_thresholds_recomputed(self):
        metrics = self.result.tables['fold_metrics']
        oof = self.result.tables['oof_predictions']
        for row in metrics.itertuples(index=False):
            pred = oof.loc[oof.fold_id.eq(row.fold_id), row.model].to_numpy()
            self.assertEqual(row.max_abs_prediction, float(np.abs(pred).max()))
            self.assertEqual(row.fraction_abs_prediction_gt_0_20, float((np.abs(pred) > .20).mean()))

    def test_extrapolation_uses_training_imputation_and_scaler(self):
        fold, model = '2026Q2', a.TREATMENTS[0]
        pp = self.result.fits[fold, model].preprocessing
        train, valid = self.members[fold]
        keyed = self.data.set_index('date', drop=False)
        z = a.baseline.transform_features(pp, a.predictor_matrix(keyed.loc[valid], True))
        rows = self.result.tables['extrapolation_audit'].loc[
            self.result.tables['extrapolation_audit'].fold_id.eq(fold) & self.result.tables['extrapolation_audit'].representation.eq('relative_fundamentals')].set_index('feature')
        for j, name in enumerate(pp.features):
            self.assertEqual(rows.loc[name, 'max_abs_validation_z'], float(np.abs(z[:, j]).max()))

    def test_nonfinite_features_fail_loudly(self):
        changed = self.data.copy()
        changed.loc[0, 'q_gross_margin'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            a.predictor_matrix(changed, True)

    def test_isolated_offline_pipeline_reproduces_every_generated_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            with a.authority.offline_only():
                a.run_pipeline(Path(directory))
            paths = sorted(Path(directory).glob(a.PREFIX + '*'))
            self.assertGreater(len(paths), 25)
            for path in paths:
                self.assertEqual(path.read_bytes(), (a.OUTPUT_DIR / path.name).read_bytes(), path.name)


if __name__ == '__main__':
    unittest.main()
