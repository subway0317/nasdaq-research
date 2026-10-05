"""Stage 14 frozen folds, source boundaries, leakage and reconstruction oracles."""

import ast
import copy
from functools import lru_cache
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import walk_forward as w
from nasdaq_research import market_representation as r
from nasdaq_research import modeling as b
from nasdaq_research import historical_expansion as h


@lru_cache(maxsize=1)
def context():
    with w.authority.offline_only():
        data,inventory,purges,members,features,evidence=w.prepare_and_freeze()
        result=w.build_walk_forward(data,inventory,members)
    return data,inventory,purges,members,features,evidence,result


class FoldAndPurgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.inventory,cls.purges,cls.members,cls.features,cls.evidence,cls.result=context()

    def test_first_fold_is_2021_Q1(self):
        row=self.inventory.iloc[0]
        self.assertEqual(row.fold_id,'2021Q1')
        self.assertEqual(row.validation_calendar_start,'2021-01-01')
        self.assertEqual(row.validation_calendar_end,'2021-03-31')
        self.assertEqual(row.effective_train_start,'2020-01-02')

    def test_all_22_folds_are_calendar_quarters(self):
        self.assertEqual(len(self.inventory),22)
        for row in self.inventory.itertuples():
            quarter=pd.Period(row.validation_calendar_start,freq='Q')
            self.assertEqual(row.validation_calendar_start,quarter.start_time.strftime('%Y-%m-%d'))
            if row.fold_id!='2026Q2':
                self.assertEqual(row.validation_calendar_end,quarter.end_time.strftime('%Y-%m-%d'))

    def test_training_expands_monotonically_from_same_start(self):
        self.assertTrue(self.inventory.effective_train_rows.diff().dropna().gt(0).all())
        previous=set()
        for training,_ in self.members.values():
            self.assertTrue(previous<=set(training));previous=set(training)
            self.assertEqual(training[0],w.DEVELOPMENT_START)

    def test_validation_windows_do_not_overlap(self):
        dates=[d for _,v in self.members.values() for d in v]
        self.assertEqual(len(dates),len(set(dates)))
        self.assertEqual(dates,sorted(dates))

    def test_last_fold_is_partial_2026_Q2(self):
        row=self.inventory.iloc[-1]
        self.assertEqual(row.fold_id,'2026Q2')
        self.assertEqual(row.validation_calendar_end,'2026-06-12')
        self.assertEqual(row.validation_last_session,'2026-06-12')
        self.assertTrue(row.partial)

    def test_no_feature_rows_touch_gap_Final_Test_or_tail(self):
        self.assertTrue(self.data.date.le(w.DEVELOPMENT_END).all())
        for training,valid in self.members.values():
            self.assertTrue(all(d<w.GAP_START for d in (*training,*valid)))

    def test_ordinary_purge_is_five_sessions_not_twenty(self):
        self.assertTrue(self.inventory.target_purged_rows.eq(5).all())
        self.assertTrue(self.purges.target_horizon.eq(5).all())
        self.assertEqual(self.evidence['gap_rows'],20)

    def test_training_labels_realize_strictly_before_validation(self):
        usable=self.purges.loc[self.purges.eligible]
        self.assertTrue(usable.target_end_date.lt(usable.validation_start).all())
        self.assertTrue(usable.training_feature_date.lt(usable.validation_start).all())
        self.assertTrue(self.purges.violations.eq(0).all())

    def test_purge_eligibility_uses_independent_observed_positions(self):
        sessions=w.read_sessions(h.OUTPUT_DIR/'NVDA_stage13_research_matrix.csv')
        positions={d:i for i,d in enumerate(sessions)}
        for row in self.purges.loc[self.purges.target_purged|self.purges.groupby('fold_id').cumcount().eq(0)].itertuples():
            self.assertEqual(row.oracle_entry_date,sessions[positions[row.training_feature_date]+1])
            self.assertEqual(row.oracle_target_end_date,sessions[positions[row.training_feature_date]+5])
            self.assertEqual(row.eligible,row.oracle_target_end_date<row.validation_start)

    def test_frozen_membership_tampering_is_rejected(self):
        inventory=self.inventory.iloc[:1].copy();fold=inventory.fold_id.iloc[0]
        members={fold:copy.deepcopy(self.members[fold])}
        members[fold][0].pop()
        with self.assertRaisesRegex(ValueError,'membership'):
            w.build_walk_forward(self.data,inventory,members)

    def test_5d_exit_metadata_tampering_is_detected(self):
        changed=self.data.copy();changed.loc[0,'target_exit_date_5d']='2025-01-01'
        sessions=w.read_sessions(h.OUTPUT_DIR/'NVDA_stage13_research_matrix.csv')
        _,purges,_=w.construct_folds(changed,sessions,w.load_protocol()['fold_calendar_definitions'])
        self.assertGreater(purges.violations.sum(),0)

    def test_no_complete_case_filtering_for_features(self):
        changed=self.data.copy();changed.loc[10,'simple_return']=np.nan
        sessions=w.read_sessions(h.OUTPUT_DIR/'NVDA_stage13_research_matrix.csv')
        inventory,_,members=w.construct_folds(changed,sessions,w.load_protocol()['fold_calendar_definitions'])
        self.assertEqual(len(changed),len(self.data))
        self.assertEqual(members,self.members)
        pd.testing.assert_frame_equal(inventory,self.inventory)


class ModelAndReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.inventory,cls.purges,cls.members,cls.features,cls.evidence,cls.result=context()

    def test_current_development_matrix_is_exact_market_only(self):
        self.assertEqual(tuple(self.data),w.MATRIX_COLUMNS)
        self.assertNotIn('volume',self.data)
        self.assertNotIn('forward_return_1d',self.data)
        self.assertNotIn('forward_return_20d',self.data)

    def test_representation_has_nine_features_and_three_substitutions(self):
        original=r.representation(self.data,'control');relative=r.representation(self.data,'treatment')
        self.assertEqual(len(original.columns),9);self.assertEqual(len(relative.columns),9)
        self.assertEqual(set(original)-set(relative),set(r.REPLACEMENTS))
        self.assertEqual(set(relative)-set(original),set(r.REPLACEMENTS.values()))

    def test_six_unchanged_features_are_exactly_identical(self):
        original=r.representation(self.data,'control');relative=r.representation(self.data,'treatment')
        common=[n for n in r.CONTROL if n not in r.REPLACEMENTS]
        pd.testing.assert_frame_equal(original[common],relative[common],check_exact=True)

    def test_relative_formula_uses_authoritative_Stage11_implementation(self):
        actual=r.representation(self.data,'treatment')
        for old,new in r.REPLACEMENTS.items():
            np.testing.assert_allclose(actual[new],self.data.close/self.data[old]-1,rtol=0,atol=1e-12)

    def test_all_estimators_are_fixed_OLS_or_Ridge_alpha_one(self):
        self.assertEqual(len(self.result.fits),88)
        for (_,model),fitted in self.result.fits.items():
            self.assertTrue(fitted.estimator.fit_intercept)
            if 'ridge' in model:
                self.assertEqual(fitted.estimator.alpha,1.0)
                self.assertEqual(fitted.estimator.solver,'svd')
            else:self.assertEqual(fitted.estimator.n_jobs,1)

    def test_all_88_saved_parameter_sets_reconstruct_predictions(self):
        parameters=self.result.tables['model_parameters'];oof=self.result.tables['oof_predictions']
        keyed=self.data.set_index('date')
        for (fold,model),rows in parameters.groupby(['fold_id','model'],sort=False):
            valid=self.members[fold][1];rep=w.MODEL_SPECS[model][0]
            x=r.representation(keyed.loc[valid],w.ARMS[rep])
            expected=w.reconstruction(rows,x)
            actual=oof.loc[oof.fold_id.eq(fold),model].to_numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)

    def test_formal_CSV_roundtrip_parameters_reconstruct_representative_folds(self):
        parameters=h.read_csv(w.OUTPUT_DIR/(w.PREFIX+'model_parameters.csv'))
        oof=h.read_csv(w.OUTPUT_DIR/(w.PREFIX+'oof_predictions.csv'))
        keyed=self.data.set_index('date')
        for fold in ('2021Q1','2023Q4','2026Q2'):
            valid=self.members[fold][1]
            for model in w.LEARNED:
                rows=parameters.loc[parameters.fold_id.eq(fold)&parameters.model.eq(model)]
                rep=w.MODEL_SPECS[model][0]
                expected=w.reconstruction(rows,r.representation(keyed.loc[valid],w.ARMS[rep]))
                np.testing.assert_allclose(expected,oof.loc[oof.fold_id.eq(fold),model],rtol=1e-10,atol=1e-12)

    def test_zero_and_historical_mean_use_only_effective_training_targets(self):
        keyed=self.data.set_index('date');oof=self.result.tables['oof_predictions']
        for fold,(train,valid) in self.members.items():
            block=oof.loc[oof.fold_id.eq(fold)]
            np.testing.assert_array_equal(block.zero_return,np.zeros(len(valid)))
            np.testing.assert_array_equal(block.historical_mean,np.full(len(valid),keyed.loc[train,w.PRIMARY_TARGET].mean()))

    def test_baselines_do_not_fit_an_estimator(self):
        with patch.object(r,'fit_arm',side_effect=AssertionError('must not fit')):
            predictions=w.baseline_predictions(pd.Series([.1,-.2,.3]),2)
        np.testing.assert_array_equal(predictions['zero_return'],[0.,0.])
        self.assertAlmostEqual(predictions['historical_mean'][0],(.1-.2+.3)/3)

    def test_OOF_Date_model_uniqueness_and_complete_coverage(self):
        oof=self.result.tables['oof_predictions']
        self.assertFalse(oof.date.duplicated().any())
        self.assertEqual(len(oof),sum(len(valid) for _,valid in self.members.values()))
        self.assertTrue(np.isfinite(oof.loc[:,w.MODELS].to_numpy()).all())
        self.assertTrue(oof.date.le(w.DEVELOPMENT_END).all())

    def test_model_parameter_rows_include_medians_scalers_coefficients(self):
        rows=self.result.tables['model_parameters']
        self.assertTrue({'training_median','training_mean_after_imputation','scaler_scale','coefficient',
                         'intercept','training_coverage','used','feature_name','feature_position'}<=set(rows))
        self.assertFalse(rows.feature_name.eq('volume').any())
        self.assertTrue(rows.fit_target.eq(w.PRIMARY_TARGET).all())

    def test_primary_target_only_in_all_metrics_and_fits(self):
        self.assertTrue(self.result.tables['fold_metrics'].target.eq('forward_return_5d').all())
        self.assertTrue(self.result.tables['model_parameters'].fit_target.eq('forward_return_5d').all())

    def test_extrapolation_z_comes_from_training_scaler(self):
        fold='2021Q1';name='original_ols';fitted=self.result.fits[fold,name]
        valid=self.members[fold][1];x=r.representation(self.data.set_index('date').loc[valid],'control')
        pp=fitted.preprocessing;filled=x.loc[:,pp.features].fillna(pp.medians)
        z=(filled.to_numpy()-pp.scaler.mean_)/pp.scaler.scale_
        rows=self.result.tables['extrapolation_audit'].loc[
            lambda f:f.fold_id.eq(fold)&f.representation.eq('original')].set_index('feature')
        for i,feature in enumerate(pp.features):
            self.assertAlmostEqual(rows.loc[feature,'max_abs_validation_z'],np.abs(z[:,i]).max(),places=12)
            self.assertAlmostEqual(rows.loc[feature,'fraction_abs_z_gt_5'],(np.abs(z[:,i])>5).mean(),places=12)

    def test_range_endpoints_include_equality_and_exclude_missing_denominator(self):
        train=pd.DataFrame({'simple_return':[-1.,1.]})
        # Use a real registered preprocessor with one retained feature.
        full=pd.DataFrame(0.,index=['a','b'],columns=r.CONTROL);full.simple_return=[-1.,1.]
        pp=b.fit_preprocessor(full,r.CONTROL)
        valid=pd.DataFrame(0.,index=['c','d','e'],columns=r.CONTROL);valid.simple_return=[-1.,1.,np.nan]
        row=w.extrapolation_rows(full,valid,pp,'fixture','original')[0]
        self.assertEqual(row['fraction_validation_outside_training_min_max'],0.)
        self.assertEqual(row['validation_non_null_count'],2)

    def test_metric_and_actual_label_tampering_fails_validation(self):
        modified=copy.deepcopy(self.result)
        modified.tables['fold_metrics'].loc[0,'MAE']+=.1
        self.assertFalse(w.validate_result(self.data,self.inventory,self.purges,self.members,modified)['valid'])
        modified=copy.deepcopy(self.result)
        modified.tables['oof_predictions'].loc[0,'y_true']+=.1
        self.assertFalse(w.validate_result(self.data,self.inventory,self.purges,self.members,modified)['valid'])

    def test_finite_missing_prediction_and_duplicate_fail_validation(self):
        for change in ('nan','duplicate'):
            modified=copy.deepcopy(self.result)
            if change=='nan':modified.tables['oof_predictions'].loc[0,'relative_ols']=np.nan
            else:modified.tables['oof_predictions']=pd.concat([modified.tables['oof_predictions'],modified.tables['oof_predictions'].iloc[:1]],ignore_index=True)
            with self.subTest(change=change):
                self.assertFalse(w.validate_result(self.data,self.inventory,self.purges,self.members,modified)['valid'])


class PreprocessingIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.inventory,cls.purges,cls.members,cls.features,cls.evidence,cls.result=context()

    def first_fold(self,data):
        inventory=self.inventory.iloc[:1].copy();fold=inventory.fold_id.iloc[0]
        return w.build_walk_forward(data,inventory,{fold:self.members[fold]})

    def assert_fit_states_equal(self,original,changed):
        for key,fitted in original.fits.items():
            other=changed.fits[key]
            np.testing.assert_array_equal(fitted.estimator.coef_,other.estimator.coef_)
            self.assertEqual(fitted.estimator.intercept_,other.estimator.intercept_)
            pd.testing.assert_frame_equal(fitted.preprocessing.usage,other.preprocessing.usage,check_exact=True)
            np.testing.assert_array_equal(fitted.preprocessing.scaler.mean_,other.preprocessing.scaler.mean_)
            np.testing.assert_array_equal(fitted.preprocessing.scaler.scale_,other.preprocessing.scaler.scale_)

    def test_validation_y_cannot_change_same_fold_fit_or_historical_mean(self):
        changed=self.data.copy();dates=self.members['2021Q1'][1]
        changed.loc[changed.date.isin(dates),w.PRIMARY_TARGET]*=100
        new=self.first_fold(changed)
        subset=w.WalkForwardResult({}, {key:value for key,value in self.result.fits.items() if key[0]=='2021Q1'})
        self.assert_fit_states_equal(subset,new)
        old=self.result.tables['oof_predictions'].loc[lambda f:f.fold_id.eq('2021Q1')]
        np.testing.assert_array_equal(new.tables['oof_predictions'].historical_mean,old.historical_mean)
        for model in w.MODELS:
            np.testing.assert_array_equal(new.tables['oof_predictions'][model],old[model])

    def test_validation_X_cannot_change_same_fold_preprocessing_or_fit(self):
        changed=self.data.copy();dates=self.members['2021Q1'][1]
        changed.loc[changed.date.isin(dates),'simple_return']*=1000
        new=self.first_fold(changed)
        subset=w.WalkForwardResult({}, {key:value for key,value in self.result.fits.items() if key[0]=='2021Q1'})
        self.assert_fit_states_equal(subset,new)
        old=self.result.tables['oof_predictions'].loc[lambda f:f.fold_id.eq('2021Q1')]
        self.assertTrue(any(not np.array_equal(new.tables['oof_predictions'][model],old[model]) for model in w.LEARNED))

    def test_future_development_rows_cannot_change_first_fold_predictions(self):
        changed=self.data.copy();mask=changed.date>'2021-03-31'
        changed.loc[mask,w.PRIMARY_TARGET]=10000
        changed.loc[mask,'simple_return']=1e6
        new=self.first_fold(changed)
        old=self.result.tables['oof_predictions'].loc[lambda f:f.fold_id.eq('2021Q1')]
        pd.testing.assert_frame_equal(new.tables['oof_predictions'],old.reset_index(drop=True),check_exact=True)

    def small_x(self):
        x=pd.DataFrame(0.,index=['a','b','c','d'],columns=r.TREATMENT)
        x['close_to_sma_5']=[-.03,.01,np.nan,.08]
        return x

    def test_median_population_scaler_and_zero_variance_from_training_only(self):
        x=self.small_x();fit=r.fit_arm('ols','treatment',x,pd.Series([0.,.1,.2,.3],index=x.index))
        pp=fit.preprocessing;filled=np.array([-.03,.01,.01,.08])
        self.assertEqual(pp.features,('close_to_sma_5',))
        self.assertEqual(pp.medians['close_to_sma_5'],.01)
        np.testing.assert_allclose(pp.scaler.mean_,[filled.mean()],rtol=0,atol=1e-12)
        np.testing.assert_allclose(pp.scaler.scale_,[filled.std(ddof=0)],rtol=0,atol=1e-12)
        valid=x.iloc[:2].copy();valid['close_to_sma_5']=[np.nan,1000.]
        usage=pp.usage.copy();b.transform_features(pp,valid)
        pd.testing.assert_frame_equal(usage,pp.usage,check_exact=True)

    def test_exact_50_percent_training_coverage_is_kept(self):
        x=self.small_x();x['close_to_sma_5']=[-.03,.08,np.nan,np.nan]
        fit=r.fit_arm('ridge','treatment',x,pd.Series([0.,.1,.2,.3],index=x.index))
        self.assertIn('close_to_sma_5',fit.preprocessing.features)
        self.assertEqual(fit.preprocessing.usage.iloc[4].training_coverage,.5)

    def test_under_50_percent_feature_is_dropped_using_training_only(self):
        x=self.small_x();x['close_to_sma_20']=[1.,np.nan,np.nan,np.nan]
        fit=r.fit_arm('ols','treatment',x,pd.Series([0.,.1,.2,.3],index=x.index))
        row=fit.preprocessing.usage.loc[lambda f:f.feature_name.eq('close_to_sma_20')].iloc[0]
        self.assertFalse(row.used);self.assertEqual(row.drop_reason,'low_training_coverage')

    def test_Y_is_not_scaled_and_intercept_is_inherited(self):
        x=self.small_x();y=pd.Series([10.,20.,30.,40.],index=x.index)
        fit=r.fit_arm('ols','treatment',x,y)
        self.assertAlmostEqual(fit.estimator.intercept_,25.,places=12)


class SourceBoundaryTests(unittest.TestCase):
    def test_locked_y_and_gap_payload_are_not_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.csv'
            path.write_text('date,forward_return_5d\n2020-01-02,0.1\n2026-06-15,FORBIDDEN_GAP_LABEL\n2026-07-15,FORBIDDEN_FINAL_LABEL\n2026-09-30,FORBIDDEN_TAIL\n')
            data=w.read_selected(path,('date','forward_return_5d'))
            self.assertEqual(data.date.tolist(),['2020-01-02'])
            self.assertEqual(b._target(data.forward_return_5d).tolist(),[.1])

    def test_forbidden_extra_columns_fail_matrix_gate(self):
        data=context()[0]
        for field in ('volume','turnover','revenue','q_gross_margin','forward_return_1d','forward_return_20d'):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Market-only'):
                w.check_matrix(data.assign(**{field:1.}))

    def test_gap_Final_Test_tail_dates_fail_matrix_gate(self):
        data=context()[0]
        for day in ('2026-06-15','2026-07-15','2026-09-30'):
            changed=data.copy();changed.loc[changed.index[-1],'date']=day
            with self.subTest(day=day),self.assertRaisesRegex(ValueError,'forbidden'):
                w.check_matrix(changed)

    def test_Volume_dependency_is_checked_not_just_feature_names(self):
        dependencies=h.read_csv(w.authority.OUTPUT_DIR/(w.authority.PREFIX+'research_dependency_manifest.csv'))
        changed=dependencies.copy();mask=changed.research_object.eq('simple_return')&changed.source_field.eq('volume')
        changed.loc[mask,'used_indirectly']=True
        with self.assertRaisesRegex(ValueError,'Volume'):
            w.feature_manifest(changed)

    def test_missing_Volume_dependency_proof_fails_closed(self):
        dependencies=h.read_csv(w.authority.OUTPUT_DIR/(w.authority.PREFIX+'research_dependency_manifest.csv'))
        changed=dependencies.loc[~(dependencies.research_object.eq('simple_return')&dependencies.source_field.eq('volume'))]
        with self.assertRaisesRegex(ValueError,'Volume'):
            w.feature_manifest(changed)

    def test_no_Volume_in_fitted_or_scaler_inputs(self):
        result=context()[-1]
        for (_,model),fit in result.fits.items():
            rep=w.MODEL_SPECS[model][0];allowed=r.SPECS[w.ARMS[rep]]
            self.assertTrue(set(fit.preprocessing.features)<=set(allowed))
            self.assertNotIn('volume',fit.preprocessing.scaler.feature_names_in_)

    def test_final_lock_audit_passes_without_predictions_or_metrics(self):
        data,inventory,_,_,_,evidence,result=context()
        audit=w.final_lock_audit(data,inventory,result,evidence)
        self.assertTrue(audit['valid']);self.assertTrue(audit['final_test_locked'])
        self.assertFalse(audit['final_test_predictions_generated']);self.assertFalse(audit['final_test_metrics_computed'])
        self.assertFalse(audit['final_training_pool_used'])
        self.assertEqual(audit['gap_feature_observations_used'],0)

    def test_frozen_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'artifact.json';w.freeze(path,'{"x":1}\n');w.freeze(path,'{"x":1}\n')
            with self.assertRaisesRegex(ValueError,'Frozen'):
                w.freeze(path,'{"x":2}\n')
            self.assertEqual(path.read_text(),'{"x":1}\n')

    def test_protocol_and_fold_registration_are_frozen(self):
        self.assertEqual(h.sha256(w.OUTPUT_DIR/(w.PREFIX+'protocol.json')),w.PROTOCOL_SHA256)
        record=w.authority.load_json(w.OUTPUT_DIR/(w.PREFIX+'fold_registration.json'))
        self.assertTrue(record['folds_frozen_before_modeling'])
        for name,digest in record['fold_sha256'].items():
            self.assertEqual(h.sha256(w.OUTPUT_DIR/(w.PREFIX+name+'.csv')),digest)

    def test_upstream_hashes_unchanged(self):
        self.assertEqual(w.verify_upstream()['mutation_violations'],0)


class ClassificationTests(unittest.TestCase):
    def comparisons(self):
        return pd.DataFrame([{'model':model,'baseline':ref,'mean_delta_MAE':-.01,
                              'median_delta_MAE':-.01,'win_rate':.75}
            for model in ('relative_ols','relative_ridge_alpha1') for ref in w.MODELS[:2]])

    def representation_inputs(self):
        paired=pd.DataFrame([{'scope':'aggregate','family':family,'win_rate':.75}
                             for family in ('ols','ridge_alpha1')])
        mechanism=pd.DataFrame([{'fold_id':str(i),'original_SMA_max_abs_z':10.,'relative_SMA_max_abs_z':3.,
            'original_SMA_outside_fraction':.5,'relative_SMA_outside_fraction':.1,
            'original_SMA_fraction_z_gt_5':.4,'relative_SMA_fraction_z_gt_5':.0,
            'relative_SMA_max_z_better':True,'new_severe_extrapolation':False} for i in range(22)])
        stability=pd.DataFrame([{'fold_id':str(i),'model':model,'fraction_abs_prediction_gt_0_20':0.,
                                 'max_abs_prediction':.1} for i in range(22) for model in w.LEARNED])
        return paired,mechanism,stability

    def test_robust_requires_all_six_for_at_least_one_relative_model(self):
        rows=self.comparisons();rows.loc[rows.model.eq('relative_ridge_alpha1'),'win_rate']=.2
        result=w.predictive_assessment(rows)
        self.assertEqual(result['classification'],'ROBUST_MULTI_YEAR_SIGNAL')
        self.assertTrue(result['model_criteria']['relative_ols']['all_six_passed'])

    def test_NO_robust_when_neither_model_meets_mean_and_majority_gates(self):
        rows=self.comparisons();rows.loc[rows.baseline.eq('zero_return'),'mean_delta_MAE']=.01
        self.assertEqual(w.predictive_assessment(rows)['classification'],'NO_ROBUST_MULTI_YEAR_SIGNAL')

    def test_mixed_when_core_mean_majority_gates_pass_but_medians_fail(self):
        rows=self.comparisons();rows['median_delta_MAE']=.01
        self.assertEqual(w.predictive_assessment(rows)['classification'],'MIXED_OR_REGIME_DEPENDENT_SIGNAL')

    def test_exact_50_percent_is_not_majority(self):
        rows=self.comparisons();rows.win_rate=.5
        self.assertEqual(w.predictive_assessment(rows)['classification'],'NO_ROBUST_MULTI_YEAR_SIGNAL')

    def test_ties_are_not_fold_wins(self):
        result=w.comparison_counts([-1.,0.,1.,0.])
        self.assertEqual((result['wins'],result['losses'],result['ties']),(1,1,2))
        self.assertEqual(result['win_rate'],.25)

    def test_signal_assessment_does_not_use_original_representation_performance(self):
        rows=self.comparisons();extra=rows.copy();extra.model='original_ols';extra.mean_delta_MAE=-100
        self.assertEqual(w.predictive_assessment(pd.concat([rows,extra],ignore_index=True)),w.predictive_assessment(rows))

    def test_broad_representation_requires_both_family_wins_and_mechanism(self):
        result=w.representation_assessment(*self.representation_inputs())
        self.assertEqual(result['classification'],'BROAD_REPRESENTATION_STABILITY_SUPPORT')
        self.assertFalse(result['new_systematic_pathology'])

    def test_one_family_improvement_is_mixed(self):
        paired,mechanism,stability=self.representation_inputs();paired.loc[paired.family.eq('ols'),'win_rate']=.4
        self.assertEqual(w.representation_assessment(paired,mechanism,stability)['classification'],'MIXED_REPRESENTATION_RESULT')

    def test_neither_family_majority_means_no_broad_support(self):
        paired,mechanism,stability=self.representation_inputs();paired.win_rate=.5
        self.assertEqual(w.representation_assessment(paired,mechanism,stability)['classification'],'NO_BROAD_REPRESENTATION_STABILITY_SUPPORT')

    def test_severe_extrapolation_requires_strict_majority(self):
        paired,mechanism,stability=self.representation_inputs();mechanism.loc[:10,'new_severe_extrapolation']=True
        self.assertFalse(w.representation_assessment(paired,mechanism,stability)['new_systematic_pathology'])
        mechanism.loc[11,'new_severe_extrapolation']=True
        self.assertEqual(w.representation_assessment(paired,mechanism,stability)['classification'],'NO_BROAD_REPRESENTATION_STABILITY_SUPPORT')

    def test_new_systematic_prediction_pathology_prevents_broad_claim(self):
        paired,mechanism,stability=self.representation_inputs()
        mask=stability.model.eq('relative_ols')&stability.fold_id.astype(int).lt(12)
        stability.loc[mask,'max_abs_prediction']=.4;stability.loc[mask,'fraction_abs_prediction_gt_0_20']=.5
        result=w.representation_assessment(paired,mechanism,stability)
        self.assertTrue(result['new_systematic_pathology'])
        self.assertEqual(result['classification'],'NO_BROAD_REPRESENTATION_STABILITY_SUPPORT')

    def test_mechanism_failure_can_yield_mixed_even_with_both_MAE_majorities(self):
        paired,mechanism,stability=self.representation_inputs()
        mechanism.relative_SMA_outside_fraction=.6
        self.assertEqual(w.representation_assessment(paired,mechanism,stability)['classification'],'MIXED_REPRESENTATION_RESULT')

    def test_prediction_thresholds_are_strict_and_descriptive(self):
        info=w.prediction_stability(np.array([-.20,-.10,.10,.20,.21]))
        self.assertEqual(info['fraction_abs_prediction_gt_0_20'],1/5)
        self.assertEqual(info['fraction_abs_prediction_gt_0_10'],3/5)

    def test_no_statistical_inference_or_tuning_imports(self):
        tree=ast.parse(Path(w.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(name and ('statsmodels' in name or 'model_selection' in name or 'bootstrap' in name) for name in imports))
        protocol=w.load_protocol()
        self.assertFalse(protocol['statistical_inference_allowed']);self.assertFalse(protocol['hyperparameter_tuning_allowed'])


if __name__=='__main__':
    unittest.main()
