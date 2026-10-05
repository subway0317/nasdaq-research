"""Stage 13.1 causal guards and exact source preservation; fixtures are test-only."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from nasdaq_research import source_reconciliation as r


def trace_fixture(legacy=100,expanded=200,*,raw_equal=False,missing_legacy_raw=False):
    rows=[]
    for side,value in [('legacy',legacy),('expanded',expanded)]:
        for layer in r.LAYERS:
            raw=layer in r.LAYERS[:2]
            available=not (missing_legacy_raw and side=='legacy' and raw)
            row_value=100 if raw and raw_equal else value
            rows.append({'side':side,'layer':layer,'value':row_value if available else None,
                'available':available,'source_file':f'{side}/{layer}.json','source_hash':f'{side}-{layer}',
                'date':r.PRIMARY_DATE,'field':'volume','notes':'synthetic causal fixture'})
    return pd.DataFrame(rows)


def complete_drift_evidence():
    return r.Evidence(legacy_raw_available=True,expanded_raw_available=True,
        legacy_raw_value=100,expanded_raw_value=200,legacy_processing_faithful=True,
        expanded_processing_faithful=True,legacy_timing_known=True,
        legacy_complete_session_proven=True,expanded_complete_session_proven=True,
        source_session_identity_comparable=True,unresolved_partial_session_possibility=False)


class ClassificationTests(unittest.TestCase):
    def test_exact_proven_processing_bug_category(self):
        e=r.Evidence(processing_bug_proven=True,bug_file='fixture.py',bug_function='normalize',bug_transformation='incorrect integer scale')
        self.assertEqual(r.classify(e),'PIPELINE_PROCESSING_BUG')

    def test_processing_bug_requires_exact_file_function_transformation(self):
        e=r.Evidence(processing_bug_proven=True,bug_file='fixture.py')
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_vendor_drift_requires_complete_provenance(self):
        self.assertEqual(r.classify(complete_drift_evidence()),'VENDOR_SOURCE_VINTAGE_DRIFT')

    def test_vendor_drift_missing_raw_falls_back(self):
        e=replace(complete_drift_evidence(),legacy_raw_available=False)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_vendor_drift_unknown_timing_falls_back(self):
        e=replace(complete_drift_evidence(),legacy_timing_known=False)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_vendor_drift_unresolved_partial_possibility_falls_back(self):
        e=replace(complete_drift_evidence(),unresolved_partial_session_possibility=True)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_vendor_drift_same_source_values_insufficient(self):
        e=replace(complete_drift_evidence(),expanded_raw_value=100)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_vendor_drift_incompatible_source_session_falls_back(self):
        e=replace(complete_drift_evidence(),source_session_identity_comparable=False)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_incomplete_session_positive_reliable_timing(self):
        e=replace(complete_drift_evidence(),legacy_complete_session_proven=False,
                  legacy_incomplete_session_proven=True,unresolved_partial_session_possibility=True)
        self.assertEqual(r.classify(e),'LEGACY_INCOMPLETE_SESSION_SNAPSHOT')

    def test_incomplete_session_no_timestamp_cannot_resolve(self):
        e=replace(complete_drift_evidence(),legacy_complete_session_proven=False,
            legacy_incomplete_session_proven=True,legacy_timing_known=False)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_lower_legacy_count_is_not_timing_evidence(self):
        e=r.Evidence(legacy_raw_available=True,expanded_raw_available=True,legacy_raw_value=100,expanded_raw_value=200)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_missing_available_source_value_cannot_resolve(self):
        e=replace(complete_drift_evidence(),legacy_raw_value=None)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_conflicting_completion_evidence_falls_back(self):
        e=replace(complete_drift_evidence(),legacy_incomplete_session_proven=True)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_unresolved_fallback_deterministic(self):
        self.assertEqual(r.classify(r.Evidence()),r.classify(r.Evidence()))
        self.assertEqual(r.classify(r.Evidence()),'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_unfaithful_source_processing_cannot_be_vendor_drift(self):
        e=replace(complete_drift_evidence(),expanded_processing_faithful=False)
        self.assertEqual(r.classify(e),'UNRESOLVED_SOURCE_DISCREPANCY')


class DivergenceAndPolicyTests(unittest.TestCase):
    def test_same_raw_different_normalized_processing_divergence(self):
        result=r.first_divergence(trace_fixture(raw_equal=True))
        self.assertEqual(result['first_divergence_layer'],'normalized_market')
        self.assertTrue(result['causal_first_divergence_identified'])

    def test_different_raw_source_level_divergence(self):
        result=r.first_divergence(trace_fixture())
        self.assertEqual(result['first_divergence_layer'],'raw_yahoo_chart')
        self.assertTrue(result['causal_first_divergence_identified'])

    def test_missing_raw_reports_observable_only(self):
        result=r.first_divergence(trace_fixture(missing_legacy_raw=True))
        self.assertEqual(result['first_observed_divergence_layer'],'normalized_market')
        self.assertEqual(result['first_divergence_layer'],'NOT_IDENTIFIABLE_WITH_AVAILABLE_EVIDENCE')
        self.assertFalse(result['causal_first_divergence_identified'])

    def test_all_equal_no_discrepancy_invented(self):
        self.assertIsNone(r.first_divergence(trace_fixture(legacy=100,expanded=100))['first_divergence_layer'])

    def test_faithful_propagation_integer_counts(self):
        result=r.propagation(trace_fixture())
        self.assertEqual(result['propagation_consistency_violations'],0)
        self.assertEqual(result['available_lineage_edges_checked'],12)

    def test_processing_mutation_detected(self):
        trace=trace_fixture();trace.loc[(trace.side=='expanded')&(trace.layer=='stage5'),'value']=999
        self.assertGreater(r.propagation(trace)['propagation_consistency_violations'],0)

    def test_missing_raw_does_not_claim_original_source_fidelity(self):
        result=r.propagation(trace_fixture(missing_legacy_raw=True))
        self.assertFalse(result['sides']['legacy']['original_source_to_normalized_verified'])
        self.assertTrue(result['sides']['legacy']['normalized_to_downstream_verified'])

    def test_four_canonical_mappings_frozen(self):
        for root,status in zip(r.ROOTS,['RESOLVED_PIPELINE_BUG','RESOLVED_LEGACY_INCOMPLETE_SNAPSHOT','RESOLVED_VENDOR_VINTAGE_DRIFT','UNRESOLVED_BLOCKER']):
            with self.subTest(root=root):
                policy=r.canonical_policy(root,corrected_and_revalidated=True)
                self.assertEqual(policy['completion_status'],status)
                self.assertEqual(policy['expanded_dataset_approved_for_downstream_research'],root!=r.ROOTS[3])
                self.assertEqual(policy['stage14_modeling_eligible'],root!=r.ROOTS[3])
                self.assertEqual(policy['Stage13_original_overlap_violations'],5)
                self.assertEqual(policy['Stage13_historical_status'],'INCOMPLETE_OR_BLOCKED')

    def test_unfixed_bug_cannot_approve_dataset(self):
        policy=r.canonical_policy(r.ROOTS[0])
        self.assertFalse(policy['expanded_dataset_approved_for_downstream_research'])
        self.assertEqual(policy['completion_status'],'UNRESOLVED_BLOCKER')

    def test_engineering_failure_cannot_approve_resolved_root(self):
        policy=r.canonical_policy(r.ROOTS[2],engineering_passed=False)
        self.assertFalse(policy['stage14_modeling_eligible'])
        self.assertIsNone(policy['future_stage14_plus_market_data_authority'])

    def test_unknown_root_category_rejected(self):
        with self.assertRaises(ValueError):r.canonical_policy('MOST_LIKELY_VOLUME_DRIFT')

    def test_current_value_not_causal_classifier_input(self):
        # Missing legacy evidence remains unresolved even if a current value matches exactly.
        e=r.Evidence(expanded_raw_available=True,expanded_raw_value=r.EXPANDED_VALUE,
                     expanded_processing_faithful=True,expanded_complete_session_proven=True)
        self.assertEqual(r.classify(e),r.ROOTS[3])


class ExactSourceAndIsolationTests(unittest.TestCase):
    def test_volume_exact_large_integer_no_float_roundoff(self):
        self.assertEqual(r.integer_volume('9007199254740993'),9007199254740993)
        self.assertNotEqual(r.integer_volume(r.LEGACY_VALUE),r.integer_volume(r.EXPANDED_VALUE))

    def test_fractional_nonfinite_negative_volume_rejected(self):
        for value in ['121269300.1','NaN','Infinity','-1',True,None]:
            with self.subTest(value=value),self.assertRaises(ValueError):r.integer_volume(value)

    def test_final_test_target_and_performance_mutations_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labeled.csv'
            data=pd.DataFrame({'date':['2026-09-23',r.PRIMARY_DATE],'volume':[999,r.LEGACY_VALUE],
                'forward_return_5d':[1e100,-99],'model_prediction':[-1e100,1]})
            data.to_csv(path,index=False);a=r.primary_csv_volume(path)
            data[['forward_return_5d','model_prediction']]=float('nan');data.loc[0,'volume']=0
            data.to_csv(path,index=False);b=r.primary_csv_volume(path)
            self.assertEqual(a,b);self.assertEqual(a,r.LEGACY_VALUE)

    def test_duplicate_primary_date_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'duplicate.csv'
            pd.DataFrame({'date':[r.PRIMARY_DATE]*2,'volume':[1,2]}).to_csv(path,index=False)
            with self.assertRaises(ValueError):r.primary_csv_volume(path)

    def test_bar_timestamp_not_retrieval_timestamp(self):
        local=json.loads(r.output_path('local_evidence').read_text())
        timezone=local['timezone_audit']
        self.assertEqual(timezone['primary_bar_US_Eastern'],'2026-09-30T09:30:00-04:00')
        self.assertFalse(timezone['bar_timestamp_is_retrieval_time'])
        self.assertFalse(timezone['vendor_quote_timestamp_is_retrieval_time'])
        self.assertEqual(timezone['issues'],0)

    def test_legacy_timing_and_version_not_guessed_from_current_environment(self):
        data=pd.read_csv(r.output_path('acquisition_metadata','.csv'))
        for key in ['retrieval_timestamp','query_start','query_end','yfinance_version']:
            record=data.loc[data.source_side.eq('legacy') & data.metadata_field.eq(key)].iloc[0]
            self.assertEqual(record.evidence_status,'NOT_AVAILABLE')
            self.assertTrue(pd.isna(record.value))

    def test_original_raw_absence_explicit_not_fabricated(self):
        trace=pd.read_csv(r.output_path('processing_trace','.csv'))
        missing=trace.loc[trace.side.eq('legacy') & trace.layer.isin(r.LAYERS[:2])]
        self.assertEqual(len(missing),2)
        self.assertFalse(missing.available.any());self.assertTrue(missing.value.isna().all())

    def test_protocol_root_rules_unchanged(self):
        self.assertEqual(json.loads(r.output_path('protocol').read_text()),r.protocol())
        self.assertIn('Absence of incomplete-session evidence is not evidence of completion',r.protocol()['root_cause_decision_rules'][r.ROOTS[3]])

    def test_stage1_12_and_stage13_hashes_preserved(self):
        check=r.verify_protected()
        self.assertEqual(check['stage1_12_mutation_violations'],0)
        self.assertEqual(check['stage13_mutation_violations'],0)


class RealReconciliationTests(unittest.TestCase):
    def test_real_trace_completeness_and_exact_sources(self):
        trace=pd.DataFrame(r.local_bundle()['processing_trace'])
        self.assertTrue(all(r.validate_trace(trace).values()))
        self.assertEqual(r.propagation(trace)['propagation_consistency_violations'],0)

    def test_real_normalization_replay_exact(self):
        local=json.loads(r.output_path('local_evidence').read_text())
        replay=local['normalized_replay']
        self.assertEqual(replay['primary_original_yfinance_volume'],r.EXPANDED_VALUE)
        self.assertEqual(replay['normalized_replay_volume'],r.EXPANDED_VALUE)
        self.assertEqual(replay['violations'],0)

    def test_current_raw_table_normalized_match(self):
        external=r.external_evidence()
        self.assertTrue(external['available']);self.assertTrue(all(external['checks'].values()))
        self.assertEqual(external['primary_volume'],r.EXPANDED_VALUE)

    def test_frozen_external_reuse_does_not_network(self):
        with patch.object(r,'download_stock_history',side_effect=AssertionError('No network during frozen reuse')):
            a=r.acquire_current();b=r.acquire_current()
        self.assertEqual(a,b)

    def test_offline_repeated_artifacts_and_unresolved_approval(self):
        with patch.object(r,'download_stock_history',side_effect=AssertionError('Formal runs offline')):
            a=r.run_pipeline()
            hashes1={p.name:r.sha256(p) for p in sorted(r.OUT.glob('*')) if p.is_file() and p.suffix in ('.json','.csv')}
            b=r.run_pipeline()
            hashes2={p.name:r.sha256(p) for p in sorted(r.OUT.glob('*')) if p.is_file() and p.suffix in ('.json','.csv')}
        self.assertEqual(a,b);self.assertEqual(hashes1,hashes2)
        self.assertEqual(a['root_cause'],'UNRESOLVED_SOURCE_DISCREPANCY')
        self.assertEqual(a['completion_status'],'UNRESOLVED_BLOCKER')
        self.assertFalse(a['stage14_modeling_eligible'])
        self.assertFalse(a['expanded_dataset_approved_for_downstream_research'])

    def test_stage13_historical_blocker_not_rewritten(self):
        original=json.loads((r.STAGE13/'NVDA_stage13_summary.json').read_text())
        self.assertEqual(original['completion_status'],'INCOMPLETE_OR_BLOCKED')
        self.assertEqual(original['overlap']['unexpected_violations'],5)

    def test_locked_final_test_flags(self):
        summary=r.run_pipeline()
        self.assertTrue(summary['final_test_locked'])
        for key in ['final_test_predictions_generated','final_test_metrics_computed','final_training_pool_used',
                    'stage14_designed','stage14_executed','new_model_fit','new_CV']:
            self.assertFalse(summary[key])


if __name__=='__main__':
    unittest.main()
