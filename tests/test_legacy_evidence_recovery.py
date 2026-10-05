"""Pure fixtures for evidence semantics and read-only audit guards."""

import json
from pathlib import Path
import unittest

from nasdaq_research import legacy_evidence_recovery as recovery


def encoded(value):
    return json.dumps(value).encode()


def chart(symbol='NVDA', epoch=1790775000, volume=121269300):
    return encoded({'chart': {'result': [{'meta': {'symbol': symbol,
        'exchangeTimezoneName': 'America/New_York'}, 'timestamp': [epoch],
        'indicators': {'quote': [{'volume': [volume], 'open': [1], 'close': [2]}]}}]}})


def eligible(**changes):
    row = dict(category='RAW_VENDOR_EVIDENCE', strength='STRONG', genuinely_new=True,
               legacy_provenance_verified=True, material_primary_information=True)
    return {**row, **changes}


class EvidenceClassificationTests(unittest.TestCase):
    def test_original_yahoo_primary_session(self):
        result = recovery.classify_candidate('raw/NVDA_chart.json', chart())
        self.assertEqual(result['category'], 'RAW_VENDOR_EVIDENCE')
        self.assertEqual(result['primary_volume'], 121269300)
        self.assertTrue(result['contains_2026_09_30'])

    def test_wrong_symbol_cannot_be_nvda_source(self):
        result = recovery.classify_candidate('raw/NVDA_chart.json', chart(symbol='AAPL'))
        self.assertEqual(result['category'], 'UNKNOWN')

    def test_wrong_session_cannot_be_primary_raw(self):
        result = recovery.classify_candidate('NVDA_chart.json', chart(epoch=1790688600))
        self.assertEqual(result['category'], 'UNKNOWN')

    def test_original_yfinance_multiheader(self):
        data=b'Price,Close,Adj Close,Volume\nTicker,NVDA,NVDA,NVDA\nDate,,,\n2026-09-30,1,1,121269300\n'
        result=recovery.classify_candidate('NVDA_yfinance.csv',data)
        self.assertEqual(result['category'],'RAW_LIBRARY_EVIDENCE')
        self.assertEqual(result['primary_volume'],'121269300')

    def test_project_normalized_raw_directory_is_level_three(self):
        data=b'date,open,high,low,close,volume\n2026-09-30,1,2,1,2,121269300\n'
        result=recovery.classify_candidate('data/raw/NVDA.csv',data)
        self.assertEqual(result['category'],'NORMALIZED_DATA')
        self.assertEqual(result['strength'],'WEAK')

    def test_downstream_labeled_is_not_original_library(self):
        data=b'date,open,high,low,close,volume,forward_return_1d\n2026-09-30,1,2,1,2,121269300,3\n'
        self.assertEqual(recovery.classify_candidate('data/research/NVDA_labeled.csv',data)['category'],
                         'DOWNSTREAM_PROCESSED_DATA')

    def test_actual_request_manifest_with_timezone_timestamp(self):
        data=encoded({'ticker':'NVDA','query':{'start':'2026-09-30','end':'2026-10-01'},
                      'retrieval_timestamp':'2026-09-30T19:58:00+00:00'})
        result=recovery.classify_candidate('NVDA_acquisition.json',data)
        self.assertEqual(result['category'],'RUNTIME_ACQUISITION_METADATA')
        self.assertEqual(result['strength'],'STRONG')
        self.assertTrue(result['reliable_acquisition_timestamp'])

    def test_actual_query_can_be_runtime_without_retrieval_time(self):
        result=recovery.classify_candidate('NVDA_acquisition.json',encoded({
            'ticker':'NVDA','query':{'start':'2026-09-30','end':'2026-10-01'}}))
        self.assertEqual(result['category'],'RUNTIME_ACQUISITION_METADATA')
        self.assertEqual(result['strength'],'MODERATE')
        self.assertFalse(result['reliable_acquisition_timestamp'])

    def test_mtime_is_not_runtime_provenance(self):
        result=recovery.classify_candidate('NVDA_metadata.json',encoded({
            'ticker':'NVDA','mtime':'2026-09-30T19:58:00+00:00','volume':121269300}))
        self.assertEqual(result['category'],'UNKNOWN')
        self.assertFalse(result['reliable_acquisition_timestamp'])

    def test_quote_time_is_not_retrieval_timestamp(self):
        result=recovery.classify_candidate('NVDA_metadata.json',encoded({
            'ticker':'NVDA','query':{'start':'2026-09-30'},'regularMarketTime':1790775000}))
        self.assertFalse(result['reliable_acquisition_timestamp'])

    def test_naive_timestamp_not_certified_reliable(self):
        result=recovery.classify_candidate('NVDA_acquisition.json',encoded({
            'ticker':'NVDA','query':{'start':'2026-09-30'},'retrieval_timestamp':'2026-09-30T15:58:00'}))
        self.assertFalse(result['reliable_acquisition_timestamp'])

    def test_configured_defaults_not_runtime(self):
        result=recovery.classify_candidate('NVDA_defaults.json',encoded({
            'ticker':'NVDA','query':{'start':'2026-09-30'},'configured_defaults':True}))
        self.assertEqual(result['category'],'UNKNOWN')

    def test_generic_financial_start_end_not_acquisition(self):
        result=recovery.classify_candidate('NVDA_facts.json',encoded({
            'ticker':'NVDA','start':'2026-01-01','end':'2026-09-30','revenue':121269300}))
        self.assertEqual(result['category'],'UNKNOWN')

    def test_implementation_cannot_be_raw_or_runtime(self):
        data=b'query = {"ticker": "NVDA", "start": "2026-09-30"}\n# 121269300\nyf.download()\n'
        self.assertEqual(recovery.classify_candidate('src/data.py',data)['category'],
                         'IMPLEMENTATION_EVIDENCE')
        self.assertEqual(recovery.classify_candidate('unmapped_blob_abc',b'import yfinance\n# NVDA 2026-09-30 121269300\n')['category'],
                         'IMPLEMENTATION_EVIDENCE')

    def test_fixture_raw_payload_is_not_historical_evidence(self):
        self.assertEqual(recovery.classify_candidate('tests/NVDA_chart.json',chart())['category'],
                         'TEST_FIXTURE')
        self.assertEqual(recovery.classify_candidate('unmapped_blob_abc',b'import unittest\n# NVDA 121269300\n')['category'],
                         'TEST_FIXTURE')

    def test_report_hit_not_original_evidence(self):
        self.assertEqual(recovery.classify_candidate('docs/NVDA_report.md',chart())['category'],
                         'REPORT_OR_DOCUMENTATION')
        self.assertEqual(recovery.classify_candidate('unmapped_blob_abc',b'# Report\nNVDA 2026-09-30 121269300\n')['category'],
                         'REPORT_OR_DOCUMENTATION')

    def test_binary_never_dumped_or_classified_as_text(self):
        self.assertIsNone(recovery.text_payload(b'\x00NVDA121269300'))

    def test_unmapped_normalized_blob_not_raw(self):
        data=b'date,open,high,low,close,volume\n2026-09-30,1,2,1,2,121269300\n'
        self.assertEqual(recovery.classify_candidate('unmapped_blob_abc',data)['category'],'NORMALIZED_DATA')


class RecoveryDecisionTests(unittest.TestCase):
    def test_strong_new_authenticated_primary_raw_qualifies(self):
        self.assertTrue(recovery.recovery_qualified(eligible()))

    def test_moderate_material_raw_library_qualifies(self):
        self.assertTrue(recovery.recovery_qualified(eligible(category='RAW_LIBRARY_EVIDENCE',strength='MODERATE')))

    def test_weak_never_qualifies(self):
        self.assertFalse(recovery.recovery_qualified(eligible(strength='WEAK')))

    def test_implementation_cannot_count_as_recovery(self):
        self.assertFalse(recovery.recovery_qualified(eligible(category='IMPLEMENTATION_EVIDENCE')))

    def test_already_known_context_cannot_count(self):
        self.assertFalse(recovery.recovery_qualified(eligible(genuinely_new=False)))

    def test_unmapped_blob_without_legacy_provenance_not_recovered(self):
        self.assertFalse(recovery.recovery_qualified(eligible(legacy_provenance_verified=False)))

    def test_unrelated_raw_not_material_primary_information(self):
        self.assertFalse(recovery.recovery_qualified(eligible(material_primary_information=False)))

    def test_completion_exact_two_categories(self):
        self.assertEqual(recovery.completion([]),'LEGACY_EVIDENCE_NOT_RECOVERABLE')
        self.assertEqual(recovery.completion([eligible()]),'LEGACY_EVIDENCE_RECOVERED')


class ScopeAndGitParsingTests(unittest.TestCase):
    def test_git_tree_paths_preserve_spaces_and_tabs(self):
        oid='a'*40
        payload=f'100644 blob {oid}\tdata/raw/NVDA old\tfile.csv\0'.encode()
        self.assertEqual(recovery.parse_ls_tree(payload)[0]['original_path'],'data/raw/NVDA old\tfile.csv')

    def test_fsck_object_types_and_sizes_inventory_input(self):
        text='\n'.join(f'unreachable {kind} '+str(index)*40 for index,kind in enumerate(('commit','tree','blob'),1))
        rows=recovery.parse_fsck(text)
        self.assertEqual([row['object_type'] for row in rows],['commit','tree','blob'])
        self.assertEqual(recovery.parse_fsck('notice: HEAD points to main'),[])

    def test_fsck_checkpoint_excludes_post_registration_objects(self):
        old='a'*40
        new='b'*40
        payload=f'unreachable blob {new}\nunreachable tree {old}\n'.encode()
        self.assertEqual(recovery.checkpoint_fsck_payload(payload,{old}),f'unreachable tree {old}\n'.encode())

    def test_missing_canonical_empty_tree_is_not_source_payload(self):
        self.assertEqual(recovery.hashlib.sha1(b'tree 0\0').hexdigest(),'4b825dc642cb6eb9a060e54bf8d69288fbee4904')

    def test_reflog_time_not_retrieval_time(self):
        rows=recovery.parse_reflog('a'*40+'\x1fHEAD@{2026-09-30 12:00:00 -0400}\x1fcommit: example\n')
        self.assertEqual(rows[0]['time_evidence_role'],'REF_MOVEMENT_ONLY_NOT_ACQUISITION_TIME')

    def test_unrelated_project_content_forbidden(self):
        base=Path('/home/test/projects')
        self.assertFalse(recovery.related_project_candidate(base/'unrelated/customer.json',base))
        self.assertTrue(recovery.related_project_candidate(base/'old-nasdaq/data/NVDA.csv',base))
        self.assertTrue(recovery.related_project_candidate(base/'other/NVDA_acquisition.json',base))

    def test_outside_projects_boundary_forbidden(self):
        self.assertFalse(recovery.related_project_candidate(Path('/home/test/NVDA.csv'),Path('/home/test/projects')))

    def test_registered_depth_limit_enforced(self):
        base=Path('/home/test/projects')
        self.assertFalse(recovery.related_project_candidate(base/'nasdaq/a/b/c/d/e/f/g/NVDA.csv',base))

    def test_known_stage13_context_excluded(self):
        protocol={'known_context_exclusions':['data/research/historical_expansion/stage13_nvda/']}
        self.assertTrue(recovery.context_path('data/research/historical_expansion/stage13_nvda/raw/NVDA.json',protocol))
        self.assertFalse(recovery.context_path('data/raw/NVDA.csv',protocol))

    def test_performance_payload_paths_not_inspected(self):
        self.assertTrue(recovery.performance_path('data/research/baseline_models/NVDA_predictions.csv'))
        self.assertFalse(recovery.performance_path('data/raw/NVDA.csv'))

    def test_mutating_git_commands_rejected_without_subprocess(self):
        audit=recovery.Audit.__new__(recovery.Audit)
        for command in ('reset','restore','clean','gc','prune','repack','checkout','add','commit','push','tag'):
            with self.subTest(command=command),self.assertRaises(ValueError):
                audit.git(command)
        with self.assertRaises(ValueError):
            audit.git('fsck','--lost-found')
        with self.assertRaises(ValueError):
            audit.git('reflog','expire','--all')

    def test_search_order_cannot_change(self):
        audit=recovery.Audit.__new__(recovery.Audit)
        audit.protocol={'search_order':['reachable_git_history','deleted_reachable_files']}
        audit.steps=[]
        with self.assertRaises(ValueError):
            audit.step('deleted_reachable_files')
        audit.step('reachable_git_history')
        audit.step('deleted_reachable_files')
        self.assertEqual(audit.steps,audit.protocol['search_order'])

    def test_serialization_contains_no_execution_timestamp(self):
        value={'classification':'LEGACY_EVIDENCE_NOT_RECOVERABLE','objects':0}
        first=recovery.json_bytes(value)
        self.assertEqual(first,recovery.json_bytes(json.loads(first)))


if __name__=='__main__':
    unittest.main()
