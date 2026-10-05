"""Stage 13.2 governance tests; preserved evidence and isolated mutations only."""

import ast
from functools import lru_cache
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import data_authority as a
from nasdaq_research import historical_expansion as h
from nasdaq_research.data import standardize_history
from nasdaq_research.features import build_features


@lru_cache(maxsize=1)
def preserved_context():
    with a.offline_only():
        market=pd.read_csv(h.OUTPUT_DIR/'raw/NVDA_market.csv')
        facts=pd.read_csv(a.DATA_DIR/'fundamentals/processed/NVDA_fundamentals.csv')
        base=a.pipeline_copy(market,facts)
        dependencies,code=a.dependency_evidence(base['daily'])
        mutations=a.volume_mutation_audit(market,facts,base)
        controls=a.positive_controls(market,facts,base)
    return market,facts,base,dependencies,code,mutations,controls


class FormulaEvidenceTests(unittest.TestCase):
    def tracer(self,source):
        tracer=a.FormulaTracer(['result'],('date','open','high','low','close','volume'))
        tracer.visit(ast.parse(source).body)
        return tracer

    def test_aliases_trace_actual_price_field(self):
        t=self.tracer('close = result["close"]\nresult["simple_return"] = close.pct_change(fill_method=None)')
        self.assertEqual(t.graph['simple_return']['raw_fields'],['close'])

    def test_indirect_dependency_propagates(self):
        t=self.tracer('result["r"] = result["close"].pct_change()\nresult["v"] = result["r"].rolling(20).std(ddof=1)')
        self.assertEqual(t.graph['v']['terms'],['r'])
        self.assertEqual(t.graph['v']['raw_fields'],['close'])

    def test_added_Volume_formula_is_detected(self):
        t=self.tracer('result["r"] = result["close"] / result["volume"]')
        self.assertIn('volume',t.graph['r']['raw_fields'])

    def test_hidden_Volume_mask_is_detected(self):
        t=self.tracer('result["r"] = result["close"].where(result["volume"] > 100)')
        self.assertIn('volume',t.graph['r']['raw_fields'])

    def test_dynamic_unknown_column_fails_closed(self):
        with self.assertRaises(a.DependencyUnknown):
            self.tracer('result["r"] = result[unknown_field]')

    def test_unknown_expression_symbol_fails_closed(self):
        with self.assertRaises(a.DependencyUnknown):
            self.tracer('result["r"] = unknown_series + result["close"]')

    def test_unsupported_conditional_fails_closed(self):
        with self.assertRaises(a.DependencyUnknown):
            self.tracer('result["r"] = result["close"] if condition else result["volume"]')

    def test_loop_column_names_resolve_from_source(self):
        t=self.tracer('for k in (5,20,60):\n    result[f"sma_{k}"] = result["close"].rolling(k).mean()')
        self.assertEqual(set(t.graph),{'sma_5','sma_20','sma_60'})
        self.assertTrue(all(n['raw_fields']==['close'] for n in t.graph.values()))


class GovernanceDecisionTests(unittest.TestCase):
    def decision(self,**changes):
        args=dict(governance_passed=True,materialities=[a.NONMATERIAL],registry_complete=True,
                  mutations_passed=True,controls_passed=True,policy_frozen=True)
        args.update(changes)
        return a.approval_decision(**args)

    def test_unresolved_nonmaterial_difference_receives_qualified_status(self):
        self.assertEqual(self.decision(),a.QUALIFIED)

    def test_unqualified_status_requires_no_known_unresolved_difference(self):
        self.assertEqual(self.decision(materialities=[]),a.APPROVED)

    def test_semantic_material_difference_rejects(self):
        self.assertEqual(self.decision(materialities=[a.MATERIAL]),a.REJECTED)

    def test_unknown_materiality_rejects(self):
        self.assertEqual(self.decision(materialities=[a.UNKNOWN]),a.REJECTED)

    def test_unknown_category_rejects(self):
        self.assertEqual(self.decision(materialities=['invented']),a.REJECTED)

    def test_critical_governance_failure_rejects(self):
        self.assertEqual(self.decision(governance_passed=False),a.REJECTED)

    def test_missing_registry_entry_rejects(self):
        self.assertEqual(self.decision(registry_complete=False),a.REJECTED)

    def test_failed_Volume_mutation_rejects(self):
        self.assertEqual(self.decision(mutations_passed=False),a.REJECTED)

    def test_broken_positive_control_rejects(self):
        self.assertEqual(self.decision(controls_passed=False),a.REJECTED)

    def test_unfrozen_future_policy_rejects(self):
        self.assertEqual(self.decision(policy_frozen=False),a.REJECTED)

    def test_downstream_flags_exact_mapping(self):
        for status in (a.APPROVED,a.QUALIFIED,a.REJECTED):
            expected=status!=a.REJECTED
            self.assertEqual(a.approval_flags(status),{
                'expanded_dataset_approved_for_downstream_research':expected,
                'stage14_modeling_eligible':expected})

    def test_unknown_status_cannot_open_downstream_gate(self):
        with self.assertRaises(ValueError):
            a.approval_flags('COMPLETE')


class PreservedVintageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market,cls.facts,cls.base,cls.deps,cls.code,cls.mutations,cls.controls=preserved_context()

    def test_Volume_mutation_does_not_alter_nine_market_features(self):
        self.assertTrue(self.mutations.market_cells_changed.eq(0).all())

    def test_Volume_mutation_does_not_alter_relative_market_features(self):
        self.assertTrue(self.mutations.relative_cells_changed.eq(0).all())

    def test_Volume_mutation_does_not_alter_three_targets(self):
        self.assertTrue(self.mutations.target_cells_changed_outside_lock.eq(0).all())

    def test_Volume_mutation_does_not_alter_PIT_states(self):
        self.assertTrue(self.mutations.PIT_cells_changed.eq(0).all())
        self.assertTrue(self.mutations.snapshot_cells_changed.eq(0).all())
        self.assertGreater(len(a.fundamental_state_columns(self.base['daily'])),100)

    def test_Volume_mutation_does_not_change_row_eligibility(self):
        self.assertTrue(self.mutations.row_eligibility_changes.eq(0).all())

    def test_primary_mutations_include_legacy_value_and_large_change(self):
        primary=self.mutations.loc[self.mutations.date.eq('2026-09-30')]
        self.assertEqual(set(primary.new_value),{121269300,1217322000})
        self.assertTrue(primary.passed.all())

    def test_general_mutations_cover_presample_multiple_years_and_splits(self):
        general=self.mutations.loc[self.mutations.case.eq('historical_times_10_plus_123')]
        self.assertGreaterEqual(len(general),4)
        self.assertIn('2010-06-30',set(general.date))
        self.assertIn('2021-07-20',set(general.date))
        self.assertIn('2024-06-10',set(general.date))
        self.assertIn('2026-06-30',set(general.date))
        self.assertGreaterEqual(len(set(general.date.str[:4])),6)

    def test_whole_Volume_vector_mutation_is_isolated(self):
        vector=self.mutations.loc[self.mutations.case.eq('all_rows_times_10_plus_123')].iloc[0]
        self.assertEqual(vector.mutated_rows,len(self.market))
        self.assertEqual(vector.violations,0)

    def test_OHLC_positive_controls_trigger_registered_dependencies(self):
        self.assertEqual(set(self.controls.field),{'open','high','low','close'})
        self.assertTrue(self.controls.passed.all())
        self.assertTrue(self.controls.market_cells_changed.gt(0).all())
        self.assertTrue(self.controls.loc[self.controls.field.isin(['open','close']),
                                          'target_cells_changed_outside_lock'].gt(0).all())

    def test_High_Low_do_not_change_targets_or_PIT(self):
        self.assertTrue(self.controls.loc[self.controls.field.isin(['high','low']),
                                          'target_cells_changed_outside_lock'].eq(0).all())
        self.assertTrue(self.controls.PIT_cells_changed.eq(0).all())

    def test_actual_daily_range_formula_and_dependencies(self):
        node=self.code['market_graph']['daily_range']
        self.assertEqual(set(node['raw_fields']),{'high','low','close'})
        self.assertIn(' / close',node['expression'])

    def test_rolling_volatility_is_indirect_Close_dependency(self):
        row=self.deps.loc[self.deps.research_object.eq('rolling_volatility_60')
                          & self.deps.source_field.eq('close')].iloc[0]
        self.assertFalse(row.used_directly)
        self.assertTrue(row.used_indirectly)

    def test_target_Open_Close_dependency_comes_from_code(self):
        for target in h.RETURN_COLUMNS:
            self.assertEqual(set(self.code['target_graph'][target]['raw_fields']),{'open','close'})

    def test_dependency_manifest_covers_fundamentals_and_validation_guards(self):
        objects=set(self.deps.research_object)
        self.assertTrue(set(h.RESEARCH_FEATURES)<=objects)
        self.assertTrue(set(h.RELATIVE_FEATURES)<=objects)
        self.assertTrue(set(h.RETURN_COLUMNS)<=objects)
        guards=self.deps.loc[self.deps.dependency_type.eq('preprocessing_validity_guard')
                            & self.deps.source_field.eq('volume')]
        self.assertEqual(len(guards),2)
        self.assertTrue(guards.used_directly.all())
        formulas=self.deps.loc[self.deps.dependency_type.ne('preprocessing_validity_guard')
                              & self.deps.source_field.eq('volume')]
        self.assertFalse((formulas.used_directly|formulas.used_indirectly).any())

    def test_registry_retains_one_discrepancy_and_five_manifestations(self):
        registry,complete=a.known_registry(self.deps,self.mutations,self.controls,self.code['complete'])
        self.assertTrue(complete)
        self.assertEqual(len(registry),1)
        row=registry.iloc[0]
        self.assertEqual(row.downstream_manifestations,5)
        self.assertEqual(row.current_research_materiality,a.NONMATERIAL)
        self.assertEqual(row.root_cause_status,'UNRESOLVED_SOURCE_DISCREPANCY')

    def test_incomplete_dependency_proof_is_unknown(self):
        self.assertEqual(a.classify_materiality('volume',self.deps,True,True,False),a.UNKNOWN)

    def test_observed_mutation_change_is_material(self):
        self.assertEqual(a.classify_materiality('volume',self.deps,False,True,True,observed_changes=True),a.MATERIAL)

    def test_future_added_Volume_dependency_would_be_material(self):
        changed=self.deps.copy()
        mask=changed.research_object.eq('simple_return') & changed.source_field.eq('volume')
        changed.loc[mask,'used_indirectly']=True
        self.assertEqual(a.classify_materiality('volume',changed,True,True,True),a.MATERIAL)

    def test_locked_target_cells_are_excluded_even_when_corrupted(self):
        left=self.base['targets'];right=left.copy()
        mask=right.date.between(*a.LOCKED)
        numeric=[c for c in right if c.startswith(('forward_return_','target_exit_close_')) or c=='target_entry_open']
        right.loc[mask,numeric]=np.inf
        self.assertEqual(a.target_diff(left,right)['cells'],0)
        self.assertEqual(int(mask.sum()),50)

    def test_real_replay_matches_frozen_matrix_and_snapshots(self):
        self.assertEqual(a.exact_diff(self.base['research'],h.read_csv(h.OUTPUT_DIR/'NVDA_stage13_research_matrix.csv'))['cells'],0)
        self.assertEqual(a.exact_diff(self.base['snapshots'],h.read_csv(h.OUTPUT_DIR/'NVDA_stage13_fundamental_snapshots.csv'))['cells'],0)

    def test_missing_Volume_can_change_normalization_eligibility(self):
        # Explicit negative case: nondependency is conditional on legal counts.
        changed=self.market.iloc[:100].copy();changed.loc[changed.index[10],'volume']=np.nan
        self.assertEqual(len(standardize_history(changed,'NVDA')),99)

    def test_negative_Volume_fails_input_validation(self):
        changed=self.market.iloc[:100].copy();changed.loc[changed.index[10],'volume']=-1
        with self.assertRaisesRegex(ValueError,'Volume'):
            build_features(changed)

    def test_upstream_inputs_unchanged_after_all_mutations(self):
        self.assertEqual(a.verify_upstream()['mutation_violations'],0)


class CanonicalContractTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        source=self.root/'source.csv';source.write_text('date,close\n2020-01-02,10\n')
        self.manifest={'canonical_vintage_id':'FIXTURE_V1','approved':True,'approval_status':a.QUALIFIED,
                       'authority_scope':{'market_features':['simple_return'],
                          'relative_market_features':['close_to_sma_5'],'fundamental_features':[],
                          'targets':['forward_return_5d']},
                       'source_sha256':{'source.csv':h.sha256(source)}}
        self.path=self.root/'manifest.json'
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def require(self,vintage='FIXTURE_V1',**kwargs):
        return a.require_canonical_vintage(vintage,manifest_path=self.path,root=self.root,**kwargs)

    def test_explicit_vintage_and_hash_identity_pass(self):
        self.assertTrue(self.require(required_features=['close_to_sma_5'])['approved'])

    def test_missing_explicit_vintage_fails(self):
        with self.assertRaisesRegex(ValueError,'Explicit'):
            self.require('')

    def test_wrong_vintage_fails(self):
        with self.assertRaisesRegex(ValueError,'identity'):
            self.require('FIXTURE_V2')

    def test_missing_manifest_fails_without_refresh(self):
        self.path.unlink()
        with a.offline_only(),self.assertRaises(FileNotFoundError):
            self.require()

    def test_missing_cache_fails_without_refresh(self):
        (self.root/'source.csv').unlink()
        with a.offline_only(),self.assertRaises(FileNotFoundError):
            self.require()

    def test_cache_hash_mismatch_fails_without_refresh(self):
        (self.root/'source.csv').write_text('changed')
        with a.offline_only(),self.assertRaisesRegex(ValueError,'hash mismatch'):
            self.require()

    def test_unapproved_vintage_fails(self):
        self.manifest['approved']=False;self.save()
        with self.assertRaisesRegex(ValueError,'not approved'):
            self.require()

    def test_unknown_approval_status_fails(self):
        self.manifest['approval_status']='COMPLETE';self.save()
        with self.assertRaises(ValueError):
            self.require()

    def test_Volume_feature_requires_new_validation(self):
        with self.assertRaisesRegex(ValueError,'new validation'):
            self.require(required_features=['volume'])

    def test_turnover_or_unknown_feature_requires_new_validation(self):
        for feature in ('turnover','volume_zscore','liquidity_proxy','new_feature'):
            with self.subTest(feature=feature),self.assertRaises(ValueError):
                self.require(required_features=[feature])

    def test_target_outside_registered_scope_fails(self):
        with self.assertRaises(ValueError):
            self.require(required_target='new_target')

    def test_source_path_escape_fails(self):
        self.manifest['source_sha256']={'../outside':'digest'};self.save()
        with self.assertRaisesRegex(ValueError,'escapes'):
            self.require()

    def test_empty_source_inventory_fails(self):
        self.manifest['source_sha256']={};self.save()
        with self.assertRaisesRegex(ValueError,'inventory'):
            self.require()


class PolicyAndSafetyTests(unittest.TestCase):
    def test_preregistered_protocol_is_byte_immutable(self):
        self.assertEqual(h.sha256(a.OUTPUT_DIR/(a.PREFIX+'protocol.json')),a.PROTOCOL_SHA256)

    def test_future_metadata_requires_timestamp_and_library_version(self):
        protocol=a.load_json(a.OUTPUT_DIR/(a.PREFIX+'protocol.json'))
        policy=a.future_policy(protocol)
        self.assertTrue({'retrieval_timestamp_utc','library_version','raw_sha256','normalized_sha256',
                         'endpoint_request_semantics'}<=set(policy['required_fields']))
        self.assertTrue(policy['future_metadata_missing_means_incomplete'])

    def test_new_acquisition_requires_new_immutable_vintage(self):
        policy=a.future_policy(a.load_json(a.OUTPUT_DIR/(a.PREFIX+'protocol.json')))
        for key in ('new_acquisition_new_vintage','used_research_vintage_immutable',
                    'cross_vintage_diff_and_materiality_required','no_cell_patching'):
            self.assertTrue(policy[key])
        self.assertFalse(policy['silent_refresh_allowed'])

    def test_volume_scope_limit_is_explicit(self):
        self.assertFalse(a.LIMITS['volume_field_approved_for_current_modeling'])
        self.assertTrue(a.LIMITS['volume_dependent_future_research_requires_revalidation'])
        self.assertTrue(a.LIMITS['volume_dependent_feature_research_requires_new_field_validation'])

    def test_offline_guard_rejects_network_attempt(self):
        with a.offline_only(),self.assertRaisesRegex(RuntimeError,'offline'):
            socket.create_connection(('example.invalid',443))

    def test_no_model_execution_imports(self):
        tree=ast.parse(Path(a.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(name and ('sklearn' in name or name.endswith(('baseline','market_representation')))
                             for name in imports))
        self.assertFalse(a.SAFETY['new_model_fit'])
        self.assertFalse(a.SCOPE['stage14_design_or_execution_authorized'])


if __name__=='__main__':
    unittest.main()
