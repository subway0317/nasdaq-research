"""Stage 13 data-only tests. Synthetic prices/filings are adversarial fixtures only."""

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import historical_expansion as h
from nasdaq_research.data import download_stock_history
from nasdaq_research.features import build_features
from nasdaq_research.targets import build_targets
from test_research_features import observation


def fixture():
    dates=pd.bdate_range('2018-01-02','2021-01-08').difference(pd.to_datetime(['2020-01-01']))
    prices=20+np.arange(len(dates))*.01+np.sin(np.arange(len(dates))*.3)*.05
    market=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'open':prices-.03,
                         'high':prices+.1,'low':prices-.1,'close':prices,'volume':1000})
    facts=pd.DataFrame([observation(fy=2018,fp='FY',basis='annual'),
                       observation(fy=2019,fp='FY',basis='annual'),
                       observation(fy=2020,fp='FY',basis='annual'),
                       observation(fy=2019,fp='Q2'),observation(fy=2020,fp='Q2'),
                       observation(fy=2021,fp='Q2')])
    return market,facts


class ExpansionConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market,cls.facts=fixture()
        cls.processed,cls.snapshots,cls.daily=h.construct_features(cls.market,cls.facts)

    def test_research_whitelist_has_original_39(self):
        self.assertEqual(len(h.RESEARCH_FEATURES),39)
        self.assertEqual(len(set(h.RESEARCH_FEATURES)),39)
        self.assertEqual(len(h.RELATIVE_FEATURES),3)

    def test_original_SMA_retained_with_relative_representation(self):
        for k in (5,20,60):
            self.assertIn(f'sma_{k}',self.daily)
            np.testing.assert_allclose(self.daily[f'close_to_sma_{k}'],self.daily.close/self.daily[f'sma_{k}']-1,equal_nan=True,rtol=0,atol=1e-12)

    def test_presample_market_matures_first_research_session(self):
        first=h.research_slice(self.daily).iloc[0]
        self.assertEqual(first.date,'2020-01-02')
        self.assertTrue(first[list(h.MARKET_FEATURES)].notna().all())

    def test_presample_fundamental_states_not_removed(self):
        first=h.research_slice(self.daily).iloc[0]
        self.assertEqual(first.q_accession,observation(fy=2020,fp='Q2')['accession'])
        self.assertEqual(first.fy_accession,observation(fy=2019,fp='FY',basis='annual')['accession'])
        self.assertLess(pd.Timestamp(first.q_effective_date),pd.Timestamp('2020-01-01'))
        self.assertFalse(first.q_effective_date_is_sample_truncated)

    def test_no_fill_of_unavailable_feature(self):
        f=self.facts.copy(); f['operating_cash_flow']=np.nan;f['free_cash_flow']=np.nan
        _,_,d=h.construct_features(self.market,f)
        self.assertTrue(d.q_ocf_margin.isna().all())
        self.assertEqual(len(d),len(self.market))

    def test_independent_rolling_oracle(self):
        result=h.market_oracle(self.processed,[0,1,4,19,20,59,60,130,len(self.processed)-1])
        self.assertEqual(result['violations'],0)

    def test_rolling_oracle_rejects_population_volatility(self):
        d=self.processed.copy();i=100
        d.loc[i,'rolling_volatility_60']=d.simple_return.iloc[i-59:i+1].std(ddof=0)
        self.assertGreater(h.market_oracle(d,[i])['violations'],0)

    def test_independent_state_oracle(self):
        positions=[self.daily.index[self.daily.date.eq('2020-01-02')][0],len(self.daily)-1]
        self.assertEqual(h.state_oracle(self.daily,self.facts,positions,self.market.date)['violations'],0)

    def test_independent_growth_oracle(self):
        self.assertEqual(h.growth_oracle(self.snapshots,self.facts)['violations'],0)

    def test_future_SEC_mutation_isolated(self):
        cutoff='2020-01-02';f=self.facts.copy();mask=f.filing_date>cutoff
        for c in h.FINANCIAL_FIELDS:f.loc[mask,c]*=10000
        _,_,d=h.construct_features(self.market,f)
        before=self.daily.date<=cutoff
        self.assertEqual(h.row_equality(self.daily.loc[before],d.loc[before]),0)

    def test_future_market_mutation_isolated(self):
        cutoff='2020-01-02';m=self.market.copy()
        m.loc[m.date>cutoff,['open','high','low','close']]*=12
        _,_,d=h.construct_features(m,self.facts)
        before=self.daily.date<=cutoff
        self.assertEqual(h.row_equality(self.daily.loc[before],d.loc[before]),0)

    def test_target_mutation_cannot_enter_research_X(self):
        labeled=self.daily.copy()
        for c in (*h.RETURN_COLUMNS,'target_entry_date','target_entry_open','target_exit_close_5d'):
            labeled[c]='adversarial future target'
        self.assertEqual(h.row_equality(h.research_x(labeled),h.research_x(self.daily)),0)

    def test_target_oracle_preserves_tail(self):
        d=h.research_slice(self.daily);t=build_targets(d)
        self.assertEqual(len(t),len(d))
        for horizon in h.HORIZONS:self.assertEqual(t[f'forward_return_{horizon}d'].isna().sum(),horizon)
        self.assertEqual(h.target_oracle(d,t,('2026-07-15','2026-09-23'))['violations'],0)

    def test_target_oracle_rejects_shifted_horizon(self):
        d=h.research_slice(self.daily);t=build_targets(d)
        t.loc[0,'target_exit_date_5d']=t.loc[0,'target_exit_date_1d']
        self.assertGreater(h.target_oracle(d,t,('2026-07-15','2026-09-23'))['violations'],0)

    def test_locked_target_values_not_read_by_oracle(self):
        d=h.research_slice(self.daily);t=build_targets(d)
        locked=('2020-01-02','2020-12-31')
        t.loc[t.date.between(*locked),list(h.RETURN_COLUMNS)]=np.inf
        self.assertEqual(h.target_oracle(d,t,locked)['violations'],0)

    def test_legal_prefix_reconstruction(self):
        for cutoff in ('2020-01-02','2020-02-25','2020-08-26','2020-08-27'):
            with self.subTest(date=cutoff):
                _,_,d=h.construct_features(self.market.loc[self.market.date<=cutoff],self.facts.loc[self.facts.filing_date<cutoff])
                self.assertEqual(h.row_equality(self.daily.loc[self.daily.date.eq(cutoff)],d.iloc[[-1]]),0)

    def test_amendment_contents_isolated_until_next_observed_session(self):
        amendment=observation(fy=2020,fp='Q2',filing_date='2020-05-01',accession='amend',form='10-Q/A')
        f=pd.concat([self.facts,pd.DataFrame([amendment])],ignore_index=True)
        _,_,a=h.construct_features(self.market,f)
        changed=f.copy();changed.loc[changed.accession.eq('amend'),'revenue']=99999
        _,_,b=h.construct_features(self.market,changed)
        before=a.date<='2020-05-01'
        self.assertEqual(h.row_equality(a.loc[before],b.loc[before]),0)
        self.assertEqual(a.loc[a.date.eq('2020-05-01'),'q_accession'].iloc[0],observation(fy=2020,fp='Q2')['accession'])
        self.assertEqual(a.loc[a.date.eq('2020-05-04'),'q_accession'].iloc[0],'amend')
        self.assertNotEqual(a.loc[a.date.eq('2020-05-04'),'q_gross_margin'].iloc[0],b.loc[b.date.eq('2020-05-04'),'q_gross_margin'].iloc[0])

    def test_same_day_multiple_accessions_deterministic(self):
        candidate=observation(fy=2020,fp='Q2',accession='zzz-tie',revenue=500)
        f=pd.concat([self.facts,pd.DataFrame([candidate])],ignore_index=True)
        _,_,a=h.construct_features(self.market,f)
        _,_,b=h.construct_features(self.market,f.iloc[::-1])
        self.assertEqual(h.row_equality(a,b),0)
        self.assertEqual(a.loc[a.date.eq('2020-01-02'),'q_accession'].iloc[0],'zzz-tie')

    def test_YTD_does_not_replace_quarterly_state(self):
        candidate=observation(fy=2020,fp='Q3',basis='ytd',period_start='2019-02-01',period_end='2019-10-31',filing_date='2019-11-20',revenue=99999)
        f=pd.concat([self.facts,pd.DataFrame([candidate])],ignore_index=True)
        _,_,a=h.construct_features(self.market,f)
        self.assertEqual(h.row_equality(a,self.daily),0)

    def test_collision_safe_temporary_market_column(self):
        m=self.market.copy();m['_trading_date']='retained source column'
        # build_features deliberately uses registered OHLCV schema; no hidden input can enter X.
        _,_,d=h.construct_features(m,self.facts)
        self.assertEqual(h.row_equality(d,self.daily),0)

    def test_yearly_coverage_not_only_average(self):
        d=h.research_slice(self.daily);t=build_targets(d)
        yearly,features,dates=h.coverage(d,t)
        self.assertEqual(yearly.year.tolist(),list(range(2020,2027)))
        self.assertEqual(len(features),42)
        self.assertEqual(dates['first_mature_60_session_market_date'],'2020-01-02')

    def test_feature_code_ignores_action_metadata(self):
        m=self.market.copy();m['future_split_ratio']=100000
        _,_,d=h.construct_features(m,self.facts)
        self.assertEqual(h.row_equality(d,self.daily),0)

    def test_fixed_absolute_prices_not_claimed_as_traded(self):
        # Retroactive vendor rescaling changes absolute SMA, even though returns/relative SMA cancel.
        m=self.market.copy();m[['open','high','low','close']]/=10
        _,_,d=h.construct_features(m,self.facts)
        np.testing.assert_allclose(d.sma_60,self.daily.sma_60/10,equal_nan=True,rtol=0,atol=1e-12)
        np.testing.assert_allclose(d.close_to_sma_60,self.daily.close_to_sma_60,equal_nan=True,rtol=0,atol=1e-12)
        self.assertIn('not an archive of as-traded',h.contract()['inherited_vintage_limitations'][0])


class AcquisitionAndAuditTests(unittest.TestCase):
    def test_legacy_download_kwargs_unchanged(self):
        source=pd.DataFrame({'Open':[1.],'High':[2.],'Low':[.5],'Close':[1.5],'Volume':[100]},index=pd.DatetimeIndex(['2020-01-02'],name='Date'))
        with patch('yfinance.download',return_value=source) as get:
            download_stock_history('nvda')
        get.assert_called_once_with('NVDA',period='1y',interval='1d',auto_adjust=False,progress=False,threads=False)

    def test_date_range_extension_preserves_adjustment_and_archives_source(self):
        source=pd.DataFrame({'Open':[1.],'High':[2.],'Low':[.5],'Close':[1.5],'Volume':[100],'Adj Close':[1.4]},index=pd.DatetimeIndex(['2020-01-02'],name='Date'))
        with tempfile.TemporaryDirectory() as tmp,patch('yfinance.download',return_value=source) as get:
            path=Path(tmp)/'source.csv';result=download_stock_history('NVDA',start='2019-01-01',end='2020-01-03',source_snapshot_path=path)
            self.assertTrue(path.exists());self.assertEqual(result.close.iloc[0],1.5)
            self.assertFalse(get.call_args.kwargs['auto_adjust']);self.assertTrue(get.call_args.kwargs['keepna'])
            with self.assertRaisesRegex(ValueError,'already exists'):
                download_stock_history('NVDA',source_snapshot_path=path)

    def test_calendar_detects_missing_session(self):
        dates=pd.DataFrame({'date':['2020-01-02','2020-01-06']})
        table,check=h.calendar_audit(dates,'2020-01-01','2020-01-06')
        self.assertEqual(check['violations'],1);self.assertEqual(table.date.iloc[0],'2020-01-03')

    def test_calendar_special_closure_and_newyear_Saturday(self):
        dates=pd.DataFrame({'date':['2021-12-30','2021-12-31','2022-01-03']})
        self.assertEqual(h.calendar_audit(dates,'2021-12-30','2022-01-03')[1]['violations'],0)
        dates=pd.DataFrame({'date':['2025-01-08','2025-01-10']})
        self.assertEqual(h.calendar_audit(dates,'2025-01-08','2025-01-10')[1]['violations'],0)

    def test_protocol_locked_without_results_dependency(self):
        p=h.contract()
        self.assertFalse(p['modeling_allowed']);self.assertFalse(p['new_cv_allowed']);self.assertFalse(p['final_test_evaluation_allowed'])
        self.assertEqual(p['research_start'],'2020-01-01')
        self.assertEqual(p['market_raw_start'],'2009-08-20')
        self.assertEqual(p['final_test_lock_rules']['start'],'2026-07-15')

    def test_upstream_hash_manifest_passes(self):
        self.assertEqual(h.verify_upstream()['mutation_violations'],0)

    def test_wrong_output_namespace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'namespace'):
            h.register(Path(tmp))

    def test_absent_corporate_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'No archived'):
            h.corporate_actions(Path(tmp))


@unittest.skipUnless((h.OUTPUT_DIR/'raw/NVDA_acquisition.json').exists(),'real frozen Stage 13 raw acquisition required')
class RealHistoricalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=pd.read_csv(h.OUTPUT_DIR/'raw/NVDA_market.csv')
        cls.facts=pd.read_csv(h.DATA_DIR/'fundamentals/processed/NVDA_fundamentals.csv')
        cls.processed,cls.snapshots,cls.daily=h.construct_features(cls.market,cls.facts)
        cls.events,cls.event_meta=h.corporate_actions(h.OUTPUT_DIR/'raw')

    def test_real_split_events_confirmed(self):
        for event in h.SPLITS:
            with self.subTest(date=event['date']):
                row=self.events.loc[self.events.date.eq(event['date']) & self.events.event_type.eq('splits')].iloc[0]
                self.assertEqual(row.split_ratio,event['ratio'])

    def test_real_split_price_SMA_target_consistency(self):
        table,check=h.corporate_audit(self.processed,self.events)
        self.assertEqual(check['violations'],0)
        split_rows=table.loc[table.event_type.eq('splits')]
        self.assertEqual(len(split_rows),2)
        self.assertFalse(split_rows.inverse_split_mechanical_jump.any())

    def test_unadjusted_split_discontinuity_rejected(self):
        changed=self.market.copy();before=changed.date<'2024-06-10'
        changed.loc[before,['open','high','low','close']]*=10
        bad=build_features(changed)
        self.assertGreater(h.corporate_audit(bad,self.events)[1]['violations'],0)

    def test_actual_trading_calendar_complete(self):
        self.assertEqual(h.calendar_audit(self.market,h.RESEARCH_START,h.RESEARCH_END)[1]['violations'],0)

    def test_real_presample_states_and_growth_oracle(self):
        self.assertEqual(h.growth_oracle(self.snapshots,self.facts)['violations'],0)
        daily=h.research_slice(self.daily)
        self.assertTrue(daily[['q_accession','fy_accession','bs_accession']].iloc[0].notna().all())

    def test_real_legal_history_prefix(self):
        for cutoff in ('2020-01-02','2021-07-20','2024-06-10'):
            with self.subTest(date=cutoff):
                _,_,d=h.construct_features(self.market.loc[self.market.date<=cutoff],self.facts.loc[self.facts.filing_date<cutoff])
                self.assertEqual(h.row_equality(self.daily.loc[self.daily.date.eq(cutoff)],d.iloc[[-1]]),0)

    def test_nontruncation_fields_strict_match_or_explicit_failure(self):
        d=h.research_slice(self.daily);targets=build_targets(d)
        old_market=pd.read_csv(h.DATA_DIR/'raw/NVDA.csv');_,_,old=h.construct_features(old_market,self.facts)
        research=h.legacy_stage5_frame(self.processed,self.facts)
        audit,expected,unexpected,summary=h.overlap_audit(self.processed,research,self.daily,targets,old,('2026-07-15','2026-09-23'),raw_market=self.market)
        self.assertEqual(summary['unexpected_violations'],len(unexpected))
        # Real vendor volume revision is retained as a failure, never relabeled expected.
        self.assertTrue(unexpected.field.eq('volume').all())
        self.assertTrue(unexpected.date.eq('2026-09-30').all())
        self.assertTrue(expected.reason.isin(['registered_market_warmup_truncated_at_legacy_sample_start','legacy_pre_window_filing_effective_date_left_censoring']).all())
        self.assertTrue(expected.evidence.notna().all())
        stable=audit.loc[audit.expected_expansion_differences.eq(0) & ~audit.field.eq('volume')]
        self.assertEqual(stable.unexpected_violations.sum(),0)

    def test_raw_cache_bytes_immutable(self):
        meta=json.loads((h.OUTPUT_DIR/'raw/NVDA_acquisition.json').read_text())
        for name,digest in meta['raw_sha256'].items():
            self.assertEqual(h.sha256(h.OUTPUT_DIR/'raw'/name),digest)


if __name__=='__main__':
    unittest.main()
