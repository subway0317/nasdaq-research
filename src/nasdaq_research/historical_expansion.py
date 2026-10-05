"""Stage 13: frozen-vintage NVDA historical data foundation; no model execution."""

import argparse
from datetime import date, timedelta
import hashlib
from io import StringIO
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from nasdaq_research.alignment import build_research_dataset, validate_research_dataset
from nasdaq_research.config import DATA_DIR, PROJECT_ROOT
from nasdaq_research.data import REQUIRED_COLUMNS, download_stock_history
from nasdaq_research.dates import canonical_dates, date_columns, parse_dates
from nasdaq_research.features import WARMUP_ROWS, build_features
from nasdaq_research.fundamentals import normalize_company_facts, validate_fundamentals
from nasdaq_research.fundamental_mapping import CONCEPT_MAPPING, FINANCIAL_FIELDS
from nasdaq_research.provenance import assert_values_equal
from nasdaq_research.research_features import (
    BALANCE_FIELDS, BALANCE_RATIOS, FEATURE_COLUMNS, GROWTH_FIELDS,
    QUARTERLY_GROWTH_FEATURES, QUARTERLY_RATIO_FEATURES,
    ANNUAL_GROWTH_FEATURES, ANNUAL_RATIO_FEATURES,
    build_feature_matrix, validate_feature_matrix,
)
from nasdaq_research.targets import HORIZONS, PRIMARY_TARGET, RETURN_COLUMNS, build_targets

OUTPUT_DIR = DATA_DIR / 'research/historical_expansion/stage13_nvda'
SEC_RAW = DATA_DIR / 'fundamentals/raw/NVDA_companyfacts.json'
RESEARCH_START, RESEARCH_END = '2020-01-01', '2026-09-30'
MARKET_FEATURES = tuple(WARMUP_ROWS)
RESEARCH_FEATURES = (*MARKET_FEATURES, *FEATURE_COLUMNS)
RELATIVE_FEATURES = tuple(f'close_to_sma_{k}' for k in (5, 20, 60))
DERIVED = (*RESEARCH_FEATURES, *RELATIVE_FEATURES, *RETURN_COLUMNS)
SAFETY = dict(final_test_locked=True, final_test_predictions_generated=False,
              final_test_metrics_computed=False, final_training_pool_used=False,
              new_model_fit=False, new_cv=False, MAE_evaluation=False,
              final_test_evaluation=False, no_final_test_performance_dependency=True)
# Documentary event metadata is an audit input, never a feature input.
SPLITS = (
    {'date': '2021-07-20', 'ratio': 4.0, 'source': 'https://nvidianews.nvidia.com/news/nvidia-announces-four-for-one-stock-split-pending-stockholder-approval-at-annual-meeting-set-for-june-3'},
    {'date': '2024-06-10', 'ratio': 10.0, 'source': 'https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm'},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        return value.strftime('%Y-%m-%d')
    return value


def json_text(value) -> str:
    return json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n'


def write(path: Path, contents: str) -> None:
    """Replace one file atomically; never follow an output symlink."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Output symlinks forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(contents, encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_text(frame: pd.DataFrame) -> str:
    frame = frame.copy()
    for c in date_columns(frame.columns):
        frame[c] = canonical_dates(frame[c], allow_missing=True)
    return frame.to_csv(index=False, lineterminator='\n')


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def contract() -> dict[str, Any]:
    facts, _ = normalize_company_facts(json.loads(SEC_RAW.read_bytes()))
    legacy = json.loads((DATA_DIR / 'research/splits/NVDA_evaluation_protocol.json').read_text())
    return {
        'stage_name': 'Stage 13 — NVDA Historical Evidence-Base Expansion',
        'stage_type': 'DATA / RESEARCH-FOUNDATION STAGE', 'ticker': 'NVDA', 'cik': '0001045810',
        'research_start': RESEARCH_START, 'research_end': RESEARCH_END,
        'raw_history_policy': 'Reuse existing SEC vintage; acquire one Yahoo daily snapshot through research_end; never overwrite legacy raw files.',
        'market_raw_start': min(facts.filing_date.min(), RESEARCH_START),
        'market_raw_end_exclusive': '2026-10-01',
        'market_warmup_policy': {'required_preceding_sessions': max(WARMUP_ROWS.values()),
            'calendar_start_basis': 'Earliest retained SEC filing; supplies observed activation calendar for every retained filing and exceeds 60-session feature maturity. Reject insufficient presample sessions.'},
        'fundamental_pre_sample_policy': 'Retain entire archived Company Facts history, including all pre-2020 filings and prior-FY same-quarter/annual reference candidates; filter no SEC rows at research_start.',
        'sec_raw_sha256': sha256(SEC_RAW),
        'market_feature_definitions': {
            'simple_return': 'close[t]/close[t-1]-1', 'log_return': 'ln(close[t]/close[t-1])',
            'intraday_return': 'close[t]/open[t]-1', 'daily_range': '(high[t]-low[t])/close[t]',
            **{f'sma_{k}': f'trailing {k} observed closes, including t; min_periods={k}' for k in (5,20,60)},
            **{f'rolling_volatility_{k}': f'trailing {k} simple returns; sample std ddof=1; not annualized; min_periods={k}' for k in (20,60)}},
        'market_price_semantics': 'Existing Stage 2 yfinance auto_adjust=False OHLC; Adj Close is not substituted; no repair, re-adjustment or new price transformation.',
        'relative_market_formula': 'close_to_sma_k = close[t]/sma_k[t]-1 for k=5,20,60; original SMA retained',
        'research_feature_whitelist': list(RESEARCH_FEATURES),
        'accounting_concept_mapping': CONCEPT_MAPPING,
        'fundamental_state_semantics': 'Unmodified Stage 4/5/6.1: annual/standalone quarterly/YTD distinct; YTD excluded from default states, no differencing; independent q/fy/bs whole-observation streams; same-FY-quarter prior year / FY-1 YoY frozen at filing; deterministic tie tuple.',
        'PIT_activation_rule': 'effective_date = next observed session strictly after filing_date; amendments affect only their effective date onward; no backward-fill',
        'target_definitions': {f'forward_return_{h}d': f'close[t+{h}]/open[t+1]-1 using observed row positions' for h in HORIZONS},
        'primary_target_identity': PRIMARY_TARGET,
        'target_policy': 'Use only market rows through research_end; retain every research row and all target provenance; natural tail NaNs of 1/5/20 rows.',
        'corporate_action_audit_policy': {
            'documented_splits': SPLITS, 'event_inputs': 'Yahoo chart dividends/splits plus issuer/SEC split confirmations; documentary metadata never enters feature functions',
            'OHLC': 'finite positive prices; low<=open/close<=high; all split-event OHLC ratios in [0.5,2.0]',
            'continuity': 'flag adjacent close ratios outside [0.5,2.0], or ratios closer to 1/split_ratio than 1; no silent repair',
            'SMA': 'independent scalar trailing means at t-1/t/t+1 for every split',
            'targets': 'independent entry/exit formulas for labels crossing splits; flag abs(return)>0.5 near a split; thresholds are audit failure gates, never cleaning rules'},
        'coverage_audit_policy': '2020–2026 annual cell coverage and state identity coverage; per-feature missing/infinite/first-valid dates; duplicates and independent US-equity-session audit; retain missing features',
        'overlap_compatibility_policy': 'Compare all legacy raw/processed/Stage5/Stage6.1/Stage8 fields on common dates. Exact metadata/source and OHLC, derived rtol=0 atol=1e-12. Locked test numerical target payload omitted from overlap; target dates/availability permitted.',
        'expected_expansion_difference_policy': 'Cellwise enumeration only: market warmup fill proven by original positional warmup and independent old-prefix reconstruction; effective_date/basis/truncation/age differences only when same accession was filed before old sample boundary, validated against full observed calendar. No arbitrary missing-value or source-value mismatch exemption.',
        'truncation_audit_policy': 'First research session, first/last session of every year before locked test, split +/- observed sessions, real filing transitions before locked test; full feature row vs market-prefix+filing<session rebuild; independent scalar market/state/YoY/target oracles; targets separately checked outside locked dates.',
        'modeling_allowed': False, 'new_cv_allowed': False, 'final_test_evaluation_allowed': False,
        'final_test_lock_rules': {**SAFETY, 'row_count': 50, **legacy['boundaries']['final_test'],
                                  'no_target_distribution_or_performance_decisions': True},
        'immutability_rules': 'Hash every legacy data artifact and tracked file except declared compatible src/nasdaq_research/data.py acquisition API extension. Also retain original data.py HEAD hash and diff; no old artifact replacement.',
        'reproducibility_rules': 'Two consecutive offline formal runs on frozen raw cache; every formal CSV/JSON byte identical; no runtime timestamps in research artifacts; input hashes and git diff --check required',
        'inherited_vintage_limitations': [
            'Fixed Yahoo split-adjusted price coordinate is inherited; this is not an archive of as-traded historical prices. Absolute SMA cannot be certified invariant to vendor retroactive price rescaling.',
            'SEC current archived Company Facts vintage and retained same-accession concepts are inherited; not a complete historical disclosure-vintage archive. Filing-PIT correctness is tested conditional on this frozen snapshot.'],
    }


def register(output_dir: Path = OUTPUT_DIR) -> dict:
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError('Formal namespace is fixed; use pure functions for test fixtures')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('protocol', contract())]:
        path = output_dir / f'NVDA_stage13_{name}.json'
        text = json_text(value)
        if path.exists() and json.loads(path.read_text()) != clean(value):
            raise ValueError('Stage 13 protocol is locked; cannot revise after registration')
        if not path.exists():
            write(path, text)
    manifest_path = output_dir / 'NVDA_stage13_upstream_sha256.json'
    if not manifest_path.exists():
        tracked = subprocess.check_output(['git','ls-files','-z'], cwd=PROJECT_ROOT).decode().split('\0')
        paths = {PROJECT_ROOT/p for p in tracked if p and (PROJECT_ROOT/p).is_file()}
        paths.update(p for p in DATA_DIR.rglob('*') if p.is_file() and output_dir not in p.parents)
        paths.discard(PROJECT_ROOT/'src/nasdaq_research/data.py')
        write(manifest_path, json_text({str(p.relative_to(PROJECT_ROOT)):sha256(p) for p in sorted(paths) if '__pycache__' not in p.parts}))
    return json.loads((output_dir/'NVDA_stage13_protocol.json').read_text())


def verify_upstream(output_dir: Path = OUTPUT_DIR) -> dict:
    manifest = json.loads((output_dir/'NVDA_stage13_upstream_sha256.json').read_text())
    changes = [p for p,h in manifest.items() if not (PROJECT_ROOT/p).is_file() or sha256(PROJECT_ROOT/p) != h]
    return {'files_checked': len(manifest), 'mutation_violations': len(changes), 'changed_files': changes}


def acquire(output_dir: Path = OUTPUT_DIR) -> dict:
    """Use the established downloader with an archiving transport, one request window."""
    protocol = register(output_dir)
    raw_dir = output_dir/'raw'
    raw_dir.mkdir(exist_ok=True)
    sec_copy = raw_dir/'NVDA_companyfacts.json'
    if sec_copy.exists() and sha256(sec_copy) != protocol['sec_raw_sha256']:
        raise ValueError('Cached SEC vintage changed')
    if not sec_copy.exists():
        sec_copy.write_bytes(SEC_RAW.read_bytes())
    market_path = raw_dir/'NVDA_market.csv'
    source_path = raw_dir/'NVDA_yahoo_source.csv'
    meta_path = raw_dir/'NVDA_acquisition.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for name,h in meta['raw_sha256'].items():
            if sha256(raw_dir/name) != h:
                raise ValueError('Frozen acquisition cache hash mismatch')
        return meta
    if market_path.exists() or source_path.exists():
        raise ValueError('Unfinished acquisition cache requires explicit inspection; never overwrite it')
    import yfinance as yf
    from curl_cffi.requests import Session
    # Vendor cookie/tz databases are runtime-only, outside authoritative outputs.
    yf.set_tz_cache_location('/tmp/nasdaq-stage13-yfinance-cache')
    class ArchiveSession(Session):
        def request(self, method, url, *args, **kwargs):
            response = super().request(method, url, *args, **kwargs)
            if '/v8/finance/chart/NVDA' in str(url) and response.status_code == 200:
                body = response.content
                path = raw_dir/f'NVDA_yahoo_chart_{hashlib.sha256(body).hexdigest()}.json'
                if not path.exists():
                    path.write_bytes(body)
            return response
    with ArchiveSession(impersonate='chrome') as session:
        market = download_stock_history('NVDA', start=protocol['market_raw_start'],
            end=protocol['market_raw_end_exclusive'], session=session, source_snapshot_path=source_path)
    market.to_csv(market_path, index=False)
    raw_files = [sec_copy, market_path, source_path, *sorted(raw_dir.glob('NVDA_yahoo_chart_*.json'))]
    meta = {'source': 'Yahoo Finance via existing nasdaq_research.data.download_stock_history',
            'auto_adjust': False, 'repair': False, 'dividend_adjustment_applied': False,
            'requested_start': protocol['market_raw_start'], 'requested_end_exclusive': protocol['market_raw_end_exclusive'],
            'sec_source': 'Existing immutable SEC Company Facts snapshot',
            'market_rows': len(market), 'market_start': str(market.date.min()), 'market_end': str(market.date.max()),
            'raw_sha256': {p.name:sha256(p) for p in raw_files}}
    write(meta_path,json_text(meta))
    return meta


def construct_features(market: pd.DataFrame, facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use trusted Stage 3/5/6.1 builders on full information history, then slice."""
    # Reproduce the registered Stage 3 -> Stage 5 -> Stage 6 CSV/parser
    # boundaries. Omitting these can change exact OHLC cells by one ULP.
    processed = pd.read_csv(StringIO(csv_text(build_features(market))))
    research = legacy_stage5_frame(processed, facts)
    snapshots, daily = build_feature_matrix(research, facts)
    for k in (5,20,60):
        daily[f'close_to_sma_{k}'] = daily.close/daily[f'sma_{k}']-1
    return processed, snapshots, daily


def legacy_stage5_frame(processed: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_text(build_research_dataset(processed, facts))))


def research_slice(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.date.astype(str).between(RESEARCH_START,RESEARCH_END)].reset_index(drop=True)


def target_oracle(features: pd.DataFrame, targets: pd.DataFrame, locked: tuple[str,str]) -> dict:
    """Independent scalar label/source oracle; never read locked numerical labels."""
    violations = checks = skipped = 0
    rows = features.reset_index(drop=True)
    for i,row in rows.iterrows():
        if locked[0] <= str(row.date) <= locked[1]:
            skipped += 1
            continue
        for h in HORIZONS:
            if i+h >= len(rows):
                violations += int(pd.notna(targets.loc[i,f'forward_return_{h}d']))
                violations += int(pd.notna(targets.loc[i,f'target_exit_date_{h}d']))
            else:
                expected = float(rows.iloc[i+h].close)/float(rows.iloc[i+1].open)-1
                violations += int(not np.isclose(targets.loc[i,f'forward_return_{h}d'],expected,rtol=0,atol=1e-12))
                violations += int(str(targets.loc[i,f'target_exit_date_{h}d']) != str(rows.iloc[i+h].date))
                violations += int(targets.loc[i,f'target_exit_close_{h}d'] != rows.iloc[i+h].close)
            checks += 1
        if i+1<len(rows):
            violations += int(str(targets.loc[i,'target_entry_date']) != str(rows.iloc[i+1].date))
            violations += int(targets.loc[i,'target_entry_open'] != rows.iloc[i+1].open)
        else:
            violations += int(pd.notna(targets.loc[i,'target_entry_date']))
            violations += int(pd.notna(targets.loc[i,'target_entry_open']))
    return {'violations':violations, 'horizon_checks':checks, 'locked_numeric_rows_skipped':skipped}


def scalar_equal(a, b, *, derived=False) -> bool:
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if derived:
        return bool(np.isclose(float(a),float(b),rtol=0,atol=1e-12))
    return bool(a == b)


def market_oracle(processed: pd.DataFrame, positions: list[int]) -> dict:
    """Scalar trailing math, no pandas rolling/std and no production builder."""
    violations = checks = 0
    closes = processed.close.tolist()
    returns = [np.nan]+[closes[i]/closes[i-1]-1 for i in range(1,len(closes))]
    for i in positions:
        expected = {
            'simple_return': returns[i],
            'log_return': np.log(closes[i]/closes[i-1]) if i else np.nan,
            'intraday_return': closes[i]/processed.open.iloc[i]-1,
            'daily_range': (processed.high.iloc[i]-processed.low.iloc[i])/closes[i],
        }
        for k in (5,20,60):
            expected[f'sma_{k}'] = mean(closes[i-k+1:i+1]) if i>=k-1 else np.nan
        for k in (20,60):
            expected[f'rolling_volatility_{k}'] = stdev(returns[i-k+1:i+1]) if i>=k else np.nan
        for c,v in expected.items():
            checks += 1
            violations += int(not scalar_equal(processed[c].iloc[i],v,derived=True))
    return {'violations': violations, 'cells_checked': checks}


def state_oracle(daily: pd.DataFrame, facts: pd.DataFrame, positions: list[int], sessions: pd.Series) -> dict:
    """Exhaustive source selection without as-of joins or production selectors."""
    records = facts.to_dict('records')
    calendar = [date.fromisoformat(str(d)) for d in sessions]
    checks = violations = 0
    for i in positions:
        current = daily.iloc[i]
        trading = date.fromisoformat(str(current.date))
        for prefix in ('q','fy','bs'):
            candidates = [r for r in records if date.fromisoformat(r['filing_date']) < trading
                and (r['data_period_type']=='quarterly' if prefix=='q' else
                     r['data_period_type']=='annual' if prefix=='fy' else
                     r['data_period_type'] in ('quarterly','annual') and any(pd.notna(r[c]) for c in BALANCE_FIELDS))]
            checks += 1
            if not candidates:
                violations += int(pd.notna(current[f'{prefix}_accession']))
                continue
            chosen = max(candidates,key=lambda r:(r['filing_date'],r['period_end'],
                r['data_period_type']=='quarterly',r['form'].endswith('/A'),r['period_start'],r['accession']))
            effective = next(d for d in calendar if d>date.fromisoformat(chosen['filing_date']))
            for c in ('accession','filing_date','period_end','period_start','form','data_period_type'):
                checks += 1
                violations += int(not scalar_equal(current[f'{prefix}_{c}'],chosen[c]))
            checks += 1
            violations += int(pd.Timestamp(current[f'{prefix}_effective_date']).strftime('%Y-%m-%d') != effective.isoformat())
            if prefix=='bs':
                for c in BALANCE_FIELDS:
                    checks += 1
                    violations += int(not scalar_equal(current[f'bs_{c}'],chosen[c]))
                for c,(n,d) in BALANCE_RATIOS.items():
                    expected = chosen[n]/chosen[d] if pd.notna(chosen[n]) and pd.notna(chosen[d]) and chosen[d]>0 else np.nan
                    checks += 1
                    violations += int(not scalar_equal(current[c],expected,derived=True))
    return {'violations':violations, 'cells_checked':checks}


def growth_oracle(snapshots: pd.DataFrame, facts: pd.DataFrame) -> dict:
    """Check recorded references, physical-year guards and scalar growth independently."""
    records = facts.to_dict('records')
    violations = checks = 0
    for _, row in snapshots.iterrows():
        prefix = 'q' if row.data_period_type=='quarterly' else 'fy'
        known = [r for r in records if r['filing_date']<=row.filing_date
                 and r['data_period_type']==row.data_period_type and r['fiscal_period']==row.fiscal_period]
        current_periods = {(r['period_start'],r['period_end']) for r in known if r['fiscal_year']==row.fiscal_year}
        previous = [r for r in known if r['fiscal_year']==row.fiscal_year-1]
        prior_periods = {(r['period_start'],r['period_end']) for r in previous}
        reference = None
        if len(current_periods)==1 and len(prior_periods)==1:
            candidate = max(previous,key=lambda r:(r['filing_date'],r['period_end'],r['form'].endswith('/A'),r['period_start'],r['accession']))
            start_gap = (date.fromisoformat(row.period_start)-date.fromisoformat(candidate['period_start'])).days
            end_gap = (date.fromisoformat(row.period_end)-date.fromisoformat(candidate['period_end'])).days
            if 330<=start_gap<=400 and 330<=end_gap<=400 and candidate['period_end']<row.period_start:
                reference = candidate
        checks += 1
        violations += int(not scalar_equal(row.yoy_reference_accession,reference['accession'] if reference else None))
        for name,field in GROWTH_FIELDS.items():
            denominator = reference[field] if reference else np.nan
            expected = row[field]/denominator-1 if pd.notna(row[field]) and pd.notna(denominator) and denominator>0 else np.nan
            checks += 1
            violations += int(not scalar_equal(row[f'{prefix}_{name}_growth_yoy'],expected,derived=True))
    return {'violations':violations,'cells_checked':checks}


def calendar_audit(market: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame,dict]:
    """US equity full-day closures; half days remain observed trading sessions."""
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, Holiday, nearest_workday, sunday_to_monday,
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        USLaborDay, USThanksgivingDay,
    )
    class EquityClosures(AbstractHolidayCalendar):
        rules = [Holiday('New Year',month=1,day=1,observance=sunday_to_monday),
                 USMartinLutherKingJr,USPresidentsDay,GoodFriday,USMemorialDay,
                 Holiday('Juneteenth',month=6,day=19,start_date='2022-01-01',observance=nearest_workday),
                 Holiday('Independence',month=7,day=4,observance=nearest_workday),
                 USLaborDay,USThanksgivingDay,
                 Holiday('Christmas',month=12,day=25,observance=nearest_workday)]
    weekdays = pd.bdate_range(start,end)
    holidays = EquityClosures().holidays(start,end)
    special = pd.to_datetime(['2012-10-29','2012-10-30','2018-12-05','2025-01-09'])
    expected = set(weekdays.difference(holidays.union(special)).strftime('%Y-%m-%d'))
    observed = set(market.loc[market.date.astype(str).between(start,end),'date'].astype(str))
    rows = [{'date':d,'status':'missing_expected_session'} for d in sorted(expected-observed)]
    rows += [{'date':d,'status':'unexpected_non_session'} for d in sorted(observed-expected)]
    audit = pd.DataFrame(rows,columns=['date','status'])
    return audit, {'expected_session_count':len(expected),'observed_session_count':len(observed),
        'violations':len(rows),'source':'https://www.nasdaq.com/market-activity/stock-market-holiday-schedule',
        'special_closure_source':'https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-announces-closure-its-us-markets-honor-national-day-0',
        'calendar_used_for_label_construction':False}


def corporate_actions(raw_dir: Path) -> tuple[pd.DataFrame,dict]:
    """Read action records from frozen response; reject missing full-window response."""
    results = []
    for p in sorted(raw_dir.glob('NVDA_yahoo_chart_*.json')):
        for result in json.loads(p.read_bytes()).get('chart',{}).get('result') or []:
            results.append((p,result))
    if not results:
        raise ValueError('No archived Yahoo chart response: corporate-action evidence unavailable')
    path,result = max(results,key=lambda item:len(item[1].get('timestamp',[])))
    rows = []
    for kind,events in result.get('events',{}).items():
        for event in events.values():
            d = pd.Timestamp(event['date'],unit='s',tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d')
            rows.append({'date':d,'event_type':kind,'split_ratio':event.get('numerator',np.nan)/event.get('denominator',1),
                         'dividend_amount':event.get('amount',np.nan),'source_sha256':sha256(path)})
    events = pd.DataFrame(rows,columns=['date','event_type','split_ratio','dividend_amount','source_sha256']).sort_values(['date','event_type']).reset_index(drop=True)
    return events, {'full_response':path.name,'timestamp_count':len(result['timestamp']),
                    'meta_symbol':result.get('meta',{}).get('symbol'), 'currency':result.get('meta',{}).get('currency')}


def corporate_audit(processed: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    rows=[]
    bound_bad = ((processed.low>processed.high) | (processed.low>processed.open) | (processed.low>processed.close)
                 | (processed.high<processed.open) | (processed.high<processed.close))
    ratios = processed.close/processed.close.shift(1)
    global_bad = ratios.notna() & ~ratios.between(.5,2.)
    for event in events.to_dict('records'):
        if not RESEARCH_START<=event['date']<=RESEARCH_END:
            continue
        matches = np.flatnonzero(processed.date.astype(str).eq(event['date']))
        if not len(matches):
            rows.append({**event,'violations':1,'reason':'event absent from observed calendar'})
            continue
        i=int(matches[0]); split=event['event_type']=='splits'
        adjacent=float(ratios.iloc[i]); gap=float(processed.open.iloc[i]/processed.close.iloc[i-1])
        ohlc=[float(processed[c].iloc[i]/processed[c].iloc[i-1]) for c in ('open','high','low','close')]
        mechanical = split and abs(adjacent-1/event['split_ratio']) < abs(adjacent-1)
        rolling = market_oracle(processed,list(range(max(60,i-1),min(len(processed),i+2))))
        # Cross-event targets use source position scalar formulas (no target distribution decisions).
        target_bad=0; max_cross=0.
        for h in HORIZONS if split else ():
            for t in range(max(0,i-h),i):
                if t+h>=len(processed): continue
                value=float(processed.close.iloc[t+h]/processed.open.iloc[t+1]-1)
                max_cross=max(max_cross,abs(value))
                target_bad += int(split and abs(value)>.5)
        violations=int(bound_bad.iloc[i])+int(split and (not all(.5<=v<=2. for v in ohlc) or mechanical))
        violations += rolling['violations']+target_bad
        rows.append({**event,'previous_session':str(processed.date.iloc[i-1]),'close_ratio':adjacent,
            'log_return':float(processed.log_return.iloc[i]),'open_to_prior_close':gap,
            'ohlc_ratios_in_registered_bounds':all(.5<=v<=2. for v in ohlc),
            'inverse_split_mechanical_jump':bool(mechanical), 'SMA_rolling_oracle_violations':rolling['violations'],
            'cross_event_max_abs_target_return':max_cross,'cross_event_target_violations':target_bad,
            'violations':violations,'reason':'documentary audit; unchanged source OHLC'})
    table=pd.DataFrame(rows)
    missing=[s['date'] for s in SPLITS if not ((events.date==s['date']) & (events.event_type=='splits') & (events.split_ratio==s['ratio'])).any()]
    return table, {'violations':int(table.violations.sum())+int(bound_bad.sum())+int(global_bad.sum())+len(missing),
                  'OHLC_bound_violations':int(bound_bad.sum()),'global_discontinuity_violations':int(global_bad.sum()),
                  'missing_documented_split_events':missing,
                  'split_events':table.loc[table.event_type.eq('splits')].to_dict('records'),
                  'dividend_events_in_research_window':int(table.event_type.eq('dividends').sum()),
                  'audit_metadata_used_in_features':False,
                  'price_vintage_note':'Vendor OHLC is split-adjusted in a fixed snapshot coordinate; dividends are not reapplied. Absolute levels are not certified historical as-traded prices.'}


def coverage(daily: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    groups={'market':MARKET_FEATURES,'relative_market':RELATIVE_FEATURES,
            'quarterly':(*QUARTERLY_RATIO_FEATURES,*QUARTERLY_GROWTH_FEATURES),
            'annual':(*ANNUAL_RATIO_FEATURES,*ANNUAL_GROWTH_FEATURES),'balance_sheet':tuple(BALANCE_RATIOS),
            'research_39':RESEARCH_FEATURES}
    yearly=[]; features=[]
    years=daily.date.astype(str).str[:4].astype(int)
    for y in range(2020,2027):
        mask=years.eq(y); part=daily.loc[mask]; n=len(part)
        row={'year':y,'market_rows':n,'duplicate_dates':int(part.date.duplicated().sum()),
             'non_finite_infinity_count':int(np.isinf(part.loc[:,[*RESEARCH_FEATURES,*RELATIVE_FEATURES]].to_numpy(dtype=float)).sum()),
             'missing_research_feature_cells':int(part.loc[:,RESEARCH_FEATURES].isna().sum().sum())}
        for name,cols in groups.items():
            row[f'{name}_feature_coverage']=float(part.loc[:,cols].notna().sum().sum()/(n*len(cols))) if n else None
        for prefix in ('q','fy','bs'):
            row[f'{prefix}_state_rows']=int(part[f'{prefix}_accession'].notna().sum())
            row[f'{prefix}_state_coverage']=float(part[f'{prefix}_accession'].notna().mean()) if n else None
        for h in HORIZONS:
            row[f'target_{h}d_valid_count']=int(targets.loc[mask,f'forward_return_{h}d'].notna().sum())
        yearly.append(row)
    for c in (*RESEARCH_FEATURES,*RELATIVE_FEATURES):
        valid=daily[c].notna(); infinite=np.isinf(daily[c].to_numpy(dtype=float))
        features.append({'feature':c,'group':next(k for k,v in groups.items() if k!='research_39' and c in v),
            'row_count':len(daily),'valid_count':int(valid.sum()),'missing_count':int((~valid).sum()),
            'infinite_count':int(infinite.sum()),'coverage':float(valid.mean()),
            'first_valid_date':str(daily.loc[valid,'date'].iloc[0]) if valid.any() else None})
    def first(mask):
        return str(daily.loc[mask,'date'].iloc[0]) if mask.any() else None
    dates={'first_research_date':str(daily.date.iloc[0]),'last_research_date':str(daily.date.iloc[-1]),
        'first_mature_60_session_market_date':first(daily[['sma_60','rolling_volatility_60']].notna().all(axis=1)),
        'first_quarterly_state':first(daily.q_accession.notna()),'first_annual_state':first(daily.fy_accession.notna()),
        'first_balance_sheet_state':first(daily.bs_accession.notna()),
        'first_all_39_features_valid':first(daily.loc[:,RESEARCH_FEATURES].notna().all(axis=1)),
        'first_structurally_eligible_5d_all_39':first(daily.loc[:,RESEARCH_FEATURES].notna().all(axis=1) & targets.forward_return_5d.notna())}
    for h in HORIZONS:
        dates[f'first_valid_{h}d_target']=first(targets[f'forward_return_{h}d'].notna())
    return pd.DataFrame(yearly),pd.DataFrame(features),dates


def overlap_audit(processed: pd.DataFrame, research: pd.DataFrame, daily: pd.DataFrame,
                  targets: pd.DataFrame, old_rebuilt: pd.DataFrame,
                  locked: tuple[str,str], *, raw_market: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    """Enumerate every cell mismatch; only proven boundary effects are accepted."""
    sources=[('raw',DATA_DIR/'raw/NVDA.csv',raw_market),
             ('market',DATA_DIR/'processed/NVDA.csv',processed),
             ('PIT_stage5',DATA_DIR/'research/NVDA_research.csv',research),
             ('research_stage61',DATA_DIR/'research/NVDA_features.csv',daily),
             ('targets_stage8',DATA_DIR/'research/targets/NVDA_targets.csv',targets),
             ('labeled_stage8',DATA_DIR/'research/NVDA_labeled.csv',daily.merge(targets.drop(columns='ticker'),on='date',validate='one_to_one'))]
    rows=[]; differences=[]; unexpected=[]; all_dates=set(); total_skipped=0
    old_rebuilt=old_rebuilt.copy()
    for c in date_columns(old_rebuilt.columns):
        old_rebuilt[c]=canonical_dates(old_rebuilt[c],allow_missing=True)
    old_rebuilt=old_rebuilt.set_index('date')
    for layer,path,new in sources:
        # Match the exact legacy parser contract for authoritative source cells.
        old=pd.read_csv(path)
        for c in date_columns(old.columns): old[c]=canonical_dates(old[c],allow_missing=True)
        for c in date_columns(new.columns):
            new=new.copy() if c==date_columns(new.columns)[0] else new
            new[c]=canonical_dates(new[c],allow_missing=True)
        boundary=str(old.date.iloc[0]); old=old.set_index('date'); new=new.set_index('date')
        common=old.index.intersection(new.index); all_dates.update(common)
        for c in old.columns:
            exact=tolerant=expected_count=violation_count=skipped=0
            if c not in new:
                unexpected.append({'layer':layer,'date':None,'field':c,'legacy_value':None,'expanded_value':None,'reason':'missing schema field'})
                rows.append({'layer':layer,'field':c,'overlap_dates':len(common),'unexpected_violations':1})
                continue
            for d in common:
                if locked[0]<=str(d)<=locked[1] and (c.startswith('forward_return_') or c.startswith(('target_entry_open','target_exit_close'))):
                    skipped+=1; continue
                a,b=old.at[d,c],new.at[d,c]
                if scalar_equal(a,b): exact+=1; continue
                if c in DERIVED and scalar_equal(a,b,derived=True): tolerant+=1; continue
                reason=evidence=None
                if c in WARMUP_ROWS and pd.isna(a) and pd.notna(b):
                    position=old.index.get_loc(d)
                    # Independent original-boundary reconstruction proves truncation;
                    # a missing cell outside registered positional warmup is never exempt.
                    if position<WARMUP_ROWS[c] and c in old_rebuilt and pd.isna(old_rebuilt.at[d,c]):
                        reason='registered_market_warmup_truncated_at_legacy_sample_start'
                        evidence=f'legacy_start={boundary}; legacy_position={position}; required_preceding_sessions={WARMUP_ROWS[c]}; old-prefix reconstruction missing; full-history scalar rolling oracle audited'
                if reason is None:
                    for prefix,age in [('fundamental','days_since_effective_date'),('q','q_days_since_effective_date'),('fy','fy_days_since_effective_date'),('bs','bs_days_since_effective_date')]:
                        allowed={f'{prefix}_effective_date',f'{prefix}_effective_date_basis',f'{prefix}_effective_date_is_sample_truncated',age}
                        accession=f'{prefix}_accession'; filed=f'{prefix}_filing_date'
                        if c in allowed and accession in old and filed in old and d in old_rebuilt.index and c in old_rebuilt:
                            source=old.at[d,accession]
                            if pd.notna(source) and scalar_equal(source,new.at[d,accession]) and old.at[d,filed]<boundary and scalar_equal(a,old_rebuilt.at[d,c]):
                                # Require actual new effective date from the raw calendar;
                                # the caller's independent state oracle verifies that calendar.
                                if c == f'{prefix}_effective_date':
                                    valid_new=old.at[d,filed]<b<boundary and a==boundary
                                elif c.endswith('is_sample_truncated'):
                                    valid_new=(a==True and b==False)
                                elif c.endswith('effective_date_basis'):
                                    valid_new=a=='sample_start_truncated' and b=='observed_next_session'
                                else:
                                    full_effective=str(new.at[d,f'{prefix}_effective_date'])
                                    valid_new=(b==(date.fromisoformat(str(d))-date.fromisoformat(full_effective)).days
                                        and a==(date.fromisoformat(str(d))-date.fromisoformat(boundary)).days)
                                if valid_new:
                                    reason='legacy_pre_window_filing_effective_date_left_censoring'
                                    evidence=f'same accession={source}; filing={old.at[d,filed]}; legacy_start={boundary}; expanded_effective={new.at[d,f"{prefix}_effective_date"]}; original-boundary reconstruction exact; independent next-session oracle'
                record={'layer':layer,'date':str(d),'field':c,'legacy_value':a,'expanded_value':b,
                        'reason':reason or 'unexpected_source_or_semantic_difference','evidence':evidence}
                if reason:
                    expected_count+=1; differences.append(record)
                else:
                    violation_count+=1; unexpected.append(record)
            total_skipped+=skipped
            rows.append({'layer':layer,'field':c,'overlap_dates':len(common),'exact_matches':exact,
                         'strict_tolerance_matches':tolerant,'expected_expansion_differences':expected_count,
                         'unexpected_violations':violation_count,'locked_numeric_target_cells_skipped':skipped})
    columns=['layer','date','field','legacy_value','expanded_value','reason','evidence']
    summary={'overlapping_dates':len(all_dates),'field_comparisons':len(rows),
        'exact_matches':sum(r.get('exact_matches',0) for r in rows),
        'strict_tolerance_matches':sum(r.get('strict_tolerance_matches',0) for r in rows),
        'expected_expansion_differences':len(differences),'unexpected_violations':len(unexpected),
        'locked_numeric_target_cells_skipped':total_skipped,
        'expected_reason_counts':pd.Series([r['reason'] for r in differences],dtype='object').value_counts().to_dict()}
    return pd.DataFrame(rows),pd.DataFrame(differences,columns=columns),pd.DataFrame(unexpected,columns=columns),summary


def row_equality(a: pd.DataFrame,b: pd.DataFrame, columns=None) -> int:
    try:
        cols=list(columns if columns is not None else a.columns)
        assert_values_equal(a.loc[:,cols].reset_index(drop=True),b.loc[:,cols].reset_index(drop=True),derived_columns=DERIVED)
        return 0
    except (AssertionError,ValueError,KeyError):
        return 1


def audit_positions(daily: pd.DataFrame, facts: pd.DataFrame, locked_start: str) -> list[int]:
    dates=daily.date.astype(str); available=(dates>=RESEARCH_START) & (dates<locked_start)
    positions=set(np.flatnonzero(available & ~dates.str[:4].duplicated()))
    positions.update(np.flatnonzero(available & ~dates.str[:4].duplicated(keep='last')))
    for event in SPLITS:
        matches=np.flatnonzero(dates.eq(event['date']))
        if len(matches): positions.update(range(max(0,int(matches[0])-1),min(len(daily),int(matches[0])+2)))
    # All research state transitions before the locked boundary, not chosen by values/performance.
    for prefix in ('q','fy','bs'):
        mask=daily[f'{prefix}_accession'].ne(daily[f'{prefix}_accession'].shift()) & available
        positions.update(np.flatnonzero(mask))
    return sorted(positions)


def leakage_audits(market: pd.DataFrame,facts: pd.DataFrame,processed: pd.DataFrame,
                   snapshots: pd.DataFrame,daily: pd.DataFrame,locked_start: str) -> tuple[pd.DataFrame,dict]:
    positions=audit_positions(daily,facts,locked_start)
    oracle_rows=[]; truncation=0
    # Full information prefix at a known feature timestamp; labels never supplied.
    for i in positions:
        d=str(daily.date.iloc[i])
        prefix_market=market.loc[market.date.astype(str)<=d].copy()
        known_facts=facts.loc[facts.filing_date<d].copy()
        _,_,prefix=construct_features(prefix_market,known_facts)
        violation=row_equality(daily.iloc[[i]],prefix.iloc[[-1]])
        truncation+=violation
        oracle_rows.append({'date':d,'audit':'full_vs_legal_information_prefix','violations':violation})
    market_check=market_oracle(processed,positions)
    state_check=state_oracle(daily,facts,positions,market.date)
    growth_check=growth_oracle(snapshots,facts)
    # Future SEC mutation, including amendment insertion, retains source chronology.
    future=facts.loc[facts.filing_date.between('2024-01-01','2024-12-31') & facts.data_period_type.eq('quarterly')].iloc[0]
    cutoff=str(future.filing_date)
    before=daily.date.astype(str)<=cutoff
    changed=facts.copy()
    same=changed.accession.eq(future.accession)
    for c in FINANCIAL_FIELDS: changed.loc[same,c]=changed.loc[same,c]*1000
    _,_,mutated=construct_features(market,changed)
    future_sec=row_equality(daily.loc[before],mutated.loc[before])
    amendment=future.copy(); amendment['accession']='stage13-adversarial-amendment'; amendment['form']='10-Q/A'
    amendment['filing_date']='2024-12-20'
    amended=pd.concat([facts,pd.DataFrame([amendment])],ignore_index=True)
    _,_,amendment_base=construct_features(market,amended)
    amended_changed=amended.copy()
    for c in FINANCIAL_FIELDS:
        mask=amended_changed.accession.eq(amendment['accession'])
        amended_changed.loc[mask,c]=amended_changed.loc[mask,c]*1000
    _,_,amendment_new=construct_features(market,amended_changed)
    before_amendment=daily.date.astype(str)<=amendment['filing_date']
    amendment_check=row_equality(amendment_base.loc[before_amendment],amendment_new.loc[before_amendment])
    # A positive control proves mutation can change state at/after activation.
    after=daily.date.astype(str)>'2024-12-20'
    amendment_effective_changed=not amendment_base.loc[after,'q_revenue_growth_yoy'].equals(amendment_new.loc[after,'q_revenue_growth_yoy'])
    changed_market=market.copy(); future_rows=changed_market.date.astype(str)>cutoff
    for c in ('open','high','low','close'): changed_market.loc[future_rows,c]*=17
    _,_,market_new=construct_features(changed_market,facts)
    future_market=row_equality(daily.loc[before],market_new.loc[before])
    shuffled=facts.sample(frac=1,random_state=13).reset_index(drop=True)
    _,_,shuffle_daily=construct_features(market,shuffled)
    ties=row_equality(daily,shuffle_daily)
    # An explicit whitelist projection is the only research-X interface.
    projected=research_x(daily)
    labeled=daily.copy()
    for c in (*RETURN_COLUMNS,'target_entry_open','target_exit_close_5d','target_entry_date'):
        labeled[c]='mutated future target/provenance'
    target_isolation=row_equality(projected,research_x(labeled))
    relative_violations=0
    for i in positions:
        for k in (5,20,60):
            expected=float(daily.close.iloc[i])/float(daily[f'sma_{k}'].iloc[i])-1
            relative_violations+=int(not scalar_equal(daily[f'close_to_sma_{k}'].iloc[i],expected,derived=True))
    checks={'PIT_truncation':{'violations':truncation,'representative_dates':len(positions)},
            'market_rolling_oracle':market_check,'independent_state_oracle':state_check,'independent_YoY_oracle':growth_check,
            'future_SEC_isolation':{'violations':future_sec,'cutoff':cutoff},
            'amendment_timing_isolation':{'violations':amendment_check,'filing_date':amendment['filing_date'],
                 'post_effective_positive_control':amendment_effective_changed,'fixture_only':True},
            'future_market_isolation':{'violations':future_market,'cutoff':cutoff},
            'future_target_isolation':{'violations':target_isolation,'whitelist':list(RESEARCH_FEATURES)},
            'deterministic_source_order':{'violations':ties},
            'relative_SMA_oracle':{'violations':relative_violations,'cells_checked':len(positions)*3},
            'future_split_metadata_isolation':{'violations':0,'basis':'construct_features takes only OHLCV and facts; no action metadata input; raw price vintage held fixed'},
            'future_split_price_vintage_invariance_certified':False}
    return pd.DataFrame(oracle_rows),checks


def research_x(data: pd.DataFrame) -> pd.DataFrame:
    """Explicit 39-feature projection; targets and metadata have no dependency."""
    return data.loc[:,RESEARCH_FEATURES].copy()


def run_pipeline(output_dir: Path = OUTPUT_DIR, *, acquire_missing=False) -> dict:
    protocol=register(output_dir)
    upstream_before=verify_upstream(output_dir)
    if upstream_before['mutation_violations']:
        raise ValueError('Legacy upstream changed before execution')
    if acquire_missing:
        acquire(output_dir)
    raw_dir=output_dir/'raw'; meta_path=raw_dir/'NVDA_acquisition.json'
    if not meta_path.exists():
        summary={'completion_status':'INCOMPLETE_OR_BLOCKED','reason':'Reliable frozen market acquisition is missing',**SAFETY}
        write(output_dir/'NVDA_stage13_summary.json',json_text(summary))
        write(output_dir/'NVDA_stage13_validation.json',json_text({'valid':False,**summary,'upstream':upstream_before}))
        return summary
    meta=json.loads(meta_path.read_text())
    for name,h in meta['raw_sha256'].items():
        if sha256(raw_dir/name)!=h: raise ValueError(f'Raw input changed: {name}')
    # Preserve the historical decimal parser contract; formal CSV reload uses round_trip.
    market=pd.read_csv(raw_dir/'NVDA_market.csv')
    facts,normalization=normalize_company_facts(json.loads((raw_dir/'NVDA_companyfacts.json').read_bytes()))
    # Reuse the exact normalized vintage cells produced by Stage 4 when identity is unchanged.
    old_facts=pd.read_csv(DATA_DIR/'fundamentals/processed/NVDA_fundamentals.csv')
    assert_values_equal(facts,old_facts)
    facts=old_facts
    processed,snapshots,daily_full=construct_features(market,facts)
    daily=research_slice(daily_full)
    targets=build_targets(daily)
    locked_rules=protocol['final_test_lock_rules']; locked=(locked_rules['start'],locked_rules['end'])
    presample=int((market.date.astype(str)<RESEARCH_START).sum())
    stage5_constructed=build_research_dataset(processed,facts)
    stage5_check=validate_research_dataset(stage5_constructed,processed,facts)
    research=legacy_stage5_frame(processed,facts)
    stage61_check=validate_feature_matrix(daily_full.drop(columns=list(RELATIVE_FEATURES)),snapshots,research,facts)
    target_check=target_oracle(daily,targets,locked)
    events,event_meta=corporate_actions(raw_dir)
    corporate_table,corporate_check=corporate_audit(processed,events)
    session_table,session_check=calendar_audit(market,RESEARCH_START,RESEARCH_END)
    raw_session_table,raw_session_check=calendar_audit(market,protocol['market_raw_start'],RESEARCH_END)
    yearly,feature_coverage,first_dates=coverage(daily,targets)
    old_market=pd.read_csv(DATA_DIR/'raw/NVDA.csv')
    _,_,old_rebuilt=construct_features(old_market,facts)
    overlap,expected,unexpected,overlap_check=overlap_audit(processed,research,daily_full,targets,old_rebuilt,locked,raw_market=market)
    pit_oracles,leakage=leakage_audits(market,facts,processed,snapshots,daily_full,locked[0])
    upstream=verify_upstream(output_dir)
    required={'historical_scope':daily.date.iloc[0]=='2020-01-02' and daily.date.iloc[-1]==RESEARCH_END and set(yearly.year)==set(range(2020,2027)),
              'warmup_sufficient':presample>=max(WARMUP_ROWS.values()),
              'stage5_PIT':stage5_check['valid'],'stage61_PIT_features':stage61_check['valid'],
              'fundamentals_valid':validate_fundamentals(facts)['valid'],
              'target_oracle':target_check['violations']==0,
              'corporate_actions':corporate_check['violations']==0,
              'calendar_complete':session_check['violations']==0,
              'raw_calendar_complete':raw_session_check['violations']==0,
              'raw_response_row_preservation':event_meta['timestamp_count']==len(market),
              'overlap_compatible':overlap_check['unexpected_violations']==0,
              'leakage_oracles':all(v['violations']==0 for v in leakage.values() if isinstance(v,dict)),
              'amendment_positive_control':leakage['amendment_timing_isolation']['post_effective_positive_control'],
              'upstream_immutable':upstream['mutation_violations']==0,
              'finite_features':not np.isinf(daily.loc[:,RESEARCH_FEATURES].to_numpy(dtype=float)).any(),
              'no_model_or_test_evaluation':all(SAFETY[k]==v for k,v in SAFETY.items())}
    # Snapshot-PIT is conditional on fixed source coordinate; no claim of as-traded archive.
    status=('COMPLETE_WITH_DOCUMENTED_EXPECTED_EXPANSION_DIFFERENCES' if len(expected) else 'COMPLETE') if all(required.values()) else 'INCOMPLETE_OR_BLOCKED'
    summary={'completion_status':status,'acceptance_failures':[k for k,v in required.items() if not v],
        'stage_name':protocol['stage_name'],'research_window':[RESEARCH_START,RESEARCH_END],
        'actual_research_dates':[str(daily.date.iloc[0]),str(daily.date.iloc[-1])],
        'research_rows':len(daily),'market_raw_rows':len(market),'market_raw_dates':[str(market.date.iloc[0]),str(market.date.iloc[-1])],
        'market_presample_rows':presample,'SEC_filing_range':[facts.filing_date.min(),facts.filing_date.max()],
        'SEC_period_range':[facts.period_end.min(),facts.period_end.max()],
        'SEC_observation_counts':facts.data_period_type.value_counts().to_dict(),
        'SEC_amendment_observation_count':int(facts.form.str.endswith('/A').sum()),
        'SEC_real_amendment_note':'No real amendment observations in retained archived source; independently tested in-memory amendment fixtures.',
        'research_feature_count':len(RESEARCH_FEATURES),'relative_feature_count':len(RELATIVE_FEATURES),
        'yearly_coverage':yearly.to_dict('records'),'first_usable_dates':first_dates,
        'targets':{f'{h}d':{'valid_count':int(targets[f'forward_return_{h}d'].notna().sum()),'tail_NaN_count':int(targets[f'forward_return_{h}d'].isna().sum())} for h in HORIZONS},
        'corporate_action_audit':corporate_check,'overlap':overlap_check,'leakage_audits':leakage,
        'upstream':upstream,'inherited_vintage_limitations':protocol['inherited_vintage_limitations'],**SAFETY}
    tables={'market_history':research_slice(processed),'fundamental_history':facts,'fundamental_snapshots':snapshots,
            'pit_states':research_slice(research),'research_matrix':daily,'targets':targets,
            'yearly_coverage':yearly,'feature_coverage':feature_coverage,'corporate_action_events':events,
            'corporate_action_audit':corporate_table,'calendar_violations':session_table,
            'overlap_compatibility':overlap,'expected_expansion_differences':expected,'unexpected_overlap_differences':unexpected,
            'PIT_truncation_oracles':pit_oracles}
    roundtrip_checks={}
    output_hashes={}
    for name,frame in tables.items():
        text=csv_text(frame)
        reloaded=pd.read_csv(StringIO(text),float_precision='round_trip')
        if name in ('expected_expansion_differences','unexpected_overlap_differences'):
            # Mixed numeric/bool/text old/new values are documentary source
            # tokens. Verify their lexical values rather than inferred dtypes.
            lexical=pd.read_csv(StringIO(text),dtype=str,keep_default_na=False)
            roundtrip_checks[name]=lexical.to_csv(index=False,lineterminator='\n')==text
        elif frame.empty:
            roundtrip_checks[name]=list(frame.columns)==list(reloaded.columns)
        else:
            roundtrip_checks[name]=row_equality(frame,reloaded)==0
        path=output_dir/f'NVDA_stage13_{name}.csv';write(path,text);output_hashes[path.name]=sha256(path)
    required['CSV_roundtrip']=all(roundtrip_checks.values())
    if not required['CSV_roundtrip']:
        summary['completion_status']='INCOMPLETE_OR_BLOCKED';summary['acceptance_failures'].append('CSV_roundtrip')
    validation={'valid':all(required.values()),'completion_status':summary['completion_status'],'checks':required,
        'stage5_validation':stage5_check,'stage61_validation':stage61_check,'target_oracle':target_check,
        'corporate_action_audit':corporate_check,'event_source':event_meta,'calendar':session_check,
        'raw_calendar':raw_session_check,
        'overlap':overlap_check,'leakage_audits':leakage,'CSV_roundtrip':roundtrip_checks,
        'normalization':normalization,'upstream':upstream,'protocol_sha256':sha256(output_dir/'NVDA_stage13_protocol.json'),
        'raw_sha256':meta['raw_sha256'],'formal_CSV_sha256':output_hashes,**SAFETY}
    write(output_dir/'NVDA_stage13_summary.json',json_text(summary))
    write(output_dir/'NVDA_stage13_validation.json',json_text(validation))
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--acquire-only',action='store_true')
    parser.add_argument('--acquire',action='store_true',help='Acquire only when immutable raw cache is missing')
    args=parser.parse_args()
    if args.register_only: result=register()
    elif args.acquire_only: result=acquire()
    else: result=run_pipeline(acquire_missing=args.acquire)
    print(json_text({k:result[k] for k in ('completion_status','acceptance_failures','research_rows','market_raw_rows') if k in result}))


if __name__=='__main__':
    main()
