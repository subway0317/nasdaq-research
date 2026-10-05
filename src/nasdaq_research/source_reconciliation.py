"""Stage 13.1: narrow source-vintage provenance investigation, no model work."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from nasdaq_research.config import DATA_DIR, PROJECT_ROOT
from nasdaq_research.data import download_stock_history, standardize_history
from nasdaq_research.historical_expansion import json_text, sha256, write

OUT = DATA_DIR/'research/historical_expansion/stage13_1_yahoo_reconciliation'
STAGE13 = DATA_DIR/'research/historical_expansion/stage13_nvda'
PRIMARY_DATE = '2026-09-30'
PRIMARY_FIELD = 'volume'
LEGACY_VALUE, EXPANDED_VALUE = 121269300, 121732200
ROOTS = ('PIPELINE_PROCESSING_BUG', 'LEGACY_INCOMPLETE_SESSION_SNAPSHOT',
         'VENDOR_SOURCE_VINTAGE_DRIFT', 'UNRESOLVED_SOURCE_DISCREPANCY')
LAYERS = ('raw_yahoo_chart', 'raw_yfinance_table', 'normalized_market',
          'stage3', 'stage5', 'stage61', 'stage8_labeled')
SAFETY = dict(final_test_locked=True, final_test_predictions_generated=False,
    final_test_metrics_computed=False, final_training_pool_used=False,
    new_model_fit=False, new_CV=False, feature_changes=False, target_changes=False,
    final_test_performance_dependency=False, stage14_designed=False, stage14_executed=False)
POLICIES = {
    ROOTS[0]: {'authority':'corrected_and_revalidated_expanded_namespace',
        'required':'Exact bug location, frozen raw preservation, separate corrected namespace and narrow revalidation; original Stage 13 blocked record remains immutable',
        'status':'RESOLVED_PIPELINE_BUG'},
    ROOTS[1]: {'authority':'stage13_complete_session_frozen_snapshot',
        'required':'Reliable legacy incomplete-session evidence and expanded complete-session evidence; do not overwrite either vintage',
        'status':'RESOLVED_LEGACY_INCOMPLETE_SNAPSHOT'},
    ROOTS[2]: {'authority':'stage13_frozen_multi_year_snapshot',
        'required':'Both original source values, faithful processing, comparable session/source semantics and legacy partial-session possibility excluded; document vintage transition, never splice values',
        'status':'RESOLVED_VENDOR_VINTAGE_DRIFT'},
    ROOTS[3]: {'authority':None,
        'required':'No expanded downstream approval until missing causal evidence is recovered; preserve both records and original Stage 13 violations',
        'status':'UNRESOLVED_BLOCKER'},
}


def protocol() -> dict:
    """Only the user-specified discrepancy and prospective decision rules."""
    return {
        'stage_name':'Stage 13.1 — Yahoo Source-Vintage Reconciliation and Canonical Snapshot Policy',
        'stage_type':'DATA PROVENANCE / SOURCE AUTHORITY RESOLUTION STAGE',
        'ticker':'NVDA','primary_date':PRIMARY_DATE,'primary_field':PRIMARY_FIELD,
        'legacy_value':LEGACY_VALUE,'expanded_value':EXPANDED_VALUE,'difference':462900,
        'investigation_scope':{'primary':'Only NVDA 2026-09-30 Volume',
            'allowed_context_dates':['2026-09-29','2026-09-30','2026-10-01'],
            'forbidden':'No current-vendor comparison of the full 4304-row historical data, new quality protocol, feature/target/split change or performance analysis'},
        'source_authority_hierarchy':[
            {'level':1,'authority':'Frozen original Yahoo HTTP response or original yfinance table; normalized CSV named raw is not an original response'},
            {'level':2,'authority':'Contemporaneous acquisition metadata, query and version/session parameters'},
            {'level':3,'authority':'Normalized market/parser/dtype/session representation and downstream artifacts'},
            {'level':4,'authority':'Current Yahoo response as current state only; cannot reconstruct missing legacy evidence'},
            {'level':5,'authority':'Third-party corroboration only; never sole causal authority'}],
        'root_cause_categories':list(ROOTS),
        'root_cause_decision_rules':{
            ROOTS[0]:'Proven same-source processing mismatch or an exact parser/cache/boundary bug causing this discrepancy; require file/function/transformation evidence.',
            ROOTS[1]:'Reliable contemporaneous timing or source-unfinal evidence proves legacy incomplete at the primary session; lower volume, file mtime and Git commit time are insufficient.',
            ROOTS[2]:'Both original frozen sources differ, each is preserved faithfully, session/source semantics comparable and unresolved legacy partial-session possibility excluded. Current Yahoo is corroboration only.',
            ROOTS[3]:'Mandatory fallback for absent legacy raw, insufficient acquisition timing, incompatible sources, conflicting evidence or multiple remaining explanations. Absence of incomplete-session evidence is not evidence of completion.'},
        'canonical_policy_rules':POLICIES,
        'approval_rule':'Resolved category plus evidence and all engineering acceptance checks; pipeline-bug branch additionally corrected and revalidated; otherwise false',
        'first_divergence_rules':'Compare source to downstream ordered layers. If earlier legacy evidence is absent, report first observable mismatch separately and true first causal divergence as not identifiable. Do not label normalized CSV as original vendor raw.',
        'allowed_external_evidence':{'order':'Only after local trace is written',
            'current_yahoo_query':{'start':'2026-09-29','end_exclusive':'2026-10-02','interval':'1d','auto_adjust':False},
            'cache':'Freeze one acquisition with original response, source table, normalized output, request parameters and timezone-aware retrieval timestamp; subsequent runs offline',
            'third_party':'Only if necessary, secondary corroboration, never sole authority'},
        'forbidden_interventions':['manual value replacement','tolerance widening','Volume omission','primary-date exclusion',
            'post-hoc expected difference reclassification','source splicing','synthetic historical raw','model/CV/MAE/PnL/Sharpe/backtest',
            'overwrite Stage 1–13 artifacts or original Stage 13 blocked result','automatic Stage 14 work'],
        'legacy_artifact_immutability_rules':'Hash all current Stage 1–12 inputs, including already modified data.py; preserve original legacy value and artifacts byte-identically',
        'stage13_artifact_immutability_rules':'Hash every current Stage 13 artifact, code/test/report; original status INCOMPLETE_OR_BLOCKED and unexpected count 5 remain immutable. Any evidence-proven bug correction uses separate versioned namespace.',
        'final_test_lock_rules':{'rows':50,'start':'2026-07-15','end':'2026-09-23',**SAFETY},
        'reproducibility_rules':'At least two consecutive formal offline runs on frozen local/external evidence; all deterministic CSV/JSON bytes identical. External timestamp recorded once, never refreshed during reproducibility.',
        'acceptance_rules':{'resolved':'Complete causal trace, supported first divergence, exactly one supported root cause, frozen mapped policy, no prohibited editing, hashes unchanged, no models/test evaluation, all historical/new tests pass, deterministic repeats',
                            'unresolved':'Record evidence gaps and remaining hypotheses; freeze no-approval policy; engineering audit can pass while root-cause resolution remains false'},
    }


def output_path(name: str, suffix='.json') -> Path:
    return OUT/f'NVDA_stage13_1_{name}{suffix}'


def register() -> dict:
    OUT.mkdir(parents=True,exist_ok=True)
    value=protocol();path=output_path('protocol')
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Pre-registered Stage 13.1 protocol is locked')
    if not path.exists():write(path,json_text(value))
    manifest_path=output_path('protected_sha256')
    if not manifest_path.exists():
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=PROJECT_ROOT).decode().split('\0')
        paths={PROJECT_ROOT/p for p in tracked if p and (PROJECT_ROOT/p).is_file()}
        paths.update(p for p in DATA_DIR.rglob('*') if p.is_file() and OUT not in p.parents)
        paths.update(PROJECT_ROOT/p for p in ['src/nasdaq_research/historical_expansion.py',
                                            'tests/test_historical_expansion.py','docs/stage13_execution_report.md'])
        manifest={str(p.relative_to(PROJECT_ROOT)):sha256(p) for p in sorted(paths) if '__pycache__' not in p.parts}
        write(manifest_path,json_text(manifest))
    initial_path=output_path('initial_repository_state')
    if not initial_path.exists():
        initial={'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT).decode().strip(),
                 'branch':subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT_ROOT).decode().strip(),
                 'git_log_5':subprocess.check_output(['git','log','-5','--oneline'],cwd=PROJECT_ROOT).decode(),
                 'pre_existing_changes':[
                    'modified src/nasdaq_research/data.py','untracked data/research/historical_expansion/stage13_nvda/',
                    'untracked docs/stage13_execution_report.md','untracked src/nasdaq_research/historical_expansion.py',
                    'untracked tests/test_historical_expansion.py'],
                 'initial_observed_status_before_stage13_1_creation':'## main...origin/main\n M src/nasdaq_research/data.py\n?? data/research/historical_expansion/\n?? docs/stage13_execution_report.md\n?? src/nasdaq_research/historical_expansion.py\n?? tests/test_historical_expansion.py\n',
                 'registration_status':subprocess.check_output(['git','status','--short','--branch'],cwd=PROJECT_ROOT).decode(),
                 'Stage13_commit_state':'NOT_COMMITTED; existing Stage 13 artifacts/code/tests/report are protected inputs',
                 'initial_diff_check_passed':subprocess.run(['git','diff','--check'],cwd=PROJECT_ROOT,capture_output=True).returncode==0}
        write(initial_path,json_text(initial))
    return value


def verify_protected() -> dict:
    manifest=json.loads(output_path('protected_sha256').read_text())
    changes=[p for p,h in manifest.items() if not (PROJECT_ROOT/p).is_file() or sha256(PROJECT_ROOT/p)!=h]
    stage13={p for p in manifest if '/stage13_nvda/' in p or p in (
        'src/nasdaq_research/historical_expansion.py','tests/test_historical_expansion.py','docs/stage13_execution_report.md')}
    return {'protected_files_checked':len(manifest),'stage1_12_files_checked':len(manifest)-len(stage13),
            'stage13_files_checked':len(stage13),'mutation_violations':len(changes),
            'stage1_12_mutation_violations':sum(p not in stage13 for p in changes),
            'stage13_mutation_violations':sum(p in stage13 for p in changes),'changed_files':changes}


@dataclass(frozen=True)
class Evidence:
    legacy_raw_available: bool=False
    expanded_raw_available: bool=False
    legacy_raw_value: int | None=None
    expanded_raw_value: int | None=None
    legacy_processing_faithful: bool=False
    expanded_processing_faithful: bool=False
    legacy_timing_known: bool=False
    legacy_complete_session_proven: bool=False
    legacy_incomplete_session_proven: bool=False
    expanded_complete_session_proven: bool=False
    source_session_identity_comparable: bool=False
    processing_bug_proven: bool=False
    bug_file: str | None=None
    bug_function: str | None=None
    bug_transformation: str | None=None
    corrected_and_revalidated: bool=False
    conflicting_evidence: bool=False
    unresolved_partial_session_possibility: bool=True


def classify(e: Evidence) -> str:
    if (e.conflicting_evidence or
            (e.legacy_complete_session_proven and e.legacy_incomplete_session_proven) or
            (e.legacy_raw_available and e.legacy_raw_value is None) or
            (e.expanded_raw_available and e.expanded_raw_value is None)):
        return ROOTS[3]
    if e.processing_bug_proven and e.bug_file and e.bug_function and e.bug_transformation:
        return ROOTS[0]
    if (e.legacy_raw_available and e.expanded_raw_available and e.legacy_processing_faithful
            and e.expanded_processing_faithful and e.legacy_incomplete_session_proven
            and e.legacy_timing_known and e.expanded_complete_session_proven):
        return ROOTS[1]
    if (e.legacy_raw_available and e.expanded_raw_available and e.legacy_raw_value!=e.expanded_raw_value
            and e.legacy_processing_faithful and e.expanded_processing_faithful
            and e.legacy_timing_known and e.legacy_complete_session_proven
            and not e.legacy_incomplete_session_proven and not e.unresolved_partial_session_possibility
            and e.expanded_complete_session_proven and e.source_session_identity_comparable):
        return ROOTS[2]
    return ROOTS[3]


def canonical_policy(root_cause: str, *, corrected_and_revalidated=False, engineering_passed=True) -> dict:
    if root_cause not in POLICIES:raise ValueError('Root cause outside four pre-registered categories')
    mapped=POLICIES[root_cause]
    approved=(root_cause!=ROOTS[3] and engineering_passed and
              (root_cause!=ROOTS[0] or corrected_and_revalidated))
    status=mapped['status'] if approved else 'UNRESOLVED_BLOCKER'
    return {'root_cause':root_cause,'canonical_policy_frozen':True,
            'future_stage14_plus_market_data_authority':mapped['authority'] if approved else None,
            'policy_required_evidence':mapped['required'],'completion_status':status,
            'expanded_dataset_approved_for_downstream_research':approved,
            'stage14_modeling_eligible':approved and status.startswith('RESOLVED_'),
            'legacy_vendor_vintage_preserved':True,'stage13_vendor_vintage_preserved':True,
            'Stage13_historical_status':'INCOMPLETE_OR_BLOCKED','Stage13_original_overlap_violations':5,
            'manual_value_override_allowed':False,'current_yahoo_replaces_legacy_evidence':False,**SAFETY}


def integer_volume(value: Any) -> int:
    """Integer counts are exact; no floating tolerance or fractional coercion."""
    from decimal import Decimal, InvalidOperation
    if isinstance(value,bool) or value is None:raise ValueError('Volume requires an integer count')
    try:number=Decimal(str(value))
    except InvalidOperation as exc:raise ValueError('Invalid Volume') from exc
    if not number.is_finite() or number!=number.to_integral_value() or number<0:
        raise ValueError('Volume requires a finite nonnegative integer count')
    return int(number)


def primary_csv_volume(path: Path) -> int:
    # Read only date/Volume, never targets or performance columns.
    data=pd.read_csv(path,usecols=['date','volume'],dtype=str)
    rows=data.loc[data.date.eq(PRIMARY_DATE),'volume']
    if len(rows)!=1:raise ValueError(f'Exactly one primary-date row required: {path}')
    return integer_volume(rows.iloc[0])


def source_table(path: Path) -> pd.DataFrame:
    frame=pd.read_csv(path,header=[0,1],index_col=0)
    frame.index=pd.to_datetime(frame.index)
    frame.index.name='Date'
    return frame


def chart_primary(path: Path) -> tuple[int | None,dict,list[dict]]:
    payload=json.loads(path.read_bytes())
    results=payload.get('chart',{}).get('result') or []
    if len(results)!=1:raise ValueError('Expected one NVDA chart result')
    result=results[0];meta=result.get('meta',{})
    if meta.get('symbol')!='NVDA':raise ValueError('Expected NVDA chart identity')
    timestamps=result.get('timestamp',[]);volumes=result.get('indicators',{}).get('quote',[{}])[0].get('volume',[])
    if len(timestamps)!=len(volumes):raise ValueError('Chart timestamp/Volume count mismatch')
    primary=[];context=[]
    for stamp,value in zip(timestamps,volumes):
        utc=pd.Timestamp(stamp,unit='s',tz='UTC');local=utc.tz_convert('America/New_York')
        d=local.strftime('%Y-%m-%d')
        if d in ('2026-09-29',PRIMARY_DATE,'2026-10-01'):
            context.append({'date':d,'bar_epoch':stamp,'bar_time_UTC':utc.isoformat(),
                'bar_time_US_Eastern':local.isoformat(),'volume':integer_volume(value),
                'UTC_and_Eastern_date_match':utc.strftime('%Y-%m-%d')==d})
        if d==PRIMARY_DATE:primary.append(integer_volume(value))
    if len(primary)>1:raise ValueError('Multiple primary-date source bars; session ambiguous')
    evidence_meta={k:meta.get(k) for k in ['symbol','exchangeTimezoneName','timezone','gmtoffset',
        'dataGranularity','range','regularMarketTime','currentTradingPeriod']}
    if meta.get('regularMarketTime'):
        evidence_meta['regularMarketTime_UTC']=pd.Timestamp(meta['regularMarketTime'],unit='s',tz='UTC').isoformat()
    evidence_meta['timestamp_semantics']='bar epoch is a daily-session timestamp; regularMarketTime is quote time; neither is HTTP retrieval timestamp'
    return primary[0] if primary else None,evidence_meta,context


def frame_write(name: str, rows: list[dict] | pd.DataFrame, columns=None) -> Path:
    frame=rows.copy() if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    if 'value' in frame and 'field' in frame:
        frame['value']=pd.array(frame['value'],dtype='Int64')
    path=output_path(name,'.csv');write(path,frame.to_csv(index=False,lineterminator='\n'))
    return path


def inventory_row(side: str,kind: str,path: Path | None,*,role: str,status=None,notes='',date_range=None,query=None) -> dict:
    exists=path is not None and path.is_file()
    return {'source_side':side,'source_type':kind,
        'file_path':str(path.relative_to(PROJECT_ROOT)) if path else None,'exists':exists,
        'evidence_status':status or ('AVAILABLE' if exists else 'NOT_AVAILABLE'),
        'sha256':sha256(path) if exists else None,'acquisition_time_if_known':None,
        'date_range_if_known':date_range,'query_if_known':query,'authoritative_role':role,'notes':notes}


def trace_row(side: str,layer: str,path: Path | None,value: int | None,*,level: int,notes='',representation='persisted_artifact') -> dict:
    return {'side':side,'layer':layer,'source_file':str(path.relative_to(PROJECT_ROOT)) if path else None,
        'source_hash':sha256(path) if path and path.is_file() else None,
        'date':PRIMARY_DATE,'field':PRIMARY_FIELD,'value':value,'available':value is not None,
        'authority_level':level,'representation':representation,'notes':notes}


def first_divergence(trace: pd.DataFrame) -> dict:
    """Distinguish first observable mismatch from an unobserved causal origin."""
    missing_earlier=[]
    for layer in LAYERS:
        rows={side:trace.loc[(trace.side==side)&(trace.layer==layer)] for side in ('legacy','expanded')}
        if any(len(rows[side])!=1 or not bool(rows[side].iloc[0].available) for side in rows):
            missing_earlier.append(layer);continue
        a,b=(integer_volume(rows[side].iloc[0].value) for side in ('legacy','expanded'))
        if a!=b:
            identified=not missing_earlier
            return {'first_divergence_layer':layer if identified else 'NOT_IDENTIFIABLE_WITH_AVAILABLE_EVIDENCE',
                'first_observed_divergence_layer':layer,'causal_first_divergence_identified':identified,
                'legacy_value':a,'expanded_value':b,'missing_earlier_layers':missing_earlier,
                'first_divergence_evidence':[
                    {'side':side,'source_file':rows[side].iloc[0].source_file,'source_hash':rows[side].iloc[0].source_hash} for side in rows],
                'interpretation':'First common available mismatch is confirmed; missing earlier evidence prevents causal source localization' if not identified else 'Both sides have earlier original evidence; first mismatch localized'}
    return {'first_divergence_layer':None,'first_observed_divergence_layer':None,
            'causal_first_divergence_identified':False,'missing_earlier_layers':missing_earlier,
            'interpretation':'No comparable available mismatch found'}


def propagation(trace: pd.DataFrame) -> dict:
    violations=edges=0;per_side={}
    for side in ('legacy','expanded'):
        rows=[trace.loc[(trace.side==side)&(trace.layer==layer)].iloc[0] for layer in LAYERS]
        available=[r for r in rows if bool(r.available)]
        bad=sum(integer_volume(a.value)!=integer_volume(b.value) for a,b in zip(available,available[1:]))
        edges+=max(0,len(available)-1);violations+=bad
        per_side[side]={'available_layers':len(available),'propagation_violations':bad,
            'original_source_to_normalized_verified':bool(rows[0].available) and bool(rows[1].available) and bool(rows[2].available) and bad==0,
            'normalized_to_downstream_verified':all(bool(r.available) for r in rows[2:]) and bad==0}
    return {'propagation_consistency_violations':violations,'available_lineage_edges_checked':edges,'sides':per_side}


def metadata_row(side: str,key: str,value,*,status='AVAILABLE',source=None,notes='') -> dict:
    return {'source_side':side,'metadata_field':key,'value':value,
            'evidence_status':status,'source_file':source,'notes':notes}


def discover_local() -> dict:
    """Only primary Volume and request/session context; never rebuild old outputs."""
    register()
    protected=verify_protected()
    if protected['mutation_violations']:raise ValueError('Protected Stage 1–13 inputs changed')
    inventory=[];trace=[];metadata=[];timezone_rows=[]
    # Search inventory is itself frozen before external acquisition.
    files=sorted(p for directory in ('data','config','docs','src') for p in (PROJECT_ROOT/directory).rglob('*')
                 if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts)
    vendor_sources=[]
    for p in files:
        if p.suffix.lower()=='.json' and '/fundamentals/' not in str(p):
            obj=json.loads(p.read_bytes())
            if isinstance(obj,dict) and isinstance(obj.get('chart'),dict):vendor_sources.append(p)
    legacy_charts=[p for p in vendor_sources if STAGE13 not in p.parents]
    # No synthetic or inferred raw source is created when discovery is empty.
    if legacy_charts:
        raise ValueError('Additional legacy chart evidence discovered; inspect and add strict identity-specific handling before deciding')
    inventory.append(inventory_row('legacy','original_yahoo_chart',None,role='Level 1 original source',notes='Repository discovery found no legacy chart response'))
    inventory.append(inventory_row('legacy','original_yfinance_table',None,role='Level 1 original source',notes='Legacy downloader standardized immediately and saved only six OHLCV columns; no original yfinance representation retained'))
    inventory.append(inventory_row('legacy','acquisition_metadata',None,role='Level 2 acquisition context',notes='No contemporaneous retrieval/request timestamp, query manifest or library version recovered'))
    trace.append(trace_row('legacy','raw_yahoo_chart',None,None,level=1,notes='Missing original response; normalized CSV is not a replacement'))
    trace.append(trace_row('legacy','raw_yfinance_table',None,None,level=1,notes='Missing original table; never reconstruct a fake source'))
    legacy_paths={'normalized_market':DATA_DIR/'raw/NVDA.csv','stage3':DATA_DIR/'processed/NVDA.csv',
        'stage5':DATA_DIR/'research/NVDA_research.csv','stage61':DATA_DIR/'research/NVDA_features.csv',
        'stage8_labeled':DATA_DIR/'research/NVDA_labeled.csv'}
    for layer,p in legacy_paths.items():
        value=primary_csv_volume(p)
        trace.append(trace_row('legacy',layer,p,value,level=3,notes='Frozen normalized/downstream count, exact integer'))
        inventory.append(inventory_row('legacy',layer,p,role='Level 3 normalized/processed historical record'))
    # Record original downloader/config evidence without pretending it is a request log.
    original=subprocess.check_output(['git','show','HEAD:src/nasdaq_research/data.py'],cwd=PROJECT_ROOT).decode()
    code_path=OUT/'evidence/legacy_HEAD_data.py';write(code_path,original)
    inventory.append(inventory_row('legacy','original_downloader_code',code_path,role='Level 2 configured behavior only',status='PARTIAL',notes='Exact HEAD source; proves available code behavior, not historical request args or installed yfinance version'))
    for key in ['retrieval_timestamp','request_timestamp','query_start','query_end','endpoint','yfinance_version','HTTP_response_time']:
        metadata.append(metadata_row('legacy',key,None,status='NOT_AVAILABLE',notes='Not recoverable from frozen acquisition evidence; file mtime/Git time are not retrieval time'))
    for key,value in [('configured_period_default','1y'),('configured_interval_default','1d'),('configured_auto_adjust',False)]:
        metadata.append(metadata_row('legacy',key,value,status='PARTIAL',source='evidence/legacy_HEAD_data.py',notes='Configured default only; actual acquisition args not recorded'))
    metadata.append(metadata_row('legacy','observed_normalized_date_range','2025-10-01 through 2026-09-30',source='data/raw/NVDA.csv',notes='Observed rows, not query boundaries'))
    metadata.append(metadata_row('legacy','end_exclusive_semantics',None,status='NOT_AVAILABLE',notes='No explicit legacy start/end query or historic library version; current docs cannot certify it'))
    acquisition=json.loads((STAGE13/'raw/NVDA_acquisition.json').read_text())
    for name,digest in acquisition['raw_sha256'].items():
        if sha256(STAGE13/'raw'/name)!=digest:raise ValueError('Stage 13 frozen raw hash mismatch')
    charts=[]
    for p in sorted((STAGE13/'raw').glob('*chart*.json')):
        value,meta,context=chart_primary(p)
        inventory.append(inventory_row('stage13','original_yahoo_chart',p,role='Level 1 original vendor response',
            status='AVAILABLE' if value is not None else 'PARTIAL',notes='Contains primary bar' if value is not None else 'Auxiliary one-day response; primary bar absent'))
        if value is not None:charts.append((p,value,meta,context))
    if len(charts)!=1:raise ValueError('Exactly one archived full-window primary Yahoo response required')
    chart_path,raw_value,chart_meta,context=charts[0]
    trace.append(trace_row('expanded','raw_yahoo_chart',chart_path,raw_value,level=1,notes='Original JSON indicators.quote.volume indexed by primary-session timestamp'))
    for row in context:timezone_rows.append({'source_side':'stage13','source_file':str(chart_path.relative_to(PROJECT_ROOT)),**row})
    table_path=STAGE13/'raw/NVDA_yahoo_source.csv';table=source_table(table_path)
    rows=table.loc[table.index==pd.Timestamp(PRIMARY_DATE),('Volume','NVDA')]
    if len(rows)!=1:raise ValueError('Exactly one primary yfinance table row required')
    table_value=integer_volume(rows.iloc[0])
    trace.append(trace_row('expanded','raw_yfinance_table',table_path,table_value,level=1,notes='Original yfinance multi-index source table'))
    inventory.append(inventory_row('stage13','original_yfinance_table',table_path,role='Level 1 original yfinance representation'))
    expanded_paths={'normalized_market':STAGE13/'raw/NVDA_market.csv','stage3':STAGE13/'NVDA_stage13_market_history.csv',
        'stage5':STAGE13/'NVDA_stage13_pit_states.csv','stage61':STAGE13/'NVDA_stage13_research_matrix.csv'}
    for layer,p in expanded_paths.items():
        trace.append(trace_row('expanded',layer,p,primary_csv_volume(p),level=3,notes='Frozen Stage 13 equivalent processing layer'))
        inventory.append(inventory_row('stage13',layer,p,role='Level 3 frozen processed representation'))
    # Stage 13 did not persist a separate expanded labeled CSV. Preserve that distinction.
    audit_path=STAGE13/'NVDA_stage13_unexpected_overlap_differences.csv'
    audit=pd.read_csv(audit_path,dtype=str)
    match=audit.loc[audit.layer.eq('labeled_stage8') & audit.date.eq(PRIMARY_DATE) & audit.field.eq(PRIMARY_FIELD)]
    if len(match)!=1:raise ValueError('Expected frozen Stage 13 labeled-layer audit manifestation')
    trace.append(trace_row('expanded','stage8_labeled',audit_path,integer_volume(match.expanded_value.iloc[0]),level=3,
        notes='Indirect frozen audit of in-memory Stage 8 labeled join; no standalone expanded labeled artifact was persisted',representation='frozen_audit_of_in_memory_join'))
    inventory.append(inventory_row('stage13','expanded_stage8_labeled_file',None,role='Level 3 standalone labeled representation',notes='Not produced as a standalone Stage 13 artifact'))
    inventory.append(inventory_row('stage13','stage8_labeled_audit',audit_path,role='Level 3 indirect frozen labeled-join evidence',status='PARTIAL'))
    for kind,p,role in [('acquisition_manifest',STAGE13/'raw/NVDA_acquisition.json','Level 2 recorded query/adjustment policy; missing retrieval timestamp/version'),
                        ('execution_verification',STAGE13/'NVDA_stage13_execution_verification.json','Frozen execution/hash evidence, not acquisition timing'),
                        ('SEC_raw_copy',STAGE13/'raw/NVDA_companyfacts.json','Out of investigation scope; preservation only')]:
        inventory.append(inventory_row('stage13',kind,p,role=role,status='PARTIAL' if kind=='acquisition_manifest' else None))
    for key in ['retrieval_timestamp','request_timestamp','yfinance_version','HTTP_response_time','recorded_exact_endpoint']:
        metadata.append(metadata_row('stage13',key,None,status='NOT_AVAILABLE',notes='Omitted from frozen acquisition manifest; current installed version is not acquisition-time proof'))
    for key,value in [('query_start',acquisition['requested_start']),('query_end_exclusive',acquisition['requested_end_exclusive']),
                      ('auto_adjust',acquisition['auto_adjust']),('repair',acquisition['repair'])]:
        metadata.append(metadata_row('stage13',key,value,source='raw/NVDA_acquisition.json'))
    metadata.append(metadata_row('stage13','interval',chart_meta['dataGranularity'],source=str(chart_path.relative_to(PROJECT_ROOT))))
    for key,value in chart_meta.items():
        metadata.append(metadata_row('stage13',key,json.dumps(value,sort_keys=True) if isinstance(value,dict) else value,source=str(chart_path.relative_to(PROJECT_ROOT))))
    metadata.append(metadata_row('stage13','back_adjust',None,status='NOT_AVAILABLE',notes='Not explicitly passed or recorded; do not replace missing historical-version evidence with current defaults'))
    import yfinance as yf
    metadata.append(metadata_row('current_environment','yfinance_version',yf.__version__,notes='Observed now; not proof of either historical acquisition version'))
    # Narrow replay only the documented context rows of original yfinance table.
    small=table.loc[table.index.strftime('%Y-%m-%d').isin(['2026-09-29',PRIMARY_DATE,'2026-10-01'])]
    replay=standardize_history(small,symbol='NVDA')
    replay_value=integer_volume(replay.loc[replay.date.astype(str).eq(PRIMARY_DATE),'volume'].iloc[0])
    replay_check={'function':'nasdaq_research.data.standardize_history',
                  'exact_transformation':'pd.to_numeric Volume; no arithmetic on Volume; observed date normalization',
                  'primary_original_yfinance_volume':table_value,'normalized_replay_volume':replay_value,
                  'normalized_recorded_volume':primary_csv_volume(expanded_paths['normalized_market']),
                  'violations':int(replay_value!=raw_value or replay_value!=table_value or replay_value!=primary_csv_volume(expanded_paths['normalized_market']))}
    timezone_check={'available_chart_context_rows_checked':len(timezone_rows),
        'issues':sum(not r['UTC_and_Eastern_date_match'] for r in timezone_rows),
        'primary_bar_epoch':next(r['bar_epoch'] for r in timezone_rows if r['date']==PRIMARY_DATE),
        'primary_bar_UTC':next(r['bar_time_UTC'] for r in timezone_rows if r['date']==PRIMARY_DATE),
        'primary_bar_US_Eastern':next(r['bar_time_US_Eastern'] for r in timezone_rows if r['date']==PRIMARY_DATE),
        'legacy_timezone_evidence':'NOT_AVAILABLE','legacy_raw_session_identity_certified':False,
        'bar_timestamp_is_retrieval_time':False,'vendor_quote_timestamp_is_retrieval_time':False,
        'expanded_regularMarketTime_UTC':chart_meta.get('regularMarketTime_UTC')}
    df=pd.DataFrame(trace);df['value']=pd.array(df.value,dtype='Int64')
    divergence=first_divergence(df);prop=propagation(df)
    # A quote dated after the primary regular session supports that primary bar is
    # historical in this response; it does not recover legacy acquisition timing.
    quote=pd.Timestamp(chart_meta['regularMarketTime'],unit='s',tz='UTC')
    primary_close=pd.Timestamp(PRIMARY_DATE+' 16:00',tz='America/New_York').tz_convert('UTC')
    e=Evidence(expanded_raw_available=True,expanded_raw_value=raw_value,
        expanded_processing_faithful=replay_check['violations']==0 and prop['sides']['expanded']['propagation_violations']==0,
        legacy_processing_faithful=False,expanded_complete_session_proven=quote>primary_close,
        unresolved_partial_session_possibility=True)
    frame_write('source_inventory',inventory)
    frame_write('processing_trace',df)
    frame_write('acquisition_metadata',metadata)
    frame_write('timezone_session_context',timezone_rows)
    write(output_path('first_divergence'),json_text(divergence))
    local={'local_trace_completed_before_external_acquisition':True,'searched_file_count':len(files),
        'legacy_original_chart_count':len(legacy_charts),'expanded_original_chart_count':len(charts),
        'legacy_original_yfinance_table_available':False,'legacy_acquisition_timing_available':False,
        'legacy_source_evidence_status':'NOT_AVAILABLE','Stage13_acquisition_metadata_status':'PARTIAL',
        'evidence_facts':asdict(e),'first_divergence':divergence,'propagation':prop,
        'normalized_replay':replay_check,'timezone_audit':timezone_check,'protected':protected,
        'remaining_hypotheses':['Unrecorded legacy acquisition before session/source finalization',
                                'Historical vendor/source-vintage change',
                                'Unrecoverable pre-normalized source/session/cache difference'],
        'missing_required_evidence':['Legacy original Yahoo HTTP response or yfinance raw table',
                                     'Legacy reliable request/retrieval timestamp and exact query/session parameters',
                                     'Historical acquisition library version/endpoint evidence'],**SAFETY}
    # Freeze the local inventory/search scope; repeated runs must not absorb new runtime files.
    write(output_path('local_evidence'),json_text(local))
    return local


def acquire_current() -> dict:
    """One narrow current-vendor acquisition after frozen local trace completion."""
    register()
    if not output_path('local_evidence').exists():
        raise ValueError('Complete and save local frozen evidence trace before networking')
    protected=verify_protected()
    if protected['mutation_violations']:raise ValueError('Protected source changed before acquisition')
    folder=OUT/'external';folder.mkdir(exist_ok=True)
    manifest=folder/'NVDA_current_acquisition.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        for name,digest in existing['sha256'].items():
            if sha256(folder/name)!=digest:raise ValueError('Frozen current evidence hash mismatch')
        return existing
    table_path=folder/'NVDA_current_yfinance.csv';normalized_path=folder/'NVDA_current_normalized.csv'
    if table_path.exists() or normalized_path.exists():
        raise ValueError('Incomplete external acquisition requires inspection; no overwriting')
    from curl_cffi.requests import Session
    import yfinance as yf
    from urllib.parse import urlsplit
    yf.set_tz_cache_location('/tmp/nasdaq-stage13-1-yfinance-cache')
    chart_records=[]
    class ArchiveSession(Session):
        def request(self,method,url,*args,**kwargs):
            started=datetime.now(timezone.utc).isoformat()
            response=super().request(method,url,*args,**kwargs)
            received=datetime.now(timezone.utc).isoformat()
            parsed=urlsplit(str(url))
            if parsed.path.endswith('/v8/finance/chart/NVDA') and response.status_code==200:
                body=response.content;digest=hashlib.sha256(body).hexdigest()
                path=folder/f'NVDA_current_chart_{digest}.json'
                if not path.exists():path.write_bytes(body)
                safe_keys={'period1','period2','interval','range','events','includePrePost','includeAdjustedClose'}
                params={k:v for k,v in (kwargs.get('params') or {}).items() if k in safe_keys}
                chart_records.append({'file':path.name,'sha256':digest,'method':str(method),
                    'endpoint':f'{parsed.scheme}://{parsed.netloc}{parsed.path}',
                    'request_parameters':params,'request_started_UTC':started,'response_received_UTC':received,
                    'HTTP_server_Date':response.headers.get('Date'),'HTTP_status':response.status_code})
            return response
    started=datetime.now(timezone.utc).isoformat()
    with ArchiveSession(impersonate='chrome') as session:
        market=download_stock_history('NVDA',start='2026-09-29',end='2026-10-02',
            interval='1d',session=session,source_snapshot_path=table_path)
    market.to_csv(normalized_path,index=False)
    if not chart_records:raise ValueError('Original chart response not archived; current evidence incomplete')
    paths=[table_path,normalized_path,*sorted(folder.glob('NVDA_current_chart_*.json'))]
    result={'evidence_role':'Level 4 current vendor state only; never a replacement for legacy frozen source',
        'acquisition_started_UTC':started,'acquisition_finished_UTC':datetime.now(timezone.utc).isoformat(),
        'query':{'ticker':'NVDA','start':'2026-09-29','end_exclusive':'2026-10-02','interval':'1d'},
        'yfinance_version':yf.__version__,'auto_adjust':False,'back_adjust':False,'repair':False,
        'back_adjust_evidence':'Current installed yf.download signature default; not a claim about legacy library',
        'threads':False,'session_impersonate':'chrome','progress':False,'actions':True,'keepna':True,
        'local_trace_completed_before_external_acquisition':True,
        'local_evidence_sha256':sha256(output_path('local_evidence')),
        'chart_requests':chart_records,'sha256':{p.name:sha256(p) for p in paths},
        'frozen_once':True,'cookies_or_crumb_archived':False}
    write(manifest,json_text(result))
    return result


def local_bundle() -> dict:
    """Freeze local tables separately before formal reports add external evidence."""
    path=OUT/'evidence/NVDA_local_trace_bundle.json'
    if path.exists():return json.loads(path.read_text())
    if not output_path('local_evidence').exists():discover_local()
    bundle={'local_evidence_sha256':sha256(output_path('local_evidence')),
        'source_inventory':pd.read_csv(output_path('source_inventory','.csv')).to_dict('records'),
        'processing_trace':pd.read_csv(output_path('processing_trace','.csv')).to_dict('records'),
        'acquisition_metadata':pd.read_csv(output_path('acquisition_metadata','.csv')).to_dict('records'),
        'timezone_session_context':pd.read_csv(output_path('timezone_session_context','.csv')).to_dict('records')}
    write(path,json_text(bundle));return json.loads(path.read_text())


def external_evidence() -> dict:
    path=OUT/'external/NVDA_current_acquisition.json'
    if not path.exists():return {'available':False,'rows':[],'inventory':[],'metadata':[],'timezone_context':[],'checks':{}}
    manifest=json.loads(path.read_text());folder=path.parent
    for name,digest in manifest['sha256'].items():
        if sha256(folder/name)!=digest:raise ValueError('Frozen external source changed')
    if manifest['local_evidence_sha256']!=sha256(output_path('local_evidence')):
        raise ValueError('Original pre-external local evidence changed')
    rows=[];inventory=[];metadata=[];timezone_context=[];values=[]
    for req in manifest['chart_requests']:
        p=folder/req['file'];value,meta,context=chart_primary(p)
        inventory.append(inventory_row('current_external','original_yahoo_chart',p,role='Level 4 current-vendor evidence only',
            status='AVAILABLE' if value is not None else 'PARTIAL',notes='Frozen once; no substitute for missing legacy source'))
        inventory[-1]['acquisition_time_if_known']=req['response_received_UTC']
        inventory[-1]['query_if_known']=json.dumps(req['request_parameters'],sort_keys=True)
        for r in context:timezone_context.append({'source_side':'current_external','source_file':str(p.relative_to(PROJECT_ROOT)),**r})
        if value is not None:
            values.append(value)
            rows.append({'source':'current_external','layer':'raw_yahoo_chart','date':PRIMARY_DATE,'field':PRIMARY_FIELD,
                'value':value,'value_type':'integer_count','retrieval_time':req['response_received_UTC'],
                'raw_or_processed':'original_HTTP_response','source_file':str(p.relative_to(PROJECT_ROOT)),
                'sha256':sha256(p),'source_authority_level':4,'available':True})
    if len(values)!=1:raise ValueError('Expected one current primary-window chart')
    for layer,p in [('raw_yfinance_table',folder/'NVDA_current_yfinance.csv'),('normalized_market',folder/'NVDA_current_normalized.csv')]:
        if layer=='raw_yfinance_table':
            table=source_table(p);value=integer_volume(table.loc[pd.Timestamp(PRIMARY_DATE),('Volume','NVDA')])
        else:value=primary_csv_volume(p)
        values.append(value)
        inventory.append(inventory_row('current_external',layer,p,role='Level 4 current representation; corroboration only'))
        inventory[-1]['acquisition_time_if_known']=manifest['acquisition_finished_UTC']
        rows.append({'source':'current_external','layer':layer,'date':PRIMARY_DATE,'field':PRIMARY_FIELD,
            'value':value,'value_type':'integer_count','retrieval_time':manifest['acquisition_finished_UTC'],
            'raw_or_processed':'raw_yfinance' if layer=='raw_yfinance_table' else 'normalized',
            'source_file':str(p.relative_to(PROJECT_ROOT)),'sha256':sha256(p),'source_authority_level':4,'available':True})
    inventory.append(inventory_row('current_external','acquisition_manifest',path,role='Level 4 retrieval/query/library metadata'))
    inventory[-1]['acquisition_time_if_known']=manifest['acquisition_finished_UTC']
    for k in ('acquisition_started_UTC','acquisition_finished_UTC','yfinance_version','auto_adjust','back_adjust','repair','actions','keepna'):
        metadata.append(metadata_row('current_external',k,manifest[k],source=str(path.relative_to(PROJECT_ROOT))))
    for k,value in manifest['query'].items():metadata.append(metadata_row('current_external',k,value,source=str(path.relative_to(PROJECT_ROOT))))
    for req in manifest['chart_requests']:
        if 'period1' not in req['request_parameters']:continue
        metadata.append(metadata_row('current_external','endpoint',req['endpoint'],source=str(path.relative_to(PROJECT_ROOT))))
        metadata.append(metadata_row('current_external','exact_request_parameters',json.dumps(req['request_parameters'],sort_keys=True),source=str(path.relative_to(PROJECT_ROOT))))
    return {'available':True,'rows':rows,'inventory':inventory,'metadata':metadata,'timezone_context':timezone_context,
        'primary_volume':values[0], 'retrieval_timestamp':manifest['acquisition_finished_UTC'],
        'checks':{'raw_yfinance_normalized_agree':len(set(values))==1,
                  'primary_UTC_Eastern_session_agree':all(r['UTC_and_Eastern_date_match'] for r in timezone_context),
                  'frozen_raw_hashes_valid':True,'local_first_order_proven':manifest['local_trace_completed_before_external_acquisition']},
        'manifest_sha256':sha256(path)}


def root_evidence_rows(evidence: Evidence, root_cause: str, current: dict) -> list[dict]:
    support={
        ROOTS[0]:('NOT_SUPPORTED','Expanded original raw/table/normalized all preserve 121732200; legacy normalized/downstream all preserve 121269300. No exact discrepancy-causing processing bug found; missing legacy original source prevents complete historical normalization certification.'),
        ROOTS[1]:('NOT_ESTABLISHED','Legacy retrieval/request/raw-source timing unavailable. Lower legacy Volume and daily-bar open timestamp are not proof of incomplete acquisition.'),
        ROOTS[2]:('NOT_ESTABLISHED','Current Yahoo matches expanded value, but original legacy source unavailable and partial-session possibility unresolved; cannot certify cross-vintage causal revision.'),
        ROOTS[3]:('SUPPORTED_FALLBACK','Missing original legacy source plus reliable acquisition timing leaves source-vintage vs incomplete-session/pre-normalized source explanations unresolved.')}
    return [{'root_cause_candidate':k,'supported_as_final_classification':k==root_cause,
        'evidence_status':support[k][0],'evidence':support[k][1],
        'current_Yahoo_corroboration':current.get('primary_volume') if k==ROOTS[2] else None,
        'legacy_original_source_available':evidence.legacy_raw_available,
        'legacy_acquisition_timing_known':evidence.legacy_timing_known} for k in ROOTS]


def validate_trace(trace: pd.DataFrame) -> dict:
    schema=set(['side','layer','source_file','source_hash','date','field','value','available','notes'])<=set(trace)
    exactly_all=all(len(trace.loc[(trace.side==side)&(trace.layer==layer)])==1 for side in ('legacy','expanded') for layer in LAYERS)
    hashes=True;source_values=True
    for _,r in trace.iterrows():
        if not bool(r.available):
            if pd.notna(r.value):source_values=False
            continue
        p=PROJECT_ROOT/r.source_file
        hashes &= p.is_file() and sha256(p)==r.source_hash
        if r.layer=='raw_yahoo_chart':expected=chart_primary(p)[0]
        elif r.layer=='raw_yfinance_table':expected=integer_volume(source_table(p).loc[pd.Timestamp(PRIMARY_DATE),('Volume','NVDA')])
        elif r.representation=='frozen_audit_of_in_memory_join':
            a=pd.read_csv(p,dtype=str)
            selected=a.loc[a.layer.eq('labeled_stage8') & a.date.eq(PRIMARY_DATE) & a.field.eq(PRIMARY_FIELD)]
            expected=integer_volume(selected.expanded_value.iloc[0])
        else:expected=primary_csv_volume(p)
        source_values &= integer_volume(r.value)==expected
    return {'required_schema':schema,'all_layers_including_explicit_missing':exactly_all,
            'available_source_hashes_valid':bool(hashes),'available_source_values_exact':bool(source_values),
            'primary_scope_only':bool(trace.date.eq(PRIMARY_DATE).all() and trace.field.eq(PRIMARY_FIELD).all()),
            'missing_original_raw_not_fabricated':not trace.loc[trace.side.eq('legacy') & trace.layer.isin(LAYERS[:2]),'available'].any()}


def enrich_inventory(rows: list[dict]) -> list[dict]:
    """Attach observed file ranges and recorded queries, never infer retrieval time."""
    result=[]
    stage13_query=json.loads((STAGE13/'raw/NVDA_acquisition.json').read_text())
    current_path=OUT/'external/NVDA_current_acquisition.json'
    current=json.loads(current_path.read_text()) if current_path.exists() else None
    for original in rows:
        row=original.copy();name=row.get('file_path')
        if not name or pd.isna(name):
            result.append(row);continue
        path=PROJECT_ROOT/name
        if path.suffix=='.csv':
            if row['source_type'] in ('original_yfinance_table','raw_yfinance_table'):
                dates=source_table(path).index.strftime('%Y-%m-%d')
            else:
                header=pd.read_csv(path,nrows=0).columns
                dates=pd.read_csv(path,usecols=['date'],dtype=str).date if 'date' in header else []
            if len(dates):row['date_range_if_known']=f'{min(dates)} through {max(dates)}'
        elif row['source_type']=='original_yahoo_chart':
            payload=json.loads(path.read_bytes())['chart']['result'][0]
            stamps=payload.get('timestamp',[])
            if stamps:
                endpoints=[pd.Timestamp(t,unit='s',tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d') for t in (stamps[0],stamps[-1])]
                row['date_range_if_known']=' through '.join(endpoints)
        if row['source_side']=='stage13' and row['source_type'] in ('normalized_market','original_yfinance_table','acquisition_manifest'):
            row['query_if_known']=json.dumps({'requested_start':stage13_query['requested_start'],
                'requested_end_exclusive':stage13_query['requested_end_exclusive'],
                'exact_original_HTTP_request_not_recorded':True},sort_keys=True)
        if row['source_side']=='current_external' and current and row['source_type']!='original_yahoo_chart':
            row['query_if_known']=json.dumps(current['query'],sort_keys=True)
        result.append(row)
    return result


def run_pipeline() -> dict:
    register()
    protected=verify_protected()
    if protected['mutation_violations']:raise ValueError('Protected Stage 1–13 files changed')
    bundle=local_bundle();local=json.loads(output_path('local_evidence').read_text())
    if bundle['local_evidence_sha256']!=sha256(output_path('local_evidence')):
        raise ValueError('Frozen local evidence hash changed')
    trace=pd.DataFrame(bundle['processing_trace']);trace['value']=pd.array(trace.value,dtype='Int64')
    divergence=first_divergence(trace);prop=propagation(trace);trace_checks=validate_trace(trace)
    current=external_evidence();evidence=Evidence(**local['evidence_facts'])
    root=classify(evidence)
    # Current-vendor agreement never changes missing historical causal facts.
    inventory=list(bundle['source_inventory'])+current['inventory']
    inventory.append(inventory_row('third_party','not_used',None,role='Corroboration only',notes='Not needed: a third-party count cannot recover missing legacy source/timing or resolve the causal alternatives'))
    inventory=enrich_inventory(inventory)
    comparisons=[]
    for _,r in trace.iterrows():
        comparisons.append({'source':r.side,'layer':r.layer,'date':r.date,'field':r.field,
            'value':integer_volume(r.value) if bool(r.available) else None,
            'value_type':'integer_count' if bool(r.available) else 'unavailable','retrieval_time':None,
            'raw_or_processed':'original_source' if r.layer in LAYERS[:2] else r.representation,
            'source_file':r.source_file,'sha256':r.source_hash,'source_authority_level':r.authority_level,'available':bool(r.available)})
    comparisons.extend(current['rows'])
    timezone_context=list(bundle['timezone_session_context'])+current['timezone_context']
    engineering={**trace_checks,
        'propagation_consistent':prop['propagation_consistency_violations']==0,
        'narrow_expanded_normalization_replay':local['normalized_replay']['violations']==0,
        'available_timezone_context_consistent':all(r['UTC_and_Eastern_date_match'] for r in timezone_context),
        'root_category_registered':root in ROOTS,
        'unresolved_fallback_when_missing_legacy_source':root==ROOTS[3] if not evidence.legacy_raw_available else True,
        'original_stage13_inputs_preserved':protected['stage13_mutation_violations']==0,
        'legacy_inputs_preserved':protected['stage1_12_mutation_violations']==0,
        'final_test_not_accessed_for_performance':True,'no_modeling_or_feature_target_split_changes':True,
        **{f'current_{k}':v for k,v in current['checks'].items()}}
    policy=canonical_policy(root,corrected_and_revalidated=evidence.corrected_and_revalidated,engineering_passed=all(engineering.values()))
    policy['frozen_stage13_snapshot_candidate']=str((STAGE13/'raw/NVDA_market.csv').relative_to(PROJECT_ROOT))
    policy['candidate_sha256']=sha256(STAGE13/'raw/NVDA_market.csv')
    policy['candidate_authority_approved']=policy['expanded_dataset_approved_for_downstream_research']
    summary={'stage_name':protocol()['stage_name'],'ticker':'NVDA','primary_date':PRIMARY_DATE,'primary_field':PRIMARY_FIELD,
        'legacy_normalized_value':LEGACY_VALUE,'expanded_original_source_value':EXPANDED_VALUE,'difference':462900,
        'current_yahoo_value':current.get('primary_volume'),'external_evidence_available':current['available'],
        'root_cause':root,'completion_status':policy['completion_status'],
        'first_divergence_layer':divergence['first_divergence_layer'],
        'first_observed_divergence_layer':divergence['first_observed_divergence_layer'],
        'causal_first_divergence_identified':divergence['causal_first_divergence_identified'],
        'observed_independent_discrepancy_count':1,'resolved_independent_discrepancy_count':0 if root==ROOTS[3] else 1,
        'original_stage13_overlap_violations':5,'original_manifestations_including_normalized_layer':5,
        'additional_downstream_manifestations':4,'propagation':prop,
        'canonical_policy_frozen':True,'future_stage14_plus_market_data_authority':policy['future_stage14_plus_market_data_authority'],
        'expanded_dataset_approved_for_downstream_research':policy['expanded_dataset_approved_for_downstream_research'],
        'stage14_modeling_eligible':policy['stage14_modeling_eligible'],
        'Stage13_historical_status':'INCOMPLETE_OR_BLOCKED','evidence_gaps':local['missing_required_evidence'],
        'remaining_hypotheses':local['remaining_hypotheses'],'engineering_checks_passed':all(engineering.values()),
        'protected':protected,**SAFETY}
    tables={'source_inventory':inventory,'source_comparison':comparisons,
        'acquisition_metadata':list(bundle['acquisition_metadata'])+current['metadata'],
        'processing_trace':trace,'root_cause_evidence':root_evidence_rows(evidence,root,current),
        'timezone_session_context':timezone_context}
    roundtrips={};hashes={}
    for name,data in tables.items():
        path=frame_write(name,data);text=path.read_text()
        lexical=pd.read_csv(StringIO(text),dtype=str,keep_default_na=False)
        roundtrips[name]=lexical.to_csv(index=False,lineterminator='\n')==text
        hashes[path.name]=sha256(path)
    write(output_path('first_divergence'),json_text(divergence));hashes[output_path('first_divergence').name]=sha256(output_path('first_divergence'))
    write(output_path('canonical_snapshot_policy'),json_text(policy));hashes[output_path('canonical_snapshot_policy').name]=sha256(output_path('canonical_snapshot_policy'))
    write(output_path('summary'),json_text(summary));hashes[output_path('summary').name]=sha256(output_path('summary'))
    engineering['deterministic_CSV_roundtrip']=all(roundtrips.values())
    validation={'valid':all(engineering.values()) and root!=ROOTS[3] and policy['expanded_dataset_approved_for_downstream_research'],
        'investigation_engineering_checks_passed':all(engineering.values()),
        'root_cause_resolved':root!=ROOTS[3],'root_cause':root,'completion_status':policy['completion_status'],
        'checks':engineering,'trace':trace_checks,'propagation':prop,'first_divergence':divergence,
        'normalized_replay':local['normalized_replay'],'timezone_audit':local['timezone_audit'],
        'evidence_facts':asdict(evidence),'current_evidence_checks':current['checks'],
        'CSV_roundtrips':roundtrips,'formal_generated_sha256':hashes,
        'protected':verify_protected(),'protocol_sha256':sha256(output_path('protocol')),
        'local_bundle_sha256':sha256(OUT/'evidence/NVDA_local_trace_bundle.json'),
        'current_acquisition_manifest_sha256':current.get('manifest_sha256'),
        'expanded_dataset_approved_for_downstream_research':policy['expanded_dataset_approved_for_downstream_research'],
        'stage14_modeling_eligible':policy['stage14_modeling_eligible'],**SAFETY}
    write(output_path('validation'),json_text(validation))
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--discover-local',action='store_true')
    parser.add_argument('--acquire-current',action='store_true')
    args=parser.parse_args()
    if args.register_only:register();result={'protocol_registered':True}
    elif args.discover_local:
        if output_path('local_evidence').exists():raise ValueError('Local evidence is already frozen; formal runs reuse it')
        result=discover_local()
    elif args.acquire_current:result=acquire_current()
    else:result=run_pipeline()
    print(json_text({k:result[k] for k in ['protocol_registered','root_cause','completion_status',
        'first_divergence_layer','first_observed_divergence_layer','expanded_dataset_approved_for_downstream_research',
        'stage14_modeling_eligible'] if k in result}))


if __name__=='__main__':
    main()
