"""Stage 13.2: offline, prospective approval of one preserved market vintage.

No acquisition or model code is executed. Existing builders run only on isolated
copies; all writes belong to the new Stage 13.2 namespace. Historical decisions
and the unresolved causal discrepancy are evidence, not decisions to rewrite.
"""

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research import historical_expansion as h
from nasdaq_research import research_features as rf
from nasdaq_research.config import PROJECT_ROOT, DATA_DIR
from nasdaq_research.data import REQUIRED_COLUMNS, standardize_history
from nasdaq_research.dates import date_columns, canonical_dates
from nasdaq_research.targets import build_targets

OUTPUT_DIR = DATA_DIR / 'research/historical_expansion/stage13_2_canonical_authority'
PREFIX = 'NVDA_stage13_2_'
VINTAGE_ID = 'NVDA_YAHOO_STAGE13_2020_2026_V1'
PROTOCOL_SHA256 = 'e15de0f4de24783ead998fa2ba14ea3648ccd36579a75f40db15333c2e00e9bc'
UPSTREAM_MANIFEST_SHA256 = '441a2544eef7c2d9f3ecb8bb7dfa435a9eb3af9021d049ab3a7be0fc44ba4966'
APPROVED = 'APPROVED_CANONICAL_VINTAGE'
QUALIFIED = 'APPROVED_WITH_DOCUMENTED_NONMATERIAL_CROSS_VINTAGE_DIFFERENCE'
REJECTED = 'NOT_APPROVED_FOR_DOWNSTREAM_RESEARCH'
MATERIAL = 'RESEARCH_SEMANTIC_MATERIAL'
NONMATERIAL = 'PROVENANCE_MATERIAL_MODEL_SEMANTIC_NONMATERIAL'
UNKNOWN = 'UNKNOWN_MATERIALITY'
LOCKED = ('2026-07-15', '2026-09-23')
SAFETY = {**h.SAFETY, 'modeling_allowed': False, 'network_allowed': False,
          'network_requests': 0, 'stage14_designed': False, 'stage14_executed': False,
          'legacy_root_cause_reinvestigated': False, 'predictive_performance_read': False}
LIMITS = {
    'volume_field_approved_for_current_modeling': False,
    'volume_dependent_future_research_requires_revalidation': True,
    'volume_dependent_feature_research_requires_new_field_validation': True,
}
SCOPE = {
    'ticker': 'NVDA', 'research_window': ['2020-01-01', '2026-09-30'],
    'market_features': list(h.MARKET_FEATURES),
    'relative_market_features': list(h.RELATIVE_FEATURES),
    'fundamental_features': list(rf.FEATURE_COLUMNS),
    'targets': list(h.RETURN_COLUMNS), 'primary_target': 'forward_return_5d',
    'prospective_direction': ['Relative Market OLS', 'inherited Ridge alpha=1',
                              'zero_return', 'historical_mean'],
    'stage14_design_or_execution_authorized': False,
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return str(Path(path).relative_to(PROJECT_ROOT))


def verify_upstream():
    """Check the preregistered file set, including untracked/ignored old data."""
    path = OUTPUT_DIR / (PREFIX + 'upstream_sha256.json')
    if h.sha256(path) != UPSTREAM_MANIFEST_SHA256:
        raise ValueError('Preregistered upstream manifest changed')
    manifest = load_json(path)
    changed = [name for name, digest in manifest.items()
               if not (PROJECT_ROOT / name).is_file()
               or h.sha256(PROJECT_ROOT / name) != digest]
    return {'files_checked': len(manifest), 'mutation_violations': len(changed),
            'changed_files': changed}


@contextmanager
def offline_only():
    """Fail on transport/DNS attempts rather than relying on missing credentials."""
    def forbidden(*args, **kwargs):
        raise RuntimeError('Stage 13.2 is offline; network access forbidden')
    with patch.object(socket.socket, 'connect', forbidden), \
         patch.object(socket.socket, 'connect_ex', forbidden), \
         patch.object(socket, 'create_connection', forbidden), \
         patch.object(socket, 'getaddrinfo', forbidden), \
         patch.object(h, 'acquire', forbidden), \
         patch.object(h, 'download_stock_history', forbidden):
        yield


class DependencyUnknown(ValueError):
    """A source expression cannot be certified by this restricted AST tracer."""


class FormulaTracer:
    """Trace published Series expressions back through aliases to raw fields.

    This deliberately accepts the small grammar actually used by the registered
    price/target builders. Unresolved subscripts, loops or Series expressions
    fail closed. DataFrame pass-through and validation are audited separately.
    """

    def __init__(self, frames, inputs, graph=None):
        self.frames = set(frames)
        self.inputs = set(inputs)
        self.graph = dict(graph or {})
        self.aliases = {}
        self.literals = {}

    def literal(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.literals:
            return self.literals[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.literal(n) for n in node.elts]
        if isinstance(node, ast.JoinedStr):
            return ''.join(str(self.literal(n.value)) if isinstance(n, ast.FormattedValue)
                           else str(self.literal(n)) for n in node.values)
        raise DependencyUnknown('Unresolved column/loop literal: ' + ast.unparse(node))

    def terms(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.aliases:
                return set(self.aliases[node.id])
            if node.id in self.literals or node.id in ('np','float','int'):
                return set()
            raise DependencyUnknown('Unresolved expression symbol: ' + node.id)
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.frames:
                if node.attr not in self.inputs and node.attr not in self.graph:
                    raise DependencyUnknown('Unregistered frame field: ' + node.attr)
                return {node.attr}
            return self.terms(node.value)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.frames:
                field = self.literal(node.slice)
                if not isinstance(field, str) or field not in self.inputs | set(self.graph):
                    raise DependencyUnknown('Unregistered frame selection: ' + str(field))
                return {field}
            return self.terms(node.value) | self.terms(node.slice)
        if isinstance(node, ast.Call):
            return (self.terms(node.func)
                    | set().union(*(self.terms(n) for n in node.args))
                    | set().union(*(self.terms(k.value) for k in node.keywords)))
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            return set().union(*(self.terms(n) for n in ast.iter_child_nodes(node)))
        if isinstance(node, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
            return set()
        raise DependencyUnknown('Unsupported Series expression: ' + ast.unparse(node))

    def expand(self, terms, seen=()):
        raw = set()
        for term in terms:
            if term in self.inputs:
                raw.add(term)
            elif term in self.graph and term not in seen:
                raw |= self.expand(self.graph[term]['terms'], (*seen, term))
            else:
                raise DependencyUnknown('Unresolved or cyclic dependency: ' + term)
        return raw

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            # The caller excludes frame construction/validation assignments.
            self.aliases[target.id] = self.terms(value)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                and target.value.id in self.frames:
            name = self.literal(target.slice)
            terms = self.terms(value)
            self.graph[name] = {'terms': sorted(terms), 'raw_fields': sorted(self.expand(terms)),
                                'expression': ast.unparse(value), 'line': value.lineno}
        else:
            raise DependencyUnknown('Unsupported formula assignment')

    def visit(self, statements):
        for node in statements:
            if isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name):
                    raise DependencyUnknown('Unsupported loop target')
                for value in self.literal(node.iter):
                    self.literals[node.target.id] = value
                    self.visit(node.body)
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise DependencyUnknown('Multiple assignment targets')
                self.assign(node.targets[0], node.value)
            else:
                raise DependencyUnknown('Unsupported formula statement')


def source_functions(module_name, names):
    path = PROJECT_ROOT / 'src/nasdaq_research' / (module_name + '.py')
    text = path.read_text()
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    return path, functions, [
        {'source': relative(path), 'source_sha256': h.sha256(path), 'function': name,
         'line': functions[name].lineno,
         'code': ast.get_source_segment(text, functions[name])}
        for name in names]


def dependency_evidence(daily):
    """Extract actual formulas/constants and preserve integration call evidence."""
    fp, funcs, evidence = source_functions('features', ['build_features', '_validate_input', 'validate_features'])
    tracer = FormulaTracer(['result'], REQUIRED_COLUMNS)
    body = funcs['build_features'].body
    # Isolate the computation interval from the actual close alias to quality.
    begin = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'close')
    end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'quality')
    tracer.visit(body[begin:end])
    if set(tracer.graph) != set(h.MARKET_FEATURES):
        raise DependencyUnknown('Computed market feature set differs from registered whitelist')
    hp, hf, ev = source_functions('historical_expansion', ['construct_features', 'legacy_stage5_frame', 'research_slice'])
    evidence += ev
    relative_tracer = FormulaTracer(['daily'], REQUIRED_COLUMNS, tracer.graph)
    relative_tracer.visit([n for n in hf['construct_features'].body if isinstance(n,ast.For)])
    if set(relative_tracer.graph) != set(h.MARKET_FEATURES) | set(h.RELATIVE_FEATURES):
        raise DependencyUnknown('Relative feature computation cannot be resolved')
    tp, tf, ev = source_functions('targets', ['build_targets', '_check_features', 'invalid_price_count'])
    evidence += ev
    targets = FormulaTracer(['features', 'targets'], REQUIRED_COLUMNS)
    target_body = tf['build_targets'].body
    # Trace entry, exit_price, returns and result[...] directly, not a prompt formula.
    for node in target_body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == 'entry':
            targets.visit([node])
        elif isinstance(node, ast.For):
            for value in h.HORIZONS:
                targets.literals[node.target.id] = value
                formula_statements = []
                for statement in node.body:
                    formula_statements.extend(statement.body if isinstance(statement,ast.With) else [statement])
                for sub in formula_statements:
                    if isinstance(sub, ast.Assign):
                        target = sub.targets[0]
                        if isinstance(target, ast.Name) and target.id in ('exit_price','returns'):
                            targets.visit([sub])
                        elif isinstance(target, ast.Subscript):
                            name = targets.literal(target.slice)
                            if name.startswith(('forward_return_', 'target_exit_close_')):
                                targets.visit([sub])
    if not set(h.RETURN_COLUMNS) <= set(targets.graph):
        raise DependencyUnknown('Target formulas not fully traced')
    for module, names in [
        ('data',['standardize_history','_flatten_yfinance_columns']),
        ('alignment',['_market','effective_dates','_observations','_join','build_research_dataset']),
        ('research_features',['build_fundamental_features','_reference','_divide',
                              '_market_from_research','build_feature_matrix','_effective_semantics'])]:
        _, _, ev = source_functions(module, names)
        evidence += ev
    policy_path = DATA_DIR / 'research/modeling/stage11_market_representation/NVDA_stage11_protocol.json'
    policy = load_json(policy_path)
    expected_treatment = [c.replace('sma_', 'close_to_sma_') if c.startswith('sma_') else c
                          for c in h.MARKET_FEATURES]
    protocol_match = (policy['control_features'] == list(h.MARKET_FEATURES)
                      and policy['treatment_features'] == expected_treatment
                      and policy['primary_target'] == h.PRIMARY_TARGET
                      and policy['ridge_alpha'] == 1.0
                      and all(policy['formulas'][f'close_to_sma_{k}'] == f'Close / sma_{k} - 1'
                              for k in (5,20,60)))
    if not protocol_match:
        raise DependencyUnknown('Authoritative representation/target protocol mismatch')
    graph = relative_tracer.graph
    rows = []

    def add(obj, kind, fields, direct, expression, path, function, line, indirect=None):
        for field in sorted(set(fields) | {'volume'}):
            used = field in fields
            rows.append({'research_object': obj, 'dependency_type': kind,
                         'source_field': field, 'used_directly': used and field in direct,
                         'used_indirectly': used and (field in indirect if indirect is not None else field not in direct),
                         'description': expression, 'source_code': relative(path),
                         'function': function, 'line': line, 'source_sha256': h.sha256(path)})

    for name in (*h.MARKET_FEATURES, *h.RELATIVE_FEATURES):
        node = graph[name]
        path = hp if name in h.RELATIVE_FEATURES else fp
        add(name, 'relative_market' if name in h.RELATIVE_FEATURES else 'market_feature',
            node['raw_fields'], set(node['terms']) & set(REQUIRED_COLUMNS),
            node['expression'], path, 'construct_features' if path == hp else 'build_features', node['line'],
            indirect=relative_tracer.expand(set(node['terms'])-set(REQUIRED_COLUMNS)))
    for name in h.RETURN_COLUMNS:
        node = targets.graph[name]
        # Aliases entry/exit_price refer directly to shifted source prices.
        add(name, 'primary_target' if name == h.PRIMARY_TARGET else 'secondary_target',
            node['raw_fields'], node['raw_fields'], node['expression'], tp, 'build_targets', node['line'])
    add('target_entry_exit_provenance', 'target_provenance', {'date','open','close'},
        {'date','open','close'}, 'Observed session shifts; entry Open[t+1]; exit Close[t+h]',
        tp, 'build_targets', tf['build_targets'].lineno)
    rp, rfuncs, _ = source_functions('research_features', ['build_fundamental_features', 'build_feature_matrix'])
    # Source-owned dictionaries are extracted by literal_eval from the AST.
    rt = ast.parse(rp.read_text())
    mappings = {n.targets[0].id: ast.literal_eval(n.value) for n in rt.body
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in ('RATIO_FIELDS','BALANCE_RATIOS','GROWTH_FIELDS')}
    fundamentals = {}
    for prefix in ('q','fy'):
        for name, field in mappings['RATIO_FIELDS'].items():
            fundamentals[f'{prefix}_{name}'] = ({'SEC.'+field,'SEC.revenue'}, 'row[numerator] / row.revenue')
        for name, field in mappings['GROWTH_FIELDS'].items():
            fundamentals[f'{prefix}_{name}_growth_yoy'] = ({'SEC.'+field}, 'row[field] / reference[field] - 1; same basis/FY-1; reference filed <= current filing')
    for name, fields in mappings['BALANCE_RATIOS'].items():
        fundamentals[name] = ({'SEC.'+f for f in fields}, 'row[numerator] / row[denominator]')
    if set(fundamentals) != set(rf.FEATURE_COLUMNS):
        raise DependencyUnknown('Fundamental feature definitions do not cover the current whitelist')
    for name, (fields, formula) in fundamentals.items():
        add(name, 'PIT_fundamental_feature', fields | {'date','SEC.filing_date','SEC.period_identity'},
            fields, formula + '; attached by whole-observation q/fy/bs PIT selection',
            rp, 'build_fundamental_features', rfuncs['build_fundamental_features'].lineno)
    pit_columns = fundamental_state_columns(daily)
    for name in pit_columns:
        if name not in rf.FEATURE_COLUMNS:
            add('PIT:' + name, 'PIT_state_provenance', {'date','SEC.whole_observation'},
                set(), 'SEC whole-observation state plus observed session dates; no numeric OHLCV state selector',
                rp, 'build_feature_matrix', rfuncs['build_feature_matrix'].lineno)
    dp, df, _ = source_functions('data',['standardize_history'])
    add('normalization_row_eligibility', 'preprocessing_validity_guard', set(REQUIRED_COLUMNS),
        set(REQUIRED_COLUMNS), 'to_numeric(errors=coerce); dropna(subset=REQUIRED_COLUMNS); date dedup/sort',
        dp, 'standardize_history', df['standardize_history'].lineno)
    add('feature_input_validity', 'preprocessing_validity_guard', set(REQUIRED_COLUMNS),
        set(REQUIRED_COLUMNS), 'OHLCV finite/numeric; prices > 0; Volume >= 0; unique ascending dates; reject, no cleaning',
        fp, '_validate_input', funcs['_validate_input'].lineno)
    # Source hashes bind broad pass-through operations and helper call semantics.
    # Mutation tests separately cover legal Volume values and all state columns.
    return pd.DataFrame(rows), {
        'complete': True, 'market_graph': graph, 'target_graph': targets.graph,
        'fundamental_constant_maps': mappings, 'functions': evidence,
        'authoritative_protocol': relative(policy_path), 'authoritative_protocol_sha256': h.sha256(policy_path),
        'authoritative_protocol_matches': protocol_match,
        'PIT_state_columns': pit_columns,
        'Volume_validation_dependency': True,
        'Volume_numeric_formula_dependency': any('volume' in n['raw_fields'] for n in graph.values())
            or any('volume' in n['raw_fields'] for n in targets.graph.values()),
        'proof_boundary': 'Registered source hashes + extracted formulas/constants + all-state isolated replay. Unsupported source changes invalidate the audit; this is not a general-purpose whole-program proof.'}


def fundamental_state_columns(daily):
    excluded = {*REQUIRED_COLUMNS, *h.MARKET_FEATURES, *h.RELATIVE_FEATURES, 'ticker'}
    return [c for c in daily if c not in excluded]


def exact_diff(left, right, columns=None):
    """Count exact cells, with paired missing sentinels; canonicalize dates only."""
    if columns is None:
        if list(left.columns) != list(right.columns):
            return {'cells': 1, 'changed_columns': ['__schema__']}
        columns = list(left.columns)
    if len(left) != len(right):
        return {'cells': abs(len(left)-len(right)) or 1, 'changed_columns': ['__row_count__']}
    changed, count = [], 0
    for col in columns:
        a,b = left[col].reset_index(drop=True),right[col].reset_index(drop=True)
        if col in date_columns(columns):
            a,b = canonical_dates(a,allow_missing=True),canonical_dates(b,allow_missing=True)
        equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
        n = int((~equal).sum())
        if n:
            changed.append(col)
            count += n
    return {'cells': count, 'changed_columns': changed}


def target_diff(left, right):
    """Exclude every locked numerical/provenance target cell from comparison."""
    mask = ~left.date.astype(str).between(*LOCKED)
    return exact_diff(left.loc[mask], right.loc[mask])


def pipeline_copy(market, facts):
    """Execute the original data builders without their historical artifact writers."""
    processed,snapshots,daily = h.construct_features(market.copy(deep=True),facts.copy(deep=True))
    research = h.research_slice(daily)
    return {'processed':processed, 'snapshots':snapshots, 'daily':daily,
            'research':research, 'targets':build_targets(research)}


def mutation_comparison(base, mutated):
    pit_columns = fundamental_state_columns(base['daily'])
    return {
        'market':exact_diff(base['daily'],mutated['daily'],h.MARKET_FEATURES),
        'relative':exact_diff(base['daily'],mutated['daily'],h.RELATIVE_FEATURES),
        'PIT':exact_diff(base['daily'],mutated['daily'],pit_columns),
        'snapshots':exact_diff(base['snapshots'],mutated['snapshots']),
        'targets':target_diff(base['targets'],mutated['targets']),
        'eligibility':exact_diff(base['daily'],mutated['daily'],['date','ticker']),
    }


def volume_mutation_audit(market, facts, base=None):
    base = base or pipeline_copy(market,facts)
    cases = [('primary_legacy_value','2026-09-30',121269300),
             ('primary_times_10','2026-09-30',None)]
    anchors = ['2010-06-30','2020-06-30','2021-07-20','2022-06-30',
               '2023-06-30','2024-06-10','2025-06-30','2026-06-30']
    dates = market.date.astype(str)
    for date in anchors:
        selected = dates.loc[dates.ge(date)].iloc[0]
        cases.append(('historical_times_10_plus_123',selected,None))
    cases.append(('all_rows_times_10_plus_123','ALL',None))
    records = []
    baseline_normalized = standardize_history(market.copy(deep=True),symbol='NVDA')
    for name,date,value in cases:
        changed = market.copy(deep=True)
        mask = pd.Series(True,index=changed.index) if date == 'ALL' else dates.eq(date)
        before = changed.loc[mask,'volume'].copy()
        changed.loc[mask,'volume'] = value if value is not None else before * 10 + (0 if name == 'primary_times_10' else 123)
        normalized = standardize_history(changed,symbol='NVDA')
        normalization_eligibility = exact_diff(baseline_normalized,normalized,['date'])['cells']
        diff = mutation_comparison(base,pipeline_copy(changed,facts))
        violations = normalization_eligibility + sum(d['cells'] for d in diff.values())
        records.append({'case':name,'date':None if date == 'ALL' else date,'field':'volume','mutated_rows':int(mask.sum()),
                        'old_value':int(before.iloc[0]) if int(mask.sum()) == 1 else None,
                        'new_value':int(changed.loc[mask,'volume'].iloc[0]) if int(mask.sum()) == 1 else None,
                        'market_cells_changed':diff['market']['cells'],
                        'relative_cells_changed':diff['relative']['cells'],
                        'target_cells_changed_outside_lock':diff['targets']['cells'],
                        'PIT_cells_changed':diff['PIT']['cells'],
                        'snapshot_cells_changed':diff['snapshots']['cells'],
                        'row_eligibility_changes':diff['eligibility']['cells']+normalization_eligibility,
                        'violations':violations,'passed':violations == 0})
    return pd.DataFrame(records)


def positive_controls(market,facts,base=None):
    base = base or pipeline_copy(market,facts)
    date = market.date.astype(str).loc[market.date.astype(str).ge('2023-06-30')].iloc[0]
    idx = market.index[market.date.astype(str).eq(date)][0]
    row = market.loc[idx]
    values = {'open':(row.open+row.close)/2,'high':row.high*1.01,
              'low':row.low*.99,'close':(row.close+row.high)/2}
    if values['open'] == row.open:
        values['open'] = (row.open+row.high)/2
    if values['close'] == row.close:
        values['close'] = (row.close+row.low)/2
    expected = {
        'open':{'intraday_return'}, 'high':{'daily_range'}, 'low':{'daily_range'},
        'close':set(h.MARKET_FEATURES),
    }
    records=[]
    for field,value in values.items():
        changed=market.copy(deep=True); changed.loc[idx,field]=value
        diff=mutation_comparison(base,pipeline_copy(changed,facts))
        targets_changed=diff['targets']['changed_columns']
        price_target_ok = (all(c in targets_changed for c in h.RETURN_COLUMNS)
                           if field in ('open','close') else diff['targets']['cells'] == 0)
        feature_ok = expected[field] <= set(diff['market']['changed_columns'])
        relative_ok = (set(h.RELATIVE_FEATURES) <= set(diff['relative']['changed_columns'])
                       if field == 'close' else diff['relative']['cells'] == 0)
        passed=(feature_ok and relative_ok and price_target_ok and
                all(diff[k]['cells'] == 0 for k in ('PIT','snapshots','eligibility')))
        records.append({'date':date,'field':field,'old_value':float(row[field]),'new_value':float(value),
                        'market_cells_changed':diff['market']['cells'],
                        'market_columns_changed':'|'.join(diff['market']['changed_columns']),
                        'relative_cells_changed':diff['relative']['cells'],
                        'target_cells_changed_outside_lock':diff['targets']['cells'],
                        'target_columns_changed':'|'.join(targets_changed),
                        'PIT_cells_changed':diff['PIT']['cells'],
                        'snapshot_cells_changed':diff['snapshots']['cells'],
                        'row_eligibility_changes':diff['eligibility']['cells'],
                        'passed':passed,'violations':0 if passed else 1})
    return pd.DataFrame(records)


def classify_materiality(field, dependencies, mutations_passed, controls_passed,
                         complete, legal_values=True, observed_changes=False):
    relevant = dependencies.loc[dependencies.dependency_type.ne('preprocessing_validity_guard')
                                & dependencies.source_field.eq(field.lower())]
    dependent = bool((relevant.used_directly | relevant.used_indirectly).any())
    if dependent or observed_changes:
        return MATERIAL
    if not complete or relevant.empty or not mutations_passed or not controls_passed or not legal_values:
        return UNKNOWN
    return NONMATERIAL


def known_registry(dependencies,mutations,controls,complete):
    original=h.read_csv(h.OUTPUT_DIR/'NVDA_stage13_unexpected_overlap_differences.csv')
    keys=['date','field','legacy_value','expanded_value']
    records=[]
    for key, group in original.groupby(keys,dropna=False,sort=True):
        date,field,old,new=key
        masks=dependencies.source_field.eq(str(field).lower())
        rel=dependencies.loc[masks]
        materiality=classify_materiality(str(field),dependencies,bool(mutations.passed.all()),
                                        bool(controls.passed.all()),complete,
                                        legal_values=bool(field == 'volume' and old >= 0 and new >= 0
                                                          and float(old).is_integer() and float(new).is_integer()),
                                        observed_changes=bool(mutations.violations.sum()))
        def used(kinds,which):
            return bool(rel.loc[rel.dependency_type.isin(kinds),which].any())
        records.append({'ticker':'NVDA','date':date,'field':str(field).title(),
                        'legacy_value':old,'canonical_candidate_value':new,
                        'difference':new-old,'root_cause_status':'UNRESOLVED_SOURCE_DISCREPANCY',
                        'legacy_evidence_recoverability':'LEGACY_EVIDENCE_NOT_RECOVERABLE',
                        'direct_feature_dependency':used(['market_feature','relative_market','PIT_fundamental_feature'],'used_directly'),
                        'indirect_feature_dependency':used(['market_feature','relative_market','PIT_fundamental_feature'],'used_indirectly'),
                        'target_dependency':used(['primary_target','secondary_target','target_provenance'],'used_directly')
                                            or used(['primary_target','secondary_target','target_provenance'],'used_indirectly'),
                        'PIT_dependency':used(['PIT_fundamental_feature','PIT_state_provenance'],'used_directly')
                                         or used(['PIT_fundamental_feature','PIT_state_provenance'],'used_indirectly'),
                        'preprocessing_validity_dependency':True,
                        'current_research_materiality':materiality,
                        'downstream_manifestations':len(group),'layers':'|'.join(group.layer.tolist()),
                        'notes':'Both counts satisfy the Volume validity guards. No current numerical X/Y/PIT or eligibility change; historical cause remains non-identifiable. Approval excludes Volume research.'})
    registry=pd.DataFrame(records)
    complete_registry=(len(original)==5 and len(registry)==1
        and registry.iloc[0]['date']=='2026-09-30' and registry.iloc[0]['field']=='Volume'
        and registry.iloc[0]['legacy_value']==121269300
        and registry.iloc[0]['canonical_candidate_value']==121732200)
    return registry,complete_registry


def approval_decision(governance_passed, materialities, registry_complete,
                      mutations_passed, controls_passed, policy_frozen):
    if not all((governance_passed,registry_complete,mutations_passed,controls_passed,policy_frozen)):
        return REJECTED
    if any(c != NONMATERIAL for c in materialities):
        return REJECTED
    return QUALIFIED if materialities else APPROVED


def approval_flags(status):
    if status not in (APPROVED,QUALIFIED,REJECTED):
        raise ValueError('Unknown approval category')
    approved=status in (APPROVED,QUALIFIED)
    return {'expanded_dataset_approved_for_downstream_research':approved,
            'stage14_modeling_eligible':approved}


def candidate_inventory():
    raw=h.OUTPUT_DIR/'raw'
    rows=[]
    for path in sorted(raw.iterdir()):
        if path.is_file():
            role=('normalized_market' if path.name=='NVDA_market.csv' else
                  'original_yfinance_table' if path.name=='NVDA_yahoo_source.csv' else
                  'original_vendor_chart' if 'chart_' in path.name else
                  'acquisition_manifest' if path.name=='NVDA_acquisition.json' else 'SEC_frozen_input')
            rows.append({'candidate_vintage_id':VINTAGE_ID,'path':relative(path),
                         'role':role,'provider':'SEC' if 'companyfacts' in path.name else 'Yahoo Finance/yfinance',
                         'sha256':h.sha256(path),'bytes':path.stat().st_size,'frozen':True})
    return pd.DataFrame(rows)


def future_policy(protocol):
    return {**protocol['future_vintage_policy'], **LIMITS,
            'policy_version':'Stage13.2-v1','no_cell_patching':True,
            'legacy_role_if_approved':'HISTORICAL_RESEARCH_RECORD_ONLY',
            'new_vintage_id_must_differ':True,
            'new_acquisition_cannot_overwrite_any_used_vintage':True,
            'approval_scope':SCOPE,
            'future_metadata_missing_means_incomplete':True,
            'candidate_legacy_metadata_exception_is_not_future_permission':True,
            'volume_revalidation_examples':protocol['volume_future_use_revalidation_rule']['examples'],
            'future_acquisition_steps':[
                'Acquire into a new immutable namespace with a distinct vintage ID.',
                'Persist original vendor response and normalized output, hashes and all required metadata.',
                'Audit internal calendar/OHLC/corporate-action/PIT/target consistency.',
                'Generate complete cross-vintage diffs and registered-purpose materiality classification.',
                'Obtain a new explicit governance decision before adopting the new vintage.']}


def governance_audit(inventory,base,market,registry_complete,code_complete,protocol):
    rows=[]
    def check(name,ok,source,notes='',violations=None):
        rows.append({'check':name,'status':'PASS' if ok else 'FAIL','evidence_source':source,
                     'violations':(0 if ok else 1) if violations is None else int(violations),'notes':notes})
    v=load_json(h.OUTPUT_DIR/'NVDA_stage13_validation.json')
    e=load_json(h.OUTPUT_DIR/'NVDA_stage13_execution_verification.json')
    acq=load_json(h.OUTPUT_DIR/'raw/NVDA_acquisition.json')
    frozen=load_json(OUTPUT_DIR/(PREFIX+'upstream_sha256.json'))
    raw_hash_ok=all((h.OUTPUT_DIR/'raw'/n).is_file() and h.sha256(h.OUTPUT_DIR/'raw'/n)==digest
                    for n,digest in acq['raw_sha256'].items())
    check('raw_source_frozen',{'original_yfinance_table','original_vendor_chart'} <= set(inventory.role),
          relative(h.OUTPUT_DIR/'raw'),'Original archived table and HTTP response present; no current vendor request')
    check('raw_and_normalized_hash_identity',raw_hash_ok and all(frozen.get(row.path)==row.sha256 for row in inventory.itertuples()),
          relative(h.OUTPUT_DIR/'raw/NVDA_acquisition.json'))
    check('provider_identity_and_query_bounds',acq['requested_start']=='2009-08-20'
          and acq['requested_end_exclusive']=='2026-10-01' and 'Yahoo Finance' in acq['source'],
          relative(h.OUTPUT_DIR/'raw/NVDA_acquisition.json'),'Research slice 2020-01-01–2026-09-30; daily bars')
    original=pd.read_csv(h.OUTPUT_DIR/'raw/NVDA_yahoo_source.csv',header=[0,1],index_col=0,
                         parse_dates=True,float_precision='round_trip')
    normalized=standardize_history(original,'NVDA')
    frozen_normalized=h.read_csv(h.OUTPUT_DIR/'raw/NVDA_market.csv')
    difference=exact_diff(normalized,frozen_normalized)['cells']
    check('original_table_normalization_exact_replay',difference==0,relative(h.OUTPUT_DIR/'raw/NVDA_yahoo_source.csv'),violations=difference)
    main_chart=max((load_json(p)['chart']['result'][0] for p in (h.OUTPUT_DIR/'raw').glob('*chart_*.json')),
                   key=lambda x:len(x.get('timestamp',[])))
    chart_dates=pd.to_datetime(main_chart['timestamp'],unit='s',utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d')
    check('raw_chart_source_identity_and_sessions',main_chart['meta']['symbol']=='NVDA'
          and main_chart['meta']['dataGranularity']=='1d'
          and main_chart['meta']['exchangeTimezoneName']=='America/New_York'
          and list(chart_dates)==market.date.astype(str).tolist(),'Stage13 frozen chart meta/timestamp',
          'Provider session date interpreted in America/New_York; timestamps are bar timestamps, not retrieval times')
    quote=main_chart['indicators']['quote'][0]
    chart_volume=pd.Series(quote['volume'])
    check('raw_chart_volume_internal_coherence',bool(chart_volume.eq(market.volume.reset_index(drop=True)).all()),
          'Stage13 original chart/table/normalized Volume','This is internal vintage coherence, not absolute vendor truth')
    prices=market.loc[:,['open','high','low','close']].to_numpy(dtype=float)
    bounds=(market.low.le(market.open)&market.low.le(market.close)&market.high.ge(market.open)&market.high.ge(market.close)&market.low.le(market.high))
    check('critical_OHLC_valid',bool(np.isfinite(prices).all() and (prices>0).all() and bounds.all()),relative(h.OUTPUT_DIR/'raw/NVDA_market.csv'))
    check('Volume_legal_values',bool(np.isfinite(market.volume).all() and market.volume.ge(0).all()
                                  and market.volume.eq(np.floor(market.volume)).all()),relative(h.OUTPUT_DIR/'raw/NVDA_market.csv'),
          'Volume is a validity input; current registered numerical features do not consume its count')
    dates=market.date.astype(str)
    check('market_rows_ranges_unique_order',len(market)==4304 and len(base['research'])==1695
          and dates.iloc[0]=='2009-08-20' and dates.iloc[-1]=='2026-09-30'
          and not dates.duplicated().any() and dates.is_monotonic_increasing,
          relative(h.OUTPUT_DIR/'raw/NVDA_market.csv'))
    check('presample_maturity',int(dates.lt('2020-01-01').sum())>=60,
          relative(h.OUTPUT_DIR/'NVDA_stage13_protocol.json'),f"{int(dates.lt('2020-01-01').sum())} pre-sample sessions; all raw history retained for PIT initialization")
    for name in ('calendar','raw_calendar'):
        item=v[name]
        check(name+'_audit',item['violations']==0 and item['observed_session_count']==item['expected_session_count'],
              relative(h.OUTPUT_DIR/'NVDA_stage13_validation.json')+'#'+name,violations=item['violations'])
    corp=v['corporate_action_audit']
    splits={(s['date'],s['split_ratio']) for s in corp['split_events']}
    check('corporate_actions',corp['violations']==0 and splits=={('2021-07-20',4.0),('2024-06-10',10.0)}
          and corp['OHLC_bound_violations']==0 and corp['global_discontinuity_violations']==0
          and all(s['violations']==0 and s['SMA_rolling_oracle_violations']==0
                  and s['cross_event_target_violations']==0 for s in corp['split_events']),
          relative(h.OUTPUT_DIR/'NVDA_stage13_validation.json')+'#corporate_action_audit',
          'Existing fixed split-adjusted OHLC coordinate retained; no dividend reapplication',corp['violations'])
    leak=v['leakage_audits']
    pit_ok=v['stage5_validation']['valid'] and v['stage61_validation']['valid'] \
        and all(x['violations']==0 for x in leak.values() if isinstance(x,dict)) \
        and leak['amendment_timing_isolation']['post_effective_positive_control']
    check('PIT_activation_presample_future_filing_amendment',pit_ok,
          relative(h.OUTPUT_DIR/'NVDA_stage13_validation.json')+'#leakage_audits',
          '45 prefix dates; strict next observed session; independent state/YoY; future SEC; amendment positive control')
    t=h.target_oracle(base['research'],base['targets'],LOCKED)
    check('targets_definition_and_independent_oracle',v['target_oracle']['violations']==0 and t['violations']==0,
          relative(h.OUTPUT_DIR/'NVDA_stage13_validation.json')+'#target_oracle',
          f"Observed Open[t+1] to Close[t+h], h=1/5/20; {t['horizon_checks']} checks; 50 locked numerical rows skipped",t['violations'])
    repro=e['reproducibility']
    mismatches=[]
    for name,digest in repro['run1_sha256'].items():
        path=h.OUTPUT_DIR/name
        if not path.is_file() or h.sha256(path)!=digest or repro['run2_sha256'].get(name)!=digest:
            mismatches.append(name)
    check('Stage13_two_run_reproducibility_and_current_identity',repro['byte_for_byte_identical']
          and repro['consecutive_formal_runs']>=2 and repro['mismatch_count']==0 and not mismatches,
          relative(h.OUTPUT_DIR/'NVDA_stage13_execution_verification.json'),
          f"20 formal artifacts + 6 frozen raw/cache; {len(repro['run1_sha256'])} current hashes verified",len(mismatches))
    for key,filename in [('research','research_matrix'),('snapshots','fundamental_snapshots')]:
        differences=exact_diff(base[key],h.read_csv(h.OUTPUT_DIR/f'NVDA_stage13_{filename}.csv'))['cells']
        check(key+'_exact_semantic_replay',differences==0,relative(h.OUTPUT_DIR/f'NVDA_stage13_{filename}.csv'),
              'Original default CSV parser boundaries replayed; comparison exact after date canonicalization',differences)
    differences=target_diff(base['targets'],h.read_csv(h.OUTPUT_DIR/'NVDA_stage13_targets.csv'))['cells']
    check('targets_exact_replay_outside_Final_Test',differences==0,relative(h.OUTPUT_DIR/'NVDA_stage13_targets.csv'),violations=differences)
    # The one historical failed gate stays failed, rather than being relaxed or deleted.
    check('historical_acceptance_failures_preserved',not v['valid'] and v['checks']['overlap_compatible'] is False
          and all(value for name,value in v['checks'].items() if name!='overlap_compatible')
          and v['overlap']['unexpected_violations']==5,
          relative(h.OUTPUT_DIR/'NVDA_stage13_validation.json'),'Original strict-overlap failure remains 5; only the separate prospective governance question is judged here')
    s13=load_json(h.OUTPUT_DIR/'NVDA_stage13_summary.json')
    s131=load_json(h.OUTPUT_DIR.parent/'stage13_1_yahoo_reconciliation/NVDA_stage13_1_summary.json')
    recovery=load_json(h.OUTPUT_DIR.parent/'legacy_evidence_recovery/NVDA_legacy_recovery_summary.json')
    check('legacy_nonidentifiability_and_historical_statuses',
          s13['completion_status']=='INCOMPLETE_OR_BLOCKED' and s131['completion_status']=='UNRESOLVED_BLOCKER'
          and s131['root_cause']=='UNRESOLVED_SOURCE_DISCREPANCY'
          and recovery['completion_classification']=='LEGACY_EVIDENCE_NOT_RECOVERABLE',
          'Preserved Stage13/13.1/Recovery summaries','No forensic investigation or causal closure attempted')
    check('known_discrepancy_registry_complete',registry_complete,relative(h.OUTPUT_DIR/'NVDA_stage13_unexpected_overlap_differences.csv'))
    check('code_and_authoritative_manifest_dependency_evidence',code_complete,PREFIX+'dependency_code_evidence.json')
    upstream=verify_upstream()
    check('upstream_immutability',upstream['mutation_violations']==0,PREFIX+'upstream_sha256.json',
          f"{upstream['files_checked']} protected files",upstream['mutation_violations'])
    locked_rows=int(base['research'].date.astype(str).between(*LOCKED).sum())
    check('Final_Test_locked_and_no_modeling',locked_rows==50 and protocol['modeling_allowed'] is False
          and protocol['network_allowed'] is False,PREFIX+'protocol.json','No predictions, metrics, fitting, training pool or Stage14 design; target numeric comparisons exclude the 50 locked rows')
    # Explicit documented, noncritical exception; never manufacture old metadata.
    rows.append({'check':'acquisition_timestamp_version_endpoint_limitation','status':'DOCUMENTED_LIMITATION',
                 'evidence_source':relative(h.OUTPUT_DIR/'raw/NVDA_acquisition.json'),'violations':0,
                 'notes':'Reliable retrieval_timestamp_utc/library_version/exact endpoint-request metadata unavailable; null in manifest. Preregistered exception conditional on preserved original payload/hash/query/exact replay/internal audit; future vintages require all metadata.'})
    return pd.DataFrame(rows),upstream


def canonical_manifest(status,inventory,policy,protocol):
    approved=status in (APPROVED,QUALIFIED)
    acq=load_json(h.OUTPUT_DIR/'raw/NVDA_acquisition.json')
    sources={r.path:r.sha256 for r in inventory.itertuples()}
    sources[relative(DATA_DIR/'fundamentals/processed/NVDA_fundamentals.csv')]=h.sha256(DATA_DIR/'fundamentals/processed/NVDA_fundamentals.csv')
    for name in ('research_matrix','fundamental_snapshots','targets','validation','protocol','execution_verification'):
        path=h.OUTPUT_DIR/f'NVDA_stage13_{name}.{"csv" if name in ("research_matrix","fundamental_snapshots","targets") else "json"}'
        sources[relative(path)]=h.sha256(path)
    for module in ('data','features','alignment','research_features','targets','historical_expansion',
                   'dates','provenance','fundamentals','comparative','fundamental_mapping'):
        path=PROJECT_ROOT/'src/nasdaq_research'/(module+'.py')
        sources[relative(path)]=h.sha256(path)
    return {
        'canonical_vintage_id':VINTAGE_ID, 'approved':approved, 'approval_status':status,
        'canonical_meaning':'Unified frozen research authority for the approved purpose; future replacement needs a new governance decision',
        'absolute_vendor_truth_claimed':False, 'latest_forever_claimed':False,
        'provider':'Yahoo Finance via yfinance', 'ticker':'NVDA', 'interval':'1d',
        'query_start':acq['requested_start'],'query_end_exclusive':acq['requested_end_exclusive'],
        'observed_raw_dates':['2009-08-20','2026-09-30'],'raw_rows':4304,
        'observed_research_dates':['2020-01-02','2026-09-30'],'research_rows':1695,
        'retrieval_timestamp_utc':None,'library_name':'yfinance','library_version_at_acquisition':None,
        'exact_endpoint_request_metadata':None,
        'metadata_gap_reason':'Not reliably preserved in Stage13 acquisition manifest; no inference from bar timestamps, file mtimes or current installed version',
        'metadata_gap_acceptance_basis':'Frozen original payloads + hashes + query bounds + exact normalization/semantic replay + preserved deterministic outputs + critical internal audits; sufficient only for preregistered prospective fit-for-purpose authority',
        'adjustment_semantics':{'auto_adjust':False,'repair':False,'dividend_adjustment_applied':False,
                                'OHLC':'Inherited vendor split-adjusted frozen price coordinate; not certified as-traded history',
                                'back_adjust_at_acquisition':None},
        'timezone_session_semantics':'Original chart exchangeTimezoneName America/New_York; normalized naive session dates; targets use observed rows',
        'corporate_actions':[{'date':'2021-07-20','ratio':4.0},{'date':'2024-06-10','ratio':10.0}],
        'source_sha256':sources,'authority_scope':SCOPE,
        'known_discrepancy_registry':relative(OUTPUT_DIR/(PREFIX+'known_cross_vintage_discrepancies.csv')),
        'known_independent_discrepancy_count':1,'historical_downstream_manifestations':5,
        'legacy_cross_vintage_root_cause':'historically_non_identifiable_from_preserved_evidence',
        'legacy_root_cause_status':'UNRESOLVED_SOURCE_DISCREPANCY',
        'legacy_role':'HISTORICAL_RESEARCH_RECORD_ONLY' if approved else 'Unchanged historical records; no prospective authority assigned',
        'historical_statuses':protocol['historical_statuses'],
        'no_cell_patching':True,'silent_refresh_allowed':False,
        'future_policy_sha256':h.sha256(OUTPUT_DIR/(PREFIX+'future_vintage_policy.json')),
        'protocol_sha256':PROTOCOL_SHA256,**LIMITS,**approval_flags(status),**SAFETY}


def require_canonical_vintage(vintage_id, *, manifest_path=None, root=PROJECT_ROOT,
                              required_features=(), required_target='forward_return_5d'):
    """Future-facing offline gate: explicit identity, approved scope, source hashes.

    This additive API does not change historical downloader behavior. Future
    formal research must call this gate and must not fall back to an ad-hoc
    acquisition. Unknown feature definitions require a new governance decision.
    """
    if not vintage_id:
        raise ValueError('Explicit canonical_vintage_id required in research mode')
    path=Path(manifest_path) if manifest_path else OUTPUT_DIR/(PREFIX+'canonical_vintage_manifest.json')
    manifest=load_json(path)  # FileNotFoundError is intentional; no refresh.
    if vintage_id != manifest.get('canonical_vintage_id'):
        raise ValueError('Canonical vintage identity mismatch')
    if manifest.get('approved') is not True or manifest.get('approval_status') not in (APPROVED,QUALIFIED):
        raise ValueError('Vintage is not approved for downstream research')
    scope=manifest.get('authority_scope',{})
    allowed=set(scope.get('market_features',[])) | set(scope.get('relative_market_features',[])) \
        | set(scope.get('fundamental_features',[]))
    if not set(required_features) <= allowed or required_target not in scope.get('targets',[]):
        raise ValueError('Requested field/feature/target needs new validation and governance approval')
    sources=manifest.get('source_sha256',{})
    if not sources:
        raise ValueError('Canonical source/hash inventory missing')
    root=Path(root).resolve()
    for name,digest in sources.items():
        source=(root/name).resolve()
        if not source.is_relative_to(root):
            raise ValueError('Canonical source path escapes root')
        if not source.is_file():
            raise FileNotFoundError('Canonical source/cache missing: '+name)
        if h.sha256(source) != digest:
            raise ValueError('Canonical source/cache hash mismatch: '+name)
    return manifest


def run_pipeline():
    """Run only this preregistered governance stage; never run old stage writers."""
    protocol_path=OUTPUT_DIR/(PREFIX+'protocol.json')
    if h.sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Stage13.2 protocol is locked; cannot revise after registration')
    protocol=load_json(protocol_path)
    upstream=verify_upstream()
    if upstream['mutation_violations']:
        raise ValueError('Protected upstream files changed; cannot certify this evidence set')
    with offline_only():
        inventory=candidate_inventory()
        # Identical parser contracts to Stage13 formal execution, not a new normalization.
        market=pd.read_csv(h.OUTPUT_DIR/'raw/NVDA_market.csv')
        facts=pd.read_csv(DATA_DIR/'fundamentals/processed/NVDA_fundamentals.csv')
        base=pipeline_copy(market,facts)
        dependencies,code=dependency_evidence(base['daily'])
        mutations=volume_mutation_audit(market,facts,base)
        controls=positive_controls(market,facts,base)
        registry,registry_complete=known_registry(dependencies,mutations,controls,code['complete'])
        governance,upstream=governance_audit(inventory,base,market,registry_complete,code['complete'],protocol)
        policy=future_policy(protocol)
        policy_frozen=all(policy.get(k) == value for k,value in protocol['future_vintage_policy'].items())
        critical_passed=bool(governance.loc[governance.status.ne('DOCUMENTED_LIMITATION'),'status'].eq('PASS').all())
        status=approval_decision(critical_passed,registry.current_research_materiality.tolist(),
                                 registry_complete,bool(mutations.passed.all()),bool(controls.passed.all()),policy_frozen)
        materiality=registry.loc[:,['ticker','date','field','direct_feature_dependency','indirect_feature_dependency',
                                    'target_dependency','PIT_dependency','current_research_materiality']].copy()
        materiality['dependency_evidence_complete']=code['complete']
        materiality['Volume_mutation_cases_passed']=bool(mutations.passed.all())
        materiality['OHLC_positive_controls_passed']=bool(controls.passed.all())
        materiality['row_eligibility_unchanged']=bool(mutations.row_eligibility_changes.eq(0).all())
        materiality['limitations']='Legal nonnegative finite Volume counts only; missing/negative/nonfinite Volume is a preprocessing gate. Future Volume use requires revalidation.'
        tables={'candidate_source_inventory':inventory,'research_dependency_manifest':dependencies,
                'known_cross_vintage_discrepancies':registry,'materiality_audit':materiality,
                'governance_audit':governance,'volume_mutation_audit':mutations,'positive_controls':controls}
        for name,frame in tables.items():
            h.write(OUTPUT_DIR/(PREFIX+name+'.csv'),h.csv_text(frame))
        h.write(OUTPUT_DIR/(PREFIX+'dependency_code_evidence.json'),h.json_text(code))
        h.write(OUTPUT_DIR/(PREFIX+'future_vintage_policy.json'),h.json_text(policy))
        manifest=canonical_manifest(status,inventory,policy,protocol)
        h.write(OUTPUT_DIR/(PREFIX+'canonical_vintage_manifest.json'),h.json_text(manifest))
        summary={
            'stage_name':protocol['stage_name'],'completion_classification':status,
            'candidate_vintage_id':VINTAGE_ID,'governance_axis_passed':critical_passed,
            'materiality_axis_complete':code['complete'],'known_registry_complete':registry_complete,
            'known_independent_discrepancy_count':len(registry),'historical_unexpected_overlap_violations':5,
            'downstream_manifestations':int(registry.downstream_manifestations.sum()),
            'materiality_counts':registry.current_research_materiality.value_counts().to_dict(),
            'market_raw_rows':len(market),'research_rows':len(base['research']),
            'market_raw_dates':[str(market.date.iloc[0]),str(market.date.iloc[-1])],
            'research_dates':[str(base['research'].date.iloc[0]),str(base['research'].date.iloc[-1])],
            'governance_critical_checks':int(governance.status.ne('DOCUMENTED_LIMITATION').sum()),
            'governance_failures':governance.loc[governance.status.eq('FAIL'),'check'].tolist(),
            'volume_mutation_cases':len(mutations),'volume_mutation_violations':int(mutations.violations.sum()),
            'positive_controls':len(controls),'positive_control_violations':int(controls.violations.sum()),
            'market_features_checked':len(h.MARKET_FEATURES),'relative_features_checked':len(h.RELATIVE_FEATURES),
            'fundamental_features_checked':len(rf.FEATURE_COLUMNS),
            'PIT_state_columns_checked':len(fundamental_state_columns(base['daily'])),
            'snapshot_rows_checked_per_case':len(base['snapshots']),
            'target_numeric_rows_skipped_per_case':int(base['targets'].date.between(*LOCKED).sum()),
            'historical_statuses':protocol['historical_statuses'],
            'legacy_cross_vintage_root_cause':'historically_non_identifiable_from_preserved_evidence',
            'legacy_future_role':manifest['legacy_role'],
            'acquisition_metadata_limitation':manifest['metadata_gap_reason'],
            'absolute_vendor_truth_claimed':False,'future_vintage_policy_frozen':policy_frozen,
            'upstream':upstream,**LIMITS,**approval_flags(status),**SAFETY}
        validation={
            'governance_decision_valid':critical_passed and registry_complete and code['complete']
                and bool(mutations.passed.all()) and bool(controls.passed.all()) and policy_frozen,
            'checks':{'protocol_frozen':True,'governance_axis_passed':critical_passed,
                      'dependency_evidence_complete':code['complete'],'known_registry_complete':registry_complete,
                      'Volume_mutations':bool(mutations.passed.all()),'positive_controls':bool(controls.passed.all()),
                      'future_policy_frozen':policy_frozen,'upstream_immutable':upstream['mutation_violations']==0,
                      'historical_decisions_preserved':True,'offline_execution':True,'no_modeling':True,
                      'Final_Test_locked':True},
            'completion_classification':status,'protocol_sha256':PROTOCOL_SHA256,
            'upstream':upstream,'formal_artifact_sha256':{
                p.name:h.sha256(p) for p in sorted(OUTPUT_DIR.iterdir())
                if p.suffix in ('.csv','.json') and p.name not in
                (PREFIX+'summary.json',PREFIX+'validation.json',PREFIX+'execution_verification.json')},
            'formal_repeat_and_tests_evidence':'NVDA_stage13_2_execution_verification.json (post-run execution evidence)',
            **approval_flags(status),**LIMITS,**SAFETY}
        h.write(OUTPUT_DIR/(PREFIX+'summary.json'),h.json_text(summary))
        h.write(OUTPUT_DIR/(PREFIX+'validation.json'),h.json_text(validation))
    return summary


def main():
    result=run_pipeline()
    print(h.json_text({k:result[k] for k in ('completion_classification','governance_axis_passed',
        'volume_mutation_cases','volume_mutation_violations','positive_control_violations',
        'expanded_dataset_approved_for_downstream_research','stage14_modeling_eligible')}))


if __name__ == '__main__':
    main()
