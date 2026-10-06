"""Stage 15: one preregistered, offline PIT information-block ablation.

Stage 14 owns folds/labels; Stage 6 owns PIT values; Stage 10 owns fitting.
Only additive Stage 15 files are written. Historical authorities stay frozen.
"""

import argparse
import csv
from io import StringIO
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from nasdaq_research import data_authority as authority
from nasdaq_research import historical_expansion as h
from nasdaq_research import market_representation as representation
from nasdaq_research import modeling as baseline
from nasdaq_research import research_features as rf
from nasdaq_research import walk_forward as w
from nasdaq_research.alignment import build_research_dataset
from nasdaq_research.config import PROJECT_ROOT, DATA_DIR
from nasdaq_research.dates import parse_dates
from nasdaq_research.fundamentals import KEY_COLUMNS
from nasdaq_research.targets import PRIMARY_TARGET

OUTPUT_DIR = DATA_DIR / 'research/stage15_pit_fundamental_ablation'
PREFIX = 'NVDA_stage15_'
CONTROL = representation.TREATMENT
FUNDAMENTALS = rf.FEATURE_COLUMNS
TREATMENT = (*CONTROL, *FUNDAMENTALS)
CONTROLS = ('relative_ols', 'relative_ridge_alpha1')
TREATMENTS = ('relative_fundamentals_ols', 'relative_fundamentals_ridge_alpha1')
BASELINES = ('zero_return', 'historical_mean')
MODELS = (*BASELINES, *CONTROLS, *TREATMENTS)
SAFETY = {**representation.SAFETY, 'network_allowed': False,
          'volume_used': False, 'secondary_targets_modeled': False,
          'final_model_fitted': False, 'hyperparameter_tuning_allowed': False,
          'new_feature_engineering_allowed': False, 'feature_family_search_allowed': False,
          'quant_mainline_auto_extension_after_stage15': False,
          'legacy_root_cause_reinvestigated': False}


def output(name, output_dir=OUTPUT_DIR, suffix='json'):
    return Path(output_dir) / (PREFIX + name + '.' + suffix)


def freeze(path, contents):
    """An existing formal artifact must reproduce exactly, never be replaced."""
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Stage15 output symlinks forbidden')
    if path.exists():
        if path.read_bytes() != contents.encode('utf-8'):
            raise ValueError('Frozen Stage15 artifact differs: ' + path.name)
    else:
        h.write(path, contents)


def contract():
    """Read historical specifications, never Treatment outcomes, to register."""
    previous = w.load_protocol()
    manifest = h.read_csv(w.OUTPUT_DIR / (w.PREFIX + 'feature_manifest.csv'))
    relative = manifest.loc[manifest.representation.eq('relative')].sort_values('feature_position')
    if tuple(relative.feature_name) != CONTROL or tuple(previous['relative_market_features']) != CONTROL:
        raise ValueError('Stage14 Relative whitelist differs from implementation')
    historical = authority.load_json(h.OUTPUT_DIR / 'NVDA_stage13_protocol.json')
    inherited = tuple(c for c in historical['research_feature_whitelist'] if c not in h.MARKET_FEATURES)
    scope = authority.load_json(authority.OUTPUT_DIR / (authority.PREFIX + 'canonical_vintage_manifest.json'))
    if inherited != FUNDAMENTALS or tuple(scope['authority_scope']['fundamental_features']) != FUNDAMENTALS:
        raise ValueError('Stage6/13/13.2 fundamental whitelists disagree')
    authorities = {name: str((w.OUTPUT_DIR / (w.PREFIX + name + suffix)).relative_to(PROJECT_ROOT))
                   for name, suffix in [('protocol', '.json'), ('fold_registration', '.json'),
                                        ('fold_inventory', '.csv'), ('target_purge_audit', '.csv'),
                                        ('feature_manifest', '.csv'), ('oof_predictions', '.csv'),
                                        ('model_parameters', '.csv'), ('fit_membership', '.csv')]}
    return {
        'stage_name': 'Stage 15 — Pre-Registered Incremental PIT Fundamentals Walk-Forward Ablation',
        'stage_type': ['INCREMENTAL INFORMATION ABLATION', 'MULTI-YEAR OOS VALIDATION', 'PIT FUNDAMENTAL VALIDATION'],
        'ticker': 'NVDA', 'canonical_vintage_id': previous['canonical_vintage_id'],
        'required_authority_status': previous['required_authority_status'],
        'stage14_control_authority': authorities,
        'stage14_authority_sha256': {key: h.sha256(PROJECT_ROOT / path) for key, path in authorities.items()},
        'stage14_fold_authority': authorities['fold_registration'],
        'fold_calendar_definitions': previous['fold_calendar_definitions'],
        'fold_count': 22, 'expected_oof_dates': 1367,
        'walk_forward_type': previous['walk_forward_type'],
        'validation_frequency': previous['validation_frequency'],
        'target': PRIMARY_TARGET, 'target_horizon': 5,
        'target_definition': previous['target_definition'], 'purge_rule': previous['purge_rule'],
        'development_start': w.DEVELOPMENT_START, 'development_end': w.DEVELOPMENT_END,
        'pretest_gap': [w.GAP_START, w.GAP_END], 'final_test': [w.FINAL_START, w.FINAL_END],
        'validation_label_policy': previous['validation_label_policy'],
        'control_features': list(CONTROL), 'fundamental_features': list(FUNDAMENTALS),
        'treatment_features': list(TREATMENT),
        'fundamental_whitelist_authority': 'research_features.FEATURE_COLUMNS; Stage13 protocol; Stage13.2 scope',
        'models': [*CONTROLS, *TREATMENTS], 'ridge_alpha': baseline.RIDGE_ALPHA,
        'ridge_solver': 'svd', 'fit_intercept': True, 'baselines': list(BASELINES),
        'preprocessing_order': previous['preprocessing_order'],
        'coverage_threshold': baseline.TRAINING_COVERAGE_THRESHOLD,
        'target_scaling': 'none', 'complete_case_filtering_allowed': False,
        'fit_authority': 'Control: Stage11 fit_arm; Treatment: bijective SMA-name adapter -> unchanged Stage10 fit_linear(ols_all/ridge_all)',
        'PIT_semantics': 'filing date -> first observed session strictly after filing; unchanged Stage6 q/fy/bs whole-observation state and amendment precedence; no period-end backfill',
        'pit_audit_scope': 'Every development date x 30 features; source snapshot identity/value, source filing, next-session activation, eligible chronology and YoY reference chronology; independent whole-observation selection oracle',
        'control_reproduction_required': True,
        'control_reproduction_tolerance': {'rtol': 0.0, 'atol': 1e-12},
        'prediction_reconstruction_tolerance': previous['prediction_reconstruction_tolerance'],
        'incremental_metric_definition': 'MAE_treatment - MAE_corresponding_control',
        'fold_win_definition': previous['fold_win_definition'],
        'primary_aggregate_metric': previous['primary_aggregate_metric'],
        'incremental_success_rules': {
            'model_robust': ['mean paired delta < 0', 'median paired delta < 0', 'strict fold win rate > 0.50'],
            'broad_OOS_improvement_definition': 'mean paired delta < 0 AND strict fold win rate > 0.50',
            'classification_precedence': ['ROBUST if either fixed family passes all three',
                                          'NO_ROBUST if neither family has broad OOS improvement and neither passes all three',
                                          'MIXED otherwise'],
            'ambiguity_resolution': 'Numeric robust predicate takes precedence over illustrative mixed examples; broad OOS improvement uses inherited Stage14 mean-and-majority semantics, frozen before Treatment.'},
        'incremental_classifications': ['ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE',
                                       'MIXED_OR_REGIME_DEPENDENT_INCREMENTAL_VALUE',
                                       'NO_ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE'],
        'absolute_signal_success_rules': previous['predictive_success_rules'],
        'absolute_signal_classifications': previous['predictive_classifications'],
        'numerical_diagnostics': ['effective feature count', 'augmented scaled design matrix rank',
                                  'rank deficiency', 'condition number', 'coefficient L2', 'max abs coefficient', 'intercept'],
        'numerical_failure_rule': 'Nonfinite prediction/parameter/input or fit failure blocks; rank/condition/coefficient magnitude descriptive only; no post-hoc feature changes',
        'prediction_stability_endpoints': previous['prediction_stability_endpoints'],
        'extrapolation_endpoints': previous['extrapolation_endpoints'],
        'range_denominator': previous['range_denominator'],
        'yearly_diagnostics': '2021–2026 pre-test, descriptive only; no regime selection',
        'statistical_inference_allowed': False, 'volume_forbidden': True,
        'missingness_indicators_allowed': False, 'final_training_pool_read_allowed': False,
        'protocol_frozen_before_treatment': True,
        'historical_statuses': previous['historical_statuses'],
        'inherited_stage14_classifications': {
            'predictive_signal': 'NO_ROBUST_MULTI_YEAR_SIGNAL',
            'representation_stability': 'BROAD_REPRESENTATION_STABILITY_SUPPORT'},
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'randomness': 'none', 'BLAS_threads': 1,
        'reproducibility_policy': 'Two consecutive offline formal runs, every CSV/JSON byte-identical; no timestamps or temporary paths; execution verification afterwards',
        **SAFETY}


def register(initial_state, output_dir=OUTPUT_DIR):
    """Freeze protocol first, then protect all existing Stage1–14 evidence."""
    initial_state = authority.load_json(initial_state)
    if not initial_state['working_tree_clean']:
        raise ValueError('Unknown initial working-tree changes; preserve and audit before registration')
    freeze(output('initial_repository_state', output_dir), h.json_text(initial_state))
    freeze(output('protocol', output_dir), h.json_text(contract()))
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=PROJECT_ROOT, check=True,
                             capture_output=True).stdout.decode().split('\0')
    paths = {PROJECT_ROOT / name for name in tracked if name}
    paths.update(p for p in DATA_DIR.rglob('*') if p.is_file() and not p.is_relative_to(OUTPUT_DIR))
    paths = {p for p in paths if p.is_file() and not p.is_relative_to(OUTPUT_DIR)
             and p != PROJECT_ROOT / 'src/nasdaq_research/fundamental_ablation.py'
             and p != PROJECT_ROOT / 'tests/test_fundamental_ablation.py'
             and p != PROJECT_ROOT / 'docs/stage15_execution_report.md'}
    hashes = {str(p.relative_to(PROJECT_ROOT)): h.sha256(p) for p in sorted(paths)}
    freeze(output('upstream_sha256', output_dir), h.json_text(hashes))
    registration = {'protocol_sha256': h.sha256(output('protocol', output_dir)),
                    'upstream_manifest_sha256': h.sha256(output('upstream_sha256', output_dir)),
                    'protected_files': len(hashes), 'frozen_before_control_and_treatment': True}
    freeze(output('registration', output_dir), h.json_text(registration))
    return registration


def verify_upstream(output_dir=OUTPUT_DIR):
    manifest = authority.load_json(output('upstream_sha256', output_dir))
    changed = [name for name, digest in manifest.items() if not (PROJECT_ROOT / name).is_file()
               or h.sha256(PROJECT_ROOT / name) != digest]
    return {'files_checked': len(manifest), 'mutation_violations': len(changed), 'changed_files': changed}


def load_protocol(output_dir=OUTPUT_DIR):
    registration = authority.load_json(output('registration', output_dir))
    for name, field in [('protocol', 'protocol_sha256'), ('upstream_sha256', 'upstream_manifest_sha256')]:
        if h.sha256(output(name, output_dir)) != registration[field]:
            raise ValueError('Stage15 preregistered ' + name + ' changed')
    protocol = authority.load_json(output('protocol', output_dir))
    if protocol != contract():
        raise ValueError('Stage15 locked specification differs from registered code/authority')
    return protocol


def load_fold_context(protocol):
    """Reconstruct membership using Stage14 code; compare every frozen row."""
    for key, name in protocol['stage14_control_authority'].items():
        if h.sha256(PROJECT_ROOT / name) != protocol['stage14_authority_sha256'][key]:
            raise ValueError('Stage14 authority changed: ' + key)
    data, sessions, features, evidence = w.load_development(w.load_protocol())
    inventory, purges, members = w.construct_folds(data, sessions, protocol['fold_calendar_definitions'])
    registration = authority.load_json(w.OUTPUT_DIR / (w.PREFIX + 'fold_registration.json'))
    for name, frame in [('fold_inventory', inventory), ('target_purge_audit', purges), ('feature_manifest', features)]:
        path = w.OUTPUT_DIR / (w.PREFIX + name + '.csv')
        if h.sha256(path) != registration['fold_sha256'][name] or path.read_text() != h.csv_text(frame):
            raise ValueError('Stage14 frozen fold/feature identity failed: ' + name)
    if len(inventory) != protocol['fold_count'] or sum(len(v) for _, v in members.values()) != protocol['expected_oof_dates']:
        raise ValueError('Stage14 fold/OOF count differs')
    if purges.violations.sum():
        raise ValueError('Stage14 target purge failed')
    return data, inventory, purges, members, features, evidence


def predictor_matrix(rows, treatment=False):
    expected = (*w.MATRIX_COLUMNS, *FUNDAMENTALS) if treatment else w.MATRIX_COLUMNS
    if tuple(rows.columns) != expected:
        raise ValueError('Exact Stage15 source schema required; forbidden feature/label/metadata')
    w.check_matrix(rows.loc[:, w.MATRIX_COLUMNS])
    x = representation.representation(rows, 'treatment')
    if treatment:
        x = pd.concat([x, baseline._numeric(rows, FUNDAMENTALS)], axis=1)
    if tuple(x) != (TREATMENT if treatment else CONTROL):
        raise ValueError('Stage15 registered predictor matrix changed')
    return x


def fit_treatment(family, training_x, target):
    """Bijective label adapter; reuse Stage10 All fitting without global edits."""
    if family not in ('ols', 'ridge') or tuple(training_x) != TREATMENT:
        raise ValueError('Fixed family and full registered Treatment block required')
    reverse = {new: old for old, new in representation.REPLACEMENTS.items()}
    candidates = (*representation.CONTROL, *FUNDAMENTALS)
    fitted = baseline.fit_linear(family + '_all', training_x.rename(columns=reverse), target, candidates)
    pp = fitted.preprocessing
    pp.features = tuple(representation.REPLACEMENTS.get(c, c) for c in pp.features)
    pp.medians = pp.medians.rename(index=representation.REPLACEMENTS)
    pp.usage['feature_name'] = pp.usage.feature_name.replace(representation.REPLACEMENTS)
    pp.scaler.feature_names_in_ = np.array(pp.features, dtype=object)
    return fitted


def build_models(data, inventory, members, *, treatment=False):
    """Fit only one registered representation, retaining all OOS dates."""
    predictor_matrix(data, treatment)
    keyed = data.set_index('date', drop=False)
    rep = 'relative_fundamentals' if treatment else 'relative'
    names = TREATMENTS if treatment else CONTROLS
    tables = {name: [] for name in ('oof_predictions', 'fold_metrics', 'preprocessing_audit',
                                   'model_parameters', 'extrapolation_audit', 'numerical_stability',
                                   'fit_membership', 'leakage_audit')}
    fits = {}
    with threadpool_limits(limits=1):
        for info in inventory.itertuples(index=False):
            fold = info.fold_id
            train_dates, valid_dates = members[fold]
            if (len(train_dates) != info.effective_train_rows or len(valid_dates) != info.validation_rows
                    or train_dates != sorted(set(train_dates)) or valid_dates != sorted(set(valid_dates))
                    or train_dates[0] != info.effective_train_start or train_dates[-1] != info.effective_train_end
                    or valid_dates[0] != info.validation_first_session or valid_dates[-1] != info.validation_last_session):
                raise ValueError('Frozen fold membership mismatch')
            training, valid = keyed.loc[train_dates], keyed.loc[valid_dates]
            if not training.target_exit_date_5d.lt(valid_dates[0]).all():
                raise ValueError('Training label not realized before validation')
            tx, vx = predictor_matrix(training, treatment), predictor_matrix(valid, treatment)
            y = baseline._target(valid[PRIMARY_TARGET])
            predictions = {} if treatment else w.baseline_predictions(training[PRIMARY_TARGET], len(valid))
            for family, name in zip(('ols', 'ridge'), names):
                fitted = (fit_treatment(family, tx, training[PRIMARY_TARGET]) if treatment else
                          representation.fit_arm(family, 'treatment', tx, training[PRIMARY_TARGET]))
                pp = fitted.preprocessing
                if fitted.train_dates != tuple(train_dates) or pp.fit_dates != tuple(train_dates):
                    raise ValueError('Training-only fit membership failed')
                pred = baseline.predict_linear(fitted, vx)
                fits[fold, name] = fitted
                predictions[name] = pred
                usage = pp.usage.assign(fold_id=fold, model=name, representation=rep)
                parameters = usage.copy()
                parameters['feature_position'] = np.arange(len(parameters))
                parameters['coefficient'] = parameters.feature_name.map(dict(zip(pp.features, map(float, fitted.estimator.coef_))))
                parameters['intercept'] = float(fitted.estimator.intercept_)
                parameters['ridge_alpha'] = 1.0 if family == 'ridge' else np.nan
                parameters['fit_target'] = PRIMARY_TARGET
                tables['model_parameters'].append(parameters)
                tables['preprocessing_audit'].append(usage)
                rebuilt = w.reconstruction(parameters, vx)
                stat_bad = 0
                for row in pp.usage.itertuples(index=False):
                    expected = baseline._audit_training_stats(tx[row.feature_name])
                    stat_bad += int(row.drop_reason != expected['drop_reason'])
                    for field in ('training_non_null_count', 'training_coverage', 'training_median',
                                  'training_mean_after_imputation', 'training_std_after_imputation', 'scaler_scale'):
                        stat_bad += int(not np.isclose(getattr(row, field), expected[field], rtol=1e-10, atol=1e-12, equal_nan=True))
                tables['leakage_audit'].append({
                    'fold_id': fold, 'model': name, 'training_only_stats_violations': stat_bad,
                    'prediction_reconstruction_violations': int(not np.allclose(rebuilt, pred, rtol=1e-10, atol=1e-12)),
                    'scaler_sample_count': int(pp.scaler.n_samples_seen_), 'effective_train_rows': len(training),
                    'fit_membership_violations': 0,
                    'forbidden_feature_violations': len(set(pp.features) - set(TREATMENT if treatment else CONTROL)),
                    'fit_target': PRIMARY_TARGET})
                tables['fit_membership'].append(pd.DataFrame({
                    'fold_id': fold, 'model': name, 'training_feature_date': train_dates,
                    'target_end_date': training.target_exit_date_5d.to_numpy(), 'validation_start': valid_dates[0]}))
                z = baseline.transform_features(pp, tx)
                design = np.column_stack([np.ones(len(z)), z])
                rank = int(np.linalg.matrix_rank(design))
                condition = float(np.linalg.cond(design))
                tables['numerical_stability'].append({
                    'fold_id': fold, 'model': name, 'initial_candidate_count': len(tx.columns),
                    'coverage_selected_count': int(pp.usage.training_coverage.ge(.50).sum()),
                    'zero_variance_removed_count': int(pp.usage.drop_reason.eq('zero_training_variance').sum()),
                    'effective_feature_count': len(pp.features), 'design_columns_including_intercept': design.shape[1],
                    'design_matrix_rank': rank, 'rank_deficiency': design.shape[1] - rank,
                    'condition_number': condition, 'condition_number_infinite': bool(np.isinf(condition)),
                    'coefficient_L2_norm': float(np.linalg.norm(fitted.estimator.coef_)),
                    'max_abs_coefficient': float(np.abs(fitted.estimator.coef_).max()),
                    'intercept': float(fitted.estimator.intercept_), 'fit_failure': False,
                    'diagnostic_only': True})
                if family == 'ols':
                    rows = w.extrapolation_rows(tx, vx, pp, fold, rep)
                    for row in rows:
                        row['feature_group'] = baseline.FEATURE_GROUPS.get(row['feature'], 'market')
                    tables['extrapolation_audit'].extend(rows)
                else:
                    ols = fits[fold, names[0]].preprocessing
                    if pp.features != ols.features or not np.array_equal(pp.scaler.mean_, ols.scaler.mean_) \
                            or not np.array_equal(pp.scaler.scale_, ols.scaler.scale_):
                        raise ValueError('OLS/Ridge preprocessing differs')
            tables['oof_predictions'].append(pd.DataFrame({
                'date': valid_dates, 'fold_id': fold, 'calendar_year': pd.to_datetime(valid_dates).year,
                'y_true': y, 'target_entry_date': valid.target_entry_date.to_numpy(),
                'target_exit_date_5d': valid.target_exit_date_5d.to_numpy(), **predictions}))
            for name, pred in predictions.items():
                tables['fold_metrics'].append({
                    'fold_id': fold, 'model': name, 'target': PRIMARY_TARGET,
                    'validation_first_session': valid_dates[0], 'validation_last_session': valid_dates[-1],
                    'train_rows': len(training), 'validation_rows': len(valid),
                    'MAE': float(np.abs(y - pred).mean()), 'bias': float((pred - y).mean()),
                    **w.prediction_stability(pred)})
    frames = {key: pd.concat(values, ignore_index=True) if values and isinstance(values[0], pd.DataFrame)
              else pd.DataFrame(values) for key, values in tables.items()}
    return w.WalkForwardResult(frames, fits)


def control_reproduction(replay, reference, tolerance):
    """Fail closed on missing/duplicate rows, actuals, provenance or fold drift."""
    keys = ['date', 'fold_id']
    identity = ['date', 'fold_id', 'calendar_year', 'y_true', 'target_entry_date', 'target_exit_date_5d']
    if replay.date.duplicated().any() or reference.date.duplicated().any():
        raise ValueError('Control reproduction duplicate date')
    # CSV reload widens calendar_year int32 to int64; compare exact cell values.
    if len(replay) != len(reference) or not all(np.array_equal(replay[c].to_numpy(), reference[c].to_numpy()) for c in identity):
        raise ValueError('Control reproduction coverage/fold/target/provenance mismatch')
    records = []
    for model in CONTROLS:
        part = replay.loc[:, keys].copy()
        part['model'] = model
        part['stage14_prediction'] = reference[model].to_numpy()
        part['stage15_replay_prediction'] = replay[model].to_numpy()
        part['absolute_difference'] = np.abs(part.stage15_replay_prediction - part.stage14_prediction)
        part['within_tolerance'] = np.isclose(part.stage15_replay_prediction, part.stage14_prediction, **tolerance)
        records.append(part)
    audit = pd.concat(records, ignore_index=True)
    anchors_equal = all(np.allclose(replay[m], reference[m], **tolerance) for m in BASELINES)
    summary = {'rows_checked': len(audit), 'OOF_dates': len(replay), 'coverage': 1.0,
               'unexpected_missing': 0, 'duplicates': int(audit.duplicated(['date', 'fold_id', 'model']).sum()),
               'max_absolute_difference': float(audit.absolute_difference.max()),
               'tolerance_violations': int((~audit.within_tolerance).sum()),
               'baselines_reproduced': anchors_equal, 'tolerance': tolerance}
    summary['passed'] = summary['tolerance_violations'] == 0 and anchors_equal
    return audit, summary


def feature_manifests(historical_features):
    dependency = h.read_csv(authority.OUTPUT_DIR / (authority.PREFIX + 'research_dependency_manifest.csv'))
    records = []
    for position, name in enumerate(FUNDAMENTALS):
        dep = dependency.loc[dependency.research_object.eq(name)]
        volume = dep.loc[dep.source_field.str.lower().eq('volume')]
        if dep.empty or len(volume) != 1 or (volume.used_directly | volume.used_indirectly).any():
            raise ValueError('Fundamental Volume/dependency evidence missing or forbidden')
        active = dep.loc[dep.used_directly | dep.used_indirectly, 'source_field'].tolist()
        if any('volume' in field.lower() for field in active):
            raise ValueError('Indirect Volume dependency forbidden')
        records.append({'feature_position': position, 'feature_name': name,
                        'feature_group': baseline.FEATURE_GROUPS[name], 'source_fields': '|'.join(active),
                        'Volume_dependency': False, 'definition': dep.description.iloc[0],
                        'implementation_authority': 'research_features.build_feature_matrix',
                        'implementation_sha256': h.sha256(PROJECT_ROOT / 'src/nasdaq_research/research_features.py')})
    control = historical_features.loc[historical_features.representation.eq('relative')].copy().reset_index(drop=True)
    control['feature_group'] = 'market'
    fundamental = pd.DataFrame(records)
    treatment = pd.concat([control.drop(columns='representation'), fundamental], ignore_index=True)
    treatment['feature_position'] = np.arange(len(treatment))
    treatment['representation'] = 'relative_fundamentals'
    if tuple(treatment.feature_name) != TREATMENT or treatment.Volume_dependency.any():
        raise ValueError('Treatment feature manifest differs')
    return control, fundamental, treatment


def read_filing_payload(path, columns):
    """Filter future filing records before numeric parsing, as for market rows."""
    records = []
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(columns) <= set(reader.fieldnames or ()):
            raise ValueError('Canonical filing schema missing')
        for row in reader:
            if row['filing_date'] < w.DEVELOPMENT_END:
                records.append([row[c] for c in columns])
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(records)
    buffer.seek(0)
    return pd.read_csv(buffer, float_precision='round_trip')


def load_fundamentals(data, protocol):
    authority.require_canonical_vintage(protocol['canonical_vintage_id'], required_features=TREATMENT,
                                       required_target=PRIMARY_TARGET)
    provenance = tuple(f'{p}_{c}' for p in ('q', 'fy', 'bs')
                       for c in ('accession', 'filing_date', 'effective_date', 'period_start', 'period_end', 'form', 'data_period_type'))
    reference = tuple(f'{p}_yoy_reference_{c}' for p in ('q', 'fy')
                      for c in ('accession', 'filing_date', 'period_start', 'period_end'))
    bs_source = tuple('bs_' + c for c in rf.BALANCE_FIELDS)
    columns = ('date', *FUNDAMENTALS, *provenance, *reference, *bs_source)
    daily = w.read_selected(h.OUTPUT_DIR / 'NVDA_stage13_research_matrix.csv', columns)
    if not daily.date.equals(data.date):
        raise ValueError('PIT rows differ from inherited Market/target population')
    augmented = pd.concat([data, daily.loc[:, FUNDAMENTALS]], axis=1)
    predictor_matrix(augmented, True)
    snapshots_path = h.OUTPUT_DIR / 'NVDA_stage13_fundamental_snapshots.csv'
    with snapshots_path.open() as stream:
        snap_columns = tuple(next(csv.reader(stream)))
    snapshots = read_filing_payload(snapshots_path, snap_columns)
    facts_path = h.OUTPUT_DIR / 'NVDA_stage13_fundamental_history.csv'
    with facts_path.open() as stream:
        fact_columns = tuple(next(csv.reader(stream)))
    facts = read_filing_payload(facts_path, fact_columns)
    # Dates only across the entire presample activation calendar, no future X/Y.
    sessions = w.read_sessions(h.OUTPUT_DIR / 'raw/NVDA_market.csv')
    sessions = [day for day in sessions if day <= w.DEVELOPMENT_END]
    return augmented, daily, snapshots, facts, sessions


def replay_pit_values(facts, sessions):
    """Isolated replay using unchanged Stage6 features and Stage5 selectors."""
    snapshots = rf.build_fundamental_features(facts)
    enriched = facts.merge(snapshots.loc[:, [*KEY_COLUMNS, *rf.SNAPSHOT_PAYLOAD_COLUMNS]],
                           on=list(KEY_COLUMNS), how='left', validate='one_to_one')
    market = pd.DataFrame({'date': list(sessions), 'ticker': 'NVDA'})
    groups = {'q': (*rf.QUARTERLY_RATIO_FEATURES, *rf.QUARTERLY_GROWTH_FEATURES),
              'fy': (*rf.ANNUAL_RATIO_FEATURES, *rf.ANNUAL_GROWTH_FEATURES),
              'bs': tuple(rf.BALANCE_RATIOS)}
    masks = {'q': enriched.data_period_type.eq('quarterly'), 'fy': enriched.data_period_type.eq('annual'),
             'bs': enriched.data_period_type.isin(['quarterly', 'annual']) & enriched.loc[:, rf.BALANCE_FIELDS].notna().any(axis=1)}
    result = market.copy()
    for prefix, names in groups.items():
        state = build_research_dataset(market, enriched.loc[masks[prefix]], allow_empty=True)
        for name in names:
            result[name] = state[name]
        for name in ('accession', 'filing_date', 'effective_date'):
            result[prefix + '_' + name] = state['fundamental_' + name]
    return result


def pit_audit(daily, snapshots, facts, sessions):
    """Audit every cell against the frozen snapshot and independent chronology."""
    keyed = snapshots.set_index(list(KEY_COLUMNS), drop=False)
    records = []
    days = pd.DatetimeIndex(parse_dates(pd.Series(sessions)))
    for row in daily.to_dict('records'):
        feature_day = row['date']
        for name in FUNDAMENTALS:
            prefix = 'q' if name in (*rf.QUARTERLY_RATIO_FEATURES, *rf.QUARTERLY_GROWTH_FEATURES) else \
                     'fy' if name in (*rf.ANNUAL_RATIO_FEATURES, *rf.ANNUAL_GROWTH_FEATURES) else 'bs'
            accession = row[prefix + '_accession']
            actual = row[name]
            if pd.isna(accession):
                records.append({'feature_date': feature_day, 'feature': name, 'state_family': prefix,
                                'source_filing': None, 'filing_date': None, 'effective_date': None,
                                'expected_effective_date': None, 'eligible_on_feature_date': False,
                                'feature_available': False, 'canonical_state_exact': pd.isna(actual),
                                'reference_filing_date': None, 'violations': int(pd.notna(actual))})
                continue
            # Use the actual repository KEY_COLUMNS order, not a reconstructed key.
            identity = {c: ('NVDA' if c == 'ticker' else row[prefix + '_' + c]) for c in KEY_COLUMNS}
            source = keyed.loc[tuple(identity[c] for c in KEY_COLUMNS)]
            if isinstance(source, pd.DataFrame):
                raise ValueError('Ambiguous frozen filing identity')
            filed = row[prefix + '_filing_date']
            effective = row[prefix + '_effective_date']
            pos = days.searchsorted(pd.Timestamp(filed), side='right')
            expected_effective = days[pos].strftime('%Y-%m-%d') if pos < len(days) else None
            eligible = filed < feature_day and isinstance(effective, str) and effective <= feature_day
            exact = (pd.isna(actual) and pd.isna(source[name])) or actual == source[name]
            ref = source.yoy_reference_filing_date if 'growth_yoy' in name else None
            ref_ok = pd.isna(ref) or ref <= filed
            metadata_ok = all(h.scalar_equal(row[prefix + '_' + c], source[c])
                              for c in ('accession', 'filing_date', 'period_start', 'period_end', 'form', 'data_period_type'))
            if 'growth_yoy' in name:
                ref_ok = ref_ok and all(h.scalar_equal(row[prefix + '_yoy_reference_' + c], source['yoy_reference_' + c])
                                        for c in ('accession', 'filing_date', 'period_start', 'period_end'))
            bad = int(not eligible) + int(effective != expected_effective) + int(not exact) + int(not ref_ok) + int(not metadata_ok)
            records.append({'feature_date': feature_day, 'feature': name, 'state_family': prefix,
                            'source_filing': accession, 'filing_date': filed, 'effective_date': effective,
                            'expected_effective_date': expected_effective, 'eligible_on_feature_date': eligible,
                            'feature_available': pd.notna(actual), 'canonical_state_exact': bool(exact),
                            'reference_filing_date': ref, 'violations': bad})
    audit = pd.DataFrame(records)
    # No as-of joins in this independent raw-source identity/BS formula oracle.
    oracle = h.state_oracle(daily, facts, list(range(len(daily))), pd.Series(sessions))
    growth = h.growth_oracle(snapshots, facts)
    replayed = replay_pit_values(facts, sessions).set_index('date').loc[daily.date]
    values = daily.loc[:, FUNDAMENTALS].to_numpy(dtype=float)
    regenerated = replayed.loc[:, FUNDAMENTALS].to_numpy(dtype=float)
    replay_bad = int((~np.isclose(values, regenerated, rtol=1e-10, atol=1e-12, equal_nan=True)).sum())
    summary = {'cell_checks': len(audit), 'future_information_violations': int(audit.violations.sum()),
               'canonical_exact_state_cells': int(audit.canonical_state_exact.sum()),
               'independent_state_oracle': oracle, 'growth_reference_oracle': growth,
               'replayed_state_cells': values.size, 'replay_violations': replay_bad,
               'future_numeric_filings_loaded': False, 'numeric_locked_payload_loaded': False}
    summary['passed'] = not (summary['future_information_violations'] or oracle['violations']
                             or growth['violations'] or replay_bad)
    return audit, summary


def fundamental_coverage(data, members, preprocessing):
    records = []
    keyed = data.set_index('date')
    for fold, (train, valid) in members.items():
        usage = preprocessing.loc[preprocessing.fold_id.eq(fold) & preprocessing.model.eq(TREATMENTS[0])].set_index('feature_name')
        for name in FUNDAMENTALS:
            tr, va = keyed.loc[train, name], keyed.loc[valid, name]
            row = usage.loc[name]
            present = data.loc[data[name].notna(), 'date']
            records.append({'fold_id': fold, 'feature': name, 'feature_group': baseline.FEATURE_GROUPS[name],
                            'train_nonmissing_count': int(tr.notna().sum()), 'train_rows': len(train),
                            'train_coverage': float(tr.notna().mean()),
                            'coverage_eligible': bool(row.training_coverage >= .50),
                            'selected': bool(row.used), 'drop_reason': row.drop_reason,
                            'training_median': row.training_median, 'validation_rows': len(valid),
                            'validation_nonmissing_count': int(va.notna().sum()),
                            'validation_missing_fraction': float(va.isna().mean()),
                            'first_available_date': present.iloc[0] if len(present) else None,
                            'first_available_date_scope': 'descriptive development-window metadata; never used by selector'})
    return pd.DataFrame(records)


def merge_results(control, treatment):
    identity = ['date', 'fold_id', 'calendar_year', 'y_true', 'target_entry_date', 'target_exit_date_5d']
    c, t = control.tables['oof_predictions'], treatment.tables['oof_predictions']
    if len(c) != len(t) or any(not np.array_equal(c[name], t[name]) for name in identity):
        raise ValueError('Treatment OOF population differs from reproduced Control')
    tables = {key: pd.concat([control.tables[key], treatment.tables[key]], ignore_index=True)
              for key in control.tables if key != 'oof_predictions'}
    tables['oof_predictions'] = c.copy()
    for name in TREATMENTS:
        tables['oof_predictions'][name] = t[name].to_numpy()
    return w.WalkForwardResult(tables, {**control.fits, **treatment.fits})


def paired_comparison(metrics, model, reference):
    per = metrics.loc[metrics.model.eq(model), ['fold_id', 'MAE']].merge(
        metrics.loc[metrics.model.eq(reference), ['fold_id', 'MAE']], on='fold_id',
        suffixes=('_treatment', '_reference'), validate='one_to_one')
    if len(per) != 22:
        raise ValueError('All 22 paired fold metrics required')
    delta = per.MAE_treatment - per.MAE_reference
    per = per.assign(scope='fold', model=model, reference=reference, paired_delta_MAE=delta,
                     win=delta.lt(0), loss=delta.gt(0), tie=delta.eq(0))
    aggregate = {'scope': 'aggregate', 'fold_id': 'ALL', 'model': model, 'reference': reference,
                 'fold_count': len(per), 'mean_treatment_MAE': float(per.MAE_treatment.mean()),
                 'mean_reference_MAE': float(per.MAE_reference.mean()), **w.comparison_counts(delta)}
    return per, aggregate


def aggregate_results(oof, metrics):
    summaries, incremental, absolute, years = [], [], [], []
    for name in MODELS:
        per = metrics.loc[metrics.model.eq(name)]
        summaries.append({'model': name, 'fold_count': len(per), 'OOF_rows': len(oof),
                          'mean_fold_MAE': float(per.MAE.mean()), 'median_fold_MAE': float(per.MAE.median()),
                          'pooled_OOF_MAE': float(np.abs(oof.y_true - oof[name]).mean())})
    for name, control in zip(TREATMENTS, CONTROLS):
        per, summary = paired_comparison(metrics, name, control)
        incremental.extend([per, pd.DataFrame([summary])])
        for reference in BASELINES:
            per, summary = paired_comparison(metrics, name, reference)
            absolute.extend([per, pd.DataFrame([summary])])
    for year, block in oof.groupby('calendar_year', sort=True):
        maes = {model: float(np.abs(block.y_true - block[model]).mean()) for model in MODELS}
        for model in MODELS:
            control = dict(zip(TREATMENTS, CONTROLS)).get(model)
            years.append({'calendar_year': year, 'model': model, 'rows': len(block),
                          'pooled_yearly_MAE': maes[model],
                          'paired_delta_vs_control': maes[model] - maes[control] if control else np.nan,
                          'delta_vs_zero': maes[model] - maes['zero_return'],
                          'delta_vs_historical_mean': maes[model] - maes['historical_mean'],
                          'descriptive_only': True})
    return (pd.DataFrame(summaries), pd.concat(incremental, ignore_index=True),
            pd.concat(absolute, ignore_index=True), pd.DataFrame(years))


def incremental_assessment(comparison):
    gates, broad = {}, {}
    rows = comparison.loc[comparison.scope.eq('aggregate')].set_index('model')
    if set(rows.index) != set(TREATMENTS):
        raise ValueError('Both incremental family comparisons required')
    for name in TREATMENTS:
        row = rows.loc[name]
        tests = {'mean_paired_delta_negative': bool(row.mean_delta_MAE < 0),
                 'median_paired_delta_negative': bool(row.median_delta_MAE < 0),
                 'strict_majority_fold_wins': bool(row.win_rate > .50)}
        gates[name] = {**tests, 'all_three_passed': all(tests.values())}
        broad[name] = tests['mean_paired_delta_negative'] and tests['strict_majority_fold_wins']
    category = ('ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE' if any(v['all_three_passed'] for v in gates.values())
                else 'NO_ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE' if not any(broad.values())
                else 'MIXED_OR_REGIME_DEPENDENT_INCREMENTAL_VALUE')
    return {'classification': category, 'model_criteria': gates, 'broad_OOS_improvement': broad}


def absolute_assessment(comparison):
    # Rename only model identities, delegate all six gates and precedence to Stage14.
    rows = comparison.loc[comparison.scope.eq('aggregate')].rename(columns={'reference': 'baseline'}).copy()
    mapping = dict(zip(TREATMENTS, CONTROLS))
    rows['model'] = rows.model.replace(mapping)
    result = w.predictive_assessment(rows)
    reverse = {v: k for k, v in mapping.items()}
    for field in ('model_criteria', 'four_mean_majority_gates'):
        result[field] = {reverse[name]: value for name, value in result[field].items()}
    return result


def validate_result(data, inventory, purges, members, result, reproduction, pit):
    tables = result.tables
    oof, metrics, leak = (tables[name] for name in ('oof_predictions', 'fold_metrics', 'leakage_audit'))
    expected = [day for _, valid in members.values() for day in valid]
    expected_folds = [fold for fold, (_, valid) in members.items() for _ in valid]
    violations = {name: int(leak[name].sum()) for name in
                  ('training_only_stats_violations', 'prediction_reconstruction_violations',
                   'fit_membership_violations', 'forbidden_feature_violations')}
    checks = {
        'protocol_frozen': True, 'Stage14_control_reproduced': reproduction['passed'],
        'fold_count': len(inventory) == 22, 'target_purge': int(purges.violations.sum()) == 0,
        'PIT_leakage': pit['passed'], 'training_only_preprocessing': not any(violations.values()),
        'scaler_sample_count': bool(leak.scaler_sample_count.eq(leak.effective_train_rows).all()),
        'OOF_exact_dates_and_folds': oof.date.tolist() == expected and oof.fold_id.tolist() == expected_folds,
        'OOF_count': len(oof) == 1367, 'OOF_unique': not oof.date.duplicated().any(),
        'OOF_complete_finite': bool(np.isfinite(oof.loc[:, ['y_true', *MODELS]].to_numpy(dtype=float)).all()),
        'OOF_six_registered_models': tuple(oof.columns[-6:]) == MODELS,
        'learned_fit_count': len(result.fits) == 88,
        'metric_grid': len(metrics) == 132 and not metrics.duplicated(['fold_id', 'model']).any() and set(metrics.model) == set(MODELS),
        'primary_target_only': bool(metrics.target.eq(PRIMARY_TARGET).all()),
        'no_complete_case_sample_selection': len(data) == 1620 and oof.date.tolist() == expected,
        'Volume_excluded': 'volume' not in predictor_matrix(data, True),
        'finite_fits': all(np.isfinite(f.estimator.coef_).all() and np.isfinite(f.estimator.intercept_) for f in result.fits.values()),
        'OOF_canonical_target_and_provenance': True, 'baseline_definitions': True, 'metrics_correct': True,
        'saved_parameters_reconstruct': True, 'saved_preprocessing_training_only': True,
        'fit_membership_exact': True}
    keyed = data.set_index('date', drop=False)
    checks['OOF_canonical_target_and_provenance'] = all(
        np.array_equal(oof[field].to_numpy(), keyed.loc[oof.date, source].to_numpy())
        for field, source in [('y_true', PRIMARY_TARGET), ('target_entry_date', 'target_entry_date'),
                              ('target_exit_date_5d', 'target_exit_date_5d')])
    for fold, (train, valid) in members.items():
        block = oof.loc[oof.fold_id.eq(fold)]
        anchors = w.baseline_predictions(keyed.loc[train, PRIMARY_TARGET], len(valid))
        checks['baseline_definitions'] &= all(np.array_equal(block[name], values) for name, values in anchors.items())
        for model in MODELS:
            row = metrics.loc[metrics.fold_id.eq(fold) & metrics.model.eq(model)]
            if len(row) != 1:
                checks['metrics_correct'] = False
                continue
            checks['metrics_correct'] &= bool(row.train_rows.iloc[0] == len(train) and row.validation_rows.iloc[0] == len(valid)
                and np.isclose(float(np.abs(block.y_true - block[model]).mean()), row.MAE.iloc[0], rtol=0, atol=1e-12))
        for model in (*CONTROLS, *TREATMENTS):
            parameters = tables['model_parameters'].loc[tables['model_parameters'].fold_id.eq(fold) & tables['model_parameters'].model.eq(model)]
            source = keyed if model in TREATMENTS else keyed.loc[:, w.MATRIX_COLUMNS]
            tx, vx = predictor_matrix(source.loc[train], model in TREATMENTS), predictor_matrix(source.loc[valid], model in TREATMENTS)
            checks['saved_parameters_reconstruct'] &= bool(np.allclose(w.reconstruction(parameters, vx), block[model], rtol=1e-10, atol=1e-12))
            checks['saved_preprocessing_training_only'] &= tuple(parameters.feature_name) == tuple(tx)
            for p in parameters.itertuples(index=False):
                expected_stats = baseline._audit_training_stats(tx[p.feature_name])
                checks['saved_preprocessing_training_only'] &= p.drop_reason == expected_stats['drop_reason']
                checks['saved_preprocessing_training_only'] &= all(np.isclose(getattr(p, f), expected_stats[f], rtol=1e-10, atol=1e-12, equal_nan=True)
                    for f in ('training_non_null_count', 'training_coverage', 'training_median',
                              'training_mean_after_imputation', 'training_std_after_imputation', 'scaler_scale'))
            membership = tables['fit_membership'].loc[tables['fit_membership'].fold_id.eq(fold) & tables['fit_membership'].model.eq(model)]
            checks['fit_membership_exact'] &= membership.training_feature_date.tolist() == train
            checks['fit_membership_exact'] &= membership.target_end_date.tolist() == keyed.loc[train, 'target_exit_date_5d'].tolist()
    return {'valid': all(checks.values()), 'checks': checks, 'violations': violations,
            'expected_OOF_sessions': len(expected), 'actual_OOF_sessions': len(oof),
            'missing_OOF_sessions': len(set(expected) - set(oof.date)),
            'duplicate_OOF_sessions': int(oof.date.duplicated().sum()),
            'prediction_reconstruction_model_folds': len(result.fits)}


def run_pipeline(output_dir=OUTPUT_DIR):
    """Offline formal execution: Control/PIT gates precede Treatment fitting."""
    output_dir = Path(output_dir).resolve()
    if output_dir != OUTPUT_DIR.resolve() and (output_dir == DATA_DIR or output_dir in DATA_DIR.parents or output_dir.is_relative_to(DATA_DIR)):
        raise ValueError('Alternative outputs must be isolated from upstream data')
    protocol = load_protocol()
    upstream = verify_upstream()
    if upstream['mutation_violations']:
        raise ValueError('Protected upstream input changed')
    with authority.offline_only():
        data, inventory, purges, members, historical_features, evidence = load_fold_context(protocol)
        control = build_models(data, inventory, members)
        replay, reproduction = control_reproduction(control.tables['oof_predictions'],
            h.read_csv(w.OUTPUT_DIR / (w.PREFIX + 'oof_predictions.csv')), protocol['control_reproduction_tolerance'])
        freeze(output('stage14_control_reproduction', output_dir, 'csv'), h.csv_text(replay))
        freeze(output('control_reproduction_summary', output_dir), h.json_text(reproduction))
        if not reproduction['passed']:
            raise ValueError('Stage15 BLOCKED: Stage14 Control reproduction failed; Treatment forbidden')
        control_manifest, fundamental_manifest, treatment_manifest = feature_manifests(historical_features)
        data, daily, snapshots, facts, sessions = load_fundamentals(data, protocol)
        pit_table, pit = pit_audit(daily, snapshots, facts, sessions)
        if not pit['passed']:
            h.write(output_dir / 'engineering_attempts/PIT_blocker.json', h.json_text(pit))
            raise ValueError('Stage15 BLOCKED: PIT source/state chronology audit failed')
        freeze(output('pit_leakage_audit', output_dir, 'csv'), h.csv_text(pit_table))
        freeze(output('pit_state_replay_audit', output_dir), h.json_text(pit))
        # Coverage evidence is calculated before Treatment predictions; estimator
        # code still uses unchanged training-only Stage10 preprocessing per model.
        coverage_pre = []
        keyed = data.set_index('date', drop=False)
        for fold, (train, _) in members.items():
            reverse = {new: old for old, new in representation.REPLACEMENTS.items()}
            pp = baseline.fit_preprocessor(predictor_matrix(keyed.loc[train], True).rename(columns=reverse),
                                           (*representation.CONTROL, *FUNDAMENTALS))
            pp.usage['feature_name'] = pp.usage.feature_name.replace(representation.REPLACEMENTS)
            coverage_pre.append(pp.usage.assign(fold_id=fold, model=TREATMENTS[0]))
        coverage = fundamental_coverage(data, members, pd.concat(coverage_pre, ignore_index=True))
        freeze(output('fundamental_coverage', output_dir, 'csv'), h.csv_text(coverage))
        treatment = build_models(data, inventory, members, treatment=True)
        result = merge_results(control, treatment)
        tables = result.tables
        actual_coverage = fundamental_coverage(data, members, tables['preprocessing_audit'])
        if not coverage.equals(actual_coverage):
            raise ValueError('Pre-model coverage differs from fitted Treatment selection')
        validation = validate_result(data, inventory, purges, members, result, reproduction, pit)
        lock = w.final_lock_audit(data, inventory, result, evidence)
        lock.update({**SAFETY, 'fundamentals_used': True})
        if not validation['valid'] or not lock['valid']:
            raise ValueError('Stage15 engineering audit failed: ' + h.json_text({'validation': validation, 'lock': lock}))
        summaries, incremental, absolute, years = aggregate_results(tables['oof_predictions'], tables['fold_metrics'])
        inc, signal = incremental_assessment(incremental), absolute_assessment(absolute)
        stability = tables['fold_metrics'].loc[:, ['fold_id', 'model', *w.prediction_stability(np.array([0., 1.])).keys()]]
        tables.update({'fold_inventory': inventory, 'target_purge_audit': purges,
                       'control_feature_manifest': control_manifest, 'fundamental_feature_manifest': fundamental_manifest,
                       'treatment_feature_manifest': treatment_manifest, 'fundamental_coverage': coverage,
                       'model_summary': summaries, 'incremental_comparison': incremental,
                       'absolute_baseline_comparison': absolute, 'yearly_diagnostics': years,
                       'prediction_stability': stability})
        # Validate the persisted decimal parser contract as well as in-memory fits.
        texts = {name: h.csv_text(frame) for name, frame in tables.items()}
        loaded = {name: pd.read_csv(StringIO(text), float_precision='round_trip') for name, text in texts.items()}
        roundtrip = validate_result(data, inventory, purges, members, w.WalkForwardResult(loaded, result.fits), reproduction, pit)
        if not roundtrip['valid']:
            raise ValueError('Stage15 CSV round-trip validation failed: ' + h.json_text(roundtrip))
        upstream = verify_upstream()
        if upstream['mutation_violations']:
            raise ValueError('Upstream mutation detected; refusing formal output')
        evidence.update({**SAFETY, 'fundamentals_used': True, 'fundamental_feature_count': len(FUNDAMENTALS),
                         'initial_treatment_feature_count': len(TREATMENT), 'PIT_audit': pit})
        summary = {'stage_name': protocol['stage_name'], 'engineering_status': 'PASS',
                   'engineering_scope': 'Pipeline audits; complete stage acceptance additionally requires execution_verification.json tests and two-run reproducibility',
                   'incremental_fundamental_value_classification': inc['classification'],
                   'absolute_predictive_signal_classification': signal['classification'],
                   'incremental_assessment': inc, 'absolute_predictive_assessment': signal,
                   'canonical_vintage_id': protocol['canonical_vintage_id'], 'development_rows': len(data),
                   'fold_count': len(inventory), 'OOF_sessions': len(tables['oof_predictions']),
                   'learned_model_fits': len(result.fits), 'control_reproduction': reproduction, 'PIT_audit': pit,
                   'candidate_counts': {'control': len(CONTROL), 'fundamentals': len(FUNDAMENTALS), 'treatment': len(TREATMENT)},
                   'model_summary': summaries.to_dict('records'),
                   'incremental_comparisons': incremental.loc[incremental.scope.eq('aggregate')].to_dict('records'),
                   'absolute_comparisons': absolute.loc[absolute.scope.eq('aggregate')].to_dict('records'),
                   'numerical_summary': {
                       'treatment_rank_deficient_fits': int(tables['numerical_stability'].loc[tables['numerical_stability'].model.isin(TREATMENTS), 'rank_deficiency'].gt(0).sum()),
                       'fit_failures': int(tables['numerical_stability'].fit_failure.sum()),
                       'thresholds_descriptive_only': True},
                   'upstream': upstream, 'statistical_inference_performed': False, **SAFETY}
        validation.update({'engineering_status': 'PASS',
                           'canonical_input': evidence, 'Control_reproduction': reproduction, 'PIT': pit,
                           'Final_Test': lock, 'upstream': upstream, 'csv_roundtrip_validated': True,
                           'tests_and_two_run_verification_authority': PREFIX + 'execution_verification.json', **SAFETY})
        for name, text in texts.items():
            freeze(output(name, output_dir, 'csv'), text)
        for name, obj in [('canonical_input_audit', evidence), ('final_test_lock_audit', lock),
                          ('summary', summary), ('validation', validation)]:
            freeze(output(name, output_dir), h.json_text(obj))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', type=Path, help='Initial repository-state JSON captured before any changes')
    args = parser.parse_args()
    if args.register:
        print(h.json_text(register(args.register)))
    else:
        print(h.json_text(run_pipeline()))


if __name__ == '__main__':
    main()
