"""Stage 14: frozen, quarterly expanding-window validation, offline.

Stage 11 supplies representation and fit_arm; Stage 10 supplies all fitting
and preprocessing semantics. Only the predeclared development rows and primary
5d labels reach model code. Protocol/fold authority cannot be rewritten.
"""

import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nasdaq_research import data_authority as authority
from nasdaq_research import historical_expansion as h
from nasdaq_research import market_representation as representation
from nasdaq_research import modeling as baseline
from nasdaq_research.config import PROJECT_ROOT, DATA_DIR
from nasdaq_research.dates import parse_dates
from nasdaq_research.targets import PRIMARY_TARGET

OUTPUT_DIR = DATA_DIR / 'research/stage14_multi_year_walk_forward'
PREFIX = 'NVDA_stage14_'
PROTOCOL_SHA256 = 'e6787c175a077a3e422d9533df00369e0450c35532b7c67276fbaeee21446908'
UPSTREAM_SHA256 = '6b9a4b082207d8ca86ebb2ea960a51e95b4d36df97fed805e8f30a87d84891bf'
DEVELOPMENT_START, DEVELOPMENT_END = '2020-01-02', '2026-06-12'
GAP_START, GAP_END = '2026-06-15', '2026-07-14'
FINAL_START, FINAL_END = '2026-07-15', '2026-09-23'
MODELS = ('zero_return', 'historical_mean', 'original_ols', 'original_ridge_alpha1',
          'relative_ols', 'relative_ridge_alpha1')
LEARNED = MODELS[2:]
ARMS = {'original': 'control', 'relative': 'treatment'}
MODEL_SPECS = {'original_ols': ('original','ols'), 'original_ridge_alpha1': ('original','ridge'),
               'relative_ols': ('relative','ols'), 'relative_ridge_alpha1': ('relative','ridge')}
X_SOURCE = ('date','ticker','open','close',*representation.CONTROL,*representation.REPLACEMENTS.values())
LABEL_COLUMNS = ('date','target_entry_date','target_entry_open','target_exit_date_5d',
                 'target_exit_close_5d',PRIMARY_TARGET)
MATRIX_COLUMNS = (*X_SOURCE, 'target_entry_date','target_exit_date_5d',PRIMARY_TARGET)
SAFETY = {**representation.SAFETY, 'network_allowed':False,'network_requests':0,
          'hyperparameter_tuning_allowed':False,'fundamentals_used':False,'volume_used':False,
          'secondary_targets_modeled':False,'final_model_fitted':False,'new_stage_design':False,
          'legacy_root_cause_reinvestigated':False}
TABLE_NAMES = ('feature_manifest','fold_inventory','target_purge_audit','preprocessing_audit',
               'oof_predictions','fold_metrics','model_summary','baseline_comparison',
               'representation_comparison','extrapolation_audit','paired_extrapolation_summary',
               'prediction_stability','yearly_diagnostics','model_parameters','fit_membership',
               'leakage_audit')


def load_protocol():
    path=OUTPUT_DIR/(PREFIX+'protocol.json')
    if h.sha256(path)!=PROTOCOL_SHA256:
        raise ValueError('Stage14 preregistered protocol changed')
    return authority.load_json(path)


def verify_upstream():
    path=OUTPUT_DIR/(PREFIX+'upstream_sha256.json')
    if h.sha256(path)!=UPSTREAM_SHA256:
        raise ValueError('Stage14 protected-file manifest changed')
    manifest=authority.load_json(path)
    changed=[name for name,digest in manifest.items() if not (PROJECT_ROOT/name).is_file()
             or h.sha256(PROJECT_ROOT/name)!=digest]
    return {'files_checked':len(manifest),'mutation_violations':len(changed),'changed_files':changed}


def freeze(path,contents):
    """Idempotent formal write: subsequent runs may only reproduce exact bytes."""
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=contents.encode('utf-8'):
            raise ValueError('Frozen Stage14 artifact differs: '+path.name)
    else:
        h.write(path,contents)


def read_selected(path,columns,*,end=DEVELOPMENT_END,start=None):
    """Filter lexical dates BEFORE parsing numeric cells, including y.

    CSV streaming necessarily traverses the frozen file. Rejected-row payloads
    are never parsed into numeric features/labels or returned to modeling code.
    This function accepts no path for the historical final training pool.
    """
    records=[]
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        if not set(columns)<=set(reader.fieldnames or ()):
            raise ValueError('Canonical input schema missing registered fields')
        for row in reader:
            day=row['date']
            if day<=end and (start is None or day>=start):
                records.append([row[c] for c in columns])
    text=StringIO()
    writer=csv.writer(text,lineterminator='\n')
    writer.writerow(columns);writer.writerows(records);text.seek(0)
    return pd.read_csv(text,float_precision='round_trip')


def read_sessions(path):
    with Path(path).open(newline='') as stream:
        return [r['date'] for r in csv.DictReader(stream)]


def check_matrix(data):
    if data.empty or list(data.columns)!=list(MATRIX_COLUMNS) or data.columns.duplicated().any():
        raise ValueError('Exact Market-only development matrix required; Volume/fundamentals/secondary labels forbidden')
    days=parse_dates(data.date).dt.strftime('%Y-%m-%d')
    if days.duplicated().any() or not days.is_monotonic_increasing:
        raise ValueError('Unique sorted development sessions required')
    if not days.between(DEVELOPMENT_START,DEVELOPMENT_END).all():
        raise ValueError('Gap/Final Test/tail rows forbidden in development matrix')
    if not data.ticker.eq('NVDA').all():
        raise ValueError('Only NVDA is registered')
    baseline._target(data[PRIMARY_TARGET])
    for arm in ARMS.values():
        representation.representation(data,arm)


def feature_manifest(dependencies):
    objects=(*representation.CONTROL,*representation.REPLACEMENTS.values())
    selected=dependencies.loc[dependencies.research_object.isin(objects)]
    volume=selected.loc[selected.source_field.str.lower().eq('volume')]
    if len(volume)!=len(objects) or (volume.used_directly|volume.used_indirectly).any():
        raise ValueError('Missing or used Volume dependency in registered representation')
    records=[]
    for rep,arm in ARMS.items():
        for position,name in enumerate(representation.SPECS[arm]):
            dep=selected.loc[selected.research_object.eq(name)]
            if dep.empty:
                raise ValueError('Feature dependency evidence missing: '+name)
            fields=dep.loc[dep.used_directly|dep.used_indirectly,'source_field'].tolist()
            if 'volume' in fields or any(field.startswith('SEC.') for field in fields):
                raise ValueError('Volume/fundamental dependency forbidden')
            original=next((old for old,new in representation.REPLACEMENTS.items() if new==name),name)
            records.append({'representation':rep,'feature_position':position,'feature_name':name,
                            'paired_original_feature':original,'substitution':name!=original,
                            'source_fields':'|'.join(fields),'Volume_dependency':False,
                            'definition':dep.description.iloc[0],
                            'implementation_authority':'market_representation.representation',
                            'implementation_sha256':h.sha256(PROJECT_ROOT/'src/nasdaq_research/market_representation.py')})
    return pd.DataFrame(records)


def target_source_oracle(data,labels,sessions,prices):
    """Observed-position and scalar-price oracle, independent of target builder."""
    positions={day:i for i,day in enumerate(sessions)}
    price=prices.set_index('date'); keyed=labels.set_index('date')
    records=[]
    for row in data.itertuples(index=False):
        i=positions[row.date];entry=sessions[i+1];end=sessions[i+5]
        label=keyed.loc[row.date]
        expected=float(price.loc[end,'close'])/float(price.loc[entry,'open'])-1
        violations=int(label.target_entry_date!=entry or label.target_exit_date_5d!=end
                       or label.target_entry_open!=price.loc[entry,'open']
                       or label.target_exit_close_5d!=price.loc[end,'close']
                       or not np.isclose(label[PRIMARY_TARGET],expected,rtol=0,atol=1e-12))
        records.append({'date':row.date,'expected_entry_date':entry,'expected_exit_date_5d':end,
                        'violations':violations})
    return pd.DataFrame(records)


def construct_folds(data,sessions,definitions):
    """Chronological expanding folds with independently positioned 5d purges."""
    check_matrix(data)
    keyed=data.set_index('date',drop=False)
    positions={day:i for i,day in enumerate(sessions)}
    inventory=[];purges=[];members={}
    previous_valid=set();previous_train=set()
    for definition in definitions:
        fold=definition['fold_id']
        block=data.loc[data.date.between(definition['validation_calendar_start'],definition['validation_calendar_end'])]
        if block.empty:
            raise ValueError('Declared validation fold has no observed sessions: '+fold)
        first,last=block.date.iloc[0],block.date.iloc[-1]
        candidate=data.loc[data.date.lt(first)]
        training=[];fold_purges=[]
        for row in candidate.itertuples(index=False):
            i=positions[row.date]
            entry=sessions[i+1];end=sessions[i+5]
            provenance_ok=row.target_entry_date==entry and row.target_exit_date_5d==end
            eligible=provenance_ok and row.date<first and end<first and pd.notna(getattr(row,PRIMARY_TARGET))
            violations=int(not provenance_ok)
            record={'fold_id':fold,'training_feature_date':row.date,'entry_date':row.target_entry_date,
                    'target_end_date':row.target_exit_date_5d,'oracle_entry_date':entry,'oracle_target_end_date':end,
                    'validation_start':first,'target_horizon':5,'eligible':bool(eligible),
                    'target_purged':bool(end>=first),'reason':'eligible' if eligible else '5d_realization_not_before_validation',
                    'violations':violations+int(eligible and end>=first)}
            purges.append(record);fold_purges.append(record)
            if eligible:training.append(row.date)
        valid=block.date.tolist()
        if not training or previous_valid.intersection(valid) or not previous_train<=set(training):
            raise ValueError('Empty/nonexpanding training or overlapping validation windows')
        if any(keyed.loc[day,'target_exit_date_5d']>=FINAL_START for day in valid):
            raise ValueError('Validation primary label reaches locked Final Test')
        previous_valid.update(valid);previous_train=set(training)
        members[fold]=(training,valid)
        inventory.append({**definition,'validation_first_session':first,'validation_last_session':last,
                          'raw_train_candidate_start':candidate.date.iloc[0],
                          'raw_train_candidate_end':candidate.date.iloc[-1],
                          'raw_train_candidate_rows':len(candidate),
                          'effective_train_start':training[0],'effective_train_end':training[-1],
                          'validation_rows':len(valid),'effective_train_rows':len(training),
                          'target_purged_rows':sum(r['target_purged'] for r in fold_purges),
                          'other_excluded_rows':len(candidate)-len(training)-sum(r['target_purged'] for r in fold_purges),
                          'purge_violations':sum(r['violations'] for r in fold_purges)})
    return pd.DataFrame(inventory),pd.DataFrame(purges),members


def load_development(protocol):
    """Validate authority/hashes, then return only registered development payload."""
    canonical=authority.require_canonical_vintage(protocol['canonical_vintage_id'],
        required_features=representation.TREATMENT,required_target=PRIMARY_TARGET)
    if canonical['approval_status']!=protocol['required_authority_status'] \
            or not canonical['expanded_dataset_approved_for_downstream_research'] \
            or not canonical['stage14_modeling_eligible']:
        raise ValueError('Stage13.2 canonical approval gate closed')
    root=h.OUTPUT_DIR
    matrix_path=root/'NVDA_stage13_research_matrix.csv'
    sessions=read_sessions(matrix_path)
    if sessions[0]!=DEVELOPMENT_START or len(sessions)!=1695 or len(set(sessions))!=1695:
        raise ValueError('Canonical research session authority changed')
    x=read_selected(matrix_path,X_SOURCE)
    labels=read_selected(root/'NVDA_stage13_targets.csv',LABEL_COLUMNS)
    if not x.date.equals(labels.date):
        raise ValueError('Canonical X/primary-target row identities differ')
    data=x.merge(labels.loc[:,['date','target_entry_date','target_exit_date_5d',PRIMARY_TARGET]],
                 on='date',validate='one_to_one',sort=False)
    data=data.loc[:,MATRIX_COLUMNS]
    check_matrix(data)
    if data.date.iloc[0]!=DEVELOPMENT_START or data.date.iloc[-1]!=DEVELOPMENT_END:
        raise ValueError('Declared development boundary missing')
    # Price-only target provenance oracle reads the five terminal realization
    # sessions, never a gap feature observation or a locked target label.
    price_end=sessions[sessions.index(DEVELOPMENT_END)+5]
    prices=read_selected(matrix_path,('date','open','close'),end=price_end)
    target_oracle=target_source_oracle(data,labels,sessions,prices)
    if target_oracle.violations.sum():
        raise ValueError('Canonical primary target fails independent source oracle')
    # Secondary provenance uses dates only; no secondary numerical Y is loaded.
    secondary=read_selected(root/'NVDA_stage13_targets.csv',('date','target_exit_date_1d','target_exit_date_20d'))
    pos={day:i for i,day in enumerate(sessions)}
    secondary_violations=sum(int(getattr(row,f'target_exit_date_{k}d')!=sessions[pos[row.date]+k])
                             for row in secondary.itertuples(index=False) for k in (1,20))
    if secondary_violations:
        raise ValueError('Secondary target date provenance changed')
    source_dependency=h.read_csv(authority.OUTPUT_DIR/(authority.PREFIX+'research_dependency_manifest.csv'))
    features=feature_manifest(source_dependency)
    relative=representation.representation(data,'treatment')
    if not np.allclose(relative.loc[:,representation.REPLACEMENTS.values()],
                       data.loc[:,representation.REPLACEMENTS.values()],rtol=0,atol=1e-12,equal_nan=True):
        raise ValueError('Stage11 relative implementation disagrees with frozen canonical values')
    evidence={'canonical_vintage_id':protocol['canonical_vintage_id'],'canonical_source_sha256':canonical['source_sha256'],
              'canonical_manifest_sha256':h.sha256(authority.OUTPUT_DIR/(authority.PREFIX+'canonical_vintage_manifest.json')),
              'canonical_research_sessions':len(sessions),'development_rows':len(data),
              'primary_source_oracle_rows':len(target_oracle),'primary_source_oracle_violations':int(target_oracle.violations.sum()),
              'secondary_provenance_date_checks':len(secondary)*2,'secondary_provenance_violations':secondary_violations,
              'numeric_secondary_labels_loaded':False,'locked_numeric_labels_loaded':False,
              'terminal_primary_label_price_oracle_end':price_end,
              'gap_price_access_scope':'Only the frozen 5d label source-price oracle for development feature dates; gap rows never enter model matrix',
              'gap_rows':sum(GAP_START<=d<=GAP_END for d in sessions),
              'final_test_rows':sum(FINAL_START<=d<=FINAL_END for d in sessions),
              'post_test_tail_rows':sum(d>FINAL_END for d in sessions),**SAFETY}
    return data,sessions,features,evidence


def prepare_and_freeze():
    protocol=load_protocol();before=verify_upstream()
    if before['mutation_violations']:raise ValueError('Protected upstream input changed')
    with authority.offline_only():
        data,sessions,features,evidence=load_development(protocol)
        inventory,purges,members=construct_folds(data,sessions,protocol['fold_calendar_definitions'])
        if len(inventory)!=22 or purges.violations.sum():
            raise ValueError('Declared fold inventory/purge audit failed')
        for name,frame in [('fold_inventory',inventory),('target_purge_audit',purges),('feature_manifest',features)]:
            freeze(OUTPUT_DIR/(PREFIX+name+'.csv'),h.csv_text(frame))
        freeze(OUTPUT_DIR/(PREFIX+'canonical_input_audit.json'),h.json_text(evidence))
        registration={'protocol_sha256':PROTOCOL_SHA256,'folds_frozen_before_modeling':True,
                      'fold_count':len(inventory),'fold_sha256':{
                          name:h.sha256(OUTPUT_DIR/(PREFIX+name+'.csv'))
                          for name in ('fold_inventory','target_purge_audit','feature_manifest')},
                      'boundary_changes_after_modeling_allowed':False}
        freeze(OUTPUT_DIR/(PREFIX+'fold_registration.json'),h.json_text(registration))
    return data,inventory,purges,members,features,evidence


def baseline_predictions(training_y,count):
    y=baseline._target(training_y)
    return {'zero_return':np.zeros(count),'historical_mean':np.full(count,float(y.mean()))}


def reconstruction(parameters,values):
    """Rebuild prediction solely from saved fitted stats, coefficients/intercept."""
    selected=parameters.loc[parameters.used].sort_values('feature_position')
    names=selected.feature_name.tolist()
    filled=values.loc[:,names].fillna(selected.set_index('feature_name').training_median)
    means=selected.set_index('feature_name').training_mean_after_imputation
    scales=selected.set_index('feature_name').scaler_scale
    z=(filled-means)/scales
    return z.to_numpy()@selected.coefficient.to_numpy()+float(selected.intercept.iloc[0])


def prediction_stability(predictions):
    values=np.asarray(predictions);absolute=np.abs(values)
    return {'prediction_mean':float(values.mean()),'prediction_std':float(values.std(ddof=1)),
            'prediction_min':float(values.min()),'prediction_max':float(values.max()),
            'max_abs_prediction':float(absolute.max()),
            'fraction_abs_prediction_gt_0_10':float((absolute>.10).mean()),
            'fraction_abs_prediction_gt_0_20':float((absolute>.20).mean())}


def extrapolation_rows(train_x,valid_x,pp,fold,rep):
    z=baseline.transform_features(pp,valid_x)
    records=[]
    for i,name in enumerate(pp.features):
        train,valid=train_x[name],valid_x[name]
        absolute=np.abs(z[:,i]);outside=(valid.lt(train.min())|valid.gt(train.max()))&valid.notna()
        records.append({'fold_id':fold,'representation':rep,'feature':name,
                        'paired_original_feature':next((k for k,v in representation.REPLACEMENTS.items() if v==name),name),
                        'is_SMA':name in (*representation.REPLACEMENTS,*representation.REPLACEMENTS.values()),
                        'train_min':float(train.min()),'train_max':float(train.max()),
                        'validation_min':float(valid.min()),'validation_max':float(valid.max()),
                        'training_mean':float(pp.scaler.mean_[i]),'training_scale':float(pp.scaler.scale_[i]),
                        'validation_non_null_count':int(valid.notna().sum()),
                        'validation_outside_training_range_count':int(outside.sum()),
                        'max_abs_validation_z':float(absolute.max()),'median_abs_validation_z':float(np.median(absolute)),
                        'fraction_abs_z_gt_3':float((absolute>3).mean()),'fraction_abs_z_gt_5':float((absolute>5).mean()),
                        'fraction_validation_outside_training_min_max':float(outside.sum()/valid.notna().sum()) if valid.notna().any() else np.nan})
    return records


@dataclass
class WalkForwardResult:
    tables:dict
    fits:dict


def build_walk_forward(data,inventory,members):
    check_matrix(data)
    keyed=data.set_index('date',drop=False)
    tables={k:[] for k in ('oof_predictions','fold_metrics','preprocessing_audit','model_parameters',
                           'extrapolation_audit','fit_membership','leakage_audit')}
    fits={}
    with threadpool_limits(limits=1):
        for info in inventory.itertuples(index=False):
            fold=info.fold_id;train_dates,valid_dates=members[fold]
            if (len(train_dates)!=info.effective_train_rows or len(valid_dates)!=info.validation_rows
                    or train_dates!=sorted(set(train_dates)) or valid_dates!=sorted(set(valid_dates))
                    or train_dates[0]!=info.effective_train_start or train_dates[-1]!=info.effective_train_end
                    or valid_dates[0]!=info.validation_first_session or valid_dates[-1]!=info.validation_last_session):
                raise ValueError('Frozen fold membership/count/boundary mismatch')
            if any(d>=info.validation_first_session for d in train_dates) \
                    or not keyed.loc[train_dates,'target_exit_date_5d'].lt(info.validation_first_session).all():
                raise ValueError('Training label realization reaches validation')
            training,valid=keyed.loc[train_dates],keyed.loc[valid_dates]
            if not valid.date.between(info.validation_first_session,info.validation_last_session).all():
                raise ValueError('Validation membership changed')
            actual=baseline._target(valid[PRIMARY_TARGET])
            predictions=baseline_predictions(training[PRIMARY_TARGET],len(valid_dates))
            for rep,arm in ARMS.items():
                train_x=representation.representation(training,arm)
                valid_x=representation.representation(valid,arm)
                if set(train_x) & {'volume','revenue',PRIMARY_TARGET} or len(train_x.columns)!=9:
                    raise ValueError('Forbidden predictor in fit/scaler matrix')
                for family in ('ols','ridge'):
                    name=f'{rep}_{family}'+('_alpha1' if family=='ridge' else '')
                    fitted=representation.fit_arm(family,arm,train_x,training[PRIMARY_TARGET])
                    predictions[name]=baseline.predict_linear(fitted,valid_x)
                    fits[fold,name]=fitted
                    pp=fitted.preprocessing
                    if fitted.train_dates!=tuple(train_dates) or pp.fit_dates!=tuple(train_dates):
                        raise ValueError('Preprocessing/model fit dates differ from effective training')
                    parameters=pp.usage.copy()
                    parameters['feature_position']=np.arange(len(parameters))
                    coef=dict(zip(pp.features,map(float,fitted.estimator.coef_)))
                    parameters['coefficient']=parameters.feature_name.map(coef)
                    parameters['intercept']=float(fitted.estimator.intercept_)
                    parameters['fold_id']=fold;parameters['model']=name;parameters['representation']=rep
                    parameters['ridge_alpha']=1.0 if family=='ridge' else np.nan
                    parameters['fit_target']=PRIMARY_TARGET
                    tables['model_parameters'].append(parameters)
                    usage=pp.usage.assign(fold_id=fold,model=name,representation=rep)
                    tables['preprocessing_audit'].append(usage)
                    rebuilt=reconstruction(parameters,valid_x)
                    reconstruction_bad=int(not np.allclose(rebuilt,predictions[name],rtol=1e-10,atol=1e-12))
                    stat_bad=0
                    for row in pp.usage.itertuples(index=False):
                        expected=baseline._audit_training_stats(train_x[row.feature_name])
                        stat_bad+=int(row.drop_reason!=expected['drop_reason'])
                        for field in ('training_non_null_count','training_coverage','training_median',
                                      'training_mean_after_imputation','training_std_after_imputation','scaler_scale'):
                            stat_bad+=int(not np.isclose(getattr(row,field),expected[field],rtol=1e-10,atol=1e-12,equal_nan=True))
                    tables['leakage_audit'].append({'fold_id':fold,'model':name,'training_only_stats_violations':stat_bad,
                        'prediction_reconstruction_violations':reconstruction_bad,
                        'scaler_sample_count':int(pp.scaler.n_samples_seen_),
                        'effective_train_rows':len(training),'fit_target':PRIMARY_TARGET,
                        'forbidden_feature_violations':len(set(pp.features)-set(representation.SPECS[arm])),
                        'fit_membership_violations':0})
                    tables['fit_membership'].append(pd.DataFrame({'fold_id':fold,'model':name,
                        'training_feature_date':train_dates,'target_end_date':training.target_exit_date_5d.to_numpy(),
                        'validation_start':info.validation_first_session}))
                    if family=='ols':
                        tables['extrapolation_audit'].extend(extrapolation_rows(train_x,valid_x,pp,fold,rep))
                    else:
                        ols=fits[fold,f'{rep}_ols'].preprocessing
                        if pp.features!=ols.features or not np.array_equal(pp.scaler.mean_,ols.scaler.mean_) \
                                or not np.array_equal(pp.scaler.scale_,ols.scaler.scale_):
                            raise ValueError('Paired family preprocessing differs')
            oof=pd.DataFrame({'date':valid_dates,'fold_id':fold,'calendar_year':pd.to_datetime(valid_dates).year,
                              'y_true':actual,'target_entry_date':valid.target_entry_date.to_numpy(),
                              'target_exit_date_5d':valid.target_exit_date_5d.to_numpy(),**predictions})
            tables['oof_predictions'].append(oof)
            for model in MODELS:
                pred=predictions[model]
                tables['fold_metrics'].append({'fold_id':fold,'model':model,'target':PRIMARY_TARGET,
                    'validation_first_session':info.validation_first_session,'validation_last_session':info.validation_last_session,
                    'train_rows':len(training),'validation_rows':len(valid),'MAE':float(np.abs(actual-pred).mean()),
                    'bias':float((pred-actual).mean()),**prediction_stability(pred)})
    frames={key:(pd.concat(value,ignore_index=True) if value and isinstance(value[0],pd.DataFrame)
                 else pd.DataFrame(value)) for key,value in tables.items()}
    return WalkForwardResult(frames,fits)


def comparison_counts(delta):
    delta=np.asarray(delta,dtype=float)
    return {'mean_delta_MAE':float(delta.mean()),'median_delta_MAE':float(np.median(delta)),
            'wins':int((delta<0).sum()),'losses':int((delta>0).sum()),'ties':int((delta==0).sum()),
            'win_rate':float((delta<0).mean())}


def aggregate_tables(oof,metrics):
    summaries=[];baseline_comparisons=[];paired=[];years=[]
    for model in MODELS:
        per=metrics.loc[metrics.model.eq(model)]
        summaries.append({'model':model,'fold_count':len(per),'OOF_rows':len(oof),
                          'mean_fold_MAE':float(per.MAE.mean()),'median_fold_MAE':float(per.MAE.median()),
                          'pooled_OOF_MAE':float(np.abs(oof.y_true-oof[model]).mean())})
        if model in LEARNED:
            for ref in MODELS[:2]:
                pair=per[['fold_id','MAE']].merge(metrics.loc[metrics.model.eq(ref),['fold_id','MAE']],on='fold_id',suffixes=('_model','_reference'),validate='one_to_one')
                baseline_comparisons.append({'model':model,'baseline':ref,'fold_count':len(pair),
                    **comparison_counts(pair.MAE_model-pair.MAE_reference)})
    for family in ('ols','ridge_alpha1'):
        original=f'original_{family}';relative=f'relative_{family}'
        a=metrics.loc[metrics.model.eq(original),['fold_id','MAE']]
        b=metrics.loc[metrics.model.eq(relative),['fold_id','MAE']]
        per=a.merge(b,on='fold_id',suffixes=('_original','_relative'),validate='one_to_one')
        for row in per.itertuples(index=False):
            paired.append({'scope':'fold','fold_id':row.fold_id,'family':family,
                           'original_MAE':row.MAE_original,'relative_MAE':row.MAE_relative,
                           'paired_delta_MAE':row.MAE_relative-row.MAE_original})
        paired.append({'scope':'aggregate','fold_id':'ALL','family':family,
                       **comparison_counts(per.MAE_relative-per.MAE_original)})
    for year,block in oof.groupby('calendar_year',sort=True):
        reference={name:float(np.abs(block.y_true-block[name]).mean()) for name in MODELS[:2]}
        for model in MODELS:
            mae=float(np.abs(block.y_true-block[model]).mean())
            years.append({'calendar_year':year,'model':model,'rows':len(block),'pooled_yearly_MAE':mae,
                          'delta_vs_zero':mae-reference['zero_return'],
                          'delta_vs_historical_mean':mae-reference['historical_mean'],
                          'descriptive_only':True})
    return pd.DataFrame(summaries),pd.DataFrame(baseline_comparisons),pd.DataFrame(paired),pd.DataFrame(years)


def predictive_assessment(comparisons):
    gates={};core={}
    for model in ('relative_ols','relative_ridge_alpha1'):
        rows=comparisons.loc[comparisons.model.eq(model)].set_index('baseline')
        if set(rows.index)!=set(MODELS[:2]):raise ValueError('Complete paired baseline comparisons required')
        tests={}
        for ref in MODELS[:2]:
            tests[f'mean_MAE_beats_{ref}']=bool(rows.loc[ref,'mean_delta_MAE']<0)
            tests[f'median_delta_vs_{ref}_negative']=bool(rows.loc[ref,'median_delta_MAE']<0)
            tests[f'win_rate_vs_{ref}_majority']=bool(rows.loc[ref,'win_rate']>.50)
        gates[model]={**tests,'all_six_passed':all(tests.values())}
        core[model]=all(value for name,value in tests.items() if not name.startswith('median_'))
    category=('ROBUST_MULTI_YEAR_SIGNAL' if any(v['all_six_passed'] for v in gates.values()) else
              'NO_ROBUST_MULTI_YEAR_SIGNAL' if not any(core.values()) else 'MIXED_OR_REGIME_DEPENDENT_SIGNAL')
    return {'classification':category,'model_criteria':gates,'four_mean_majority_gates':core}


def paired_extrapolation(extrapolation):
    records=[]
    for fold,group in extrapolation.groupby('fold_id',sort=False):
        row={'fold_id':fold}
        for rep in ARMS:
            sma=group.loc[group.representation.eq(rep)&group.is_SMA]
            if len(sma)!=3:raise ValueError('Three selected paired SMA features required for mechanism audit')
            row[rep+'_SMA_max_abs_z']=float(sma.max_abs_validation_z.max())
            row[rep+'_SMA_outside_fraction']=float(sma.validation_outside_training_range_count.sum()/sma.validation_non_null_count.sum())
            row[rep+'_SMA_fraction_z_gt_5']=float(sma.fraction_abs_z_gt_5.mean())
        row['relative_SMA_max_z_better']=row['relative_SMA_max_abs_z']<row['original_SMA_max_abs_z']
        row['new_severe_extrapolation']=bool(row['relative_SMA_max_abs_z']>5
            and row['relative_SMA_max_abs_z']>row['original_SMA_max_abs_z']
            and row['relative_SMA_fraction_z_gt_5']>row['original_SMA_fraction_z_gt_5'])
        records.append(row)
    return pd.DataFrame(records)


def representation_assessment(paired,mechanism,stability):
    aggregates=paired.loc[paired.scope.eq('aggregate')].set_index('family')
    wins={family:bool(aggregates.loc[family,'win_rate']>.50) for family in ('ols','ridge_alpha1')}
    tests={
        'median_SMA_max_z_lower':bool(mechanism.relative_SMA_max_abs_z.median()<mechanism.original_SMA_max_abs_z.median()),
        'mean_SMA_outside_fraction_lower':bool(mechanism.relative_SMA_outside_fraction.mean()<mechanism.original_SMA_outside_fraction.mean()),
        'majority_fold_SMA_max_z_lower':bool(mechanism.relative_SMA_max_z_better.mean()>.50),
        'mean_SMA_fraction_gt5_no_higher':bool(mechanism.relative_SMA_fraction_z_gt_5.mean()<=mechanism.original_SMA_fraction_z_gt_5.mean())}
    severe_z=bool(mechanism.new_severe_extrapolation.mean()>.50)
    prediction_pathology={}
    for family in ('ols','ridge_alpha1'):
        rows=stability.loc[stability.model.isin([f'original_{family}',f'relative_{family}'])]
        original=rows.loc[rows.model.eq(f'original_{family}')].set_index('fold_id')
        relative=rows.loc[rows.model.eq(f'relative_{family}')].set_index('fold_id')
        worse=((relative.fraction_abs_prediction_gt_0_20>original.fraction_abs_prediction_gt_0_20)
               &(relative.max_abs_prediction>original.max_abs_prediction)&(relative.max_abs_prediction>.20))
        prediction_pathology[family]={'worse_fold_count':int(worse.sum()),'fold_count':len(worse),
                                     'systematic':bool(worse.mean()>.50)}
    pathology=severe_z or any(v['systematic'] for v in prediction_pathology.values())
    category=('NO_BROAD_REPRESENTATION_STABILITY_SUPPORT' if not any(wins.values()) or pathology else
              'BROAD_REPRESENTATION_STABILITY_SUPPORT' if all(wins.values()) and all(tests.values()) else
              'MIXED_REPRESENTATION_RESULT')
    return {'classification':category,'family_MAE_majority':wins,'mechanism_criteria':tests,
            'new_systematic_pathology':pathology,'systematic_severe_extrapolation':severe_z,
            'severe_extrapolation_fold_count':int(mechanism.new_severe_extrapolation.sum()),
            'prediction_pathology':prediction_pathology,
            'SMA_mechanism_summary':{
                'original_median_fold_max_z':float(mechanism.original_SMA_max_abs_z.median()),
                'relative_median_fold_max_z':float(mechanism.relative_SMA_max_abs_z.median()),
                'original_mean_fold_outside_fraction':float(mechanism.original_SMA_outside_fraction.mean()),
                'relative_mean_fold_outside_fraction':float(mechanism.relative_SMA_outside_fraction.mean()),
                'relative_lower_max_z_folds':int(mechanism.relative_SMA_max_z_better.sum())}}


def final_lock_audit(data,inventory,result,evidence):
    oof=result.tables['oof_predictions'];membership=result.tables['fit_membership']
    checks={
        'development_rows_before_gap':bool(data.date.le(DEVELOPMENT_END).all()),
        'validation_rows_before_gap':bool(oof.date.le(DEVELOPMENT_END).all()),
        'fit_rows_before_gap':bool(membership.training_feature_date.le(DEVELOPMENT_END).all()),
        'fit_labels_realized_before_validation':bool(membership.target_end_date.lt(membership.validation_start).all()),
        'validation_labels_end_before_Final_Test':bool(oof.target_exit_date_5d.lt(FINAL_START).all()),
        'locked_final_row_count_50':evidence['final_test_rows']==50,
        'no_locked_numeric_target_loaded':not evidence['locked_numeric_labels_loaded'],
        'no_secondary_numeric_labels_loaded':not evidence['numeric_secondary_labels_loaded'],
        'no_gap_test_tail_predictions':bool(oof.date.lt(GAP_START).all()),
        'no_final_training_pool_or_final_fit':True}
    return {'valid':all(checks.values()),'checks':checks,'development':[DEVELOPMENT_START,DEVELOPMENT_END],
            'frozen_pretest_gap':[GAP_START,GAP_END],'Final_Test':[FINAL_START,FINAL_END],
            'post_test_tail':['2026-09-24','2026-09-30'],
            'validation_labels_realizing_in_gap':int(oof.target_exit_date_5d.ge(GAP_START).sum()),
            'gap_feature_observations_used':0,'locked_numeric_labels_loaded':False,**SAFETY}


def validate_result(data,inventory,purges,members,result):
    tables=result.tables;oof=tables['oof_predictions'];metrics=tables['fold_metrics']
    expected=[day for _,valid in members.values() for day in valid]
    leak=tables['leakage_audit']
    violations={name:int(leak[name].sum()) for name in
        ('training_only_stats_violations','prediction_reconstruction_violations','forbidden_feature_violations','fit_membership_violations')}
    checks={
        'target_purge':not bool(purges.violations.sum()),
        'OOF_unique':not oof.date.duplicated().any(),
        'OOF_dates_complete_exact':oof.date.tolist()==expected,
        'OOF_six_models_only':list(oof.columns[-6:])==list(MODELS),
        'OOF_complete_finite':bool(np.isfinite(oof.loc[:,['y_true',*MODELS]].to_numpy(dtype=float)).all()),
        'OOF_y_matches_canonical_source':bool(np.array_equal(oof.y_true.to_numpy(),
            data.set_index('date').loc[oof.date,PRIMARY_TARGET].to_numpy())),
        'OOF_provenance_matches_source':bool(oof.target_entry_date.tolist()==data.set_index('date').loc[oof.date,'target_entry_date'].tolist()
            and oof.target_exit_date_5d.tolist()==data.set_index('date').loc[oof.date,'target_exit_date_5d'].tolist()),
        'fold_metric_grid':len(metrics)==len(inventory)*len(MODELS)
            and not metrics.duplicated(['fold_id','model']).any() and set(metrics.model)==set(MODELS),
        'learned_fit_count':len(result.fits)==len(inventory)*4,
        'training_only_preprocessing':not any(violations.values()),
        'scaler_sample_count':bool(leak.scaler_sample_count.eq(leak.effective_train_rows).all()),
        'primary_target_only':bool(metrics.target.eq(PRIMARY_TARGET).all()),
        'no_locked_or_gap_dates':bool(oof.date.le(DEVELOPMENT_END).all()),
        'prediction_metrics_integrity':True,'baseline_definitions':True}
    keyed=data.set_index('date')
    for fold,(train,valid) in members.items():
        block=oof.loc[oof.fold_id.eq(fold)]
        anchors=baseline_predictions(keyed.loc[train,PRIMARY_TARGET],len(valid))
        checks['baseline_definitions'] &= all(np.array_equal(block[model].to_numpy(),anchors[model]) for model in anchors)
        for model in MODELS:
            mae=float(np.abs(block.y_true-block[model]).mean())
            actual=metrics.loc[metrics.fold_id.eq(fold)&metrics.model.eq(model),'MAE'].iloc[0]
            checks['prediction_metrics_integrity'] &= bool(np.isclose(mae,actual,rtol=0,atol=1e-12))
    return {'valid':all(checks.values()),'checks':checks,'violations':violations,
            'expected_OOF_sessions':len(expected),'actual_OOF_sessions':len(oof),
            'missing_OOF_sessions':len(set(expected)-set(oof.date)),
            'duplicate_OOF_sessions':int(oof.date.duplicated().sum()),
            'prediction_reconstruction_model_folds':len(result.fits)}


def run_pipeline():
    protocol=load_protocol()
    with authority.offline_only():
        data,inventory,purges,members,features,evidence=prepare_and_freeze()
        result=build_walk_forward(data,inventory,members)
        tables=result.tables
        validation=validate_result(data,inventory,purges,members,result)
        lock=final_lock_audit(data,inventory,result,evidence)
        if not validation['valid'] or not lock['valid']:
            raise ValueError('Stage14 engineering audit failed: '+h.json_text({'validation':validation,'lock':lock}))
        summaries,comparisons,paired,years=aggregate_tables(tables['oof_predictions'],tables['fold_metrics'])
        mechanism=paired_extrapolation(tables['extrapolation_audit'])
        stability=tables['fold_metrics'].loc[:,['fold_id','model',*prediction_stability(np.array([0.,1.])).keys()]]
        signal=predictive_assessment(comparisons)
        representation_result=representation_assessment(paired,mechanism,stability)
        tables.update({'feature_manifest':features,'fold_inventory':inventory,'target_purge_audit':purges,
                       'model_summary':summaries,'baseline_comparison':comparisons,
                       'representation_comparison':paired,'yearly_diagnostics':years,
                       'paired_extrapolation_summary':mechanism,'prediction_stability':stability})
        for name in TABLE_NAMES:
            freeze(OUTPUT_DIR/(PREFIX+name+'.csv'),h.csv_text(tables[name]))
        freeze(OUTPUT_DIR/(PREFIX+'final_test_lock_audit.json'),h.json_text(lock))
        upstream=verify_upstream()
        if upstream['mutation_violations']:raise ValueError('Upstream mutation detected')
        summary={'stage_name':protocol['stage_name'],'engineering_status':'PASS',
                 'predictive_signal_classification':signal['classification'],
                 'representation_stability_classification':representation_result['classification'],
                 'canonical_vintage_id':protocol['canonical_vintage_id'],
                 'development_rows':len(data),'development_dates':[data.date.iloc[0],data.date.iloc[-1]],
                 'fold_count':len(inventory),'first_fold':inventory.fold_id.iloc[0],
                 'last_fold':inventory.fold_id.iloc[-1],'OOF_sessions':len(tables['oof_predictions']),
                 'OOF_prediction_values':len(tables['oof_predictions'])*len(MODELS),
                 'learned_model_fits':len(result.fits),'target_purged_rows_per_fold':inventory.set_index('fold_id').target_purged_rows.to_dict(),
                 'predictive_assessment':signal,'representation_assessment':representation_result,
                 'model_summary':summaries.to_dict('records'),'baseline_comparisons':comparisons.to_dict('records'),
                 'OOF_coverage':{k:validation[k] for k in ('expected_OOF_sessions','actual_OOF_sessions','missing_OOF_sessions','duplicate_OOF_sessions')},
                 'new_catastrophic_threshold_monitoring':{
                     'folds_with_abs_prediction_gt_0_20':stability.groupby('model').fraction_abs_prediction_gt_0_20.apply(lambda s:int(s.gt(0).sum())).to_dict(),
                     'max_abs_predictions':stability.groupby('model').max_abs_prediction.max().to_dict(),
                     'thresholds_descriptive_only':True},
                 'label_realization_in_pretest_gap':lock['validation_labels_realizing_in_gap'],
                 'historical_statuses':protocol['historical_statuses'],'upstream':upstream,
                 'statistical_inference_performed':False,**SAFETY}
        validation.update({'engineering_status':'PASS','canonical_input':evidence,
                           'final_test_lock_valid':lock['valid'],'upstream':upstream,
                           'protocol_sha256':PROTOCOL_SHA256,
                           'implementation_sha256':h.sha256(Path(__file__)),
                           'formal_table_sha256':{name:h.sha256(OUTPUT_DIR/(PREFIX+name+'.csv')) for name in TABLE_NAMES},
                           'post_run_tests_and_reproduction_evidence':PREFIX+'execution_verification.json',**SAFETY})
        freeze(OUTPUT_DIR/(PREFIX+'summary.json'),h.json_text(summary))
        freeze(OUTPUT_DIR/(PREFIX+'validation.json'),h.json_text(validation))
    return summary


def main():
    summary=run_pipeline()
    print(h.json_text({k:summary[k] for k in ('engineering_status','predictive_signal_classification',
                        'representation_stability_classification','fold_count','OOF_sessions')}))


if __name__=='__main__':
    main()
