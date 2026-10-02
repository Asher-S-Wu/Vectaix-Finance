"""按年份向前验证，冻结方案后做历史确认与月度重训。"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import HORIZONS
from .contracts import FORECAST_RETURN_BASIS
from .prediction_tasks import PREDICTION_SCHEMA, TASK_STATUS_COLUMNS
from .collect import write_json
from .paths import DATA, MODELS, RESULTS

KINDS = ('factor', 'linear', 'lightgbm_small', 'lightgbm_large')
METHOD_VERSION = 'universal-20260911-v1'
MODEL_REVISION = 'all-market-pit-task-outputs-v31'
TRAIN_STOCKS_PER_DATE = 256
PREDICTION_OUTPUTS = ('date','security_id','horizon','fwd_return','label_end','score',
                      'probability_up','expected_return','q10','q50','q90',
                      'baseline_probability','baseline_q10','baseline_q50','baseline_q90',
                      'status','reasons','model_version','data_as_of','model_trained_as_of','return_basis',
                      *TASK_STATUS_COLUMNS.values(),'prediction_schema')
LINEAR_FEATURES = ('momentum_20', 'momentum_60', 'volatility_20', 'near_high_252',
                   'log_amount_20', 'intraday_range', 'intraday_return',
                   'market_momentum_60', 'market_volatility_60', 'market_breadth_60')
FACTOR_FEATURES = ('momentum_60','momentum_5','volatility_20','log_amount_20')


def cross_sectional_inputs(frame, features):
    result = frame.copy()
    result['sigma_daily'] = result.volatility_20 / np.sqrt(252)
    ranked = [c for c in features if not c.startswith(('market_', 'offer_'))]
    valid = result.status.eq('ok')
    result.loc[valid, ranked] = result.loc[valid].groupby('date')[ranked].rank(pct=True).astype(np.float32)
    return result


def make_examples(frame, features):
    pieces = []
    columns = ['date','security_id','status','sigma_daily'] + list(features)
    for horizon in HORIZONS:
        part = frame[columns].copy()
        part['horizon'] = horizon
        part['fwd_return'] = frame[f'fwd_return_{horizon}']
        part['label_end'] = frame[f'label_end_{horizon}']
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def sample_observations(frame, max_stocks_per_date=TRAIN_STOCKS_PER_DATE):
    samples = []
    for date, group in frame.groupby('date', sort=True):
        if len(group) > max_stocks_per_date:
            rng = np.random.default_rng(20260910 + pd.Timestamp(date).toordinal())
            group = group.iloc[np.sort(rng.choice(len(group), max_stocks_per_date, replace=False))]
        samples.append(group)
    return pd.concat(samples, ignore_index=True)


def split_window(frame, features, as_of, max_stocks_per_date=TRAIN_STOCKS_PER_DATE):
    as_of = pd.Timestamp(as_of)
    calibration_start = as_of - pd.DateOffset(months=12)
    current = frame.loc[frame.date.le(as_of) & frame.status.eq('ok')].copy()
    # 固定随机抽样减少重叠日标签；每天均有训练观察，市场推理不作抽样。
    sampled = sample_observations(current, max_stocks_per_date)
    examples = make_examples(sampled, features)
    mature = examples.fwd_return.notna() & examples.label_end.notna() & examples.sigma_daily.gt(0)
    training = examples.loc[mature & examples.date.lt(calibration_start) & examples.label_end.lt(calibration_start)]
    calibration = examples.loc[mature & examples.date.ge(calibration_start) & examples.label_end.le(as_of)]
    return training.reset_index(drop=True), calibration.reset_index(drop=True)


def normalize_dataset(data_root=DATA):
    manifest = json.loads((data_root / 'feature_manifest.json').read_text(encoding='utf-8'))
    features = manifest['features']
    output = data_root / 'normalized'
    output.mkdir(parents=True, exist_ok=True)
    sample_output = data_root / 'training_samples'
    sample_output.mkdir(parents=True, exist_ok=True)
    latest = []
    for year in range(int(manifest['start_date'][:4]), int(manifest['data_as_of'][:4]) + 1):
        start, end = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
        frame = pd.read_parquet(data_root / 'features', filters=[('date','>=',start),('date','<=',end)])
        if frame.empty:
            raise ValueError(f'{year}无因子数据')
        transformed = cross_sectional_inputs(frame, features)
        transformed.to_parquet(output / f'{year}.parquet', index=False, row_group_size=200_000)
        sample_observations(transformed[transformed.status.eq('ok')]).to_parquet(sample_output/f'{year}.parquet',index=False)
        latest.append(transformed.loc[transformed.date.eq(pd.Timestamp(manifest['data_as_of']))])
        print(f'normalized {year}: {len(frame)} rows', flush=True)
    pd.concat(latest, ignore_index=True).to_parquet(data_root / 'latest_inputs.parquet', index=False)
    from .market_context import risk_returns_from_inputs
    daily_returns=pd.read_parquet(data_root/'features',columns=['date','security_id','return_1'])
    returns = risk_returns_from_inputs(daily_returns,pd.read_parquet(data_root/'securities.parquet'))
    returns.reset_index().to_parquet(data_root / 'risk_returns.parquet', index=False)
    write_json(data_root / 'normalization_manifest.json', dataset_lineage(data_root))


def dataset_lineage(data_root=DATA):
    path = data_root / 'feature_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    return {'method_version': METHOD_VERSION, 'feature_manifest_mtime_ns': path.stat().st_mtime_ns,
            'data_as_of': manifest['data_as_of'], 'rows': manifest['rows'],
            'features': manifest['features'], 'training_stocks_per_date': TRAIN_STOCKS_PER_DATE,
            'risk_return_source':'raw_daily_feature_return_1_before_cross_sectional_normalization'}


def initialize_run(data_root, results_root):
    build=json.loads((data_root/'build_status.json').read_text(encoding='utf-8'))
    if build['status']!='complete':raise ValueError('数据版本仍在构建，不能开始训练')
    lineage = dataset_lineage(data_root)
    normalized = json.loads((data_root / 'normalization_manifest.json').read_text(encoding='utf-8'))
    if normalized != lineage:
        raise ValueError('标准化数据与本轮因子版本不一致，必须重新标准化')
    path = results_root / 'training_protocol.json'
    protocol = {
        'lineage': lineage, 'model_revision':MODEL_REVISION, 'kinds': list(KINDS), 'development_years': list(range(2016, 2024)),
        'confirmation_start': '2024-01-01', 'calibration_months': 12,
        'selection': 'Fixed per-horizon task models selected on identical development observations for that task. Unrelated task failures do not remove a valid task observation. Each complex task model must improve its full-period metric and >=60% of all eight annual metrics over the stronger simple baseline.',
        'training_sampling': 'Deterministic uniform sample of 256 eligible securities per date; all securities are scored and evaluated.',
        'target_schema': 'arithmetic-return-sqrt-horizon-v1',
        'prediction_schema': PREDICTION_SCHEMA,
        'learned_target': 'fwd_return / sqrt(horizon); includes verified -100% outcomes without clipping',
        'prediction_batching': 'Complete date/horizon cross sections; preserves source row order and does not split market centering groups.',
        'tree_intervals': 'Conditional LightGBM quantile models with held-out joint positive-scale log1p calibration; raw crossing intervals are unavailable, never sorted to hide crossing.',
        'linear_intervals': 'Common empirical log1p-return residual distribution using only the held-out calibration window; invalid point or volatility inputs remain unavailable.',
        'ranking': {'horizon':20, 'ic_ci95_lower_gt':0, 'positive_year_ratio_min':.60},
        'probability': {'brier_skill_gt':0, 'ece_max':.05},
        'interval': {'coverage_min':.75, 'coverage_max':.85, 'pinball_improves':True},
        'portfolio': {'annual_excess_gt':0, 'information_ratio_min':.5, 'drawdown_min':-.30,
                      'positive_year_ratio_min':.60, 'stress_annual_excess_gt':0},
        'execution': {'fee_per_side':.0025, 'stress_fee_per_side':.005, 'max_participation':.01,
                      'price':'next_session_raw_vwap', 'cash_policy':'issuer_final_schedule_simulated'},
        'calibration': {
            'selection_period': '2016-2023 development only',
            'probability_centering': 'date_horizon_cross_section for factor probability head',
            'probability_baseline_blend': {'1': 0.0, '5': 0.0, '20': 0.0, '60': 0.0},
            'probability_anchor': 'held-out_logistic_without_training_rate_shift',
            'probability_anchor_offset': {'1': 0.0, '5': 0.0, '20': 0.0, '60': 0.0},
            'interval_width_multiplier': {'1': 1.0, '5': 1.0, '20': 1.0, '60': 1.0},
            'selection_rule': 'No manual probability offsets or interval width tuning. Task selection uses only 2016-2023 development results under unchanged acceptance thresholds.',
            'confirmation_parameter_selection': False,
        },
        'confirmation_note': 'Historical confirmation; some recent results had previously been viewed. Not an untouched holdout.'}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('本轮训练方案或数据已变化，请为新实验指定独立结果目录')
    else:
        write_json(path, protocol)
    return protocol


def load_inputs(data_root=DATA, end=None, training_sample=False):
    frames = []
    folder = 'training_samples' if training_sample else 'normalized'
    for path in sorted((data_root / folder).glob('*.parquet')):
        if end is not None and int(path.stem) > pd.Timestamp(end).year:
            continue
        frame = pd.read_parquet(path)
        if end is not None:
            frame = frame[frame.date.le(pd.Timestamp(end))]
        frames.append(frame)
    if not frames:
        raise ValueError('尚无标准化因子数据')
    return pd.concat(frames, ignore_index=True)


def fit_at(frame, kind, as_of, version, features):
    from .models import UniversalModel
    chosen_features = list(FACTOR_FEATURES) if kind=='factor' else list(LINEAR_FEATURES) if kind == 'linear' else list(features)
    # 仅训练期决定特征可用性，不让确认期的数据覆盖率决定输入。
    training_boundary = pd.Timestamp(as_of) - pd.DateOffset(months=12)
    coverage = frame.loc[frame.date.lt(training_boundary), chosen_features].notna().mean()
    chosen_features = coverage.index[coverage.ge(.20)].tolist()
    if not chosen_features:
        raise ValueError('训练期没有可用因子')
    training, calibration = split_window(frame, chosen_features, as_of)
    model = UniversalModel(kind=kind, model_version=version)
    return model.fit(training, calibration, chosen_features, pd.Timestamp(as_of))


def predict_frame(model, frame, batch_size=100_000):
    """分批执行只改变内存用量，同日同期限的市场中心化不受批大小影响。"""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('预测批大小必须为正整数')
    examples = make_examples(frame, model.feature_columns)
    outputs = []
    pending = []
    size = 0
    def emit(indices):
        positions = np.concatenate(indices)
        result = model.predict(examples.iloc[positions])[list(PREDICTION_OUTPUTS)].copy()
        result['_source_position'] = positions
        outputs.append(result)
    for positions in examples.groupby(['date', 'horizon'], sort=False, dropna=False).indices.values():
        if pending and size + len(positions) > batch_size:
            emit(pending)
            pending, size = [], 0
        pending.append(positions)
        size += len(positions)
    if pending:
        emit(pending)
    return (pd.concat(outputs, ignore_index=True).sort_values('_source_position')
            .drop(columns='_source_position').reset_index(drop=True))


def run_development(data_root=DATA, results_root=RESULTS, model_root=MODELS):
    initialize_run(data_root, results_root)
    manifest = json.loads((data_root / 'feature_manifest.json').read_text(encoding='utf-8'))
    if pd.Timestamp(manifest['data_as_of']) < pd.Timestamp('2023-12-31'):
        raise ValueError('数据尚未覆盖完整开发期')
    frame = load_inputs(data_root, '2023-12-31', training_sample=True)
    output = results_root / 'development'
    output.mkdir(parents=True, exist_ok=True)
    selection = {}
    for kind in KINDS:
        paths = []
        for year in range(2016, 2024):
            path = output / f'{kind}_{year}.parquet'
            if path.exists():
                paths.append(path)
                continue
            else:
                started = time.monotonic()
                cutoff = frame.loc[frame.date.lt(f'{year}-01-01'),'date'].max()
                write_json(results_root/'training_status.json',{'stage':'development','status':'training',
                    'kind':kind,'validation_year':year,'training_as_of':str(cutoff.date())})
                print(f'training {kind} as_of={cutoff.date()}', flush=True)
                model = fit_at(frame, kind, cutoff, f'{MODEL_REVISION}-{kind}-{year}', manifest['features'])
                test = pd.read_parquet(data_root / 'normalized' / f'{year}.parquet')
                predictions = predict_frame(model, test)
                predictions.to_parquet(path, index=False)
                model_path = model_root / 'development' / f'{kind}-{year}.pkl'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with model_path.open('wb') as stream:
                    pickle.dump(model, stream)
                write_json(results_root/'training_status.json',{'stage':'development','status':'year_complete',
                    'kind':kind,'validation_year':year,'prediction_rows':len(predictions),
                    'scored_rows':int(predictions.status.eq('ok').sum()),'elapsed_seconds':time.monotonic()-started})
                print(f'completed {kind} {year}: {len(predictions)} forecasts in {time.monotonic()-started:.1f}s',flush=True)
            paths.append(path)
            del predictions
        metrics = evaluate_prediction_files(paths, '2023-12-31')
        write_json(output / f'{kind}_metrics.json', metrics)
        selection[kind] = metrics
    write_json(results_root / 'development_metrics.json', selection)
    write_json(results_root/'training_status.json',{'stage':'development','status':'complete','kinds':list(KINDS)})
    return selection


def evaluate_prediction_files(paths, as_of):
    """逐期限读取全市场预测，避免一次复制四个期限的十余百万行。"""
    from .evaluation import evaluate_predictions, PREDICTION_COLUMNS
    merged = None
    status_counts = {}
    totals = ('input_rows','matured_rows','excluded_unmatured_rows','pending_label_end_rows',
              'matured_label_rows','missing_matured_label_rows','total_forecast_rows',
              'scored_forecast_rows','excluded_status_rows','input_unavailable_rows',
              'input_eligible_forecast_rows','model_rejected_rows','matured_model_rejected_rows',
              'partially_input_unavailable_rows','any_task_scored_forecast_rows','partially_scored_forecast_rows')
    task_totals = ('total_forecast_rows','scored_forecast_rows','excluded_status_rows',
                   'input_unavailable_rows','input_eligible_forecast_rows','model_rejected_rows',
                   'matured_model_rejected_rows')
    task_coverage = {task:{**{key:0 for key in task_totals},'status_counts':{}} for task in TASK_STATUS_COLUMNS}
    task_valid_samples = {task:0 for task in TASK_STATUS_COLUMNS}
    for horizon in HORIZONS:
        frames = [pd.read_parquet(path, columns=list(PREDICTION_COLUMNS),
                                  filters=[('horizon','=',horizon)]) for path in paths]
        metrics = evaluate_predictions(pd.concat(frames, ignore_index=True), as_of)
        if merged is None:
            merged = {k:v for k,v in metrics.items() if k != 'horizons'}
            merged['horizons'] = {}
            for key in totals: merged[key] = 0
        for key in totals: merged[key] += metrics[key]
        for key,value in metrics['status_counts'].items():status_counts[key]=status_counts.get(key,0)+value
        for task in TASK_STATUS_COLUMNS:
            values=metrics['task_coverage'][task]
            for key in task_totals:task_coverage[task][key]+=values[key]
            for key,value in values['status_counts'].items():
                counts=task_coverage[task]['status_counts'];counts[key]=counts.get(key,0)+value
            task_valid_samples[task]+=metrics['task_valid_samples'][task]
        merged['horizons'][str(horizon)] = metrics['horizons'][horizon]
        print(f'evaluated horizon {horizon}: {metrics["matured_rows"]} matured rows', flush=True)
    merged['matured_label_coverage']=(merged['matured_label_rows']/merged['matured_rows']
                                      if merged['matured_rows'] else None)
    merged['label_coverage_complete']=(merged['matured_rows']>0 and merged['missing_matured_label_rows']==0)
    merged['model_rejection_rate']=(merged['model_rejected_rows']/merged['input_eligible_forecast_rows']
                                    if merged['input_eligible_forecast_rows'] else None)
    merged['model_output_coverage_complete']=(merged['input_eligible_forecast_rows']>0 and merged['model_rejected_rows']==0)
    merged['valid']=all(value['matured_rows']>0 and value['label_coverage_complete'] and value['model_output_coverage_complete']
                        for value in merged['horizons'].values())
    merged['status_counts'] = status_counts
    for values in task_coverage.values():
        total=values['input_eligible_forecast_rows']
        values['model_rejection_rate']=values['model_rejected_rows']/total if total else None
        values['model_output_coverage_complete']=total>0 and values['model_rejected_rows']==0
    merged['task_coverage']=task_coverage
    merged['task_valid_samples']=task_valid_samples
    merged['model_output_coverage_complete']=all(values['model_output_coverage_complete'] for values in task_coverage.values())
    return merged


def fit_bundle_at(frame, heads, as_of, version, features):
    from .bundle import MultiTaskBundle
    kinds=sorted({kind for configuration in heads.values() for kind in configuration.values()})
    models={kind:fit_at(frame,kind,as_of,f'{version}-{kind}',features) for kind in kinds}
    return MultiTaskBundle(models,heads,version)


def evaluate_bundle_development(heads, results_root, model_root):
    from .bundle import MultiTaskBundle
    kinds=sorted({kind for configuration in heads.values() for kind in configuration.values()})
    revision=json.loads((results_root/'training_protocol.json').read_text(encoding='utf-8'))['model_revision']
    paths=[]
    output=results_root/'bundle_development'
    output.mkdir(parents=True,exist_ok=True)
    for year in range(2016,2024):
        models={}
        for kind in kinds:
            with (model_root/'development'/f'{kind}-{year}.pkl').open('rb') as stream:
                models[kind]=pickle.load(stream)
        bundle=MultiTaskBundle(models,heads,f'{revision}-bundle-{year}')
        predictions={kind:pd.read_parquet(results_root/'development'/f'{kind}_{year}.parquet') for kind in kinds}
        examples=predictions[kinds[0]][['date','security_id','horizon','fwd_return','label_end']]
        combined=bundle.combine_predictions(examples,predictions)
        path=output/f'{year}.parquet'
        combined.to_parquet(path,index=False)
        paths.append(path)
        del predictions,combined,models,bundle,examples
    metrics=evaluate_prediction_files(paths,'2023-12-31')
    metrics['model_revision']=revision
    write_json(results_root/'bundle_development_metrics.json',metrics)
    return metrics


def freeze_architecture(data_root=DATA, results_root=RESULTS, model_root=MODELS):
    from .component_selection import compare_components, choose_component_heads
    protocol = initialize_run(data_root, results_root)
    comparison=compare_components(results_root)
    write_json(results_root/'component_comparison.json',comparison)
    heads=choose_component_heads(comparison)
    write_json(results_root/'component_selection.json',{'heads':heads,'as_of':'2023-12-31',
                                                       'comparison':comparison,'confirmation_used':False,
                                                       'diagnostic_only':True,'comparison_scope':comparison['comparison_scope']})
    path = results_root/'frozen_model.json'
    decision = {'heads':heads, 'protocol':protocol,'selection_end':'2023-12-31'}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != decision:
            raise ValueError('方案已经冻结，不得使用确认期结果重选')
    else:
        write_json(path, decision)
    evaluate_bundle_development(heads,results_root,model_root)
    return decision


def confirmation_readiness(development, data_audit):
    from .evaluation import release_decision
    checks=release_decision(development,{},data_audit)['checks']
    relevant={name:value for name,value in checks.items() if not name.startswith('confirmation.')}
    return {'ready':all(value['passed'] for value in relevant.values()),
            'checks':relevant,'failed_checks':[name for name,value in relevant.items() if not value['passed']]}


def confirmation_month_record(model, forecasts, inputs, period):
    """从实际模型及当月输出核对四期限覆盖，不以文件存在代替完成。"""
    keys=['date','security_id','horizon']
    expected=pd.concat([inputs[['date','security_id']].assign(horizon=h) for h in HORIZONS],ignore_index=True)
    actual=forecasts[keys].sort_values(keys).reset_index(drop=True)
    complete=actual.equals(expected.sort_values(keys).reset_index(drop=True))
    complete &= forecasts.model_version.eq(model.model_version).all()
    complete &= pd.to_datetime(forecasts.model_trained_as_of).eq(pd.Timestamp(model.as_of)).all()
    components={}
    for kind,component in model.models.items():
        components[kind]={
            'probability_baseline_blend':{str(h):v for h,v in component.probability_baseline_blend.items()},
            'probability_anchor_offset':{str(h):v for h,v in component.probability_anchor_offsets.items()},
            'interval_width_multiplier':{str(h):v for h,v in component.interval_width_multiplier.items()},
        }
    return {'forecast_month':str(period),'model_revision':model.model_version.rsplit('-bundle-',1)[0],
            'model_version':model.model_version,'trained_as_of':pd.Timestamp(model.as_of).isoformat(),
            'heads':model.heads,'component_calibration_parameters':components,
            'horizons':sorted(int(h) for h in forecasts.horizon.unique()),
            'forecast_rows':len(forecasts),'prediction_coverage_complete':bool(complete)}


def run_confirmation(data_root=DATA, results_root=RESULTS, model_root=MODELS):
    initialize_run(data_root, results_root)
    development={key:json.loads((results_root/path).read_text(encoding='utf-8'))
                 for key,path in [('predictions','bundle_development_metrics.json'),('portfolio','development_portfolio_metrics.json')]}
    readiness=confirmation_readiness(development,json.loads((data_root/'data_audit.json').read_text(encoding='utf-8')))
    write_json(results_root/'confirmation_readiness.json',readiness)
    if not readiness['ready']:
        raise ValueError('开发期或数据验收未通过，不进入确认区间：'+', '.join(readiness['failed_checks']))
    frozen = json.loads((results_root/'frozen_model.json').read_text(encoding='utf-8'))
    manifest = json.loads((data_root/'feature_manifest.json').read_text(encoding='utf-8'))
    frame = load_inputs(data_root, training_sample=True)
    output = results_root/'confirmation'
    output.mkdir(parents=True, exist_ok=True)
    paths, records = [], []
    for period in pd.period_range('2024-01',pd.Timestamp(manifest['data_as_of']),freq='M'):
        path = output/f'{period:%Y%m}.parquet'
        version = f'{MODEL_REVISION}-bundle-{period:%Y%m}'
        saved = model_root/'confirmation'/f'{version}.pkl'
        if not path.exists():
            cutoff = frame.loc[frame.date.lt(period.start_time),'date'].max()
            print(f'confirmation fit {version} as_of={cutoff.date()}',flush=True)
            model = fit_bundle_at(frame,frozen['heads'],cutoff,version,manifest['features'])
            test = pd.read_parquet(data_root/'normalized'/f'{period.year}.parquet',
                                   filters=[('date','>=',period.start_time),('date','<=',period.end_time)])
            predictions = predict_frame(model,test)
            predictions.to_parquet(path,index=False)
            saved.parent.mkdir(parents=True,exist_ok=True)
            with saved.open('wb') as stream: pickle.dump(model,stream)
        else:
            with saved.open('rb') as stream: model=pickle.load(stream)
            predictions=pd.read_parquet(path,columns=['date','security_id','horizon','model_version','model_trained_as_of'])
            test=pd.read_parquet(data_root/'normalized'/f'{period.year}.parquet',columns=['date','security_id'],
                                 filters=[('date','>=',period.start_time),('date','<=',period.end_time)])
        records.append(confirmation_month_record(model,predictions,test,period))
        del predictions, model, test
        paths.append(path)
    metrics = evaluate_prediction_files(paths,manifest['data_as_of'])
    calibration=frozen['protocol']['calibration']
    metrics.update(model_revision=MODEL_REVISION,frozen_model_revision=frozen['protocol']['model_revision'],
                   frozen_heads=frozen['heads'],
                   frozen_calibration_parameters={key:calibration[key] for key in (
                       'probability_baseline_blend','probability_anchor_offset','interval_width_multiplier')},
                   formal_confirmation=True,parameter_selection_used=False,retrain_frequency='monthly',
                   monthly_model_records=records)
    write_json(results_root/'confirmation_metrics.json',metrics)
    return metrics


def complete_market_snapshot(forecasts, securities, day, model_version, trained_as_of):
    """未有行情的新股/REIT仍显示四个期限，缺失预测不补造。"""
    day = pd.Timestamp(day).normalize()
    listing = pd.to_datetime(securities.list_date)
    delisting = pd.to_datetime(securities.delist_date)
    active = securities[(listing.isna() | listing.le(day)) & (delisting.isna() | delisting.gt(day))].copy()
    forecasts = forecasts[forecasts.security_id.isin(active.security_id)].copy()
    expected=active[['security_id']].merge(pd.DataFrame({'horizon':HORIZONS}),how='cross')
    if not forecasts.horizon.isin(HORIZONS).all():raise ValueError('快照出现不支持的预测期限')
    if not pd.to_datetime(forecasts.date).dt.normalize().eq(day).all():raise ValueError('快照预测日期不一致')
    missing=expected.merge(forecasts[['security_id','horizon']],on=['security_id','horizon'],
                           how='left',indicator=True,validate='one_to_one')
    missing=missing.loc[missing['_merge'].eq('left_only')]
    additions = []
    for row in missing.itertuples(index=False):
        additions.append({'date':day,'security_id':row.security_id,'horizon':row.horizon,
            'status':'insufficient_model_inputs','data_status':'missing_prediction_record',
            'reasons':'该证券当日该期限没有预测记录，未生成预测数值',
            **{field:'insufficient_model_inputs' for field in TASK_STATUS_COLUMNS.values()},
            'prediction_schema':PREDICTION_SCHEMA,'explanations':[],'explanation_basis':None,
            'model_version':model_version,'data_as_of':day.date().isoformat(),
            'model_trained_as_of':pd.Timestamp(trained_as_of).isoformat(),'return_basis':FORECAST_RETURN_BASIS})
    if additions:
        forecasts = pd.concat([forecasts,pd.DataFrame(additions)],ignore_index=True)
    identity_columns = ['security_id','exchange_code','name','isin','currency','asset_type','identity_status']
    return forecasts.merge(active[identity_columns],on='security_id',validate='many_to_one')


def export_candidate(data_root=DATA, results_root=RESULTS, model_root=MODELS):
    initialize_run(data_root,results_root)
    frozen = json.loads((results_root/'frozen_model.json').read_text(encoding='utf-8'))
    manifest = json.loads((data_root/'feature_manifest.json').read_text(encoding='utf-8'))
    day = pd.Timestamp(manifest['data_as_of'])
    frame = load_inputs(data_root,training_sample=True)
    cutoff = frame.loc[frame.date.lt(day.to_period('M').start_time),'date'].max()
    version = f'{MODEL_REVISION}-bundle-{day:%Y%m}'
    model = fit_bundle_at(frame,frozen['heads'],cutoff,version,manifest['features'])
    latest = pd.read_parquet(data_root/'latest_inputs.parquet')
    forecast = model.predict(make_examples(latest,model.feature_columns),explain=True)
    forecast = forecast[[*PREDICTION_OUTPUTS,'explanations','explanation_basis']]
    forecast = complete_market_snapshot(forecast,pd.read_parquet(data_root/'securities.parquet'),day,version,cutoff)
    folder = model_root/'candidates'/version
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'model.pkl').open('wb') as stream:pickle.dump(model,stream)
    snapshot = folder/f'forecasts_{day:%Y%m%d}.parquet'
    forecast.to_parquet(snapshot,index=False)
    forecast.drop(columns=['explanations']).to_csv(folder/f'forecasts_{day:%Y%m%d}.csv',index=False)
    result = {'status':'candidate_only','eligible':False,'reason':'候选模型，正式发布仍须完整验收',
        'model_version':version,'data_as_of':day.date().isoformat(),'model_trained_as_of':str(cutoff.date()),
        'model_path':str((folder/'model.pkl').resolve()),'forecast_path':str(snapshot.resolve()),
        'securities':int(forecast.security_id.nunique()),'scored_rows':int(forecast.status.eq('ok').sum()),
        'model_metadata':model.metadata}
    write_json(folder/'candidate.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description='全港股模型训练流程')
    parser.add_argument('stage', choices=['normalize','development','freeze','confirmation','candidate'])
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--results-root',type=Path,default=RESULTS)
    parser.add_argument('--model-root',type=Path,default=MODELS)
    args = parser.parse_args()
    if args.stage == 'normalize':
        normalize_dataset(args.data_root)
    elif args.stage == 'development':
        run_development(args.data_root,args.results_root,args.model_root)
    elif args.stage == 'freeze':
        print(json.dumps(freeze_architecture(args.data_root,args.results_root,args.model_root)['heads']))
    elif args.stage == 'confirmation':
        run_confirmation(args.data_root,args.results_root,args.model_root)
    else:
        print(export_candidate(args.data_root,args.results_root,args.model_root)['forecast_path'])


if __name__ == '__main__':
    main()
