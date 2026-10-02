"""Reproducible local US research build, actual training and evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import pickle
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from hk_quant.training import KINDS,FACTOR_FEATURES,LINEAR_FEATURES,predict_frame
from . import HORIZONS
from .data import write_json,write_immutable_json,sha256
from .features import security_features,add_market_context,normalize_features,ALL_FEATURES,PRICE_FEATURES
from .models import USModel
from .training import protocol,split_examples,compare_candidates,freeze_selection
from .evaluation import evaluate

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/us/oef2015'
RESULTS=ROOT/'backtests/us/oef2015'
MODELS=ROOT/'models/us/oef2015/frozen'


def validate_calendar(calendar):
    dates=pd.DatetimeIndex(calendar)
    if dates.duplicated().any() or not dates.is_monotonic_increasing or (dates.dayofweek>4).any():raise ValueError('Invalid shared US session calendar')
    return dates


def absent_features(identity,calendar,status):
    f=pd.DataFrame({'date':calendar,'security_id':identity,'status':status,'quote_present':False,'adj_close':np.nan,'adv20_amount':np.nan})
    for c in PRICE_FEATURES:f[c]=np.nan
    for h in HORIZONS:f[f'fwd_return_{h}']=np.nan;f[f'label_end_{h}']=pd.Series(calendar).shift(-h).values
    return f


def select_training_features(frame,kind):
    if kind=='factor':return list(FACTOR_FEATURES)
    if kind=='linear':return list(LINEAR_FEATURES)
    observed=frame.loc[frame.date.between('2016-01-01','2022-12-31')&frame.status.eq('ok'),ALL_FEATURES]
    coverage=observed.notna().mean()
    return coverage.index[coverage.ge(.2)].tolist()


def validate_approved_identity(record):
    if record.get('instrument')!='EQUITY':raise ValueError('Original cohort must be an equity, not a reused ETF ticker')
    if pd.Timestamp(record['first_date'])>pd.Timestamp('2015-06-30'):
        raise ValueError('Source does not support the historical identity; review ticker reuse or incomplete history')


def require_fresh_run(data,results,models,stage):
    data,results,models=Path(data),Path(results),Path(models)
    if stage=='build':
        protected=[results/'source_audit.json',data/'features.parquet',data/'normalized.parquet',data/'prices.parquet',data/'calendar.csv']
    else:
        protected=[results/'training_status.json',results/'frozen_architecture.json',data/'predictions',models.parent/'latest',*models.glob('*.pkl'),*models.glob('*.json')]
    if any(path.exists() for path in protected):
        raise FileExistsError('Run already contains artifacts; use new data/results/model namespaces, never overwrite completed or partial evidence')


def build(data=DATA,results=RESULTS):
    require_fresh_run(data,results,MODELS,'build')
    data=Path(data);results=Path(results);results.mkdir(parents=True,exist_ok=True)
    universe=ROOT/'docs/us/declared_universe.csv';cohort=pd.read_csv(universe)
    collection=json.loads((data/'collection_audit.json').read_text())
    review_path=ROOT/'docs/us/identity_review.json'
    if not review_path.exists():raise ValueError('Explicit source identity review required before build')
    identity_review=json.loads(review_path.read_text())
    allowed=set(identity_review['accepted_original_identities'])
    source_records={r['security_id']:r for r in collection['records']}
    for identity in allowed:validate_approved_identity(source_records[identity])
    if not allowed.issubset(set(cohort.ticker_2015)):raise ValueError('Unexpected cohort identity')
    spy=pd.read_parquet(data/'bars/SPY.parquet')
    if not spy.data_valid.all():raise ValueError('Benchmark has invalid quote; calendar needs review')
    calendar=validate_calendar(spy.date)
    if calendar.min()!=pd.Timestamp('2014-01-02') or calendar.max()!=pd.Timestamp('2026-09-30'):raise ValueError('Source window differs from declared protocol')
    write_immutable_json(results/'protocol.json',protocol(sha256(universe)))
    frames=[];prices=[];audit=[]
    for row in cohort.itertuples():
        identity=row.ticker_2015;path=data/'bars'/f'{identity}.parquet'
        if identity in allowed:
            if not path.exists():raise ValueError(f'Missing reviewed source {identity}')
            bars=pd.read_parquet(path)
            outside=bars.loc[~bars.date.isin(calendar)]
            if len(outside):raise ValueError(f'{identity} quotes outside shared SPY session calendar')
            features=security_features(bars,calendar)
            daily=bars.set_index('date').reindex(calendar).reset_index(names='date');daily['security_id']=identity
            daily['quote_present']=daily.quote_present.fillna(False).astype(bool);daily['data_valid']=daily.data_valid.fillna(False).astype(bool)
            daily=daily.merge(features[['date','adv20_amount']],on='date',validate='one_to_one')
            prices.append(daily)
            returns=features.return_1
            audit.append(dict(security_id=identity,status='available_current_vintage_history',rows=len(bars),valid_quote_rows=int((bars.quote_present&bars.data_valid).sum()),eligible_sessions=int(features.status.eq('ok').sum()),first_date=str(bars.date.min().date()),last_date=str(bars.date.max().date()),large_adjusted_moves=int(returns.abs().gt(.4).sum())))
        else:
            status=identity_review.get('excluded_reasons',{}).get(identity,'source_history_unavailable')
            features=absent_features(identity,calendar,status)
            audit.append(dict(security_id=identity,status=status,rows=0,eligible_sessions=0))
            prices.append(pd.DataFrame(dict(date=calendar,security_id=identity,adj_close=np.nan,quote_present=False,data_valid=False,dollar_volume_proxy=np.nan,adv20_amount=np.nan)))
        frames.append(features)
    features=add_market_context(pd.concat(frames,ignore_index=True),calendar)
    normalized=normalize_features(features)
    features.to_parquet(data/'features.parquet',index=False)
    normalized.to_parquet(data/'normalized.parquet',index=False)
    pd.concat(prices,ignore_index=True).to_parquet(data/'prices.parquet',index=False)
    pd.DataFrame({'date':calendar}).to_csv(data/'calendar.csv',index=False)
    coverage=dict(declared_cohort=101,available_original_identities=len(allowed),unavailable_original_identities=101-len(allowed),rows=len(features),calendar_sessions=len(calendar),start=str(calendar.min().date()),end=str(calendar.max().date()),
        calendar_source='SPY daily observed US sessions, shared across every original cohort identity; no per-ticker row compression',
        cohort_survivorship_free=False,full_us_market=False,source_current_vintage=True,records=audit,
        collection_audit_sha256=sha256(data/'collection_audit.json'),identity_review_sha256=sha256(review_path),universe_sha256=sha256(universe),
        normalized_sha256=sha256(data/'normalized.parquet'),features_sha256=sha256(data/'features.parquet'),
        prices_sha256=sha256(data/'prices.parquet'),calendar_sha256=sha256(data/'calendar.csv'),
        raw_sha256={p.name:sha256(p) for p in sorted((data/'raw').glob('*.json'))},
        bar_sha256={p.name:sha256(p) for p in sorted((data/'bars').glob('*.parquet'))},
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [*sorted((ROOT/'us_quant').glob('*.py')),*[ROOT/'hk_quant'/f for f in ['models.py','training.py','market_context.py','prediction_tasks.py','contracts.py']]]})
    write_json(results/'source_audit.json',coverage)
    return coverage


def train(data=DATA,results=RESULTS,models=MODELS):
    require_fresh_run(data,results,models,'train')
    data=Path(data);results=Path(results);models=Path(models);models.mkdir(parents=True,exist_ok=True)
    contract=protocol(sha256(ROOT/'docs/us/declared_universe.csv'));write_immutable_json(results/'protocol.json',contract)
    source=json.loads((results/'source_audit.json').read_text())
    if source['normalized_sha256']!=sha256(data/'normalized.parquet'):raise ValueError('Dataset changed after source audit')
    for name,digest in source['code_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError(f'Code changed after source audit: {name}')
    frame=pd.read_parquet(data/'normalized.parquet');dev=frame.loc[frame.date.between('2024-01-01','2024-12-31')].copy()
    development={};devframes={};fit_metadata={};started=datetime.now(timezone.utc).isoformat()
    write_json(results/'training_status.json',dict(status='running',started_at=started,protocol_sha256=sha256(results/'protocol.json')))
    for kind in KINDS:
        t0=time.monotonic();print('fit',kind,flush=True)
        features=select_training_features(frame,kind);training,calibration=split_examples(frame,features)
        model=USModel(kind=kind,model_version=f'us-oef2015-{kind}-frozen-20261002-v1').fit(training,calibration,features,pd.Timestamp('2023-12-29'))
        model.metadata.update(protocol_sha256=sha256(results/'protocol.json'),source_audit_sha256=sha256(results/'source_audit.json'),universe_sha256=source['universe_sha256'])
        path=models/f'{kind}.pkl';path.write_bytes(pickle.dumps(model,protocol=pickle.HIGHEST_PROTOCOL));write_json(models/f'{kind}.json',model.metadata)
        predictions=predict_frame(model,dev);(data/'predictions/development').mkdir(parents=True,exist_ok=True)
        predictions.to_parquet(data/'predictions/development'/f'{kind}.parquet',index=False)
        metrics,daily=evaluate(predictions,'2024-12-31');development[kind]=metrics;devframes[kind]=predictions
        write_json(results/'development'/f'{kind}.json',metrics);daily.to_csv(results/'development'/f'{kind}_daily_ic.csv',index=False)
        fit_metadata[kind]=dict(seconds=time.monotonic()-t0,model_sha256=sha256(path),features=features,train_rows=len(training),calibration_rows=len(calibration))
        write_json(results/'fit_metadata.json',fit_metadata)
        print('fitted',kind,'seconds',round(fit_metadata[kind]['seconds'],1),'IC20',metrics['horizons']['20']['rank_ic_mean'],flush=True)
        del model,training,calibration
    scores=compare_candidates(devframes,'2024-12-31');write_json(results/'development_common_universe.json',scores)
    write_json(results/'development_summary.json',development)
    frozen=freeze_selection(scores,results/'frozen_architecture.json')
    print('frozen',frozen,flush=True)
    selected=frozen['selected_kind'];model=pickle.loads((models/f'{selected}.pkl').read_bytes())
    # Confirmation is scored only after architecture selection is sealed.
    confirm=frame.loc[frame.date.between('2025-01-01','2026-09-30')].copy()
    predictions=predict_frame(model,confirm);(data/'predictions/confirmation').mkdir(parents=True,exist_ok=True)
    predictions.to_parquet(data/'predictions/confirmation'/f'{selected}.parquet',index=False)
    summary,daily=evaluate(predictions,'2026-09-30');write_json(results/'confirmation_summary.json',summary);daily.to_csv(results/'confirmation_daily_ic.csv',index=False)
    # A separate latest refit is a usable research artifact, never a substitute
    # for the frozen confirmation model and never assigned confirmation metrics.
    latest_features=select_training_features(frame,selected)
    latest_train,latest_cal=split_examples(frame,latest_features,'2025-10-01','2026-09-30')
    latest=USModel(kind=selected,model_version=f'us-oef2015-{selected}-latest-20260930-v1').fit(latest_train,latest_cal,latest_features,pd.Timestamp('2026-09-30'))
    latest.metadata.update(out_of_sample_evaluated=False,architecture_source='2024 development frozen selection',protocol_sha256=sha256(results/'protocol.json'),source_audit_sha256=sha256(results/'source_audit.json'))
    latest_dir=models.parent/'latest';latest_dir.mkdir(parents=True,exist_ok=True)
    (latest_dir/'model.pkl').write_bytes(pickle.dumps(latest,protocol=pickle.HIGHEST_PROTOCOL));write_json(latest_dir/'metadata.json',latest.metadata)
    latest_predictions=predict_frame(latest,frame.loc[frame.date.eq(pd.Timestamp('2026-09-30'))])
    latest_predictions.to_parquet(data/'predictions/latest.parquet',index=False)
    write_json(results/'latest_refit_summary.json',dict(model_version=latest.model_version,kind=selected,as_of='2026-09-30',model_sha256=sha256(latest_dir/'model.pkl'),out_of_sample_evaluated=False,eligible=False,execution_validated=False))
    write_json(results/'training_status.json',dict(status='complete',started_at=started,finished_at=datetime.now(timezone.utc).isoformat(),selected_kind=selected,models=fit_metadata,confirmation_used_for_selection=False,execution_validated=False,eligible=False))
    import sklearn,lightgbm,scipy,matplotlib
    write_json(results/'environment_versions.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    return frozen,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['build','train'])
    parser.add_argument('--data-root',type=Path,default=DATA);parser.add_argument('--results-root',type=Path,default=RESULTS);parser.add_argument('--models-root',type=Path,default=MODELS)
    args=parser.parse_args()
    if args.stage=='build':build(args.data_root,args.results_root)
    else:train(args.data_root,args.results_root,args.models_root)
