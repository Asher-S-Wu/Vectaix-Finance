"""Chronological A-share training; artifacts never share the HK namespace."""
from __future__ import annotations
import numpy as np
import pandas as pd
from hk_quant.training import make_examples, sample_observations, cross_sectional_inputs, KINDS, FACTOR_FEATURES, LINEAR_FEATURES
from .models import AShareModel
from .paths import DATA,MODELS,RESULTS


def split_window(frame,features,as_of,max_stocks_per_date=256,calibration_start=None):
    as_of=pd.Timestamp(as_of)
    boundary=pd.Timestamp(calibration_start) if calibration_start is not None else as_of-pd.DateOffset(months=12)
    current=frame.loc[frame.date.le(as_of)&frame.status.eq('ok')].copy()
    examples=make_examples(sample_observations(current,max_stocks_per_date),features)
    mature=examples.fwd_return.notna()&examples.label_end.notna()&examples.sigma_daily.gt(0)
    train=examples.loc[mature&examples.date.lt(boundary)&examples.label_end.lt(boundary)].reset_index(drop=True)
    cal=examples.loc[mature&examples.date.ge(boundary)&examples.label_end.le(as_of)].reset_index(drop=True)
    return train,cal

# Protocol is fixed before looking at development or confirmation performance.
PROTOCOL = {
    'version':'cn-price37-chronological-v1',
    'history_start':'2016-01-01',
    'training_as_of':'2023-12-29',
    'calibration_start':'2023-01-01',
    'development_start':'2024-01-01',
    'development_end':'2024-12-31',
    'confirmation_start':'2025-01-01',
    'selection_rule':'maximum 20-session mean daily rank IC',
    'selection_universe':'common contemporaneous score-available identities across all four candidates',
    'tie_break':['factor','linear','lightgbm_small','lightgbm_large'],
    'training_stocks_per_date':256,
    'inference_scope':'all available eligible historical A shares; no test-data-based universe screen',
    'purge':'train label_end < calibration_start; calibration label_end <= training_as_of',
    'preprocessing':'same-day cross-sectional ranks; unranked sigma; training-only linear scaler',
    'horizons':[1,5,20,60],
}


def initialize_run(results_root,contract=None):
    import json
    from pathlib import Path
    from .dataset import write_json
    root=Path(results_root);root.mkdir(parents=True,exist_ok=True)
    contract=PROTOCOL if contract is None else contract
    path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Existing protocol is immutable; use a separate results namespace')
    if not path.exists():write_json(path,contract)
    return contract


def freeze_architecture(development,results_root):
    import json,hashlib
    from pathlib import Path
    from .dataset import write_json
    digest=hashlib.sha256(json.dumps(development,sort_keys=True,allow_nan=False).encode()).hexdigest()
    path=Path(results_root)/'frozen_architecture.json'
    candidates=[]
    for index,kind in enumerate(KINDS):
        record=development.get(kind,{})
        value=record.get('common_score_universe',record.get('horizons',{}).get('20',{})).get('rank_ic_mean')
        if value is not None and np.isfinite(value):candidates.append((float(value),-index,kind))
    if not candidates:raise ValueError('No legal development score evidence')
    best=max(candidates)
    result=dict(selected_kind=best[2],selection_set='development',selection_rule=PROTOCOL['selection_rule'],
                development_sha256=digest,selection_statistic=best[0],confirmation_used=False,
                selection_universe='common contemporaneous score availability' if all('common_score_universe' in r for r in development.values()) else 'provided development evidence')
    if path.exists():
        existing=json.loads(path.read_text())
        if existing!=result:raise ValueError('Architecture frozen selection integrity mismatch')
        return existing
    write_json(path,result);return result



def compare_common_score_universe(prediction_paths_by_kind,cutoff):
    """Paired development IC on the same observable score universe.

    Availability defines membership before label maturity is considered. The
    full prediction files and per-model task coverage remain separately saved.
    This evaluation subset is never used to filter live/replay entry pools.
    """
    columns=['date','security_id','horizon','score','score_status','fwd_return','label_end']
    frames={};common=None
    for kind,paths in prediction_paths_by_kind.items():
        parts=[pd.read_parquet(path,columns=columns,filters=[('horizon','==',20)]) for path in paths]
        frame=pd.concat(parts,ignore_index=True)
        frame=frame.loc[pd.to_datetime(frame.date).le(pd.Timestamp(cutoff))].set_index(['date','security_id'])
        if frame.index.has_duplicates:raise ValueError('Duplicate common-universe prediction keys')
        frames[kind]=frame
        available=frame.index[frame.score_status.eq('ok') & np.isfinite(frame.score)]
        common=available if common is None else common.intersection(available)
    if common is None or not len(common):raise ValueError('No common score-available development universe')
    common=common.sort_values();first=next(iter(frames.values())).reindex(common)
    labels=pd.to_numeric(first.fwd_return,errors='coerce')
    maturity=pd.to_datetime(first.label_end).le(pd.Timestamp(cutoff)) & np.isfinite(labels)
    results={}
    for kind,frame in frames.items():
        paired=frame.reindex(common)
        if not np.allclose(paired.fwd_return.to_numpy(float),labels.to_numpy(float),equal_nan=True) or not pd.to_datetime(paired.label_end).equals(pd.to_datetime(first.label_end)):
            raise ValueError('Candidate source labels disagree on shared keys')
        observed=paired.loc[maturity].reset_index();ics=[]
        for date,day in observed.groupby('date'):
            if len(day)>=20 and day.score.nunique()>1 and day.fwd_return.nunique()>1:
                value=day.score.corr(day.fwd_return,method='spearman')
                if np.isfinite(value):ics.append(float(value))
        results[kind]=dict(rank_ic_mean=float(np.mean(ics)) if ics else None,ic_dates=len(ics),
            common_score_rows=len(common),common_mature_rows=int(maturity.sum()),
            universe='intersection of contemporaneously score-available identities across all candidates',
            outcome_policy='membership independent of unknown future outcomes; IC uses mature observed outcomes only')
    return results


def fit_at(frame,kind,as_of,version,features,calibration_start=None):
    chosen=list(FACTOR_FEATURES if kind=='factor' else LINEAR_FEATURES if kind=='linear' else features)
    boundary=pd.Timestamp(calibration_start) if calibration_start else pd.Timestamp(as_of)-pd.DateOffset(months=12)
    coverage=frame.loc[frame.date.lt(boundary)&frame.status.eq('ok'),chosen].notna().mean()
    if kind not in ('factor','linear'):chosen=coverage.index[coverage.ge(.20)].tolist()
    if not chosen:raise ValueError('No usable training-period features')
    train,cal=split_window(frame,chosen,as_of,calibration_start=calibration_start)
    return AShareModel(kind=kind,model_version=version).fit(train,cal,chosen,pd.Timestamp(as_of))


def evaluate_files(paths,as_of):
    """Accurate streaming sufficient statistics, full cross-section IC per day.

    Fixed-width probability bins are explicit (different to equal-mass HK bins).
    Missing horizon outcomes remain in coverage denominators; task failures do
    not suppress usable outputs belonging to other tasks.
    """
    from hk_quant.evaluation import _block_bootstrap_mean
    from . import HORIZONS
    states={h:dict(predictions=0,mature_labels=0,score_available=0,probability_available=0,
        interval_available=0,expected_return_available=0,correct_direction=0,actual_up=0,
        brier_sum=0.,baseline_brier_sum=0.,interval_covered=0,pinball_sum=0.,baseline_pinball_sum=0.,mae_sum=0.,
        ic=[],bins=np.zeros((10,3)),task_failures={}) for h in HORIZONS}
    for path in sorted(paths):
        frame=pd.read_parquet(path)
        for h,g in frame.groupby('horizon'):
            state=states[int(h)];state['predictions']+=len(g)
            mature=pd.to_datetime(g.label_end).le(pd.Timestamp(as_of))&np.isfinite(g.fwd_return)
            state['mature_labels']+=int(mature.sum())
            for field in ['score_status','probability_status','interval_status','expected_return_status']:
                for key,value in g[field].value_counts(dropna=False).items():
                    label=field+':'+str(key);state['task_failures'][label]=state['task_failures'].get(label,0)+int(value)
            q=g.loc[mature&g.score_status.eq('ok')]
            state['score_available']+=len(q)
            for date,day in q.groupby('date'):
                if len(day)>=20 and day.score.nunique()>1 and day.fwd_return.nunique()>1:
                    ic=day.score.corr(day.fwd_return,method='spearman')
                    if np.isfinite(ic):state['ic'].append((str(pd.Timestamp(date).date()),float(ic)))
            q=g.loc[mature&g.probability_status.eq('ok')]
            if len(q):
                y=q.fwd_return.gt(0).to_numpy(float);p=q.probability_up.to_numpy(float);b=q.baseline_probability.to_numpy(float)
                state['probability_available']+=len(q);state['correct_direction']+=int(((p>=.5)==y).sum());state['actual_up']+=int(y.sum())
                state['brier_sum']+=float(((p-y)**2).sum());state['baseline_brier_sum']+=float(((b-y)**2).sum())
                bins=np.minimum((p*10).astype(int),9)
                for i in range(10):
                    z=bins==i;state['bins'][i]+=np.array([z.sum(),p[z].sum(),y[z].sum()])
            q=g.loc[mature&g.interval_status.eq('ok')]
            if len(q):
                state['interval_available']+=len(q);y=q.fwd_return.to_numpy(float)
                state['interval_covered']+=int(((y>=q.q10)&(y<=q.q90)).sum())
                for alpha,name,bname in [(.1,'q10','baseline_q10'),(.5,'q50','baseline_q50'),(.9,'q90','baseline_q90')]:
                    d=y-q[name].to_numpy(float);bd=y-q[bname].to_numpy(float)
                    state['pinball_sum']+=float(np.maximum(alpha*d,(alpha-1)*d).sum()/3)
                    state['baseline_pinball_sum']+=float(np.maximum(alpha*bd,(alpha-1)*bd).sum()/3)
            q=g.loc[mature&g.expected_return_status.eq('ok')]
            state['expected_return_available']+=len(q)
            state['mae_sum']+=float((q.fwd_return-q.expected_return).abs().sum())
    result={'as_of':str(pd.Timestamp(as_of).date()),'horizons':{},'probability_bin_method':'10 fixed equal-width bins','ic_minimum_cross_section':20}
    for h,s in states.items():
        ratio=lambda a,b:float(a/b) if b else None
        values=np.array([v for _,v in s.pop('ic')],float)
        ci=_block_bootstrap_mean(values)
        p=s['probability_available'];n=s['interval_available'];mean=ratio(s['brier_sum'],p);base=ratio(s['baseline_brier_sum'],p)
        bins=s.pop('bins')
        ece=sum(abs(row[1]-row[2]) for row in bins if row[0])
        result['horizons'][str(h)]={k:v for k,v in s.items() if not k.endswith('_sum')}
        result['horizons'][str(h)].update(rank_ic_mean=float(values.mean()) if len(values) else None,ic_dates=len(values),ic_ci_lower=ci[0],ic_ci_upper=ci[1],
            direction_accuracy=ratio(s['correct_direction'],p),actual_up_ratio=ratio(s['actual_up'],p),brier=mean,baseline_brier=base,
            brier_skill=1-mean/base if base else None,ece=ratio(ece,p),interval_coverage=ratio(s['interval_covered'],n),
            pinball=ratio(s['pinball_sum'],n),baseline_pinball=ratio(s['baseline_pinball_sum'],n),mae=ratio(s['mae_sum'],s['expected_return_available']),
            probability_bins=[dict(lower=i/10,upper=(i+1)/10,count=int(v[0]),mean_probability=ratio(v[1],v[0]),observed_up=ratio(v[2],v[0])) for i,v in enumerate(bins)])
    return result


def evaluate_development(paths):
    # Labels ending in confirmation cannot influence architecture selection.
    return evaluate_files(paths,PROTOCOL['development_end'])


def publish_research_snapshot(model,forecasts,model_root,limitations,data_lineage):
    import hashlib,json,pickle
    from pathlib import Path
    from .dataset import write_json
    root=Path(model_root)
    if forecasts.empty or forecasts.date.nunique()!=1:raise ValueError('Snapshot must contain one nonempty date')
    date=str(pd.Timestamp(forecasts.date.iloc[0]).date());version=model.model_version
    if any(c in version for c in ['/','\\','..']):raise ValueError('Unsafe model version')
    directory=root/'snapshots'/version;directory.mkdir(parents=True,exist_ok=True)
    paths={key:str((directory/name).relative_to(root)) for key,name in [('forecast_path','forecasts.parquet'),('metadata_path','metadata.json'),('acceptance_path','acceptance.json')]}
    forecasts.to_parquet(root/paths['forecast_path'],index=False)
    write_json(root/paths['metadata_path'],{**model.metadata,'data_as_of':date,'model_trained_as_of':model.as_of.isoformat(),'data_lineage':data_lineage})
    write_json(root/paths['acceptance_path'],dict(model_version=version,data_as_of=date,research_ready=True,eligible=False,execution_validated=False,limitations=limitations))
    (directory/'model.pkl').write_bytes(pickle.dumps(model,protocol=pickle.HIGHEST_PROTOCOL))
    active=dict(schema_version='ashare-v1',market='CN',currency='CNY',model_version=version,data_as_of=date,
        **paths,research_ready=True,eligible=False,execution_validated=False,
        model_path=str((directory/'model.pkl').relative_to(root)),model_sha256=hashlib.sha256((directory/'model.pkl').read_bytes()).hexdigest(),
        artifact_sha256={key:hashlib.sha256((root/path).read_bytes()).hexdigest() for key,path in paths.items()})
    write_json(root/'active.json',active)
    return active


def predict_period(model,data_root,output,start,end):
    from pathlib import Path
    from hk_quant.training import predict_frame
    root=Path(data_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for year in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1):
        source=root/'normalized'/f'{year}.parquet'
        if not source.exists():raise ValueError(f'Missing normalized year {year}')
        frame=pd.read_parquet(source)
        frame=frame.loc[frame.date.between(pd.Timestamp(start),pd.Timestamp(end))]
        for period,month in frame.groupby(frame.date.dt.to_period('M'),sort=True):
            path=output/f'{period}.parquet'
            predictions=predict_frame(model,month)
            predictions=predictions.merge(month[['date','security_id','adv20_amount']],on=['date','security_id'],validate='many_to_one')
            predictions.to_parquet(path,index=False,row_group_size=50000)
            paths.append(path)
            print(f'predicted model={model.kind} month={period} rows={len(predictions)}',flush=True)
    return paths


def verify_dataset_lineage(data_root,manifest):
    import hashlib
    from pathlib import Path
    root=Path(data_root)
    expected=manifest['training_sample_sha256']
    actual={p.name for p in (root/'training_samples').glob('*.parquet')}
    if actual!=set(expected):raise ValueError('Training sample lineage partitions changed')
    for name,digest in expected.items():
        if hashlib.sha256((root/'training_samples'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Training sample lineage hash mismatch: '+name)
    for relative,digest in manifest.get('artifact_sha256',{}).items():
        path=root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Dataset lineage hash mismatch: '+relative)


def load_frozen_model(model_path,metadata_path,expected_manifest_hash):
    import json,pickle,hashlib
    from pathlib import Path
    metadata=json.loads(Path(metadata_path).read_text())
    if metadata.get('feature_manifest_sha256')!=expected_manifest_hash:raise ValueError('Frozen model data lineage changed')
    payload=Path(model_path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=metadata.get('model_sha256'):raise ValueError('Frozen model hash mismatch')
    return pickle.loads(payload)


def run_training(data_root=DATA,results_root=RESULTS,model_root=MODELS):
    import json,pickle,hashlib,platform
    from pathlib import Path
    from datetime import datetime,timezone
    from .dataset import write_json
    data_root,results_root,model_root=map(Path,[data_root,results_root,model_root])
    initialize_run(results_root)
    manifest=json.loads((data_root/'feature_manifest.json').read_text())
    if manifest['status']!='complete' or manifest['data_as_of']<'2026-01-01':raise ValueError('Complete requested historical dataset required')
    verify_dataset_lineage(data_root,manifest)
    frozen_dir=model_root/'frozen';frozen_dir.mkdir(parents=True,exist_ok=True)
    samples=pd.concat([pd.read_parquet(p) for p in sorted((data_root/'training_samples').glob('*.parquet')) if int(p.stem)<=2023],ignore_index=True)
    candidates={};development={};development_paths={}
    for kind in KINDS:
        model_path=frozen_dir/f'{kind}.pkl';meta_path=frozen_dir/f'{kind}.json'
        if model_path.exists() and meta_path.exists():
            model=load_frozen_model(model_path,meta_path,hashlib.sha256((data_root/'feature_manifest.json').read_bytes()).hexdigest())
        else:
            print(f'training {kind}: authentic historical sample rows={len(samples)}',flush=True)
            model=fit_at(samples,kind,PROTOCOL['training_as_of'],f'cn-{kind}-frozen-20231229-v1',manifest['features'],PROTOCOL['calibration_start'])
            model_path.write_bytes(pickle.dumps(model,protocol=pickle.HIGHEST_PROTOCOL))
            write_json(meta_path,{**model.metadata,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'feature_manifest_sha256':hashlib.sha256((data_root/'feature_manifest.json').read_bytes()).hexdigest()})
        candidates[kind]=model
        paths=predict_period(model,data_root,results_root/'development'/kind,PROTOCOL['development_start'],PROTOCOL['development_end'])
        development_paths[kind]=paths
        development[kind]=evaluate_development(paths)
        write_json(results_root/'development'/f'{kind}.json',development[kind])
    del samples
    common=compare_common_score_universe(development_paths,PROTOCOL['development_end'])
    write_json(results_root/'development_common_universe.json',common)
    for kind in KINDS:
        development[kind]['common_score_universe']=common[kind]
        write_json(results_root/'development'/f'{kind}.json',development[kind])
    frozen=freeze_architecture(development,results_root)
    write_json(results_root/'development_summary.json',development)
    selected=candidates[frozen['selected_kind']]
    print(f'architecture frozen kind={selected.kind}; confirmation starts next',flush=True)
    # Include December signal for a first-January execution without using its
    # outcomes for any architecture choice.
    paths=predict_period(selected,data_root,results_root/'confirmation'/selected.kind,'2024-12-01',manifest['data_as_of'])
    confirmation=evaluate_files([p for p in paths if p.stem>='2025-01'],manifest['data_as_of'])
    write_json(results_root/'confirmation_summary.json',confirmation)
    latest=pd.concat([pd.read_parquet(p) for p in sorted((data_root/'training_samples').glob('*.parquet'))],ignore_index=True)
    as_of=manifest['data_as_of']
    model=fit_at(latest,selected.kind,as_of,f'cn-{selected.kind}-latest-{as_of.replace("-","")}-v1',manifest['features'])
    del latest
    current=pd.read_parquet(data_root/'latest_inputs.parquet')
    forecasts=model.predict(make_examples(current,model.feature_columns),explain=True)
    snapshot=publish_research_snapshot(model,forecasts,model_root,manifest['limitations']+['Research readiness does not imply performance acceptance or broker execution validation.'],
        {'feature_manifest_sha256':hashlib.sha256((data_root/'feature_manifest.json').read_bytes()).hexdigest(),'training_sample_sha256':manifest['training_sample_sha256']})
    status=dict(status='trained',selected_kind=selected.kind,snapshot=snapshot,completed_at=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
        data_as_of=as_of,frozen_model_version=selected.model_version,latest_model_version=model.model_version,execution_replay_status='pending')
    write_json(results_root/'training_status.json',status)
    return status


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--results-root',type=Path,default=RESULTS);p.add_argument('--model-root',type=Path,default=MODELS)
    a=p.parse_args();print(json.dumps(run_training(a.data_root,a.results_root,a.model_root),ensure_ascii=False))
