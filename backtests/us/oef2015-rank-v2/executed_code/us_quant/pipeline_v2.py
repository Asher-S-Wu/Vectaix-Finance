"""Predeclared nested chronological rank study with explicit reused diagnostics."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import pickle
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from hk_quant.training import make_examples
from .data import write_json,write_immutable_json,sha256
from .training import split_examples,compare_candidates
from .ranking_v2 import RankingModel
from .evaluation import evaluate

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/us/oef2015_v2'
DEFAULT_RUN='oef2015-rank-v2'


def fold_examples(frame,evaluation_year,window_years,features,horizons=(20,)):
    calibration_start=pd.Timestamp(f'{evaluation_year-1}-01-01');asof=pd.Timestamp(f'{evaluation_year-1}-12-31')
    start=max(pd.Timestamp('2016-01-01'),calibration_start-pd.DateOffset(years=window_years))
    tr,ca=split_examples(frame,features,calibration_start,asof,start)
    return tr.loc[tr.horizon.isin(horizons)].reset_index(drop=True),ca.loc[ca.horizon.isin(horizons)].reset_index(drop=True),asof


def select_kind(annual,candidate_order,years):
    stats={}
    for kind in candidate_order:
        values=np.array([annual[str(y)][kind]['rank_ic_mean'] for y in years],float)
        if not np.isfinite(values).all():raise ValueError('Every declared year must have a finite common-cohort metric')
        stats[kind]=dict(mean_annual_ic=float(values.mean()),median_annual_ic=float(np.median(values)),worst_annual_ic=float(values.min()),positive_years=int((values>0).sum()),years=list(years),annual_ic=values.tolist())
    selected=max((stats[k]['mean_annual_ic'],-i,k) for i,k in enumerate(candidate_order))[2]
    return selected,stats


def nested_selection(annual,candidate_order):
    result=[]
    for year in [2022,2023,2024]:
        inner=list(range(2020,year));kind,stats=select_kind(annual,candidate_order,inner)
        result.append(dict(outer_year=year,inner_years=inner,selected_kind=kind,inner_mean_ic=stats[kind]['mean_annual_ic'],outer_ic=annual[str(year)][kind]['rank_ic_mean'],own_outer_year_used_for_selection=False))
    return result


def examples_for(frame,features,horizons):
    return make_examples(frame,features).loc[lambda f:f.horizon.isin(horizons)].reset_index(drop=True)


def dump_model(model,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Model artifact already exists')
    path.write_bytes(pickle.dumps(model,protocol=pickle.HIGHEST_PROTOCOL));write_json(path.with_suffix('.json'),model.metadata)
    return sha256(path)


def run(data=DATA,run_name=DEFAULT_RUN):
    if not run_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in run_name):raise ValueError('Unsafe run name')
    data=Path(data);results=ROOT/'backtests/us'/run_name;models=ROOT/'models/us'/run_name;outputs=data/'predictions'/run_name
    if any(p.exists() for p in [results,models,outputs]):raise FileExistsError('Use a new run namespace; prior complete or partial experiment stays intact')
    cfg_path=ROOT/'docs/us/v2/proposed_protocol.json';cfg=json.loads(cfg_path.read_text());config_hash=sha256(cfg_path)
    source=json.loads((data/'source_audit.json').read_text())
    if source['normalized_sha256']!=sha256(data/'normalized.parquet'):raise ValueError('Recovered normalized data changed')
    code=[*sorted((ROOT/'us_quant').glob('*.py')),*[ROOT/'hk_quant'/f for f in ['models.py','training.py','prediction_tasks.py','contracts.py','market_context.py']]]
    code_hash={str(p.relative_to(ROOT)):sha256(p) for p in code}
    results.mkdir(parents=True);models.mkdir(parents=True);outputs.mkdir(parents=True)
    contract={**cfg,'status':'frozen_before_fit','config_sha256':config_hash,'source_audit_sha256':sha256(data/'source_audit.json'),'code_sha256':code_hash,'data_source_version':source['dataset_version'],'recent_diagnostic_is_untouched':False}
    write_immutable_json(results/'protocol.json',contract);write_json(results/'source_audit.json',source)
    for path in code:
        dest=results/'executed_code'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    started=datetime.now(timezone.utc).isoformat();write_json(results/'training_status.json',dict(status='running',started_at=started,stage='walk_forward_development'))
    frame=pd.read_parquet(data/'normalized.parquet');features=cfg['features'];order=list(cfg['candidates']);annual={};fit_log=[]
    for year in cfg['walk_forward_years']:
        predictions={};evaluation=frame.loc[frame.date.between(f'{year}-01-01',f'{year}-12-31')]
        for kind in order:
            if sha256(cfg_path)!=config_hash:raise ValueError('Candidate protocol changed during training')
            t0=time.monotonic();params=cfg['candidates'][kind];tr,ca,asof=fold_examples(frame,year,params['fit_window_years'],features)
            print('fit',year,kind,len(tr),len(ca),flush=True)
            model=RankingModel(kind,model_version=f'us-v2-{kind}-eval{year}',horizons=(20,)).fit(tr,ca,features,asof)
            model.metadata.update(protocol_sha256=sha256(results/'protocol.json'),source_audit_sha256=contract['source_audit_sha256'],fold_evaluation_year=year)
            digest=dump_model(model,models/'folds'/str(year)/f'{kind}.pkl')
            predicted=model.predict(examples_for(evaluation,model.feature_columns,(20,)));predictions[kind]=predicted
            dest=outputs/'development'/str(year);dest.mkdir(parents=True,exist_ok=True);predicted.to_parquet(dest/f'{kind}.parquet',index=False)
            metrics,daily=evaluate(predicted,f'{year}-12-31');write_json(results/'development'/str(year)/f'{kind}.json',metrics['horizons']['20']);daily.to_csv(results/'development'/str(year)/f'{kind}_daily_ic.csv',index=False)
            fit_log.append(dict(year=year,kind=kind,seconds=time.monotonic()-t0,model_sha256=digest,train_rows=len(tr),calibration_rows=len(ca)))
            print('done',year,kind,'IC20',metrics['horizons']['20']['rank_ic_mean'],flush=True)
        annual[str(year)]=compare_candidates(predictions,f'{year}-12-31')
        write_json(results/'annual_common_ic.json',annual);write_json(results/'fit_log.json',fit_log)
    selected,selection=select_kind(annual,order,cfg['walk_forward_years']);nested=nested_selection(annual,order)
    write_json(results/'development_selection.json',selection);write_json(results/'nested_selection.json',dict(outer_path=nested,mean_outer_ic=float(np.mean([r['outer_ic'] for r in nested])),scope='Chronological outer architecture choices use only earlier inner folds, still retrospective vendor data'))
    frozen=dict(selected_kind=selected,selection_statistic=selection[selected]['mean_annual_ic'],selection_set='2020–2024 rolling development',selection_rule=cfg['selection_rule'],recent_diagnostic_used_for_selection=False,
        annual_common_ic_sha256=sha256(results/'annual_common_ic.json'),selection_sha256=sha256(results/'development_selection.json'),nested_selection_sha256=sha256(results/'nested_selection.json'))
    write_immutable_json(results/'frozen_architecture.json',frozen);print('FROZEN',frozen,flush=True)
    params=cfg['candidates'][selected];tr,ca,asof=fold_examples(frame,2025,params['fit_window_years'],features,horizons=(1,5,20,60))
    model=RankingModel(selected,model_version=f'us-v2-{selected}-frozen-20241231',horizons=(1,5,20,60)).fit(tr,ca,features,asof)
    model.metadata.update(protocol_sha256=sha256(results/'protocol.json'),source_audit_sha256=contract['source_audit_sha256'],evaluation_context='2025–2026 is an already-viewed reused diagnostic, not a fresh holdout')
    frozen_digest=dump_model(model,models/'frozen/model.pkl')
    recent=frame.loc[frame.date.between('2024-12-31','2026-09-30')]
    predicted=model.predict(examples_for(recent,model.feature_columns,(1,5,20,60)));predicted.to_parquet(outputs/'frozen_recent.parquet',index=False)
    diagnostic=predicted.loc[predicted.date.ge('2025-01-01')]
    summary,daily=evaluate(diagnostic,'2026-09-30');summary['evaluation_status']='reused_diagnostic_after_experiment1';summary['untouched_holdout']=False
    write_json(results/'reused_diagnostic_summary.json',summary);daily.to_csv(results/'reused_diagnostic_daily_ic.csv',index=False)
    # Latest model has no independent OOS evaluation, regardless of diagnostic outcome.
    boundary=pd.Timestamp('2025-10-01');start=max(pd.Timestamp('2016-01-01'),boundary-pd.DateOffset(years=params['fit_window_years']))
    tr,ca=split_examples(frame,features,boundary,'2026-09-30',start)
    latest=RankingModel(selected,model_version=f'us-v2-{selected}-latest-20260930',horizons=(1,5,20,60)).fit(tr,ca,features,'2026-09-30')
    latest.metadata.update(protocol_sha256=sha256(results/'protocol.json'),source_audit_sha256=contract['source_audit_sha256'],out_of_sample_evaluated=False)
    latest_digest=dump_model(latest,models/'latest/model.pkl')
    latest.predict(examples_for(frame.loc[frame.date.eq('2026-09-30')],latest.feature_columns,(1,5,20,60))).to_parquet(outputs/'latest.parquet',index=False)
    write_json(results/'latest_refit_summary.json',dict(kind=selected,model_version=latest.model_version,as_of='2026-09-30',model_sha256=latest_digest,out_of_sample_evaluated=False,eligible=False,execution_validated=False))
    write_json(results/'training_status.json',dict(status='complete',started_at=started,finished_at=datetime.now(timezone.utc).isoformat(),selected_kind=selected,
        candidate_fold_fits=len(fit_log),frozen_model_sha256=frozen_digest,latest_model_sha256=latest_digest,untouched_recent_holdout=False,recent_period_used_for_selection=False,eligible=False,execution_validated=False))
    return frozen,summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--run-name',default=DEFAULT_RUN);args=p.parse_args();run(args.data_root,args.run_name)
