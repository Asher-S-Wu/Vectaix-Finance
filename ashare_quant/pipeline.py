"""Reproducible A-share collection -> training -> confirmation -> report command."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from .paths import DATA,MODELS,RESULTS
from .dataset import prepare_data,build_features,write_json
from .training import run_training,initialize_run
from .collect import collect_market,audit_collection
from .report import benchmark_curves,build_report


def run_backtests(data_root,results_root,*,collect_actions=False):
    from .replay import run_replay,run_reference_replay
    data_root,results_root=Path(data_root),Path(results_root)
    frozen=json.loads((results_root/'frozen_architecture.json').read_text())
    manifest=json.loads((data_root/'feature_manifest.json').read_text())
    calendar_frame=pd.read_parquet(data_root/'calendar.parquet')
    calendar=pd.DatetimeIndex(calendar_frame.loc[calendar_frame.is_open.eq(1),'cal_date']).sort_values()
    month_ends=set(calendar[:-1][calendar[:-1].to_period('M')!=calendar[1:].to_period('M')])
    paths=sorted((results_root/'confirmation'/frozen['selected_kind']).glob('*.parquet'))
    rankings=[]
    for path in paths:
        part=pd.read_parquet(path,filters=[('horizon','==',20)])
        part=part.loc[part.date.isin(month_ends)]
        rankings.append(part)
    forecasts=pd.concat(rankings,ignore_index=True)
    if forecasts.empty:raise ValueError('No confirmation month-end scores')
    eligible=forecasts.loc[forecasts.score_status.eq('ok') & pd.to_numeric(forecasts.score,errors='coerce').notna()]
    selected=eligible.sort_values(['date','score','security_id'],ascending=[True,False,True]).groupby('date',sort=False).head(30)
    ids=set(selected.security_id)
    # Full rankings are already persisted; load raw bars only for selected names.
    frames=[]
    for year in range(forecasts.date.min().year,pd.Timestamp(manifest['data_as_of']).year+1):
        frame=pd.read_parquet(data_root/'bars'/f'{year}.parquet',filters=[('security_id','in',sorted(ids))])
        frames.append(frame.loc[frame.date.ge(forecasts.date.min())])
    bars=pd.concat(frames,ignore_index=True)
    # Retiring holdings still need the current signal's trailing liquidity.
    # This is a same-date join, never future execution-day ADV.
    signal_liquidity=forecasts[['date','security_id','adv20_amount']]
    bars=bars.drop(columns='adv20_amount',errors='ignore').merge(signal_liquidity,on=['date','security_id'],how='left',validate='one_to_one')
    issues=pd.DataFrame(manifest.get('adjustment_reference_issues',[]))
    bars['reference_continuity_break']=False
    if not issues.empty:
        issues['date']=pd.to_datetime(issues.date)
        issue_keys=pd.MultiIndex.from_frame(issues[['date','security_id']].drop_duplicates())
        bars['reference_continuity_break']=pd.MultiIndex.from_frame(bars[['date','security_id']]).isin(issue_keys)
    if collect_actions:
        from .corporate_actions import collect_selected_dividends
        terminal_path=data_root/'raw'/'collection_run.json'
        terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
        completed=(terminal.get('status') in ['complete','incomplete'] and terminal.get('requested_dates')==len(calendar) and str(terminal.get('end','')).replace('-','')==manifest['data_as_of'].replace('-',''))
        if completed:
            evidence=collect_selected_dividends(data_root,sorted(ids),model_frozen=True,core_collection_stopped=True,asof_date=manifest['data_as_of'])
        else:
            evidence={'collection_status':'not_collected_core_completion_unverified','queried':False}
        write_json(results_root/'corporate_actions_status.json',evidence)
    actions_path=data_root/'references'/'verified_corporate_actions.parquet'
    actions=pd.read_parquet(actions_path) if actions_path.exists() else None
    actual=run_replay(forecasts,bars,calendar,actions=actions,start_date='2025-01-01',end_date=manifest['data_as_of'])
    directory=results_root/'execution';directory.mkdir(exist_ok=True)
    for name,value in actual.items():
        if name!='summary':pd.DataFrame(value).to_parquet(directory/f'{name}.parquet',index=False)
    write_json(results_root/'execution_summary.json',actual['summary'])
    references={};dates=None
    for fee in [.001,.0025,.005]:
        result=run_reference_replay(forecasts,bars,calendar,fee=fee,end_date=manifest['data_as_of'])
        label=f'fee_{fee:.4f}';sub=results_root/'reference'/label;sub.mkdir(parents=True,exist_ok=True)
        for name,value in result.items():
            if name=='summary':continue
            frame=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
            frame.to_parquet(sub/f'{name}.parquet',index=False)
            if name in ['equity','daily'] and fee==.0025:dates=pd.to_datetime(frame.date)
        references[label]=result['summary']
    write_json(results_root/'reference_summary.json',references)
    if dates is not None:
        curves,summaries=benchmark_curves(pd.read_parquet(data_root/'references'/'benchmarks.parquet'),dates)
        curves.to_parquet(results_root/'benchmark_curves.parquet',index=False)
        write_json(results_root/'benchmark_summary.json',summaries)
    status=json.loads((results_root/'training_status.json').read_text());status['execution_replay_status']='complete_with_disclosed_limitations';write_json(results_root/'training_status.json',status)
    return actual['summary'],references


def run_pipeline(data_root=DATA,results_root=RESULTS,model_root=MODELS,*,skip_collection=False,wait_for_collection=False,skip_corporate_actions=False):
    data_root,results_root,model_root=map(Path,[data_root,results_root,model_root])
    initialize_run(results_root)
    raw=data_root/'raw'
    if not skip_collection:
        collect_market(root=raw,start='20160101',end='20260930',max_workers=12,requests_per_minute=60,newest_first=False)
    while True:
        readiness=audit_collection(raw,verify_checksums=False)
        write_json(results_root/'source_readiness.json',readiness)
        if readiness['core_status']=='complete':break
        progress_path=raw/'collection_progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        if not wait_for_collection or progress.get('stopped_endpoints'):
            raise ValueError('Core source dataset incomplete; consult source_readiness.json')
        print('waiting for authorized historical source collection',flush=True)
        time.sleep(120)
    # This call re-reads and verifies the hash of every source partition.
    prepare_data(data_root)
    build_features(data_root)
    trained=run_training(data_root,results_root,model_root)
    run_backtests(data_root,results_root,collect_actions=not skip_corporate_actions)
    report=build_report(data_root,results_root,model_root)
    write_json(results_root/'pipeline_status.json',dict(status='complete',training=trained,report=report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--results-root',type=Path,default=RESULTS);p.add_argument('--model-root',type=Path,default=MODELS)
    p.add_argument('--skip-collection',action='store_true');p.add_argument('--wait-for-collection',action='store_true');p.add_argument('--skip-corporate-actions',action='store_true')
    a=p.parse_args()
    print(json.dumps(run_pipeline(a.data_root,a.results_root,a.model_root,skip_collection=a.skip_collection,wait_for_collection=a.wait_for_collection,skip_corporate_actions=a.skip_corporate_actions),ensure_ascii=False))
