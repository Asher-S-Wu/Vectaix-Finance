"""保存组合与同约束基准的逐笔回放，供验收与对账读取。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import RiskProfile
from .strategy import run_strategy_backtest


def _json_value(value):
    if isinstance(value,pd.DataFrame):
        return json.loads(value.to_json(orient='records',date_format='iso'))
    if isinstance(value,dict):return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_value(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if isinstance(value,np.generic):return _json_value(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(_json_value(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def export_backtest(data_root,forecast_dir,results_root,*,start,end,initial_cash=1000000.,profile=RiskProfile(),phase='development'):
    if phase not in ('development','confirmation'):raise ValueError('回测阶段无效')
    output=Path(results_root);output.mkdir(parents=True,exist_ok=True)
    metric_path=output/f'{phase}_portfolio_metrics.json'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest={'status':'running','phase':phase,'run_id':stamp,'data_root':str(Path(data_root).resolve()),
              'forecast_dir':str(Path(forecast_dir).resolve()),'start':str(pd.Timestamp(start).date()),
              'end':str(pd.Timestamp(end).date()),'initial_cash':initial_cash,'profile':asdict(profile),
              'cash_payment_policy':'issuer_final_schedule_simulated or actual_account_receipt; receivables are not cash'}
    _write(metric_path,{'portfolio_valid':False,'reason':'本次回测尚未完成','run_id':stamp})
    _write(output/f'{phase}_portfolio_run.json',manifest)
    result=run_strategy_backtest(data_root,forecast_dir,start=start,end=end,initial_cash=initial_cash,profile=profile)
    if result['runs']:
        folder=output/'portfolio_replay'/stamp
        folder.mkdir(parents=True)
        for name,run in result['runs'].items():
            destination=folder/name;destination.mkdir()
            for key,value in run.items():
                if isinstance(value,pd.DataFrame):value.to_parquet(destination/f'{key}.parquet',index=False)
            _write(destination/'summary.json',run['summary'])
            _write(destination/'policy_reports.json',result['policies'][name])
        manifest['replay_directory']=str(folder.resolve())
    metrics=result['portfolio_metrics']
    if metrics is None:metrics={'portfolio_valid':False,'reason':result['reason']}
    _write(metric_path,{**metrics,'run_id':stamp})
    manifest.update(status=result['status'],reason=result.get('reason'),execution_summary=result.get('execution_summary'),
                    completed_at=datetime.now(timezone.utc).isoformat())
    _write(output/f'{phase}_portfolio_run.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description='组合回测及逐笔对账输出')
    for name in ('data-root','forecast-dir','results-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--start',required=True);parser.add_argument('--end',required=True)
    parser.add_argument('--initial-cash',type=float,default=1000000.)
    parser.add_argument('--phase',choices=('development','confirmation'),default='development')
    args=parser.parse_args()
    result=export_backtest(args.data_root,args.forecast_dir,args.results_root,start=args.start,end=args.end,
                           initial_cash=args.initial_cash,phase=args.phase)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='ok' else 2


if __name__=='__main__':raise SystemExit(main())
