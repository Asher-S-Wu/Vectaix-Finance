"""Aggregate paired reused diagnostics and terminal-aware v2 portfolio illustration."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from us_quant.data import sha256,write_json
from us_quant.evaluation import block_bootstrap
from us_quant.portfolio import replay
from scripts.run_us_replay import cohort_benchmark_scores

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/us/oef2015_v2';RUN='oef2015-rank-v2';RESULTS=ROOT/'backtests/us'/RUN


def paired_comparison(first,second,cutoff):
    out=dict(evaluation_status='posthoc paired diagnostic on previously viewed2025–2026 outcomes',untouched_holdout=False,horizons={})
    for h in [1,5,20,60]:
        frames=[]
        for frame in [first,second]:
            frame=frame.copy()
            for column in ['date','label_end']:frame[column]=pd.to_datetime(frame[column]).astype('datetime64[ns]')
            f=frame.loc[frame.horizon.eq(h)&frame.date.ge('2025-01-01')&frame.date.le(pd.Timestamp(cutoff))].set_index(['date','security_id'])
            if f.index.has_duplicates:raise ValueError('Duplicate prediction keys')
            frames.append(f.loc[f.score_status.eq('ok')&np.isfinite(f.score)])
        common=frames[0].index.intersection(frames[1].index).sort_values();a,b=[f.reindex(common) for f in frames]
        if not np.allclose(a.fwd_return,b.fwd_return,equal_nan=True) or not a.label_end.equals(b.label_end):raise ValueError('Paired source labels disagree')
        mature=a.label_end.le(pd.Timestamp(cutoff))&np.isfinite(a.fwd_return);a=a.loc[mature];b=b.loc[mature]
        values=[]
        for date,day in a.groupby(level='date'):
            if len(day)<20 or day.fwd_return.nunique()<2:continue
            old=0. if day.score.nunique()<2 else float(day.score.corr(day.fwd_return,method='spearman'))
            newday=b.xs(date,level='date');new=0. if newday.score.nunique()<2 else float(newday.score.corr(newday.fwd_return,method='spearman'))
            values.append((date,old,new))
        if values:
            dates=[x[0] for x in values];old=np.array([x[1] for x in values]);new=np.array([x[2] for x in values]);ci=block_bootstrap(new-old)
            out['horizons'][str(h)]=dict(first_run_ic=float(old.mean()),second_run_ic=float(new.mean()),paired_ic_change=float((new-old).mean()),change_ci_lower=ci[0],change_ci_upper=ci[1],ic_dates=len(values),common_score_rows=len(common),common_mature_rows=len(a),date_sha256=hashlib.sha256('|'.join(str(d.date()) for d in dates).encode()).hexdigest())
        else:out['horizons'][str(h)]=dict(ic_dates=0,common_score_rows=len(common),common_mature_rows=len(a))
    return out


def main():
    source=json.loads((DATA/'source_audit.json').read_text());status=json.loads((RESULTS/'training_status.json').read_text())
    if status['status']!='complete':raise ValueError('Training incomplete')
    for name in ['prices','normalized']:
        if sha256(DATA/f'{name}.parquet')!=source[f'{name}_sha256']:raise ValueError(f'{name} changed')
    new_path=DATA/'predictions'/RUN/'frozen_recent.parquet';scores=pd.read_parquet(new_path)
    first_root=ROOT/'data/us/oef2015';first_kind=json.loads((ROOT/'backtests/us/oef2015/frozen_architecture.json').read_text())['selected_kind']
    old_path=first_root/'predictions/confirmation'/f'{first_kind}.parquet';paired=paired_comparison(pd.read_parquet(old_path),scores,'2026-09-30')
    paired.update(first_predictions_sha256=sha256(old_path),second_predictions_sha256=sha256(new_path));write_json(RESULTS/'paired_recent_summary.json',paired)
    prices=pd.read_parquet(DATA/'prices.parquet');normalized=pd.read_parquet(DATA/'normalized.parquet');benchmark=cohort_benchmark_scores(normalized)
    spy=pd.read_parquet(first_root/'bars/SPY.parquet');spy['adv20_amount']=spy.dollar_volume_proxy.rolling(20,min_periods=12).mean()
    spy_scores=spy[['date','security_id']].copy();spy_scores['horizon']=20;spy_scores['score']=0.;spy_scores['score_status']='ok'
    jobs=[('selected_15bps',prices,scores,.0015,10),('selected_30bps',prices,scores,.003,10),('cohort_101slots_15bps',prices,benchmark,.0015,101),('spy_15bps',spy,spy_scores,.0015,1)]
    curves=[];summaries={};private=DATA/'replay'/RUN;private.mkdir(parents=True,exist_ok=True)
    for name,p,s,fee,n in jobs:
        result=replay(p,s,'2025-01-02','2026-09-30',fee_per_side=fee,top_n=n)
        result['trades'].to_parquet(private/f'{name}_trades.parquet',index=False);write_json(private/f'{name}_summary.json',result['summary'])
        summary={k:v for k,v in result['summary'].items() if k!='terminal_holdings'}
        summary['interpretation']='Reused vendor-adjusted research illustration; unresolved terminal paths are reference-only, never broker-executable performance'
        summary['average_equity_exposure']=float((result['curve'].reference_equity/result['curve'].reference_nav).mean())
        summaries[name]=summary;curve=result['curve'].copy();curve.insert(0,'series',name);curves.append(curve)
        print(name,'return',summary['total_return'],'reference',summary['reference_return_metrics']['total_return'],'unknown',summary['terminal_unknown_count'],'gap_sessions',summary['valuation_gap_sessions'],flush=True)
    combined=pd.concat(curves,ignore_index=True);combined.to_csv(RESULTS/'replay_curves.csv',index=False)
    write_json(RESULTS/'replay_summary.json',dict(selected_kind=status['selected_kind'],start='2025-01-02',end='2026-09-30',reused_diagnostic=True,untouched_holdout=False,execution_validated=False,
        original_identity_terminal_entitlements_complete=False,cohort_benchmark='All101originalslots; unavailable names remain cash, WBA disappearance retains unresolved holding rather than drop or assumed cash liquidation',
        strategies=summaries,prices_sha256=source['prices_sha256'],predictions_sha256=sha256(new_path),curves_sha256=sha256(RESULTS/'replay_curves.csv'),replay_code_sha256=sha256(ROOT/'us_quant/portfolio.py')))


if __name__=='__main__':main()
