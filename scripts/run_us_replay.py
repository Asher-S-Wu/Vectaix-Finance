"""Reproduce bounded-cohort adjusted-unit portfolio illustrations after freeze."""
from pathlib import Path
import json
import pandas as pd
from us_quant.data import sha256,write_json
from us_quant.portfolio import replay

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/us/oef2015';RESULTS=ROOT/'backtests/us/oef2015'


def cohort_benchmark_scores(frame):
    f=frame[['date','security_id','status']].copy();f['horizon']=20;f['score']=0.
    f['score_status']=f.status.where(f.status.ne('ok'),'ok')
    return f.drop(columns='status')


def main():
    source=json.loads((RESULTS/'source_audit.json').read_text());frozen=json.loads((RESULTS/'frozen_architecture.json').read_text())
    if sha256(DATA/'prices.parquet')!=source['prices_sha256']:raise ValueError('Replay price evidence changed')
    if sha256(DATA/'normalized.parquet')!=source['normalized_sha256']:raise ValueError('Signal evidence changed')
    if sha256(ROOT/'us_quant/portfolio.py')!=source['code_sha256']['us_quant/portfolio.py']:raise ValueError('Replay code changed after protocol freeze')
    kind=frozen['selected_kind'];prices=pd.read_parquet(DATA/'prices.parquet')
    dev=pd.read_parquet(DATA/'predictions/development'/f'{kind}.parquet')
    confirm=pd.read_parquet(DATA/'predictions/confirmation'/f'{kind}.parquet')
    scores=pd.concat([dev.loc[dev.date.ge('2024-12-01')],confirm],ignore_index=True)
    normalized=pd.read_parquet(DATA/'normalized.parquet');benchmark=cohort_benchmark_scores(normalized)
    spy=pd.read_parquet(DATA/'bars/SPY.parquet');spy['adv20_amount']=spy.dollar_volume_proxy.rolling(20,min_periods=12).mean()
    spy_scores=spy[['date','security_id']].copy();spy_scores['horizon']=20;spy_scores['score']=0.;spy_scores['score_status']='ok'
    jobs=[('selected_15bps',prices,scores,.0015,10),('selected_30bps',prices,scores,.003,10),('cohort_101slots_15bps',prices,benchmark,.0015,101),('spy_15bps',spy,spy_scores,.0015,1)]
    curves=[];summaries={};raw_output=DATA/'replay';raw_output.mkdir(parents=True,exist_ok=True)
    for name,p,s,fee,n in jobs:
        result=replay(p,s,'2025-01-02','2026-09-30',fee_per_side=fee,top_n=n)
        summary=result['summary'];summary['interpretation']='Conditional vendor-adjusted research illustration, not actual-share/broker executable performance'
        summaries[name]=summary
        curve=result['curve'].copy();curve.insert(0,'series',name);curves.append(curve)
        result['trades'].to_parquet(raw_output/f'{name}_trades.parquet',index=False)
        write_json(raw_output/f'{name}_summary.json',summary)
        print(name,summary['total_return'],summary['annualized_return'],summary['max_drawdown'],flush=True)
    aggregate=pd.concat(curves,ignore_index=True);aggregate.to_csv(RESULTS/'replay_curves.csv',index=False)
    # Per-security holdings and order records stay local; public summaries are aggregate only.
    public={name:{k:v for k,v in s.items() if k not in ['terminal_holdings']} for name,s in summaries.items()}
    write_json(RESULTS/'replay_summary.json',dict(selected_kind=kind,start='2025-01-02',end='2026-09-30',conditional_observable_cohort=True,
        full_us_market=False,survivorship_free=False,execution_validated=False,actual_share_corporate_action_ledger_reconstructed=False,
        scope='89 observable original-cohort share classes; twelve missing identities do not become model inputs',
        cohort_benchmark='101 original fixed slots; unavailable or unfillable slots stay cash; same monthly timing/cost/participation',
        spy_reference='External ETF adjusted-price proxy, monthly 95% target and same bps cost assumption; not a pure price index',
        strategies=public,price_sha256=source['prices_sha256'],calendar_sha256=source['calendar_sha256'],curves_sha256=sha256(RESULTS/'replay_curves.csv')))


if __name__=='__main__':main()
