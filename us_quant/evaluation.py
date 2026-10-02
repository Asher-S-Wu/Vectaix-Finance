"""Conditional observed-outcome metrics with complete cohort coverage counts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import HORIZONS


def block_bootstrap(values,block=60,repetitions=2000):
    values=np.asarray(values,float)
    if len(values)<block or np.isfinite(values).sum()<block:return [None,None]
    rng=np.random.default_rng(42);n=len(values);length=min(block,n);means=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=int(np.ceil(n/length)))
        indices=((starts[:,None]+np.arange(length))%n).ravel()[:n]
        sample=values[indices]
        if np.isfinite(sample).any():means.append(float(np.nanmean(sample)))
    return np.quantile(means,[.025,.975]).tolist()


def evaluate(frame,cutoff):
    output=dict(as_of=str(pd.Timestamp(cutoff).date()),horizons={},cohort_outcome_scope='conditional on observed mature endpoints; unavailable identities retained in predictions',
        bootstrap='60-session circular blocks, 2000 repetitions seed42; missing-day slots preserved',ic_minimum_cross_section=20,
        probability_bin_method='10 fixed equal-width bins')
    daily=[]
    ratio=lambda a,b:float(a/b) if b else None
    for h in HORIZONS:
        g=frame.loc[frame.horizon.eq(h)&frame.date.le(pd.Timestamp(cutoff))]
        mature=g.label_end.le(pd.Timestamp(cutoff))&np.isfinite(g.fwd_return)
        result=dict(predictions=len(g),mature_labels=int(mature.sum()),missing_or_immature_outcomes=int((~mature).sum()),task_failures={})
        for field in ['score_status','probability_status','interval_status','expected_return_status']:
            for key,n in g[field].value_counts(dropna=False).items():result['task_failures'][f'{field}:{key}']=int(n)
        score=g.loc[mature&g.score_status.eq('ok')];result['score_available']=len(score)
        ordered_dates=pd.DatetimeIndex(g.loc[g.label_end.le(pd.Timestamp(cutoff)),'date'].unique()).sort_values()
        ics=pd.Series(np.nan,index=ordered_dates,dtype=float)
        for date,day in score.groupby('date'):
            if len(day)>=20 and day.fwd_return.nunique()>1:
                ic=0. if day.score.nunique()<2 else float(day.score.corr(day.fwd_return,method='spearman'))
                ics.loc[date]=ic;daily.append(dict(date=date,horizon=h,rank_ic=ic,observations=len(day)))
        ci=block_bootstrap(ics.to_numpy())
        result.update(rank_ic_mean=float(ics.mean()) if ics.notna().any() else None,ic_dates=int(ics.notna().sum()),ic_calendar_slots=len(ics),ic_ci_lower=ci[0],ic_ci_upper=ci[1])
        p=g.loc[mature&g.probability_status.eq('ok')];result['probability_available']=len(p)
        if len(p):
            y=p.fwd_return.gt(0).to_numpy(float);prob=p.probability_up.to_numpy(float);base=p.baseline_probability.to_numpy(float)
            brier=float(((prob-y)**2).mean());bb=float(((base-y)**2).mean());bins=[];ece=0.
            for i in range(10):
                mask=np.minimum((prob*10).astype(int),9)==i;n=int(mask.sum())
                mp=float(prob[mask].mean()) if n else None;obs=float(y[mask].mean()) if n else None
                bins.append(dict(lower=i/10,upper=(i+1)/10,count=n,mean_probability=mp,observed_up=obs))
                if n:ece+=abs(mp-obs)*n/len(p)
            result.update(brier=brier,baseline_brier=bb,brier_skill=1-brier/bb if bb else None,direction_accuracy=float(((prob>=.5)==y).mean()),actual_up_ratio=float(y.mean()),ece=ece,probability_bins=bins)
        q=g.loc[mature&g.interval_status.eq('ok')];result['interval_available']=len(q)
        covered=int(((q.fwd_return>=q.q10)&(q.fwd_return<=q.q90)).sum());loss=0.;baseline_loss=0.
        for alpha,col,bcol in [(.1,'q10','baseline_q10'),(.5,'q50','baseline_q50'),(.9,'q90','baseline_q90')]:
            err=q.fwd_return-q[col];be=q.fwd_return-q[bcol]
            loss+=float(np.maximum(alpha*err,(alpha-1)*err).sum())/3
            baseline_loss+=float(np.maximum(alpha*be,(alpha-1)*be).sum())/3
        result.update(interval_covered=covered,interval_coverage=ratio(covered,len(q)),pinball=ratio(loss,len(q)),baseline_pinball=ratio(baseline_loss,len(q)))
        e=g.loc[mature&g.expected_return_status.eq('ok')];result['expected_return_available']=len(e)
        result['mae']=float((e.fwd_return-e.expected_return).abs().mean()) if len(e) else None
        output['horizons'][str(h)]=result
    return output,pd.DataFrame(daily)
