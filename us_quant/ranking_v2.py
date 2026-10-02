"""Bounded v2 rank-target models with strictly earlier calibration heads."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from hk_quant.models import NUMERIC_OUTPUTS,_fit_log_residual_quantiles,_apply_log_residual_quantiles
from hk_quant.prediction_tasks import TASK_STATUS_COLUMNS,PREDICTION_SCHEMA,store_task_outputs
RETURN_BASIS='USD source-adjusted research-price arithmetic return; Yahoo plus explicitly reconstructed original identities'


def candidate_settings():
    path=Path(__file__).resolve().parents[1]/'docs/us/v2/proposed_protocol.json'
    return json.loads(path.read_text())['candidates']


def rank_targets(frame):
    y=frame.fwd_return.where(np.isfinite(frame.fwd_return)&frame.fwd_return.ge(-1)&frame.status.eq('ok')&frame.sigma_daily.gt(0))
    groups=y.groupby([frame.date,frame.horizon]);count=groups.transform('count')
    return (groups.rank(method='average')/count-(count+1)/(2*count)).where(count.ge(20))


class RankingModel:
    def __init__(self,kind,model_version=None,horizons=(20,)):
        settings=candidate_settings()
        if kind not in settings:raise ValueError('Unknown predeclared candidate')
        self.kind=kind;self.parameters=settings[kind];self.model_version=model_version or f'us-v2-{kind}'
        self.horizons=tuple(horizons);self.metadata={}

    def _valid(self,frame):
        f=frame[self.feature_columns].astype(float)
        valid=~np.isinf(f.to_numpy()).any(axis=1)
        if not self.kind.startswith('gbm_'):valid&=f.notna().all(axis=1).to_numpy()
        valid&=np.isfinite(frame.sigma_daily)&frame.sigma_daily.gt(0)&frame.horizon.isin(self.horizons)
        if 'status' in frame:valid&=frame.status.eq('ok')
        return np.asarray(valid,bool)

    def _scores(self,frame,h):
        x=frame[self.feature_columns].astype(float)
        if self.kind=='factor_reference':return (.5*(x.momentum_60-x.momentum_5-x.volatility_20+x.log_amount_20)).to_numpy()
        if self.kind=='momentum_12_1':return (2*x.momentum_252_skip20-1).to_numpy()
        if self.kind.startswith('ridge_'):return self.estimators[h].predict(self.scalers[h].transform(x))
        return self.estimators[h].predict(x)

    def fit(self,training,calibration,features,as_of):
        self.feature_columns=list(features)
        forbidden={'date','security_id','horizon','fwd_return','label_end','status'}
        if not self.feature_columns or len(set(self.feature_columns))!=len(self.feature_columns) or any(c in forbidden or c.startswith(('fwd_return_','label_end_')) for c in self.feature_columns):raise ValueError('Invalid or future-leaking feature list')
        if self.kind=='factor_reference':self.feature_columns=['momentum_60','momentum_5','volatility_20','log_amount_20']
        if self.kind=='momentum_12_1':self.feature_columns=['momentum_252_skip20']
        self.as_of=pd.Timestamp(as_of);tr=training.copy();ca=calibration.copy()
        if tr.empty or ca.empty:raise ValueError('Empty temporal split')
        for f in (tr,ca):
            if f.duplicated(['date','security_id','horizon']).any():raise ValueError('Duplicate training examples')
            if f[['date','label_end']].isna().any().any():raise ValueError('Missing temporal endpoints')
            if (f.label_end<f.date).any():raise ValueError('label_end precedes signal date')
        if (tr.label_end>=ca.date.min()).any() or (tr.date>=ca.date.min()).any():raise ValueError('Training/calibration purge violated')
        if (ca.label_end>self.as_of).any():raise ValueError('Calibration outcome exceeds as_of')
        tr['_rank_target']=rank_targets(tr)
        baseline_valid=np.isfinite(tr.fwd_return)&tr.fwd_return.ge(-1)&tr.status.eq('ok')&np.isfinite(tr.sigma_daily)&tr.sigma_daily.gt(0)
        common_frequency={h:float(tr.loc[baseline_valid&tr.horizon.eq(h),'fwd_return'].gt(0).mean()) for h in self.horizons}
        tr=tr.loc[self._valid(tr)&np.isfinite(tr.fwd_return)&tr.fwd_return.ge(-1)&np.isfinite(tr['_rank_target'])].copy()
        ca=ca.loc[self._valid(ca)&np.isfinite(ca.fwd_return)&ca.fwd_return.ge(-1)].copy()
        self.estimators={};self.scalers={};self.calibrators={};self.mean_affine={};self.quantiles={};self.interval_audit={};self.baseline_probability=common_frequency;counts={}
        for h in self.horizons:
            train=tr.loc[tr.horizon.eq(h)];cal=ca.loc[ca.horizon.eq(h)]
            if min(len(train),len(cal))<30 or cal.fwd_return.gt(0).nunique()<2:raise ValueError('Insufficient calibrated training evidence')
            weights=1/train.groupby('date').date.transform('size').to_numpy(float);weights*=len(weights)/weights.sum()
            x=train[self.feature_columns].astype(float);y=train['_rank_target']
            if self.kind.startswith('ridge_'):
                scaler=StandardScaler().fit(x,sample_weight=weights);self.scalers[h]=scaler
                self.estimators[h]=Ridge(alpha=self.parameters['alpha']).fit(scaler.transform(x),y,sample_weight=weights)
            elif self.kind.startswith('gbm_'):
                params={k:v for k,v in self.parameters.items() if k not in ('fit_window_years','estimator')}
                self.estimators[h]=lgb.LGBMRegressor(**params,verbosity=-1).fit(x,y,sample_weight=weights)
            score=self._scores(cal,h)
            cw=1/cal.groupby('date').date.transform('size').to_numpy(float);cw*=len(cw)/cw.sum()
            self.calibrators[h]=LogisticRegression(C=1.,random_state=42,max_iter=1000).fit(score[:,None],cal.fwd_return.gt(0),sample_weight=cw)
            mx=np.average(score,weights=cw);my=np.average(cal.fwd_return,weights=cw)
            variance=np.average((score-mx)**2,weights=cw)
            slope=max(0.,float(np.average((score-mx)*(cal.fwd_return.to_numpy()-my),weights=cw)/variance)) if variance>0 else 0.
            offset=float(my-slope*mx);self.mean_affine[h]=dict(offset=offset,slope=slope)
            point=offset+slope*score
            fitted=_fit_log_residual_quantiles(cal.fwd_return.to_numpy(float),point,cal.sigma_daily.to_numpy(float)*np.sqrt(h))
            self.quantiles[h]=fitted.pop('quantiles');self.interval_audit[h]=fitted;counts[str(h)]=dict(train=len(train),calibration=len(cal),train_start=str(train.date.min().date()),train_end=str(train.date.max().date()),train_label_end=str(train.label_end.max().date()),calibration_start=str(cal.date.min().date()),calibration_end=str(cal.date.max().date()),calibration_label_end=str(cal.label_end.max().date()))
        self.metadata=dict(kind=self.kind,market='US',currency='USD',model_version=self.model_version,as_of=self.as_of.isoformat(),horizons=list(self.horizons),features=self.feature_columns,parameters=self.parameters,
            ranking_target='within_date_horizon_centered_percentile',fitted_ranking_weights=self.kind not in ('factor_reference','momentum_12_1'),
            calibration=dict(probability='held-out per-horizon logistic',mean_return='held-out nonnegative-slope affine transform of rank score; score itself is never reversed',interval='empirical log1p-return residual quantiles scaled by historical daily sigma*sqrt(horizon)'),
            interval_fit_audit={str(k):v for k,v in self.interval_audit.items()},mean_affine={str(k):v for k,v in self.mean_affine.items()},residual_quantiles={str(k):v.tolist() for k,v in self.quantiles.items()},sample_counts=counts,
            return_basis=RETURN_BASIS,fit_executed_at=datetime.now(timezone.utc).isoformat(),eligible=False,execution_validated=False)
        return self

    def predict(self,examples,explain=False):
        f=examples.reset_index(drop=True).copy();out=f[['date','security_id','horizon','fwd_return','label_end']].copy()
        for column in NUMERIC_OUTPUTS:out[column]=np.nan
        for column in TASK_STATUS_COLUMNS.values():out[column]='insufficient_model_inputs'
        out['status']='insufficient_model_inputs';out['reasons']='';out['model_version']=self.model_version
        out['data_as_of']=pd.to_datetime(f.date).dt.strftime('%Y-%m-%d');out['model_trained_as_of']=self.as_of.isoformat();out['return_basis']=RETURN_BASIS;out['prediction_schema']=PREDICTION_SCHEMA
        legal=self._valid(f)&pd.to_datetime(f.date).ge(self.as_of).to_numpy()
        for h in self.horizons:
            ids=f.index[legal&f.horizon.eq(h)];part=f.loc[ids]
            if not len(part):continue
            score=self._scores(part,h);point=self.mean_affine[h]['offset']+self.mean_affine[h]['slope']*score
            q=_apply_log_residual_quantiles(point,part.sigma_daily.to_numpy(float)*np.sqrt(h),self.quantiles[h])
            values=pd.DataFrame(index=ids,columns=NUMERIC_OUTPUTS,dtype=float)
            values['score']=score;values['probability_up']=self.calibrators[h].predict_proba(score[:,None])[:,1];values['expected_return']=point
            values[['q10','q50','q90']]=q;values['baseline_probability']=self.baseline_probability[h]
            values[['baseline_q10','baseline_q50','baseline_q90']]=np.expm1(part.sigma_daily.to_numpy(float)[:,None]*np.sqrt(h)*np.array([-1.2815515655,0,1.2815515655])[None,:])
            store_task_outputs(out,values)
        return out
