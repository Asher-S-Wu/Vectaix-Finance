"""共享多期限的港元来源复权价格收益模型与独立时间段校准。"""
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from scipy.optimize import minimize, minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from .contracts import FORECAST_RETURN_BASIS, CASH_DIVIDEND_ACCOUNTING
from .prediction_tasks import TASK_STATUS_COLUMNS, PREDICTION_SCHEMA, store_task_outputs


HORIZONS=(1,5,20,60)
NUMERIC_OUTPUTS=('score','probability_up','expected_return','q10','q50','q90','baseline_probability','baseline_q10','baseline_q50','baseline_q90')
TARGET_SCHEMA='arithmetic-return-sqrt-horizon-v1'
PROBABILITY_BASELINE_BLEND={1:0.0,5:0.0,20:0.0,60:0.0}
PROBABILITY_ANCHOR_OFFSET={1:0.0,5:0.0,20:0.0,60:0.0}
INTERVAL_WIDTH_MULTIPLIER={1:1.0,5:1.0,20:1.0,60:1.0}
QUANTILE_LEVELS=np.array([.1,.5,.9],dtype=float)


def _fit_joint_quantile_log_affine(raw, actual):
    """在简单收益的log1p域用一个正尺度和一个位置同时校准三个分位。"""
    raw=np.asarray(raw,dtype=float);actual=np.asarray(actual,dtype=float)
    finite=np.isfinite(raw).all(axis=1) & np.isfinite(actual) & (actual>=-1)
    ordered=(raw[:,0]<=raw[:,1]) & (raw[:,1]<=raw[:,2])
    usable=finite & ordered
    if usable.sum()<30:
        raise ValueError('联合分位校准缺少至少30条有序样本')
    boundary_rows=(raw[usable]<-1).any(axis=1)
    # A calibration prediction below the arithmetic return domain is represented
    # by the domain boundary only while fitting the transform. The prediction
    # path still returns the original below-minus-one value as unavailable.
    with np.errstate(divide='ignore',invalid='ignore'):
        fit_raw=np.log1p(np.maximum(raw[usable],-1))
    fit_actual=actual[usable]
    finite_actual=fit_actual>-1
    initial_location=float(np.median(np.log1p(fit_actual[finite_actual])-
                                     fit_raw[finite_actual,1])) if finite_actual.any() else 0.
    def objective(parameters):
        location=float(parameters[0]);scale=float(np.exp(parameters[1]))
        if not np.isfinite(location) or not np.isfinite(scale) or scale<=0:return np.inf
        predicted=np.expm1(location+scale*fit_raw)
        error=fit_actual[:,None]-predicted
        return float(np.maximum(QUANTILE_LEVELS*error,(QUANTILE_LEVELS-1)*error).mean())
    result=minimize(objective,[initial_location,0.],method='Powell',bounds=[(-5.,5.),(-5.,5.)],
                    options={'xtol':1e-7,'ftol':1e-10,'maxiter':500})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('联合分位校准未收敛')
    location=float(result.x[0]);scale=float(np.exp(result.x[1]))
    return {'location':float(result.x[0]),'scale':scale,'fit_rows':int(usable.sum()),
            'excluded_unordered_or_nonfinite_rows':int((~usable).sum()),
            'boundary_represented_rows':int(boundary_rows.sum()),
            'pinball':float(result.fun),'objective_evaluations':int(result.nfev),
            'positive_scale':True,'domain':'log1p_arithmetic_return'}


def _apply_joint_quantile_log_affine(raw, affine):
    """应用正单调变换；原始无序或低于-100%的输出保留为失败依据。"""
    raw=np.asarray(raw,dtype=float)
    result=np.full_like(raw,np.nan)
    finite=np.isfinite(raw).all(axis=1)
    ordered=(raw[:,0]<=raw[:,1]) & (raw[:,1]<=raw[:,2])
    domain=finite & ordered & (raw>=-1).all(axis=1)
    if domain.any():
        transformed=np.log1p(raw[domain])
        result[domain]=np.expm1(float(affine['location'])+float(affine['scale'])*transformed)
    result[finite & ~domain]=raw[finite & ~domain]
    return result


def _fit_log_residual_quantiles(actual, point, scale):
    """用共同的标准化log1p收益误差分布构造线性模型区间。"""
    actual=np.asarray(actual,dtype=float);point=np.asarray(point,dtype=float);scale=np.asarray(scale,dtype=float)
    valid=np.isfinite(actual) & (actual>=-1) & np.isfinite(point) & (point>-1) & np.isfinite(scale) & (scale>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        residual=(np.log1p(actual[valid])-np.log1p(point[valid]))/scale[valid]
    finite_residual=np.isfinite(residual) | np.isneginf(residual)
    if len(residual)<30 or not finite_residual.all():
        raise ValueError('线性区间经验分布缺少至少30条有效log1p收益误差')
    ordered=np.sort(residual)
    positions=[max(0,int(np.ceil(q*len(ordered)))-1) for q in QUANTILE_LEVELS]
    return {'quantiles':ordered[positions],'fit_rows':int(valid.sum()),
            'excluded_invalid_point_or_scale_rows':int((~valid).sum()),
            'negative_infinity_atoms':int(np.isneginf(ordered).sum()),
            'domain':'log1p_arithmetic_return'}


def _apply_log_residual_quantiles(point, scale, residual_quantiles):
    point=np.asarray(point,dtype=float);scale=np.asarray(scale,dtype=float)
    result=np.full((len(point),3),np.nan)
    valid=np.isfinite(point) & (point>-1) & np.isfinite(scale) & (scale>0)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        result[valid]=np.expm1(np.log1p(point[valid,None])+scale[valid,None]*residual_quantiles)
    return result


class UniversalModel:
    def __init__(self,kind='lightgbm_small',model_version='universal-v1'):
        if kind not in ('factor','linear','lightgbm_small','lightgbm_large'):
            raise ValueError('未知模型类型: '+str(kind))
        self.kind=kind
        self.model_version=model_version
        self.metadata={}

    def _matrix(self,frame):
        x=frame[self.feature_columns].astype(float).copy()
        x['log_horizon']=np.log(frame.horizon.astype(float))
        return x

    def _legal(self,frame,labels=False):
        features=frame[self.feature_columns].apply(pd.to_numeric,errors='coerce')
        # LightGBM uses its native NaN branch. Infinity is never a valid input.
        valid=~np.isinf(features.to_numpy()).any(axis=1)
        valid &= ~(features.isna() & frame[self.feature_columns].notna()).any(axis=1).to_numpy()
        if self.kind in ('factor','linear'):valid &= features.notna().all(axis=1).to_numpy()
        sigma=pd.to_numeric(frame.sigma_daily,errors='coerce')
        valid &= sigma.notna().to_numpy() & np.isfinite(sigma).to_numpy() & (sigma>0).to_numpy()
        valid &= frame.horizon.isin(HORIZONS).to_numpy()
        if 'status' in frame:valid &= frame.status.eq('ok').to_numpy()
        if labels:
            y=pd.to_numeric(frame.fwd_return,errors='coerce')
            valid &= np.isfinite(y).to_numpy() & (y>=-1).to_numpy()
        return valid

    def fit(self,train,calibration,feature_columns,as_of):
        self.feature_columns=list(feature_columns)
        if not self.feature_columns or len(set(self.feature_columns))!=len(self.feature_columns):
            raise ValueError('特征列表为空或重复')
        forbidden={'date','security_id','horizon','fwd_return','label_end','log_horizon','status'}
        if forbidden.intersection(self.feature_columns):
            raise ValueError('特征列表包含标签、标识或内部期限列')
        self.as_of=pd.Timestamp(as_of)
        if pd.isna(self.as_of):raise ValueError('as_of无效')
        required={'date','security_id','horizon','fwd_return','label_end','sigma_daily',*self.feature_columns}
        for name,frame in [('train',train),('calibration',calibration)]:
            if required.difference(frame.columns):raise ValueError(name+'缺少输入列: '+','.join(sorted(required.difference(frame.columns))))
            if frame.empty:raise ValueError(name+'为空')
        tr=train.copy();ca=calibration.copy()
        for frame in (tr,ca):
            frame['date']=pd.to_datetime(frame.date,errors='coerce');frame['label_end']=pd.to_datetime(frame.label_end,errors='coerce')
            if frame[['date','label_end']].isna().any().any():raise ValueError('date或label_end缺失')
            if (frame.label_end<frame.date).any():raise ValueError('label_end早于样本日期')
            if frame.duplicated(['date','security_id','horizon']).any():raise ValueError('重复样本日期、证券及期限')
        boundary=ca.date.min()
        if (tr.date>=boundary).any():raise ValueError('train.date必须早于calibration.date')
        if (tr.label_end>=boundary).any():raise ValueError('train.label_end必须早于calibration.date')
        if (ca.label_end>self.as_of).any():raise ValueError('calibration.label_end超过as_of')
        # 概率基准用共同历史标签，不能随候选模型的特征缺失处理改变。
        baseline_return=pd.to_numeric(tr.fwd_return,errors='coerce')
        baseline_sigma=pd.to_numeric(tr.sigma_daily,errors='coerce')
        baseline_valid=(np.isfinite(baseline_return) & baseline_return.ge(-1)
                        & np.isfinite(baseline_sigma) & baseline_sigma.gt(0) & tr.horizon.isin(HORIZONS))
        if 'status' in tr:baseline_valid &= tr.status.eq('ok')
        common_baseline={h:float(baseline_return[baseline_valid & tr.horizon.eq(h)].gt(0).mean()) for h in HORIZONS}
        used={};counts={}
        for name,frame in [('train',tr),('calibration',ca)]:
            legal=frame.loc[self._legal(frame,labels=True)].copy()
            counts[name]={'input':len(frame),'used':len(legal),'excluded':len(frame)-len(legal),'horizons':{}}
            for h in HORIZONS:
                subset=legal.loc[legal.horizon==h]
                if len(subset)<30 or (subset.fwd_return>0).nunique()!=2:
                    raise ValueError(f'{name}期限{h}至少需要30条合法样本且包含正负两类方向')
                counts[name]['horizons'][str(h)]={'samples':len(subset),'positive':int((subset.fwd_return>0).sum()),
                                                'negative_or_zero':int((subset.fwd_return<=0).sum()),
                                                'full_loss':int(subset.fwd_return.eq(-1).sum())}
            used[name]=legal
        tr=used['train'];ca=used['calibration']
        x=self._matrix(tr);xc=self._matrix(ca)
        # 真实收益仅按期限归一；历史sigma接近零不应放大训练标签。
        y=tr.fwd_return.to_numpy(float)/np.sqrt(tr.horizon.to_numpy(float))
        weights=1/tr.groupby(['date','horizon']).horizon.transform('size').to_numpy(float)
        weights*=len(weights)/weights.sum()
        self.ranker=None;self.classifier=None;self.scaler=None;self.quantile_models={}
        if self.kind=='factor':
            required_factor=['momentum_60','momentum_5','volatility_20','log_amount_20']
            if self.feature_columns!=required_factor:raise ValueError('简单因子基准输入必须为固定四项因子')
            self.parameters={'weights':dict(zip(required_factor,[.25,-.25,-.25,.25])),
                             'definition':'Equal-weight trend, short reversal, low volatility and liquidity ranks; no fitted weights'}
        elif self.kind=='linear':
            self.parameters={'alpha':10.0,'scaler':'StandardScaler'}
            self.scaler=StandardScaler().fit(x,sample_weight=weights)
            self.regressor=Ridge(alpha=10.).fit(self.scaler.transform(x),y,sample_weight=weights)
        else:
            large=self.kind=='lightgbm_large'
            self.parameters=dict(num_leaves=31 if large else 15,n_estimators=240 if large else 160,min_child_samples=100,reg_lambda=5.,learning_rate=.04 if large else .05,n_jobs=8,random_state=42,verbosity=-1)
            self.regressor=lgb.LGBMRegressor(**self.parameters).fit(x,y,sample_weight=weights)
            for alpha in (.1,.5,.9):
                self.quantile_models[alpha]=lgb.LGBMRegressor(**self.parameters,objective='quantile',alpha=alpha).fit(x,y,sample_weight=weights)
            self.classifier=lgb.LGBMClassifier(**self.parameters).fit(x,(tr.fwd_return>0).astype(int),sample_weight=weights)
            ranking=tr.loc[tr.horizon==20].sort_values('date',kind='stable')
            relevance=(ranking.groupby('date').fwd_return.rank(method='average',pct=True)*10).apply(np.floor).clip(upper=9).astype(int)
            self.ranker=lgb.LGBMRanker(**self.parameters,objective='lambdarank',lambdarank_truncation_level=30)
            self.ranker.fit(self._matrix(ranking),relevance,group=ranking.groupby('date',sort=False).size().tolist())
        z=self._regression(xc)
        margins=self._margin(xc,z)
        cal_scale=self._return_scale(ca)
        predicted_return=z*cal_scale
        return_residual=ca.fwd_return.to_numpy(float)-predicted_return
        self.calibrators={};self.residual_quantiles={};self.quantile_affine={};self.interval_log_residual_quantiles={};self.interval_log_residual_audit={};self.mean_return_residual={};self.baseline_probability={}
        self.probability_baseline_blend=PROBABILITY_BASELINE_BLEND.copy()
        self.probability_anchor_offsets=PROBABILITY_ANCHOR_OFFSET.copy()
        self.interval_width_multiplier=INTERVAL_WIDTH_MULTIPLIER.copy()
        if self.quantile_models:
            calibration_return_quantiles=np.column_stack([self.quantile_models[q].predict(xc) for q in (.1,.5,.9)])*cal_scale[:,None]
        for h in HORIZONS:
            mask=ca.horizon.to_numpy()==h
            calibration_margins=self._center_margins(ca,margins) if self.kind=='factor' else margins
            self.calibrators[h]=LogisticRegression(random_state=42,max_iter=1000).fit(calibration_margins[mask,None],(ca.loc[mask,'fwd_return']>0).astype(int))
            self.mean_return_residual[h]=float(np.mean(return_residual[mask]))
            if self.quantile_models:
                self.quantile_affine[h]=_fit_joint_quantile_log_affine(
                    calibration_return_quantiles[mask], ca.loc[mask,'fwd_return'].to_numpy(float))
            elif self.kind=='linear':
                fitted=_fit_log_residual_quantiles(
                    ca.loc[mask,'fwd_return'].to_numpy(float),
                    predicted_return[mask]+self.mean_return_residual[h],
                    ca.loc[mask,'sigma_daily'].to_numpy(float)*np.sqrt(h))
                self.interval_log_residual_quantiles[h]=fitted.pop('quantiles')
                self.interval_log_residual_audit[h]=fitted
            else:
                self.residual_quantiles[h]=np.quantile(return_residual[mask],[.1,.5,.9])
            self.baseline_probability[h]=common_baseline[h]
        self.target_schema=TARGET_SCHEMA
        self.metadata=dict(
            kind=self.kind,model_version=self.model_version,features=self.feature_columns+['log_horizon'],
            target_schema=self.target_schema,
            prediction_schema=PREDICTION_SCHEMA,
            horizons=list(HORIZONS),as_of=self.as_of.isoformat(),sample_counts=counts,parameters=self.parameters,
            return_basis=FORECAST_RETURN_BASIS,cash_dividend_accounting=CASH_DIVIDEND_ACCOUNTING,sigma_basis='historical unranked HKD source-adjusted daily log-price-return standard deviation',
            training_target='fixed factor score; no fitted return weights' if self.kind=='factor' else 'fwd_return / sqrt(horizon); arithmetic returns include -1 without sigma division or return clipping',
            prediction_arithmetic_return='factor_score * sigma_daily * sqrt(horizon)' if self.kind=='factor' else 'regressor_prediction * sqrt(horizon)',
            interval_model='three LightGBM quantile regressors with alpha 0.1, 0.5, 0.9, jointly calibrated in log1p return space' if self.quantile_models else 'held-out arithmetic-return residual quantiles around the point arithmetic-return prediction',
            baseline={'probability':'common legal historical training label frequency, independent of candidate feature missingness',
                      'quantiles':'exp(normal_quantile * historical sigma_daily * sqrt(horizon)) - 1'},
            calibration={
                'probability':'per-horizon Platt logistic calibration on held-out raw margin (factor/linear: regression score)',
                'probability_anchor':'held_out_logistic_without_training_rate_shift',
                'probability_baseline_blend':{str(h):self.probability_baseline_blend[h] for h in HORIZONS},
                'probability_centering':'date_horizon_cross_section' if self.kind=='factor' else 'none',
                'probability_anchor_offset':{str(h):self.probability_anchor_offsets[h] for h in HORIZONS},
                'intervals':('per-horizon joint positive-scale log1p calibration; raw crossing rows remain unavailable, never sorted'
                             if self.quantile_models else 'per-horizon empirical log1p-return residual distribution'),
                'expected_return':'predicted_arithmetic_return + mean(held_out_arithmetic_return_residual) per horizon',
                'residual_assumption':'Within each horizon, held-out arithmetic-return residuals represent prediction errors; the correction is unconditional within horizon.',
                'residual_quantiles':{str(h):v.tolist() for h,v in self.residual_quantiles.items()},
                'interval_log_residual_quantiles':{str(h):v.tolist() for h,v in self.interval_log_residual_quantiles.items()},
                'interval_log_residual_audit':{str(h):v for h,v in self.interval_log_residual_audit.items()},
                'quantile_affine':{str(h):v for h,v in self.quantile_affine.items()},
                'mean_return_residual':{str(h):self.mean_return_residual[h] for h in HORIZONS},
                'interval_width_multiplier':{str(h):self.interval_width_multiplier[h] for h in HORIZONS}},
            train_start=tr.date.min().isoformat(),train_end=tr.date.max().isoformat(),
            calibration_start=ca.date.min().isoformat(),calibration_end=ca.date.max().isoformat(),
            latest_label_end=max(tr.label_end.max(),ca.label_end.max()).isoformat(),
            versions={'numpy':np.__version__,'sklearn':sklearn.__version__,'lightgbm':lgb.__version__})
        return self

    def _return_scale(self,frame):
        scale=np.sqrt(frame.horizon.to_numpy(float))
        if self.kind=='factor':scale=scale*frame.sigma_daily.to_numpy(float)
        return scale

    def _regression(self,x):
        if self.kind=='factor':return ((2*x[self.feature_columns].to_numpy()-1)*np.array([.25,-.25,-.25,.25])).sum(axis=1)
        if self.kind=='linear':return self.regressor.predict(self.scaler.transform(x))
        return self.regressor.predict(x)

    def _margin(self,x,z):
        if self.kind in ('factor','linear'):return z
        return self.classifier.predict(x,raw_score=True)

    def _center_margins(self, frame, margins):
        """按信号日及期限中心化横截面分数，不使用未来收益。"""
        values = pd.DataFrame({
            'date': pd.to_datetime(frame.date).to_numpy(),
            'horizon': frame.horizon.to_numpy(),
            'margin': np.asarray(margins, dtype=float),
        })
        return (values['margin'] - values.groupby(['date', 'horizon'])['margin'].transform('mean')).to_numpy(float)

    def predict(self,examples,explain=False):
        if not self.metadata:raise ValueError('模型尚未训练')
        if getattr(self,'target_schema',None)!=TARGET_SCHEMA:
            raise ValueError('模型收益目标版本不匹配，必须使用当前收益目标重新训练')
        out=examples.copy().reset_index(drop=True)
        for col in NUMERIC_OUTPUTS:out[col]=np.nan
        out['status']='insufficient_model_inputs';out['reasons']='';out['model_version']=self.model_version
        for field in TASK_STATUS_COLUMNS.values():out[field]='insufficient_model_inputs'
        out['prediction_schema']=PREDICTION_SCHEMA
        out['data_as_of']=pd.to_datetime(out['date'],errors='coerce').dt.strftime('%Y-%m-%d') if 'date' in out else None
        out['model_trained_as_of']=self.as_of.isoformat();out['return_basis']=FORECAST_RETURN_BASIS
        if explain:
            out['explanations']=[[] for _ in range(len(out))];out['explanation_basis']=None
        required={'date','security_id','horizon','sigma_daily',*self.feature_columns}
        missing=required.difference(examples.columns)
        if missing:
            out['reasons']='缺少必要输入列: '+','.join(sorted(missing))
            return out
        frame=examples.copy().reset_index(drop=True)
        data_valid=self._legal(frame)
        valid=data_valid.copy()
        dates=pd.to_datetime(frame.date,errors='coerce')
        valid &= dates.notna().to_numpy() & (dates>=self.as_of).to_numpy()
        def append_reason(mask,message):
            existing=out.loc[mask,'reasons']
            separator=np.where(existing.ne(''),'；','')
            out.loc[mask,'reasons']=existing+separator+message
        append_reason(dates.isna(),'预测日期缺失')
        append_reason(dates.notna() & (dates<self.as_of),'预测日期早于模型训练截止，须使用当时模型')
        if 'status' in frame:
            blocked=frame.status.ne('ok')
            append_reason(blocked,'上游状态不可预测: '+frame.loc[blocked,'status'].astype(str))
        append_reason(~data_valid,'期限、历史波动率或数值特征不可用')
        indices=np.flatnonzero(valid)
        if not len(indices):return out
        selected=frame.iloc[indices];x=self._matrix(selected)
        z=self._regression(x);margins=self._margin(x,z)
        probability_margins=self._center_margins(selected,margins) if self.kind=='factor' else margins
        scale=self._return_scale(selected)
        predicted_return=z*scale
        probabilities=np.empty(len(indices))
        for h in HORIZONS:
            mask=selected.horizon.to_numpy()==h
            if mask.any():
                calibrated=self.calibrators[h].predict_proba(probability_margins[mask,None])[:,1]
                calibrated=1.0/(1.0+np.exp(-np.log(calibrated/(1.0-calibrated))))
                calibrated=calibrated+self.probability_anchor_offsets[h]
                blend=self.probability_baseline_blend[h]
                probabilities[mask]=(1.0-blend)*calibrated+blend*self.baseline_probability[h]
        ranking=np.array(z,copy=True)
        twenty=selected.horizon.to_numpy()==20
        if self.ranker is not None and twenty.any():ranking[twenty]=self.ranker.predict(x.loc[twenty])
        if self.quantile_models:
            return_quantiles=np.column_stack([self.quantile_models[q].predict(x) for q in (.1,.5,.9)])*scale[:,None]
        values=np.empty((len(indices),len(NUMERIC_OUTPUTS)),dtype=float)
        values[:,0]=ranking;values[:,1]=probabilities
        for h in HORIZONS:
            mask=selected.horizon.to_numpy()==h
            if not mask.any():continue
            values[mask,2]=predicted_return[mask]+self.mean_return_residual[h]
            if self.quantile_models:
                values[mask,3:6]=_apply_joint_quantile_log_affine(return_quantiles[mask],self.quantile_affine[h])
            elif self.kind=='linear':
                values[mask,3:6]=_apply_log_residual_quantiles(
                    predicted_return[mask]+self.mean_return_residual[h],
                    selected.loc[mask,'sigma_daily'].to_numpy(float)*np.sqrt(h),
                    self.interval_log_residual_quantiles[h])
            else:
                values[mask,3:6]=predicted_return[mask,None]+self.residual_quantiles[h]
            width=self.interval_width_multiplier[h]
            if width != 1.0:
                median=values[mask,4].copy()
                values[mask,3]=median+width*(values[mask,3]-median)
                values[mask,5]=median+width*(values[mask,5]-median)
            values[mask,6]=self.baseline_probability[h]
        baseline_scale=selected.sigma_daily.to_numpy(float)*np.sqrt(selected.horizon.to_numpy(float))
        values[:,7:10]=np.expm1(np.array([-1.2815515655,0,1.2815515655])[None,:]*baseline_scale[:,None])
        store_task_outputs(out,pd.DataFrame(values,index=indices,columns=NUMERIC_OUTPUTS))
        if explain:
            if self.kind=='factor':
                contrib=np.column_stack(((2*x[self.feature_columns].to_numpy()-1)*np.array([.25,-.25,-.25,.25])*scale[:,None],np.zeros(len(x))))
            elif self.kind=='linear':contrib=self.scaler.transform(x)*self.regressor.coef_[None,:]*scale[:,None]
            else:contrib=self.regressor.predict(x,pred_contrib=True)[:,:-1]*scale[:,None]
            if self.ranker is not None and twenty.any():contrib[twenty]=self.ranker.predict(x.loc[twenty],pred_contrib=True)[:,:-1]
            names=self.feature_columns+['log_horizon']
            for j,i in enumerate(indices):
                if out.at[i,'score_status']!='ok' and out.at[i,'expected_return_status']!='ok':continue
                basis='rank_score' if self.ranker is not None and int(selected.iloc[j].horizon)==20 else 'predicted_arithmetic_return'
                out.at[i,'explanation_basis']=basis
                order=np.argsort(-np.abs(contrib[j]))[:5]
                out.at[i,'explanations']=[{'feature':names[k],'contribution':float(contrib[j,k]),'feature_value':float(x.iloc[j,k]),'basis':basis} for k in order]
        return out
