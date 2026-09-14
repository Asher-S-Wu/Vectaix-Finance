"""只读取已正式发布快照的港股量化服务。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .paths import DATA, MODELS
from .contracts import RiskProfile
from .portfolio import advise_portfolio
from .registry import publication_artifact
from . import HORIZONS
from .prediction_tasks import PREDICTION_SCHEMA, TASK_COLUMNS, TASK_STATUS_COLUMNS, validate_task_outputs


class ModelNotPublished(Exception): pass
class SecurityNotFound(Exception): pass
class RequestError(Exception): pass


def parse_holdings_csv(text):
    from io import StringIO
    frame = pd.read_csv(StringIO(text),dtype={'security_id':'string'})
    if set(frame.columns) - {'security_id', 'quantity', 'average_cost'} or not {'security_id','quantity'} <= set(frame):
        raise ValueError('持仓CSV只接受 security_id、quantity、average_cost（可选）')
    frame['quantity']=pd.to_numeric(frame.quantity,errors='raise')
    if 'average_cost' in frame:frame['average_cost']=pd.to_numeric(frame.average_cost,errors='raise')
    if frame.security_id.isna().any() or frame.security_id.str.strip().eq('').any() or frame.security_id.duplicated().any() or not np.isfinite(frame.quantity).all() or (frame.quantity < 0).any():
        raise ValueError('持仓CSV含重复证券、空值或无效数量')
    if 'average_cost' in frame and (not np.isfinite(frame.average_cost.dropna()).all() or (frame.average_cost.dropna()<0).any()):
        raise ValueError('average_cost无效')
    return frame


def jsonable(value):
    if isinstance(value, pd.DataFrame): return [jsonable(row) for row in value.to_dict('records')]
    if isinstance(value, pd.Series): return jsonable(value.to_dict())
    if isinstance(value, dict): return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp,np.datetime64)): return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic): value=value.item()
    return None if value is None or (isinstance(value,float) and not np.isfinite(value)) or pd.isna(value) else value


class HKQuantService:
    def __init__(self, data=DATA, models=MODELS): self.data, self.models = Path(data), Path(models)
    def _active(self):
        try:
            active=json.loads((self.models/'active.json').read_text(encoding='utf-8'))
            acceptance=json.loads(publication_artifact(self.models,active['acceptance_path']).read_text(encoding='utf-8'))
            required={'model_version','data_as_of','forecast_path','eligible','acceptance_path'}
            if (not required <= set(active) or active['eligible'] is not True or acceptance.get('eligible') is not True
                or any(acceptance.get(key)!=active[key] for key in ('model_version','data_as_of'))): raise ValueError()
            forecast=publication_artifact(self.models,active['forecast_path'])
            if not forecast.is_file(): raise ValueError()
            frame=pd.read_parquet(forecast)
            required_forecast={'model_version','data_as_of','date','security_id','horizon',
                              'prediction_schema','status','reasons','explanations','return_basis',
                              *TASK_STATUS_COLUMNS.values(),
                              *(column for columns in TASK_COLUMNS.values() for column in columns)}
            if frame.empty or not required_forecast <= set(frame) or not frame.model_version.eq(active['model_version']).all() or not pd.to_datetime(frame.data_as_of).dt.normalize().eq(pd.Timestamp(active['data_as_of']).normalize()).all() or not pd.to_datetime(frame.date).dt.normalize().eq(pd.Timestamp(active['data_as_of']).normalize()).all(): raise ValueError()
            if (not frame.prediction_schema.eq(PREDICTION_SCHEMA).all()
                or frame.security_id.isna().any() or frame.security_id.astype(str).str.strip().eq('').any()
                or not frame.horizon.isin(HORIZONS).all() or frame.duplicated(['security_id','horizon']).any()): raise ValueError()
            validate_task_outputs(frame)
            return active, frame
        except Exception as exc: raise ModelNotPublished('没有可用的正式发布模型') from exc
    def _snapshot(self, as_of=None):
        active, forecast=self._active(); day=pd.Timestamp(active['data_as_of']).normalize()
        if as_of is not None and pd.Timestamp(as_of).normalize()!=day: raise ModelNotPublished('未提供该历史日期的正式快照')
        return active, day, forecast
    def _forecast_availability(self, active, day, forecast, securities):
        requested = securities[['security_id']].merge(pd.DataFrame({'horizon': HORIZONS}), how='cross')
        rows = requested.merge(forecast, on=['security_id', 'horizon'], how='left',
                               validate='one_to_one', indicator=True)
        absent = rows.pop('_merge').eq('left_only')
        # 缺少快照的证券仍显示查询范围；不生成预测数值或模型解释。
        rows.loc[absent, 'status'] = 'not_scored'
        rows.loc[absent, list(TASK_STATUS_COLUMNS.values())] = 'not_scored'
        rows.loc[absent, 'prediction_schema'] = PREDICTION_SCHEMA
        rows.loc[absent, 'reasons'] = '该证券在此日期和期限没有已发布预测'
        rows.loc[absent, 'model_version'] = active['model_version']
        for field in ('date', 'data_as_of'):
            rows[field] = pd.to_datetime(rows[field])
            rows.loc[absent, field] = day
        return rows
    def _securities(self, day):
        sec=pd.read_parquet(self.data/'securities.parquet').copy()
        listed=pd.to_datetime(sec.get('list_date'),errors='coerce').le(day) | pd.to_datetime(sec.get('list_date'),errors='coerce').isna()
        alive=pd.to_datetime(sec.get('delist_date'),errors='coerce').gt(day) | pd.to_datetime(sec.get('delist_date'),errors='coerce').isna()
        return sec.loc[listed & alive & ~sec.asset_type.astype(str).str.lower().isin({'etf','warrant'})]
    def _bars(self, day):
        bars=pd.read_parquet(self.data/'bars'/f'{day.year}.parquet'); bars['date']=pd.to_datetime(bars.date)
        bars=bars.loc[bars.date.dt.normalize()==day].copy()
        features=pd.read_parquet(self.data/'latest_features.parquet'); features['date']=pd.to_datetime(features.date)
        features=features.loc[features.date.dt.normalize()==day, ['security_id','adv20_amount']]
        return bars.merge(features,on='security_id',how='left',validate='one_to_one')
    def status(self):
        active, day, _=self._snapshot(); return jsonable({'model_version':active['model_version'],'data_as_of':day,'eligible':True,'prediction_schema':PREDICTION_SCHEMA})
    def rank_market(self,horizon,limit=None,as_of=None):
        if int(horizon) not in (1,5,20,60): raise RequestError('预测期限仅支持 1、5、20、60')
        if limit is not None and (not isinstance(limit,int) or isinstance(limit,bool) or limit<=0): raise RequestError('limit 必须为正整数')
        active,day,forecast=self._snapshot(as_of); sec=self._securities(day)
        forecast=self._forecast_availability(active,day,forecast,sec)
        rows=sec.merge(forecast.loc[forecast.horizon==int(horizon)],on='security_id',how='left',suffixes=('','_forecast'))
        rows['score']=pd.to_numeric(rows['score'],errors='raise')
        rows=rows.sort_values(['score','security_id'],ascending=[False,True],na_position='last')
        if limit is not None: rows=rows.head(int(limit))
        return jsonable({'model_version':active['model_version'],'data_as_of':day,'prediction_schema':PREDICTION_SCHEMA,'horizon':int(horizon),'items':rows})
    def forecast_stock(self,code,as_of=None):
        active,day,forecast=self._snapshot(as_of); sec=self._securities(day)
        exact=sec.security_id.astype(str).eq(str(code))
        matches=sec.loc[exact if exact.any() else sec.exchange_code.astype(str).eq(str(code))]
        if len(matches)==0: raise SecurityNotFound('未找到证券')
        if len(matches)>1: raise RequestError('证券代码对应多个实体，无法确定')
        row=matches.iloc[0]
        items=self._forecast_availability(active,day,forecast,matches).sort_values('horizon')
        return jsonable({'model_version':active['model_version'],'data_as_of':day,'prediction_schema':PREDICTION_SCHEMA,'security':row,'forecasts':items})
    def advise(self,holdings,cash,risk_profile=None,as_of=None):
        if holdings is None or cash is None: raise RequestError('必须提供持仓和现金')
        if not {'security_id','quantity'} <= set(holdings): raise RequestError('持仓必须含 security_id 和 quantity')
        if not holdings.empty and (holdings.security_id.isna().any() or holdings.security_id.astype(str).str.strip().eq('').any() or holdings.security_id.duplicated().any() or not np.isfinite(holdings.quantity).all() or (holdings.quantity<0).any()): raise RequestError('持仓含重复证券、空值或无效数量')
        if not holdings.empty and 'average_cost' in holdings and (not np.isfinite(holdings.average_cost.dropna()).all() or (holdings.average_cost.dropna()<0).any()): raise RequestError('average_cost 无效')
        allowed=set(RiskProfile.__dataclass_fields__)
        if risk_profile is not None and set(risk_profile)-allowed: raise RequestError('risk_profile 含未知字段')
        active,day,forecast=self._snapshot(as_of); sec=self._securities(day); bars=self._bars(day)
        returns=pd.read_parquet(self.data/'risk_returns.parquet'); returns['date']=pd.to_datetime(returns.date); returns=returns.set_index('date')
        try: profile=RiskProfile(**(risk_profile or {}))
        except (TypeError,ValueError) as exc: raise RequestError(str(exc)) from exc
        try: result=advise_portfolio(forecast,holdings,float(cash),bars,sec,returns,profile)
        except (TypeError,ValueError) as exc: raise RequestError(str(exc)) from exc
        result.update(model_version=active['model_version'],data_as_of=day,return_basis=forecast.get('return_basis',pd.Series([None])).dropna().iloc[0] if forecast.get('return_basis',pd.Series()).notna().any() else None)
        return jsonable(result)
