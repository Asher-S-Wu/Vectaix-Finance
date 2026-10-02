"""Independent, explicitly research-scoped A-share forecast and advice service."""
from io import BytesIO, StringIO
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# These contracts contain no market-specific execution or accounting assumptions.
from hk_quant.prediction_tasks import PREDICTION_SCHEMA, TASK_COLUMNS, TASK_STATUS_COLUMNS, validate_task_outputs
from hk_quant.registry import publication_artifact

from .portfolio import RiskProfile, SECURITY_PATTERN, advise_portfolio, validate_holdings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cn/universal'
MODELS = ROOT / 'models/cn/universal'
HORIZONS = (1, 5, 20, 60)
RETURN_BASIS = 'CNY source-adjusted price return'


class ModelNotPublished(Exception):
    pass


class SecurityNotFound(Exception):
    pass


class RequestError(ValueError):
    pass


def jsonable(value):
    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict('records')]
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value) else value


def parse_holdings_csv(text):
    try:
        frame = pd.read_csv(StringIO(text), dtype={'security_id':'string', 'acquired_date':'string'})
        return validate_holdings(frame)
    except (TypeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('无效持仓CSV: ' + str(exc)) from exc


def _day(value):
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tz is not None:
            raise ValueError()
        return result.normalize()
    except (ValueError, TypeError) as exc:
        raise RequestError('日期必须为有效的无时区日期') from exc


class AShareQuantService:
    def __init__(self, data=DATA, models=MODELS):
        self.data, self.models = Path(data), Path(models)

    def _active(self):
        try:
            active = json.loads((self.models/'active.json').read_text(encoding='utf-8'))
            required = {'schema_version','market','currency','model_version','data_as_of','forecast_path',
                        'metadata_path','acceptance_path','research_ready','eligible','artifact_sha256'}
            if not required <= set(active) or active['schema_version'] != 'ashare-v1' or active['market'] != 'CN' or active['currency'] != 'CNY' or active['research_ready'] is not True:
                raise ValueError('A股快照未就绪或市场标识无效')
            artifacts = {}
            for field in ('forecast_path','metadata_path','acceptance_path'):
                payload = publication_artifact(self.models,active[field]).read_bytes()
                if hashlib.sha256(payload).hexdigest() != active['artifact_sha256'].get(field):
                    raise ValueError('快照文件完整性检查失败: ' + field)
                artifacts[field] = payload
            certificate = json.loads(artifacts['acceptance_path'])
            metadata = json.loads(artifacts['metadata_path'])
            if certificate.get('research_ready') is not True or not isinstance(active['eligible'],bool) or certificate.get('eligible') is not active['eligible']:
                raise ValueError('研究就绪与绩效验收声明不一致')
            if any(certificate.get(k) != active[k] for k in ('model_version','data_as_of')):
                raise ValueError('验收凭证与快照版本不一致')
            if any(metadata.get(k) != active[k] for k in ('model_version','data_as_of')):
                raise ValueError('训练元数据与快照版本不一致')
            if (not isinstance(metadata.get('data_lineage'),dict) or not metadata['data_lineage']
                or not isinstance(metadata.get('features'),list) or not metadata['features']
                or metadata.get('model_trained_as_of') is None):
                raise ValueError('训练来源、特征或训练截止日期缺失')
            if metadata.get('market','CN') != 'CN' or metadata.get('currency','CNY') != 'CNY':
                raise ValueError('训练元数据市场或币种无效')
            if not isinstance(certificate.get('limitations'),list) or not isinstance(certificate.get('execution_validated'),bool):
                raise ValueError('研究限制或执行验证状态缺失')
            day = _day(active['data_as_of'])
            frame = pd.read_parquet(BytesIO(artifacts['forecast_path']))
            required_forecast = {'model_version','data_as_of','date','model_trained_as_of','security_id','horizon',
                'prediction_schema','status','reasons','explanations','return_basis', *TASK_STATUS_COLUMNS.values(),
                *(column for columns in TASK_COLUMNS.values() for column in columns)}
            if frame.empty or not required_forecast <= set(frame):
                raise ValueError('预测快照字段缺失')
            if not frame.model_version.eq(active['model_version']).all() or not frame.prediction_schema.eq(PREDICTION_SCHEMA).all():
                raise ValueError('预测格式或版本不一致')
            if not frame.return_basis.eq(RETURN_BASIS).all():
                raise ValueError('预测收益口径必须明确为CNY')
            if not frame.security_id.map(lambda s:isinstance(s,str) and bool(SECURITY_PATTERN.fullmatch(s))).all():
                raise ValueError('预测证券代码不是A股身份')
            if not frame.horizon.isin(HORIZONS).all() or frame.duplicated(['security_id','horizon']).any():
                raise ValueError('预测期限无效或重复')
            for field in ('date','data_as_of'):
                if not pd.to_datetime(frame[field],errors='raise').dt.normalize().eq(day).all():
                    raise ValueError('预测日期与快照日期不一致')
            trained = pd.to_datetime(frame.model_trained_as_of,errors='raise')
            if trained.isna().any() or not trained.dt.normalize().le(day).all():
                raise ValueError('训练截止日期缺失或晚于预测日')
            if metadata.get('model_trained_as_of') is not None:
                trained_day = _day(metadata['model_trained_as_of'])
                if not trained.dt.normalize().eq(trained_day).all():
                    raise ValueError('训练截止日期与元数据不一致')
            validate_task_outputs(frame)
            # Keep evidence attached to the exact in-memory snapshot to avoid mixed-version reads.
            active = {**active, '_certificate':certificate, '_metadata':metadata}
            return active, frame
        except Exception as exc:
            raise ModelNotPublished('没有可用且通过结构与来源检查的A股研究快照') from exc

    def _snapshot(self, as_of=None):
        requested = _day(as_of) if as_of is not None else None
        active, forecast = self._active()
        day = _day(active['data_as_of'])
        if requested is not None and requested != day:
            raise ModelNotPublished('未提供该历史日期的A股快照')
        return active, day, forecast

    def _disclosure(self, active, day):
        certificate = active['_certificate']
        return dict(model_version=active['model_version'], data_as_of=day, market='CN', currency='CNY',
            prediction_schema=PREDICTION_SCHEMA, return_basis=RETURN_BASIS, research_ready=True,
            eligible=active['eligible'], execution_validated=certificate['execution_validated'],
            publication_mode='research', limitations=certificate['limitations'],
            scope=active['_metadata'].get('scope', '证券主表中快照日仍上市的A股；实际训练与预测覆盖以各证券状态为准'),
            performance_acceptance=certificate.get('performance_acceptance',active['eligible']))

    def _securities(self, day):
        try:
            sec = pd.read_parquet(self.data/'securities.parquet').copy()
            required = {'security_id','exchange_code','name','list_date','delist_date','asset_type','currency','lot_size','identity_status'}
            if not required <= set(sec) or sec.security_id.duplicated().any():
                raise ValueError('证券身份表字段缺失或重复')
            if not sec.security_id.map(lambda s:isinstance(s,str) and bool(SECURITY_PATTERN.fullmatch(s))).all():
                raise ValueError('证券身份表包含非A股代码')
            listed = pd.to_datetime(sec.list_date,errors='coerce')
            delisted = pd.to_datetime(sec.delist_date,errors='coerce')
            # Unknown listing dates must not manufacture point-in-time universe membership.
            return sec.loc[listed.notna() & listed.le(day) & (delisted.isna() | delisted.gt(day)) &
                           sec.asset_type.eq('equity') & sec.currency.eq('CNY')].copy()
        except (ValueError, OSError, KeyError) as exc:
            raise ModelNotPublished('A股证券身份数据不可用') from exc

    def _forecast_availability(self, active, day, forecast, securities):
        requested = securities[['security_id']].merge(pd.DataFrame({'horizon':HORIZONS}),how='cross')
        rows = requested.merge(forecast,on=['security_id','horizon'],how='left',validate='one_to_one',indicator=True)
        absent = rows.pop('_merge').eq('left_only')
        rows.loc[absent,'status'] = 'not_scored'
        rows.loc[absent,list(TASK_STATUS_COLUMNS.values())] = 'not_scored'
        rows.loc[absent,'prediction_schema'] = PREDICTION_SCHEMA
        rows.loc[absent,'reasons'] = '该证券在此日期和期限没有研究预测'
        rows.loc[absent,'model_version'] = active['model_version']
        rows.loc[absent,'return_basis'] = RETURN_BASIS
        for field in ('date','data_as_of'):
            rows[field] = pd.to_datetime(rows[field])
            rows.loc[absent,field] = day
        return rows

    def _bars(self, day):
        try:
            bars = pd.read_parquet(self.data/'bars'/f'{day.year}.parquet')
            bars = bars.loc[pd.to_datetime(bars.date).dt.normalize().eq(day)].copy()
            features = pd.read_parquet(self.data/'latest_features.parquet')
            feature_fields = ['security_id',*[field for field in ('adv20_amount','volatility_20') if field in features]]
            features = features.loc[pd.to_datetime(features.date).dt.normalize().eq(day),feature_fields]
            return bars.drop(columns=[f for f in feature_fields if f != 'security_id' and f in bars]).merge(features,on='security_id',how='left',validate='one_to_one')
        except (ValueError, OSError, KeyError) as exc:
            raise ModelNotPublished('当前A股行情及流动性特征不可用') from exc

    def status(self):
        active, day, frame = self._snapshot()
        result = self._disclosure(active,day)
        result.update(forecast_security_count=int(frame.security_id.nunique()),
                      forecast_count=len(frame),model_trained_as_of=frame.model_trained_as_of.iloc[0])
        return jsonable(result)

    def rank_market(self, horizon, limit=None, as_of=None):
        if isinstance(horizon,bool) or horizon not in HORIZONS:
            raise RequestError('预测期限仅支持 1、5、20、60')
        if limit is not None and (not isinstance(limit,int) or isinstance(limit,bool) or limit <= 0):
            raise RequestError('limit 必须为正整数')
        active, day, forecast = self._snapshot(as_of)
        sec = self._securities(day)
        available = self._forecast_availability(active,day,forecast,sec)
        rows = sec.merge(available.loc[available.horizon.eq(horizon)],on='security_id',validate='one_to_one')
        rows = rows.sort_values(['score','security_id'],ascending=[False,True],na_position='last')
        if limit is not None:
            rows = rows.head(limit)
        return jsonable({**self._disclosure(active,day),'horizon':int(horizon),'items':rows})

    def forecast_stock(self, code, as_of=None):
        code = str(code).strip().upper()
        active, day, forecast = self._snapshot(as_of)
        sec = self._securities(day)
        exact = sec.security_id.eq(code)
        matches = sec.loc[exact if exact.any() else sec.exchange_code.astype(str).eq(code) |
                          sec.security_id.str.split('.').str[0].eq(code)]
        if matches.empty:
            raise SecurityNotFound('未找到A股证券')
        if len(matches) != 1:
            raise RequestError('证券代码对应多个交易所，请提供六位代码及交易所后缀')
        rows = self._forecast_availability(active,day,forecast,matches).sort_values('horizon')
        return jsonable({**self._disclosure(active,day),'security':matches.iloc[0],'forecasts':rows})

    def advise(self, holdings, cash, risk_profile=None, as_of=None):
        try:
            holdings = validate_holdings(holdings)
            cash = float(cash)
            if not np.isfinite(cash) or cash < 0:
                raise ValueError('现金必须为非负有限数')
            profile = RiskProfile(**(risk_profile or {}))
        except (TypeError, ValueError) as exc:
            raise RequestError(str(exc)) from exc
        active, day, forecast = self._snapshot(as_of)
        securities, bars = self._securities(day), self._bars(day)
        try:
            returns = pd.read_parquet(self.data/'risk_returns.parquet')
            returns['date'] = pd.to_datetime(returns.date)
            result = advise_portfolio(forecast,holdings,cash,bars,securities,returns.set_index('date'),profile)
        except (OSError, KeyError) as exc:
            raise ModelNotPublished('A股风险收益历史不可用') from exc
        except (ValueError, TypeError) as exc:
            raise RequestError(str(exc)) from exc
        return jsonable({**result,**self._disclosure(active,day)})


# Readable spelling for callers that do not capitalize the initialism.
AShareService = AShareQuantService
