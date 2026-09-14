import os
import argparse
import math
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .service import ModelNotPublished, SecurityNotFound, RequestError,parse_holdings_csv

class Holding(BaseModel):
    model_config=ConfigDict(extra='forbid')
    security_id: str
    quantity: float = Field(ge=0)
    average_cost: float | None = Field(default=None,ge=0)
    @field_validator('security_id')
    @classmethod
    def nonempty(cls,value):
        if not value.strip(): raise ValueError('security_id 不能为空')
        return value
    @field_validator('quantity','average_cost')
    @classmethod
    def finite(cls,value):
        if value is not None and not math.isfinite(value): raise ValueError('数值必须有限')
        return value

class RiskRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    max_position_weight: float | None = Field(default=None,gt=0)
    max_equity_weight: float | None = Field(default=None,gt=0,le=1)
    target_annual_volatility: float | None = Field(default=None,gt=0)
    fee_per_side: float | None = Field(default=None,ge=0,lt=1)
    max_participation: float | None = Field(default=None,gt=0,le=1)
    horizon: int | None = None

class AdviceRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    holdings: list[Holding] | None = None
    cash: float | None = Field(default=None,ge=0)
    risk_profile: RiskRequest | None = None
    as_of: str | None = None
    @field_validator('cash')
    @classmethod
    def cash_finite(cls,value):
        if value is not None and not math.isfinite(value): raise ValueError('现金必须有限')
        return value


class CSVAdviceRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    holdings_csv: str = Field(description='CSV持仓内容，表头为security_id,quantity，可包含average_cost')
    cash: float = Field(ge=0,allow_inf_nan=False)
    risk_profile: RiskRequest | None = None
    as_of: str | None = None

def create_app(service, api_key):
    if not api_key: raise ValueError('必须提供 API 密钥')
    app=FastAPI()
    def guard(key):
        if key != api_key: raise HTTPException(401,'API密钥无效')
    def call(fn):
        try:return fn()
        except ModelNotPublished as e: raise HTTPException(503,str(e))
        except SecurityNotFound as e: raise HTTPException(404,str(e))
        except (RequestError,ValueError) as e: raise HTTPException(422,str(e))
    @app.get('/health')
    def health(): return {'status':'ok'}
    @app.get('/v1/model/status')
    def status(x_api_key: str|None=Header(None)): guard(x_api_key); return call(service.status)
    @app.get('/v1/rankings')
    def rankings(horizon:int=20,limit:int|None=None,as_of:str|None=None,x_api_key:str|None=Header(None)):
        guard(x_api_key); return call(lambda:service.rank_market(horizon,limit,as_of))
    @app.get('/v1/stocks/{code}/forecast')
    def stock(code:str,as_of:str|None=None,x_api_key:str|None=Header(None)):
        guard(x_api_key); return call(lambda:service.forecast_stock(code,as_of))
    @app.post('/v1/portfolio/advice')
    def advice(body:AdviceRequest,x_api_key:str|None=Header(None)):
        guard(x_api_key)
        if body.holdings is None or body.cash is None: raise HTTPException(422,'必须提供持仓和现金')
        records=[item.model_dump() for item in body.holdings]
        if len({item['security_id'] for item in records}) != len(records): raise HTTPException(422,'持仓含重复证券')
        risk=body.risk_profile.model_dump(exclude_none=True) if body.risk_profile else None
        return call(lambda:service.advise(pd.DataFrame(records,columns=['security_id','quantity','average_cost']),body.cash,risk,body.as_of))
    @app.post('/v1/portfolio/advice/csv')
    def advice_csv(body:CSVAdviceRequest,x_api_key:str|None=Header(None)):
        guard(x_api_key)
        risk=body.risk_profile.model_dump(exclude_none=True) if body.risk_profile else None
        return call(lambda:service.advise(parse_holdings_csv(body.holdings_csv),body.cash,risk,body.as_of))
    return app

if __name__=='__main__':
    from .service import HKQuantService
    from .paths import DATA,MODELS
    from pathlib import Path
    import uvicorn
    key=os.environ.get('HK_QUANT_API_KEY')
    if not key: raise RuntimeError('必须设置 HK_QUANT_API_KEY')
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default=os.environ.get('HOST','127.0.0.1'))
    parser.add_argument('--port',type=int,default=int(os.environ.get('PORT','8000')))
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--models-root',type=Path,default=MODELS)
    args=parser.parse_args()
    uvicorn.run(create_app(HKQuantService(data=args.data_root,models=args.models_root),key),host=args.host,port=args.port)
