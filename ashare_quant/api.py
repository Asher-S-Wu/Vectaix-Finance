"""Authenticated research API for the independent CNY A-share snapshot."""
import argparse
from datetime import date
import hmac
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .service import ModelNotPublished, RequestError, SecurityNotFound, parse_holdings_csv
from .portfolio import SECURITY_PATTERN


class Holding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    security_id: str
    quantity: float = Field(ge=0,allow_inf_nan=False)
    average_cost: float | None = Field(default=None,ge=0,allow_inf_nan=False)
    acquired_date: date | None = None
    sellable_quantity: float | None = Field(default=None,ge=0,allow_inf_nan=False)

    @field_validator('security_id')
    @classmethod
    def security(cls,value):
        if not SECURITY_PATTERN.fullmatch(value):
            raise ValueError('证券代码必须为六位数字加 .SH、.SZ 或 .BJ')
        return value

    @field_validator('quantity','sellable_quantity')
    @classmethod
    def integral(cls,value):
        if value is not None and value % 1 != 0:
            raise ValueError('数量必须为整数股')
        return value

    @model_validator(mode='after')
    def sellable(self):
        if self.sellable_quantity is not None and self.sellable_quantity > self.quantity:
            raise ValueError('可卖数量不得超过持仓数量')
        return self


class RiskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    max_position_weight: float | None = Field(default=None,gt=0,le=1)
    max_equity_weight: float | None = Field(default=None,gt=0,le=1)
    target_annual_volatility: float | None = Field(default=None,gt=0)
    max_participation: float | None = Field(default=None,gt=0,le=1)
    horizon: int | None = None
    commission_rate: float | None = Field(default=None,ge=0,lt=1)
    min_commission: float | None = Field(default=None,ge=0)
    fee_per_side: float | None = Field(default=None,ge=0,lt=1,description='Additional explicit cost stress, above CN fees')


class AdviceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    holdings: list[Holding]
    cash: float = Field(ge=0,allow_inf_nan=False)
    risk_profile: RiskRequest | None = None
    as_of: date | None = None


class CSVAdviceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    holdings_csv: str = Field(description='CSV: security_id,quantity; optional average_cost,acquired_date,sellable_quantity')
    cash: float = Field(ge=0,allow_inf_nan=False)
    risk_profile: RiskRequest | None = None
    as_of: date | None = None


def create_app(service, api_key):
    if not isinstance(api_key,str) or not api_key.strip():
        raise ValueError('必须提供 A-share API 密钥')
    app = FastAPI(title='A-share quantitative research API',description='CNY research snapshots; no order submission')

    def guard(key):
        if key is None or not hmac.compare_digest(key,api_key):
            raise HTTPException(401,'API密钥无效')

    def call(fn):
        try:
            return fn()
        except ModelNotPublished as exc:
            raise HTTPException(503,str(exc)) from exc
        except SecurityNotFound as exc:
            raise HTTPException(404,str(exc)) from exc
        except (RequestError,ValueError,TypeError) as exc:
            raise HTTPException(422,str(exc)) from exc

    @app.get('/health')
    def health():
        return {'status':'ok','market':'CN','currency':'CNY'}

    @app.get('/v1/model/status')
    def status(x_api_key: str | None = Header(None)):
        guard(x_api_key)
        return call(service.status)

    @app.get('/v1/rankings')
    def rankings(horizon:int=20,limit:int|None=None,as_of:date|None=None,x_api_key:str|None=Header(None)):
        guard(x_api_key)
        return call(lambda:service.rank_market(horizon,limit,as_of))

    @app.get('/v1/stocks/{code}/forecast')
    def stock(code:str,as_of:date|None=None,x_api_key:str|None=Header(None)):
        guard(x_api_key)
        return call(lambda:service.forecast_stock(code,as_of))

    @app.post('/v1/portfolio/advice')
    def advice(body:AdviceRequest,x_api_key:str|None=Header(None)):
        guard(x_api_key)
        records = [holding.model_dump(mode='json',exclude_none=True) for holding in body.holdings]
        holdings = pd.DataFrame(records) if records else pd.DataFrame(columns=['security_id','quantity'])
        risk = body.risk_profile.model_dump(exclude_none=True) if body.risk_profile else None
        return call(lambda:service.advise(holdings,body.cash,risk,body.as_of))

    @app.post('/v1/portfolio/advice/csv')
    def advice_csv(body:CSVAdviceRequest,x_api_key:str|None=Header(None)):
        guard(x_api_key)
        risk = body.risk_profile.model_dump(exclude_none=True) if body.risk_profile else None
        return call(lambda:service.advise(parse_holdings_csv(body.holdings_csv),body.cash,risk,body.as_of))

    return app


def main():
    import uvicorn
    from .service import AShareQuantService, DATA, MODELS
    key = os.environ.get('ASHARE_QUANT_API_KEY')
    if not key:
        raise RuntimeError('必须设置 ASHARE_QUANT_API_KEY')
    parser = argparse.ArgumentParser(description='Serve an independent CNY A-share research snapshot')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--models-root',type=Path,default=MODELS)
    args = parser.parse_args()
    uvicorn.run(create_app(AShareQuantService(args.data_root,args.models_root),key),host=args.host,port=args.port)


if __name__ == '__main__':
    main()
