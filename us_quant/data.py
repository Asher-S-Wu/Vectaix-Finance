"""Public daily data ingestion with immutable provenance and explicit price semantics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_immutable_json(path, value):
    path=Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Existing evidence is immutable: {path.name}')
    if not path.exists():write_json(path,value)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_chart(payload, expected_symbol):
    chart=payload.get('chart',{})
    if chart.get('error') or not chart.get('result'):raise ValueError('Provider returned no history')
    result=chart['result'][0];meta=result['meta']
    if meta.get('symbol')!=expected_symbol:raise ValueError('Provider symbol mismatch')
    if meta.get('currency')!='USD':raise ValueError('Source currency must be USD')
    if meta.get('instrumentType') not in ('EQUITY','ETF'):raise ValueError('Unsupported asset type')
    if meta.get('exchangeTimezoneName')!='America/New_York':raise ValueError('Unexpected exchange timezone')
    timestamps=result.get('timestamp',[])
    if not timestamps:raise ValueError('No daily timestamps')
    dates=pd.to_datetime(timestamps,unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    if dates.duplicated().any():raise ValueError('duplicate source dates')
    q=result['indicators']['quote'][0];adj=result['indicators']['adjclose'][0]['adjclose']
    values={f'split_adjusted_{x}':q[x] for x in ('open','high','low','close')}
    values.update(volume=q['volume'],adj_close=adj)
    if any(len(v)!=len(dates) for v in values.values()):raise ValueError('Mismatched source array lengths')
    out=pd.DataFrame(values,dtype=float);out.insert(0,'security_id',expected_symbol);out.insert(0,'date',dates)
    numerical=out.select_dtypes('number')
    positive=numerical.drop(columns='volume').gt(0).all(axis=1)
    finite=np.isfinite(numerical).all(axis=1)
    o,h,l,c=[out[f'split_adjusted_{x}'] for x in ('open','high','low','close')]
    consistent=(h>=np.maximum(o,c))&(l<=np.minimum(o,c))&(h>=l)
    out['quote_present']=out.volume.gt(0)&c.notna()
    out['data_valid']=positive&finite&consistent&out.volume.ge(0)
    # Yahoo historical close and volume are split-adjusted. Their product is a
    # vendor-derived trading-value proxy, not verified raw trade consideration.
    out['dollar_volume_proxy']=out.split_adjusted_close*out.volume
    return out.sort_values('date').reset_index(drop=True),meta,result.get('events',{})
