"""Slow, resumable public research-history collector. Never bypasses challenges."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import pandas as pd
from .data import parse_chart,write_json,write_immutable_json,sha256


def validate_request_symbol(symbol):
    if not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,15}',symbol):raise ValueError('Unsafe symbol')
    return symbol


def classify_http_status(status):
    if status in (401,403,429):return 'stop'
    if status==404:return 'unavailable'
    return 'ok' if status==200 else 'error'


def effective_symbol(row,overrides):
    override=overrides.get(row['ticker_2015'])
    if override:
        if not override.get('source'):raise ValueError('Symbol override requires issuer evidence')
        return validate_request_symbol(override['symbol'])
    return validate_request_symbol(row['yahoo_query_symbol'])


def collect(root,universe,start='2014-01-01',end='2026-09-30',delay=1.5,overrides=None):
    root=Path(root);raw=root/'raw';bars=root/'bars';raw.mkdir(parents=True,exist_ok=True);bars.mkdir(parents=True,exist_ok=True)
    cohort=pd.read_csv(universe,keep_default_na=False)
    overrides={} if overrides is None else overrides
    if overrides:write_immutable_json(root/'symbol_overrides_contract.json',overrides)
    progress=root/'collection_progress.json'
    previous={r['security_id']:r for r in json.loads(progress.read_text())} if progress.exists() else {}
    write_immutable_json(root/'collection_contract.json',dict(source='Yahoo Finance public daily chart',start=start,end=end,universe_sha256=sha256(universe),universe_rows=len(cohort),policy='all original cohort identities retained; unresolved complex lineages quarantined, not mapped to successor'))
    begin=int(pd.Timestamp(start,tz='UTC').timestamp());stop=int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)).timestamp())
    records=[]
    requests=[dict(row) for row in cohort.to_dict('records')]+[dict(ticker_2015='SPY',yahoo_query_symbol='SPY',mapping_status='benchmark_etf',issuer_name='SPDR S&P 500 ETF Trust')]
    for index,row in enumerate(requests,1):
        symbol=effective_symbol(row,overrides);identity=row['ticker_2015'];status='unavailable'
        record=dict(security_id=identity,query_symbol=symbol,mapping_status=row['mapping_status'],issuer_name=row['issuer_name'])
        if previous.get(identity,{}).get('status')=='unavailable' and previous[identity].get('query_symbol')==symbol:
            record=previous[identity]
        elif row['mapping_status']=='complex_lineage_require_review':
            record.update(status='identity_unresolved',reason='Original 2015 identity cannot safely be joined to current/successor ticker')
        else:
            url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={begin}&period2={stop}&interval=1d&events=div%2Csplits'
            path=raw/f'{identity}.json'
            try:
                if not path.exists():
                    time.sleep(delay)
                    with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as response:payload=json.loads(response.read())
                    write_json(path,payload)
                else:payload=json.loads(path.read_text())
                frame,meta,events=parse_chart(payload,symbol)
                frame=frame.loc[frame.date.between(start,end)].copy();frame['security_id']=identity
                # Listing dates are a validation guard, never a completeness screen.
                if frame.empty:raise ValueError('No history in declared date range')
                frame.to_parquet(bars/f'{identity}.parquet',index=False)
                write_json(raw/f'{identity}.metadata.json',dict(meta=meta,events=events,url=url,retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),raw_sha256=sha256(path)))
                record.update(status='downloaded_pending_identity_review',rows=len(frame),first_date=str(frame.date.min().date()),last_date=str(frame.date.max().date()),invalid_rows=int((~frame.data_valid).sum()),quoted_rows=int(frame.quote_present.sum()),provider_name=meta.get('longName',meta.get('shortName','')),exchange=meta.get('exchangeName'),instrument=meta.get('instrumentType'),raw_sha256=sha256(path),source_url=url)
            except HTTPError as exc:
                action=classify_http_status(exc.code)
                record.update(status='unavailable',reason=f'HTTP {exc.code}')
                if action=='stop':
                    records.append(record);write_json(root/'collection_progress.json',records)
                    raise RuntimeError(f'Provider access/rate limit HTTP {exc.code}; collection paused, no bypass') from None
            except (ValueError,KeyError,TimeoutError,OSError) as exc:
                record.update(status='unavailable',reason=str(exc)[:220])
        records.append(record);write_json(root/'collection_progress.json',records)
        print(f'{index}/{len(requests)} {identity}: {record["status"]}',flush=True)
    write_json(root/'collection_audit.json',dict(source='Yahoo Finance',declared_cohort_count=len(cohort),requested_benchmark_count=1,records=records,raw_redistribution_allowed=False,point_in_time_vendor_vintages=False,delisted_completeness=False))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('data/us/oef2015'));p.add_argument('--universe',type=Path,default=Path('docs/us/declared_universe.csv'));p.add_argument('--symbol-overrides',type=Path);args=p.parse_args();collect(args.data_root,args.universe,overrides=json.loads(args.symbol_overrides.read_text()) if args.symbol_overrides else None)
