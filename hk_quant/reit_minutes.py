"""REIT显式频率原始采集：不构造日线、复权或停牌状态。"""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime,timezone
from urllib.parse import urlsplit

import pandas as pd

FIELDS=('ts_code','trade_time','open','close','high','low','vol','amount')
DEFAULT_ROOT=Path(__file__).resolve().parents[1]/'data/hk/universal/reit_minutes_raw'


def quarter_windows(start,end):
    start=pd.Timestamp(start).normalize();end=pd.Timestamp(end).normalize()
    if pd.isna(start) or pd.isna(end) or start>end:raise ValueError('采集日期无效')
    for period in pd.period_range(start,end,freq='Q'):
        yield max(start,period.start_time).strftime('%Y-%m-%d'),min(end,period.end_time.normalize()).strftime('%Y-%m-%d')


def half_month_windows(start,end):
    start=pd.Timestamp(start).normalize();end=pd.Timestamp(end).normalize()
    if pd.isna(start) or pd.isna(end) or start>end:raise ValueError('采集日期无效')
    for month in pd.period_range(start,end,freq='M'):
        first=month.start_time;middle=first+pd.Timedelta(days=15)
        for left,right in ((first,middle-pd.Timedelta(days=1)),(middle,month.end_time.normalize())):
            left=max(start,left);right=min(end,right)
            if left<=right:yield left.strftime('%Y-%m-%d'),right.strftime('%Y-%m-%d')


def _validate_mode(freq,window):
    if (freq,window) not in (('60min','quarter'),('1min','half_month')):
        raise ValueError('必须明确选择60min/quarter或1min/half_month；不自动更换频率或窗口')


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def load_universe(current_csv,historical_json):
    frame=pd.read_csv(current_csv,dtype={'Stock Code':str})
    current=frame.loc[frame.Category.eq('Real Estate Investment Trusts')]
    records=[{'ts_code':str(row['Stock Code']).zfill(5)+'.HK','name':row['Name of Securities'],'isin':row['ISIN'],
              'source_url':'https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx',
              'source_file':str(Path(current_csv).resolve()),'identity_scope':'current HKEX REIT list; historical lifecycle not inferred'} for row in current.to_dict('records')]
    history=json.loads(Path(historical_json).read_text(encoding='utf-8'))
    for row in history:
        if not {'ts_code','verified','source_url','source_file','last_dealing_date'}.issubset(row):raise ValueError('历史REIT缺少来源或最后交易日')
        host=urlsplit(row['source_url']).hostname
        if row['verified'] is not True or not host or not (host.endswith('hkex.com.hk') or host.endswith('hkexnews.hk')) or not Path(row['source_file']).is_file():
            raise ValueError('历史REIT必须有已核验的港交所来源文件')
        if pd.isna(pd.Timestamp(row['last_dealing_date'])):raise ValueError('历史REIT最后交易日无效')
        records.append(row)
    codes=[row['ts_code'] for row in records]
    if len(set(codes))!=len(codes) or not all(re.fullmatch(r'\d{5}\.HK',code) for code in codes):raise ValueError('REIT代码重复或格式无效')
    return records


def collect_partition(client,ts_code,start,end,root,*,freq='60min',window='quarter'):
    _validate_mode(freq,window)
    if not re.fullmatch(r'\d{5}\.HK',ts_code):raise ValueError('证券代码格式无效')
    start=pd.Timestamp(start).normalize();end=pd.Timestamp(end).normalize()
    if pd.isna(start) or pd.isna(end) or start>end or (end-start).days>=92:raise ValueError('每个请求必须不超过一个季度92日')
    if window=='half_month' and (start.to_period('M')!=end.to_period('M') or (start.day<=15)!=(end.day<=15)):
        raise ValueError('1min请求必须限制在同一个半月窗口内')
    request={'api_name':'hk_mins','params':{'ts_code':ts_code,'freq':freq,'start_date':start.strftime('%Y-%m-%d 00:00:00'),
                                          'end_date':end.strftime('%Y-%m-%d 23:59:59')},'fields':','.join(FIELDS)}
    folder=Path(root)/ts_code;folder.mkdir(parents=True,exist_ok=True)
    stem=start.strftime('%Y%m%d')+'_'+end.strftime('%Y%m%d')
    meta_path=folder/(stem+'.meta.json');parquet_path=folder/(stem+'.parquet')
    if meta_path.exists():
        cached=json.loads(meta_path.read_text(encoding='utf-8'))
        if cached['request']!=request:raise ValueError('缓存日期、频率或字段与请求不一致')
        if cached['status'] in ('downloaded','api_empty'):
            if not Path(cached['raw_file']).is_file() or (cached['status']=='downloaded' and not Path(cached['parquet_file']).is_file()):
                raise ValueError('缓存已完成记录缺少原始响应或数据文件')
            return cached
    stamp=datetime.now(timezone.utc)
    raw_path=folder/(stem+'.'+stamp.strftime('%Y%m%dT%H%M%S%f')+'.json.gz')
    envelope={'request':request,'started_at':stamp.isoformat(),'response':None,'http_status':None}
    result={'request':request,'status':'error','rows':0,'first_observed':None,'last_observed':None,
            'raw_file':str(raw_path.resolve()),'parquet_file':None,'history_coverage_complete':False}
    try:
        response=client._request('POST',client.base_url+'/v1/tushare/query',request)
        envelope['response']=response;envelope['http_status']=200
        if not isinstance(response,dict) or response.get('code')!=0:raise ValueError('上游业务返回非零code')
        data=response['data'];fields=data['fields'];items=data['items']
        if len(items)>=8000 or data['has_more'] is not False:raise ValueError('响应可能截断或要求分页；分区请求不接受截断数据，未猜测offset参数')
        if len(set(fields))!=len(fields) or not set(FIELDS).issubset(fields):raise ValueError('返回缺少必需字段或字段重复')
        frame=pd.DataFrame(items,columns=fields)
        times=pd.to_datetime(frame.trade_time,errors='coerce')
        if times.isna().any() or not frame.ts_code.eq(ts_code).all():raise ValueError('返回代码或交易时间不匹配')
        if times.lt(start).any() or times.ge(end+pd.Timedelta(days=1)).any():raise ValueError('返回交易时间超出请求日期')
        if frame.assign(_time=times).duplicated(['ts_code','_time']).any():raise ValueError('返回存在重复证券/时间键')
        if not len(frame):
            result.update(status='api_empty',empty_meaning='API returned no records; not evidence of suspension or historical coverage')
        else:
            frame[list(FIELDS)].to_parquet(parquet_path,index=False)
            result.update(status='downloaded',rows=len(items),first_observed=times.min().strftime('%Y-%m-%d %H:%M:%S'),
                          last_observed=times.max().strftime('%Y-%m-%d %H:%M:%S'),parquet_file=str(parquet_path.resolve()))
    except Exception as exc:
        result['error']=str(exc);envelope['error']=str(exc)
    envelope['completed_at']=datetime.now(timezone.utc).isoformat()
    result['updated_at']=envelope['completed_at']
    with gzip.open(raw_path,'wt',encoding='utf-8') as stream:json.dump(envelope,stream,ensure_ascii=False,allow_nan=False)
    _write_json(meta_path,result)
    return result


def collect_reit_minutes(client,universe,start,end,root,*,freq='60min',window='quarter'):
    _validate_mode(freq,window)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    previous=root/'collection_status.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))['frequency']!=freq:
        raise ValueError('输出目录已有不同频率的采集，必须使用独立目录')
    _write_json(root/'universe.json',universe)
    status={'status':'running','pid':os.getpid(),'start':pd.Timestamp(start).strftime('%Y-%m-%d'),
            'end':pd.Timestamp(end).strftime('%Y-%m-%d'),'frequency':freq,'window':window,'history_coverage_complete':False,
            'interpretation':'Raw provider minutes only. Adjustment, currency and units are not assumed. Empty requests are not suspensions.',
            'securities':{},'partitions':[],'completed_partitions':0,'errors':0}
    work=[]
    windows=quarter_windows if window=='quarter' else half_month_windows
    for row in universe:
        stop=min(pd.Timestamp(end),pd.Timestamp(row['last_dealing_date'])) if 'last_dealing_date' in row else pd.Timestamp(end)
        if stop<pd.Timestamp(start):continue
        for left,right in windows(start,stop):work.append((row['ts_code'],left,right))
        status['securities'][row['ts_code']]={'rows':0,'first_observed':None,'last_observed':None,'api_empty_partitions':0,'errors':0}
    status['planned_partitions']=len(work)
    _write_json(root/'collection_status.json',status)
    for code,left,right in work:
        try:
            record=collect_partition(client,code,left,right,root,freq=freq,window=window)
        except Exception as exc:
            status.update(status='stopped_on_error',current_error=str(exc),current_partition={'ts_code':code,'start':left,'end':right})
            _write_json(root/'collection_status.json',status)
            raise
        status['partitions'].append(record);status['completed_partitions']+=1
        stock=status['securities'][code]
        if record['status']=='downloaded':
            stock['rows']+=record['rows']
            if stock['first_observed'] is None or record['first_observed']<stock['first_observed']:stock['first_observed']=record['first_observed']
            if stock['last_observed'] is None or record['last_observed']>stock['last_observed']:stock['last_observed']=record['last_observed']
        elif record['status']=='api_empty':stock['api_empty_partitions']+=1
        else:stock['errors']+=1;status['errors']+=1
        status['updated_at']=datetime.now(timezone.utc).isoformat()
        _write_json(root/'collection_status.json',status)
        print(f'{code} {left} {right} {record["status"]} rows={record["rows"]} progress={status["completed_partitions"]}/{len(work)}',flush=True)
        if record['status']=='error' and freq=='1min':
            status.update(status='stopped_on_error',current_error=record['error'])
            _write_json(root/'collection_status.json',status)
            return status
    status['status']='finished_with_errors' if status['errors'] else 'finished'
    _write_json(root/'collection_status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description='单并发采集有来源依据的REIT原始分钟数据，显式指定频率和窗口')
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--current-csv',type=Path,default=DEFAULT_ROOT.parent/'references/reit_research/equities_and_reits.csv')
    parser.add_argument('--historical-json',type=Path,default=DEFAULT_ROOT/'sources/historical_reits.json')
    parser.add_argument('--skill-root',type=Path,default=Path('C:/Users/Noah/.agents/skills/tushare-relay'))
    parser.add_argument('--start',default='20100101');parser.add_argument('--end',default='20260910')
    parser.add_argument('--freq',choices=['60min','1min'],default='60min')
    parser.add_argument('--window',choices=['quarter','half_month'],default='quarter')
    args=parser.parse_args()
    _validate_mode(args.freq,args.window)
    if args.freq=='1min' and args.root.resolve()==DEFAULT_ROOT.resolve():
        raise ValueError('1min必须显式指定独立输出目录，不能写入原60min目录')
    universe=load_universe(args.current_csv,args.historical_json)
    sys.path.insert(0,str(args.skill_root/'scripts'))
    from tushare_client import RelayClient,load_settings
    settings=load_settings(root=args.skill_root)
    client=RelayClient(settings.api_key,settings.base_url,timeout=90,max_retries=2,interval_seconds=2.,trust_env=False)
    print(f'REIT minute collector PID={os.getpid()} securities={len(universe)} freq={args.freq} window={args.window}',flush=True)
    result=collect_reit_minutes(client,universe,args.start,args.end,args.root,freq=args.freq,window=args.window)
    return 1 if result['status']=='stopped_on_error' else 0


if __name__=='__main__':raise SystemExit(main())
