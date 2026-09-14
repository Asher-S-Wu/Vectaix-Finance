import gzip
import json

import pandas as pd
import pytest

from hk_quant.reit_minutes import collect_partition, quarter_windows, collect_reit_minutes, load_universe

FIELDS=['ts_code','trade_time','open','close','high','low','vol','amount']


class FakeRelayClient:
    base_url='https://example.test'
    def __init__(self,response):self.response=response;self.calls=[]
    def _request(self,method,url,payload):
        self.calls.append((method,url,payload))
        return self.response


def response(items=None,**extra):
    return {'code':0,'source':'tushare','data':{'fields':FIELDS,'items':items if items is not None else [['00823.HK','2021-01-04 09:30:00',70.,70.,70.,70.,100,7000.]],'count':0,'has_more':False,**extra}}


def test_quarters_are_bounded_inclusive_and_clipped_to_requested_dates():
    assert list(quarter_windows('2021-02-01','2021-04-13'))==[('2021-02-01','2021-03-31'),('2021-04-01','2021-04-13')]
    assert max((pd.Timestamp(end)-pd.Timestamp(start)).days+1 for start,end in quarter_windows('20100101','20260910'))<=92


def test_original_fields_items_and_params_are_retained_and_count_uses_items(tmp_path):
    client=FakeRelayClient(response())
    result=collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path)
    assert result['status']=='downloaded' and result['rows']==1
    payload=client.calls[0][2]
    assert payload=={'api_name':'hk_mins','params':{'ts_code':'00823.HK','freq':'60min','start_date':'2021-01-01 00:00:00','end_date':'2021-03-31 23:59:59'},'fields':','.join(FIELDS)}
    saved=pd.read_parquet(result['parquet_file'])
    assert saved.columns.tolist()==FIELDS and saved.iloc[0].trade_time=='2021-01-04 09:30:00'
    with gzip.open(result['raw_file'],'rt',encoding='utf-8') as stream:raw=json.load(stream)
    assert raw['response']==client.response and raw['request']==payload
    again=collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path)
    assert again['rows']==1 and len(client.calls)==1


def test_empty_partition_is_not_historical_or_suspension_coverage(tmp_path):
    client=FakeRelayClient(response([]))
    result=collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path)
    assert result['status']=='api_empty' and result['rows']==0
    assert not result['history_coverage_complete']
    assert not result['parquet_file']
    collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path)
    assert len(client.calls)==1


@pytest.mark.parametrize('bad',[
    response(has_more=True),
    response([['00823.HK','2021-01-04 09:30:00',70.,70.,70.,70.,100,7000.]]*8000),
    response([['00823.HK','2021-01-04 09:30:00',70.,70.,70.,70.,100,7000.]]*2),
    response([['02778.HK','2021-01-04 09:30:00',70.,70.,70.,70.,100,7000.]]),
    response([['00823.HK','2021-04-01 09:30:00',70.,70.,70.,70.,100,7000.]]),
    {'code':-1,'msg':'not permitted','data':None},
    {'code':0,'data':{'fields':['ts_code'],'items':[['00823.HK']],'has_more':False}},
])
def test_truncation_pagination_conflicts_and_business_errors_never_publish_partition(tmp_path,bad):
    result=collect_partition(FakeRelayClient(bad),'00823.HK','2021-01-01','2021-03-31',tmp_path)
    assert result['status']=='error' and result['error']
    assert not list(tmp_path.rglob('*.parquet'))
    assert list(tmp_path.rglob('*.json.gz'))


def test_cached_partition_requires_exact_request_metadata(tmp_path):
    client=FakeRelayClient(response())
    result=collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path)
    path=tmp_path/'00823.HK'/'20210101_20210331.meta.json'
    meta=json.loads(path.read_text());meta['request']['params']['freq']='1min';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='缓存'):
        collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path)
    assert len(client.calls)==1


def test_summary_tracks_empty_partitions_and_actual_observed_time(tmp_path):
    client=FakeRelayClient(response())
    universe=[{'ts_code':'00823.HK','name':'LINK','source_url':'https://www.hkex.com.hk/list.xlsx'}]
    result=collect_reit_minutes(client,universe,'20210101','20210331',tmp_path)
    stock=result['securities']['00823.HK']
    assert stock['rows']==1 and stock['first_observed']=='2021-01-04 09:30:00'
    assert stock['last_observed']=='2021-01-04 09:30:00'
    assert result['status']=='finished' and result['history_coverage_complete'] is False


def test_historic_code_stops_at_verified_last_dealing_date(tmp_path):
    client=FakeRelayClient(response([]))
    universe=[{'ts_code':'00625.HK','last_dealing_date':'2010-04-19','verified':True,'source_url':'https://www.hkexnews.hk/source.pdf'}]
    result=collect_reit_minutes(client,universe,'20100101','20260910',tmp_path)
    assert len(client.calls)==2
    assert client.calls[-1][2]['params']['end_date']=='2010-04-19 23:59:59'
    assert result['securities']['00625.HK']['api_empty_partitions']==2


def test_universe_uses_current_reit_category_and_verified_historical_sources(tmp_path):
    csv=tmp_path/'master.csv'
    pd.DataFrame({'Stock Code':['00823','00625'],'Name of Securities':['LINK','SHEIN'],'ISIN':['HK0823032773','OTHER'],'Category':['Real Estate Investment Trusts','Equity']}).to_csv(csv,index=False)
    pdf=tmp_path/'official.pdf';pdf.write_bytes(b'%PDF test evidence fixture')
    source=tmp_path/'history.json';source.write_text(json.dumps([{'ts_code':'00625.HK','verified':True,'source_url':'https://www1.hkexnews.hk/history.pdf','source_file':str(pdf),'last_dealing_date':'2010-04-19'}]))
    result=load_universe(csv,source)
    assert [r['ts_code'] for r in result]==['00823.HK','00625.HK']
    assert result[1]['last_dealing_date']=='2010-04-19'
    data=json.loads(source.read_text());data[0]['verified']=False;source.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='核验'):load_universe(csv,source)


def test_failed_partition_is_retained_in_collection_status(tmp_path):
    result=collect_reit_minutes(FakeRelayClient({'code':-1,'msg':'error','data':None}),[{'ts_code':'00823.HK'}],'20210101','20210331',tmp_path)
    assert result['status']=='finished_with_errors'
    assert result['errors']==1 and result['securities']['00823.HK']['errors']==1
    assert result['history_coverage_complete'] is False


def test_half_month_windows_cover_boundaries_without_overlap():
    from hk_quant.reit_minutes import half_month_windows
    assert list(half_month_windows('2016-02-14','2016-03-02'))==[
        ('2016-02-14','2016-02-15'),('2016-02-16','2016-02-29'),('2016-03-01','2016-03-02')]


def test_explicit_one_minute_request_and_cache_frequency_isolation(tmp_path):
    client=FakeRelayClient(response())
    first=collect_partition(client,'00823.HK','2021-01-01','2021-01-15',tmp_path,freq='1min',window='half_month')
    assert first['status']=='downloaded' and client.calls[0][2]['params']['freq']=='1min'
    with pytest.raises(ValueError,match='缓存'):
        collect_partition(client,'00823.HK','2021-01-01','2021-01-15',tmp_path)
    assert len(client.calls)==1


def test_one_minute_cannot_request_quarter_or_cross_half_month(tmp_path):
    client=FakeRelayClient(response())
    with pytest.raises(ValueError):collect_partition(client,'00823.HK','2021-01-01','2021-03-31',tmp_path,freq='1min')
    with pytest.raises(ValueError):collect_partition(client,'00823.HK','2021-01-14','2021-01-29',tmp_path,freq='1min',window='half_month')
    assert not client.calls


@pytest.mark.parametrize('bad',[response(has_more=True),response([['00823.HK','2021-01-04 09:30:00',70.,70.,70.,70.,100,7000.]]*8000),{'code':-1,'msg':'denied','data':None}])
def test_one_minute_stops_after_refusal_or_truncated_partition(tmp_path,bad):
    client=FakeRelayClient(bad)
    result=collect_reit_minutes(client,[{'ts_code':'00823.HK'}],'20210101','20210331',tmp_path,freq='1min',window='half_month')
    assert result['status']=='stopped_on_error' and result['completed_partitions']==1
    assert len(client.calls)==1 and not list(tmp_path.rglob('*.parquet'))


def test_one_minute_preserves_historical_last_dealing_cutoff(tmp_path):
    client=FakeRelayClient(response([]))
    universe=[{'ts_code':'00625.HK','last_dealing_date':'2010-04-19'}]
    result=collect_reit_minutes(client,universe,'20100101','20170630',tmp_path,freq='1min',window='half_month')
    assert result['frequency']=='1min' and result['window']=='half_month'
    assert len(client.calls)==8
    assert client.calls[-1][2]['params']['start_date']=='2010-04-16 00:00:00'
    assert client.calls[-1][2]['params']['end_date']=='2010-04-19 23:59:59'
