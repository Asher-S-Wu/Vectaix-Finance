import io
import json
from urllib.error import HTTPError

import pandas as pd
import pytest
from us_quant.collect import validate_request_symbol, classify_http_status


def test_symbol_validation_blocks_path_or_url_injection():
    assert validate_request_symbol('BRK-B')=='BRK-B'
    for bad in ['../AAPL','AAPL?x=1','https://bad','']:
        with pytest.raises(ValueError):validate_request_symbol(bad)


def test_rate_limit_and_access_denial_stop_collection():
    assert classify_http_status(429)=='stop'
    assert classify_http_status(403)=='stop'
    assert classify_http_status(404)=='unavailable'
    assert classify_http_status(200)=='ok'


def test_verified_symbol_override_preserves_original_identity():
    from us_quant.collect import effective_symbol
    row={'ticker_2015':'BK','yahoo_query_symbol':'BK'}
    assert effective_symbol(row,{'BK':{'symbol':'BNY','source':'issuer announcement'}})=='BNY'
    assert row['ticker_2015']=='BK'
    with pytest.raises(ValueError):effective_symbol(row,{'BK':{'symbol':'BNY','source':''}})


@pytest.mark.parametrize('failure,expected', [(403,'paused'), (429,'paused'), (500,'error'), (TimeoutError('timed out'),'error')])
def test_manual_resume_revisits_access_and_transport_failures(tmp_path,monkeypatch,failure,expected):
    from us_quant import collect as collector
    from tests.test_us_data_features import chart
    universe=tmp_path/'universe.csv'
    pd.DataFrame(columns=['ticker_2015','yahoo_query_symbol','mapping_status','issuer_name']).to_csv(universe,index=False)
    root=tmp_path/'data';calls=[]

    def failing(request,timeout):
        calls.append(request.full_url)
        if isinstance(failure,int):raise HTTPError(request.full_url,failure,'test error',None,None)
        raise failure

    monkeypatch.setattr(collector,'urlopen',failing)
    if expected=='paused':
        with pytest.raises(RuntimeError,match='paused'):collector.collect(root,universe,delay=0)
    else:collector.collect(root,universe,delay=0)
    progress=json.loads((root/'collection_progress.json').read_text())
    assert progress[0]['status']==expected
    payload=chart();payload['chart']['result'][0]['meta']['symbol']='SPY'

    def successful(request,timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(collector,'urlopen',successful)
    records=collector.collect(root,universe,delay=0)
    assert len(calls)==2
    assert records[0]['status']=='downloaded_pending_identity_review'


def test_manual_resume_keeps_confirmed_missing_history(tmp_path,monkeypatch):
    from us_quant import collect as collector
    universe=tmp_path/'universe.csv'
    pd.DataFrame(columns=['ticker_2015','yahoo_query_symbol','mapping_status','issuer_name']).to_csv(universe,index=False)
    root=tmp_path/'data';root.mkdir()
    record={'security_id':'SPY','query_symbol':'SPY','status':'unavailable','reason':'HTTP 404'}
    (root/'collection_progress.json').write_text(json.dumps([record]))
    monkeypatch.setattr(collector,'urlopen',lambda *args,**kwargs:pytest.fail('Confirmed missing history was requested again'))
    assert collector.collect(root,universe,delay=0)==[record]
