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
