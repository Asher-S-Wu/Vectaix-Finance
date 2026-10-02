"""Contract tests for audited, resumable mainland market collection."""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.collect import (
    CollectionError, HelperClient, PermanentAPIError, TransientAPIError,
    collect_query, is_a_share, normalize_units,
)


class Pages:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def query(self, endpoint, params, fields):
        self.calls.append((endpoint, dict(params), fields))
        return {"status": "success", "api_code": 0, "data": {
            "fields": ["ts_code", "trade_date", "close"],
            "items": self.pages[params["offset"]],
        }}


def test_short_final_page_and_complete_provenance(tmp_path):
    client = Pages({0: [["600000.SH", "20260102", 5.], ["000001.SZ", "20260102", 7.]], 2: [["920001.BJ", "20260102", 8.]]})
    result = collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2)
    assert result["rows"] == 3
    assert result["status"] == "complete"
    assert [p[1]["offset"] for p in client.calls] == [0, 2]
    parquet = tmp_path / result["artifact"]
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == result["sha256"]
    assert len(pd.read_parquet(parquet)) == 3
    assert all(p["fetched_at"] and p["rows"] and p["params"]["limit"] == 2 for p in result["requests"])
    assert result["params"] == {"trade_date": "20260102"}


def test_cache_is_verified_and_does_not_repeat_queries(tmp_path):
    client = Pages({0: [["600000.SH", "20260102", 5.]]})
    result = collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2)
    again = collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2)
    assert result == again
    assert len(client.calls) == 1
    (tmp_path / result["artifact"]).write_bytes(b"bad data")
    with pytest.raises(CollectionError, match="checksum"):
        collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2)


def test_overlapping_pages_are_rejected_and_not_marked_complete(tmp_path):
    client = Pages({0: [["600000.SH", "20260102", 5.]], 1: [["600000.SH", "20260102", 5.]]})
    with pytest.raises(CollectionError, match="duplicate"):
        collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=1)
    assert not list(tmp_path.rglob("*.parquet"))
    audit = json.loads(next(tmp_path.rglob("*.error.json")).read_text())
    assert audit["status"] == "error"


def test_empty_response_is_explicit_and_accepted_only_when_allowed(tmp_path):
    with pytest.raises(CollectionError, match="empty"):
        collect_query(Pages({0: []}), "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2)
    result = collect_query(Pages({0: []}), "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2, allow_empty=True)
    assert result["status"] == "empty" and result["rows"] == 0


def test_exact_full_page_requires_terminal_page(tmp_path):
    client = Pages({0: [["600000.SH", "20260102", 5.]], 1: []})
    result = collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=1)
    assert len(result["requests"]) == 2 and result["requests"][-1]["rows"] == 0


def test_page_larger_than_requested_and_schema_changes_are_rejected(tmp_path):
    with pytest.raises(CollectionError, match="page size"):
        collect_query(Pages({0: [["1", "20260102", 5.], ["2", "20260102", 5.]]}), "daily", {}, "ts_code,trade_date,close", tmp_path, page_size=1)


def test_helper_retries_transient_result_without_credential_access():
    outputs = iter([{"status": "timeout"}, {"status": "success", "api_code": 0, "data": {"fields": [], "items": []}}])
    calls, sleeps = [], []
    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps(next(outputs)), "")
    client = HelperClient(runner=runner, sleep=sleeps.append, requests_per_minute=0, retries=2)
    assert client.query("daily", {"trade_date": "20260102"}, "close")["status"] == "success"
    assert len(calls) == 2 and sleeps == [1.0]
    assert calls[0][1] == "/workspace/shared/tushare_persistent.py"
    assert calls[0][2:] == ["query", "daily", '{"trade_date":"20260102"}', "close"]


def test_permission_error_stops_without_retry():
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, '{"status":"api_rejected","api_code":-2001}', "")
    client = HelperClient(runner=runner, requests_per_minute=0)
    with pytest.raises(PermanentAPIError):
        client.query("daily", {}, "close")
    assert len(calls) == 1


def test_a_share_universe_keeps_delisted_and_old_beijing_excludes_b_and_funds():
    accepted = ["600000.SH", "688001.SH", "000001.SZ", "300001.SZ", "430047.BJ", "833914.BJ", "920001.BJ"]
    rejected = ["900901.SH", "200001.SZ", "510300.SH", "159915.SZ", "000300.SH", "00001.HK"]
    assert all(is_a_share(s) for s in accepted)
    assert not any(is_a_share(s) for s in rejected)


def test_units_are_explicit_without_overwriting_source_fields():
    frame = pd.DataFrame({"vol": [3.], "amount": [2.], "total_mv": [5.], "circ_mv": [4.], "total_share": [6.], "float_share": [7.], "free_share": [8.], "turnover_rate": [9.], "pct_chg": [10.]})
    out = normalize_units(frame)
    assert out.loc[0, "volume_shares"] == 300
    assert out.loc[0, "amount_cny"] == 2000
    assert out.loc[0, "total_mv_cny"] == 50000
    assert out.loc[0, "total_share_shares"] == 60000
    assert out.loc[0, "turnover_rate_fraction"] == .09
    assert out.loc[0, "return_fraction"] == .1
    pd.testing.assert_frame_equal(frame, out[frame.columns])


def test_exact_six_digit_beijing_codes():
    assert is_a_share('920001.BJ')
    assert not is_a_share('9200001.BJ')
    assert not is_a_share('43001.BJ')


def test_loader_accepts_only_verified_manifests(tmp_path):
    from ashare_quant.collect import load_partitions
    client = Pages({0: [["600000.SH", "20260102", 5.]]})
    result = collect_query(client, "daily", {"trade_date": "20260102"}, "ts_code,trade_date,close", tmp_path, page_size=2)
    assert len(load_partitions(tmp_path, 'daily')) == 1
    (tmp_path / result['artifact']).write_bytes(b'corrupt')
    with pytest.raises(CollectionError, match='checksum'):
        load_partitions(tmp_path, 'daily')


def test_stock_st_type_is_part_of_unique_identity(tmp_path):
    class Client:
        def query(self, *args):
            return {'data': {'fields': ['ts_code', 'trade_date', 'type'], 'items': [
                ['600000.SH', '20260102', 'ST'], ['600000.SH', '20260102', '*ST']]}}
    result = collect_query(Client(), 'stock_st', {'trade_date': '20260102'}, '', tmp_path)
    assert result['rows'] == 2


def test_failed_field_schema_is_recorded(tmp_path):
    with pytest.raises(CollectionError, match='requested fields'):
        collect_query(Pages({0: [['600000.SH', '20260102', 5.]]}), 'daily', {}, 'close,trade_date,ts_code', tmp_path)
    assert list(tmp_path.rglob('*.error.json'))


def test_partition_iterator_filters_year_before_materializing(tmp_path):
    from ashare_quant.collect import iter_partitions, cached_partition
    for date in ('20250102', '20260102'):
        collect_query(Pages({0: [['600000.SH', date, 5.]]}), 'daily', {'trade_date': date}, 'ts_code,trade_date,close', tmp_path)
    found = list(iter_partitions(tmp_path, 'daily', year=2026))
    assert len(found) == 1
    frame, manifest = found[0]
    assert frame.iloc[0]['trade_date'] == '20260102'
    assert manifest['params']['trade_date'] == '20260102'
    hit_frame, hit_manifest = cached_partition(tmp_path, 'daily', {'trade_date': '20250102'})
    assert hit_frame.iloc[0]['trade_date'] == '20250102'
    assert cached_partition(tmp_path, 'daily', {'trade_date': '19900101'}) is None


def test_retry_budget_is_bounded():
    sleeps = []
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 40)
    client = HelperClient(runner=runner, sleep=sleeps.append, requests_per_minute=0, retries=2)
    with pytest.raises(TransientAPIError, match='retry budget'):
        client.query('daily', {}, '')
    assert len(calls) == 3
    assert sleeps == [1., 2.]


def test_schema_change_across_pages_rejected(tmp_path):
    class Client:
        def query(self, endpoint, params, fields):
            if params['offset'] == 0:
                return {'data': {'fields': ['ts_code', 'trade_date'], 'items': [['600000.SH', '20260102']]}}
            return {'data': {'fields': ['trade_date', 'ts_code'], 'items': []}}
    with pytest.raises(CollectionError, match='schema changed'):
        collect_query(Client(), 'daily', {}, '', tmp_path, page_size=1)


def test_unexpected_date_rejected(tmp_path):
    with pytest.raises(CollectionError, match='unexpected trade_date'):
        collect_query(Pages({0: [['600000.SH', '20250102', 5.]]}), 'daily', {'trade_date': '20260102'}, 'ts_code,trade_date,close', tmp_path)


def test_rate_and_concurrency_limits_are_bounded():
    from ashare_quant.collect import collect_market
    with pytest.raises(ValueError, match='rate budget'):
        HelperClient(requests_per_minute=201)
    with pytest.raises(ValueError, match='1..12'):
        collect_market(root=Path('/unused'), start='20160101', end='20260101', max_workers=13)


def test_max_pages_fails_closed(tmp_path):
    client = Pages({0: [['600000.SH', '20260102', 5.]]})
    with pytest.raises(CollectionError, match='max_pages'):
        collect_query(client, 'daily', {}, 'ts_code,trade_date,close', tmp_path, page_size=1, max_pages=1)
    assert not list(tmp_path.rglob('*.manifest.json'))


def test_range_query_cannot_silently_ignore_requested_dates(tmp_path):
    with pytest.raises(CollectionError, match='outside requested date range'):
        collect_query(Pages({0: [['600000.SH', '20260930', 5.]]}), 'daily', {'start_date': '20160101', 'end_date': '20161231'}, 'ts_code,trade_date,close', tmp_path)


def test_monthly_range_boundaries_are_calendar_exact():
    from ashare_quant.collect import monthly_ranges
    assert monthly_ranges('20240215', '20240305') == [
        {'start_date': '20240215', 'end_date': '20240229'},
        {'start_date': '20240301', 'end_date': '20240305'}]


def test_core_readiness_is_independent_of_optional_endpoint_failure(tmp_path):
    from ashare_quant.collect import audit_collection
    for endpoint in ('daily', 'daily_basic', 'adj_factor', 'stk_limit'):
        collect_query(Pages({0: [['600000.SH', '20260102', 5.]]}), endpoint, {'trade_date': '20260102'}, 'ts_code,trade_date,close', tmp_path)
    (tmp_path / 'namechange').mkdir()
    (tmp_path / 'namechange' / 'fixture.error.json').write_text(json.dumps({'endpoint': 'namechange', 'error': 'duplicate'}))
    report = audit_collection(tmp_path, expected_dates=['20260102'])
    assert report['core_status'] == 'complete'
    assert report['overall_status'] == 'partial'
    assert report['endpoints']['daily']['missing_dates'] == []
    assert report['optional_errors'][0]['endpoint'] == 'namechange'
    report = audit_collection(tmp_path, expected_dates=['20260102', '20260105'])
    assert report['core_status'] == 'incomplete'
    assert report['endpoints']['daily']['missing_dates'] == ['20260105']


def test_recovered_optional_error_does_not_poison_readiness(tmp_path):
    from ashare_quant.collect import audit_collection
    for endpoint in ('daily', 'daily_basic', 'adj_factor', 'stk_limit'):
        collect_query(Pages({0: [['600000.SH', '20260102', 5.]]}), endpoint, {'trade_date': '20260102'}, 'ts_code,trade_date,close', tmp_path)
    with pytest.raises(CollectionError):
        collect_query(Pages({0: []}), 'stock_st', {'trade_date': '20260102'}, 'ts_code,trade_date,close', tmp_path)
    collect_query(Pages({0: []}), 'stock_st', {'trade_date': '20260102'}, 'ts_code,trade_date,close', tmp_path, allow_empty=True)
    report = audit_collection(tmp_path, expected_dates=['20260102'])
    assert report['overall_status'] == 'complete'
    assert report['optional_errors'] == []
    assert len(report['resolved_optional_errors']) == 1


def test_client_can_pause_through_control_file_without_network_call(tmp_path):
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'paused': True}))
    sleeps, calls = [], []
    def sleep(seconds):
        sleeps.append(seconds)
        assert not calls
        control.write_text(json.dumps({'paused': False, 'requests_per_minute': 30}))
    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '{"status":"success","data":{"fields":[],"items":[]}}', '')
    client = HelperClient(runner=runner, sleep=sleep, control_path=control)
    client.query('daily', {}, '')
    assert len(calls) == 1 and sleeps == [0.5]
    assert client.interval == 2.


def test_listed_star_cdr_is_not_silently_dropped_from_market_universe():
    assert is_a_share('689009.SH')
