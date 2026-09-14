import json
from pathlib import Path

import pandas as pd
import pytest

from hk_quant.hkex import (
    discover,
    import_actions_csv,
    import_board_lot_csv,
    infer_lot_history,
    parse_board_lot_text,
    parse_dividend_form_text,
    parse_search_html,
)
from hk_quant.paths import DATA


SAMPLE_PAGE = Path("data/hk/universal/references/settlement_research/") / (
    "hkexnews_boardlot_category_current_2024_page.html"
)


def test_all_category_stock_search_is_sent_to_server_and_scoped_cache(tmp_path):
    from types import SimpleNamespace
    html=b"<div>Total records found: 1</div><table><tr><td>Release Time: 02/01/2024 18:30</td><td>Stock Code: 00405</td><td>Stock Short Name: YUEXIU</td><td><a href='/listedco/listconews/sehk/2024/0102/example.pdf'>DISTRIBUTION PER UNIT</a></td></tr></table>"
    class Session:
        def __init__(self):self.forms=[]
        def request(self,url,data=None,**kwargs):
            if data is not None:self.forms.append(data)
            return SimpleNamespace(body=html,url=url)
    session=Session()
    run=discover('all','20240102','20240102','00405.HK',tmp_path,session,stock_id='10324')
    assert run['status']=='complete'
    assert all(f['stockId']=='10324' and f['t1code']==f['t2Gcode']==f['t2code']=='' for f in session.forms)
    different=Session()
    discover('all','20240102','20240102','00405.HK',tmp_path,different,stock_id='10325')
    assert len(different.forms)==2
    wrong=discover('all','20240102','20240102','00823.HK',tmp_path,Session(),stock_id='10148')
    assert wrong['status']=='failed'
    assert wrong['discovered_records']==0


def test_default_storage_root_does_not_duplicate_universal_segment():
    assert discover.__defaults__[1] == DATA / "hkex"


def test_saved_official_search_page_preserves_count_timezone_and_pdf_url():
    parsed = parse_search_html(SAMPLE_PAGE.read_text(encoding="utf-8"))

    assert parsed["reported_count"] == 160
    assert len(parsed["records"]) == 100
    first = parsed["records"][0]
    assert first["published_at"].endswith("+08:00")
    assert first["stock_codes"] == ["01327"]
    assert first["pdf_url"].startswith("https://www1.hkexnews.hk/")


def test_search_page_count_mismatch_is_not_accepted_as_complete():
    parsed = parse_search_html(SAMPLE_PAGE.read_text(encoding="utf-8"))
    assert parsed["complete"] is False
    assert parsed["reported_count"] > len(parsed["records"])


def test_discover_reconciles_counts_and_reuses_successful_cache(tmp_path):
    html = b"""<div>Total records found: 1</div><table><tr>
    <td>Release Time: 02/01/2024 18:30</td><td>Stock Code: 00005</td>
    <td>Stock Short Name: HSBC HOLDINGS</td><td><a
    href='/listedco/listconews/sehk/2024/0102/example.pdf'>CHANGE IN BOARD LOT SIZE</a></td>
    </tr></table>"""

    class Response:
        body = html
        url = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en"

    class Session:
        def __init__(self):
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    result = discover("board_lot", "20240102", "20240102", root=tmp_path, session=session)
    assert result["status"] == "complete"
    assert session.calls == 4  # landing page and search POST, for current and delisted
    assert result["windows"][0]["reported_count"] == 1

    class MustNotRequest:
        def request(self, *args, **kwargs):
            raise AssertionError("successful cached windows must not be downloaded again")

    cached = discover("board_lot", "20240102", "20240102", root=tmp_path, session=MustNotRequest())
    assert cached["status"] == "complete"


def test_discover_load_more_uses_hkex_compact_dates_and_exact_count(tmp_path):
    initial = b"""<div>Total records found: 2</div><table><tr>
    <td>Release Time: 02/01/2024 18:30</td><td>Stock Code: 00005</td>
    <td>Stock Short Name: HSBC</td><td><a href='/listedco/a.pdf'>A</a></td>
    </tr></table>"""
    ajax_rows = [{
        "DATE_TIME": f"0{i + 2}/01/2024 18:30", "STOCK_CODE": f"{i + 5:05d}",
        "STOCK_NAME": f"S{i}", "TITLE": f"T{i}", "FILE_LINK": f"/listedco/{i}.pdf",
        "SHORT_TEXT": "Change in Board Lot Size", "FILE_TYPE": "PDF", "FILE_INFO": "1KB",
        "TOTAL_COUNT": "2",
    } for i in range(2)]
    ajax = json.dumps({"result": json.dumps(ajax_rows), "recordCnt": 2}).encode()

    class Response:
        def __init__(self, body):
            self.body = body
            self.url = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en"

    class Session:
        def request(self, url, data=None, referer=None):
            if "titleSearchServlet" in url:
                assert "fromDate=20240102" in url and "toDate=20240102" in url
                return Response(ajax)
            return Response(initial if data is not None else b"landing")

    result = discover("board_lot", "20240102", "20240102", root=tmp_path, session=Session())
    assert result["status"] == "complete"
    assert result["discovered_records"] == 2


def test_empty_or_restricted_discovery_is_persisted_as_failure(tmp_path):
    class EmptyResponse:
        body = b""
        url = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en"

    class Session:
        def request(self, *args, **kwargs):
            return EmptyResponse()

    result = discover("board_lot", "20240102", "20240102", root=tmp_path, session=Session())
    assert result["status"] == "failed"
    assert len(result["failures"]) == 2
    saved = json.loads(next((tmp_path / "runs").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["status"] == "failed"


def test_board_lot_proposal_is_never_enabled():
    candidate = parse_board_lot_text(
        "The Board proposes to change the board lot size from 2,000 shares to 10,000 shares.",
        published_at="2024-01-02T18:00:00+08:00",
        source_url="https://www1.hkexnews.hk/example.pdf",
    )
    assert candidate["verified"] is False
    assert candidate["evidence_status"] == "needs_review"
    assert candidate["effective_date"] is None


def test_effective_board_lot_candidate_requires_all_three_facts():
    candidate = parse_board_lot_text(
        "The board lot size was changed from 20,000 Existing Shares to 10,000 "
        "Consolidated Shares and became effective on 15 April 2024.",
        published_at="2024-09-27T18:00:00+08:00",
        source_url="https://www1.hkexnews.hk/listedco/example.pdf",
    )
    assert candidate["old_lot"] == 20_000
    assert candidate["new_lot"] == 10_000
    assert candidate["effective_date"] == "2024-04-15"
    assert candidate["evidence_status"] == "candidate_effective"


def test_dividend_final_schedule_is_explicitly_simulated_not_actual_receipt():
    candidate = parse_dividend_form_text(
        "Status Updated Final cash dividend. Dividend declared HKD 0.42 per share. "
        "Payment date 26 September 2025.",
        published_at="2025-09-15T18:00:00+08:00",
        source_url="https://www1.hkexnews.hk/final.pdf",
    )
    assert candidate["payment_date"] == "2025-09-26"
    assert candidate["cash_currency"] == "HKD"
    assert candidate["cash_per_share"] == pytest.approx(.42)
    assert candidate["payment_basis"] == "issuer_final_schedule_simulated"
    assert candidate["evidence_status"] == "candidate_final_schedule"


@pytest.mark.parametrize("phrase", [
    "Payment date To be announced",
    "The proposed payment date is conditional on shareholder approval",
    "Cheques will be despatched on or before 26 September 2025",
])
def test_uncertain_or_cheque_only_date_remains_a_gap(phrase):
    candidate = parse_dividend_form_text(
        f"Final dividend HKD 0.42 per share. {phrase}.",
        published_at="2025-07-30T18:00:00+08:00",
        source_url="https://www1.hkexnews.hk/unresolved.pdf",
    )
    assert candidate["payment_date"] is None
    assert candidate["verified"] is False
    assert candidate["evidence_status"] == "needs_review"


def test_verified_csv_imports_require_facts_units_uniqueness_and_basis(tmp_path):
    lots = tmp_path / "lots.csv"
    pd.DataFrame([{
        "security_id": "A", "announced_at": "2024-01T18:00:00+08:00",
        "effective_date": "2024-04-15", "old_lot": 20000, "new_lot": 10000,
        "verified": True, "source_url": "https://www1.hkexnews.hk/a.pdf",
    }]).to_csv(lots, index=False)
    lot_result = import_board_lot_csv(lots)
    assert lot_result["status"] == "complete"
    assert lot_result["events"][0]["new_lot"] == 10000

    actions = tmp_path / "actions.csv"
    row = {
        "security_id": "A", "announced_at": "2025-09-15T18:00:00+08:00",
        "effective_date": "2025-08-14", "payment_date": "2025-09-26",
        "cash_currency": "HKD", "cash_per_share": .42, "action_type": "cash_dividend",
        "share_multiplier": 1.0, "verified": True,
        "payment_basis": "issuer_final_schedule_simulated",
        "source_url": "https://www1.hkexnews.hk/d.pdf",
    }
    pd.DataFrame([row]).to_csv(actions, index=False)
    assert import_actions_csv(actions)["events"][0]["payment_date"] == "2025-09-26"

    pd.DataFrame([row, row]).to_csv(actions, index=False)
    with pytest.raises(ValueError, match="重复"):
        import_actions_csv(actions)

    row["payment_basis"] = "actual_receipt"
    pd.DataFrame([row]).to_csv(actions, index=False)
    with pytest.raises(ValueError, match="payment_basis"):
        import_actions_csv(actions)


def test_lot_history_only_reverses_a_complete_chain():
    anchors = pd.DataFrame([{"security_id": "A", "current_lot": 100}])
    changes = pd.DataFrame([
        {"security_id": "A", "effective_date": "2023-01-03", "old_lot": 500, "new_lot": 100, "verified": True},
        {"security_id": "A", "effective_date": "2020-01-02", "old_lot": 1000, "new_lot": 500, "verified": True},
    ])
    result = infer_lot_history(anchors, changes, {"A": {"complete": True, "unresolved_count": 0}})
    assert result["status"] == "complete"
    assert [period["lot_size"] for period in result["histories"]["A"]] == [1000, 500, 100]

    broken = changes.copy()
    broken.loc[0, "new_lot"] = 200
    result = infer_lot_history(anchors, broken, {"A": {"complete": True, "unresolved_count": 0}})
    assert result["status"] == "incomplete"
    assert result["histories"]["A"] == []
    assert "链" in result["gaps"][0]["reason"]


def test_lot_history_rejects_unresolved_announcements_even_when_chain_matches():
    anchors = pd.DataFrame([{"security_id": "A", "current_lot": 100}])
    changes = pd.DataFrame([
        {"security_id": "A", "effective_date": "2023-01-03", "old_lot": 500, "new_lot": 100, "verified": True},
    ])
    result = infer_lot_history(anchors, changes, {"A": {"complete": True, "unresolved_count": 1}})
    assert result["status"] == "incomplete"
    assert "未解决" in result["gaps"][0]["reason"]


def test_download_preserves_word_original_instead_of_rejecting_it_as_non_pdf(tmp_path):
    from hk_quant.hkex import download, document_path, _Response
    import json
    root=tmp_path
    (root/'catalog').mkdir()
    urls=['https://www1.hkexnews.hk/example.pdf','https://www1.hkexnews.hk/example.doc']
    records=[{'pdf_url':url,'stock_codes':['00001']} for url in urls]
    (root/'catalog/board_lot.json').write_text(json.dumps({'records':records}))
    class Session:
        def request(self,url):
            body=b'%PDF-example' if url.endswith('.pdf') else bytes.fromhex('d0cf11e0a1b11ae1')+b'word-original'
            return _Response(body,url)
    result=download('board_lot',root=root,session=Session())
    assert result['status']=='complete' and result['downloaded']==2
    assert document_path(root,'board_lot',urls[1]).read_bytes().endswith(b'word-original')
    assert document_path(root,'board_lot',urls[0]).parent==root/'pdf/board_lot'
    assert document_path(root,'board_lot',urls[1]).parent==root/'documents/board_lot'


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('lot_section,other_section,expected',[
    ('''CHANGE IN BOARD LOT SIZE BECOMING EFFECTIVE ON 3 JANUARY 2025
As disclosed in the Circular, the Change in Board Lot Size will become effective on Friday,
3 January 2025. The board lot size of the Shares for trading on the Stock Exchange will be
changed from 2,000 Existing Shares to 10,000 Consolidated Shares (or 10,000 New Shares,
subject to the Capital Reduction and the Sub-division becoming effective).''',
     '''SHARE CONSOLIDATION BECOMING EFFECTIVE ON 17 DECEMBER 2024
As all the conditions of the Share Consolidation as stated in the Circular have been fulfilled,
the Share Consolidation will become effective on Tuesday, 17 December 2024.''',
     ('2025-01-03',2000,10000)),
    ('''CHANGE IN BOARD LOT SIZE BECOMING EFFECTIVE ON 27 AUGUST 2024
As disclosed in the EGM Circular, the Change in Board Lot Size will become
effective on Tuesday, 27 August 2024. The board lot size of the Shares for trading
on the Stock Exchange will be changed from 2,500 Existing Shares to 5,000
Consolidated Shares (or 5,000 New Shares, subject to the Capital Reduction and the
Share Sub-division becoming effective).''',
     '''SHARE CONSOLIDATION BECOMING EFFECTIVE ON 13 AUGUST 2024
As all the conditions of the Share Consolidation as stated in the EGM Circular have
been fulfilled, the Share Consolidation will become effective on Tuesday, 13 August
2024. The par value of each issued Consolidated Share reduced from HK$0.25 to HK$0.01.''',
     ('2024-08-27',2500,5000)),
])
def test_actual_multi_action_excerpts_bind_date_to_board_lot(reverse,lot_section,other_section,expected):
    sections=[other_section,lot_section]
    if reverse:sections.reverse()
    result=parse_board_lot_text('\n\n'.join(sections),'2024-01-02T18:00:00+08:00','https://www1.hkexnews.hk/announcement.pdf')
    assert (result['effective_date'],result['old_lot'],result['new_lot'])==expected
    assert result['verified'] is False and result['evidence_status']=='candidate_effective'


@pytest.mark.parametrize('text',[
    'The board lot size will change from 2,000 Shares to 10,000 Shares. Share consolidation became effective on 17 December 2024.',
    'CHANGE IN BOARD LOT SIZE. The share consolidation from 25 Shares to 1 Share became effective on 17 December 2024.',
    'The board lot size became effective on 3 January 2025. Capital value was reduced from 25 to 1.',
    'The board lot size was changed from HK$0.25 to HK$0.01 and became effective on 3 January 2025.',
    'The board lot size was changed from 2,000 Shares to 10,000 Shares and became effective on 3 January 2025. The board lot size became effective on 7 January 2025.',
    'The board lot size became effective on 3 January 2025. The board lot size changed from 2,000 Shares to 10,000 Shares. The board lot size changed from 5,000 Shares to 20,000 Shares.',
    'The proposed board lot size will change from 2,000 Shares to 10,000 Shares and will become effective on 3 January 2025 subject to shareholder approval.',
])
def test_unbound_capital_or_conflicting_board_lot_facts_need_review(text):
    result=parse_board_lot_text(text,'2024-01-02T18:00:00+08:00','https://www1.hkexnews.hk/announcement.pdf')
    assert result['evidence_status']=='needs_review'
    assert result['verified'] is False
    assert result['effective_date'] is None and result['old_lot'] is None and result['new_lot'] is None


def test_consolidation_short_name_cannot_supply_board_lot_date_or_ratio():
    text=('The board lot size will change after consolidation from 25 Shares to 1 Share, '
          'becoming effective on 17 December 2024.')
    candidate=parse_board_lot_text(text,'2024-12-13T18:00:00+08:00','https://www1.hkexnews.hk/announcement.pdf')
    assert candidate['evidence_status']=='needs_review' and candidate['effective_date'] is None


MULTI_CODE_ROWS=[
 '03008<br/>03009<br/>09008<br/>09009','02461<br/>02562','00291<br/>80291',
 '01109<br/>05786<br/>05819','04841 07841','00616 01218','03199<br/>83199',
 '01211<br/>81211','03437<br/>09437<br/>83437','03008<br/>03009<br/>09008<br/>09009',
 '02814<br/>03132<br/>09814','03187<br/>09187']


def test_blank_official_index_code_preserves_announcement_without_guessing_from_title():
    from hk_quant.hkex import _parse_ajax_response,_select_security
    row={'DATE_TIME':'24/11/2014 18:31','STOCK_CODE':'','STOCK_NAME':'',
         'TITLE':'83139, 3139 Dividend Announcement','FILE_LINK':'/listedco/listconews/sehk/2014/1124/ltn20141124412.pdf'}
    parsed=_parse_ajax_response(json.dumps({'result':json.dumps([row]),'recordCnt':1}))
    assert parsed['complete'] and parsed['parsed_count']==1
    assert parsed['records'][0]['stock_codes']==[]
    assert parsed['unassigned_announcements']==1
    assert _select_security(parsed['records'],'03139.HK')==[]
    with pytest.raises(ValueError):_select_security(parsed['records'],'')


def test_blank_html_code_keeps_original_document_row_count():
    html='<div>Total records found: 1</div><table><tr><td>Release Time: 20/03/2014 18:22</td><td>Stock Code: </td><td>Stock Short Name: </td><td><a href="/listedco/notice.pdf">NOTICE</a></td></tr></table>'
    parsed=parse_search_html(html)
    assert parsed['complete'] and parsed['unassigned_announcements']==1
    assert parsed['records'][0]['stock_code_status']=='unassigned'


@pytest.mark.parametrize('raw',MULTI_CODE_ROWS)
def test_actual_multicode_index_rows_preserve_all_associations(raw):
    from hk_quant.hkex import _parse_ajax_response
    expected=raw.replace('<br/>',' ').split()
    row={'DATE_TIME':'02/01/2024 18:30','STOCK_CODE':raw,'STOCK_NAME':'Names',
         'TITLE':'Notice','FILE_LINK':'/listedco/test.pdf'}
    ajax=_parse_ajax_response(json.dumps({'result':json.dumps([row]),'recordCnt':1}))
    html=parse_search_html('<div>Total records found: 1</div><table><tr><td>Release Time: 02/01/2024 18:30</td><td>Stock Code: '+raw+'</td><td>Stock Short Name: Names</td><td><a href="/listedco/test.pdf">Notice</a></td></tr></table>')
    for parsed in (ajax,html):
        assert parsed['complete'] is True and parsed['reported_count']==1
        assert parsed['parsed_count']==1 and parsed['security_associations']==len(expected)
        assert parsed['records'][0]['stock_codes']==expected
        assert 'stock_code' not in parsed['records'][0]


def test_same_document_version_merges_codes_without_miscounting_source_rows():
    from hk_quant.hkex import _parse_ajax_response
    rows=[{'DATE_TIME':time,'STOCK_CODE':code,'STOCK_NAME':'N','TITLE':'Notice','FILE_LINK':'/listedco/shared.pdf'}
          for time,code in [('02/01/2024 18:30','00291'),('02/01/2024 18:30','80291'),('03/01/2024 18:30','29291')]]
    result=_parse_ajax_response(json.dumps({'result':json.dumps(rows),'recordCnt':3}))
    assert result['complete'] and result['parsed_count']==3
    assert len(result['records'])==2 and result['security_associations']==3
    assert result['records'][0]['stock_codes']==['00291','80291']
    assert result['records'][1]['stock_codes']==['29291']


def test_every_index_code_is_available_to_security_filter():
    from hk_quant.hkex import _select_security
    records=[{'stock_codes':['03008','03009','09008','09009'],'pdf_url':'shared.pdf'}]
    for code in records[0]['stock_codes']:
        assert _select_security(records,code+'.HK')==records
    assert _select_security(records,'00001.HK')==[]


def test_multisecurity_extracted_fact_remains_unassigned_review_candidate(tmp_path):
    from hk_quant.hkex import document_path,extract_cached
    url='https://www1.hkexnews.hk/listedco/test.pdf'
    (tmp_path/'catalog').mkdir();(tmp_path/'catalog/board_lot.json').write_text(json.dumps({'records':[{'published_at':'2024-01-01T18:00:00+08:00','stock_codes':['00291','80291'],'title':'Joint notice','pdf_url':url}]}))
    original=document_path(tmp_path,'board_lot',url);original.parent.mkdir(parents=True);original.write_bytes(b'%PDF cached fixture')
    textpath=tmp_path/'extracted/board_lot'/f'{original.stem}.txt';textpath.parent.mkdir(parents=True)
    textpath.write_text('The board lot size was changed from 2,000 Shares to 10,000 Shares and became effective on 3 January 2024.')
    extract_cached('board_lot',tmp_path)
    row=json.loads((textpath.parent/'candidates.json').read_text())['records'][0]
    assert row['stock_codes']==['00291','80291'] and 'stock_code' not in row
    assert row['verified'] is False and row['evidence_status']=='needs_review'
    assert row['index_association_only'] is True


def test_discovery_security_filter_includes_secondary_code_and_reparses_raw_cache(tmp_path):
    body=b'<div>Total records found: 1</div><table><tr><td>Release Time: 02/01/2024 18:30</td><td>Stock Code: 00291<br/>80291</td><td>Stock Short Name: N</td><td><a href="/listedco/test.pdf">Notice</a></td></tr></table>'
    class Response:
        url='https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en'
        def __init__(self):self.body=body
    class Session:
        def request(self,*args,**kwargs):return Response()
    first=discover('board_lot','20240102','20240102',security='80291.HK',root=tmp_path,session=Session())
    assert first['status']=='complete' and first['discovered_records']==1
    class NoNetwork:
        def request(self,*args,**kwargs):raise AssertionError('cached source must be reused')
    second=discover('board_lot','20240102','20240102',security='80291.HK',root=tmp_path,session=NoNetwork())
    assert second['status']=='complete'
    row=json.loads(Path(second['catalog_path']).read_text())['records'][0]
    assert row['stock_codes']==['00291','80291'] and 'stock_code' not in row


def test_download_security_filter_selects_any_associated_code(tmp_path):
    from hk_quant.hkex import download,_Response
    (tmp_path/'catalog').mkdir()
    (tmp_path/'catalog/board_lot.json').write_text(json.dumps({'records':[{'stock_codes':['00291','80291'],'pdf_url':'https://www1.hkexnews.hk/listedco/a.pdf'}]}))
    class Session:
        def request(self,url):return _Response(b'%PDF local fixture',url)
    first=download('board_lot',security='80291.HK',root=tmp_path,session=Session())
    second=download('board_lot',security='00291.HK',root=tmp_path,session=Session())
    assert first['selected']==1 and first['downloaded']==1
    assert second['selected']==1 and second['cached']==1


LOCAL_EFFECT_STATEMENTS = [('https://www1.hkexnews.hk/listedco/listconews/sehk/2016/1223/ltn20161223742.pdf', '2016-12-23T19:30:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size of the Shares for trading on the Stock Exchange will be changed from 2,000 Shares to 1,000 Shares with effect from 9:00 a.m. on Thursday, 19 January 2017', 2000, 1000, '2017-01-19'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2016/1012/ltn20161012562.pdf', '2016-10-12T19:20:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size of the Shares for trading on the Stock Exchange will be changed from 10,000 Shares to 2,000 Shares with effect from 9:00 a.m. on Wednesday, 2 November 2016', 10000, 2000, '2016-11-02'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2016/0909/ltn20160909945.pdf', '2016-09-09T19:14:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size of the Shares for trading on the Stock Exchange will be changed from 10,000 Shares to 2,000 Shares with effect from 9:00 a.m. on Monday, 3 October 2016', 10000, 2000, '2016-10-03'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2017/1227/ltn20171227579.pdf', '2017-12-27T18:12:00+08:00', 'BOARD LOT SIZE The Board wishes to announce that the board lot size of the Shares for trading on the Stock Exchange will be changed from 1,000 Shares to 100 Shares with effect from 9:00 a.m. on Friday, 19 January 2018', 1000, 100, '2018-01-19'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2017/1127/ltn20171127335.pdf', '2017-11-27T17:42:00+08:00', 'BOARD LOT SIZE In order to improve the liquidity of the Shares and broaden the Company’s shareholder base, the Board announces that the board lot size of the Shares for trading on the Stock Exchange will be changed from 2,000 Shares each to 200 Shares each with effect from 9:00 a.m. on Monday, 18 December 2017', 2000, 200, '2017-12-18'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2017/1019/ltn20171019487.pdf', '2017-10-19T17:34:00+08:00', 'BOARD LOT SIZE In order to improve the liquidity of the Shares and broaden the Company’s shareholder base, the Board announces that the board lot size of the Shares for trading on the Stock Exchange will be changed from 2 0,000 Shares to 5,000 Shares with effect from 9:00 a.m. on Thursday, 9 November 2017', None, None, None), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2018/1220/ltn20181220283.pdf', '2018-12-20T12:19:00+08:00', 'board lot size of the ordinary shares in the share capital of the Company (the “Shares”) for trading on the Main Board of The Stock Exchange of Hong Kong Limited (the “Stock Exchange”) will be changed from 1,000 Shares to 100 Shares with effect from 9:00 a.m. on Tuesday, 15 January 2019', 1000, 100, '2019-01-15'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2018/1029/ltn201810291289.pdf', '2018-10-29T21:38:00+08:00', 'board lot size of the ordinary shares in the share capital of the Company (the “Shares”) for trading on the Main Board of The Stock Exchange of Hong Kong Limited (the “Stock Exchange”) will be changed from 1,500 Shares to 500 Shares with effect from 9:00 a.m. on Monday, 19 November 2018', 1500, 500, '2018-11-19'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2018/1024/ltn20181024699.pdf', '2018-10-24T20:42:00+08:00', 'board lot size of the Shares for trading on the Main Board of the Stock Exchange will be reduced from 20,000 Shares to 2,000 Shares with effect from 9:00 a.m. on Wednesday, 14 November 2018', 20000, 2000, '2018-11-14'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2019/1218/2019121800787.pdf', '2019-12-18T19:03:00+08:00', 'board lot size of the ordinary shares in the share capital of the Company (the “Shares”) for trading on the Main Board of The Stock Exchange of Hong Kong Limited (the “ Stock Exchange ”) will be changed from 5,000 Shares to 2,500 Shares with effect from 9:00 a.m. on Thursday, 16 January 2020', 5000, 2500, '2020-01-16'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2019/1113/2019111300472.pdf', '2019-11-13T12:11:00+08:00', 'board lot size of the units of Link (the Units) for trading on The Stock Exchange of Hong Kong Limited (the Stock Exchange) will be changed from 500 Units to 100 Units with effect from 9:00 a.m. on Thursday, 2 January 2020', 500, 100, '2020-01-02'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2019/0829/ltn20190829702.pdf', '2019-08-29T19:22:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size for trading in the H Shares on the Stock Exchange will be changed from 1,000 H Shares to 200 H Shares with effect from 9:00 a.m. on Thursday, 19 September 2019', 1000, 200, '2019-09-19'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2020/0730/2020073001148.pdf', '2020-07-30T17:54:00+08:00', 'board lot size of the ordinary shares in the Company (the “ Shares ”) for trading on the Main Board of The Stock Exchange of Hong Kong Limited (the “ Stock Exchange ”) will be changed from 2,000 Shares to 500 Shares with effect from 9:00 a.m. on Friday, 21 August 2020', 2000, 500, '2020-08-21'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2020/0715/2020071500405.pdf', '2020-07-15T16:35:00+08:00', 'board lot size of the ordinary shares in the Company (the ‘‘Shares ’’) for trading on the Main Board of The Stock Exchange of Hong Kong Limited (the ‘‘Stock Exchange ’’) will be changed from 5,000 Shares to 500 Shares with effect from 9:00 a.m. on Thursday, 6 August 2020', 5000, 500, '2020-08-06'), ('https://www1.hkexnews.hk/listedco/listconews/gem/2021/1115/2021111501251.pdf', '2021-11-15T20:34:00+08:00', 'board lot size of the ordinary shares in the Company (the “Shares”) for trading on The Stock Exchange of Hong Kong Limited (the “Stock Exchange”) will be changed from 6,000 Shares to 1,500 Shares with effect from 9:00 a.m. on Wednesday, 8 December 2021', 6000, 1500, '2021-12-08'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2021/1012/2021101200717.pdf', '2021-10-12T17:05:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size for trading in the Shares on the Stock Exchange will be changed from 4,000 Shares to 400 Shares with effect from 9:00 a.m. on Friday, 5 November 2021', 4000, 400, '2021-11-05'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2022/1129/2022112900694.pdf', '2022-11-29T17:46:00+08:00', 'board lot size of the ordinary shares in the Company (the “Shares”) for trading on The Stock Exchange of Hong Kong Limited (the “Stock Exchange ”) will be changed from 20,000 Shares to 4,000 Shares with effect from 9:00 a.m. on Tuesday, 20 December 2022', 20000, 4000, '2022-12-20'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2022/1101/2022110103656.pdf', '2022-11-01T20:54:00+08:00', 'board lot size of the ordinary shares in the Company (the “Shares ”) for trading on the Main Board of The Stock Exchange of Hong Kong Limited (the “Stock Exchange ”) will be changed from 2,000 Shares to 1,000 Shares with effect from 9:00 a.m. on Tuesday, November 22, 2022', 2000, 1000, '2022-11-22'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2023/1213/2023121300945.pdf', '2023-12-13T20:08:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size for trading of the Shares on the Stock Exchange will be changed from 400 Shares each to 100 Shares each with effect from 9:00 a.m. on Monday, 8 January 2024', 400, 100, '2024-01-08'), ('https://www1.hkexnews.hk/listedco/listconews/sehk/2023/1207/2023120700852.pdf', '2023-12-07T18:50:00+08:00', 'BOARD LOT SIZE The Board announces that the board lot size of the ordinary shares of US$0.0001 each in the Shares for trading on the Stock Exchange will be changed from 100 Shares to 3,000 Shares with effect from 9:00 a.m. on Wednesday, 3 January 2024', 100, 3000, '2024-01-03')]

@pytest.mark.parametrize('url,published,text,old,new,date',LOCAL_EFFECT_STATEMENTS)
def test_local_2016_2023_effect_statements_and_source_excerpt(url,published,text,old,new,date):
    candidate=parse_board_lot_text(text,published,url)
    assert (candidate['old_lot'],candidate['new_lot'],candidate['effective_date'])==(old,new,date)
    assert candidate['verified'] is False
    if date:
        assert candidate['evidence_status']=='candidate_effective'
        assert candidate['evidence_sentences']
        assert all(sentence in text for sentence in candidate['evidence_sentences'])
    else:
        assert candidate['evidence_status']=='needs_review'


@pytest.mark.parametrize('text',[
    'The Board proposes that the board lot size will be changed from 4,000 Shares to 20,000 Shares with effect from Friday, 10 March 2017.',
    'The board lot size will be changed from 4,000 Shares to 20,000 Shares with effect from 9:00 a.m. on Friday, 10 March 2017, subject to completion of the Rights Issue.',
    'The board lot size is proposed to be changed from 500 H Shares to 1,000 H Shares with effect from 9:00 a.m. on Thursday, 19 September 2019.',
    'The board lot size will change from 500 Units to 100 Units. The appointment of directors will take effect from 2 January 2020.',
])
def test_proposed_conditional_or_other_event_effect_is_not_enabled(text):
    candidate=parse_board_lot_text(text,'2016-12-01T18:00:00+08:00','https://www1.hkexnews.hk/example.pdf')
    assert candidate['evidence_status']=='needs_review' and candidate['effective_date'] is None


def test_unrelated_reference_to_proposed_change_does_not_override_final_effect_statement():
    url,published,statement,old,new,date=next(case for case in LOCAL_EFFECT_STATEMENTS if case[0].endswith('2019121800787.pdf'))
    context=('The Company does not have any intention to carry out other corporate actions or arrangements '
             'in the next 12 months that may affect the trading in the Shares or have any undermining or '
             'negating the intended purpose of the proposed change in the board lot size;')
    candidate=parse_board_lot_text(statement+'. '+context,published,url)
    assert candidate['evidence_status']=='candidate_effective'
    assert (candidate['old_lot'],candidate['new_lot'],candidate['effective_date'])==(old,new,date)


def test_general_dividend_category_uses_13250_and_local_dividend_parser(tmp_path):
    from datetime import date
    from hk_quant.hkex import _search_form,document_path,extract_cached
    form=_search_form('dividend',date(2016,1,1),date(2016,12,31),'current')
    assert (form['t1code'],form['t2Gcode'],form['t2code'])==('10000','3','13250')
    url='https://www1.hkexnews.hk/listedco/dividend.pdf'
    (tmp_path/'catalog').mkdir()
    (tmp_path/'catalog/dividend.json').write_text(json.dumps({'records':[{'stock_codes':['00001'],'published_at':'2025-09-15T18:00:00+08:00','pdf_url':url,'title':'Dividend'}]}))
    original=document_path(tmp_path,'dividend',url);original.parent.mkdir(parents=True);original.write_bytes(b'%PDF local fixture')
    text=tmp_path/'extracted/dividend'/f'{original.stem}.txt';text.parent.mkdir(parents=True)
    text.write_text('Status Updated Final cash dividend. Dividend declared HKD 0.42 per share. Payment date 26 September 2025.')
    extract_cached('dividend',tmp_path)
    candidate=json.loads((text.parent/'candidates.json').read_text())['records'][0]
    assert candidate['payment_date']=='2025-09-26' and candidate['verified'] is False
