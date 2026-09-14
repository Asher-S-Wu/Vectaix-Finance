import pandas as pd
import pytest

from hk_quant.lot_snapshots import build_identity_index, parse_snapshot


def inputs():
    master = pd.DataFrame([
        dict(security_id='08233.HK', exchange_code='08233.HK', isin='OLD', asset_type='equity',
             identity_status='verified', identity_valid_from='2005-09-16', identity_valid_to='2018-01-29'),
        dict(security_id='01719.HK', exchange_code='01719.HK', isin='NEW', asset_type='equity',
             identity_status='verified', identity_valid_from='2018-01-29', identity_valid_to=None),
    ])
    listings = pd.DataFrame([
        dict(ID=1, IssueID=4627, StockCode='8233', StockExID=20, FirstTradeDate='2005-09-16',
             FinalTradeDate='2018-01-26', DelistDate='2018-01-29', isin=None),
        dict(ID=2, IssueID=4627, StockCode='1719', StockExID=1, FirstTradeDate='2018-01-29',
             FinalTradeDate=None, DelistDate=None, isin='NEW'),
    ])
    mapping = pd.DataFrame([dict(security_id='01719.HK', IssueID=4627, matched_isin='NEW',
                                 mapping_status='matched_unique_isin')])
    return master, listings, mapping


def page(day='2016-06-01', code='8233', issue=4627, lot='4,000', observation=None):
    observation = day if observation is None else observation
    return f'''<input type="date" name="d" value="{day}">
    <table><tr><th>Stock<br>Code</th><th>Issuer</th><th>Date</th><th>Board<br>lot</th></tr>
    <tr><td><a href="str.asp?i={issue}">{code}</a></td><td>Issuer</td><td>{observation}</td><td>{lot}</td></tr></table>'''


def parse(body, index, day='2016-06-01'):
    return parse_snapshot(body, pd.Timestamp(day), index, 'https://example.test/snapshot', '/raw/source.html.gz')


def test_old_counter_is_mapped_by_its_own_listing_period():
    result = parse(page(), build_identity_index(*inputs()))
    row = result.iloc[0]
    assert row.security_id == '08233.HK'
    assert row.source_issue_id == 4627 and row.source_listing_id == 1
    assert row.lot_size == 4000 and row.historical_value_date_verified
    assert row.source_observation_date == pd.Timestamp('2016-06-01')


def test_provider_issue_link_must_match_and_unresolved_row_is_kept():
    row = parse(page(issue=9999), build_identity_index(*inputs())).iloc[0]
    assert pd.isna(row.security_id)
    assert not row.identity_mapping_verified
    assert row.lot_size == 4000 and row.source_issue_id == 9999


def test_transfer_day_excludes_old_counter():
    row = parse(page('2018-01-29'), build_identity_index(*inputs()), '2018-01-29').iloc[0]
    assert pd.isna(row.security_id)


def test_new_counter_cannot_receive_prelisting_lot():
    row = parse(page(code='1719'), build_identity_index(*inputs())).iloc[0]
    assert pd.isna(row.security_id)


def test_date_selection_must_match_file_date():
    with pytest.raises(ValueError, match='日期'):
        parse(page('2016-06-02'), build_identity_index(*inputs()))


def test_same_code_with_different_isin_is_not_lifecycle_proof():
    master, listings, mapping = inputs()
    listings.loc[0, 'isin'] = 'ANOTHER'
    row = parse(page(), build_identity_index(master, listings, mapping)).iloc[0]
    assert pd.isna(row.security_id)


def test_unknown_and_invalid_lot_are_preserved():
    result = parse(page(lot='N/A'), build_identity_index(*inputs()))
    assert len(result) == 1 and pd.isna(result.lot_size.iloc[0])
    assert result.status.iloc[0] == 'invalid_lot'
    assert not result.historical_value_date_verified.iloc[0]


def test_missing_issuer_link_cannot_be_filled_from_code():
    body = page().replace('<a href="str.asp?i=4627">8233</a>', '8233')
    row = parse(body, build_identity_index(*inputs())).iloc[0]
    assert pd.isna(row.security_id) and not row.identity_mapping_verified


def test_two_valid_entities_do_not_select_one_arbitrarily():
    master, listings, mapping = inputs()
    other = master.iloc[[0]].copy()
    other['security_id'] = '08233!AA.HK'
    master = pd.concat([master, other], ignore_index=True)
    row = parse(page(), build_identity_index(master, listings, mapping)).iloc[0]
    assert pd.isna(row.security_id) and not row.identity_mapping_verified


def test_currency_total_is_not_a_security():
    body = page().replace('</table>', '<tr class="total"><td>HKD</td><td colspan="2">Total/average</td></tr></table>')
    assert len(parse(body, build_identity_index(*inputs()))) == 1


@pytest.mark.parametrize('observed,status', [('2016-05-31','stale_observation'),
                                           ('','missing_observation_date'),
                                           ('2016-06-02','future_observation')])
def test_only_same_day_observation_can_supply_a_current_lot(observed,status):
    row=parse(page(observation=observed),build_identity_index(*inputs())).iloc[0]
    assert row.status==status
    assert not row.historical_value_date_verified
    assert pd.isna(row.lot_size)
    assert row.source_lot_size==4000 and row.lot_literal=='4,000'
    assert row.source_observation_date_literal==observed


def test_missing_date_column_does_not_inherit_page_asof():
    body=page().replace('<th>Date</th>','').replace('<td>2016-06-01</td>','')
    row=parse(body,build_identity_index(*inputs())).iloc[0]
    assert row.status=='missing_observation_date'
    assert pd.isna(row.source_observation_date) and pd.isna(row.lot_size)


@pytest.mark.parametrize('day,observed,lot,status', [
    ('2022-06-29','2022-06-23','6,000','stale_observation'),
    ('2022-06-30','2022-06-23','6,000','stale_observation'),
    ('2022-07-04','2022-07-04','2,000','ok'),
])
def test_real_01049_halted_effective_date_and_resumption(day,observed,lot,status):
    # 原始行来自 raw/all_hk/20220629、20220630、20220704.html.gz。
    # 6月2日原公告及6月30日修订均明确6月29日生效；停牌页面仍显示6月23日观察。
    # 仅保留这些原行中参与本次解析的三列。
    body=f'''<input type="date" name="d" value="{day}"><table>
        <tr><th>Stock Code</th><th>Date</th><th>Board lot</th></tr>
        <tr><td><a href="str.asp?i=218">1049</a></td><td>{observed}</td><td>{lot}</td></tr></table>'''
    index={('01049',218):[dict(security_id='01049.HK',start=pd.Timestamp('1994-04-27'),end=None,
                              source_listing_id=579,identity_mapping_basis='source_issue_isin_same_counter')]}
    row=parse(body,index,day).iloc[0]
    assert row.status==status and row.source_observation_date==pd.Timestamp(observed)
    assert row.historical_value_date_verified==(status=='ok')
    assert row.lot_size==2000 if status=='ok' else pd.isna(row.lot_size)
    assert row.source_lot_size==int(lot.replace(',',''))
