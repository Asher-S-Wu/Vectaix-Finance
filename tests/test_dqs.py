import pandas as pd
import pytest

from hk_quant.dqs import parse_daily_quotations


def page(body):
    return ('<html><pre>DATE: 04 JAN 2010 (MONDAY)<a name="quotations">QUOTATIONS</a>'
        '\nCODE NAME OF STOCK PRV.CLO./ ASK/ HIGH/ SHARES TRADED/\n CLOSING BID LOW TURNOVER ($)\n'+body+
        '<a name="sales_all">SALES RECORDS FOR ALL STOCKS</a>\n823 LINK REIT 999 999 999 999</pre></html>')


def test_actual_2010_link_two_line_layout_and_no_invented_open():
    text='   823 LINK REIT              19.82    19.60    19.70            1,945,905\n                              19.62    19.58    19.58           38,218,570\n'
    frame=parse_daily_quotations(page(text),['00823.HK'])
    row=frame.iloc[0]
    assert row.raw_close==19.62 and row.previous_close==19.82
    assert row.raw_high==19.70 and row.raw_low==19.58
    assert row.volume==1945905 and row.amount==38218570
    assert row.vwap==pytest.approx(38218570/1945905)
    assert row.date==pd.Timestamp('2010-01-04') and 'raw_open' not in frame


def test_missing_flow_is_not_reported_as_halt_or_zero_and_halt_is_explicit():
    text=('   823 LINK REIT              19.82    19.60     -                      -\n'
          '                              19.62    19.58     -                      -\n'
          '   778 FORTUNE REIT        TRADING SUSPENDED\n')
    frame=parse_daily_quotations(page(text),['00823.HK','00778.HK']).set_index('exchange_code')
    assert frame.loc['00823.HK','status']=='no_reported_trade'
    assert pd.isna(frame.loc['00823.HK','volume'])
    assert frame.loc['00778.HK','status']=='exchange_reported_suspension'


def test_broken_continuation_or_duplicate_cannot_create_quote():
    one='   823 LINK REIT              19.82    19.60    19.70            1,945,905\n'
    with pytest.raises(ValueError,match='续行'):parse_daily_quotations(page(one),['00823.HK'])
    full=one+'                              19.62    19.58    19.58           38,218,570\n'
    with pytest.raises(ValueError,match='重复'):parse_daily_quotations(page(full+full),['00823.HK'])


def test_ipo_or_ex_date_na_previous_close_keeps_actual_trade():
    text='   778 FORTUNE REIT             N/A     3.84     4.15           14,361,000\n                               3.84     3.83     3.84           57,331,660\n'
    row=parse_daily_quotations(page(text),['00778.HK']).iloc[0]
    assert pd.isna(row.previous_close) and row.raw_close==3.84
    assert row.status=='reported_trade' and row.volume==14361000


def test_starred_quote_row_is_preserved_with_its_source_marker():
    text='*  823 LINK REIT              19.82    19.60    19.70            1,945,905\n                              19.62    19.58    19.58           38,218,570\n'
    frame=parse_daily_quotations(page(text),['00823.HK'])
    assert len(frame)==1
    assert frame.iloc[0].quote_marker=='*' and frame.iloc[0].raw_close==19.62


def test_actual_attached_hash_marker_does_not_hide_suspended_reit():
    frame=parse_daily_quotations(page('   625#RREEF CCT REIT       TRADING SUSPENDED\n'),['00625.HK'])
    assert len(frame)==1 and frame.iloc[0].code_marker=='#'
    assert frame.iloc[0]['name']=='RREEF CCT REIT'
    assert frame.iloc[0].status=='exchange_reported_suspension'
