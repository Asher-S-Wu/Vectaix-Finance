import numpy as np
import pandas as pd

from hk_quant.reit_daily import build_reit_master, transform_reit_quotes, compare_dqs_2010


def inputs():
    listings=pd.DataFrame([{'stockExID':23,'longName':'SEHK Real Estate Investment Trusts'},{'stockExID':38,'longName':'SEHK Collective Investment Schemes'}])
    periods=pd.DataFrame([
        {'ID':1,'IssueID':4875,'StockCode':'0625','StockExID':23,'FirstTradeDate':'2007-06-22','FinalTradeDate':'2010-04-19','DelistDate':'2021-10-26','isin':'OLD_ISIN','2ndCtr':0},
        {'ID':2,'IssueID':28689,'StockCode':'1503','StockExID':38,'FirstTradeDate':'2019-12-10','FinalTradeDate':None,'DelistDate':None,'isin':'REIT_ISIN','2ndCtr':0}])
    issues=pd.DataFrame([{'ID1':4875,'issuer':1,'typeID':10},{'ID1':28689,'issuer':2,'typeID':10}])
    current=pd.DataFrame([{'Stock Code':'01503','ISIN':'REIT_ISIN','Name of Securities':'CMC REIT','Category':'Real Estate Investment Trusts'},
                          {'Stock Code':'00625','ISIN':'NEW_ISIN','Name of Securities':'NEW ISSUER','Category':'Equity'}])
    return periods,issues,current,listings


def quote(issue=4875,day='2010-01-04',**changes):
    row=dict(issueID=issue,atDate=day,prevClose=3.,closing=np.float32(3.23),ask=3.23,bid=3.22,high=3.25,low=3.22,vol=100,turn=323,susp=0,newsusp=0,noclose=0)
    row.update(changes);return row


def test_current_isin_reit_on_cis_board_and_old_code_namespace_are_kept_distinct():
    periods,issues,current,listings=inputs();master,gaps=build_reit_master(periods,issues,current,listings)
    assert master.set_index('issueID').loc[4875,'security_id']=='HKREIT:4875'
    assert master.set_index('issueID').loc[28689,'security_id']=='01503.HK'
    assert '00625.HK' not in set(master.security_id)
    assert not gaps


def test_zero_or_noclose_is_missing_price_but_flags_and_integers_are_preserved():
    periods,issues,current,listings=inputs();master,_=build_reit_master(periods,issues,current,listings)
    raw=pd.DataFrame([quote(closing=0.,noclose=1,susp=1,vol=0,turn=0),quote(day='2010-01-05',vol=9007199254740993,turn=9007199254740993),quote(day='2010-01-06',closing=3.,noclose=1)])
    result,gaps=transform_reit_quotes(raw,master,periods,listings)
    assert pd.isna(result.raw_close.iloc[0]) and pd.isna(result.raw_close.iloc[2])
    assert result.reported_close.iloc[0]==0 and result.susp.iloc[0]==1
    assert int(result.volume.iloc[1])==9007199254740993 and int(result.amount.iloc[1])==9007199254740993
    assert pd.isna(result.vwap.iloc[0]) and result.vwap.iloc[1]==1.
    assert result.exchange_code.iloc[0]=='00625.HK' and result.security_id.iloc[0]=='HKREIT:4875'
    assert 'open' not in result and not gaps


def test_overlapping_listing_periods_stay_ambiguous_not_first_match():
    periods,issues,current,listings=inputs();master,_=build_reit_master(periods,issues,current,listings)
    extra=periods.iloc[[0]].copy();extra['ID']=3;extra['StockCode']='9999';periods=pd.concat([periods,extra],ignore_index=True)
    result,gaps=transform_reit_quotes(pd.DataFrame([quote()]),master,periods,listings)
    assert pd.isna(result.exchange_code.iloc[0])
    assert result.identity_period_status.iloc[0]=='ambiguous_listing_period' and gaps


def test_date_gap_is_not_filled_between_listing_periods():
    periods,issues,current,listings=inputs();master,_=build_reit_master(periods,issues,current,listings)
    periods.loc[0,'DelistDate']='2010-02-01'
    result,gaps=transform_reit_quotes(pd.DataFrame([quote(day='2010-02-02')]),master,periods,listings)
    assert result.identity_period_status.iloc[0]=='outside_source_listing_period' and pd.isna(result.exchange_code.iloc[0])
    assert gaps


def test_parallel_quotes_have_no_claimed_normal_counter_code():
    periods,issues,current,listings=inputs();master,_=build_reit_master(periods,issues,current,listings)
    result,_=transform_reit_quotes(pd.DataFrame([quote()]),master,periods,listings,quote_table='pquotes')
    assert result.quote_table.eq('pquotes').all() and result.exchange_code.isna().all()
    assert result.identity_period_status.eq('parallel_counter_code_unresolved').all()


def test_dqs_comparison_respects_float32_but_reports_amount_and_suspension_differences():
    periods,issues,current,listings=inputs();master,_=build_reit_master(periods,issues,current,listings)
    daily,_=transform_reit_quotes(pd.DataFrame([quote()]),master,periods,listings)
    dqs=pd.DataFrame([{'date':pd.Timestamp('2010-01-04'),'exchange_code':'00625.HK','raw_close':3.23,'volume':100.,'amount':324.,'status':'exchange_reported_suspension','source_file':'official.htm'}])
    comparison,audit=compare_dqs_2010(daily,dqs)
    row=comparison.iloc[0]
    assert row.close_equal_float32 and row.volume_equal
    assert not row.amount_equal and not row.suspension_marker_equal
    assert audit['difference_counts']['amount']==1 and audit['difference_counts']['suspension_marker']==1


def test_zero_volume_does_not_invent_suspension_and_partial_day_trade_keeps_vwap():
    periods,issues,current,listings=inputs();master,_=build_reit_master(periods,issues,current,listings)
    raw=pd.DataFrame([quote(vol=0,turn=0,susp=0),quote(day='2010-01-05',susp=1,newsusp=1)])
    result,_=transform_reit_quotes(raw,master,periods,listings)
    assert result.susp.tolist()==[0,1]
    assert result.raw_close.notna().all()
    assert pd.isna(result.vwap.iloc[0]) and result.vwap.iloc[1]==3.23
