from decimal import Decimal
import numpy as np
import pandas as pd
from hk_quant.reit_minute_daily import aggregate_minutes, map_minute_identity, compare_with_ccass, build_panel_candidate, frequency_for_date


def minutes():
    return pd.DataFrame({'ts_code':['00823.HK']*3,'trade_time':['2015-09-10 09:30:00','2015-09-10 12:00:00','2015-09-10 16:00:00'],
        'open':[10.,10.2,10.1],'high':[10.2,10.3,10.4],'low':[9.9,10.,10.1],'close':[10.1,10.1,10.3],
        'vol':[100,-20,50],'amount':[1000.,-200.,515.]})


def test_minute_only_valid_high_low_reach_factor_input_panel():
    source=minutes().assign(trade_time=['2026-09-10 09:30:00','2026-09-10 12:00:00','2026-09-10 16:00:00'],
                            vol=[100,20,50],amount=[1000.,200.,515.])
    minute=map_minute_identity(aggregate_minutes(source,'60min','source.parquet'),*identities())
    ccass=pd.DataFrame(columns=['issueID','security_id','date','exchange_code','raw_close','volume','amount'])
    compared=compare_with_ccass(minute,ccass)
    panel=build_panel_candidate(ccass,compared)
    assert panel.iloc[0].high==10.4 and panel.iloc[0].low==9.9
    source.loc[1,'high']=0
    bad=compare_with_ccass(map_minute_identity(aggregate_minutes(source,'60min','source.parquet'),*identities()),ccass)
    missing=build_panel_candidate(ccass,bad)
    assert pd.isna(missing.iloc[0].high) and missing.iloc[0].low==9.9


def identities():
    master=pd.DataFrame([{'issueID':4685,'security_id':'00823.HK'},{'issueID':4875,'security_id':'HKREIT:4875'}])
    periods=pd.DataFrame([{'ID':1,'IssueID':4685,'StockCode':'0823','StockExID':23,'FirstTradeDate':'2005-11-25','DelistDate':None},
        {'ID':2,'IssueID':4875,'StockCode':'0625','StockExID':23,'FirstTradeDate':'2007-06-22','DelistDate':'2021-10-26'},
        {'ID':3,'IssueID':99000,'StockCode':'0625','StockExID':1,'FirstTradeDate':'2026-09-01','DelistDate':None}])
    boards=pd.DataFrame([{'stockExID':23,'longName':'SEHK Real Estate Investment Trusts'},{'stockExID':1,'longName':'SEHK main board'}])
    return master,periods,boards


def test_complete_observed_day_aggregation_keeps_negative_corrections():
    row=aggregate_minutes(minutes(),'1min','source.parquet').iloc[0]
    assert row.volume==130 and row.amount==1315.
    assert row.amount_decimal=='1315.0'
    assert row.open==10 and row.close==10.3 and row.reported_high_all_rows==10.4 and row.reported_low_all_rows==9.9
    assert pd.isna(row.high) and pd.isna(row.low) and not row.open_input_valid
    assert row.negative_volume_rows==1 and row.negative_amount_rows==1 and row.minute_rows==3
    assert row.source_files==['source.parquet']


def test_missing_first_open_and_amount_are_not_filled_or_skipped():
    frame=minutes();frame.loc[0,'open']=np.nan;frame.loc[1,'amount']=np.nan
    row=aggregate_minutes(frame,'1min','source.parquet').iloc[0]
    assert pd.isna(row.open) and pd.isna(row.amount)
    assert row.volume==130 and row.has_missing_numeric
    assert 'open' in row.missing_numeric_fields and 'amount' in row.missing_numeric_fields


def test_frequency_segments_are_fixed_at_boundary():
    assert frequency_for_date('2010-12-31') is None
    assert frequency_for_date('2011-01-01')=='1min'
    assert frequency_for_date('2017-06-30')=='1min'
    assert frequency_for_date('2017-07-01')=='60min'
    assert frequency_for_date('2026-09-11')=='60min'


def test_historical_code_is_not_assigned_to_current_company_or_filled_across_gap():
    daily=pd.DataFrame([{'ts_code':'00625.HK','date':pd.Timestamp('2010-01-04')},{'ts_code':'00625.HK','date':pd.Timestamp('2026-09-02')}])
    mapped=map_minute_identity(daily,*identities())
    assert mapped.security_id.iloc[0]=='HKREIT:4875'
    assert pd.isna(mapped.security_id.iloc[1]) and mapped.identity_status.iloc[1]=='not_selected_reit_issue'


def test_mismatched_daily_totals_cannot_supply_panel_open():
    daily=map_minute_identity(aggregate_minutes(minutes(),'1min','source.parquet'),*identities())
    ccass=pd.DataFrame([{'issueID':4685,'security_id':'00823.HK','date':pd.Timestamp('2015-09-10'),'exchange_code':'00823.HK','raw_close':np.float32(10.3),'volume':131,'amount':1315,'susp':0,'newsusp':0,'noclose':0}])
    compared=compare_with_ccass(daily,ccass)
    assert compared.comparison_status.iloc[0]=='daily_mismatch'
    panel=build_panel_candidate(ccass,compared)
    assert pd.isna(panel.raw_open.iloc[0])
    assert panel.volume.iloc[0]==131 and panel.amount.iloc[0]==1315
    assert panel.minute_reported_open.iloc[0]==10.


def test_matching_day_supplies_source_open_but_never_training_approval():
    daily=map_minute_identity(aggregate_minutes(minutes().assign(vol=[100,20,50],amount=[1000.,200.,515.]),'1min','source.parquet'),*identities())
    ccass=pd.DataFrame([{'issueID':4685,'security_id':'00823.HK','date':pd.Timestamp('2015-09-10'),'exchange_code':'00823.HK','raw_close':np.float32(10.3),'volume':170,'amount':1715,'susp':0,'newsusp':0,'noclose':0}])
    compared=compare_with_ccass(daily,ccass)
    assert compared.comparison_status.iloc[0]=='matched_daily_totals'
    panel=build_panel_candidate(ccass,compared)
    assert panel.raw_open.iloc[0]==10. and panel.approved_for_training.eq(False).all()


def test_after_ccass_cutoff_is_explicit_minute_only_and_2010_has_no_open():
    daily=aggregate_minutes(minutes().assign(trade_time=['2026-09-10 09:30:00','2026-09-10 12:00:00','2026-09-10 16:10:00']),'60min','new.parquet')
    daily=map_minute_identity(daily,*identities())
    ccass=pd.DataFrame([{'issueID':4685,'security_id':'00823.HK','date':pd.Timestamp('2010-01-04'),'exchange_code':'00823.HK','raw_close':10.,'volume':100,'amount':1000,'susp':0,'newsusp':0,'noclose':0}])
    compared=compare_with_ccass(daily,ccass)
    assert compared.comparison_status.iloc[0]=='minute_only_after_ccass_cutoff'
    panel=build_panel_candidate(ccass,compared)
    assert pd.isna(panel.iloc[0].raw_open)
    assert panel.iloc[1].price_source=='tushare_minutes_only'
    assert pd.isna(panel.iloc[1].susp)


def test_subunit_amount_difference_matches_reported_precision_without_changing_values():
    frame=minutes().assign(vol=[100,20,50],amount=[1000.,200.,515.32])
    daily=map_minute_identity(aggregate_minutes(frame,'1min','source.parquet'),*identities())
    ccass=pd.DataFrame([{'issueID':4685,'security_id':'00823.HK','date':pd.Timestamp('2015-09-10'),'exchange_code':'00823.HK','raw_close':np.float32(10.3),'volume':170,'amount':1715}])
    row=compare_with_ccass(daily,ccass).iloc[0]
    assert row.comparison_status=='matched_at_reported_amount_precision'
    assert row.amount_difference_scale=='below_one_reported_unit'
    assert Decimal(row.amount_delta_decimal)==Decimal('0.32')
    assert not row.amount_match and row.amount_matches_reported_precision and row.source_open_usable
    assert row.ccass_amount==1715 and row.amount_decimal=='1715.32'
    frame.loc[2,'amount']=517.32
    row=compare_with_ccass(map_minute_identity(aggregate_minutes(frame,'1min','source.parquet'),*identities()),ccass).iloc[0]
    assert row.comparison_status=='daily_mismatch'
    assert row.amount_difference_scale=='at_least_one_reported_unit'
    assert not row.amount_matches_reported_precision and not row.source_open_usable


def test_zero_flow_first_minute_uses_first_actual_trade_open():
    frame=minutes().assign(vol=[0,10000,50],amount=[0.,43500.,217.5],open=[4.35,4.35,4.35],high=[0.,4.36,4.36],low=[0.,4.34,4.34],close=[0.,4.35,4.35])
    row=aggregate_minutes(frame,'1min','00405_20110107_pattern.parquet').iloc[0]
    assert row.open==4.35 and row.first_positive_trade_time=='2015-09-10 12:00:00'
    assert row.reported_first_open==4.35 and row.reported_first_volume==0
    assert row.open_input_valid and row.high==4.36 and row.low==4.34


def test_later_positive_trade_zero_prices_only_invalidate_affected_fields():
    frame=minutes().assign(vol=[100,20,50],amount=[1000.,200.,515.])
    frame.loc[1,['open','high','low','close']]=0.
    daily=map_minute_identity(aggregate_minutes(frame,'1min','00405_20110106_pattern.parquet'),*identities())
    row=daily.iloc[0]
    assert row.open==10. and row.open_input_valid
    assert pd.isna(row.high) and pd.isna(row.low)
    assert row.positive_trade_invalid_high_rows==1 and row.positive_trade_invalid_low_rows==1
    ccass=pd.DataFrame([{'issueID':4685,'security_id':'00823.HK','date':pd.Timestamp('2015-09-10'),'exchange_code':'00823.HK','raw_close':np.float32(10.3),'volume':170,'amount':1715}])
    assert compare_with_ccass(daily,ccass).source_open_usable.iloc[0]


def test_first_actual_trade_bad_open_is_not_replaced_and_unknown_prior_flow_blocks():
    frame=minutes().assign(vol=[0,20,50],amount=[0.,200.,515.])
    frame.loc[1,'open']=0.
    row=aggregate_minutes(frame,'1min','source.parquet').iloc[0]
    assert row.open==0. and not row.open_input_valid
    frame.loc[1,'open']=10.;frame.loc[0,'vol']=np.nan
    row=aggregate_minutes(frame,'1min','source.parquet').iloc[0]
    assert row.unknown_flow_before_first_trade and not row.open_input_valid


def test_negative_correction_kept_in_totals_but_blocks_open():
    row=aggregate_minutes(minutes(),'1min','source.parquet').iloc[0]
    assert row.volume==130 and row.amount==1315
    assert not row.open_input_valid and row.negative_volume_rows==1
