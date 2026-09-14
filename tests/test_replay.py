import numpy as np
import pandas as pd
import pytest

from hk_quant.contracts import RiskProfile
from hk_quant.replay import replay_account


def inputs():
    dates=pd.date_range('2024-01-01',periods=5)
    targets=pd.DataFrame([dict(signal_date=dates[0],security_id='A',target_weight=.1)])
    bars=pd.DataFrame(dict(date=dates,security_id=['A']*5,raw_close=[10.]*5,vwap=[10.]*5,volume=[100000]*5,amount=[1000000.]*5,quote_present=[True]*5,data_valid=[True]*5,fx_to_hkd=[1.]*5))
    sec=pd.DataFrame([dict(security_id='A',lot_size=100,lot_valid_from='2020-01-01',identity_status='verified',currency='HKD')])
    actions=pd.DataFrame(columns=['security_id','action_type','effective_date','payment_date','verified','cash_per_share','share_multiplier'])
    actions.attrs['coverage_complete']=True
    return [targets,bars,pd.DataFrame({'date':dates}),sec,actions,RiskProfile()]


def test_late_cash_amount_keeps_entitled_units_without_future_value():
    a=inputs()
    a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',
        effective_date='2024-01-03',payment_date='2024-01-05',
        amount_known_at='2024-01-04T19:44:00+08:00',
        cash_per_share_hkd=1.,cash_currency='HKD',verified=True,
        source_url='https://www.hkexnews.hk/final.pdf',
        payment_basis='issuer_final_schedule_simulated')])
    a[4].attrs['coverage_complete']=True
    r=replay_account(*a,initial_cash=100000)
    daily=r['daily'].set_index('date')
    for day in ['2024-01-03','2024-01-04']:
        assert pd.isna(daily.loc[day,'equity'])
        assert pd.isna(daily.loc[day,'receivables'])
        assert daily.loc[day,'cash']==89975
    assert daily.loc['2024-01-05','cash']==90975
    assert daily.loc['2024-01-05','equity']==100975
    entitlement=r['events'].query("event_type == 'entitlement'").iloc[0]
    assert entitlement.entitled_quantity==1000
    assert pd.isna(entitlement.receivable_amount)
    recognized=r['events'].query("event_type == 'amount_recognition'").iloc[0]
    assert recognized.date==pd.Timestamp('2024-01-05')
    assert recognized.receivable_amount==1000


def test_unknown_cash_publication_is_not_assumed_known():
    a=inputs()
    a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',
        effective_date='2024-01-03',payment_date='2024-01-05',amount_known_at=None,
        cash_per_share_hkd=1.,cash_currency='HKD',verified=True,
        source_url='https://www.hkexnews.hk/final.pdf',
        payment_basis='issuer_final_schedule_simulated')])
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='data_error'
    assert r['trades'].empty


def test_next_day_lots_fees_cash_identity():
    r=replay_account(*inputs(),initial_cash=100000)
    t=r['trades'].iloc[0]
    assert t.date==pd.Timestamp('2024-01-02')
    assert t.quantity==1000 and t.fee==25
    assert r['daily'].iloc[-1]['cash']==89975
    assert r['daily'].iloc[-1]['equity']==99975


def test_suspension_no_trade_and_no_fabricated_quote():
    a=inputs();a[1].loc[1,['quote_present','volume','amount']]=[False,0,0.]
    r=replay_account(*a)
    assert r['trades'].empty
    assert len(r['unfilled'])==1


@pytest.mark.parametrize('amount',[0.,np.nan])
def test_confirmed_zero_volume_without_lot_keeps_cash_and_unfilled_buy(amount):
    a=inputs();a[3]['lot_valid_from']='2025-01-01'
    a[1].loc[1,['quote_present','volume','amount']]=[False,0,amount]
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='ok'
    assert r['gaps'].empty and r['trades'].empty and r['holdings'].empty
    assert r['daily'].cash.eq(100000).all() and r['summary']['fees']==0
    assert len(r['unfilled'])==1 and r['unfilled'].iloc[0].quantity==1000
    assert r['unfilled'].iloc[0].reason=='当日没有成交，订单未执行'


def test_confirmed_zero_volume_without_lot_keeps_owned_shares_and_cash():
    a=inputs();a[3]['lot_valid_to']='2024-01-02'
    a[0]=pd.concat([a[0],pd.DataFrame([dict(signal_date=pd.Timestamp('2024-01-02'),security_id='A',target_weight=0.)])],ignore_index=True)
    a[1].loc[2,['quote_present','volume','amount']]=[False,0,0.]
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='ok' and r['gaps'].empty
    assert len(r['trades'])==1 and r['summary']['fees']==25
    assert r['daily'].iloc[1:].cash.eq(89975).all()
    assert r['holdings'].iloc[0].quantity==1000
    assert r['daily'].iloc[2].positions[0]['quantity']==1000
    assert r['unfilled'].iloc[0].quantity==-1000
    assert r['unfilled'].iloc[0].reason=='当日没有成交，订单未执行'


@pytest.mark.parametrize('present,volume,amount,valid',[
    (False,np.nan,0.,True),
    (False,100.,1000.,True),
    (True,0.,0.,True),
    (False,0.,100.,True),
    (False,0.,-1.,True),
    (False,0.,0.,False),
    (False,0.,0.,None),
    (None,0.,0.,True),
])
@pytest.mark.parametrize('lot_available',[True,False])
def test_unknown_or_conflicting_no_trade_evidence_is_gap(present,volume,amount,valid,lot_available):
    a=inputs();a[1]=a[1].astype({'volume':float,'quote_present':object,'data_valid':object})
    a[1].loc[1,['quote_present','volume','amount','data_valid']]=[present,volume,amount,valid]
    if not lot_available:a[3]['lot_valid_from']='2025-01-01'
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete' and not r['gaps'].empty
    assert r['trades'].empty and r['holdings'].empty
    assert r['daily'].cash.eq(100000).all()
    assert not r['unfilled'].reason.str.contains('当日没有成交').any()


def test_no_trade_lot_exemption_does_not_remove_held_price_gap():
    a=inputs();a[3]['lot_valid_to']='2024-01-02'
    a[0]=pd.concat([a[0],pd.DataFrame([dict(signal_date=pd.Timestamp('2024-01-02'),security_id='A',target_weight=0.)])],ignore_index=True)
    a[1].loc[2,['quote_present','volume','amount','raw_close']]=[False,0,0.,np.nan]
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['gaps'].reason.str.contains('未复权估值').any()
    assert not r['gaps'].reason.str.contains('交易单位').any()
    assert pd.isna(r['daily'].iloc[2].equity)
    assert r['holdings'].iloc[0].quantity==1000 and r['daily'].iloc[2].cash==89975


@pytest.mark.parametrize('volume,amount,valid',[
    (np.nan,0.,True),
    (100.,1000.,True),
    (0.,100.,True),
    (0.,0.,False),
])
def test_held_unknown_or_conflicting_trade_record_is_not_counted_as_no_trade(volume,amount,valid):
    a=inputs();a[1]=a[1].astype({'volume':float})
    a[1].loc[3,['quote_present','volume','amount','data_valid']]=[False,volume,amount,valid]
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    day=r['daily'].iloc[3]
    assert day.positions[0]['quote_present'] is None
    assert day.positions[0]['sessions_without_trade'] is None
    assert day.cash==89975 and day.equity==99975
    assert r['holdings'].iloc[0].quantity==1000


def test_historical_lot_required():
    a=inputs();a[3]=a[3].drop(columns='lot_valid_from')
    r=replay_account(*a)
    assert r['trades'].empty
    assert r['summary']['status']=='incomplete'


def test_dividend_only_payment_date_and_split_identity():
    a=inputs();a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,cash_currency='HKD',verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated'),dict(security_id='A',action_type='split',effective_date='2024-01-04',share_multiplier=2,verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    a[4].attrs['coverage_complete']=True
    a[1].loc[3:,'raw_close']=5
    r=replay_account(*a,initial_cash=100000)
    d=r['daily'].set_index('date')
    assert d.loc['2024-01-04','cash']==89975
    assert d.loc['2024-01-05','cash']==90975
    assert d.loc['2024-01-04','holdings_value']==10000
    assert d.loc['2024-01-05','equity']==100975


def test_unknown_privatisation_payment_is_gap():
    a=inputs();a[4]=pd.DataFrame([dict(security_id='A',action_type='acquisition',effective_date='2024-01-03',payment_date=None,cash_per_share=12,cash_currency='HKD',verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['daily'].iloc[-1]['cash']==89975


def test_absent_quote_does_not_forward_fill():
    a=inputs();a[1]=a[1].drop(index=3)
    r=replay_account(*a,initial_cash=100000)
    assert pd.isna(r['daily'].set_index('date').loc['2024-01-04','equity'])


def test_unverified_dividend_never_creates_cash():
    a=inputs();a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,cash_currency='HKD',verified=False)])
    r=replay_account(*a,initial_cash=100000)
    assert r['daily'].iloc[-1]['cash']==89975
    assert r['summary']['status']=='incomplete'


def test_future_lot_record_does_not_authorize_past_trade():
    a=inputs();a[3]['lot_valid_from']='2025-01-01'
    r=replay_account(*a)
    assert r['trades'].empty
    assert not r['gaps'].empty


def test_daily_participation_and_lots_cap_volume():
    a=inputs();a[1]['volume']=25000
    r=replay_account(*a,initial_cash=100000)
    assert r['trades'].iloc[0]['quantity']==200
    assert r['trades'].iloc[0]['fee']==5


def test_next_day_price_cannot_overdraw_cash():
    a=inputs();a[1].loc[1,'vwap']=1000
    r=replay_account(*a,initial_cash=100000)
    assert (r['daily']['cash']>=0).all()
    assert r['trades'].empty


def test_reference_quote_values_suspension_without_claiming_trade():
    a=inputs();a[1].loc[3,['quote_present','volume','amount']]=[False,0,0.]
    r=replay_account(*a,initial_cash=100000)
    day=r['daily'].iloc[3]
    assert day.equity==99975
    assert day.positions[0]['quote_present'] is False
    assert day.positions[0]['sessions_without_trade']==1


def test_rebalance_snapshot_sells_omitted_holding():
    a=inputs()
    a[0]=pd.concat([a[0],pd.DataFrame([dict(signal_date=pd.Timestamp('2024-01-04'),security_id='B',target_weight=.1)])],ignore_index=True)
    extra=a[1].copy();extra['security_id']='B';a[1]=pd.concat([a[1],extra],ignore_index=True)
    extra=a[3].copy();extra['security_id']='B';a[3]=pd.concat([a[3],extra],ignore_index=True)
    r=replay_account(*a,initial_cash=100000)
    assert 'A' not in set(r['holdings'].security_id)
    assert r['trades'].query("security_id == 'A' and side == 'sell'").iloc[0]['date']==pd.Timestamp('2024-01-05')


def test_split_adjusts_overnight_orders_before_capacity():
    a=inputs();a[1].loc[1:,['raw_close','vwap']]=5.;a[1]['volume']=1000000
    a[4]=pd.DataFrame([dict(security_id='A',action_type='split',effective_date='2024-01-02',share_multiplier=2,verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    a[4].attrs['coverage_complete']=True
    r=replay_account(*a,initial_cash=100000)
    assert r['trades'].iloc[0]['quantity']==2000
    assert r['daily'].iloc[1]['holdings_value']==10000


def test_unknown_identity_prevents_replay_trade():
    a=inputs();a[3]['identity_status']='unknown'
    r=replay_account(*a,initial_cash=100000)
    assert r['trades'].empty
    assert r['summary']['status']=='incomplete'


def test_receivable_preserves_equity_until_dividend_payment():
    a=inputs();a[1].loc[2:,'raw_close']=9
    a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,cash_currency='HKD',verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    a[4].attrs['coverage_complete']=True
    r=replay_account(*a,initial_cash=100000)
    d=r['daily'].set_index('date')
    assert d.loc['2024-01-03','equity']==99975
    assert d.loc['2024-01-03','receivables']==1000
    assert d.loc['2024-01-05','receivables']==0
    assert d.loc['2024-01-05','equity']==99975
    assert d.loc['2024-01-05','cash']==90975


def test_positive_volume_missing_vwap_is_data_gap():
    a=inputs();a[1].loc[1,'vwap']=float('nan')
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert not r['gaps'].empty
    assert r['trades'].empty


def test_missing_quote_indicator_is_not_suspension():
    a=inputs();a[1]['quote_present']=a[1]['quote_present'].astype(object);a[1].loc[1,'quote_present']=None
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert not r['gaps'].empty


def test_missing_held_quote_state_is_gap_not_suspension():
    a=inputs();a[1]['quote_present']=a[1]['quote_present'].astype(object);a[1].loc[3,'quote_present']=None
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['daily'].iloc[3].positions[0]['quote_present'] is None
    assert r['daily'].iloc[3].positions[0]['sessions_without_trade'] is None


def test_usd_replay_uses_hkd_cash_and_native_shares():
    a=inputs();a[3]['currency']='USD';a[1]['fx_to_hkd']=8.
    r=replay_account(*a,initial_cash=100000)
    trade=r['trades'].iloc[0]
    assert trade.quantity==100 and trade.price==10
    assert trade.value_hkd==8000 and trade.fee==20
    assert r['daily'].iloc[1]['cash']==91980


def test_fx_change_changes_hkd_valuation():
    a=inputs();a[3]['currency']='USD';a[1]['fx_to_hkd']=8.;a[1].loc[3:,'fx_to_hkd']=9.
    r=replay_account(*a,initial_cash=100000)
    assert r['daily'].iloc[3]['holdings_value']==9000
    assert r['daily'].iloc[3]['equity']==100980


def test_missing_held_fx_marks_incomplete_nav():
    a=inputs();a[1].loc[3,'fx_to_hkd']=float('nan')
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert pd.isna(r['daily'].iloc[3]['equity'])


def test_native_dividend_converts_at_payment_day_fx():
    a=inputs();a[3]['currency']='USD';a[1]['fx_to_hkd']=8.;a[1].loc[4,'fx_to_hkd']=9.
    a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,cash_currency='USD',verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    a[4].attrs['coverage_complete']=True
    r=replay_account(*a,initial_cash=100000)
    assert r['daily'].iloc[2]['receivables']==800
    assert r['daily'].iloc[4]['cash']==92880


def test_dividend_without_currency_cannot_create_hkd_cash():
    a=inputs();a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['daily'].iloc[4]['cash']==89975


def test_missing_payment_fx_never_posts_dividend_cash():
    a=inputs();a[3]['currency']='USD';a[1]['fx_to_hkd']=8.;a[1].loc[4,'fx_to_hkd']=float('nan')
    a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,cash_currency='USD',verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['daily'].iloc[4]['cash']==91980


def test_explicit_hkd_dividend_amount_needs_no_currency_guess():
    a=inputs();a[4]=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share_hkd=1,verified=True,source_url='https://www.hkexnews.hk/test-final.pdf',payment_basis='issuer_final_schedule_simulated')])
    a[4].attrs['coverage_complete']=True
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='ok'
    assert r['daily'].iloc[4]['cash']==90975


def test_replay_rejects_same_verified_isin_counter_overweight():
    a=inputs();a[3]['isin']='HK000000001'
    extra=a[3].copy();extra['security_id']='B';a[3]=pd.concat([a[3],extra],ignore_index=True)
    a[0]=pd.concat([a[0],pd.DataFrame([dict(signal_date=pd.Timestamp('2024-01-01'),security_id='B',target_weight=.1)])],ignore_index=True)
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='data_error'
    assert r['trades'].empty


def test_replay_does_not_merge_different_isins_by_name():
    a=inputs();a[3]['isin']='HK000000001';a[3]['name']='Same name'
    extra=a[3].copy();extra['security_id']='B';extra['isin']='HK000000002';a[3]=pd.concat([a[3],extra],ignore_index=True)
    extra=a[1].copy();extra['security_id']='B';a[1]=pd.concat([a[1],extra],ignore_index=True)
    a[0]=pd.concat([a[0],pd.DataFrame([dict(signal_date=pd.Timestamp('2024-01-01'),security_id='B',target_weight=.1)])],ignore_index=True)
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='ok'


def test_dynamic_policy_gets_post_trade_cash_and_only_current_bars():
    a=inputs();seen=[];a[1].loc[1,'vwap']=12.
    def policy(day,account):
        seen.append((day,account['cash'],account['holdings'].copy(),account['bars'].copy()))
        return pd.DataFrame([{'security_id':'A','target_weight':.1}])
    r=replay_account(None,*a[1:],initial_cash=100000,policy=policy)
    assert r['summary']['status']=='ok'
    assert seen[1][1]==87970 and seen[1][2].iloc[0].quantity==1000
    assert r['trades'].iloc[0].price==12.
    assert all(frame.date.eq(day).all() for day,_,_,frame in seen)
    assert r['trades'].iloc[0].date==pd.Timestamp('2024-01-02')


def test_dynamic_policy_failure_preserves_account_without_replacement():
    a=inputs()
    def policy(day,account):
        if day==pd.Timestamp('2024-01-01'):return pd.DataFrame([{'security_id':'A','target_weight':.1}])
        raise ValueError('预测文件缺失')
    r=replay_account(None,*a[1:],initial_cash=100000,policy=policy)
    assert r['summary']['status']=='incomplete'
    assert len(r['trades'])==1 and r['holdings'].iloc[0].quantity==1000
    assert r['gaps'].reason.str.contains('预测文件缺失').any()


def test_dynamic_and_nonempty_static_targets_cannot_mix():
    a=inputs()
    r=replay_account(*a,policy=lambda day,account:pd.DataFrame())
    assert r['summary']['status']=='data_error'


def test_dynamic_policy_still_executes_real_next_vwap_and_cash_limit():
    a=inputs();a[1].loc[1,'vwap']=1000
    r=replay_account(None,*a[1:],initial_cash=100000,policy=lambda day,account:pd.DataFrame([{'security_id':'A','target_weight':.1}]))
    assert r['trades'].empty or r['trades'].iloc[0].date>pd.Timestamp('2024-01-02')
    assert r['daily'].cash.ge(0).all()
