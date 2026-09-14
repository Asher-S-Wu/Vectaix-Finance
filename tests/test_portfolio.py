import numpy as np
import pandas as pd
import pytest
from hk_quant.contracts import RiskProfile
from hk_quant.portfolio import advise_portfolio


def inputs():
    ids = ['A', 'B']
    forecasts = pd.DataFrame(dict(security_id=ids, date=['2024-06-01']*2,horizon=[20]*2,expected_return=[.2,.1],status=['ok']*2))
    bars=pd.DataFrame(dict(security_id=ids,date=['2024-06-01']*2,raw_close=[10.,20.],volume=[1000000]*2,quote_present=[True]*2,fx_to_hkd=[1.]*2,adv20_amount=[10000000]*2))
    sec=pd.DataFrame(dict(security_id=ids,lot_size=[100,100],currency=['HKD']*2,identity_status=['verified']*2,lot_valid_from=['2020-01-01']*2,lot_valid_to=[None]*2))
    rng=np.random.default_rng(7)
    returns=pd.DataFrame(rng.normal(0,.02,(80,2)),columns=ids,index=pd.date_range('2024-01-01',periods=80))
    return forecasts,pd.DataFrame(columns=['security_id','quantity']),100000.,bars,sec,returns


def test_lots_cash_cost_risk_and_concentration():
    result=advise_portfolio(*inputs())
    assert result['status']=='ok'
    assert result['summary']['target_cash']>=0
    assert result['summary']['annual_volatility']<=.15+1e-6
    rows=result['recommendations']
    assert rows.target_weight.max()<=.1+1e-6
    assert (rows.target_quantity % 100 == 0).all()
    assert result['summary']['estimated_fees']>0
    assert abs(result['summary']['target_cash']+sum(rows.target_quantity*rows.reference_price)+result['summary']['estimated_fees']-100000)<1e-5


def test_negative_returns_can_remain_cash():
    args=list(inputs()); args[0]['expected_return']=-.1
    r=advise_portfolio(*args)
    assert r['status']=='ok'
    assert r['recommendations'].target_quantity.sum()==0


def test_missing_holding_price_is_data_error():
    args=list(inputs()); args[1]=pd.DataFrame([dict(security_id='C',quantity=100)])
    assert advise_portfolio(*args)['status']=='data_error'


def test_suspended_holding_cannot_be_sold():
    args=list(inputs());args[1]=pd.DataFrame([dict(security_id='A',quantity=100)])
    args[3].loc[0,'quote_present']=False
    r=advise_portfolio(*args)
    assert r['status']=='ok'
    assert r['recommendations'].set_index('security_id').loc['A','target_quantity']==100


def test_missing_history_and_lot_cannot_buy():
    args=list(inputs());args[4].loc[0,'lot_size']=np.nan;args[5].loc[:,'B']=np.nan
    r=advise_portfolio(*args)
    assert r['status']=='data_error'


def test_frozen_overweight_keeps_quantity_and_optimizes_other_security():
    args=list(inputs());args[1]=pd.DataFrame([dict(security_id='A',quantity=1200)])
    args[3].loc[0,'quote_present']=False
    result=advise_portfolio(*args)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['A','target_quantity']==1200 and rows.loc['A','order_quantity']==0
    assert rows.loc['B','target_quantity']>0
    assert rows.loc['B','target_weight']<=.1+1e-7
    assert result['summary']['constraints_satisfied'] is False
    assert any(v['constraint']=='single_position' and v['security_id']=='A' for v in result['summary']['constraint_violations'])


def test_known_adv_caps_new_purchases():
    args=list(inputs());args[3]['adv20_amount']=100000
    r=advise_portfolio(*args)
    assert r['status']=='ok'
    assert (r['recommendations'].target_quantity*r['recommendations'].reference_price<=1000).all()


def test_future_returns_cannot_change_allocation():
    args=list(inputs());before=advise_portfolio(*args)
    args[5]=pd.concat([args[5],pd.DataFrame([[100,-100]],columns=['A','B'],index=[pd.Timestamp('2030-01-01')])])
    after=advise_portfolio(*args)
    assert before['recommendations'].target_quantity.tolist()==after['recommendations'].target_quantity.tolist()


def test_negative_holdings_rejected():
    args=list(inputs());args[1]=pd.DataFrame([dict(security_id='A',quantity=-1)])
    assert advise_portfolio(*args)['status']=='data_error'


def test_round_sales_up_to_restore_concentration():
    args=list(inputs());args[1]=pd.DataFrame([dict(security_id='A',quantity=1200)])
    r=advise_portfolio(*args)
    assert r['status']=='ok'
    assert r['recommendations'].set_index('security_id').loc['A','target_quantity']<=1100


def test_sell_participation_is_limited_to_whole_lot_capacity():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=1000)])
    a[0]['expected_return']=-.1;a[3]['adv20_amount']=150000
    r=advise_portfolio(*a)
    assert r['status']=='ok'
    row=r['recommendations'].set_index('security_id').loc['A']
    assert row.target_quantity==900
    assert row.order_quantity*row.reference_price<=1500


def test_unknown_identity_holding_cannot_be_sold():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=1000)])
    a[0]['expected_return']=-.1;a[4].loc[0,'identity_status']='unknown'
    r=advise_portfolio(*a)
    assert r['status']=='ok'
    row=r['recommendations'].set_index('security_id').loc['A']
    assert row.target_quantity==1000
    assert '身份' in row.reason


def test_usd_portfolio_uses_hkd_values_and_keeps_native_reference():
    a=list(inputs());a[4]['currency']='USD';a[3]['fx_to_hkd']=8.
    r=advise_portfolio(*a)
    assert r['status']=='ok'
    row=r['recommendations'].set_index('security_id').loc['A']
    assert row.target_quantity==100
    assert row.reference_price==10 and row.currency=='USD'
    assert row.target_value_hkd==8000


def test_missing_fx_holding_rejects_account_valuation():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=100)])
    a[3].loc[0,'fx_to_hkd']=float('nan')
    assert advise_portfolio(*a)['status']=='data_error'


def test_missing_fx_candidate_is_excluded_with_reason():
    a=list(inputs());a[3].loc[0,'fx_to_hkd']=float('nan')
    r=advise_portfolio(*a)
    assert r['status']=='ok'
    assert 'A' not in set(r['recommendations'].security_id)
    assert '汇率' in r['excluded'].set_index('security_id').loc['A','reason']


def test_same_isin_counters_share_single_stock_concentration_limit():
    a=list(inputs());a[4]['isin']='HK000000001'
    a[4].loc[1,'currency']='RMB';a[3].loc[1,'fx_to_hkd']=1.1
    r=advise_portfolio(*a)
    assert r['status']=='ok'
    assert r['recommendations'].target_weight.sum()<=.1+1e-7


def test_frozen_same_isin_holdings_cannot_hide_joint_overweight():
    a=list(inputs());a[4]['isin']='HK000000001'
    a[1]=pd.DataFrame([dict(security_id='A',quantity=1000),dict(security_id='B',quantity=500)])
    a[3]['quote_present']=False
    result=advise_portfolio(*a)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['A','target_quantity']==1000 and rows.loc['B','target_quantity']==500
    assert rows.order_quantity.eq(0).all()
    assert any(v['constraint']=='isin_concentration' and v['isin']=='HK000000001' for v in result['summary']['constraint_violations'])


def test_usd_adv_capacity_converts_both_sides_to_hkd():
    a=list(inputs());a[4]['currency']='USD';a[3]['fx_to_hkd']=8.;a[3]['adv20_amount']=100000.
    r=advise_portfolio(*a)
    assert r['status']=='ok'
    row=r['recommendations'].set_index('security_id').loc['A']
    assert row.target_quantity==100
    assert row.target_value_hkd==8000


def test_equal_weight_uses_same_candidates_without_fee_induced_all_cash():
    a=list(inputs());a[0]['expected_return']=-.1
    r=advise_portfolio(*a,allocation='equal_weight')
    assert r['status']=='ok'
    assert set(r['recommendations'].security_id)=={'A','B'}
    assert r['recommendations'].target_weight.min()>.05
    assert r['summary']['estimated_fees']>0
    assert r['summary']['allocation']=='equal_weight'


def test_equal_weight_pressure_fees_are_charged_and_cash_is_not_borrowed():
    a=list(inputs())
    standard=advise_portfolio(*a,allocation='equal_weight')
    stress=advise_portfolio(*a,profile=RiskProfile(fee_per_side=.005),allocation='equal_weight')
    assert stress['status']=='ok' and stress['summary']['target_cash']>=0
    assert stress['summary']['estimated_fees']>standard['summary']['estimated_fees']


@pytest.mark.parametrize('allocation',['utility','equal_weight'])
@pytest.mark.parametrize('fee',[.0025,.005])
def test_new_orders_respect_concentration_after_actual_fees(allocation,fee):
    result=advise_portfolio(*inputs(),profile=RiskProfile(fee_per_side=fee),allocation=allocation)
    assert result['status']=='ok'
    rows=result['recommendations'];summary=result['summary']
    net_equity=summary['target_cash']+rows.target_value_hkd.sum()
    assert (rows.target_value_hkd/net_equity<=.1+1e-7).all()
    assert np.isclose(net_equity,100000-summary['estimated_fees'])
    assert np.allclose(rows.target_weight,rows.target_value_hkd/net_equity)
    assert summary['constraints_satisfied'] is True
    assert summary['constraint_violations']==[]


def test_same_isin_new_orders_share_limit_after_fees():
    a=list(inputs());a[4]['isin']='HK000000001'
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    rows=result['recommendations']
    net_equity=result['summary']['target_cash']+rows.target_value_hkd.sum()
    assert rows.target_value_hkd.sum()/net_equity<=.1+1e-7


def test_frozen_isin_overweight_forbids_new_risk_in_tradable_counter():
    a=list(inputs());a[4]['isin']='HK000000001'
    a[1]=pd.DataFrame([dict(security_id='A',quantity=2000)])
    a[3].loc[0,'quote_present']=False
    result=advise_portfolio(*a)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['A','target_quantity']==2000
    assert rows.loc['B','target_quantity']==0 and rows.loc['B','order_quantity']==0
    assert result['summary']['estimated_fees']==0


def test_frozen_total_equity_breach_is_reported_without_fabricated_sale():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=1000)])
    a[2]=50.;a[3].loc[0,'quote_present']=False;a[5]*=.01
    result=advise_portfolio(*a)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['A','target_quantity']==1000 and rows.loc['B','target_quantity']==0
    assert rows.order_quantity.eq(0).all()
    assert result['summary']['target_cash']==50
    assert result['summary']['target_equity_weight']>.95
    assert any(v['constraint']=='total_equity' and v['limit']==.95 for v in result['summary']['constraint_violations'])


def test_unreachable_volatility_returns_minimum_shortfall_and_original_target():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=20000)])
    a[3].loc[0,'quote_present']=False
    result=advise_portfolio(*a)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id');summary=result['summary']
    assert rows.loc['A','target_quantity']==20000 and rows.loc['A','order_quantity']==0
    assert summary['volatility_target_reachable'] is False
    assert summary['minimum_volatility_constraint_shortfall']>0
    assert summary['annual_volatility']>.15
    assert any(v['constraint']=='annual_volatility' and v['limit']==.15 for v in summary['constraint_violations'])


def test_frozen_risk_can_be_offset_by_an_adjustable_negative_beta_asset():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=2500)])
    a[3].loc[0,'quote_present']=False
    market=np.random.default_rng(51).normal(0,.05,80)
    a[5]=pd.DataFrame({'A':market,'B':-market,'market_reference':.8*market},index=a[5].index)
    result=advise_portfolio(*a)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['A','target_quantity']==2500 and rows.loc['B','target_quantity']>0
    assert result['summary']['volatility_target_reachable'] is True
    assert result['summary']['annual_volatility']<=.15+1e-7


def test_fees_cross_frozen_limit_without_blocking_other_allowed_buys():
    a=list(inputs());a[1]=pd.DataFrame([dict(security_id='A',quantity=4999)])
    a[2]=50010.;a[3].loc[0,'quote_present']=False;a[5]*=.1
    profile=RiskProfile(max_position_weight=.5,max_equity_weight=1.)
    result=advise_portfolio(*a,profile=profile)
    assert result['status']=='constraints_unmet'
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['A','target_quantity']==4999 and rows.loc['A','order_quantity']==0
    assert rows.loc['A','target_weight']>.5
    assert rows.loc['B','target_value_hkd']>8000 and rows.loc['B','target_weight']<=.5+1e-7
    assert result['summary']['target_cash']>=0
