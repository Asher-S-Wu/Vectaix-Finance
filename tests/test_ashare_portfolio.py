import numpy as np
import pandas as pd
import pytest


def inputs(ids=None):
    ids=ids or ['600000.SH','000001.SZ']; day='2024-06-03'
    f=pd.DataFrame([dict(security_id=sid,date=day,horizon=20,expected_return=.2,status='ok') for sid in ids])
    b=pd.DataFrame([dict(security_id=sid,date=day,raw_open=10.,raw_close=10.,high=10.1,low=9.9,
                        volume=1000000.,amount=10000000.,quote_present=True,adv20_amount=10000000.,
                        up_limit=11.,down_limit=9.) for sid in ids])
    s=pd.DataFrame([dict(security_id=sid,lot_size=200 if sid.startswith('688') else 100,
                        identity_status='verified',currency='CNY',asset_type='equity') for sid in ids])
    r=pd.DataFrame(np.random.default_rng(7).normal(0,.01,(80,len(ids))),columns=ids,index=pd.bdate_range('2024-01-01',periods=80))
    return [f,pd.DataFrame(columns=['security_id','quantity']),100000.,b,s,r]


def test_cny_lots_risk_cash_and_actual_side_fees():
    from ashare_quant.portfolio import advise_portfolio
    from ashare_quant.rules import transaction_fee
    result=advise_portfolio(*inputs()); rows=result['recommendations']; summary=result['summary']
    assert result['status']=='ok' and result['currency']=='CNY'
    assert (rows.target_quantity % 100==0).all() and rows.target_quantity.sum()>0
    assert rows.target_weight.max()<=.1+1e-7 and summary['annual_volatility']<=.15+1e-7
    assert summary['target_cash']>=0 and summary['estimated_fees']>0
    assert np.isclose(summary['target_cash']+rows.target_value_cny.sum()+summary['estimated_fees'],100000)
    assert np.isclose(summary['estimated_fees'],sum(transaction_fee(row.order_quantity*row.reference_price,'buy','2024-06-03',row.security_id) for row in rows.itertuples()))
    assert not any('hkd' in c for c in rows.columns)


@pytest.mark.parametrize('cause',['today','unknown_date','lower_limit','suspended','missing_limit'])
def test_unavailable_sales_freeze_positions_with_explanation(cause):
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(); a[0]['expected_return']=-.2
    holding=dict(security_id='600000.SH',quantity=100)
    if cause!='unknown_date': holding['acquired_date']='2024-06-03' if cause=='today' else '2024-05-30'
    a[1]=pd.DataFrame([holding])
    if cause=='lower_limit': a[3].loc[0,['raw_close','raw_open']]=9.
    if cause=='suspended': a[3].loc[0,'quote_present']=False
    if cause=='missing_limit': a[3].loc[0,'down_limit']=np.nan
    result=advise_portfolio(*a); row=result['recommendations'].set_index('security_id').loc['600000.SH']
    assert row.target_quantity==100 and row.order_quantity==0
    assert row.sell_allowed==False and row.sell_block_reason


def test_upper_limit_blocks_buy_but_not_eligible_sale():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(); a[0]['expected_return']=-.2
    a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=100,acquired_date='2024-05-30')])
    a[3].loc[0,['raw_open','raw_close']]=11.
    result=advise_portfolio(*a); row=result['recommendations'].set_index('security_id').loc['600000.SH']
    assert row.buy_allowed==False and row.sell_allowed==True and row.target_quantity==0


def test_sellable_quantity_respects_t_plus_one_for_mixed_lots():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(); a[0]['expected_return']=-.2
    a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=300,sellable_quantity=100)])
    result=advise_portfolio(*a); row=result['recommendations'].set_index('security_id').loc['600000.SH']
    assert row.target_quantity==200 and row.order_quantity==100


def test_candidate_requires_daily_limits_identity_and_cny():
    from ashare_quant.portfolio import advise_portfolio
    for field,value in [('up_limit',np.nan),('quote_present',False)]:
        a=inputs(); a[3].loc[0,field]=value
        r=advise_portfolio(*a)
        assert '600000.SH' not in set(r['recommendations'].security_id)
        assert '600000.SH' in set(r['excluded'].security_id)
    a=inputs();a[4].loc[0,'currency']='HKD'
    assert '600000.SH' not in set(advise_portfolio(*a)['recommendations'].security_id)


def test_future_risk_and_forecasts_do_not_leak():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(); before=advise_portfolio(*a)
    a[5]=pd.concat([a[5],pd.DataFrame([[100,-100]],index=[pd.Timestamp('2030-01-01')],columns=a[5].columns)])
    after=advise_portfolio(*a)
    assert before['recommendations'].target_quantity.tolist()==after['recommendations'].target_quantity.tolist()
    a[0]['date']='2030-01-01'; assert advise_portfolio(*a)['status']=='data_error'


def test_frozen_overweight_reports_violation_and_keeps_other_candidates():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(); a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=2000,acquired_date='2024-06-03')])
    r=advise_portfolio(*a); rows=r['recommendations'].set_index('security_id')
    assert r['status']=='constraints_unmet' and rows.loc['600000.SH','target_quantity']==2000
    assert rows.loc['000001.SZ','target_quantity']>0
    assert r['summary']['constraints_satisfied'] is False


def test_insufficient_history_and_fractional_holdings_fail_closed():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(); a[5]=a[5].iloc[:30]
    assert advise_portfolio(*a)['status']=='data_error'
    a=inputs();a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=1.5)])
    assert advise_portfolio(*a)['status']=='data_error'


def test_star_board_orders_respect_minimum_200():
    from ashare_quant.portfolio import advise_portfolio, RiskProfile
    a=inputs(['688001.SH','430047.BJ'])
    result=advise_portfolio(*a,profile=RiskProfile(max_position_weight=.15))
    rows=result['recommendations'].set_index('security_id')
    assert rows.loc['688001.SH','order_quantity']>=200
    assert rows.loc['430047.BJ','order_quantity']>=100


def test_star_partial_sale_never_below_200_shares():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(['688001.SH','430047.BJ']); a[0]['expected_return']=-.2
    a[1]=pd.DataFrame([dict(security_id='688001.SH',quantity=300,sellable_quantity=100)])
    result=advise_portfolio(*a)
    row=result['recommendations'].set_index('security_id').loc['688001.SH']
    assert row.order_quantity==0 and row.target_quantity==300


def test_star_final_odd_lot_can_be_liquidated():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs(['688001.SH','430047.BJ']); a[0]['expected_return']=-.2
    a[1]=pd.DataFrame([dict(security_id='688001.SH',quantity=150,acquired_date='2024-05-30')])
    result=advise_portfolio(*a)
    row=result['recommendations'].set_index('security_id').loc['688001.SH']
    assert row.order_quantity==150 and row.target_quantity==0


def test_zero_cash_frozen_holding_returns_constraint_report_not_infeasible():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs();a[2]=0.;a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=100,acquired_date='2024-06-03')])
    r=advise_portfolio(*a)
    assert r['status']=='constraints_unmet'
    row=r['recommendations'].set_index('security_id').loc['600000.SH']
    assert row.target_quantity==100 and row.order_quantity==0
    assert r['summary']['target_cash']==0


def test_buy_references_close_price_limit_not_historical_open():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs();a[3].loc[0,'raw_close']=11.
    r=advise_portfolio(*a)
    assert '600000.SH' not in set(r['recommendations'].security_id)


def test_sell_participation_uses_valid_lots_without_exceeding_cap():
    from ashare_quant.portfolio import advise_portfolio
    a=inputs();a[0]['expected_return']=-.2
    a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=250,acquired_date='2024-05-30')])
    a[3].loc[0,'adv20_amount']=150000.
    r=advise_portfolio(*a);row=r['recommendations'].set_index('security_id').loc['600000.SH']
    assert row.order_quantity==100 and row.target_quantity==150


def test_side_fee_sell_includes_stamp_duty_and_buy_does_not():
    from ashare_quant.portfolio import advise_portfolio
    from ashare_quant.rules import transaction_fee
    a=inputs();a[0]['expected_return']=-.2
    a[1]=pd.DataFrame([dict(security_id='600000.SH',quantity=100,acquired_date='2024-05-30')])
    r=advise_portfolio(*a);row=r['recommendations'].set_index('security_id').loc['600000.SH']
    assert row.estimated_fee==transaction_fee(1000.,'sell','2024-06-03','600000.SH')
    assert row.estimated_fee>transaction_fee(1000.,'buy','2024-06-03','600000.SH')


def test_many_candidates_do_not_reserve_impossible_commissions():
    from ashare_quant.portfolio import advise_portfolio, RiskProfile
    ids=[f'{600000+i:06d}.SH' for i in range(100)]
    a=inputs(ids);a[2]=1000.;a[0]['expected_return']=-.1;a[0].loc[0,'expected_return']=1.
    # Only one candidate has a positive preference; no cash is spent on unused candidates.
    r=advise_portfolio(*a,profile=RiskProfile(max_position_weight=.9,max_equity_weight=.95))
    assert r['status']=='ok'
    # With price 10 and a 100-share minimum, preserving cash is the legitimate result.
    assert r['summary']['target_cash']==1000.
    a[2]=10000.
    r=advise_portfolio(*a,profile=RiskProfile(max_position_weight=.9,max_equity_weight=.95,min_commission=200.))
    row=r['recommendations'].set_index('security_id').loc[ids[0]]
    assert row.target_quantity>=100
    assert r['summary']['target_cash']>=0


def test_full_universe_concentration_matrix_remains_sparse(monkeypatch):
    import cvxpy as cp
    from ashare_quant.portfolio import advise_portfolio
    ids=[f'{600000+i:06d}.SH' for i in range(100)]
    a=inputs(ids);original=cp.Problem.solve;nonzeros=[]
    def observed_solve(problem,*args,**kwargs):
        data=problem.get_problem_data('CLARABEL')[0]
        nonzeros.append(data['A'].nnz)
        return original(problem,*args,**kwargs)
    monkeypatch.setattr(cp.Problem,'solve',observed_solve)
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    assert nonzeros and max(nonzeros)<100*len(ids)
