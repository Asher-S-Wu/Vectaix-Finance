import numpy as np
import pandas as pd
import pytest

from hk_quant.portfolio import advise_portfolio
from test_portfolio import inputs


def test_every_candidate_beyond_200_reaches_risk_and_account_assessment():
    a=list(inputs());ids=[f'S{i:03}' for i in range(205)]
    for pos in (0,3,4):
        a[pos]=pd.concat([a[pos].iloc[[0]]]*len(ids),ignore_index=True)
        a[pos]['security_id']=ids
    rng=np.random.default_rng(6)
    market=rng.normal(0,.01,100)
    a[5]=pd.DataFrame(market[:,None]+rng.normal(0,.005,(100,len(ids))),index=pd.bdate_range('2023-12-01',periods=100),columns=ids)
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    assert set(result['recommendations'].security_id)==set(ids)
    assert result['summary']['candidate_count']==205
    assert set(result['summary']['risk_observations'])==set(ids)


def test_asynchronous_missing_returns_do_not_require_complete_case_panel():
    a=list(inputs());rng=np.random.default_rng(4)
    market=rng.normal(0,.01,180)
    a[5]=pd.DataFrame({'A':market+rng.normal(0,.003,180),'B':market+rng.normal(0,.003,180)},index=pd.bdate_range(end='2024-05-31',periods=180))
    a[5].iloc[::2,0]=np.nan;a[5].iloc[1::2,1]=np.nan
    assert a[5].dropna().empty
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    assert result['summary']['risk_observations']=={'A':90,'B':90}
    assert result['summary']['market_observations']==180


def test_new_candidate_insufficient_risk_history_is_explicitly_excluded():
    a=list(inputs());a[5].loc[a[5].index[:30],'A']=np.nan
    forecasts=a[0].copy()
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    assert result['recommendations'].security_id.tolist()==['B']
    assert '60' in result['excluded'].set_index('security_id').loc['A','reason']
    pd.testing.assert_frame_equal(a[0],forecasts)


def test_existing_position_insufficient_risk_history_blocks_account():
    a=list(inputs());a[1]=pd.DataFrame([{'security_id':'A','quantity':100}]);a[5].loc[a[5].index[:30],'A']=np.nan
    result=advise_portfolio(*a)
    assert result['status']=='data_error'
    assert '持仓' in result['reason'] and 'A' in result['reason'] and '风险' in result['reason']


@pytest.mark.parametrize('changes',[{'lot_valid_from':'2025-01-01'},{'lot_valid_from':None},{'lot_valid_to':'2023-12-31'}])
def test_lot_evidence_must_cover_signal_date(changes):
    a=list(inputs())
    for field,value in changes.items():a[4].loc[0,field]=value
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    assert result['recommendations'].security_id.tolist()==['B']
    assert '交易单位' in result['excluded'].set_index('security_id').loc['A','reason']


def test_existing_holding_with_unverified_lot_date_is_frozen():
    a=list(inputs());a[4].loc[0,'lot_valid_from']=None;a[1]=pd.DataFrame([{'security_id':'A','quantity':100}])
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    row=result['recommendations'].set_index('security_id').loc['A']
    assert row.target_weight>0 and pd.isna(row.order_quantity)
    assert '交易单位' in row.reason


def test_exact_factor_risk_reconciles_to_target_weights_using_all_market_columns():
    a=list(inputs());rng=np.random.default_rng(123)
    market=rng.normal(0,.008,300)
    noise=rng.normal(0,.004,300)
    hist=pd.DataFrame({'A':1.2*market+noise,'B':.8*market-noise,'outside':market*1.7+.002},index=pd.bdate_range(end='2024-05-31',periods=300))
    a[5]=hist
    result=advise_portfolio(*a)
    assert result['status']=='ok'
    window=hist.iloc[-252:];factor=window.median(axis=1)
    market_var=factor.var(ddof=1)*252
    expected=0.;exposure=0.
    for row in result['recommendations'].itertuples():
        paired=pd.concat([window[row.security_id],factor],axis=1).dropna()
        x=np.column_stack([np.ones(len(paired)),paired.iloc[:,1]])
        intercept,beta=np.linalg.lstsq(x,paired.iloc[:,0],rcond=None)[0]
        residual=paired.iloc[:,0].to_numpy()-x@np.array([intercept,beta])
        variance=float(residual@residual/(len(paired)-2)*252)
        exposure+=beta*row.target_weight
        expected+=variance*row.target_weight**2
        evidence=result['summary']['risk_estimates'][row.security_id]
        assert np.isclose(evidence['beta'],beta) and np.isclose(evidence['residual_variance_annual'],variance)
    expected+=market_var*exposure**2
    assert np.isclose(result['summary']['annual_volatility'],np.sqrt(expected))
    assert result['summary']['risk_window_sessions']==252
    assert result['summary']['risk_observations']=={'A':252,'B':252}
    changed=hist.copy();changed.iloc[:-252]=999
    a[5]=pd.concat([changed,pd.DataFrame({'A':[100.],'B':[-100.],'outside':[100.]},index=pd.to_datetime(['2024-06-01']))])
    after=advise_portfolio(*a)
    assert result['summary']['risk_estimates']==after['summary']['risk_estimates']
    pd.testing.assert_frame_equal(result['recommendations'],after['recommendations'])
