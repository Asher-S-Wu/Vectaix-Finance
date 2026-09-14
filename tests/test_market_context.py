import numpy as np
import pandas as pd

from hk_quant.market_context import market_context_from_inputs, risk_returns_from_inputs


def test_market_uses_daily_information_returns_not_snapshot_price_levels():
    dates=pd.bdate_range('2020-01-01',periods=70)
    a=pd.DataFrame(dict(date=dates,security_id='A',return_1=.01,bias_60=.1,quote_present=True,adj_close_hkd=100.))
    b=pd.DataFrame(dict(date=dates,security_id='B',return_1=-.01,bias_60=-.1,quote_present=True,adj_close_hkd=200.))
    b.loc[69,'quote_present']=False
    source=pd.concat([a,b],ignore_index=True)
    actual=market_context_from_inputs(source,dates)
    assert np.isclose(actual.iloc[-1].market_momentum_20,.01)
    assert actual.iloc[-1].market_breadth_60==1
    source['adj_close_hkd']=np.arange(len(source))+1000.
    pd.testing.assert_frame_equal(actual,market_context_from_inputs(source,dates))


def test_risk_keeps_pit_dividend_return_and_identity_boundary():
    dates=pd.bdate_range('2020-01-01',periods=4)
    source=pd.DataFrame(dict(date=dates,security_id='A',return_1=[np.nan,0.,.01,-.5],
                             adj_close_hkd=[100.,99.,99.99,49.995]))
    master=pd.DataFrame(dict(security_id=['A'],asset_type=['equity'],identity_status=['verified'],identity_valid_from=[dates[0]],identity_valid_to=[pd.NaT]))
    risk=risk_returns_from_inputs(source,master)
    assert risk.A.iloc[1]==0. and risk.A.iloc[3]==-.5
    master['identity_valid_from']=dates[2]
    scoped=risk_returns_from_inputs(source,master)
    assert scoped.A.iloc[:3].isna().all()
    assert scoped.A.iloc[3]==-.5
    master['asset_type']='preference'
    assert risk_returns_from_inputs(source,master).A.isna().all()
