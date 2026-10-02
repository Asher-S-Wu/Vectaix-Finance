import copy
import numpy as np
import pandas as pd
import pytest

from us_quant.data import parse_chart, write_immutable_json
from us_quant.features import security_features, FEATURE_COLUMNS, normalize_features
from us_quant.training import split_examples


def chart():
    return {'chart':{'error':None,'result':[{'meta':{'symbol':'AAPL','currency':'USD','instrumentType':'EQUITY','exchangeTimezoneName':'America/New_York'},'timestamp':[1577975400,1578061800],'indicators':{'quote':[{'open':[74,75],'high':[76,77],'low':[73,74],'close':[75,76],'volume':[100,120]}],'adjclose':[{'adjclose':[70,71]}]},'events':{'dividends':{'1':{'date':1578061800,'amount':.2}}}}]}}


def bars(n=400):
    d=pd.bdate_range('2020-01-01',periods=n);p=100*np.exp(np.arange(n)*.001+np.sin(np.arange(n))*.01)
    return pd.DataFrame({'date':d,'security_id':'A','split_adjusted_open':p*.998,'split_adjusted_high':p*1.01,'split_adjusted_low':p*.99,'split_adjusted_close':p,'adj_close':p*.95,'volume':1e6,'dollar_volume_proxy':p*1e6,'quote_present':True,'data_valid':True})


def test_parse_chart_preserves_split_adjusted_and_total_return_prices():
    df,meta,events=parse_chart(chart(),'AAPL')
    assert df.split_adjusted_close.tolist()==[75,76]
    assert df.adj_close.tolist()==[70,71]
    assert 'raw_close' not in df
    assert df.date.dt.strftime('%Y-%m-%d').tolist()==['2020-01-02','2020-01-03']
    assert df.dollar_volume_proxy.tolist()==[7500,9120]
    assert events['dividends']


@pytest.mark.parametrize('field,value',[('symbol','OTHER'),('currency','HKD'),('instrumentType','CRYPTOCURRENCY')])
def test_parse_chart_rejects_wrong_identity_or_market(field,value):
    p=chart();p['chart']['result'][0]['meta'][field]=value
    with pytest.raises(ValueError):parse_chart(p,'AAPL')


def test_parse_chart_rejects_duplicate_dates():
    p=chart();p['chart']['result'][0]['timestamp'][1]=p['chart']['result'][0]['timestamp'][0]
    with pytest.raises(ValueError,match='duplicate'):parse_chart(p,'AAPL')


def test_missing_quote_is_retained_invalid_not_forward_filled():
    p=chart();p['chart']['result'][0]['indicators']['quote'][0]['close'][1]=None
    f,_,_=parse_chart(p,'AAPL');assert len(f)==2 and not f.data_valid.iloc[1]


def test_protocol_is_immutable(tmp_path):
    p=tmp_path/'protocol.json';write_immutable_json(p,{'x':1});write_immutable_json(p,{'x':1})
    with pytest.raises(ValueError,match='immutable'):write_immutable_json(p,{'x':2})


def test_features_have_no_future_leakage():
    b=bars();a=security_features(b,b.date)
    changed=b.copy(); changed.loc[changed.index>=300,['adj_close','split_adjusted_close']]*=2
    z=security_features(changed,b.date)
    pd.testing.assert_frame_equal(a.loc[:299,FEATURE_COLUMNS],z.loc[:299,FEATURE_COLUMNS])
    assert a.loc[290,'fwd_return_20'] != z.loc[290,'fwd_return_20']


def test_missing_calendar_session_does_not_compress_horizon():
    b=bars();calendar=b.date.copy();b=b.drop(index=300)
    f=security_features(b,calendar)
    assert f.loc[300,'status']=='no_trade_quote'
    assert pd.isna(f.loc[299,'fwd_return_1'])
    assert f.loc[299,'label_end_1']==calendar.iloc[300]
    assert pd.isna(f.loc[399,'fwd_return_1'])


def test_normalization_preserves_unranked_sigma():
    a=security_features(bars(),bars().date);b=a.copy();b.security_id='B';b.volatility_20*=2
    f=normalize_features(pd.concat([a,b],ignore_index=True));expected=pd.concat([a,b],ignore_index=True).volatility_20/np.sqrt(252)
    np.testing.assert_allclose(f.sigma_daily,expected,equal_nan=True)


def test_temporal_split_purges_boundary_outcomes():
    d=pd.bdate_range('2022-01-03','2023-12-29');n=len(d)
    f=pd.DataFrame({'date':d,'security_id':'A','status':'ok','sigma_daily':.01,'x':1.})
    for h in (1,5,20,60):f[f'fwd_return_{h}']=.1;f[f'label_end_{h}']=pd.Series(d).shift(-h).values
    train,cal=split_examples(f,['x'],'2023-01-01','2023-12-29')
    assert train.label_end.max()<pd.Timestamp('2023-01-01')
    assert cal.date.min()>=pd.Timestamp('2023-01-01')
    assert cal.label_end.max()<=pd.Timestamp('2023-12-29')
    assert len(train)>0 and len(cal)>0
