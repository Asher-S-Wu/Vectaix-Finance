import importlib.util
import numpy as np
import pandas as pd
import pytest


def module(name):
    assert importlib.util.find_spec('ashare_quant.' + name) is not None, 'A-share core module is not implemented'
    return __import__('ashare_quant.' + name, fromlist=['*'])


def test_market_namespaces_do_not_overlap_hk():
    p = module('paths')
    from hk_quant.paths import DATA, MODELS, RESULTS
    assert p.DATA != DATA and p.MODELS != MODELS and p.RESULTS != RESULTS
    assert '/cn/' in str(p.DATA)


def test_normalize_tushare_units_keeps_cny_raw_and_adjusted_distinct():
    m = module('data')
    daily = pd.DataFrame([dict(ts_code='600000.SH', trade_date='20200102',open=10.,high=11.,low=9.,close=10.5,pre_close=10.,vol=200.,amount=210.)])
    basic = pd.DataFrame([dict(ts_code='600000.SH',trade_date='20200102',total_mv=100.,circ_mv=80.,total_share=10.,float_share=8.,free_share=6.,turnover_rate=2.)])
    factors=pd.DataFrame([dict(ts_code='600000.SH',trade_date='20200102',adj_factor=2.)])
    limits=pd.DataFrame([dict(ts_code='600000.SH',trade_date='20200102',up_limit=11.,down_limit=9.)])
    result,audit=m.normalize_daily(daily,basic,factors,limits)
    r=result.iloc[0]
    assert r.volume==20000 and r.amount==210000
    assert r.total_mv==1000000 and r.free_mv==630000
    assert r.total_share==100000 and r.free_share==60000
    assert r.turnover_ratio==.02
    assert r.raw_close==10.5 and r.adj_close==21.
    assert r.quote_present and r.data_valid and audit['missing_adjustment_rows']==0


def test_normalize_does_not_invent_adjustment_or_limits():
    m=module('data')
    d=pd.DataFrame([dict(ts_code='000001.SZ',trade_date='20200102',open=1.,high=1.,low=1.,close=1.,pre_close=1.,vol=1.,amount=.1)])
    out,audit=m.normalize_daily(d,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    assert not out.iloc[0].data_valid and pd.isna(out.iloc[0].up_limit)
    assert audit['missing_adjustment_rows']==1


def test_duplicate_daily_keys_are_rejected():
    m=module('data')
    d=pd.DataFrame([dict(ts_code='000001.SZ',trade_date='20200102')]*2)
    with pytest.raises(ValueError,match='duplicate'):
        m.normalize_daily(d,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())


def test_historical_master_keeps_delisted_and_uses_listing_dates():
    m=module('data')
    master=m.normalize_securities(pd.DataFrame([
      dict(ts_code='000001.SZ',name='A',list_date='19910403',delist_date=None,market='主板'),
      dict(ts_code='600001.SH',name='B',list_date='19960101',delist_date='20220101',market='主板'),
      dict(ts_code='200001.SZ',name='B share',list_date='19920101',delist_date=None,market='主板')]))
    assert set(master.security_id)=={'000001.SZ','600001.SH'}
    assert set(m.universe_asof(master,'20210101').security_id)=={'000001.SZ','600001.SH'}
    assert set(m.universe_asof(master,'20230101').security_id)=={'000001.SZ'}


def test_announcement_time_join_does_not_use_future_or_restatement():
    m=module('data')
    obs=pd.DataFrame(dict(date=pd.to_datetime(['2020-04-01','2020-04-02','2020-04-15','2020-05-02']),security_id=['600000.SH']*4))
    records=pd.DataFrame(dict(security_id=['600000.SH']*2,ann_date=['20200401','20200501'],end_date=['20191231']*2,roe=[10.,20.]))
    out=m.fundamentals_asof(obs,records,['roe'])
    assert pd.isna(out.roe.iloc[0])
    assert out.roe.iloc[1:].tolist()==[10.,10.,20.]


def bars_fixture():
    dates=pd.bdate_range('2018-01-01',periods=400)
    p=10*np.exp(np.arange(400)*.001 + np.sin(np.arange(400))*.01)
    return pd.DataFrame(dict(date=dates,security_id='600000.SH',raw_close=p,adj_close=p,raw_open=p*.999,high=p*1.02,low=p*.98,adj_factor=1.,volume=1e6,amount=p*1e6,total_mv=p*1e8,free_mv=p*6e7,total_share=1e8,free_share=6e7,turnover_ratio=.01,quote_present=True,data_valid=True)), dates


def test_features_causal_and_labels_calendar_based_without_hkd_outputs():
    m=module('features'); bars,cal=bars_fixture()
    full=m.security_features(bars,cal)
    changed=bars.copy();changed.loc[changed.index>300,'adj_close']*=3
    again=m.security_features(changed,cal)
    names=[c for c in full if not c.startswith(('fwd_return_','label_end_'))]
    pd.testing.assert_frame_equal(full.loc[:300,names],again.loc[:300,names])
    assert not any('hkd' in c for c in full)
    assert full.label_end_20.iloc[100]==cal[120]
    assert full.fwd_return_20.iloc[100]==pytest.approx(bars.adj_close.iloc[120]/bars.adj_close.iloc[100]-1)
    assert full.status.iloc[10]=='insufficient_history'


def test_model_adapter_currency_and_split_purging():
    m=module('models');t=module('training')
    assert m.FORECAST_RETURN_BASIS.startswith('CNY')
    frame=pd.DataFrame(dict(date=pd.to_datetime(['20200101','20200601','20210101','20210701']),security_id=['600000.SH']*4,status='ok',volatility_20=.2,sigma_daily=.2/np.sqrt(252),momentum_5=.1))
    for h in (1,5,20,60):
        frame[f'fwd_return_{h}']=.1;frame[f'label_end_{h}']=pd.to_datetime(['20201231','20210102','20210201','20220101'])
    train,cal=t.split_window(frame,['momentum_5'],'20211231',calibration_start='20210101')
    assert (train.label_end<pd.Timestamp('20210101')).all()
    assert (cal.label_end<=pd.Timestamp('20211231')).all()
    assert train.date.nunique()==1 and cal.date.nunique()==1


def test_beijing_pre_exchange_otc_history_is_not_ashare_membership():
    m=module('data')
    frame=m.normalize_securities(pd.DataFrame([dict(ts_code='920001.BJ',name='A',list_date='20200727',delist_date=None)]))
    assert frame.list_date.iloc[0]==pd.Timestamp('2021-11-15')
    assert frame.source_list_date.iloc[0]==pd.Timestamp('2020-07-27')
    assert m.universe_asof(frame,'2021-11-12').empty


def test_star_cdr_master_lot_matches_execution_rule():
    m=module('data')
    master=m.normalize_securities(pd.DataFrame([dict(ts_code='689009.SH',name='Fixture CDR',list_date='20201029',delist_date=None)]))
    from ashare_quant.rules import lot_rule
    assert master.lot_size.iloc[0]==lot_rule('689009.SH')[0]==200


def test_adjustment_reference_mismatch_is_not_used_for_features():
    m=module('features');bars,calendar=bars_fixture()
    bars['pre_close']=bars.raw_close.shift().fillna(bars.raw_close)
    bars.loc[250:,'adj_factor']=2.
    bars['adj_close']=bars.raw_close*bars.adj_factor
    with pytest.raises(ValueError,match='adjustment'):
        m.security_features(bars,calendar,quarantine_unresolved=False)


def test_consistent_cash_ex_date_adjustment_is_accepted():
    m=module('features');bars,calendar=bars_fixture()
    bars['pre_close']=bars.raw_close.shift().fillna(bars.raw_close)
    before=bars.raw_close.iloc[249];cash=.2
    multiplier=before/(before-cash)
    bars.loc[250:,'adj_factor']=multiplier
    bars.loc[250,'pre_close']=before-cash
    bars['adj_close']=bars.raw_close*bars.adj_factor
    assert m.security_features(bars,calendar).status.iloc[250]=='ok'


def test_unresolved_adjustment_quarantines_crossing_labels_and_resets_history():
    m=module('features');bars,calendar=bars_fixture()
    bars['pre_close']=bars.raw_close.shift().fillna(bars.raw_close)
    bars.loc[250:,'adj_factor']=2.;bars['adj_close']=bars.raw_close*bars.adj_factor
    out=m.security_features(bars,calendar)
    assert pd.isna(out.fwd_return_20.iloc[240])
    assert out.status.iloc[250]=='unresolved_adjustment'
    assert out.status.iloc[251]=='insufficient_history'
    assert out.status.iloc[330]=='ok'
    assert len(out.attrs['adjustment_issues'])==1
    assert out.fwd_return_5.iloc[230]==pytest.approx(bars.adj_close.iloc[235]/bars.adj_close.iloc[230]-1)


def test_future_adjustment_break_cannot_change_prior_feature_values():
    m=module('features');bars,calendar=bars_fixture()
    bars['pre_close']=bars.raw_close.shift().fillna(bars.raw_close)
    original=m.security_features(bars,calendar)
    changed=bars.copy();changed.loc[350:,'adj_factor']=2.;changed['adj_close']=changed.raw_close*changed.adj_factor
    altered=m.security_features(changed,calendar)
    names=[c for c in original if not c.startswith(('fwd_return_','label_end_'))]
    left=original.loc[:300,names].copy();right=altered.loc[:300,names].copy()
    left.attrs={};right.attrs={}
    pd.testing.assert_frame_equal(left,right)
