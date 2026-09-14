import json
import numpy as np
import pandas as pd
from hk_quant import features


def test_build_market_uses_prebuilt_reit_information_returns(tmp_path,monkeypatch):
    dates=pd.bdate_range('2020-01-01',periods=70)
    def bars(security):
        return pd.DataFrame(dict(date=dates,security_id=security,raw_close=100.,adj_close=100.,adj_close_hkd=100.,
            raw_open=100.,high=100.,low=100.,open_adj=100.,high_adj=100.,low_adj=100.,
            quote_present=True,data_valid=True,fx_to_hkd=1.,volume=1000.,amount=100000.,amount_hkd=100000.,
            total_mv=np.nan,free_mv=np.nan,total_share=np.nan,free_share=np.nan,turnover_ratio=np.nan))
    equity=bars('00001.HK');reit=bars('00778.HK')
    prebuilt=features.security_features(reit,dates)
    reit.loc[60:,['raw_close','adj_close','adj_close_hkd']]=90.
    prebuilt.loc[60:,['raw_close','adj_close_hkd']]=90.
    (tmp_path/'bars').mkdir();(tmp_path/'references').mkdir();(tmp_path/'reit_price_features').mkdir()
    pd.concat([equity,reit]).to_parquet(tmp_path/'bars/2020.parquet',index=False)
    prebuilt.to_parquet(tmp_path/'reit_price_features/00778.HK.parquet',index=False)
    master=pd.DataFrame([dict(security_id=s,exchange_code=s,asset_type=a,currency='HKD',
        identity_status='verified',identity_valid_from=dates[0],identity_valid_to=pd.NaT,
        list_date=dates[0],delist_date=pd.NaT) for s,a in [('00001.HK','equity'),('00778.HK','reit')]])
    master.to_parquet(tmp_path/'securities.parquet',index=False)
    pd.DataFrame(dict(cal_date=dates,is_open=1)).to_parquet(tmp_path/'references/calendar.parquet',index=False)
    pd.DataFrame({'metric':pd.Series(dtype=str),'verified':pd.Series(dtype=bool),'source_url':pd.Series(dtype=str)}).to_parquet(tmp_path/'financial_versions.parquet',index=False)
    (tmp_path/'data_audit.json').write_text(json.dumps({'start':'2020-01-01','end':str(dates[-1].date())}),encoding='utf-8')
    offer=tmp_path/'data/hk/raw/corporate_events_research';offer.mkdir(parents=True)
    pd.DataFrame(columns=['code','offer_start','offer_end']).to_csv(offer/'sfc_offer_periods_all.csv',index=False)
    monkeypatch.setattr(features,'ROOT',tmp_path)
    monkeypatch.setattr(features,'index_rate_features',lambda directory,calendar:pd.DataFrame({'date':calendar}))
    features.build_features(tmp_path)
    result=pd.read_parquet(tmp_path/'features')
    assert np.allclose(result.loc[result.date.eq(dates[-1]),'market_momentum_20'],0.)
    assert len(result)==140
    master.loc[master.security_id.eq('00001.HK'),'asset_type']='preference'
    master.to_parquet(tmp_path/'securities.parquet',index=False)
    equity.loc[60:,['raw_close','adj_close','adj_close_hkd']]=200.
    pd.concat([equity,reit]).to_parquet(tmp_path/'bars/2020.parquet',index=False)
    features.build_features(tmp_path)
    scoped=pd.read_parquet(tmp_path/'features')
    assert scoped.loc[scoped.security_id.eq('00001.HK'),'status'].eq('outside_model_asset_scope').all()
    assert np.allclose(scoped.loc[scoped.date.eq(dates[-1]),'market_momentum_20'],0.)
