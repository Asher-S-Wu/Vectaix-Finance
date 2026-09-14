import numpy as np
import pandas as pd

from hk_quant.reit_dataset import build_reit_feature_history
from hk_quant.features import reit_features_asof,reit_forward_labels


def raw():
    dates=pd.bdate_range('2020-01-01',periods=12)
    prices=np.arange(10.,22.)
    return pd.DataFrame(dict(date=dates,security_id='HKREIT:4875',raw_close=prices,
        raw_open=prices,high=prices,low=prices,fx_to_hkd=1.,quote_present=True,
        data_valid=True,volume=1000.,amount=prices*1000,amount_hkd=prices*1000,
        total_mv=np.nan,free_mv=np.nan,total_share=np.nan,free_share=np.nan,turnover_ratio=np.nan))


def test_history_preserves_missing_day_and_matures_labels(tmp_path):
    bars=raw();calendar=pd.DatetimeIndex(bars.date);missing=calendar[3]
    bars=bars.drop(index=3)
    versions=pd.DataFrame(columns=['security_id','event_id','published_at'])
    result=build_reit_feature_history(bars,calendar,versions,
        calendar[-1].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19),'HKD',tmp_path)
    frame=pd.read_parquet(result['path']).set_index('date')
    assert len(frame)==12
    assert frame.loc[missing,'status']=='data_issue'
    assert pd.isna(frame.loc[missing,'raw_close'])
    assert pd.isna(frame.loc[calendar[2],'fwd_return_1'])
    assert np.isclose(frame.loc[calendar[0],'fwd_return_5'],15/10-1)
    assert pd.isna(frame.iloc[-1].fwd_return_1)
    assert frame.security_id.eq('HKREIT:4875').all()
    assert not result['approved_for_training']
    assert len(pd.read_parquet(tmp_path))==12


def test_history_cutoff_does_not_admit_later_quotes(tmp_path):
    bars=raw();calendar=pd.DatetimeIndex(bars.date)
    versions=pd.DataFrame(columns=['security_id','event_id','published_at'])
    result=build_reit_feature_history(bars,calendar,versions,
        calendar[5].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19),'HKD',tmp_path)
    frame=pd.read_parquet(result['path'])
    assert len(frame)==6 and frame.date.max()==calendar[5]
    assert pd.isna(frame.iloc[-1].fwd_return_1)


def test_segmented_build_matches_individual_daily_calls_across_revision(tmp_path):
    bars=raw();calendar=pd.DatetimeIndex(bars.date)
    original=dict(security_id='HKREIT:4875',event_id='dividend',event_type='cash_distribution',
        published_at=calendar[1].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=20),
        ex_date=calendar[4],cash_per_unit_decimal='1',cash_currency='HKD',
        conditional_cash_amount=True,cash_amount_status='conditional_pending_units',announcement_status='New')
    revision={**original,'published_at':calendar[6].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=20),
        'cash_per_unit_decimal':'.5','conditional_cash_amount':False,'cash_amount_status':'declared_payment_currency_amount'}
    versions=pd.DataFrame([original,revision]);cutoff=calendar[-1].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19)
    expected=[]
    for day in calendar:
        f=reit_features_asof(bars,calendar,versions,day.tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19),'HKD')
        labels=reit_forward_labels(bars,calendar,versions,day,cutoff,'HKD')
        item=f.merge(labels,on=['date','security_id']);item.attrs={};expected.append(item)
    expected=pd.concat(expected,ignore_index=True)
    audit=build_reit_feature_history(bars,calendar,versions,cutoff,'HKD',tmp_path)
    actual=pd.read_parquet(audit['path'])
    for name in ['date']+[f'label_end_{h}' for h in (1,5,20,60)]:
        actual[name]=actual[name].astype('datetime64[ns]')
        expected[name]=expected[name].astype('datetime64[ns]')
    pd.testing.assert_frame_equal(actual[expected.columns],expected)
    assert audit['adjustment_state_evaluations']<len(calendar)
