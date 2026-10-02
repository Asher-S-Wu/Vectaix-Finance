import numpy as np
import pandas as pd
import pytest
from us_quant.training import compare_candidates, freeze_selection, split_examples, protocol
from us_quant.models import USModel


def predictions(constant=False):
    rows=[]
    for d in pd.to_datetime(['2024-01-02','2024-01-03']):
        for s in range(20):rows.append(dict(date=d,security_id=str(s),horizon=20,score=1 if constant and d.day==3 else s,score_status='ok',fwd_return=s/100,label_end=d+pd.Timedelta(days=28)))
    return pd.DataFrame(rows)


def test_comparison_uses_same_days_and_zero_for_constant_predictions():
    r=compare_candidates({'factor':predictions(),'linear':predictions(True)},'2024-12-31')
    assert r['factor']['ic_dates']==r['linear']['ic_dates']==2
    assert r['factor']['date_sha256']==r['linear']['date_sha256']
    assert r['factor']['rank_ic_mean']==pytest.approx(1)
    assert r['linear']['rank_ic_mean']==pytest.approx(.5)
    assert r['linear']['constant_score_days']==1


def test_comparison_rejects_disagreeing_labels():
    a=predictions();b=predictions();b.loc[0,'fwd_return']=10
    with pytest.raises(ValueError,match='labels'):compare_candidates({'factor':a,'linear':b},'2024-12-31')


def test_selection_is_development_only_and_immutable(tmp_path):
    scores={k:dict(rank_ic_mean=v) for k,v in [('factor',.1),('linear',.2),('lightgbm_small',.05),('lightgbm_large',.15)]}
    r=freeze_selection(scores,tmp_path/'freeze.json');assert r['selected_kind']=='linear' and not r['confirmation_used']
    scores['factor']['rank_ic_mean']=.3
    with pytest.raises(ValueError,match='immutable'):freeze_selection(scores,tmp_path/'freeze.json')


def test_warmup_data_never_in_fit():
    dates=pd.to_datetime(['2014-06-01','2016-06-01','2023-06-01']);f=pd.DataFrame(dict(date=dates,security_id='X',status='ok',sigma_daily=.01,x=1.))
    for h in (1,5,20,60):f[f'fwd_return_{h}']=.1;f[f'label_end_{h}']=dates+pd.Timedelta(days=h)
    t,c=split_examples(f,['x']);assert t.date.min()>=pd.Timestamp('2016-01-01')


def test_us_model_does_not_share_currency_identity():
    assert USModel().market=='US'
    p=protocol('abc');assert p['currency']=='USD' and p['horizons']==[1,5,20,60]
    assert p['point_in_time_vendor_vintages'] is False
