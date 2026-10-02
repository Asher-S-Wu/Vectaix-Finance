import numpy as np
import pandas as pd
from us_quant.pipeline_v2 import fold_examples, select_kind, nested_selection


def frame():
    d=pd.bdate_range('2014-01-01','2026-09-30');f=pd.DataFrame({'date':d,'security_id':'A','status':'ok','sigma_daily':.01,'x':1.})
    for h in (1,5,20,60):f[f'fwd_return_{h}']=.1;f[f'label_end_{h}']=pd.Series(d).shift(-h).values
    return f


def test_fold_boundaries_exclude_future_and_truncate_early_window():
    tr,ca,asof=fold_examples(frame(),2020,5,['x'])
    assert tr.date.min()==pd.Timestamp('2016-01-01')
    assert tr.label_end.max()<pd.Timestamp('2019-01-01')
    assert ca.date.min()>=pd.Timestamp('2019-01-01')
    assert ca.label_end.max()<=pd.Timestamp('2019-12-31')
    assert tr.horizon.eq(20).all() and ca.horizon.eq(20).all()


def annual():
    return {str(y):{'a':{'rank_ic_mean':.1},'b':{'rank_ic_mean':.2 if y<2022 else -.5}} for y in range(2020,2025)}


def test_nested_outer_selection_never_uses_its_own_or_future_year():
    a=annual();x=nested_selection(a,['a','b'])
    assert x[0]['outer_year']==2022 and x[0]['selected_kind']=='b'
    a['2022']['a']['rank_ic_mean']=100
    a['2024']['a']['rank_ic_mean']=100
    z=nested_selection(a,['a','b'])
    assert z[0]['selected_kind']=='b'


def test_final_selection_uses_prescribed_years_and_tie_order():
    a=annual();kind,scores=select_kind(a,['a','b'],[2020,2021]);assert kind=='b'
    a['2020']['a']['rank_ic_mean']=.2;a['2021']['a']['rank_ic_mean']=.2
    assert select_kind(a,['a','b'],[2020,2021])[0]=='a'
