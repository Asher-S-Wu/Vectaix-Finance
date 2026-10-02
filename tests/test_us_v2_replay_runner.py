import pandas as pd
import pytest
from scripts.run_us_v2_replay import paired_comparison


def test_paired_diagnostic_excludes_unmatched_names_and_unknown_outcomes():
    rows=[]
    for h in [1,5,20,60]:
        for d in pd.bdate_range('2025-01-02',periods=65):
            for i in range(20):rows.append({'date':d,'security_id':str(i),'horizon':h,'score':i,'score_status':'ok','fwd_return':i/100,'label_end':d+pd.Timedelta(days=1)})
    a=pd.DataFrame(rows);b=a.copy();b.score=-b.score
    r=paired_comparison(a,b,'2025-12-31')
    assert r['horizons']['20']['first_run_ic']==pytest.approx(1.)
    assert r['horizons']['20']['second_run_ic']==pytest.approx(-1.)
    assert r['horizons']['20']['paired_ic_change']==pytest.approx(-2.)
    assert r['horizons']['20']['ic_dates']==65


def test_paired_diagnostic_requires_matching_underlying_labels():
    a=pd.DataFrame([{'date':pd.Timestamp('2025-01-02'),'security_id':'A','horizon':20,'score':1.,'score_status':'ok','fwd_return':.1,'label_end':pd.Timestamp('2025-01-03')}]);b=a.copy();b.fwd_return=.2
    with pytest.raises(ValueError,match='labels'):paired_comparison(a,b,'2025-12-31')


def test_equivalent_label_dates_with_different_parquet_resolution_match():
    a=pd.DataFrame([{'date':pd.Timestamp('2025-01-02'),'security_id':'A','horizon':20,'score':1.,'score_status':'ok','fwd_return':.1,'label_end':pd.Timestamp('2025-01-03')}])
    a['date']=a.date.astype('datetime64[ms]');a['label_end']=a.label_end.astype('datetime64[ms]')
    b=a.copy();b['date']=b.date.astype('datetime64[us]');b['label_end']=b.label_end.astype('datetime64[us]')
    r=paired_comparison(a,b,'2025-12-31');assert r['horizons']['20']['common_mature_rows']==1
