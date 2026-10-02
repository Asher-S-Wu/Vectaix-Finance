import numpy as np
import pandas as pd
import pytest
from us_quant.recovery_v2 import reconstruct_original_bars


def raw():
    d=pd.bdate_range('2020-01-02',periods=4)
    return pd.DataFrame({'date':d,'open':[100,50,49,51],'high':[101,51,50,52],'low':[99,49,48,50],'close':[100,50,49,51],'volume':[10,20,20,20]})


def events():
    d=raw().date
    return pd.DataFrame({'ex_date':[d.iloc[1],d.iloc[2]],'split_ratio_new_per_old':[2.,1.],'cash_usd_per_pre_event_share':[0.,1.]})


def test_split_and_dividend_adjustment_is_multiplicative_not_affine():
    r=reconstruct_original_bars(raw(),events(),raw().date,'OLD')
    assert r.adj_close.iloc[1]/r.adj_close.iloc[0]==pytest.approx(1.)
    assert r.adj_close.iloc[2]/r.adj_close.iloc[1]==pytest.approx(1.)
    assert r.adj_close.iloc[3]/r.adj_close.iloc[2]==pytest.approx(51/49)
    assert r.dollar_volume_proxy.tolist()==[1000,1000,980,1020]


def test_recovered_calendar_gap_remains_unknown():
    f=raw().drop(index=3);r=reconstruct_original_bars(f,events(),raw().date,'OLD')
    assert len(r)==4 and not r.quote_present.iloc[3] and pd.isna(r.adj_close.iloc[3])


def test_unsupported_cash_event_prior_quote_blocks_reconstruction():
    f=raw();f.loc[1,'close']=np.nan
    with pytest.raises(ValueError,match='prior'):reconstruct_original_bars(f,events(),f.date,'OLD')


def test_material_price_disagreement_is_masked_without_choosing_other_source():
    f=raw();f['crosscheck_close_difference_usd']=0.;f.loc[3,'crosscheck_close_difference_usd']=1.
    r=reconstruct_original_bars(f,events(),f.date,'OLD')
    assert not r.data_valid.iloc[3] and pd.isna(r.adj_close.iloc[3])


def test_raw_values_after_original_identity_end_never_enter_features():
    f=raw();r=reconstruct_original_bars(f,events(),f.date,'OLD',valid_end=str(f.date.iloc[2].date()))
    assert not r.quote_present.iloc[3] and pd.isna(r.adj_close.iloc[3])
