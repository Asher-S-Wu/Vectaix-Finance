import numpy as np
import pandas as pd
import pytest
from hk_quant.minute_audit import audit_daily_observations


def bars():
    return pd.DataFrame({'ts_code':'00823.HK','trade_time':['2026-09-10 09:30:00','2026-09-10 16:10:00'],
        'open':[10.,11.],'close':[10.,11.],'high':[10.,11.],'low':[10.,11.],'vol':[100.,200.],'amount':[1000.,2200.]})


def test_diagnostics_do_not_approve_daily_data_or_infer_full_sessions():
    result=audit_daily_observations(bars(),pd.to_datetime(['2026-09-10']))
    row=result.iloc[0]
    assert row.bars==2 and row.volume_sum==300 and np.isclose(row.amount_volume_ratio,3200/300)
    assert row.is_exchange_session and row.ratio_within_reported_day_range
    assert not row.approved_for_daily_training


def test_missing_amount_is_not_treated_as_zero_and_bad_days_remain_visible():
    frame=bars();frame.loc[0,'amount']=np.nan
    result=audit_daily_observations(frame,[])
    assert result.iloc[0].invalid_numeric_bars==1
    assert pd.isna(result.iloc[0].amount_sum) and not result.iloc[0].is_exchange_session
    with pytest.raises(ValueError,match='重复'):
        audit_daily_observations(pd.concat([frame,frame]),[])
