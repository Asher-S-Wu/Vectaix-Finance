import json
import pandas as pd
from us_quant.pipeline import absent_features, validate_calendar, select_training_features
from us_quant.features import ALL_FEATURES


def test_absent_identity_kept_at_every_session():
    cal=pd.bdate_range('2024-01-01',periods=5);f=absent_features('OLD',cal,'identity_unresolved')
    assert len(f)==5 and f.security_id.eq('OLD').all()
    assert f.status.eq('identity_unresolved').all()
    assert f['fwd_return_20'].isna().all()


def test_calendar_rejects_duplicate_and_weekend():
    import pytest
    with pytest.raises(ValueError):validate_calendar(pd.to_datetime(['2025-01-04']))
    with pytest.raises(ValueError):validate_calendar(pd.to_datetime(['2025-01-02','2025-01-02']))


def test_training_availability_ignores_warmup_and_confirmation():
    f=pd.DataFrame({'date':pd.to_datetime(['2014-01-02','2016-01-04','2022-01-03','2025-01-02']),'status':'ok',**{x:[1.,None,None,1.] for x in ALL_FEATURES}})
    f['momentum_60']=[None,1.,1.,None]
    assert select_training_features(f,'lightgbm_small')==['momentum_60']


def test_identity_gate_rejects_reused_equity_and_etf_tickers():
    from us_quant.pipeline import validate_approved_identity
    import pytest
    row={'security_id':'APC','instrument':'EQUITY','provider_name':'ARKO Petroleum Corp.','first_date':'2026-02-12'}
    with pytest.raises(ValueError,match='historical identity'):validate_approved_identity(row)
    row={'security_id':'EMC','instrument':'ETF','provider_name':'Global X ETF','first_date':'2014-01-02'}
    with pytest.raises(ValueError,match='equity'):validate_approved_identity(row)
