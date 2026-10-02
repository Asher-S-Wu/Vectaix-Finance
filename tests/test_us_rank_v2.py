import numpy as np
import pandas as pd
import pytest
from us_quant.ranking_v2 import RankingModel, rank_targets, candidate_settings

FEATURES=['momentum_60','momentum_5','volatility_20','log_amount_20','momentum_252_skip20']

def examples(year):
    rng=np.random.default_rng(year);rows=[]
    for date in pd.bdate_range(f'{year}-01-02',periods=35):
        for s in range(22):
            x=rng.uniform(size=5);ret=(x[0]-.5)*.1+rng.normal(scale=.03)
            rows.append(dict(date=date,security_id=str(s),horizon=20,fwd_return=ret,label_end=date+pd.Timedelta(days=28),status='ok',sigma_daily=.02,**dict(zip(FEATURES,x))))
    return pd.DataFrame(rows)


def test_rank_targets_ignore_shared_market_return_and_keep_unknown():
    f=examples(2020);a=rank_targets(f)
    f.fwd_return+=.7;b=rank_targets(f)
    np.testing.assert_allclose(a,b)
    f.loc[0,'fwd_return']=np.nan;assert pd.isna(rank_targets(f).iloc[0])


def test_factor_reference_is_original_direction_not_inverted():
    m=RankingModel('factor_reference').fit(examples(2020),examples(2021),FEATURES,'2021-12-31')
    f=examples(2022);p=m.predict(f)
    expected=.5*(f.momentum_60-f.momentum_5-f.volatility_20+f.log_amount_20)
    np.testing.assert_allclose(p.score,expected)
    assert m.metadata['fitted_ranking_weights'] is False


def test_ridge_ranking_fit_predict_and_independent_task_outputs():
    m=RankingModel('ridge_rank_3y').fit(examples(2020),examples(2021),FEATURES,'2021-12-31')
    p=m.predict(examples(2022))
    assert p.score_status.eq('ok').all()
    assert p.probability_up.between(0,1).all()
    assert (p.q10<=p.q50).all() and (p.q50<=p.q90).all()
    assert (p.q10>=-1).all()
    assert m.metadata['ranking_target']=='within_date_horizon_centered_percentile'
    assert m.metadata['market']=='US' and m.metadata['currency']=='USD'


def test_v2_rejects_labels_crossing_calibration():
    train=examples(2020);train.loc[0,'label_end']=pd.Timestamp('2021-01-04')
    with pytest.raises(ValueError,match='purge'):RankingModel('ridge_rank_3y').fit(train,examples(2021),FEATURES,'2021-12-31')


def test_v2_does_not_score_unavailable_or_pre_fit_dates():
    m=RankingModel('momentum_12_1').fit(examples(2020),examples(2021),FEATURES,'2021-12-31')
    f=examples(2022);f.loc[0,'status']='source_history_unavailable';f.loc[1,'date']=pd.Timestamp('2020-01-01')
    p=m.predict(f);assert p.iloc[:2].score.isna().all()


def test_fixed_budget_has_six_candidates():
    assert len(candidate_settings())==6


def test_rank_target_minimum_and_tied_label_center_are_fixed():
    f=examples(2020).iloc[:19].copy();assert rank_targets(f).isna().all()
    f=examples(2020);f.fwd_return=1.;np.testing.assert_allclose(rank_targets(f),0.,atol=1e-15)


def test_unknown_training_outcome_is_not_negative_probability_baseline():
    tr=examples(2020);tr.loc[0,'fwd_return']=np.nan
    m=RankingModel('factor_reference').fit(tr,examples(2021),FEATURES,'2021-12-31')
    expected=tr.loc[tr.fwd_return.notna(),'fwd_return'].gt(0).mean()
    assert m.baseline_probability[20]==pytest.approx(expected)


def test_interval_calibration_audit_is_saved():
    m=RankingModel('factor_reference').fit(examples(2020),examples(2021),FEATURES,'2021-12-31')
    a=m.metadata['interval_fit_audit']['20'];assert a['fit_rows']==770 and a['excluded_invalid_point_or_scale_rows']==0


def test_future_label_columns_cannot_be_features():
    with pytest.raises(ValueError,match='feature'):RankingModel('ridge_rank_3y').fit(examples(2020),examples(2021),FEATURES+['fwd_return'],'2021-12-31')


def test_label_end_cannot_precede_signal_date():
    tr=examples(2020);tr.loc[0,'label_end']=tr.loc[0,'date']-pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='label'):RankingModel('ridge_rank_3y').fit(tr,examples(2021),FEATURES,'2021-12-31')
