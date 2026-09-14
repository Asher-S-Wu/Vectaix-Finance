import pickle
import numpy as np
import pandas as pd
import pytest
from hk_quant.models import UniversalModel


def fixture():
    rng=np.random.default_rng(42)
    def make(start,days,n):
        rows=[]
        for date in pd.date_range(start,periods=days):
            for h in (1,5,20,60):
                for i in range(n):
                    x=rng.normal();sigma=.02
                    z=.5*x+rng.normal(0,.5)
                    rows.append(dict(date=date,security_id=str(i),horizon=h,fwd_return=np.expm1(z*sigma*np.sqrt(h)),label_end=date+pd.Timedelta(days=h),sigma_daily=sigma,x=x))
        return pd.DataFrame(rows)
    train=make('2023-01-01',10,40);cal=make('2023-04-01',5,40)
    examples=cal.iloc[:4].copy();examples['horizon']=[1,5,20,60];examples['date']=pd.Timestamp('2023-06-10')
    return train,cal,examples


def test_simple_factor_baseline_has_fixed_weights_and_usable_four_horizon_outputs():
    train,cal,examples=fixture()
    names=['momentum_60','momentum_5','volatility_20','log_amount_20']
    for frame in (train,cal,examples):
        for i,name in enumerate(names):
            frame[name]=1/(1+np.exp(-frame.x*(i+1)))
    model=UniversalModel('factor').fit(train,cal,names,'2023-06-09')
    predictions=model.predict(examples,explain=True)
    assert predictions.status.eq('ok').all()
    assert predictions.probability_up.between(0,1).all()
    assert predictions.q10.le(predictions.q50).all() and predictions.q50.le(predictions.q90).all()
    assert model.parameters['weights']==dict(zip(names,[.25,-.25,-.25,.25]))
    for i,row in predictions.iterrows():
        expected=((2*examples.iloc[i][names].to_numpy(float)-1)*np.array([.25,-.25,-.25,.25])).sum()
        assert np.isclose(row.score,expected)
        assert np.isclose(sum(item['contribution'] for item in row.explanations),
                          expected*examples.sigma_daily.iloc[i]*np.sqrt(row.horizon))
    pd.testing.assert_frame_equal(predictions,pickle.loads(pickle.dumps(model)).predict(examples,explain=True))


def test_probability_baseline_uses_same_history_despite_model_missingness_policy():
    train,cal,examples=fixture()
    positive=train.index[train.fwd_return.gt(0)]
    train.loc[positive[::2],'x']=np.nan
    linear=UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')
    tree=UniversalModel('lightgbm_small').fit(train,cal,['x'],'2023-06-09')
    assert linear.baseline_probability==tree.baseline_probability
    for h in (1,5,20,60):
        assert linear.baseline_probability[h]==float(train.loc[train.horizon.eq(h),'fwd_return'].gt(0).mean())


@pytest.mark.parametrize('kind',['linear','lightgbm_small'])
def test_real_fit_all_horizons_probabilities_intervals_and_pickle(kind):
    train,cal,examples=fixture()
    model=UniversalModel(kind=kind,model_version='test').fit(train,cal,['x'],as_of='2023-06-09')
    out=model.predict(examples,explain=True)
    assert (out.status=='ok').all()
    assert out.probability_up.between(0,1).all()
    assert (out.q10<=out.q50).all() and (out.q50<=out.q90).all()
    assert (out.q10>-1).all()
    assert set(out.horizon)=={1,5,20,60}
    assert (out.return_basis=='HKD source-adjusted price return').all()
    assert model.metadata['return_basis']=='HKD source-adjusted price return'
    assert 'separately by replay' in model.metadata['cash_dividend_accounting']
    assert all(len(v)>0 for v in out.explanations)
    assert all(v[0]['feature'] in ('x','log_horizon') for v in out.explanations)
    restored=pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(out,restored.predict(examples,explain=True))
    assert model.metadata['latest_label_end']<='2023-06-09T00:00:00'


def test_label_boundary_is_rejected():
    train,cal,_=fixture();train.loc[0,'label_end']=cal.date.min()
    with pytest.raises(ValueError,match='label_end'):
        UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')
    train,cal,_=fixture();cal.loc[0,'label_end']='2024-01-01'
    with pytest.raises(ValueError,match='label_end'):
        UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')


def test_labels_never_enter_prediction_and_bad_rows_preserved():
    train,cal,x=fixture();model=UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')
    expected=model.predict(x)
    x['fwd_return']=100.;x['label_end']=pd.Timestamp('2040-01-01')
    actual=model.predict(x)
    pd.testing.assert_series_equal(expected.expected_return,actual.expected_return)
    x.loc[x.index[0],'x']=np.nan;x.loc[x.index[1],'sigma_daily']=0
    x.loc[x.index[2],'date']=pd.Timestamp('2020-01-01');x['status']='ok';x.loc[x.index[3],'status']='blocked'
    out=model.predict(x)
    assert len(out)==4 and (out.status=='insufficient_model_inputs').all()
    assert out.expected_return.isna().all()
    assert out.reasons.map(bool).all()


def test_lgb_native_missing_but_missing_feature_column_is_not_invented():
    train,cal,x=fixture();train.loc[0,'x']=np.nan
    m=UniversalModel('lightgbm_small').fit(train,cal,['x'],'2023-06-09')
    x['x']=np.nan
    out=m.predict(x)
    assert out.status.isin(['ok','invalid_quantile_order']).all()
    assert out.status.eq('ok').any()
    assert out.loc[out.status.eq('ok'),'expected_return'].notna().all()
    assert (m.predict(x.drop(columns='x')).status=='insufficient_model_inputs').all()


def test_missing_horizon_or_direction_rejected_without_constant_model():
    train,cal,_=fixture()
    with pytest.raises(ValueError):UniversalModel('linear').fit(train[train.horizon!=60],cal,['x'],'2023-06-09')
    train.loc[train.horizon==1,'fwd_return']=.1
    with pytest.raises(ValueError):UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')


def test_linear_explanation_is_local_coefficient_contribution():
    train,cal,x=fixture();m=UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')
    out=m.predict(x,explain=True)
    matrix=pd.DataFrame({'x':x.x.to_numpy(),'log_horizon':np.log(x.horizon.to_numpy())})
    standardized=m.scaler.transform(matrix)
    expected=standardized*m.regressor.coef_[None,:]*np.sqrt(x.horizon.to_numpy())[:,None]
    for i,factors in enumerate(out.explanations):
        values={factor['feature']:factor['contribution'] for factor in factors}
        assert np.isclose(values['x'],expected[i,0])
        assert np.isclose(values['log_horizon'],expected[i,1])
        assert out.explanation_basis.iloc[i]=='predicted_arithmetic_return'


def test_lgb_nonnumeric_feature_does_not_raise_or_score():
    train,cal,x=fixture();m=UniversalModel('lightgbm_small').fit(train,cal,['x'],'2023-06-09')
    x['x']=x.x.astype(object);x.loc[x.index[0],'x']='invalid'
    out=m.predict(x)
    assert out.iloc[0].status=='insufficient_model_inputs'
    assert out.iloc[1].status=='ok'


def test_future_target_cannot_be_requested_as_feature():
    train,cal,_=fixture()
    with pytest.raises(ValueError):UniversalModel('linear').fit(train,cal,['x','fwd_return'],'2023-06-09')


def test_twenty_day_explanation_comes_from_ranker_and_sigma_stays_absolute():
    train,cal,x=fixture();m=UniversalModel('lightgbm_small').fit(train,cal,['x'],'2023-06-09')
    out=m.predict(x,explain=True)
    row=out[out.horizon==20].iloc[0]
    assert row.explanation_basis=='rank_score'
    selected=x[x.horizon==20]
    matrix=pd.DataFrame({'x':selected.x.to_numpy(),'log_horizon':np.log(selected.horizon.to_numpy())})
    contributions=m.ranker.predict(matrix,pred_contrib=True)[0,:-1]
    actual={factor['feature']:factor['contribution'] for factor in row.explanations}
    assert np.isclose(actual['x'],contributions[0])
    x['sigma_daily']=.04
    doubled=m.predict(x)
    np.testing.assert_allclose(np.log1p(doubled.baseline_q90),2*np.log1p(out.baseline_q90))


def test_prediction_data_date_is_separate_from_training_cutoff():
    tr,ca,x=fixture();m=UniversalModel('linear').fit(tr,ca,['x'],'2023-06-09')
    out=m.predict(x)
    assert (out.data_as_of=='2023-06-10').all()
    assert (out.model_trained_as_of=='2023-06-09T00:00:00').all()
    assert m.metadata['as_of']=='2023-06-09T00:00:00'


@pytest.mark.parametrize('kind',['linear','lightgbm_small'])
def test_batch_prediction_matches_chunks_with_invalid_rows(kind):
    tr,ca,x=fixture();m=UniversalModel(kind).fit(tr,ca,['x'],'2023-06-09')
    batch=pd.concat([x]*20,ignore_index=True)
    batch.loc[3,'sigma_daily']=0.;batch.loc[7,'date']=pd.Timestamp('2020-01-01')
    batch.loc[11,'x']=np.nan;batch['status']='ok';batch.loc[13,'status']='blocked'
    whole=m.predict(batch)
    chunks=pd.concat([m.predict(batch.iloc[i:i+7]) for i in range(0,len(batch),7)],ignore_index=True)
    pd.testing.assert_frame_equal(whole,chunks,rtol=1e-12,atol=1e-12)
