import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from hk_quant.models import UniversalModel, NUMERIC_OUTPUTS
from test_models import fixture


@pytest.mark.parametrize('kind',['linear','lightgbm_small','lightgbm_large'])
def test_near_zero_sigma_true_jump_does_not_amplify_fitted_target(kind):
    train,cal,examples=fixture()
    jump=(train.security_id=='0')
    train.loc[jump,'fwd_return']=.05
    model=UniversalModel(kind).fit(train,cal,['x'],'2023-06-09')
    train.loc[jump,'sigma_daily']=2.743181e-9
    stressed=UniversalModel(kind).fit(train,cal,['x'],'2023-06-09')
    ordinary=model.predict(examples)
    actual=stressed.predict(examples)
    assert actual.status.eq('ok').all()
    assert np.isfinite(actual[list(NUMERIC_OUTPUTS)].to_numpy()).all()
    assert actual.expected_return.abs().max()<.5
    assert max(abs(v) for v in stressed.mean_return_residual.values())<.1
    pd.testing.assert_frame_equal(ordinary,actual)
    if kind=='linear':
        assert np.abs(stressed.regressor.coef_).max()<.1


@pytest.mark.parametrize('kind',['linear','lightgbm_small','lightgbm_large'])
def test_prediction_sigma_changes_only_volatility_baseline_not_fitted_returns(kind):
    train,cal,examples=fixture()
    model=UniversalModel(kind).fit(train,cal,['x'],'2023-06-09')
    ordinary=model.predict(examples)
    examples['sigma_daily']=2.743181e-9
    actual=model.predict(examples)
    pd.testing.assert_frame_equal(ordinary[['score','probability_up','expected_return']],actual[['score','probability_up','expected_return']])
    if kind=='linear':
        assert not np.allclose(ordinary[['q10','q50','q90']],actual[['q10','q50','q90']])
    else:
        np.testing.assert_allclose(ordinary[['q10','q50','q90']],actual[['q10','q50','q90']])
    assert (actual.baseline_q90<ordinary.baseline_q90).all()


@pytest.mark.parametrize('kind',['lightgbm_small','lightgbm_large'])
def test_real_quantile_models_calibrate_each_horizon_and_ignore_prediction_labels(kind):
    train,cal,examples=fixture()
    model=UniversalModel(kind).fit(train,cal,['x'],'2023-06-09')
    assert set(model.quantile_models)=={.1,.5,.9}
    matrix=pd.DataFrame({'x':cal.x.to_numpy(),'log_horizon':np.log(cal.horizon.to_numpy())})
    for alpha,regressor in model.quantile_models.items():
        assert isinstance(regressor,lgb.LGBMRegressor)
        assert regressor.get_params()['objective']=='quantile' and regressor.get_params()['alpha']==alpha
        assert regressor.get_params()['num_leaves']==model.parameters['num_leaves']
        assert regressor.get_params()['n_estimators']==model.parameters['n_estimators']
        return_quantile=regressor.predict(matrix)*np.sqrt(cal.horizon.to_numpy())
        for h in (1,5,20,60):
            mask=cal.horizon.eq(h)
            assert np.isfinite(return_quantile[mask]).all()
        assert model.quantile_affine[h]['scale'] > 0
        assert model.quantile_affine[h]['domain'] == 'log1p_arithmetic_return'
    predictions=model.predict(examples)
    assert predictions.status.eq('ok').all() and set(predictions.horizon)=={1,5,20,60}
    assert predictions.q10.le(predictions.q50).all() and predictions.q50.le(predictions.q90).all()
    pd.testing.assert_frame_equal(predictions,pickle.loads(pickle.dumps(model)).predict(examples))
    examples['fwd_return']=999.;examples['label_end']=pd.Timestamp('2099-01-01')
    pd.testing.assert_frame_equal(predictions[list(NUMERIC_OUTPUTS)],model.predict(examples)[list(NUMERIC_OUTPUTS)])


def test_crossed_quantiles_are_unavailable_and_never_sorted():
    train,cal,examples=fixture()
    model=UniversalModel('lightgbm_small').fit(train,cal,['x'],'2023-06-09')
    class FixedQuantile:
        def __init__(self, value): self.value = value
        def predict(self, matrix): return np.full(len(matrix), self.value, dtype=float)
    model.quantile_models[.1] = FixedQuantile(.3)
    model.quantile_models[.5] = FixedQuantile(.2)
    model.quantile_models[.9] = FixedQuantile(.1)
    predictions=model.predict(examples)
    assert predictions.status.eq('invalid_quantile_order').all()
    assert predictions[['q10','q50','q90','baseline_q10','baseline_q50','baseline_q90']].isna().all().all()
    assert predictions[['score','probability_up','expected_return','baseline_probability']].notna().all().all()
    assert predictions.reasons.str.contains('交叉').all()


def test_linear_intervals_use_actual_arithmetic_residual():
    train,cal,examples=fixture()
    model=UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')
    matrix=pd.DataFrame({'x':cal.x.to_numpy(),'log_horizon':np.log(cal.horizon.to_numpy())})
    scale=np.sqrt(cal.horizon.to_numpy())
    predicted_return=model.regressor.predict(model.scaler.transform(matrix))*scale
    for h in (1,5,20,60):
        mask=cal.horizon.eq(h)
        point=predicted_return[mask]+model.mean_return_residual[h]
        standardized=(np.log1p(cal.loc[mask,'fwd_return'].to_numpy(float))-np.log1p(point))/(
            cal.loc[mask,'sigma_daily'].to_numpy(float)*np.sqrt(h))
        expected=np.quantile(standardized,[.1,.5,.9],method='inverted_cdf')
        np.testing.assert_allclose(model.interval_log_residual_quantiles[h],expected)


def test_linear_intervals_use_common_empirical_log_residuals():
    train, cal, examples = fixture()
    model = UniversalModel('linear').fit(train, cal, ['x'], '2023-06-09')
    assert model.metadata['calibration']['intervals'] == 'per-horizon empirical log1p-return residual distribution'
    assert set(model.interval_log_residual_quantiles) == {1, 5, 20, 60}
    predictions = model.predict(examples)
    assert predictions.interval_status.eq('ok').all()
    assert predictions.q10.le(predictions.q50).all() and predictions.q50.le(predictions.q90).all()


def test_nonranking_tree_explanations_use_arithmetic_return_horizon_scale():
    train,cal,examples=fixture()
    model=UniversalModel('lightgbm_small').fit(train,cal,['x'],'2023-06-09')
    predictions=model.predict(examples,explain=True)
    matrix=pd.DataFrame({'x':examples.x.to_numpy(),'log_horizon':np.log(examples.horizon.to_numpy())})
    contributions=model.regressor.predict(matrix,pred_contrib=True)
    for i,row in predictions.iterrows():
        if row.horizon==20:
            continue
        assert row.explanation_basis=='predicted_arithmetic_return'
        explained=sum(item['contribution'] for item in row.explanations)
        scale=np.sqrt(row.horizon)
        assert np.isclose(explained,contributions[i,:-1].sum()*scale)
        assert np.isclose(explained+contributions[i,-1]*scale,model.regressor.predict(matrix.iloc[[i]])[0]*scale)
