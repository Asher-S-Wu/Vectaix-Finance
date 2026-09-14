import pickle

import numpy as np
import pandas as pd
import pytest

from hk_quant.models import UniversalModel, NUMERIC_OUTPUTS


FACTOR_FEATURES = ['momentum_60', 'momentum_5', 'volatility_20', 'log_amount_20']


def full_loss_samples():
    def make(start, days):
        rows = []
        for date in pd.date_range(start, periods=days):
            for horizon in (1, 5, 20, 60):
                for i in range(40):
                    loss = i < 20
                    rows.append(dict(date=date, security_id=str(i), horizon=horizon,
                                     fwd_return=-1.0 if loss else .2,
                                     label_end=date + pd.Timedelta(days=horizon),
                                     sigma_daily=.02, x=-1. if loss else 1.,
                                     **{feature: .5 for feature in FACTOR_FEATURES}))
        return pd.DataFrame(rows)
    train, calibration = make('2023-01-01', 10), make('2023-04-01', 5)
    examples = calibration.iloc[:4].copy()
    examples['date'] = pd.Timestamp('2023-06-10')
    examples['horizon'] = [1, 5, 20, 60]
    return train, calibration, examples


@pytest.mark.parametrize('kind', ['factor', 'linear', 'lightgbm_small', 'lightgbm_large'])
def test_full_loss_is_a_training_outcome_and_below_full_loss_is_excluded(kind):
    train, calibration, examples = full_loss_samples()
    invalid = train.iloc[[0]].copy()
    invalid['security_id'] = 'invalid-below-minus-one'
    invalid['fwd_return'] = -1.01
    train = pd.concat([train, invalid], ignore_index=True)
    features = FACTOR_FEATURES if kind == 'factor' else ['x']
    model = UniversalModel(kind).fit(train, calibration, features, '2023-06-09')
    assert model.metadata['sample_counts']['train']['used'] == 1600
    assert model.metadata['sample_counts']['train']['excluded'] == 1
    assert model.metadata['sample_counts']['calibration']['used'] == 800
    for horizon in (1, 5, 20, 60):
        assert model.metadata['sample_counts']['train']['horizons'][str(horizon)]['negative_or_zero'] == 200
        assert model.baseline_probability[horizon] == .5
        assert model.metadata['sample_counts']['train']['horizons'][str(horizon)]['full_loss'] == 200
    if kind == 'linear':
        expected_intercept = -.4 * np.mean(1 / np.sqrt([1, 5, 20, 60]))
        assert model.regressor.intercept_ == pytest.approx(expected_intercept)
    if kind.startswith('lightgbm'):
        matrix = pd.DataFrame({'x': [-1., 1.], 'log_horizon': [np.log(20), np.log(20)]})
        assert model.classifier.predict_proba(matrix)[0, 1] < model.classifier.predict_proba(matrix)[1, 1]
        ranking = model.ranker.predict(matrix)
        assert ranking[0] < ranking[1]


def test_factor_mean_and_intervals_retain_exact_minus_one_outcome():
    train, calibration, examples = full_loss_samples()
    model = UniversalModel('factor').fit(train, calibration, FACTOR_FEATURES, '2023-06-09')
    output = model.predict(examples, explain=True)
    assert output.status.eq('ok').all()
    np.testing.assert_allclose(output.expected_return, -.4, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(output.q10, -np.ones(4))
    np.testing.assert_allclose(output.q50, -.4, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(output.q90, np.full(4, .2))
    assert output.explanation_basis.eq('predicted_arithmetic_return').all()
    assert model.metadata['target_schema'] == 'arithmetic-return-sqrt-horizon-v1'
    pd.testing.assert_frame_equal(output, pickle.loads(pickle.dumps(model)).predict(examples, explain=True))


@pytest.mark.parametrize('output', ['expected_return', 'lower_quantile'])
def test_predictions_below_minus_one_remain_unavailable_without_clipping(output):
    train, calibration, examples = full_loss_samples()
    model = UniversalModel('factor').fit(train, calibration, FACTOR_FEATURES, '2023-06-09')
    if output == 'expected_return':
        model.mean_return_residual[20] = -1.01
    else:
        model.residual_quantiles[20][0] = -1.01
    predictions = model.predict(examples)
    invalid = predictions.loc[predictions.horizon.eq(20)].iloc[0]
    assert invalid.status == 'invalid_return_range'
    if output == 'expected_return':
        assert pd.isna(invalid.expected_return)
        assert invalid[['q10','q50','q90']].notna().all()
    else:
        assert invalid[['q10','q50','q90']].isna().all()
        assert pd.notna(invalid.expected_return)
    assert invalid[['score','probability_up','baseline_probability']].notna().all()
    assert '-100%' in invalid.reasons
    assert predictions.loc[predictions.horizon.ne(20), 'status'].eq('ok').all()


@pytest.mark.parametrize('old_schema', [None, 'log-return-v5'])
def test_historical_target_schema_is_rejected_before_prediction(old_schema):
    train, calibration, examples = full_loss_samples()
    train.loc[train.fwd_return.eq(-1), 'fwd_return'] = -.5
    calibration.loc[calibration.fwd_return.eq(-1), 'fwd_return'] = -.5
    model = UniversalModel('linear').fit(train, calibration, ['x'], '2023-06-09')
    if old_schema is None:
        model.__dict__.pop('target_schema', None)
    else:
        model.target_schema = old_schema
    with pytest.raises(ValueError, match='收益目标版本'):
        model.predict(examples)
