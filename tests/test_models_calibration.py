import numpy as np

from hk_quant.models import UniversalModel
from tests.test_models import fixture


def test_factor_probability_is_centered_and_uses_the_held_out_calibrator():
    train, calibration, examples = fixture()
    names = ['momentum_60', 'momentum_5', 'volatility_20', 'log_amount_20']
    for frame in (train, calibration, examples):
        for i, name in enumerate(names):
            frame[name] = 1 / (1 + np.exp(-frame.x * (i + 1)))
    model = UniversalModel('factor').fit(train, calibration, names, '2023-06-09')
    output = model.predict(examples)
    assert model.metadata['calibration']['probability_baseline_blend']['20'] == 0.0
    assert model.metadata['calibration']['probability_centering'] == 'date_horizon_cross_section'
    mask = examples.horizon.eq(20).to_numpy()
    x = model._matrix(examples)
    raw_margin = model._margin(x, model._regression(x))
    centered = model._center_margins(examples, raw_margin)
    raw_probability = model.calibrators[20].predict_proba(centered[mask, None])[:, 1]
    expected = raw_probability + model.probability_anchor_offsets[20]
    np.testing.assert_allclose(output.loc[mask, 'probability_up'], expected)


def test_new_experiment_has_no_manual_probability_or_interval_adjustments():
    train, calibration, examples = fixture()
    model = UniversalModel('linear').fit(train, calibration, ['x'], '2023-06-09')
    widths = model.metadata['calibration']['interval_width_multiplier']
    assert widths == {str(h): 1.0 for h in (1, 5, 20, 60)}
    assert model.probability_anchor_offsets == {h: 0.0 for h in (1, 5, 20, 60)}


def test_factor_probability_does_not_shift_a_held_out_calibrator_to_old_training_rate():
    train, calibration, examples = fixture()
    names = ['momentum_60', 'momentum_5', 'volatility_20', 'log_amount_20']
    for frame in (train, calibration, examples):
        for i, name in enumerate(names):
            frame[name] = 1 / (1 + np.exp(-frame.x * (i + 1)))
    model = UniversalModel('factor').fit(train, calibration, names, '2023-06-09')
    assert model.metadata['calibration']['probability_anchor'] == 'held_out_logistic_without_training_rate_shift'
    assert not hasattr(model, 'probability_anchor_shifts')
