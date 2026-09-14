import numpy as np
import pandas as pd

from hk_quant.prediction_tasks import TASK_STATUS_COLUMNS, TASK_COLUMNS, classify_task, aggregate_task_status
from hk_quant.models import UniversalModel
from hk_quant.bundle import MultiTaskBundle
from test_models import fixture
from test_bundle import inputs


def test_each_task_classifies_only_its_own_values():
    frame = pd.DataFrame({'score': [.2, np.nan], 'probability_up': [.6, 1.2],
        'baseline_probability': [.5, .5], 'expected_return': [-1., -1.1],
        'q10': [-1., .3], 'q50': [0., .2], 'q90': [.5, .1],
        'baseline_q10': [-.2, -.2], 'baseline_q50': [0., 0.], 'baseline_q90': [.2, .2]})
    assert classify_task(frame, 'score').tolist() == ['ok', 'unrepresentable_prediction']
    assert classify_task(frame, 'probability_up').tolist() == ['ok', 'unrepresentable_prediction']
    assert classify_task(frame, 'expected_return').tolist() == ['ok', 'invalid_return_range']
    assert classify_task(frame, 'intervals').tolist() == ['ok', 'invalid_quantile_order']
    states = pd.DataFrame({field: classify_task(frame, task) for task, field in TASK_STATUS_COLUMNS.items()})
    assert aggregate_task_status(states).tolist() == ['ok', 'invalid_quantile_order']


def test_crossed_intervals_preserve_actual_rank_probability_and_return():
    training, calibration, examples = fixture()
    model = UniversalModel('lightgbm_small').fit(training, calibration, ['x'], '2023-06-09')
    normal = model.predict(examples, explain=True)
    class FixedQuantile:
        def __init__(self, value): self.value = value
        def predict(self, matrix): return np.full(len(matrix), self.value, dtype=float)
    model.quantile_models[.1] = FixedQuantile(.3)
    model.quantile_models[.5] = FixedQuantile(.2)
    model.quantile_models[.9] = FixedQuantile(.1)
    for horizon in (1, 5, 20, 60):
        model.quantile_affine[horizon] = {'location': 0., 'scale': 1.}
    actual = model.predict(examples, explain=True)
    assert actual.interval_status.eq('invalid_quantile_order').all()
    assert actual[TASK_COLUMNS['intervals']].isna().all().all()
    for task in ('score', 'probability_up', 'expected_return'):
        assert actual[TASK_STATUS_COLUMNS[task]].eq('ok').all()
        pd.testing.assert_frame_equal(actual[TASK_COLUMNS[task]], normal[TASK_COLUMNS[task]])
    assert actual.explanations.map(bool).all()
    assert actual.status.eq('invalid_quantile_order').all()


def test_bundle_ignores_failures_in_unselected_tasks_but_keeps_selected_failure():
    models, heads, examples = inputs()
    bundle = MultiTaskBundle(models, heads, 'task-output-test')
    predictions = {kind: model.predict(examples) for kind, model in models.items()}
    for frame in predictions.values():
        for field in TASK_STATUS_COLUMNS.values():
            frame[field] = 'ok'
    # Ranking/return model has bad intervals; interval head comes from a different model.
    predictions['linear']['status'] = 'invalid_quantile_order'
    predictions['linear']['interval_status'] = 'invalid_quantile_order'
    predictions['linear'][TASK_COLUMNS['intervals']] = np.nan
    output = bundle.combine_predictions(examples, predictions)
    assert output.status.eq('ok').all()
    assert output.score.eq(.2).all() and output.q50.eq(.3).all()
    # Once the selected interval head fails, the other three tasks remain available.
    predictions['lightgbm_small']['status'] = 'invalid_quantile_order'
    predictions['lightgbm_small']['interval_status'] = 'invalid_quantile_order'
    predictions['lightgbm_small'][TASK_COLUMNS['intervals']] = np.nan
    failed = bundle.combine_predictions(examples, predictions)
    assert failed.status.eq('invalid_quantile_order').all()
    assert failed[TASK_COLUMNS['intervals']].isna().all().all()
    assert failed.score.eq(.2).all() and failed.probability_up.eq(.1).all()
    assert failed.expected_return.eq(.2).all()
