import numpy as np
import pandas as pd

from hk_quant.training import make_examples, predict_frame, PREDICTION_OUTPUTS


class CenteredModel:
    feature_columns = ['x']

    def predict(self, examples):
        result = examples.copy()
        for column in PREDICTION_OUTPUTS:
            if column not in result:
                result[column] = 0.0
        result['score'] = result.x - result.groupby(['date', 'horizon'], dropna=False).x.transform('mean')
        return result


def test_batches_preserve_entire_date_horizon_cross_sections_and_original_order():
    # Same date interleaved in source security order, as real feature partitions are stored.
    frame = pd.DataFrame(dict(date=pd.to_datetime(['2020-01-02','2020-01-03'] * 9),
        security_id=np.repeat([f'S{i}' for i in range(9)], 2),
        status='ok', sigma_daily=.01, x=np.arange(18, dtype=float)))
    for horizon in (1, 5, 20, 60):
        frame[f'fwd_return_{horizon}'] = .01
        frame[f'label_end_{horizon}'] = pd.Timestamp('2020-04-01')
    model = CenteredModel()
    expected = model.predict(make_examples(frame, model.feature_columns))[list(PREDICTION_OUTPUTS)].reset_index(drop=True)
    for batch_size in (1, 10, 25):
        actual = predict_frame(model, frame, batch_size=batch_size)
        pd.testing.assert_frame_equal(actual, expected)


def test_unknown_date_is_preserved_in_batched_output():
    frame = pd.DataFrame(dict(date=pd.to_datetime(['2020-01-02',None]), security_id=['A','B'],
        status='ok', sigma_daily=.01, x=[1.,2.]))
    for horizon in (1, 5, 20, 60):
        frame[f'fwd_return_{horizon}'] = .01
        frame[f'label_end_{horizon}'] = pd.Timestamp('2020-04-01')
    actual = predict_frame(CenteredModel(), frame, batch_size=1)
    assert len(actual) == 8 and actual.date.isna().sum() == 4
