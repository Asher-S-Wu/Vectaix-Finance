import pickle
import numpy as np
import pandas as pd
import pytest

from hk_quant.bundle import MultiTaskBundle
from hk_quant.models import NUMERIC_OUTPUTS
from hk_quant.prediction_tasks import TASK_COLUMNS,TASK_STATUS_COLUMNS


class Component:
    def __init__(self,value,failed=False,as_of='2023-12-29'):
        self.value=value;self.failed=failed;self.as_of=pd.Timestamp(as_of)
        self.feature_columns=['factor'];self.metadata={'component_value':value}
    def predict(self,examples,explain=False):
        out=examples.copy().reset_index(drop=True)
        for column in NUMERIC_OUTPUTS:out[column]=self.value
        out['status']='insufficient_model_inputs' if self.failed else 'ok'
        for field in TASK_STATUS_COLUMNS.values():out[field]=out['status']
        if self.failed:out[list(NUMERIC_OUTPUTS)]=np.nan
        out['reasons']='missing factor' if self.failed else ''
        if explain:
            out['explanations']=[[{'feature':'factor','contribution':self.value,'basis':'test','feature_value':1.}] for _ in range(len(out))]
        return out


def inputs():
    models={'factor':Component(.1),'linear':Component(.2),'lightgbm_small':Component(.3)}
    heads={str(h):{'score':'linear','probability_up':'factor','intervals':'lightgbm_small','expected_return':'linear'} for h in (1,5,20,60)}
    frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-02']*4),'security_id':'A','horizon':[1,5,20,60],'factor':1.})
    return models,heads,frame


def test_each_output_uses_its_fixed_task_model_and_survives_pickle():
    models,heads,frame=inputs();bundle=MultiTaskBundle(models,heads,'bundle-v4')
    out=bundle.predict(frame,explain=True)
    assert out.status.eq('ok').all() and out.score.eq(.2).all()
    assert out.probability_up.eq(.1).all() and out.q50.eq(.3).all()
    assert out.expected_return.eq(.2).all() and out.model_version.eq('bundle-v4').all()
    assert out.return_basis.eq('HKD source-adjusted price return').all()
    assert bundle.metadata['return_basis']=='HKD source-adjusted price return'
    assert 'separately by replay' in bundle.metadata['cash_dividend_accounting']
    assert {x['task'] for x in out.explanations.iloc[0]}=={'score','expected_return'}
    assert {x['component_model'] for x in out.explanations.iloc[0]}=={'linear'}
    pd.testing.assert_frame_equal(out,pickle.loads(pickle.dumps(bundle)).predict(frame,explain=True))
    cached={kind:model.predict(frame,explain=True) for kind,model in bundle.models.items()}
    pd.testing.assert_frame_equal(out,bundle.combine_predictions(frame,cached,explain=True))


def test_missing_task_never_switches_to_another_component_or_drops_security():
    models,heads,frame=inputs();models['factor'].failed=True
    out=MultiTaskBundle(models,heads,'v4').predict(frame)
    assert len(out)==4 and out.status.eq('insufficient_model_inputs').all()
    assert out[TASK_COLUMNS['probability_up']].isna().all().all()
    assert out.score.eq(.2).all() and out.q50.eq(.3).all() and out.expected_return.eq(.2).all()
    assert out.reasons.str.contains('probability_up').all()


@pytest.mark.parametrize('status', ['invalid_return_range', 'unrepresentable_prediction', 'invalid_quantile_order'])
def test_bundle_preserves_model_output_failure_separately_from_missing_inputs(status):
    models, heads, frame = inputs()
    models['factor'].failed = True
    bundle = MultiTaskBundle(models, heads, 'v6')
    predictions = {kind: model.predict(frame) for kind, model in models.items()}
    predictions['lightgbm_small']['status'] = status
    predictions['lightgbm_small']['interval_status'] = status
    predictions['lightgbm_small']['reasons'] = 'Model output unavailable'
    predictions['lightgbm_small'][TASK_COLUMNS['intervals']] = np.nan
    output = bundle.combine_predictions(frame, predictions)
    assert output.status.eq(status).all()
    assert len(output) == 4
    assert output[TASK_COLUMNS['intervals']+TASK_COLUMNS['probability_up']].isna().all().all()
    assert output.score.eq(.2).all() and output.expected_return.eq(.2).all()


def test_bundle_rejects_missing_component_or_mismatched_training_cutoffs():
    models,heads,_=inputs()
    with pytest.raises(ValueError,match='缺少配置'):MultiTaskBundle({'linear':models['linear']},heads,'v4')
    models['factor'].as_of=pd.Timestamp('2024-01-01')
    with pytest.raises(ValueError,match='截止日期'):MultiTaskBundle(models,heads,'v4')


def test_real_linear_rank_explanations_use_score_units_not_return_units():
    from test_models import fixture
    from hk_quant.models import UniversalModel
    train,cal,examples=fixture()
    model=UniversalModel('linear').fit(train,cal,['x'],'2023-06-09')
    heads={str(h):{task:'linear' for task in ('score','probability_up','intervals','expected_return')} for h in (1,5,20,60)}
    bundle=MultiTaskBundle({'linear':model},heads,'v4')
    out=bundle.predict(examples,explain=True)
    matrix=model._matrix(examples)
    expected=model.scaler.transform(matrix)*model.regressor.coef_[None,:]
    for index,row in out.iterrows():
        actual={item['feature']:item for item in row.explanations if item['task']=='score'}
        assert actual['x']['basis']=='rank_score'
        assert np.isclose(actual['x']['contribution'],expected[index,0])
