import importlib.util
import pickle
import json
import hashlib
import pandas as pd
import pytest
from tests.test_models import fixture
from ashare_quant.models import AShareModel
from ashare_quant.training import publish_research_snapshot


def test_daily_update_produces_candidate_without_replacing_active(tmp_path):
    assert importlib.util.find_spec('ashare_quant.daily_update') is not None,'daily candidate flow missing'
    from ashare_quant.daily_update import run_daily_update
    train,cal,examples=fixture();m=AShareModel('linear','daily-synthetic-test').fit(train,cal,['x'],'2023-06-09')
    data=tmp_path/'data';models=tmp_path/'models';results=tmp_path/'results';data.mkdir()
    out=m.predict(examples,explain=True)
    active=publish_research_snapshot(m,out,models,limitations=['synthetic test'],data_lineage={'fixture':'synthetic'})
    before=(models/'active.json').read_bytes()
    frame=examples[['date','security_id','sigma_daily','x']].iloc[:1].copy();frame['status']='ok';frame['date']=pd.Timestamp('2023-06-12')
    for h in (1,5,20,60):frame[f'fwd_return_{h}']=float('nan');frame[f'label_end_{h}']=pd.NaT
    frame.to_parquet(data/'latest_inputs.parquet',index=False)
    (data/'feature_manifest.json').write_text(json.dumps({'status':'complete','data_as_of':'2023-06-12'}))
    result=run_daily_update(data,models,results)
    assert result['status']=='candidate_only' and not result['published']
    assert (models/'active.json').read_bytes()==before
    candidate=pd.read_parquet(result['forecast_path'])
    assert set(candidate.horizon)=={1,5,20,60}
    assert candidate.model_trained_as_of.eq('2023-06-09T00:00:00').all()


def test_daily_update_rejects_modified_pickle(tmp_path):
    assert importlib.util.find_spec('ashare_quant.daily_update') is not None
    from ashare_quant.daily_update import load_active_model
    models=tmp_path/'models';models.mkdir()
    p=models/'model.pkl';p.write_bytes(b'bad bytes')
    (models/'active.json').write_text(json.dumps({'model_path':'model.pkl','model_sha256':'0'*64}))
    with pytest.raises(ValueError,match='hash'):
        load_active_model(models)
