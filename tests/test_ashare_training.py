import json
import pickle
import numpy as np
import pandas as pd
import pytest
from ashare_quant import training
from ashare_quant.models import AShareModel
from tests.test_models import fixture


def test_all_horizon_real_estimator_adapter_is_cny_and_roundtrips():
    train,cal,examples=fixture()
    model=AShareModel('linear','cn-unit-test').fit(train,cal,['x'],'2023-06-09')
    out=model.predict(examples,explain=True)
    assert out.return_basis.str.startswith('CNY').all()
    assert 'HKD' not in json.dumps(model.metadata)
    assert out.score_status.eq('ok').all()
    pd.testing.assert_frame_equal(out,pickle.loads(pickle.dumps(model)).predict(examples,explain=True))


def test_split_contract_is_saved_before_results_and_cannot_change(tmp_path):
    assert hasattr(training,'initialize_run'),'run initialization not implemented'
    contract={'training_as_of':'2023-12-29','development_start':'2024-01-01','development_end':'2024-12-31','confirmation_start':'2025-01-01','selection_rule':'maximum 20-session mean daily rank IC'}
    training.initialize_run(tmp_path,contract)
    assert json.loads((tmp_path/'protocol.json').read_text())==contract
    with pytest.raises(ValueError,match='protocol'):
        training.initialize_run(tmp_path,{**contract,'selection_rule':'maximum realized test return'})


def test_architecture_selection_uses_only_development_and_tie_break(tmp_path):
    assert hasattr(training,'freeze_architecture'),'architecture freeze not implemented'
    dev={'linear':{'horizons':{'20':{'rank_ic_mean':.05}}},'lightgbm_small':{'horizons':{'20':{'rank_ic_mean':.05}}},'lightgbm_large':{'horizons':{'20':{'rank_ic_mean':.03}}},'factor':{'horizons':{'20':{'rank_ic_mean':.01}}}}
    result=training.freeze_architecture(dev,tmp_path)
    assert result['selected_kind']=='linear'
    assert result['selection_set']=='development'
    with pytest.raises(ValueError,match='frozen'):
        training.freeze_architecture({**dev,'factor':{'horizons':{'20':{'rank_ic_mean':.99}}}},tmp_path)


def test_partition_evaluation_preserves_partial_tasks_and_maturity(tmp_path):
    assert hasattr(training,'evaluate_files'),'partitioned evaluation not implemented'
    train,cal,examples=fixture();model=AShareModel('linear','cn-test').fit(train,cal,['x'],'2023-06-09')
    out=model.predict(examples)
    out['label_end']=pd.Timestamp('2023-07-01')
    out.loc[0,'label_end']=pd.Timestamp('2024-01-01')
    path=tmp_path/'predictions.parquet';out.to_parquet(path,index=False)
    result=training.evaluate_files([path],'2023-12-31')
    assert result['horizons']['1']['mature_labels']==0
    assert result['horizons']['5']['mature_labels']==1
    assert result['horizons']['5']['probability_available']==1


def test_snapshot_publish_is_research_only_and_has_hashes(tmp_path):
    assert hasattr(training,'publish_research_snapshot'),'snapshot publication not implemented'
    train,cal,examples=fixture();m=AShareModel('linear','cn-snapshot').fit(train,cal,['x'],'2023-06-09')
    out=m.predict(examples,explain=True)
    active=training.publish_research_snapshot(m,out,tmp_path,limitations=['unit test'],data_lineage={'sample':'synthetic test only'})
    assert active['research_ready'] and not active['eligible']
    assert active['market']=='CN' and active['currency']=='CNY'
    assert len(active['artifact_sha256'])>=3
    assert json.loads((tmp_path/active['acceptance_path']).read_text())['execution_validated'] is False


def test_development_excludes_late_2024_labels_reaching_confirmation(tmp_path):
    assert hasattr(training,'evaluate_development'),'development cutoff not implemented'
    train,cal,examples=fixture();m=AShareModel('linear','boundary').fit(train,cal,['x'],'2023-06-09')
    out=m.predict(examples)
    out['date']=pd.Timestamp('2024-12-20');out['label_end']=pd.Timestamp('2025-01-17')
    out.loc[out.horizon.eq(1),'label_end']=pd.Timestamp('2024-12-23')
    path=tmp_path/'2024-12.parquet';out.to_parquet(path,index=False)
    result=training.evaluate_development([path])
    assert result['as_of']=='2024-12-31'
    assert result['horizons']['20']['mature_labels']==0
    assert result['horizons']['1']['mature_labels']==1


def test_training_sample_lineage_is_verified_before_fit(tmp_path):
    assert hasattr(training,'verify_dataset_lineage'),'lineage verification missing'
    from hashlib import sha256
    directory=tmp_path/'training_samples';directory.mkdir()
    p=directory/'2020.parquet';p.write_bytes(b'original test bytes')
    manifest={'training_sample_sha256':{'2020.parquet':sha256(p.read_bytes()).hexdigest()}}
    training.verify_dataset_lineage(tmp_path,manifest)
    p.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='lineage'):
        training.verify_dataset_lineage(tmp_path,manifest)


def test_frozen_model_bytes_and_manifest_are_verified_before_loading(tmp_path):
    assert hasattr(training,'load_frozen_model'),'frozen artifact verifier missing'
    from hashlib import sha256
    payload=pickle.dumps({'fixture':'synthetic artifact'})
    model=tmp_path/'frozen.pkl';model.write_bytes(payload)
    metadata=tmp_path/'metadata.json';metadata.write_text(json.dumps({'model_sha256':sha256(payload).hexdigest(),'feature_manifest_sha256':'expected'}))
    assert training.load_frozen_model(model,metadata,'expected')['fixture']=='synthetic artifact'
    model.write_bytes(b'changed artifact')
    with pytest.raises(ValueError,match='hash'):
        training.load_frozen_model(model,metadata,'expected')
