from pathlib import Path
import pytest
from us_quant.pipeline import require_fresh_run


def test_training_rejects_complete_or_partial_results_before_writing(tmp_path):
    data=tmp_path/'data';results=tmp_path/'results';models=tmp_path/'models';results.mkdir()
    (results/'training_status.json').write_text('{"status":"running"}')
    with pytest.raises(FileExistsError,match='new'):require_fresh_run(data,results,models,'train')
    assert not models.exists()


def test_training_rejects_untracked_existing_model(tmp_path):
    models=tmp_path/'models';models.mkdir();(models/'factor.pkl').write_bytes(b'keep')
    with pytest.raises(FileExistsError):require_fresh_run(tmp_path/'data',tmp_path/'results',models,'train')
    assert (models/'factor.pkl').read_bytes()==b'keep'


def test_build_refuses_derived_data_and_accepts_new_namespace(tmp_path):
    data=tmp_path/'data';data.mkdir();results=tmp_path/'results';models=tmp_path/'models'
    require_fresh_run(data,results,models,'build')
    (data/'normalized.parquet').write_bytes(b'keep')
    with pytest.raises(FileExistsError):require_fresh_run(data,results,models,'build')
    assert not results.exists()


def test_fresh_model_name_cannot_overwrite_data_predictions(tmp_path):
    data=tmp_path/'data';(data/'predictions').mkdir(parents=True)
    (data/'predictions/latest.parquet').write_bytes(b'keep')
    with pytest.raises(FileExistsError):require_fresh_run(data,tmp_path/'new_results',tmp_path/'new_models','train')


def test_fresh_frozen_subdir_cannot_overwrite_sibling_latest(tmp_path):
    models=tmp_path/'models';(models/'latest').mkdir(parents=True)
    (models/'latest/model.pkl').write_bytes(b'keep')
    with pytest.raises(FileExistsError):require_fresh_run(tmp_path/'data',tmp_path/'results',models/'fresh_frozen','train')
