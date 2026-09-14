import json

import pytest

from hk_quant import daily_update


def test_daily_update_stops_when_data_build_is_incomplete(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    results = tmp_path / 'results'
    models = tmp_path / 'models'
    data.mkdir()
    (data / 'build_status.json').write_text(json.dumps({'status': 'building'}), encoding='utf-8')
    monkeypatch.setattr(daily_update, 'export_candidate', lambda *args, **kwargs: pytest.fail('不应生成候选'))
    with pytest.raises(ValueError, match='数据版本仍在构建'):
        daily_update.run_daily_update(data, results, models)


def test_daily_update_records_candidate_without_publishing(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    results = tmp_path / 'results'
    models = tmp_path / 'models'
    data.mkdir()
    (data / 'build_status.json').write_text(json.dumps({'status': 'complete'}), encoding='utf-8')
    monkeypatch.setattr(daily_update, 'export_candidate', lambda *args, **kwargs: {
        'status': 'candidate_only', 'eligible': False,
        'model_version': 'v5', 'data_as_of': '2026-09-11',
        'forecast_path': str(models / 'forecast.parquet'),
    })
    monkeypatch.setattr(daily_update, 'build_report', lambda *args, **kwargs: {
        'eligible': False, 'data_as_of': '20260911',
    })
    result = daily_update.run_daily_update(data, results, models)
    assert result['status'] == 'candidate_only'
    assert result['published'] is False
    saved = json.loads((results / 'daily_update_status.json').read_text(encoding='utf-8'))
    assert saved['model_version'] == 'v5'
    assert saved['published'] is False


def test_daily_update_never_claims_publication_without_registry_write(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    results = tmp_path / 'results'
    models = tmp_path / 'models'
    data.mkdir()
    (data / 'build_status.json').write_text(json.dumps({'status': 'complete'}), encoding='utf-8')
    monkeypatch.setattr(daily_update, 'export_candidate', lambda *args, **kwargs: {
        'status': 'candidate_only', 'eligible': True,
        'model_version': 'v5', 'data_as_of': '2026-09-11',
        'forecast_path': str(models / 'forecast.parquet'),
    })
    monkeypatch.setattr(daily_update, 'build_report', lambda *args, **kwargs: {
        'eligible': True, 'data_as_of': '20260911',
    })
    result = daily_update.run_daily_update(data, results, models)
    assert result['status'] == 'candidate_only'
    assert result['published'] is False
