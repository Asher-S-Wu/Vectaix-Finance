import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import json

from hk_quant import training
from hk_quant.training import evaluate_prediction_files
from test_evaluation import pending_label_fixture
from hk_quant.prediction_tasks import TASK_COLUMNS, TASK_STATUS_COLUMNS


def test_file_evaluation_counts_missing_labels_across_all_horizons(tmp_path):
    rows = []
    for horizon in (1, 5, 20, 60):
        for security in ('A', 'B'):
            rows.append(dict(pending_label_fixture(), horizon=horizon,
                             security_id=security, status='ok',
                             fwd_return=np.nan if horizon == 20 and security == 'B' else .1))
    path = tmp_path / 'predictions.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    metrics = evaluate_prediction_files([path], '2024-01-31')
    assert metrics['valid'] is False
    assert metrics['matured_rows'] == 8
    assert metrics['matured_label_rows'] == 7
    assert metrics['missing_matured_label_rows'] == 1
    assert metrics['matured_label_coverage'] == .875
    assert metrics['label_coverage_complete'] is False


def test_month_record_is_derived_from_actual_model_and_forecast_rows():
    heads = {str(h): {'score': 'factor'} for h in (1, 5, 20, 60)}
    component = SimpleNamespace(probability_baseline_blend={20: 0.},
                                probability_anchor_offsets={20: -.005},
                                interval_width_multiplier={20: 1.05})
    model = SimpleNamespace(heads=heads, model_version='v5-bundle-202401',
                            as_of=pd.Timestamp('2023-12-29'), models={'factor': component})
    source = pd.DataFrame({'date': [pd.Timestamp('2024-01-02')], 'security_id': ['A']})
    forecasts = pd.concat([source.assign(horizon=h, model_version=model.model_version,
                                       model_trained_as_of='2023-12-29') for h in (1, 5, 20, 60)])
    record = training.confirmation_month_record(model, forecasts, source, pd.Period('2024-01'))
    assert record['prediction_coverage_complete'] is True
    assert record['model_revision'] == 'v5'
    assert record['trained_as_of'] == '2023-12-29T00:00:00'
    assert record['component_calibration_parameters']['factor']['interval_width_multiplier'] == {'20': 1.05}
    incomplete = training.confirmation_month_record(model, forecasts.iloc[:-1], source, pd.Period('2024-01'))
    assert incomplete['prediction_coverage_complete'] is False


@pytest.mark.parametrize('status', ['invalid_return_range', 'unrepresentable_prediction', 'invalid_quantile_order'])
@pytest.mark.parametrize('missing_label', [False, True])
def test_model_rejections_stay_in_mature_coverage_and_fail_complete_evaluation(tmp_path, status, missing_label):
    rows = []
    for horizon in (1, 5, 20, 60):
        good = dict(pending_label_fixture(), horizon=horizon, security_id='GOOD', status='ok')
        rejected = dict(good, security_id='EXTREME', status=status,
                        interval_status=status, fwd_return=np.nan if missing_label else -1.)
        unavailable = dict(good, security_id='NO_INPUTS', status='insufficient_model_inputs')
        for column in TASK_COLUMNS['intervals']:
            rejected[column] = np.nan
        for task, columns in TASK_COLUMNS.items():
            unavailable[TASK_STATUS_COLUMNS[task]] = 'insufficient_model_inputs'
            for column in columns:
                unavailable[column] = np.nan
        rows.extend([good, rejected, unavailable])
    path = tmp_path / 'predictions.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    metrics = evaluate_prediction_files([path], '2024-01-31')
    assert metrics['valid'] is False
    assert metrics['total_forecast_rows'] == 12
    assert metrics['scored_forecast_rows'] == 4
    assert metrics['input_unavailable_rows'] == 4
    assert metrics['input_eligible_forecast_rows'] == 8
    assert metrics['model_rejected_rows'] == 4
    assert metrics['matured_model_rejected_rows'] == 4
    assert metrics['model_rejection_rate'] == .5
    assert metrics['model_output_coverage_complete'] is False
    assert metrics['matured_rows'] == 8
    assert metrics['missing_matured_label_rows'] == (4 if missing_label else 0)
    assert metrics['matured_label_coverage'] == (.5 if missing_label else 1.)
    for horizon in (1, 5, 20, 60):
        values = metrics['horizons'][str(horizon)]
        assert values['status_counts'] == {'ok': 1, status: 1, 'insufficient_model_inputs': 1}
        assert values['model_rejected_rows'] == 1
        assert values['matured_rows'] == 2
        assert values['valid_samples'] == 1
        assert values['brier']['skill'] is not None


def test_saved_report_label_audit_does_not_restore_scorable_only_denominator(tmp_path):
    from tmp.refresh_calibrated_release_decision import refresh_label_coverage
    rows = []
    for horizon in (1, 5, 20, 60):
        for status in ('ok', 'invalid_return_range'):
            row = dict(pending_label_fixture(), date='2023-11-01', status=status,
                       horizon=horizon, label_end=pd.Timestamp('2023-12-01'),
                       fwd_return=.1 if status == 'ok' else np.nan)
            if status != 'ok':
                row['interval_status'] = status
                for column in TASK_COLUMNS['intervals']:
                    row[column] = np.nan
            rows.append(row)
    forecast = tmp_path / 'forecast.parquet'
    pd.DataFrame(rows).to_parquet(forecast, index=False)
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps({'as_of': '2023-12-31', 'scored_forecast_rows': 4,
                               'horizons': {str(h): {} for h in (1, 5, 20, 60)}}), encoding='utf-8')
    metrics = refresh_label_coverage(path, [forecast], 'test-revision')
    assert metrics['matured_rows'] == 8
    assert metrics['matured_label_coverage'] == .5
    assert metrics['model_rejected_rows'] == 4
    assert metrics['model_output_coverage_complete'] is False
    assert metrics['valid'] is False


def test_file_and_audit_totals_preserve_partial_tasks_and_their_label_gaps(tmp_path):
    from tmp.refresh_calibrated_release_decision import refresh_label_coverage
    first, second = [], []
    for horizon in (1, 5, 20, 60):
        good = dict(pending_label_fixture(), horizon=horizon, security_id='GOOD', status='ok')
        rejected = dict(good, security_id='REJECTED', status='invalid_quantile_order', interval_status='invalid_quantile_order')
        for column in TASK_COLUMNS['intervals']:
            rejected[column] = np.nan
        partial = dict(good, security_id='PARTIAL', status='insufficient_model_inputs',
                       score_status='insufficient_model_inputs', score=np.nan, fwd_return=np.nan)
        first.extend([good, rejected])
        second.append(partial)
    paths = [tmp_path / 'first.parquet', tmp_path / 'second.parquet']
    pd.DataFrame(first).to_parquet(paths[0], index=False)
    pd.DataFrame(second).to_parquet(paths[1], index=False)
    metrics = evaluate_prediction_files(paths, '2024-01-31')
    report = tmp_path / 'metrics.json'
    report.write_text(json.dumps({'as_of': '2024-01-31', 'scored_forecast_rows': 4,
        'horizons': {str(h): {} for h in (1, 5, 20, 60)}}), encoding='utf-8')
    refreshed = refresh_label_coverage(report, paths, 'test-revision')
    for result in (metrics, refreshed):
        assert result['matured_rows'] == 12 and result['missing_matured_label_rows'] == 4
        assert result['input_eligible_forecast_rows'] == 12
        assert result['input_unavailable_rows'] == 0
        assert result['partially_input_unavailable_rows'] == 4
        assert result['any_task_scored_forecast_rows'] == 12
        assert result['partially_scored_forecast_rows'] == 8
        assert result['task_valid_samples'] == {'score': 8, 'probability_up': 8, 'intervals': 4, 'expected_return': 8}
        assert result['task_coverage']['intervals']['model_rejected_rows'] == 4
        assert result['task_coverage']['score']['model_rejected_rows'] == 0
        assert not result['model_output_coverage_complete'] and not result['valid']
        assert result['horizons']['20']['task_coverage']['intervals']['model_rejected_rows'] == 1
