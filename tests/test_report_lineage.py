import json
import pytest

from hk_quant.report import build_report


def test_report_exposes_current_feature_manifest_separately_from_source_audit(tmp_path):
    results = tmp_path / 'results'
    data = tmp_path / 'data'
    results.mkdir()
    data.mkdir()
    (results / 'training_protocol.json').write_text(json.dumps({
        'kinds': [], 'training_sampling': 'test sampling',
        'execution': {'cash_policy': 'issuer_final_schedule_simulated'},
    }), encoding='utf-8')
    (results / 'training_status.json').write_text(json.dumps({'status': 'complete'}), encoding='utf-8')
    (data / 'data_audit.json').write_text(json.dumps({
        'rows': 100, 'unresolved_identities': 80,
        'historical_lot_coverage_complete': False,
        'corporate_action_cash_coverage_complete': False,
    }), encoding='utf-8')
    (data / 'feature_manifest.json').write_text(json.dumps({
        'data_as_of': '2026-09-11', 'rows': 200, 'eligible_rows': 150,
        'securities': 12,
    }), encoding='utf-8')
    (data / 'dated_identity_audit.json').write_text(json.dumps({
        'remaining_fully_unresolved_securities': 27,
        'remaining_unverified_rows': 91,
    }), encoding='utf-8')

    payload = build_report(results, data)
    assert payload['data_as_of'] == '2026-09-11'
    assert payload['feature_rows'] == 200
    assert payload['feature_eligible_rows'] == 150
    assert payload['feature_securities'] == 12
    assert payload['data_audit_snapshot']['rows'] == 100
    assert payload['identity_audit']['remaining_fully_unresolved_securities'] == 27


def test_report_does_not_mark_unified_bundle_pending_when_component_metrics_are_not_copied(tmp_path):
    results = tmp_path / 'results'
    data = tmp_path / 'data'
    results.mkdir()
    data.mkdir()
    (results / 'training_protocol.json').write_text(json.dumps({
        'kinds': ['factor'], 'training_sampling': 'test sampling',
        'execution': {'cash_policy': 'issuer_final_schedule_simulated'},
    }), encoding='utf-8')
    (results / 'training_status.json').write_text(json.dumps({'status': 'complete'}), encoding='utf-8')
    (results / 'bundle_development_metrics.json').write_text(json.dumps({
        'total_forecast_rows': 1, 'scored_forecast_rows': 1,
        'excluded_status_rows': 0, 'status_counts': {'ok': 1},
        'horizons': {'1': {
            'ic': {'mean': 0.1, 'ci_lower': 0.01, 'positive_year_ratio': 1.0},
            'brier': {'skill': 0.01}, 'calibration': {'ece': 0.01},
            'interval': {'coverage': 0.8, 'pinball_model': 0.1,
                         'pinball_baseline': 0.2, 'pinball_improved': True},
        }},
    }), encoding='utf-8')
    (data / 'data_audit.json').write_text(json.dumps({
        'rows': 100, 'unresolved_identities': 0,
        'historical_lot_coverage_complete': False,
        'corporate_action_cash_coverage_complete': False,
    }), encoding='utf-8')
    (data / 'feature_manifest.json').write_text(json.dumps({
        'data_as_of': '2026-09-11', 'rows': 200, 'eligible_rows': 150,
        'securities': 12,
    }), encoding='utf-8')

    payload = build_report(results, data)
    assert payload['pending_models'] == []
    assert payload['pending_component_metrics'] == ['factor']


@pytest.mark.parametrize('linear_label_audit_present', [True, False])
def test_completed_families_show_label_gaps_before_bundle_exists(tmp_path, linear_label_audit_present):
    from lxml import html

    results, data = tmp_path / 'results', tmp_path / 'data'
    (results / 'development').mkdir(parents=True)
    data.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(results / 'training_protocol.json', {
        'kinds': ['factor', 'linear', 'lightgbm_small'], 'training_sampling': 'test sampling',
        'execution': {'cash_policy': 'issuer_final_schedule_simulated'},
    })
    write(results / 'training_status.json', {'status': 'running'})
    write(data / 'data_audit.json', {
        'historical_lot_coverage_complete': False,
        'corporate_action_cash_coverage_complete': False,
    })
    write(data / 'feature_manifest.json', {'data_as_of': '2023-12-29'})
    metrics = {
        'total_forecast_rows': 20000, 'scored_forecast_rows': 19993,
        'excluded_status_rows': 7, 'status_counts': {'ok': 19993, 'invalid_return_range': 7},
        'input_unavailable_rows': 0, 'model_rejected_rows': 7,
        'model_output_coverage_complete': False, 'horizons': {},
    }
    factor_labels = dict(matured_rows=20000, matured_label_rows=7000,
        missing_matured_label_rows=13000, matured_label_coverage=.35, label_coverage_complete=False)
    linear_labels = dict(matured_rows=20000, matured_label_rows=6800,
        missing_matured_label_rows=13200, matured_label_coverage=.34, label_coverage_complete=False)
    write(results / 'development/factor_metrics.json', {**metrics, **factor_labels})
    write(results / 'development/linear_metrics.json', {**metrics, **(linear_labels if linear_label_audit_present else {})})

    payload = build_report(results, data)

    saved = json.loads((results / 'training_report.json').read_text(encoding='utf-8'))
    assert saved['component_mature_label_coverage']['factor'] == factor_labels
    assert set(saved['component_mature_label_coverage']) == {'factor', 'linear'}
    assert saved['component_mature_label_coverage'] == payload['component_mature_label_coverage']
    assert payload['mature_label_coverage'] == {}
    assert payload['pending_models'] == ['lightgbm_small']
    assert payload['forecast_coverage'][0]['model_rejected_rows'] == 7
    assert not payload['prediction_acceptance_passed']
    document = html.fromstring((results / 'training_report.html').read_text(encoding='utf-8'))
    table = document.xpath('//table[@aria-label="候选模型收益标签覆盖"]')[0]
    factor_row = table.xpath('.//tr[td[1]="factor"]/td/text()')
    assert factor_row == ['factor', '20,000', '7,000', '13,000', '35.00%', '不完整']
    assert '输入具备但模型输出无效' in document.text_content()
    linear_row = table.xpath('.//tr[td[1]="linear"]/td/text()')
    if linear_label_audit_present:
        assert saved['component_mature_label_coverage']['linear'] == linear_labels
        assert linear_row == ['linear', '20,000', '6,800', '13,200', '34.00%', '不完整']
    else:
        assert all(value is None for value in saved['component_mature_label_coverage']['linear'].values())
        assert linear_row == ['linear', '未核验', '未核验', '未核验', '未核验', '未核验']


@pytest.mark.parametrize('stored_acceptance', [False, True])
def test_regular_report_retains_label_gaps_and_static_confirmation_limits(tmp_path, stored_acceptance):
    results, data = tmp_path / 'results', tmp_path / 'data'
    results.mkdir()
    data.mkdir()
    source = results / 'static_diagnostic'
    source.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(results / 'training_protocol.json', {
        'kinds': [], 'training_sampling': 'test sampling',
        'execution': {'cash_policy': 'issuer_final_schedule_simulated'},
    })
    write(results / 'training_status.json', {'status': 'complete'})
    write(data / 'data_audit.json', {
        'historical_lot_coverage_complete': False,
        'corporate_action_cash_coverage_complete': False,
        'terminal_return_coverage_complete': False,
    })
    write(data / 'feature_manifest.json', {'data_as_of': '2026-09-11'})
    horizon = {
        'ic': {'mean': .1, 'ci_lower': .01, 'positive_year_ratio': 1.},
        'brier': {'skill': .01}, 'calibration': {'ece': .01},
        'interval': {'coverage': .8, 'pinball_model': .1, 'pinball_baseline': .2,
                     'pinball_improved': True},
        'status_counts': {'ok': 23, 'invalid_return_range': 2},
        'model_rejected_rows': 2, 'input_unavailable_rows': 0,
        'model_output_coverage_complete': False,
    }
    write(results / 'bundle_development_metrics.json', {
        'total_forecast_rows': 100, 'scored_forecast_rows': 92,
        'excluded_status_rows': 8, 'status_counts': {'ok': 92, 'invalid_return_range': 8},
        'input_unavailable_rows': 0, 'model_rejected_rows': 8,
        'model_output_coverage_complete': False,
        'horizons': {str(h): horizon for h in (1, 5, 20, 60)},
        'matured_rows': 100, 'matured_label_rows': 97,
        'missing_matured_label_rows': 3, 'matured_label_coverage': .97,
    })
    write(source / 'metrics.json', {
        'formal_confirmation': False, 'retrain_frequency': 'once',
        'matured_rows': 50, 'matured_label_rows': 48,
        'missing_matured_label_rows': 2, 'matured_label_coverage': .96,
    })
    write(results / 'calibrated_release_decision.json', {
        'eligible': False, 'prediction_acceptance_passed': stored_acceptance,
        'prediction_confirmation_source': 'static_diagnostic',
        'checks': {'confirmation.predictions.monthly_frozen_validation': {'passed': False}},
    })
    build_report(results, data)
    payload = build_report(results, data)
    saved = json.loads((results / 'training_report.json').read_text(encoding='utf-8'))
    document = (results / 'training_report.html').read_text(encoding='utf-8')
    assert payload['prediction_acceptance_passed'] is False
    assert payload['mature_label_coverage']['development']['missing_matured_label_rows'] == 3
    assert payload['mature_label_coverage']['static_confirmation_diagnostic']['missing_matured_label_rows'] == 2
    assert saved['mature_label_coverage'] == payload['mature_label_coverage']
    assert '静态' in payload['confirmation_evaluation']
    assert '预测能力尚未完成正式验收' in document
    assert '成熟收益标签缺失 3 条' in document
    assert '成熟收益标签缺失 2 条' in document
    assert payload['forecast_coverage'][0]['model_rejected_rows'] == 8
    assert payload['forecast_coverage'][0]['horizon_status_coverage']['20']['status_counts'] == {'ok': 23, 'invalid_return_range': 2}
    assert '输入具备但模型输出无效' in document
    assert '8 条' in document
    assert '共同可评分子样本' in payload['component_comparison_scope']
    assert payload['component_comparison_scope'] in document


def test_report_displays_independent_task_denominators_and_unknown_legacy_counts(tmp_path):
    from lxml import html

    results, data = tmp_path / 'results', tmp_path / 'data'
    (results / 'development').mkdir(parents=True)
    data.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(results / 'training_protocol.json', {
        'kinds': ['factor', 'linear'], 'training_sampling': 'test sampling',
        'execution': {'cash_policy': 'issuer_final_schedule_simulated'},
    })
    write(results / 'training_status.json', {'status': 'complete'})
    write(data / 'data_audit.json', {
        'historical_lot_coverage_complete': False,
        'corporate_action_cash_coverage_complete': False,
    })
    write(data / 'feature_manifest.json', {'data_as_of': '2023-12-29'})
    task_coverage = {}
    for task in ('score', 'probability_up', 'intervals', 'expected_return'):
        task_coverage[task] = {
            'total_forecast_rows': 10, 'input_eligible_forecast_rows': 9,
            'scored_forecast_rows': 7 if task == 'intervals' else 9,
            'model_rejected_rows': 2 if task == 'intervals' else 0,
            'input_unavailable_rows': 1,
        }
    task_samples = {'score': 8, 'probability_up': 8, 'intervals': 6, 'expected_return': 8}
    values = {
        'ic': {'mean': .1, 'ci_lower': .01, 'positive_year_ratio': 1.},
        'brier': {'skill': .01}, 'calibration': {'ece': .01},
        'interval': {'coverage': .8, 'pinball_model': .1, 'pinball_baseline': .2,
                     'pinball_improved': True},
    }
    metrics = {
        'total_forecast_rows': 10, 'scored_forecast_rows': 7,
        'excluded_status_rows': 3,
        'status_counts': {'ok': 7, 'invalid_quantile_order': 2, 'insufficient_model_inputs': 1},
        'input_unavailable_rows': 1, 'model_rejected_rows': 2,
        'model_output_coverage_complete': False,
    }
    labels = dict(matured_rows=8, matured_label_rows=8, missing_matured_label_rows=0,
                  matured_label_coverage=1., label_coverage_complete=True)
    write(results / 'development/factor_metrics.json', {
        **metrics, **labels, 'task_coverage': task_coverage, 'task_valid_samples': task_samples,
        'horizons': {'20': {**values, 'task_coverage': task_coverage, 'task_valid_samples': task_samples}},
    })
    write(results / 'development/linear_metrics.json', {**metrics, 'horizons': {'20': values}})

    payload = build_report(results, data)
    saved = json.loads((results / 'training_report.json').read_text(encoding='utf-8'))
    factor = next(row for row in saved['forecast_coverage'] if row['model'] == 'factor')
    legacy = next(row for row in saved['forecast_coverage'] if row['model'] == 'linear')
    assert factor['task_coverage'] == task_coverage
    assert factor['task_valid_samples'] == task_samples
    assert factor['horizon_status_coverage']['20']['task_coverage'] == task_coverage
    assert factor['horizon_status_coverage']['20']['task_valid_samples'] == task_samples
    assert legacy['task_coverage'] is None and legacy['task_valid_samples'] is None
    assert legacy['horizon_status_coverage']['20']['task_coverage'] is None
    assert saved['component_mature_label_coverage']['factor'] == labels
    assert payload['mature_label_coverage'] == {}
    assert payload['eligible'] is False and payload['prediction_acceptance_passed'] is False
    assert '同一任务' in payload['component_comparison_scope']
    assert '各自' in payload['component_comparison_scope']

    document = html.fromstring((results / 'training_report.html').read_text(encoding='utf-8'))
    assert '四任务均可用' in document.xpath('//th/text()')
    assert '可评分记录' not in document.xpath('//th/text()')
    table = document.xpath('//table[@aria-label="预测任务覆盖"]')[0]
    ranking = table.xpath('.//tr[td[1]="factor" and td[2]="排名"]/td/text()')
    intervals = table.xpath('.//tr[td[1]="factor" and td[2]="收益区间"]/td/text()')
    assert ranking == ['factor', '排名', '10', '9', '9', '8', '0', '1']
    assert intervals == ['factor', '收益区间', '10', '9', '7', '6', '2', '1']
    unknown = table.xpath('.//tr[td[1]="linear" and td[2]="排名"]/td/text()')
    assert unknown == ['linear', '排名', *(['未核验'] * 6)]
