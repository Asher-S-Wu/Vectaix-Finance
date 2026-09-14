import numpy as np
import pandas as pd
import pytest

from hk_quant.evaluation import (
    evaluate_portfolio,
    evaluate_predictions,
    release_decision,
)
from hk_quant.prediction_tasks import TASK_COLUMNS, TASK_STATUS_COLUMNS


PREDICTION_COLUMNS = [
    "date", "security_id", "horizon", "fwd_return", "label_end", "score",
    "probability_up", "q10", "q50", "q90", "baseline_probability",
    "baseline_q10", "baseline_q50", "baseline_q90",
]


def prediction_frame(rows):
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    frame['status'] = [row.get('status', 'ok') for row in rows]
    frame['expected_return'] = [row.get('expected_return', .05) for row in rows]
    for field in TASK_STATUS_COLUMNS.values():
        frame[field] = [row.get(field, 'ok') for row in rows]
    return frame


def test_future_labels_are_excluded_before_metrics():
    rows = []
    for i in range(20):
        common = dict(
            security_id=f"S{i:02d}", horizon=20, fwd_return=i / 100., score=float(i),
            probability_up=.9, q10=(i - 1) / 100., q50=i / 100., q90=(i + 1) / 100.,
            baseline_probability=.5, baseline_q10=(i - 3) / 100., baseline_q50=(i - 2) / 100.,
            baseline_q90=(i + 3) / 100.,
        )
        rows.append(dict(date="2024-01-02", label_end="2024-01-31", **common))
        rows.append(dict(date="2024-02-01", label_end="2024-02-29", **common))

    result = evaluate_predictions(prediction_frame(rows), "2024-01-31")

    assert result["excluded_unmatured_rows"] == 20
    assert result["matured_rows"] == 20
    assert result["horizons"][20]["valid_samples"] == 20
    assert result["horizons"][20]["ic"]["daily_count"] == 1
    assert result["horizons"][20]["ic"]["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("probability_up", 1.01, "probability_up"),
        ("baseline_probability", -0.01, "probability_up"),
        ("q50", -.2, "intervals"),
    ],
)
def test_invalid_probabilities_or_unsorted_quantiles_raise(column, value, match):
    row = dict(
        date="2024-01-02", security_id="A", horizon=1, fwd_return=0.1,
        label_end="2024-01-03", score=1.0, probability_up=.8,
        q10=-.1, q50=0.0, q90=.1, baseline_probability=.5,
        baseline_q10=-.2, baseline_q50=0.0, baseline_q90=.2,
    )
    row[column] = value
    with pytest.raises(ValueError, match=match):
        evaluate_predictions(prediction_frame([row]), "2024-01-31")


def test_perfect_prediction_beats_constant_probability_and_quantile_baselines():
    rows = []
    for i in range(20):
        y = (i - 10) / 100.
        inside = i < 16
        rows.append(dict(
            date="2024-01-02", security_id=f"S{i:02d}", horizon=20,
            fwd_return=y, label_end="2024-01-31", score=y,
            probability_up=.99 if y > 0 else .01,
            q10=y - .01 if inside else y + .01,
            q50=y if inside else y + .02,
            q90=y + .01 if inside else y + .03,
            baseline_probability=.5,
            baseline_q10=y + .08, baseline_q50=y + .09, baseline_q90=y + .10,
        ))

    metrics = evaluate_predictions(prediction_frame(rows), "2024-01-31")["horizons"][20]

    assert metrics["ic"]["mean"] == pytest.approx(1.0)
    assert metrics["brier"]["model"] < metrics["brier"]["baseline"]
    assert metrics["brier"]["skill"] > 0
    # The equal-frequency boundary bin contains one .01 and one .99 forecast;
    # its mean exactly matches its 50% observed rate, so nine bins contribute .01.
    assert metrics["calibration"]["ece"] == pytest.approx(.009)
    assert metrics["interval"]["coverage"] == pytest.approx(.8)
    assert metrics["interval"]["coverage_pass"] is True
    assert metrics["interval"]["pinball_model"] < metrics["interval"]["pinball_baseline"]


def test_constant_scores_do_not_create_fake_ic():
    rows = [dict(
        date="2024-01-02", security_id=f"S{i:02d}", horizon=1,
        fwd_return=i / 100., label_end="2024-01-03", score=1.0,
        probability_up=.5, q10=-.1, q50=0.0, q90=.3,
        baseline_probability=.5, baseline_q10=-.2, baseline_q50=0.0,
        baseline_q90=.31,
    ) for i in range(20)]

    ic = evaluate_predictions(prediction_frame(rows), "2024-01-31")["horizons"][1]["ic"]

    assert ic["daily_count"] == 0
    assert ic["mean"] is None


def test_sixty_day_block_bootstrap_separates_perfect_from_random_scores():
    rng = np.random.default_rng(20260910)
    perfect_rows = []
    random_rows = []
    for day in pd.bdate_range("2024-01-02", periods=60):
        returns = (np.arange(20, dtype=float) - 10) / 100.
        random_scores = rng.permutation(returns)
        for i, (actual, random_score) in enumerate(zip(returns, random_scores)):
            common = dict(
                date=day, security_id=f"S{i:02d}", horizon=20,
                fwd_return=actual, label_end=day + pd.Timedelta(days=30),
                probability_up=.75 if actual > 0 else .25,
                q10=actual - .01, q50=actual, q90=actual + .01,
                baseline_probability=.5,
                baseline_q10=actual - .02, baseline_q50=actual + .01,
                baseline_q90=actual + .02,
            )
            perfect_rows.append(dict(score=actual, **common))
            random_rows.append(dict(score=random_score, **common))

    as_of = pd.Timestamp(perfect_rows[-1]["label_end"])
    perfect = evaluate_predictions(prediction_frame(perfect_rows), as_of)["horizons"][20]["ic"]
    random = evaluate_predictions(prediction_frame(random_rows), as_of)["horizons"][20]["ic"]

    assert perfect["ci_lower"] == pytest.approx(1.0)
    assert perfect["ci_upper"] == pytest.approx(1.0)
    assert perfect["positive_year_ratio"] == pytest.approx(1.0)
    assert abs(random["mean"]) < .1
    assert random["ci_lower"] < perfect["ci_lower"]
    assert perfect["bootstrap_seed"] == random["bootstrap_seed"] == 20260910
    assert perfect["block_length_trading_days"] == 60


def test_portfolio_rejects_mismatched_dates_instead_of_inner_joining():
    daily = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "equity": [100, 101]})
    benchmark = pd.DataFrame({"date": ["2024-01-01"], "equity": [100]})
    with pytest.raises(ValueError, match="日期集合"):
        evaluate_portfolio(daily, benchmark, daily, daily, {
            "status": "complete", "gap_count": 0, "corporate_action_coverage": True,
        })


def test_initial_equity_is_a_drawdown_peak_and_metrics_match_by_hand():
    dates = pd.date_range("2024-01-01", periods=3)
    daily = pd.DataFrame({"date": dates, "equity": [100.0, 80.0, 90.0]})
    benchmark = pd.DataFrame({"date": dates, "equity": [100.0, 100.0, 100.0]})
    stress = pd.DataFrame({"date": dates, "equity": [100.0, 90.0, 99.0]})
    result = evaluate_portfolio(daily, benchmark, stress, benchmark, {
        "status": "complete", "gap_count": 0, "corporate_action_coverage": True,
    })

    expected_cagr = (90.0 / 100.0) ** (252 / 2) - 1
    expected_excess = ((-.2 + .125) / 2) * 252
    expected_ir = np.sqrt(252) * np.mean([-.2, .125]) / np.std([-.2, .125], ddof=1)
    expected_stress_excess = ((-.1 + .1) / 2) * 252
    assert result["cagr"] == pytest.approx(expected_cagr)
    assert result["max_drawdown"] == pytest.approx(-.2)
    assert result["ann_excess"] == pytest.approx(expected_excess)
    assert result["information_ratio"] == pytest.approx(expected_ir)
    assert result["stress_ann_excess"] == pytest.approx(expected_stress_excess)


def passing_evaluation_pair():
    horizon = {
        "ic": {"ci_lower": .01, "positive_year_ratio": .75},
        "brier": {"skill": .1},
        "calibration": {"ece": .04},
        "interval": {"coverage": .8, "pinball_model": .1, "pinball_baseline": .2},
    }
    predictions = {
        "valid": True, "horizons": {h: horizon for h in (1, 5, 20, 60)},
        "as_of": "2024-01-31", "model_revision": "architecture-v1",
        "label_coverage_complete": True, "missing_matured_label_rows": 0,
        "matured_label_coverage": 1.0,
        "model_output_coverage_complete": True, "model_rejected_rows": 0,
        "formal_confirmation": True, "parameter_selection_used": False,
        "retrain_frequency": "monthly", "frozen_model_revision": "architecture-v1",
        "frozen_heads": {str(h): {'score': 'linear'} for h in (1, 5, 20, 60)},
        "frozen_calibration_parameters": {'interval_width_multiplier': {'20': 1.05}},
        "monthly_model_records": [{
            "forecast_month": "2024-01", "model_revision": "architecture-v1",
            "trained_as_of": "2023-12-29", "horizons": [1, 5, 20, 60],
            "forecast_rows": 80, "prediction_coverage_complete": True,
            "heads": {str(h): {'score': 'linear'} for h in (1, 5, 20, 60)},
            "component_calibration_parameters": {
                'linear': {'interval_width_multiplier': {'20': 1.05}}},
        }],
    }
    portfolio = {
        "portfolio_valid": True, "ann_excess": .05, "information_ratio": .6,
        "max_drawdown": -.2, "positive_year_ratio": .75, "stress_ann_excess": .01,
        "execution_complete": True,
    }
    return {"predictions": predictions, "portfolio": portfolio}


def complete_audit():
    return {
        "critical_gap_count": 0,
        "future_leakage_detected": False,
        "historical_lot_coverage_complete": True,
        "corporate_action_cash_coverage_complete": True,
        "terminal_return_coverage_complete": True,
    }


@pytest.mark.parametrize("execution", [
    {"status": "incomplete", "gap_count": 0, "corporate_action_coverage": True},
    {"status": "complete", "gap_count": 1, "corporate_action_coverage": True},
    {"status": "complete", "gap_count": 0, "corporate_action_coverage": False},
])
def test_execution_gap_or_corporate_action_gap_invalidates_portfolio(execution):
    dates = pd.date_range("2024-01-01", periods=3)
    equity = pd.DataFrame({"date": dates, "equity": [100.0, 101.0, 102.0]})
    result = evaluate_portfolio(equity, equity.copy(), equity, equity.copy(), execution)
    assert result["execution_complete"] is False
    assert result["portfolio_valid"] is False


def test_release_requires_every_piece_of_evidence_and_both_periods():
    development = passing_evaluation_pair()
    confirmation = passing_evaluation_pair()
    assert release_decision(development, confirmation, complete_audit())["eligible"] is True

    missing_execution = passing_evaluation_pair()
    missing_execution["portfolio"] = dict(missing_execution["portfolio"])
    missing_execution["portfolio"]["execution_complete"] = False
    rejected = release_decision(development, missing_execution, complete_audit())
    assert rejected["eligible"] is False
    assert any("confirmation.portfolio.execution_complete" in reason for reason in rejected["failed_reasons"])

    missing_audit = complete_audit()
    del missing_audit["historical_lot_coverage_complete"]
    rejected = release_decision(development, confirmation, missing_audit)
    assert rejected["eligible"] is False
    assert any("historical_lot_coverage_complete" in reason for reason in rejected["failed_reasons"])


def pending_label_fixture():
    return dict(date='2024-01-02',security_id='A',horizon=20,fwd_return=.1,
                label_end='2024-01-31',score=.2,probability_up=.6,
                q10=-.1,q50=.05,q90=.2,baseline_probability=.5,
                baseline_q10=-.2,baseline_q50=0.,baseline_q90=.3,
                expected_return=.05, score_status='ok', probability_status='ok',
                interval_status='ok', expected_return_status='ok')


def test_interval_rejection_does_not_remove_valid_ranking_or_probability():
    rows = []
    for i in range(20):
        actual = (i - 10) / 100.
        rows.append(dict(pending_label_fixture(), security_id=f'S{i}', fwd_return=actual,
            score=actual, probability_up=.9 if actual > 0 else .1,
            q10=actual-.01, q50=actual, q90=actual+.01))
    bad_interval = rows[0]
    bad_interval.update(status='invalid_quantile_order', interval_status='invalid_quantile_order', probability_up=.8)
    for column in TASK_COLUMNS['intervals']:
        bad_interval[column] = np.nan
    result = evaluate_predictions(prediction_frame(rows), '2024-01-31')
    metrics = result['horizons'][20]
    assert metrics['ic']['mean'] == pytest.approx(1.)
    assert metrics['ic']['daily_count'] == 1
    assert metrics['brier']['model'] == pytest.approx((.64 + 19 * .01) / 20)
    assert sum(row['count'] for row in metrics['calibration']['bins']) == 20
    assert metrics['task_valid_samples'] == {'score': 20, 'probability_up': 20, 'intervals': 19, 'expected_return': 20}
    assert result['task_valid_samples'] == metrics['task_valid_samples']
    assert metrics['valid_samples'] == 19
    assert result['model_rejected_rows'] == 1 and not result['valid']
    assert not result['task_coverage']['intervals']['model_output_coverage_complete']
    assert result['task_coverage']['score']['model_output_coverage_complete']
    assert result['any_task_scored_forecast_rows'] == 20
    assert result['partially_scored_forecast_rows'] == 1


def test_partial_input_row_stays_in_label_coverage_and_available_tasks():
    good = dict(pending_label_fixture(), security_id='GOOD')
    partial = dict(good, security_id='PARTIAL', status='insufficient_model_inputs',
                   score_status='insufficient_model_inputs', score=np.nan)
    missing_label = dict(partial, security_id='UNRESOLVED', fwd_return=np.nan)
    unavailable = dict(good, security_id='UNAVAILABLE', status='insufficient_model_inputs')
    for task, fields in TASK_COLUMNS.items():
        unavailable[TASK_STATUS_COLUMNS[task]] = 'insufficient_model_inputs'
        for field in fields:
            unavailable[field] = np.nan
    result = evaluate_predictions(prediction_frame([good, partial, missing_label, unavailable]), '2024-01-31')
    assert result['input_eligible_forecast_rows'] == 3
    assert result['input_unavailable_rows'] == 1
    assert result['partially_input_unavailable_rows'] == 2
    assert result['matured_rows'] == 3 and result['missing_matured_label_rows'] == 1
    assert result['task_valid_samples'] == {'score': 1, 'probability_up': 2, 'intervals': 2, 'expected_return': 2}
    assert result['task_coverage']['score']['input_unavailable_rows'] == 3
    assert result['task_coverage']['probability_up']['input_unavailable_rows'] == 1
    assert result['model_rejected_rows'] == 0 and not result['valid']


def test_legacy_file_without_task_status_is_not_inferred():
    frame = prediction_frame([pending_label_fixture()]).drop(columns=list(TASK_STATUS_COLUMNS.values()))
    with pytest.raises(ValueError, match='score_status'):
        evaluate_predictions(frame, '2024-01-31')


def test_unformed_future_label_is_excluded_and_counted_separately():
    mature=pending_label_fixture()
    pending=dict(mature,security_id='B',date='2024-02-01',label_end=None,fwd_return=np.nan)
    immature=dict(mature,security_id='C',date='2024-02-01',label_end='2024-02-29',fwd_return=np.nan)
    result=evaluate_predictions(prediction_frame([mature,pending,immature]),'2024-01-31')
    assert result['pending_label_end_rows']==1
    assert result['excluded_unmatured_rows']==1
    assert result['matured_rows']==1
    assert result['horizons'][20]['valid_samples']==1


@pytest.mark.parametrize('field',['date','label_end'])
def test_bad_date_text_is_not_treated_as_pending_label(field):
    row=pending_label_fixture();row[field]='not-a-date';row['fwd_return']=np.nan
    with pytest.raises(ValueError,match='日期'):
        evaluate_predictions(prediction_frame([row]),'2024-01-31')


def test_return_without_label_end_is_rejected():
    row=pending_label_fixture();row['label_end']=None
    with pytest.raises(ValueError,match='label_end'):
        evaluate_predictions(prediction_frame([row]),'2024-01-31')


def test_mature_missing_return_retains_diagnostics_but_fails_label_coverage():
    complete = pending_label_fixture()
    missing = dict(complete, security_id='DELISTED', fwd_return=np.nan)
    future = dict(complete, security_id='FUTURE', label_end='2024-02-29', fwd_return=np.nan)
    result = evaluate_predictions(prediction_frame([complete, missing, future]), '2024-01-31')
    assert result['valid'] is False
    assert result['missing_matured_label_rows'] == 1
    assert result['matured_label_rows'] == 1
    assert result['matured_label_coverage'] == .5
    assert result['label_coverage_complete'] is False
    assert result['horizons'][20]['valid_samples'] == 1
    assert result['horizons'][20]['missing_matured_label_rows'] == 1
    assert result['horizons'][20]['matured_label_coverage'] == .5
    assert result['horizons'][20]['brier']['skill'] is not None


@pytest.mark.parametrize('mutation', [
    'static', 'not_formal', 'missing_month', 'different_revision',
    'missing_horizon', 'training_after_month_start', 'incomplete_predictions',
    'different_heads', 'different_calibration',
])
def test_release_rejects_incomplete_or_different_monthly_confirmation(mutation):
    development, confirmation = passing_evaluation_pair(), passing_evaluation_pair()
    proof = confirmation['predictions']
    record = proof['monthly_model_records'][0]
    if mutation == 'static':
        proof['retrain_frequency'] = 'once'
    elif mutation == 'not_formal':
        proof['formal_confirmation'] = False
    elif mutation == 'missing_month':
        proof['as_of'] = '2024-02-29'
    elif mutation == 'different_revision':
        record['model_revision'] = 'different-architecture'
    elif mutation == 'missing_horizon':
        record['horizons'] = [20, 60]
    elif mutation == 'training_after_month_start':
        record['trained_as_of'] = '2024-01-02'
    elif mutation == 'incomplete_predictions':
        record['prediction_coverage_complete'] = False
    elif mutation == 'different_heads':
        record['heads']['20']['score'] = 'factor'
    elif mutation == 'different_calibration':
        record['component_calibration_parameters']['linear']['interval_width_multiplier']['20'] = 1.1
    result = release_decision(development, confirmation, complete_audit())
    assert result['eligible'] is False
    assert result['checks']['confirmation.predictions.monthly_frozen_validation']['passed'] is False


def test_release_requires_mature_label_and_terminal_return_evidence():
    development, confirmation = passing_evaluation_pair(), passing_evaluation_pair()
    development['predictions']['label_coverage_complete'] = False
    development['predictions']['missing_matured_label_rows'] = 1
    development['predictions']['matured_label_coverage'] = .99
    audit = complete_audit()
    del audit['terminal_return_coverage_complete']
    result = release_decision(development, confirmation, audit)
    assert result['eligible'] is False
    assert result['checks']['development.predictions.label_coverage_complete']['passed'] is False
    assert result['checks']['data_audit.terminal_return_coverage_complete']['present'] is False


def test_release_rejects_model_failure_even_when_observed_scores_and_labels_pass():
    development, confirmation = passing_evaluation_pair(), passing_evaluation_pair()
    development['predictions']['model_output_coverage_complete'] = False
    development['predictions']['model_rejected_rows'] = 1
    result = release_decision(development, confirmation, complete_audit())
    assert result['eligible'] is False
    assert result['checks']['development.predictions.model_output_coverage_complete']['passed'] is False
    assert result['checks']['development.predictions.model_rejected_rows']['passed'] is False
