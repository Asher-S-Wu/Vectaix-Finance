import numpy as np
import pandas as pd
import pytest

from hk_quant.research import interval_returns, summarize_intervals, prediction_accuracy, common_signal_universe, observed_quote_periods


def test_reference_only_security_keeps_an_unresolved_quote_lifecycle():
    bars = pd.DataFrame({'security_id': ['A', 'A', 'A', 'B'],
                         'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-02']),
                         'quote_present': [False, True, True, False]})
    periods = observed_quote_periods(bars).set_index('security_id')
    assert set(periods.index) == {'A', 'B'}
    assert periods.loc['A', 'first_quote'] == pd.Timestamp('2020-01-02')
    assert periods.loc['A', 'last_quote'] == pd.Timestamp('2020-01-03')
    assert periods.loc['B'].isna().all()


def test_interval_uses_next_session_and_keeps_missing_selected_return():
    dates = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-01', '2024-02-29', '2024-03-01'])
    prices = pd.DataFrame({'A': [10., 100., 10., 20., 12.], 'B': [10., 10., 10., 10., np.nan]}, index=dates)
    result = interval_returns(prices, dates, dates[1], dates[3], ['A', 'B'])
    assert result.entry_date.eq(dates[2]).all()
    assert result.exit_date.eq(dates[4]).all()
    assert result.set_index('security_id').loc['A', 'gross_return'] == pytest.approx(.2)
    assert pd.isna(result.set_index('security_id').loc['B', 'gross_return'])


def test_roundtrip_fees_cash_and_compounding():
    intervals = pd.DataFrame({'entry_date': pd.to_datetime(['2024-01-02', '2024-07-02']),
                              'exit_date': pd.to_datetime(['2024-07-02', '2025-01-02']),
                              'gross_return': [.1, -.05], 'selected': [30, 30], 'missing': [0, 0],
                              'benchmark_gross_return': [.02, .01],
                              'profitable_hit_rate': [.6, .4], 'median_hit_rate': [.7, .5]})
    result = summarize_intervals(intervals, fee=.0025, equity_weight=.95)
    wealth = np.prod(.05 + .95 * (1 + np.array([.1, -.05])) * .9975 / 1.0025)
    assert result['total_return'] == pytest.approx(wealth - 1)
    assert result['cagr'] == pytest.approx(wealth ** (365.25 / 366) - 1)
    assert result['profitable_hit_rate'] == pytest.approx(.5)
    assert result['median_hit_rate'] == pytest.approx(.6)


def test_missing_selected_outcome_withholds_whole_portfolio_performance():
    intervals = pd.DataFrame({'entry_date': pd.to_datetime(['2024-01-02']),
                              'exit_date': pd.to_datetime(['2024-02-02']),
                              'gross_return': [np.nan], 'selected': [30], 'missing': [1],
                              'benchmark_gross_return': [.01],
                              'profitable_hit_rate': [np.nan], 'median_hit_rate': [np.nan]})
    result = summarize_intervals(intervals)
    assert result['status'] == 'incomplete_selected_returns'
    assert result['total_return'] is None
    assert result['cagr'] is None
    assert result['missing_selected_outcomes'] == 1


def test_higher_fees_use_the_corresponding_stock_profit_outcomes():
    stock_gains = np.array([.007, .02])
    intervals = pd.DataFrame({'entry_date': pd.to_datetime(['2024-01-02']),
                              'exit_date': pd.to_datetime(['2024-02-02']),
                              'gross_return': [stock_gains.mean()], 'selected': [2], 'missing': [0],
                              'benchmark_gross_return': [.01], 'median_hit_rate': [.5],
                              'profitable_hit_rate': [((1 + stock_gains) * .9975 / 1.0025 > 1).mean()],
                              'stress_profitable_hit_rate': [((1 + stock_gains) * .995 / 1.005 > 1).mean()]})
    normal = summarize_intervals(intervals)
    stress = summarize_intervals(intervals, fee=.005, hit_rate_column='stress_profitable_hit_rate')
    assert normal['profitable_hit_rate'] == 1.
    assert stress['profitable_hit_rate'] == .5
    assert stress['total_return'] < normal['total_return']


def test_accuracy_uses_probability_task_status_and_only_mature_labels():
    predictions = pd.DataFrame({'horizon': [20]*4, 'probability_up': [.8, .3, .8, .8],
                                'probability_status': ['ok', 'ok', 'unavailable', 'ok'],
                                'fwd_return': [.1, -.1, -.1, .1],
                                'label_end': pd.to_datetime(['2024-02-01']*3 + ['2025-01-01'])})
    result = prediction_accuracy(predictions, '2024-12-31')
    assert result['20']['samples'] == 2
    assert result['20']['direction_accuracy'] == 1.


def test_candidate_comparison_shares_signal_pool_without_future_label_filter():
    base = pd.DataFrame({'date': pd.to_datetime(['2024-01-31']*3), 'security_id': ['A', 'B', 'C'],
                         'horizon': [20]*3, 'score_status': ['ok']*3,
                         'adv20_amount': [2_000_000.]*3, 'fwd_return': [np.nan, .1, .2]})
    other = base.copy()
    other.loc[other.security_id.eq('C'), 'score_status'] = 'insufficient_model_inputs'
    pool = common_signal_universe([base, other], 1_000_000.)
    assert set(pool.get_level_values('security_id')) == {'A', 'B'}
