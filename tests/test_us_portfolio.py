"""Chronology, missing observations, and USD research-unit accounting."""
import importlib.util

import numpy as np
import pandas as pd
import pytest


def replay(*args, **kwargs):
    # An absent implementation is an assertion failure during the red phase.
    assert importlib.util.find_spec('us_quant.portfolio') is not None, 'US replay is not implemented'
    from us_quant.portfolio import replay as implementation
    return implementation(*args, **kwargs)


def prices(dates=('2024-01-30', '2024-01-31', '2024-02-01', '2024-02-02'), ids=('A', 'B', 'C')):
    return pd.DataFrame([
        {'date': pd.Timestamp(date), 'security_id': sid, 'adj_close': 100.,
         'quote_present': True, 'data_valid': True, 'dollar_volume_proxy': 1e9,
         'adv20_amount': 1e9}
        for date in dates for sid in ids
    ])


def predictions(date='2024-01-31', ids=('A', 'B', 'C'), scores=(3., 2., 1.)):
    return pd.DataFrame({'date': pd.Timestamp(date), 'security_id': ids,
                         'horizon': 20, 'score': scores, 'score_status': 'ok'})


def run(p=None, s=None, **kwargs):
    p = prices() if p is None else p
    s = predictions() if s is None else s
    return replay(p, s, p.date.min(), p.date.max(), initial_cash=1000., top_n=1, **kwargs)


def mutate(p, date, sid, **fields):
    for key, value in fields.items():
        p.loc[(p.date == pd.Timestamp(date)) & (p.security_id == sid), key] = value


def test_month_end_signal_executes_next_session_close_and_no_future_scores():
    p = prices()
    mutate(p, '2024-02-01', 'A', adj_close=200.)
    future = predictions('2024-02-01', scores=(-100., 100., 0.))
    result = run(p, pd.concat([predictions(), future]), fee_per_side=0)
    trades = result['trades'].query('filled_notional > 0')
    assert trades.security_id.tolist() == ['A']
    assert trades.date.tolist() == [pd.Timestamp('2024-02-01')]
    assert trades.signal_date.tolist() == [pd.Timestamp('2024-01-31')]
    assert trades.fill_price.tolist() == [200.]
    assert trades.adjusted_research_units.tolist() == [4.75]
    assert result['curve'].iloc[1].cash == 1000.


def test_unfillable_top_selection_is_not_replaced_and_capital_stays_cash():
    p = prices()
    mutate(p, '2024-02-01', 'A', adj_close=np.nan, quote_present=False, data_valid=False)
    result = run(p)
    assert result['summary']['filled_orders'] == 0
    assert result['trades'].security_id.tolist() == ['A']
    assert result['trades'].status.tolist() == ['unfilled']
    assert result['trades'].reason.tolist() == ['missing_execution_quote']
    assert (result['curve'].cash == 1000.).all()
    assert result['summary']['unfilled_selected_buys'] == 1


def test_only_signal_eligible_scores_are_ranked_with_security_id_tie_break():
    p = prices()
    mutate(p, '2024-01-31', 'A', quote_present=False, data_valid=False)
    s = predictions(ids=('C', 'B', 'A'), scores=(2., 2., 100.))
    result = run(p, s, fee_per_side=0)
    assert result['trades'].security_id.tolist() == ['B']
    assert result['trades'].selection_rank.tolist() == [1]


def test_nonavailable_and_other_horizon_scores_cannot_create_positions():
    s = predictions()
    s.loc[s.security_id == 'A', 'score_status'] = 'unavailable'
    s.loc[s.security_id == 'B', 'horizon'] = 5
    assert run(s=s)['trades'].security_id.tolist() == ['C']


def test_missing_holding_quote_is_stale_reference_and_terminal_metrics_withheld():
    p = prices()
    mutate(p, '2024-02-02', 'A', adj_close=np.nan, quote_present=False, data_valid=False)
    result = run(p, fee_per_side=0)
    last = result['curve'].iloc[-1]
    assert last.reference_nav == 1000.
    assert pd.isna(last.nav)
    assert last.unresolved_holdings_count == 1
    assert last.stale_reference_value == 950.
    summary = result['summary']
    assert summary['terminal_unknown_count'] == 1
    assert summary['terminal_holdings'][0]['adjusted_research_units'] == 9.5
    assert summary['performance_validated'] is False
    assert summary['return_metrics'] is None
    assert summary['total_return'] is None
    assert summary['reference_return_metrics']['total_return'] == 0


def test_recovered_quote_resolves_terminal_but_gap_performance_remains_reference_only():
    p = prices(dates=('2024-01-31', '2024-02-01', '2024-02-02', '2024-02-05'))
    mutate(p, '2024-02-02', 'A', quote_present=False, data_valid=False)
    mutate(p, '2024-02-05', 'A', adj_close=110.)
    result = run(p, fee_per_side=0)
    assert result['curve'].iloc[-1].nav == 1095.
    assert result['summary']['terminal_unknown_count'] == 0
    assert result['summary']['valuation_gap_sessions'] == 1
    assert result['summary']['performance_validated'] is False


def test_fee_cash_nav_and_investment_cap_are_consistent():
    result = run(fee_per_side=.0015)
    row = result['trades'].iloc[0]
    final = result['curve'].iloc[-1]
    assert row.fees == pytest.approx(row.filled_notional * .0015)
    assert final.cash == pytest.approx(1000 - row.filled_notional - row.fees)
    assert final.reference_nav == pytest.approx(final.cash + final.reference_equity)
    assert final.reference_nav == pytest.approx(1000 - row.fees)
    assert final.reference_equity <= .95 * final.reference_nav + 1e-9
    assert (result['curve'].cash >= 0).all()
    assert result['summary']['total_fees'] == pytest.approx(row.fees)
    assert result['summary']['turnover'] == pytest.approx(row.filled_notional / 1000)
    assert result['summary']['return_metrics']['total_return'] == pytest.approx(-row.fees / 1000)
    assert result['summary']['execution_validated'] is False


def test_participation_uses_signal_adv_and_execution_volume_not_future_adv():
    p = prices()
    mutate(p, '2024-01-31', 'A', adv20_amount=2000.)
    mutate(p, '2024-02-01', 'A', adv20_amount=1e12, dollar_volume_proxy=3000.)
    result = run(p, fee_per_side=0, participation=.01)
    row = result['trades'].iloc[0]
    assert row.liquidity_cap_notional == 20.
    assert row.filled_notional == 20.
    assert row.status == 'partial'
    assert result['curve'].iloc[-1].cash == 980.


def test_existing_names_are_not_sold_and_rebought_when_targets_unchanged():
    dates = ('2024-01-31', '2024-02-01', '2024-02-29', '2024-03-01')
    s = pd.concat([predictions(), predictions('2024-02-29')])
    result = run(prices(dates), s, fee_per_side=0)
    filled = result['trades'].query('filled_notional > 0')
    assert filled.side.tolist() == ['buy']
    assert filled.filled_notional.tolist() == [950.]
    assert result['summary']['terminal_valuation'] == 'mark_to_market_no_forced_liquidation'


def test_monthly_rotation_sells_before_buys_and_charges_both_sides():
    dates = ('2024-01-31', '2024-02-01', '2024-02-29', '2024-03-01')
    s = pd.concat([predictions(), predictions('2024-02-29', scores=(0., 3., 1.))])
    result = run(prices(dates), s, fee_per_side=.0015)
    fills = result['trades'].query('filled_notional > 0')
    assert fills.side.tolist() == ['buy', 'sell', 'buy']
    assert fills.security_id.tolist() == ['A', 'A', 'B']
    assert result['summary']['total_fees'] == pytest.approx(fills.filled_notional.sum() * .0015)
    assert (result['curve'].cash >= 0).all()
    assert result['curve'].iloc[-1].nav == pytest.approx(1000. - fills.fees.sum())


def test_unresolved_holding_is_not_dropped_at_failed_exit_or_reinvested():
    dates = ('2024-01-31', '2024-02-01', '2024-02-29', '2024-03-01')
    p = prices(dates)
    mutate(p, '2024-03-01', 'A', adj_close=np.nan, quote_present=False, data_valid=False)
    s = pd.concat([predictions(), predictions('2024-02-29', scores=(0., 3., 1.))])
    result = run(p, s, fee_per_side=0)
    last = result['curve'].iloc[-1]
    assert last.holdings_count == 1
    assert last.cash == 50.
    assert last.reference_equity == 950.
    assert result['summary']['terminal_unknown_count'] == 1
    assert result['trades'].query("security_id == 'B'").filled_notional.sum() == 0


def test_each_selected_slot_keeps_fixed_top_n_weight_when_fewer_names_qualify():
    p = prices()
    s = predictions()
    s.loc[s.security_id != 'A', 'score_status'] = 'unavailable'
    result = replay(p, s, p.date.min(), p.date.max(), initial_cash=1000., top_n=10, fee_per_side=0)
    assert result['trades'].filled_notional.sum() == 95.
    assert result['curve'].iloc[-1].cash == 905.


def test_future_changes_do_not_change_earlier_curve_or_trade_decisions():
    p = prices()
    first = run(p, fee_per_side=0)
    mutate(p, '2024-02-02', 'A', adj_close=300., adv20_amount=0., dollar_volume_proxy=0.)
    second = run(p, fee_per_side=0)
    pd.testing.assert_frame_equal(first['curve'].iloc[:-1], second['curve'].iloc[:-1])
    pd.testing.assert_frame_equal(first['trades'], second['trades'])


def test_first_session_can_execute_prior_month_signal_and_final_partial_month_is_not_signal():
    p = prices()
    result = replay(p, predictions(), '2024-02-01', '2024-02-02', initial_cash=1000., top_n=1, fee_per_side=0)
    assert len(result['curve']) == 2
    assert result['trades'].signal_date.tolist() == [pd.Timestamp('2024-01-31')]
    assert result['summary']['rebalance_count'] == 1


def test_inputs_are_not_mutated():
    p, s = prices(), predictions()
    p_original, s_original = p.copy(deep=True), s.copy(deep=True)
    run(p, s)
    pd.testing.assert_frame_equal(p, p_original)
    pd.testing.assert_frame_equal(s, s_original)


@pytest.mark.parametrize('change, match', [
    (lambda p: pd.concat([p, p.iloc[[0]]]), 'duplicate'),
    (lambda p: p.iloc[1:].copy(), 'complete shared calendar'),
    (lambda p: p.drop(columns='quote_present'), 'missing price columns'),
])
def test_rejects_invalid_price_panel(change, match):
    with pytest.raises(ValueError, match=match):
        run(change(prices()))


def test_rejects_duplicate_prediction_identity():
    s = predictions()
    with pytest.raises(ValueError, match='duplicate'):
        run(s=pd.concat([s, s.iloc[[0]]]))


@pytest.mark.parametrize('kwargs', [{'fee_per_side': -.1}, {'participation': 0}, {'participation': 2}, {'invest_fraction': 1.1}])
def test_rejects_invalid_replay_parameters(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)
