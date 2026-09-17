import numpy as np
import pandas as pd
import pytest

from hk_quant.price_replay import replay_prices


def market(dates, prices, traded=None, amounts=None):
    n = len(dates)
    return pd.DataFrame({'date': pd.to_datetime(dates), 'security_id': ['A'] * n,
                         'adj_close': prices, 'quote_present': traded or [True] * n,
                         'data_valid': [True] * n, 'amount': amounts or [100000.] * n})


def signals(dates):
    return pd.DataFrame({'date': pd.to_datetime(dates), 'security_id': ['A'] * len(dates),
                         'score': [1.] * len(dates), 'adv20_amount': [100000.] * len(dates)})


def run(bars, signal_days, **kwargs):
    return replay_prices(signals(signal_days), bars, pd.DatetimeIndex(bars.date),
                         initial_cash=100., equity_weight=1., top_n=1, fee=0.,
                         max_participation=1., **kwargs)


def test_next_session_purchase_and_actual_elapsed_year_return():
    bars = market(['2024-01-01', '2024-01-02', '2025-01-01', '2025-01-02'], [1., 10., 2., 12.])
    result = run(bars, ['2024-01-01', '2025-01-01'])
    assert result['trades'].side.tolist() == ['buy', 'sell']
    assert result['summary']['total_return'] == pytest.approx(.2)
    assert result['summary']['cagr'] == pytest.approx(1.2 ** (365.25 / 366) - 1)


def test_no_quote_buy_is_cancelled_and_does_not_choose_another_stock():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02'],
                  [10., 1., 20., 20.], traded=[True, False, True, True])
    result = run(bars, ['2024-01-01', '2024-02-01'])
    assert result['trades'].empty
    assert result['summary']['ending_cash'] == 100.
    assert result['orders'].status.tolist() == ['buy_unfilled_no_quote']


def test_unfilled_sale_keeps_position_and_sells_on_resumption():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02', '2024-02-05'],
                  [10., 10., 11., 11., 12.], traded=[True, True, True, False, True])
    result = run(bars, ['2024-01-01', '2024-02-01'], end='2024-02-05')
    assert result['trades'].iloc[-1].date == pd.Timestamp('2024-02-05')
    assert result['summary']['ending_cash'] == 120.
    assert result['daily'].set_index('date').loc['2024-02-02', 'cash'] == 0.


def test_volume_capacity_and_both_side_fees_conserve_cash():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02'], [10.] * 4,
                  amounts=[100000., 500., 100000., 100000.])
    result = replay_prices(signals(['2024-01-01', '2024-02-01']), bars, pd.DatetimeIndex(bars.date),
                           initial_cash=100., equity_weight=1., top_n=1, fee=.01, max_participation=.1)
    assert result['trades'].iloc[0].value == 50.
    assert result['summary']['fees'] == 1.
    assert result['summary']['ending_cash'] == 99.
    assert result['summary']['profitable_hit_rate'] == 0.


def test_missing_held_valuation_stays_missing_and_never_becomes_zero():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02'],
                  [10., 10., 11., np.nan], traded=[True, True, True, False])
    result = run(bars, ['2024-01-01', '2024-02-01'])
    assert pd.isna(result['daily'].iloc[-1].equity)
    assert result['summary']['cagr'] is None
    assert result['summary']['open_positions'] == 1


def test_terminal_cash_is_receivable_until_evidenced_payment_date():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02', '2024-02-05'],
                  [10., 10., np.nan, np.nan, np.nan], traded=[True, True, False, False, False])
    actions = pd.DataFrame([{'security_id': 'A', 'effective_date': pd.Timestamp('2024-02-01'),
                             'payment_date': pd.Timestamp('2024-02-05'), 'known_date': pd.Timestamp('2024-01-31'),
                             'adjusted_cash_per_unit': 12., 'source_url': 'https://issuer.example/final.pdf',
                             'payment_basis': 'issuer_final_schedule_simulated'}])
    result = run(bars, ['2024-01-01', '2024-02-01'], end='2024-02-05', terminal_actions=actions)
    day = result['daily'].set_index('date').loc['2024-02-02']
    assert day.cash == 0.
    assert day.receivables == 120.
    assert result['summary']['ending_cash'] == 120.
    assert result['summary']['profitable_hit_rate'] == 1.


def test_future_cash_evidence_is_rejected_instead_of_backfilled():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02'], [10.] * 4)
    actions = pd.DataFrame([{'security_id': 'A', 'effective_date': pd.Timestamp('2024-02-01'),
                             'payment_date': pd.Timestamp('2024-02-02'), 'known_date': pd.Timestamp('2024-02-05'),
                             'adjusted_cash_per_unit': 12., 'source_url': 'https://issuer.example/final.pdf',
                             'payment_basis': 'issuer_final_schedule_simulated'}])
    with pytest.raises(ValueError, match='公开时间'):
        run(bars, ['2024-01-01', '2024-02-01'], terminal_actions=actions)


def test_sell_and_rebuy_share_the_same_daily_turnover_capacity():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02',
                   '2024-03-01', '2024-03-04'], [10.] * 6, amounts=[1000.] * 6)
    result = replay_prices(signals(['2024-01-01', '2024-02-01', '2024-03-01']), bars,
                           pd.DatetimeIndex(bars.date), initial_cash=100., equity_weight=1.,
                           top_n=1, fee=0., max_participation=.1)
    values = result['trades'].groupby('date').value.sum()
    assert values.le(100.).all()


def test_board_transfer_preserves_position_cost_and_units_and_pending_sale():
    bars = market(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-02', '2024-02-05'],
                  [10., 10., 12., 12., 6.], traded=[True, True, True, False, True])
    bars.loc[bars.date.eq('2024-02-05'), 'security_id'] = 'B'
    transfers = pd.DataFrame([dict(security_id='A', successor_id='B',
                                   effective_date=pd.Timestamp('2024-02-05'),
                                   known_date=pd.Timestamp('2024-01-31'), unit_multiplier=2.,
                                   source_url='https://issuer.example/transfer.pdf')])
    result = run(bars, ['2024-01-01', '2024-02-01'], end='2024-02-05', transfers=transfers)
    assert result['trades'].iloc[-1].security_id == 'B'
    assert result['trades'].iloc[-1].units == 20.
    assert result['summary']['ending_cash'] == 120.
    assert result['completed'].iloc[0].net_return == pytest.approx(.2)
