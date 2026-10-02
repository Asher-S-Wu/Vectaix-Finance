"""Historical costs and conservative, raw-price A-share order constraints."""
import pytest

from ashare_quant.rules import (buy_quantity, execution_allowed, execution_status,
                                sell_quantity, transaction_fee)


def bar(**changes):
    result = dict(date='2024-02-01', raw_open=10., raw_close=10., high=10.5,
                  low=9.5, up_limit=11., down_limit=9., volume=100_000,
                  amount=1_000_000., quote_present=True)
    result.update(changes)
    return result


def test_fee_applies_minimum_commission_and_sell_only_stamp_duty():
    assert transaction_fee(1_000, 'buy', '2024-01-02') == pytest.approx(5.01)
    assert transaction_fee(1_000, 'sell', '2024-01-02') == pytest.approx(5.51)
    assert transaction_fee(0, 'buy', '2024-01-02') == 0


def test_historical_stamp_and_transfer_cutovers():
    assert transaction_fee(100_000, 'sell', '2023-08-27') == pytest.approx(131.)
    assert transaction_fee(100_000, 'sell', '2023-08-28') == pytest.approx(81.)
    assert transaction_fee(100_000, 'buy', '2022-04-28') == pytest.approx(32.)
    assert transaction_fee(100_000, 'buy', '2022-04-29') == pytest.approx(31.)
    assert transaction_fee(100_000, 'buy', '2022-04-28', '430047.BJ') == pytest.approx(32.5)


@pytest.mark.parametrize('security,budget,expected', [
    ('600000.SH', 2999, 200), ('000001.SZ', 2999, 200),
    ('688001.SH', 1999, 0), ('688001.SH', 2349, 234),
    ('430047.BJ', 999, 0), ('430047.BJ', 1239, 123),
    ('920001.BJ', 1239, 123)])
def test_market_specific_whole_share_buy_orders(security, budget, expected):
    assert buy_quantity(budget, 10., security) == expected


def test_sell_lots_allow_complete_odd_lot_liquidation():
    assert sell_quantity(150, 150, '600000.SH') == 150
    assert sell_quantity(140, 250, '600000.SH') == 100
    assert sell_quantity(199, 399, '688001.SH') == 0
    assert sell_quantity(230, 399, '688001.SH') == 230
    assert sell_quantity(120, 250, '430047.BJ') == 120


@pytest.mark.parametrize('changes,side,reason', [
    ({'quote_present': False}, 'buy', 'missing_quote'),
    ({'volume': 0}, 'buy', 'suspended_or_no_liquidity'),
    ({'amount': 0}, 'sell', 'suspended_or_no_liquidity'),
    ({'up_limit': None}, 'buy', 'missing_price_limit'),
    ({'down_limit': float('nan')}, 'sell', 'missing_price_limit'),
    ({'raw_open': 11., 'high': 11., 'low': 11.}, 'buy', 'at_upper_limit'),
    ({'raw_open': 9., 'high': 9., 'low': 9.}, 'sell', 'at_lower_limit'),
    ({'corporate_action_unresolved': True}, 'buy', 'unresolved_corporate_action')])
def test_invalid_or_unverifiable_fills_fail_closed(changes, side, reason):
    assert execution_status(bar(**changes), side) == (False, reason)
    assert execution_allowed(bar(**changes), side) is False


def test_daily_authoritative_limit_replaces_board_or_st_assumptions():
    assert execution_allowed(bar(raw_open=10.6, up_limit=11.0, is_st=True), 'buy')
    assert not execution_allowed(bar(raw_open=10.5, up_limit=10.5, is_st=False), 'buy')


def test_t_plus_one_uses_execution_and_acquisition_dates():
    assert execution_status(bar(), 'sell', '2024-02-01') == (False, 't_plus_one')
    assert execution_allowed(bar(), 'sell', '2024-01-31')


def test_explicit_verified_limit_exemption_allows_missing_bounds():
    assert execution_allowed(bar(up_limit=None, down_limit=None,
                                 price_limit_exempt=True), 'buy')


def test_invalid_inputs_are_rejected_instead_of_generating_negative_cash():
    with pytest.raises(ValueError):
        transaction_fee(-1, 'buy', '2024-01-01')
    with pytest.raises(ValueError):
        transaction_fee(100, 'short', '2024-01-01')
    assert buy_quantity(1000, 0, '600000.SH') == 0


def test_execution_rejects_prices_outside_authoritative_range_on_both_sides():
    assert execution_status(bar(raw_open=12.), 'sell') == (False, 'price_outside_limit_range')
    assert execution_status(bar(raw_open=8.), 'buy') == (False, 'price_outside_limit_range')
