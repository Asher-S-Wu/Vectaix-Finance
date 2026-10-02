"""Auditable historical A-share cash-equity execution constraints.

Prices and notional are unadjusted CNY; quantities are actual integer shares.
The default commission is an assumed *gross/all-in* brokerage rate (including
exchange/regulatory handling charges), not a claim about any investor's tariff.
Stamp duty and ChinaClear transfer costs are additional. This avoids counting
handling fees twice. The historical schedule is supported from 2015-08-01.
Daily provider/exchange price-limit values are authoritative, including ST and
IPO exceptions. Missing bounds fail closed unless an explicit exemption exists.

Sources:
- https://one.sse.com.cn/onething/gptz/ (fees, included handling fees)
- https://m.mof.gov.cn/czxw/202308/t20230827_3904226.htm (stamp-duty cut)
- https://jrj.wuhan.gov.cn/ynzx_57/xwzx/202204/t20220429_1964160.shtml
  (ChinaClear transfer-fee announcement, government republication)
- https://edu.sse.com.cn/tib/qa/ (STAR order quantities)
- https://www.bse.cn/jygl_list/200028217.html (BSE quantities, T+1)
"""
from __future__ import annotations

import math
from datetime import date as Date, datetime
from typing import Any, Mapping


def _date(value: Any) -> Date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, '%Y%m%d').date()
    return Date.fromisoformat(text[:10])


def _positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _true(value: Any) -> bool:
    return str(value).lower() in {'true', '1', '1.0'}


def _side(side: str) -> str:
    side = str(side).lower()
    if side not in {'buy', 'sell'}:
        raise ValueError('side must be buy or sell')
    return side


def lot_rule(security_id: str) -> tuple[int, int]:
    """Return minimum regular order and increment: STAR 200/1, BJ 100/1."""
    sid = str(security_id).upper()
    code = sid.split('.')[0]
    if sid.endswith('.BJ'):
        return 100, 1
    if code.startswith(('688', '689')):
        return 200, 1
    return 100, 100


def buy_quantity(budget: float, price: float, security_id: str) -> int:
    """Round a notional budget down to a valid order; fees are not included."""
    if not _positive(budget) or not _positive(price):
        return 0
    minimum, step = lot_rule(security_id)
    quantity = int(math.floor(float(budget) / float(price) + 1e-10))
    return quantity // step * step if quantity >= minimum else 0


def sell_quantity(requested: float, holding: int, security_id: str) -> int:
    """Round a partial sale; a whole holding may include an odd-lot residual."""
    if not _positive(requested) or holding <= 0:
        return 0
    quantity = min(int(requested), int(holding))
    if quantity == holding:
        return quantity
    minimum, step = lot_rule(security_id)
    return quantity // step * step if quantity >= minimum else 0


def fee_breakdown(notional: float, side: str, date: Any,
                  security_id: str | None = None, *,
                  commission_rate: float = 0.0003,
                  min_commission: float = 5.0) -> dict[str, float]:
    """Fees per executed order, rounded at the total to CNY cents.

    commission_rate is a gross rate including exchange/regulatory charges for
    the chosen venue, including BSE. No additional exchange fee is added.
    Use an appropriate broker/venue gross rate when it differs from this
    research assumption. Unsupported pre-August-2015 dates are rejected.
    """
    side = _side(side)
    if not math.isfinite(float(notional)) or notional < 0:
        raise ValueError('notional must be finite and nonnegative')
    if (not math.isfinite(commission_rate) or commission_rate < 0 or
            not math.isfinite(min_commission) or min_commission < 0):
        raise ValueError('commission assumptions must be finite and nonnegative')
    day = _date(date)
    if day < Date(2015, 8, 1):
        raise ValueError('historical fee schedule is supported from 2015-08-01')
    if notional == 0:
        return dict(commission=0., stamp_duty=0., transfer_fee=0., total=0.)
    commission = max(float(min_commission), float(notional) * commission_rate)
    stamp_rate = 0.0005 if day >= Date(2023, 8, 28) else 0.001
    transfer_rate = (0.00001 if day >= Date(2022, 4, 29) else
                     0.000025 if str(security_id).upper().endswith('.BJ') else 0.00002)
    stamp = float(notional) * stamp_rate if side == 'sell' else 0.
    transfer = float(notional) * transfer_rate
    return dict(commission=commission, stamp_duty=stamp, transfer_fee=transfer,
                total=round(commission + stamp + transfer + 1e-10, 2))


def transaction_fee(notional: float, side: str, date: Any,
                    security_id: str | None = None, *,
                    commission_rate: float = 0.0003,
                    min_commission: float = 5.0) -> float:
    """Return total CNY fee; see fee_breakdown for historical assumptions."""
    return fee_breakdown(notional, side, date, security_id,
                         commission_rate=commission_rate,
                         min_commission=min_commission)['total']


def execution_status(bar: Mapping[str, Any], side: str,
                     acquired_date: Any = None) -> tuple[bool, str]:
    """Check an opening-auction research fill with conservative limit handling.

    execution_price can explicitly override raw_open for separately disclosed
    quote-based advice. A supplied acquisition date checks T+1; callers must
    independently reject unknown holding ages. Positive daily volume does not
    prove auction liquidity; the replay's participation cap is an approximation.
    """
    side = _side(side)
    if not _true(bar.get('quote_present', False)):
        return False, 'missing_quote'
    if _true(bar.get('suspended', False)) or not _positive(bar.get('volume')) or not _positive(bar.get('amount')):
        return False, 'suspended_or_no_liquidity'
    if _true(bar.get('corporate_action_unresolved', False)):
        return False, 'unresolved_corporate_action'
    price = bar.get('execution_price', bar.get('raw_open'))
    if not _positive(price):
        return False, 'invalid_price'
    if side == 'sell' and acquired_date is not None:
        try:
            if _date(bar.get('date')) <= _date(acquired_date):
                return False, 't_plus_one'
        except (TypeError, ValueError):
            return False, 'unknown_acquisition_date'
    exempt = _true(bar.get('price_limit_exempt', False))
    up, down = bar.get('up_limit'), bar.get('down_limit')
    if not exempt:
        if not _positive(up) or not _positive(down):
            return False, 'missing_price_limit'
        if float(up) <= float(down):
            return False, 'invalid_price_limit'
        if float(price) > float(up) + 1e-8 or float(price) < float(down) - 1e-8:
            return False, 'price_outside_limit_range'
        # Open at a one-sided limit never assumes a place at the head of queue.
        if side == 'buy' and float(price) >= float(up) - 1e-8:
            return False, 'at_upper_limit'
        if side == 'sell' and float(price) <= float(down) + 1e-8:
            return False, 'at_lower_limit'
    return True, 'ok'


def execution_allowed(bar: Mapping[str, Any], side: str,
                      acquired_date: Any = None) -> bool:
    """Boolean convenience wrapper around execution_status."""
    return execution_status(bar, side, acquired_date)[0]
