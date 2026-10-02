"""Independent USD, monthly, long-only adjusted-price research replay.

This is a valuation experiment, never a live/executable trading simulation. Yahoo
adjusted close incorporates splits and distributions, so quantities below are
*adjusted research units*, not exchange shares. No separate dividend cash is added.

Month-end scores determine a fixed selection and USD target budgets using only
that signal's observations. Budgets are converted to research units at the next
shared-calendar session's close, subject to cash, fees, and dollar liquidity.
This idealised close-price budget allocation does not assert an implementable
market-on-close order. Execution information never changes selected identities.
Rebalances trade target deltas (sells before buys), not full liquidation/re-entry.
A failed order expires that day; there is no later fill or replacement name.

Raw prices are never forward-filled. Missing holding prices retain the previous
mark only in explicitly named reference fields. Any unresolved daily valuation
invalidates whole-path performance even if the holding subsequently recovers.
Final positions are marked, not assumed liquidated or discarded.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


_PRICE_COLUMNS = {
    'date', 'security_id', 'adj_close', 'quote_present', 'data_valid',
    'dollar_volume_proxy', 'adv20_amount',
}
_PREDICTION_COLUMNS = {'date', 'security_id', 'horizon', 'score', 'score_status'}
_TRADE_COLUMNS = [
    'date', 'execution_date', 'signal_date', 'security_id', 'side', 'status',
    'reason', 'selection_rank', 'target_notional', 'planned_notional',
    'filled_notional', 'adjusted_research_units', 'fill_price', 'fees',
    'signal_adv20_amount', 'execution_dollar_volume_proxy',
    'liquidity_cap_notional', 'planned_value_uses_reference',
]


def _positive(value):
    return bool(pd.notna(value) and np.isfinite(value) and value > 0)


def _valid_quote(row):
    return bool(row['quote_present'] and row['data_valid'] and _positive(row['adj_close']))


def _prepare(prices, predictions):
    missing = _PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f'missing price columns: {sorted(missing)}')
    missing = _PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f'missing prediction columns: {sorted(missing)}')
    p, s = prices.copy(deep=True), predictions.copy(deep=True)
    for frame, label in ((p, 'price'), (s, 'prediction')):
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        if frame.date.isna().any() or frame.security_id.isna().any():
            raise ValueError(f'{label} identity and date must be nonnull')
        if frame.date.dt.tz is not None or not frame.date.equals(frame.date.dt.normalize()):
            raise ValueError(f'{label} dates must be timezone-naive session dates')
        if not frame.security_id.map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f'{label} security_id must be a nonempty string')
    if p.empty:
        raise ValueError('price panel must not be empty')
    if p.duplicated(['date', 'security_id']).any():
        raise ValueError('duplicate price date/security_id')
    if s.duplicated(['date', 'security_id', 'horizon']).any():
        raise ValueError('duplicate prediction date/security_id/horizon')
    if len(p) != p.date.nunique() * p.security_id.nunique():
        raise ValueError('prices require a complete shared calendar including missing-quote rows')
    for field in ('quote_present', 'data_valid'):
        if not p[field].isin([True, False]).all():
            raise ValueError(f'{field} must be a nonnull boolean')
        p[field] = p[field].astype(bool)
    for field in ('adj_close', 'dollar_volume_proxy', 'adv20_amount'):
        p[field] = pd.to_numeric(p[field], errors='raise').astype(float)
    s['score'] = pd.to_numeric(s['score'], errors='raise').astype(float)
    return p.sort_values(['date', 'security_id']), s


def _reference_metrics(curve, initial_cash):
    values = np.r_[initial_cash, curve.reference_nav.to_numpy(dtype=float)]
    returns = values[1:] / values[:-1] - 1.0
    total_return = float(values[-1] / initial_cash - 1.0)
    # One close-to-close observation per shared session; initial capital is the
    # pre-first-session baseline. No actual-day/252-session hybrid annualisation.
    years = len(curve) / 252.0
    exponent = np.log(values[-1] / initial_cash) / years
    annualized = float(np.expm1(exponent)) if exponent < np.log(np.finfo(float).max) else None
    volatility = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else None
    peaks = np.maximum.accumulate(values)
    return {
        'total_return': total_return,
        'annualized_return': annualized,
        'annualized_volatility': volatility,
        'max_drawdown': float(np.min(values / peaks - 1.0)),
        'sessions': len(curve),
        'annualization': '252_shared_calendar_sessions',
    }


def replay(prices, predictions, start, end, fee_per_side=.0015, top_n=10,
           initial_cash=1e6, invest_fraction=.95, participation=.01):
    """Return ``{'curve': DataFrame, 'trades': DataFrame, 'summary': dict}``.

    Required schemas are the module constants above; h20 score_status='ok' is
    eligible only with a valid signal-day quote, positive signal ADV20 and
    positive signal dollar-volume proxy. Ties use ascending security_id. Every
    selected slot receives ``signal_NAV * invest_fraction / top_n`` dollars;
    insufficient eligible names leave slots in cash. A previous month-end signal
    can execute on the first requested session when that signal is in the input.
    The provided shared calendar determines sessions; a month-end requires the
    next input session to lie in a different month. No holiday calendar is guessed.

    Turnover is sum(abs(filled dollar notional)) / initial_cash (both sides), not
    half-turnover. Buys respect cash and post-fee invested-fraction capacity.
    Existing equity may exceed the cap after appreciation or a capped/failed exit;
    no new equity is added beyond the capacity. Failed orders are never retried.

    ``nav``/``equity`` are NaN on unresolved valuation dates; ``reference_*``
    retain last observed marks. return_metrics is None after ANY valuation gap.
    Reference annualisation is withheld in that case as well. execution_validated
    is always false, regardless of research performance_validated.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1:
        raise ValueError('top_n must be a positive integer')
    for name, value in (('fee_per_side', fee_per_side), ('initial_cash', initial_cash),
                        ('invest_fraction', invest_fraction), ('participation', participation)):
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if not 0 <= fee_per_side < 1:
        raise ValueError('fee_per_side must be between zero and one (exclusive)')
    if initial_cash <= 0 or not 0 <= invest_fraction <= 1 or not 0 < participation <= 1:
        raise ValueError('invalid initial_cash, invest_fraction, or participation')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None or start > end:
        raise ValueError('start/end must be ordered timezone-naive dates')
    p, scores = _prepare(prices, predictions)
    calendar = pd.DatetimeIndex(p.date.unique()).sort_values()
    requested = calendar[(calendar >= start) & (calendar <= end)]
    if requested.empty:
        raise ValueError('no shared calendar sessions in requested replay range')
    signals = {
        signal: execution for signal, execution in zip(calendar[:-1], calendar[1:])
        if signal.to_period('M') != execution.to_period('M') and start <= execution <= end
    }
    sessions = requested.union(pd.DatetimeIndex(list(signals))).sort_values()
    needed = set(sessions)
    price_days = {date: frame.set_index('security_id').to_dict('index')
                  for date, frame in p[p.date.isin(needed)].groupby('date', sort=False)}
    score_days = {date: frame for date, frame in scores.groupby('date', sort=False)}
    cash = float(initial_cash)
    positions, last_marks, last_mark_dates, pending = {}, {}, {}, {}
    curve_rows, orders = [], []
    cumulative_fees = 0.0
    rebalance_count = selections = 0
    tolerance = max(initial_cash * 1e-12, 1e-10)

    def valuation(rows):
        unresolved = [sid for sid in positions if not _valid_quote(rows[sid])]
        value = sum(units * last_marks[sid] for sid, units in positions.items())
        stale = sum(positions[sid] * last_marks[sid] for sid in unresolved)
        return float(value), unresolved, float(stale)

    for date in sessions:
        rows = price_days[date]
        for sid in positions:
            if _valid_quote(rows[sid]):
                last_marks[sid] = float(rows[sid]['adj_close'])
                last_mark_dates[sid] = date
        daily_fees = daily_turnover = 0.0
        if date in pending:
            plan = pending.pop(date)
            rebalance_count += 1
            # Determine deltas for the frozen set before any fill. Missing quotes
            # can support a reference planned amount, but never an execution.
            deltas = []
            for sid in sorted(set(positions) | set(plan['ranks'])):
                row = rows[sid]
                valid = _valid_quote(row)
                mark = float(row['adj_close']) if valid else plan['marks'][sid]
                held_value = positions.get(sid, 0.) * mark
                target = plan['target'] if sid in plan['ranks'] else 0.0
                delta = target - held_value
                side = 'buy' if delta > tolerance else 'sell' if delta < -tolerance else 'hold'
                deltas.append((side, sid, abs(delta), target, mark, valid))
            priority = {'sell': 0, 'hold': 1, 'buy': 2}
            deltas.sort(key=lambda item: (priority[item[0]], plan['ranks'].get(item[1], 0), item[1]))
            for side, sid, amount, target, mark, valid in deltas:
                row = rows[sid]
                adv = plan['adv'][sid]
                volume = row['dollar_volume_proxy']
                cap = min(adv, volume) * participation if _positive(adv) and _positive(volume) else 0.0
                fill = 0.0
                reason = ''
                if side == 'hold':
                    amount, reason = 0.0, 'at_target'
                elif not valid:
                    reason = 'missing_execution_quote'
                elif cap <= 0:
                    reason = 'no_liquidity_capacity'
                elif side == 'sell':
                    fill = min(amount, cap, positions[sid] * mark)
                    if fill < amount - tolerance:
                        reason = 'liquidity_cap'
                else:
                    equity, unresolved, _ = valuation(rows)
                    if unresolved:
                        reason = 'unresolved_execution_valuation'
                    elif not plan['valuation_valid']:
                        reason = 'unresolved_signal_valuation'
                    else:
                        capacity = max(0., (invest_fraction * (cash + equity) - equity)
                                       / (1.0 + invest_fraction * fee_per_side))
                        available_cash = max(0., cash / (1.0 + fee_per_side))
                        fill = min(amount, cap, available_cash, capacity)
                        if fill < amount - tolerance:
                            reason = 'liquidity_cap' if cap <= min(available_cash, capacity) else 'cash_or_equity_cap'
                units = fill / mark if fill > 0 else 0.0
                fee = fill * fee_per_side
                if side == 'sell' and fill > 0:
                    remaining = positions[sid] - units
                    if remaining * mark <= tolerance:
                        positions.pop(sid)
                    else:
                        positions[sid] = remaining
                    cash += fill - fee
                elif side == 'buy' and fill > 0:
                    positions[sid] = positions.get(sid, 0.) + units
                    last_marks[sid], last_mark_dates[sid] = mark, date
                    cash -= fill + fee
                if cash < -tolerance:
                    raise ArithmeticError('negative cash in long-only replay')
                cash = max(0., cash)
                daily_fees += fee
                daily_turnover += fill
                status = ('held' if side == 'hold' else 'unfilled' if fill <= 0
                          else 'partial' if fill < amount - tolerance else 'filled')
                orders.append({
                    'date': date, 'execution_date': date, 'signal_date': plan['signal_date'],
                    'security_id': sid, 'side': side, 'status': status, 'reason': reason,
                    'selection_rank': plan['ranks'].get(sid), 'target_notional': target,
                    'planned_notional': amount, 'filled_notional': fill,
                    'adjusted_research_units': units, 'fill_price': mark if fill > 0 else np.nan,
                    'fees': fee, 'signal_adv20_amount': adv,
                    'execution_dollar_volume_proxy': volume, 'liquidity_cap_notional': cap,
                    'planned_value_uses_reference': not valid,
                })
        cumulative_fees += daily_fees
        equity, unresolved, stale = valuation(rows)
        reference_nav = cash + equity
        if start <= date <= end:
            curve_rows.append({
                'date': date, 'cash': cash, 'equity': equity if not unresolved else np.nan,
                'nav': reference_nav if not unresolved else np.nan,
                'reference_equity': equity, 'reference_nav': reference_nav,
                'holdings_count': len(positions), 'unresolved_holdings_count': len(unresolved),
                'stale_reference_value': stale,
                'valuation_status': 'reference_only' if unresolved else 'observed',
                'fees': daily_fees, 'cumulative_fees': cumulative_fees,
                'turnover_notional': daily_turnover,
            })
        if date in signals:
            candidates = []
            day_scores = score_days.get(date)
            if day_scores is not None:
                for score in day_scores.itertuples(index=False):
                    if score.horizon != 20 or score.score_status != 'ok' or not np.isfinite(score.score):
                        continue
                    row = rows.get(score.security_id)
                    if (row is not None and _valid_quote(row) and _positive(row['adv20_amount'])
                            and _positive(row['dollar_volume_proxy'])):
                        candidates.append((float(score.score), score.security_id))
            selected = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:top_n]
            ranks = {sid: rank for rank, (_, sid) in enumerate(selected, 1)}
            relevant = set(positions) | set(ranks)
            selections += len(ranks)
            pending[signals[date]] = {
                'signal_date': date, 'ranks': ranks,
                'target': reference_nav * invest_fraction / top_n,
                'valuation_valid': not unresolved,
                'adv': {sid: float(rows[sid]['adv20_amount']) for sid in relevant},
                'marks': {sid: float(rows[sid]['adj_close']) if _valid_quote(rows[sid])
                          else last_marks[sid] for sid in relevant},
            }

    curve = pd.DataFrame(curve_rows)
    peaks = np.maximum.accumulate(np.r_[initial_cash, curve.reference_nav.to_numpy()])[1:]
    curve['reference_drawdown'] = curve.reference_nav.to_numpy() / peaks - 1.0
    trades = pd.DataFrame(orders, columns=_TRADE_COLUMNS)
    valid_performance = bool((curve.unresolved_holdings_count == 0).all())
    metrics = _reference_metrics(curve, initial_cash)
    if not valid_performance:
        metrics['annualized_return'] = None
        metrics['annualized_volatility'] = None
    final_rows = price_days[requested[-1]]
    terminal = [{
        'security_id': sid, 'adjusted_research_units': float(positions[sid]),
        'last_observed_adj_close': last_marks[sid],
        'last_observed_date': last_mark_dates[sid].strftime('%Y-%m-%d'),
        'reference_value': float(positions[sid] * last_marks[sid]),
        'valuation_resolved': _valid_quote(final_rows[sid]),
    } for sid in sorted(positions)]
    planned_notional = float(trades.planned_notional.sum())
    filled_notional = float(trades.filled_notional.sum())
    summary = {
        'mode': 'adjusted_price_research_only', 'currency': 'USD',
        'quantity_semantics': 'adjusted_research_units_not_exchange_shares',
        'distribution_cash': 'included_in_adjusted_prices_no_separate_dividend_cash',
        'execution_validated': False, 'performance_validated': valid_performance,
        'performance_scope': 'research_valuation_only' if valid_performance else 'reference_only',
        'return_metrics': dict(metrics) if valid_performance else None,
        'reference_return_metrics': dict(metrics),
        'total_return': metrics['total_return'] if valid_performance else None,
        'annualized_return': metrics['annualized_return'] if valid_performance else None,
        'max_drawdown': metrics['max_drawdown'] if valid_performance else None,
        'initial_cash': float(initial_cash), 'final_cash': cash,
        'final_nav': float(curve.nav.iloc[-1]) if not pd.isna(curve.nav.iloc[-1]) else None,
        'final_reference_nav': float(curve.reference_nav.iloc[-1]),
        'fee_per_side': float(fee_per_side), 'total_fees': cumulative_fees,
        'turnover': filled_notional / initial_cash, 'turnover_notional': filled_notional,
        'turnover_definition': 'both_side_absolute_filled_notional_divided_by_initial_cash',
        'rebalance_count': rebalance_count, 'selected_slots': selections,
        'planned_orders': int((trades.side != 'hold').sum()),
        'filled_orders': int((trades.filled_notional > 0).sum()),
        'partial_orders': int((trades.status == 'partial').sum()),
        'unfilled_orders': int((trades.status == 'unfilled').sum()),
        'unfilled_selected_buys': int(((trades.side == 'buy') & (trades.status == 'unfilled')).sum()),
        'planned_notional': planned_notional, 'filled_notional': filled_notional,
        'fill_ratio': filled_notional / planned_notional if planned_notional > 0 else None,
        'valuation_gap_sessions': int((curve.unresolved_holdings_count > 0).sum()),
        'terminal_unknown_count': int(curve.unresolved_holdings_count.iloc[-1]),
        'terminal_unknown_reference_value': float(curve.stale_reference_value.iloc[-1]),
        'terminal_holdings': terminal,
        'terminal_valuation': 'mark_to_market_no_forced_liquidation',
        'start': requested[0].strftime('%Y-%m-%d'), 'end': requested[-1].strftime('%Y-%m-%d'),
        'signal_horizon': 20, 'top_n': int(top_n), 'invest_fraction': float(invest_fraction),
        'participation': float(participation),
        'sizing': 'fixed_signal_date_usd_targets_converted_at_next_session_close',
        'fill_policy': 'target_deltas_sells_before_buys_no_replacement_no_later_retry',
    }
    return {'curve': curve, 'trades': trades, 'summary': summary}
