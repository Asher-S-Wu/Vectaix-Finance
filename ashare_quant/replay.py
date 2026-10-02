"""Long-only monthly top-score A-share research replay, independent of HK code.

Interface: run_replay(forecasts, bars, calendar, ...) returns JSON-compatible
lists ``equity``, ``trades``, ``positions``, ``cash``, ``audit``, ``events``
and a ``summary`` dictionary.
Inputs accept pandas DataFrames or lists of dictionaries. Calendar accepts an
iterable of dates or a frame with date/cal_date and optional is_open.

Signals: score_status == 'ok', requested horizon, last confirmed exchange
session of a month. A following session in another month must exist in the
calendar; a truncated calendar never proves a month end. Signals and ADV are
frozen at that close. Raw opening prices on the next exchange session drive
fills, shares and fees. No intraday or limit-queue certainty is claimed. The
execution-day amount is an ex-post participation bound, not a signal feature.

Buys expire after that next session. Sells retry on subsequent sessions until
filled or replaced by the next month's targets. Each session's aggregate fill
for a security is at most participation_rate * min(signal ADV20, day amount).
Verified dated cash/bonus actions can be supplied through actions.
No adjusted-price share multipliers, automatic splits or fictional dividends
are applied. An unexplained adjustment-factor change freezes that security.
An affected holding invalidates accounting equity; separate raw-mark and
adjusted-price reference values are reported explicitly as research values.
Cash dividends use an upfront 20% conservative reserve by default, not a claim
that the broker withholds tax on payment day. Actual individual withholding
depends on holding duration and can occur on sale (SAT notice 2015 No.101):
https://www.chinatax.gov.cn/n810341/n810755/c1797427/content.html
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from .rules import (_date, _positive, _true, buy_quantity, execution_status,
                    fee_breakdown, lot_rule, sell_quantity)


def _records(value: Any) -> list[dict]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict('records')
    return [dict(row) for row in value]


def _evidence(value: Any) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and
            value.strip().lower() not in {'nan', 'none', 'null', 'nat', '<na>'})


def _day(value: Any) -> str:
    return _date(value).isoformat()


def _calendar(value: Any) -> list[str]:
    if isinstance(value, pd.DataFrame):
        rows = value.to_dict('records')
    else:
        rows = list(value)
    result = []
    for row in rows:
        if isinstance(row, dict):
            if 'is_open' in row and not _true(row['is_open']):
                continue
            row = row.get('date', row.get('cal_date'))
        result.append(_day(row))
    return sorted(set(result))


def run_replay(forecasts: Any, bars: Any, calendar: Any, *,
               initial_cash: float = 1_000_000., top_k: int = 30,
               horizon: int = 20, participation_rate: float = 0.01,
               commission_rate: float = 0.0003, min_commission: float = 5.,
               start_date: Any = None, end_date: Any = None,
               actions: Any = None) -> dict:
    """Replay frozen predictions without training or using forward outcomes.

    Equal target weights are 1/top_k, leaving unused slots in cash when fewer
    than top_k eligible stocks exist. Input ``amount``/``adv20_amount`` must be
    CNY (Tushare's thousand-CNY amount must be converted upstream). ``volume``
    must be positive, and raw_open/raw_close must be actual unadjusted prices.
    ``adj_close`` is used only for the explicitly non-accounting reference.
    Starting capital is unlevered CNY. Date bounds affect reporting; the latest
    prior signal is still processed so a leading month end can seed day one.
    """
    if not _positive(initial_cash):
        raise ValueError('initial_cash must be finite and positive')
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not _positive(participation_rate) or participation_rate > 0.01:
        raise ValueError('participation_rate must be positive and at most 0.01')
    # Validate assumptions even when no orders eventually fill.
    fee_breakdown(0., 'buy', '2024-01-01', commission_rate=commission_rate,
                  min_commission=min_commission)
    days = _calendar(calendar)
    if not days:
        raise ValueError('calendar contains no trading sessions')
    next_day = dict(zip(days, days[1:]))
    month_ends = {day for day, following in next_day.items()
                  if day[:7] != following[:7]}
    bar_rows = _records(bars)
    indexed: dict[tuple[str, str], dict] = {}
    by_security: dict[str, list[dict]] = defaultdict(list)
    for source in bar_rows:
        row = dict(source)
        row['date'] = _day(row.get('date', row.get('trade_date')))
        row['security_id'] = str(row['security_id'])
        key = row['date'], row['security_id']
        if key in indexed:
            raise ValueError(f'duplicate bar identity: {key}')
        indexed[key] = row
        by_security[row['security_id']].append(row)
    if not indexed:
        raise ValueError('bars are empty')
    # Mark only the change date. The replay freezes affected owned inventory
    # and contemporaneous orders, without using future factors to size orders.
    jumps: set[tuple[str, str]] = set()
    previous_quotes: dict[tuple[str, str], dict] = {}
    for sid, rows in by_security.items():
        prior = None
        previous_row = None
        for row in sorted(rows, key=lambda item: item['date']):
            if previous_row is not None:
                previous_quotes[(row['date'], sid)] = previous_row
            previous_row = row
            factor = row.get('adj_factor')
            if _positive(factor):
                if prior is not None and not math.isclose(float(factor), prior, rel_tol=1e-8):
                    jumps.add((row['date'], sid))
                prior = float(factor)
    signal_rows: dict[str, list[dict]] = defaultdict(list)
    seen_predictions = set()
    all_prediction_dates = []
    for source in _records(forecasts):
        row = dict(source)
        day = _day(row.get('date', row.get('signal_date', row.get('asof_date'))))
        all_prediction_dates.append(day)
        if row.get('horizon', horizon) != horizon:
            continue
        sid = str(row['security_id'])
        key = day, sid
        if key in seen_predictions:
            raise ValueError(f'duplicate forecast identity for horizon {horizon}: {key}')
        seen_predictions.add(key)
        if row.get('score_status') != 'ok':
            continue
        try:
            score = float(row.get('score'))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        row.update(date=day, security_id=sid, score=score)
        signal_rows[day].append(row)
    bar_start = min(day for day, _ in indexed)
    bar_end = max(day for day, _ in indexed)
    default_start = min(all_prediction_dates) if all_prediction_dates else bar_start
    report_start = _day(start_date) if start_date is not None else default_start
    report_end = _day(end_date) if end_date is not None else bar_end
    if report_end > days[-1]:
        raise ValueError('end_date is outside supplied exchange calendar')
    if report_end < report_start:
        raise ValueError('end_date precedes start_date')
    prior_signals = [d for d in signal_rows if d < report_start and d in month_ends]
    loop_start = max(prior_signals) if prior_signals else report_start
    loop_days = [d for d in days if loop_start <= d <= report_end]
    cash = float(initial_cash)
    holdings: dict[str, list[dict]] = defaultdict(list)
    last_quotes: dict[str, dict] = {}
    unresolved: set[str] = set()
    unresolved_observations: set[str] = set()
    orders: list[dict] = []
    output = dict(equity=[], trades=[], positions=[], cash=[], audit=[], events=[])
    output['cash'].append(dict(date=loop_start, event='initial_cash', amount=cash,
                               cash_after=cash, currency='CNY'))
    accounting_failed = False
    incomplete_execution = False
    had_missing_valuation = False
    fees_total = 0.
    tax_reserve = 0.
    dividends_paid = 0.
    cash_timing_complete = True
    receivables: list[dict] = []
    record_events: dict[str, list[dict]] = defaultdict(list)
    ex_events: dict[str, list[dict]] = defaultdict(list)
    action_keys = set()
    for index, source in enumerate(_records(actions) if actions is not None else []):
        action = dict(source, action_id=str(source.get('action_id', index)))
        valid = _true(action.get('verified')) and (_evidence(action.get('source_url')) or _evidence(action.get('source_endpoint')))
        try:
            for field in ('ann_date', 'record_date', 'ex_date'):
                action[field] = _day(action.get(field))
            for field in ('pay_date', 'share_list_date'):
                value = action.get(field)
                action[field] = _day(value) if value is not None and not pd.isna(value) else None
            valid = valid and action['ann_date'] <= action['record_date'] < action['ex_date']
            gross = float(action.get('cash_per_share_pre_tax', 0.))
            multiplier = float(action.get('share_multiplier', 1.))
            valid = valid and math.isfinite(gross) and gross >= 0 and math.isfinite(multiplier) and multiplier >= 1
            valid = valid and (gross == 0 or _evidence(action.get('payment_basis')))
            valid = valid and (multiplier == 1 or (action['share_list_date'] is not None and action['share_list_date'] >= action['ex_date']))
            valid = valid and (not action['pay_date'] or action['pay_date'] >= action['ex_date'])
            action.update(gross=gross, multiplier=multiplier, valid=bool(valid), entitled_quantity=0, record_seen=False)
        except (TypeError, ValueError):
            action['valid'] = False
        if not action['valid']:
            output['events'].append(dict(event_type='unverified_action', security_id=str(action.get('security_id','')), action_id=action['action_id']))
            continue
        action['security_id'] = str(action['security_id'])
        key = action['security_id'], action['ex_date']
        if key in action_keys:
            raise ValueError(f'duplicate corporate action: {key}')
        action_keys.add(key)
        record_events[action['record_date']].append(action)
        ex_events[action['ex_date']].append(action)

    def audit(day: str, sid: str, reason: str, *, order: dict | None = None,
              quantity: int = 0, **extra: Any) -> None:
        nonlocal incomplete_execution
        if reason in {'missing_quote', 'missing_price_limit', 'invalid_price_limit',
                      'missing_signal_adv20', 'invalid_price', 'price_outside_limit_range', 'missing_signal_quote',
                      'unresolved_corporate_action'}:
            incomplete_execution = True
        item = dict(date=day, security_id=sid, reason=reason, quantity=int(quantity))
        if order:
            item.update(side=order['side'], signal_date=order['signal_date'])
        item.update(extra)
        output['audit'].append(item)

    def quantity(sid: str) -> int:
        return sum(lot['quantity'] for lot in holdings[sid])

    def mark(day: str, write: bool = False) -> tuple[float, bool]:
        nonlocal accounting_failed, had_missing_valuation
        raw_value = 0.
        ref_value = 0.
        valid = not accounting_failed
        reference_valid = True
        for sid in sorted(holdings):
            count = quantity(sid)
            if not count:
                continue
            quote = last_quotes.get(sid, {})
            raw = quote.get('raw_close')
            current = indexed.get((day, sid), {})
            stale = quote.get('date') != day or not _true(current.get('quote_present'))
            if sid in unresolved:
                status = 'unresolved_corporate_action'
                valid = False
                accounting_failed = True
            elif not _positive(raw) or stale:
                status = 'stale_or_missing_quote'
                valid = False
                had_missing_valuation = True
            else:
                status = 'ok'
            value = count * float(raw) if _positive(raw) else 0.
            raw_value += value
            adj = quote.get('adj_close')
            reference = 0.
            for lot in holdings[sid]:
                if _positive(adj) and _positive(lot['reference_adj']):
                    reference += (lot['quantity'] * lot['entry_price'] *
                                  float(adj) / lot['reference_adj'])
                else:
                    reference_valid = False
            ref_value += reference
            if write:
                output['positions'].append(dict(
                    date=day, security_id=sid, quantity=count,
                    sellable_quantity=sum(lot['quantity'] for lot in holdings[sid]
                                          if lot['acquired_date'] < day and lot.get('available_date', day) <= day),
                    raw_close=float(raw) if _positive(raw) else None,
                    raw_market_value=value if _positive(raw) else None,
                    reference_market_value=reference if reference_valid else None,
                    valuation_status=status, currency='CNY'))
        receivable_value = sum(item['net_amount'] for item in receivables)
        raw_equity = cash + raw_value + receivable_value
        if write:
            output['equity'].append(dict(
                date=day, cash=cash, raw_market_value=raw_value, receivables=receivable_value,
                raw_mark_equity=raw_equity,
                equity=raw_equity if valid else None,
                reference_equity=cash + ref_value + receivable_value if reference_valid else None,
                valuation_valid=valid, currency='CNY'))
        return raw_equity, valid

    def exposed(sid: str) -> bool:
        return quantity(sid) > 0 or any(order['security_id'] == sid for order in orders)

    for day in loop_days:
        # A past event with no owned entitlement cannot taint newly purchased
        # raw shares forever. Keep held inventory frozen; a new event below
        # still blocks contemporaneous orders and remains in the audit history.
        unresolved.intersection_update({sid for sid in unresolved if quantity(sid)})
        # Only verified terms that reconcile the actual factor change release
        # the adjustment guard. Entitlements are frozen at record-date close.
        resolved_today = set()
        for action in ex_events.get(day, []):
            sid = action['security_id']
            bar = indexed.get((day, sid), {})
            prior = previous_quotes.get((day, sid), {})
            gross, multiplier = action['gross'], action['multiplier']
            consistent = all(_positive(v) for v in (prior.get('raw_close'), prior.get('adj_factor'), bar.get('adj_factor')))
            if consistent:
                expected_ex = (float(prior['raw_close']) - gross) / multiplier
                implied_ex = float(prior['raw_close']) * float(prior['adj_factor']) / float(bar['adj_factor'])
                consistent = expected_ex > 0 and abs(expected_ex - implied_ex) <= 0.011
            entitled = action['entitled_quantity']
            bonus = entitled * (multiplier - 1.)
            consistent = consistent and math.isclose(bonus, round(bonus), abs_tol=1e-7)
            # A position already present needs observed record-date evidence.
            consistent = consistent and (action['record_seen'] or quantity(sid) == 0)
            if not consistent:
                if exposed(sid):
                    audit(day, sid, 'unresolved_corporate_action', detail='Verified terms do not reconcile adjustment or entitlement')
                else:
                    output['events'].append(dict(date=day, security_id=sid, event_type='unresolved_unheld_action'))
                unresolved.add(sid)
                unresolved_observations.add(sid)
                if quantity(sid):
                    accounting_failed = True
                continue
            resolved_today.add(sid)
            net = gross * .8
            net_basis = 'conservative_20_percent_reserve'
            if action.get('tax_basis') == 'verified_investor_specific':
                supplied_net = action.get('cash_per_share_after_tax')
                if supplied_net is not None and math.isfinite(float(supplied_net)) and 0 <= float(supplied_net) <= gross:
                    net = float(supplied_net)
                    net_basis = 'verified_investor_specific'
            net_amount = round(entitled * net, 2)
            reserve = round(entitled * (gross - net), 2)
            tax_reserve += reserve
            if net_amount:
                receivables.append(dict(action_id=action['action_id'], security_id=sid,
                                        pay_date=action['pay_date'], net_amount=net_amount))
                if not action['pay_date']:
                    cash_timing_complete = False
                    audit(day, sid, 'unknown_dividend_payment_date')
            if bonus:
                holdings[sid].append(dict(quantity=int(round(bonus)), acquired_date=action['record_date'],
                                         available_date=action['share_list_date'], entry_price=0., reference_adj=None))
            # A resolved action restarts reference anchors so cash is not counted
            # both in a total-return factor and in the cash/receivable ledger.
            if _positive(bar.get('raw_close')) and _positive(bar.get('adj_close')):
                for lot in holdings[sid]:
                    lot['entry_price'] = float(bar['raw_close'])
                    lot['reference_adj'] = float(bar['adj_close'])
            output['events'].append(dict(date=day, event_type='ex_entitlement', security_id=sid,
                                         action_id=action['action_id'], entitled_quantity=entitled,
                                         gross_amount=round(entitled*gross,2), net_amount=net_amount,
                                         tax_reserve=reserve, tax_basis=net_basis,
                                         bonus_shares=int(round(bonus)), factor_reconciled=True,
                                         source_url=action.get('source_url') if _evidence(action.get('source_url')) else None,
                                         source_endpoint=action.get('source_endpoint') if _evidence(action.get('source_endpoint')) else None,
                                         payment_basis=action.get('payment_basis') if _evidence(action.get('payment_basis')) else None))
            # Existing pending quantities were sized before this ex-date. Adjust
            # them only from the verified event, never from the factor itself.
            for order in orders:
                if order['security_id'] == sid and order['signal_date'] <= action['record_date']:
                    order['quantity'] = int(math.floor(order['quantity'] * multiplier + 1e-8))
        retained_receivables = []
        for receipt in receivables:
            if receipt['pay_date'] is not None and receipt['pay_date'] <= day:
                cash += receipt['net_amount']
                dividends_paid += receipt['net_amount']
                output['events'].append(dict(date=day, event_type='cash_payment', **receipt))
                output['cash'].append(dict(date=day, security_id=receipt['security_id'],
                                           event='cash_dividend', amount=receipt['net_amount'], cash_after=cash, currency='CNY'))
            else:
                retained_receivables.append(receipt)
        receivables = retained_receivables
        # Corporate-action evidence is checked before any next-session fill.
        for sid in by_security:
            bar = indexed.get((day, sid))
            if bar is None:
                continue
            if ((day, sid) in jumps and sid not in resolved_today) or _true(bar.get('corporate_action_unresolved')):
                if sid not in unresolved:
                    if exposed(sid):
                        audit(day, sid, 'unresolved_corporate_action',
                              held_quantity=quantity(sid),
                              detail='Adjustment-factor change is not a verified share/cash action')
                    else:
                        output['events'].append(dict(date=day, security_id=sid, event_type='unresolved_unheld_action'))
                unresolved.add(sid)
                unresolved_observations.add(sid)
                if quantity(sid):
                    accounting_failed = True
            if _true(bar.get('quote_present')) and _positive(bar.get('raw_close')):
                last_quotes[sid] = bar
        remaining = []
        used_notional: dict[str, float] = defaultdict(float)
        for order in sorted(orders, key=lambda o: (o['side'] != 'sell', o['rank'], o['security_id'])):
            if order['execute_date'] > day:
                remaining.append(order)
                continue
            sid, side = order['security_id'], order['side']
            original = indexed.get((day, sid), {})
            bar = dict(original, date=day)
            if sid in unresolved:
                bar['corporate_action_unresolved'] = True
            allowed, reason = execution_status(bar, side)
            if not allowed:
                audit(day, sid, reason, order=order, quantity=order['quantity'])
                if side == 'sell':
                    remaining.append(order)
                continue
            if not _positive(order['signal_adv20_amount']):
                audit(day, sid, 'missing_signal_adv20', order=order,
                      quantity=order['quantity'])
                if side == 'sell':
                    remaining.append(order)
                continue
            price = float(bar['raw_open'])
            limit = participation_rate * min(float(order['signal_adv20_amount']),
                                             float(bar['amount']))
            cap = max(0., limit - used_notional[sid])
            count = int(order['quantity'])
            if side == 'buy':
                if accounting_failed:
                    audit(day, sid, 'accounting_invalid', order=order, quantity=count)
                    continue
                count = min(count, buy_quantity(min(cap, cash), price, sid))
                count = buy_quantity(count * price, price, sid)
                while count > 0:
                    fee = fee_breakdown(count * price, side, day, sid,
                                        commission_rate=commission_rate,
                                        min_commission=min_commission)
                    if count * price + fee['total'] <= cash + 1e-8:
                        break
                    _, step = lot_rule(sid)
                    count = buy_quantity((count - step) * price, price, sid)
            else:
                available = sum(lot['quantity'] for lot in holdings[sid]
                                if lot['acquired_date'] < day and lot.get('available_date', day) <= day)
                if available == 0 and quantity(sid):
                    audit(day, sid, 't_plus_one', order=order, quantity=count)
                    remaining.append(order)
                    continue
                count = min(count, available, int(math.floor(cap / price + 1e-10)))
                count = sell_quantity(count, quantity(sid), sid)
            if count <= 0:
                audit(day, sid, 'below_lot_or_cash_or_participation', order=order,
                      quantity=order['quantity'])
                if side == 'sell' and quantity(sid):
                    remaining.append(order)
                continue
            fee = fee_breakdown(count * price, side, day, sid,
                                commission_rate=commission_rate,
                                min_commission=min_commission)
            notional = count * price
            if side == 'buy':
                cash -= notional + fee['total']
                adj, close = bar.get('adj_close'), bar.get('raw_close')
                ref_adj = (float(adj) * price / float(close)
                           if _positive(adj) and _positive(close) else None)
                holdings[sid].append(dict(quantity=count, acquired_date=day,
                                          entry_price=price, reference_adj=ref_adj))
            else:
                cash += notional - fee['total']
                to_sell = count
                for lot in holdings[sid]:
                    if lot['acquired_date'] >= day or lot.get('available_date', day) > day:
                        continue
                    sold = min(lot['quantity'], to_sell)
                    lot['quantity'] -= sold
                    to_sell -= sold
                holdings[sid] = [lot for lot in holdings[sid] if lot['quantity']]
            used_notional[sid] += notional
            fees_total += fee['total']
            output['trades'].append(dict(
                date=day, signal_date=order['signal_date'], security_id=sid,
                side=side, quantity=count, price=price, notional=notional,
                fee=fee['total'], commission=fee['commission'],
                stamp_duty=fee['stamp_duty'], transfer_fee=fee['transfer_fee'],
                cash_after=cash, signal_adv20_amount=order['signal_adv20_amount'],
                execution_amount=float(bar['amount']),
                participation_limit_notional=limit, currency='CNY',
                execution_basis='next_session_raw_open_daily_liquidity_proxy'))
            output['cash'].append(dict(
                date=day, security_id=sid, event=side,
                amount=(-notional - fee['total'] if side == 'buy' else notional - fee['total']),
                cash_after=cash, currency='CNY'))
            if count < order['quantity']:
                left = dict(order, quantity=order['quantity'] - count)
                audit(day, sid, 'partial_fill', order=order, quantity=left['quantity'])
                if side == 'sell':
                    remaining.append(left)
        orders = remaining
        for action in record_events.get(day, []):
            action['entitled_quantity'] = quantity(action['security_id'])
            action['record_seen'] = True
            output['events'].append(dict(date=day, event_type='entitlement', security_id=action['security_id'],
                                         action_id=action['action_id'], entitled_quantity=action['entitled_quantity']))
        nav, valuation_ok = mark(day, write=day >= report_start)
        if day not in month_ends or day not in signal_rows:
            continue
        if not valuation_ok:
            audit(day, '', 'rebalance_valuation_unavailable')
            continue
        candidates = sorted(signal_rows[day], key=lambda r: (-r['score'], r['security_id']))
        selected = candidates[:top_k]
        targets, advs, ranks = {}, {}, {}
        for rank, row in enumerate(selected):
            sid = row['security_id']
            quote = indexed.get((day, sid), {})
            raw = quote.get('raw_close')
            if sid in unresolved:
                audit(day, sid, 'unresolved_corporate_action')
                continue
            if not _true(quote.get('quote_present')) or not _positive(raw):
                audit(day, sid, 'missing_signal_quote')
                continue
            adv = quote.get('adv20_amount', row.get('adv20_amount'))
            if not _positive(adv):
                audit(day, sid, 'missing_signal_adv20')
                continue
            targets[sid] = buy_quantity(nav / top_k, float(raw), sid)
            advs[sid], ranks[sid] = float(adv), rank
        new_orders = []
        for sid in sorted(set(targets) | {s for s in holdings if quantity(s)}):
            delta = targets.get(sid, 0) - quantity(sid)
            if not delta:
                continue
            quote = indexed.get((day, sid), {})
            adv = advs.get(sid, quote.get('adv20_amount'))
            side = 'buy' if delta > 0 else 'sell'
            count = abs(delta) if side == 'buy' else sell_quantity(-delta, quantity(sid), sid)
            if not count:
                continue
            new_orders.append(dict(signal_date=day, execute_date=next_day[day],
                                   security_id=sid, side=side, quantity=count,
                                   signal_adv20_amount=float(adv) if _positive(adv) else None,
                                   rank=ranks.get(sid, top_k)))
        # New targets supersede old delayed sells rather than double-selling.
        for order in orders:
            audit(day, order['security_id'], 'superseded_by_rebalance', order=order,
                  quantity=order['quantity'])
        orders = new_orders

    for order in orders:
        audit(report_end, order['security_id'], 'pending_at_end', order=order,
              quantity=order['quantity'])
    equity = output['equity']
    last = equity[-1] if equity else dict(equity=initial_cash,
                                         reference_equity=initial_cash,
                                         raw_mark_equity=initial_cash)
    valid_values = [e['equity'] for e in equity if e['equity'] is not None]
    valuation_valid = not accounting_failed and not had_missing_valuation
    final_equity = last['equity'] if valuation_valid else None
    total_return = final_equity / initial_cash - 1 if final_equity is not None else None
    drawdown = None
    if valuation_valid and valid_values:
        peak, drawdown = float(initial_cash), 0.
        for value in valid_values:
            peak = max(peak, value)
            drawdown = min(drawdown, value / peak - 1)
    reasons = dict(sorted(Counter(row['reason'] for row in output['audit']).items()))
    output['summary'] = dict(
        status='ok' if valuation_valid and not incomplete_execution else 'research_limited',
        initial_cash=float(initial_cash), ending_cash=cash,
        ending_equity=final_equity, ending_raw_mark_equity=last['raw_mark_equity'],
        ending_reference_equity=last['reference_equity'], total_return=total_return,
        max_drawdown=drawdown, fees=fees_total, trade_count=len(output['trades']),
        open_positions=sum(quantity(sid) > 0 for sid in holdings),
        pending_sell_orders=sum(o['side'] == 'sell' for o in orders),
        valuation_valid=valuation_valid, execution_valid=not accounting_failed and not incomplete_execution,
        execution_data_complete=not incomplete_execution,
        unresolved_corporate_actions=sorted(unresolved_observations),
        frozen_corporate_action_positions=sorted(sid for sid in unresolved if quantity(sid)),
        audit_counts=reasons,
        start_date=report_start, end_date=report_end, currency='CNY',
        top_k=top_k, horizon=horizon, participation_rate=participation_rate,
        commission_rate=commission_rate, min_commission=min_commission,
        commission_basis='gross_all_in_including_exchange_and_regulatory_fees',
        execution_model='next_session_raw_open_with_daily_liquidity_cap_not_auction_queue',
        corporate_action_policy='freeze_unresolved_never_convert_adjustment_ratio_to_shares',
        reference_valuation='adjusted_price_reference_only_not_verified_cash_or_share_accounting',
        cash_dividends_paid=dividends_paid, dividend_tax_reserve=tax_reserve,
        dividend_tax_policy='conservative_20_percent_reserve',
        exact_investor_tax_known=tax_reserve == 0.,
        accounting_basis='cash_equity_with_conservative_upfront_tax_reserve',
        cash_timing_complete=cash_timing_complete,
        ending_receivables=sum(item['net_amount'] for item in receivables),
        training_performed=False)
    return output


def run_reference_replay(forecasts: Any, bars: Any, calendar: Any, *,
                         fee: float = .0025, end_date: Any = None,
                         initial_cash: float = 1_000_000., top_k: int = 30,
                         horizon: int = 20, equity_weight: float = .95,
                         participation_rate: float = .01,
                         start_date: Any = None) -> dict:
    """HK-comparable fractional adjusted-unit research, never actual shares.

    Monthly positions are sold and re-entered at the *next session's adjusted
    close*. ``fee`` is a simplified symmetric, per-side cost/slippage stress;
    exact brokerage minimums, tax and lots do not apply to these research units.
    Side-specific limits are nevertheless checked against raw closing prices.
    Buys expire that session and sells retry. Only explicitly suspended quotes
    can carry last marks, disclosed as stale; an unresolved terminal quote
    never generates a certified ending equity or an assumed zero recovery.
    A dated reference_continuity_break freezes previously held adjusted units:
    their unknown bridge cannot generate a valuation or sale proceeds. A flat
    security is blocked only on the break session, allowing later clean entries.
    If that block affects a selected entry, portfolio performance remains
    unavailable: retaining cash is not evidence for the missing trade outcome.
    """
    if not _positive(initial_cash) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError('positive initial_cash and integer top_k required')
    if not math.isfinite(fee) or not 0 <= fee < 1:
        raise ValueError('fee must be a finite per-side fraction in [0,1)')
    if not _positive(equity_weight) or equity_weight > 1:
        raise ValueError('equity_weight must be in (0,1]')
    if not _positive(participation_rate) or participation_rate > .01:
        raise ValueError('participation_rate must be in (0,.01]')
    days = _calendar(calendar)
    following = dict(zip(days, days[1:]))
    month_ends = {d for d,n in following.items() if d[:7] != n[:7]}
    indexed = {}
    for source in _records(bars):
        row = dict(source, date=_day(source['date']), security_id=str(source['security_id']))
        key = row['date'], row['security_id']
        if key in indexed:
            raise ValueError(f'duplicate bar identity: {key}')
        indexed[key] = row
    if not indexed or not days:
        raise ValueError('nonempty bars and exchange calendar required')
    signals: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    forecast_days = []
    for source in _records(forecasts):
        day, sid = _day(source['date']), str(source['security_id'])
        forecast_days.append(day)
        if source.get('horizon', horizon) != horizon:
            continue
        if (day,sid) in seen:
            raise ValueError(f'duplicate forecast identity: {(day,sid)}')
        seen.add((day,sid))
        if source.get('score_status') != 'ok' or day not in month_ends:
            continue
        try:
            score = float(source['score'])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(score):
            signals[day].append(dict(source, security_id=sid, date=day, score=score))
    stop = _day(end_date) if end_date is not None else max(d for d,_ in indexed)
    if stop > days[-1]:
        raise ValueError('end_date is outside supplied exchange calendar')
    first_executions = [following[d] for d in signals if following[d] <= stop]
    report_start = (_day(start_date) if start_date is not None else
                    min(first_executions or forecast_days or days))
    if stop < report_start:
        raise ValueError('end_date precedes start_date')
    prior = [d for d in signals if d < report_start]
    begin = max(prior) if prior else report_start
    cash = float(initial_cash)
    positions, pending, last_marks = {}, {}, {}
    executions = {following[d]: d for d in signals if d >= begin}
    out = dict(equity=[], trades=[], positions=[], audit=[], completed=[])
    fees = 0.
    next_position_id = 0
    missing_days = stale_days = 0
    terminal_uncertain = False
    unverified_entry_outcomes: set[str] = set()

    def record(day: str, sid: str, side: str, reason: str) -> None:
        out['audit'].append(dict(date=day, security_id=sid, side=side, reason=reason))

    def executable(day: str, sid: str, side: str) -> tuple[bool, dict]:
        quote = indexed.get((day,sid), {})
        check = dict(quote, date=day, execution_price=quote.get('raw_close'))
        # Ordinary provider adjustments remain part of this reference model,
        # but known discontinuities cannot be treated as a valid unit bridge.
        check.pop('corporate_action_unresolved', None)
        if positions.get(sid, {}).get('continuity_unverified', False):
            record(day, sid, side, 'unverified_reference_continuity')
            return False, quote
        if _true(quote.get('reference_continuity_break')):
            if side == 'buy':
                unverified_entry_outcomes.add(sid)
            record(day, sid, side, 'reference_continuity_break')
            return False, quote
        allowed, reason = execution_status(check, side)
        if allowed and not _positive(quote.get('adj_close')):
            allowed, reason = False, 'missing_adjusted_execution_price'
        if not allowed:
            record(day, sid, side, reason)
        return allowed, quote

    for day in (d for d in days if begin <= d <= stop):
        # Check existing units before any sale/rebalance on the flagged date.
        # Fresh entries after an earlier flat break have no old units to bridge.
        for sid, position in positions.items():
            if _true(indexed.get((day, sid), {}).get('reference_continuity_break')):
                if not position.get('continuity_unverified', False):
                    position['continuity_unverified'] = True
                    position['continuity_break_date'] = day
                    record(day, sid, 'valuation', 'unverified_reference_continuity')
        used = defaultdict(float)
        signal_day = executions.get(day)
        if signal_day is not None:
            for sid, position in positions.items():
                quote = indexed.get((signal_day,sid), {})
                adv = quote.get('adv20_amount')
                pending[sid] = float(adv) if _positive(adv) else None
        for sid in sorted(list(pending)):
            position = positions[sid]
            allowed, quote = executable(day, sid, 'sell')
            if not allowed:
                continue
            if position['entry_date'] >= day:
                record(day,sid,'sell','t_plus_one')
                continue
            adv = pending[sid]
            if not _positive(adv):
                record(day,sid,'sell','missing_signal_adv20')
                continue
            price = float(quote['adj_close'])
            cap = participation_rate * min(adv, float(quote['amount']))
            units = min(position['units'], cap / price)
            value, cost = units * price, units * price * fee
            cash += value - cost
            fees += cost
            used[sid] += value
            position['proceeds'] += value - cost
            out['trades'].append(dict(date=day, security_id=sid, side='sell',
                                      units=units, price=price, notional=value, value=value,
                                      fee=cost, cash_after=cash, position_id=position['id'],
                                      signal_adv20_amount=adv, execution_amount=float(quote['amount']),
                                      participation_limit_notional=cap, actual_shares=False))
            position['units'] -= units
            if position['units'] <= 1e-10:
                out['completed'].append(dict(security_id=sid, position_id=position['id'],
                                             entry_date=position['entry_date'], exit_date=day,
                                             net_return=position['proceeds']/position['original_cost']-1))
                del positions[sid]
                del pending[sid]
            else:
                record(day,sid,'sell','partial_fill_pending')
        if signal_day is not None:
            budget = cash * equity_weight / top_k
            selected = sorted(signals[signal_day], key=lambda row: (-row['score'],row['security_id']))[:top_k]
            for row in selected:
                sid = row['security_id']
                if sid in positions:
                    record(day,sid,'buy','existing_position_waiting_sell')
                    continue
                allowed, quote = executable(day,sid,'buy')
                if not allowed:
                    continue
                signal_quote = indexed.get((signal_day,sid), {})
                adv = signal_quote.get('adv20_amount',row.get('adv20_amount'))
                if not _positive(adv):
                    record(day,sid,'buy','missing_signal_adv20')
                    continue
                cap = participation_rate * min(float(adv),float(quote['amount']))
                value = min(budget/(1+fee),max(0.,cap-used[sid]),cash/(1+fee))
                if value <= 0:
                    record(day,sid,'buy','participation_capacity_used')
                    continue
                price = float(quote['adj_close'])
                cost, units = value * fee, value / price
                cash -= value + cost
                fees += cost
                used[sid] += value
                next_position_id += 1
                positions[sid] = dict(units=units, entry_date=day, original_cost=value+cost,
                                      proceeds=0., id=next_position_id)
                out['trades'].append(dict(date=day, signal_date=signal_day, security_id=sid,
                                          side='buy', units=units, price=price, value=value,
                                          notional=value, fee=cost, cash_after=cash,
                                          position_id=next_position_id,
                                          signal_adv20_amount=float(adv), execution_amount=float(quote['amount']),
                                          participation_limit_notional=cap, actual_shares=False))
        market_value = 0.
        missing = stale = 0
        for sid, position in sorted(positions.items()):
            quote = indexed.get((day,sid), {})
            adj = quote.get('adj_close')
            observed = _true(quote.get('quote_present')) and _positive(adj)
            if position.get('continuity_unverified', False):
                mark, status = None, 'unverified_reference_continuity'
                missing += 1
            elif observed:
                last_marks[sid] = float(adj)
                mark, status = float(adj), 'observed_adjusted_reference'
            elif _true(quote.get('suspended')) and sid in last_marks:
                mark, status = last_marks[sid], 'stale_suspension_reference'
                stale += 1
            else:
                mark, status = None, 'missing_reference_valuation'
                missing += 1
            value = position['units'] * mark if mark is not None else None
            if value is not None:
                market_value += value
            if day >= report_start:
                out['positions'].append(dict(date=day, security_id=sid, units=position['units'],
                                             mark=mark, value=value, valuation_status=status,
                                             actual_shares=False, sell_pending=sid in pending,
                                             reference_continuity_valid=not position.get('continuity_unverified', False),
                                             continuity_break_date=position.get('continuity_break_date')))
        if day >= report_start:
            missing_days += bool(missing)
            stale_days += bool(stale)
            terminal_uncertain = bool(missing or stale or unverified_entry_outcomes)
            out['equity'].append(dict(date=day, cash=cash,
                                      holdings_value=market_value if not missing else None,
                                      equity=cash+market_value if not missing and not unverified_entry_outcomes else None,
                                      missing_marks=missing, stale_marks=stale,
                                      unverified_continuity_positions=sum(bool(p.get('continuity_unverified')) for p in positions.values()),
                                      unverified_entry_outcomes=len(unverified_entry_outcomes),
                                      reference_only=True))
    daily = out['equity']
    ending_reference = daily[-1]['equity'] if daily else initial_cash
    ending = ending_reference if not terminal_uncertain else None
    drawdown, peak = 0., initial_cash
    if missing_days or unverified_entry_outcomes:
        drawdown = None
    else:
        for row in daily:
            peak = max(peak,row['equity'])
            drawdown = min(drawdown,row['equity']/peak-1)
    elapsed = (_date(stop)-_date(report_start)).days
    out['summary'] = dict(
        status='incomplete_valuation' if missing_days or terminal_uncertain else 'complete_adjusted_unit_research',
        methodology='research_only_adjusted_unit_monthly_replay',
        execution_validated=False, actual_shares=False, training_performed=False,
        initial_cash=initial_cash, ending_cash=cash, ending_equity=ending,
        ending_reference_equity=ending_reference,
        total_return=ending/initial_cash-1 if ending is not None else None,
        cagr=(ending/initial_cash)**(365.25/elapsed)-1 if ending is not None and elapsed>0 else None,
        max_drawdown_daily=drawdown, max_drawdown=drawdown, fees=fees, fee_per_side=fee,
        cost_model='simplified_symmetric_fee_plus_slippage_not_exact_tax_or_brokerage',
        top_k=top_k, equity_weight=equity_weight, participation_rate=participation_rate,
        open_positions=len(positions), pending_sells=len(pending),
        trade_count=len(out['trades']), completed_positions=len(out['completed']),
        profitable_hit_rate=(sum(p['net_return']>0 for p in out['completed'])/len(out['completed'])
                             if out['completed'] and not unverified_entry_outcomes and not any(p.get('continuity_unverified') for p in positions.values()) else None),
        missing_valuation_days=missing_days, stale_valuation_days=stale_days,
        terminal_valuation_complete=not terminal_uncertain,
        reference_continuity_valid=not unverified_entry_outcomes and not any(p.get('continuity_unverified') for p in positions.values()),
        unresolved_reference_continuity=sorted({sid for sid,p in positions.items() if p.get('continuity_unverified')} | unverified_entry_outcomes),
        unverified_entry_outcomes=sorted(unverified_entry_outcomes),
        start_date=report_start, end_date=stop, currency='CNY',
        execution_price_basis='next_session_adjusted_close_raw_close_limit_check',
        corporate_action_policy='provider_adjusted_reference_no_verified_share_cash_accounting')
    out['daily'] = out['equity']
    out['holdings'] = out['positions']
    out['orders'] = out['audit']
    return out
