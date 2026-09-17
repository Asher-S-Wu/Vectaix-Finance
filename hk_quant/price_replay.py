"""冻结预测的逐日复权单位回放，成交、参考估值与现金分别记账。"""
import numpy as np
import pandas as pd


def replay_prices(ranking, bars, calendar, *, initial_cash=1_000_000., equity_weight=.95,
                  top_n=30, fee=.0025, max_participation=.01, terminal_actions=None, transfers=None, end=None):
    dates = pd.DatetimeIndex(calendar).sort_values()
    if dates.has_duplicates or bars.duplicated(['date', 'security_id']).any():
        raise ValueError('日历或行情存在重复记录')
    signals = pd.DatetimeIndex(sorted(ranking.date.unique()))
    if len(signals) < 2 or not signals.isin(dates).all():
        raise ValueError('至少需要两个交易日信号')
    if signals[-1] >= dates[-1]:
        raise ValueError('最后信号日之后缺少执行交易日')
    executions = {dates[dates.get_loc(day) + 1]: day for day in signals}
    first_entry = min(executions)
    final_rebalance = max(executions)
    end = pd.Timestamp(end) if end is not None else final_rebalance
    if end < final_rebalance or end > dates[-1]:
        raise ValueError('回放截止日超出行情区间或早于最后调仓日')
    if not (initial_cash > 0 and 0 < equity_weight <= 1 and 0 <= fee < 1
            and 0 < max_participation <= 1 and top_n > 0):
        raise ValueError('现金、仓位、费用或成交参与率无效')
    actions = pd.DataFrame(columns=['security_id', 'effective_date', 'payment_date', 'known_date',
                                    'adjusted_cash_per_unit', 'source_url', 'payment_basis'])
    if terminal_actions is not None:
        actions = terminal_actions.copy()
    for column in ['effective_date', 'payment_date', 'known_date']:
        actions[column] = pd.to_datetime(actions[column])
    if actions.security_id.duplicated().any():
        raise ValueError('同一证券有重复终止结算')
    for a in actions.itertuples():
        if (pd.isna(a.known_date) or pd.isna(a.effective_date) or pd.isna(a.payment_date)
                or a.known_date > a.effective_date or a.payment_date < a.effective_date):
            raise ValueError('终止结算公开时间、生效日或派付日无效')
        if (not isinstance(a.source_url, str) or not a.source_url.startswith('https://')
                or a.payment_basis != 'issuer_final_schedule_simulated'
                or not np.isfinite(a.adjusted_cash_per_unit) or a.adjusted_cash_per_unit < 0):
            raise ValueError('终止结算缺少金额或明确公告依据')
    transfers = (pd.DataFrame(columns=['security_id', 'successor_id', 'effective_date', 'known_date',
                                       'unit_multiplier', 'source_url']) if transfers is None else transfers.copy())
    for column in ['effective_date', 'known_date']:
        transfers[column] = pd.to_datetime(transfers[column])
    if transfers.security_id.duplicated().any():
        raise ValueError('同一证券有重复转板记录')
    for transfer in transfers.itertuples():
        if (pd.isna(transfer.known_date) or pd.isna(transfer.effective_date)
                or transfer.known_date > transfer.effective_date or transfer.unit_multiplier <= 0
                or not np.isfinite(transfer.unit_multiplier) or not transfer.source_url.startswith('https://')):
            raise ValueError('转板公开时间、换股比例或来源无效')
    timeline = dates[(dates >= signals[0]) & (dates <= end)]
    event_dates = pd.DatetimeIndex(pd.concat([actions.effective_date, actions.payment_date,
                                            transfers.effective_date]))
    timeline = timeline.union(event_dates[(event_dates >= timeline[0]) & (event_dates <= end)].unique()).sort_values()
    markets = {day: group.set_index('security_id') for day, group in bars.groupby('date')}
    selections = {day: group.sort_values(['score', 'security_id'], ascending=[False, True]).head(top_n)
                  for day, group in ranking.groupby('date')}
    cash = float(initial_cash)
    positions, pending, receivables, terminated = {}, set(), {}, set()
    daily, trades, orders, completed, events, gaps, holdings = [], [], [], [], [], [], []
    next_id = 0

    def close_position(sid, position, day, proceeds):
        position['proceeds'] += proceeds
        completed.append(dict(position_id=position['id'], security_id=sid, entry_date=position['entry_date'],
                              exit_date=day, cost=position['original_cost'], proceeds=position['proceeds'],
                              net_return=position['proceeds'] / position['original_cost'] - 1))

    for day in timeline:
        quotes = markets.get(day, pd.DataFrame(columns=['adj_close', 'quote_present', 'data_valid', 'amount']))
        used_turnover = {}
        for transfer in transfers.loc[transfers.effective_date.eq(day)].itertuples():
            sid, successor = transfer.security_id, transfer.successor_id
            terminated.add(sid)
            if sid in positions:
                if successor in positions:
                    raise ValueError('转板新旧证券持仓冲突')
                positions[successor] = positions.pop(sid)
                positions[successor]['units'] *= transfer.unit_multiplier
                if sid in pending:
                    pending.remove(sid)
                    pending.add(successor)
                events.append(dict(date=day, security_id=sid, successor_id=successor, event='board_transfer',
                                   source_url=transfer.source_url, unit_multiplier=transfer.unit_multiplier))
        for a in actions.loc[actions.effective_date.eq(day)].itertuples():
            sid = a.security_id
            terminated.add(sid)
            pending.discard(sid)
            if sid in positions:
                position = positions.pop(sid)
                amount = position['units'] * a.adjusted_cash_per_unit
                receivables[sid] = dict(amount=amount, position=position, payment_date=a.payment_date)
                events.append(dict(date=day, security_id=sid, event='receivable', value=amount,
                                   source_url=a.source_url, payment_basis=a.payment_basis))
        for a in actions.loc[actions.payment_date.eq(day)].itertuples():
            if a.security_id in receivables:
                claim = receivables.pop(a.security_id)
                cash += claim['amount']
                close_position(a.security_id, claim['position'], day, claim['amount'])
                events.append(dict(date=day, security_id=a.security_id, event='payment', value=claim['amount'],
                                   source_url=a.source_url, payment_basis=a.payment_basis))
        if day not in dates:
            continue
        is_rebalance = day in executions
        if is_rebalance:
            pending.update(positions)

        def tradeable(sid):
            if sid not in quotes.index or sid in terminated:
                return False
            q = quotes.loc[sid]
            return (q.quote_present == True and q.data_valid == True
                    and np.isfinite(q.adj_close) and q.adj_close > 0
                    and np.isfinite(q.amount) and q.amount > 0)

        for sid in sorted(pending):
            position = positions[sid]
            if not tradeable(sid):
                orders.append(dict(date=day, security_id=sid, status='sell_pending_no_quote'))
                continue
            q = quotes.loc[sid]
            units = min(position['units'], max_participation * q.amount / q.adj_close)
            value = units * q.adj_close
            used_turnover[sid] = value
            commission = value * fee
            cash += value - commission
            trades.append(dict(date=day, security_id=sid, position_id=position['id'], side='sell',
                               units=units, price=q.adj_close, value=value, fee=commission, cash_after=cash,
                               day_amount=q.amount))
            if units == position['units']:
                close_position(sid, position, day, value - commission)
                del positions[sid]
                pending.remove(sid)
            else:
                position['units'] -= units
                position['proceeds'] += value - commission
                orders.append(dict(date=day, security_id=sid, status='sell_pending_capacity'))

        if is_rebalance and day < final_rebalance:
            budget = cash * equity_weight / top_n
            for selected in selections[executions[day]].itertuples():
                sid = selected.security_id
                if sid in positions or sid in terminated:
                    orders.append(dict(date=day, security_id=sid, status='buy_unfilled_locked_or_terminated'))
                    continue
                if not tradeable(sid):
                    orders.append(dict(date=day, security_id=sid, status='buy_unfilled_no_quote'))
                    continue
                q = quotes.loc[sid]
                if not np.isfinite(selected.adv20_amount) or selected.adv20_amount <= 0:
                    raise ValueError('已选证券缺少有效信号日流动性')
                # 订单规模同时受信号日已知流动性和执行日实际成交额约束。
                value = min(budget / (1 + fee), max_participation * selected.adv20_amount,
                            max_participation * q.amount - used_turnover.get(sid, 0.))
                if value <= 0:
                    orders.append(dict(date=day, security_id=sid, status='buy_unfilled_capacity'))
                    continue
                commission = value * fee
                cash -= value + commission
                next_id += 1
                units = value / q.adj_close
                positions[sid] = dict(id=next_id, units=units, original_cost=value + commission,
                                      proceeds=0., entry_date=day)
                trades.append(dict(date=day, security_id=sid, position_id=next_id, side='buy', units=units,
                                   price=q.adj_close, value=value, fee=commission, cash_after=cash,
                                   day_amount=q.amount))
                if value * (1 + fee) < budget - 1e-8:
                    orders.append(dict(date=day, security_id=sid, status='buy_unfilled_capacity'))

        value, missing, reference_only = 0., 0, 0
        for sid, position in positions.items():
            usable = (sid in quotes.index and quotes.loc[sid, 'data_valid'] == True
                      and np.isfinite(quotes.loc[sid, 'adj_close']) and quotes.loc[sid, 'adj_close'] > 0)
            mark = float(quotes.loc[sid, 'adj_close']) if usable else np.nan
            if usable:
                value += position['units'] * mark
                if not tradeable(sid):
                    reference_only += 1
            else:
                missing += 1
                gaps.append(dict(date=day, security_id=sid, reason='missing_held_valuation'))
            holdings.append(dict(date=day, security_id=sid, position_id=position['id'], units=position['units'],
                                 mark=mark, value=position['units'] * mark, sell_pending=sid in pending,
                                 trade_quote=tradeable(sid)))
        claims = sum(claim['amount'] for claim in receivables.values())
        daily.append(dict(date=day, cash=cash, receivables=claims, holdings_value=value if not missing else np.nan,
                          equity=cash + value + claims if not missing else np.nan, positions=len(positions),
                          missing_marks=missing, reference_only_positions=reference_only))
    daily = pd.DataFrame(daily)
    trades = pd.DataFrame(trades, columns=['date', 'security_id', 'position_id', 'side', 'units', 'price',
                                         'value', 'fee', 'cash_after', 'day_amount'])
    completed = pd.DataFrame(completed, columns=['position_id', 'security_id', 'entry_date', 'exit_date',
                                                'cost', 'proceeds', 'net_return'])
    ending = daily.equity.iloc[-1]
    covered = daily.equity.notna().all()
    elapsed = (end - first_entry).days
    summary = dict(status='complete_adjusted_unit_replay' if np.isfinite(ending) else 'incomplete_valuation',
                   training_performed=False, execution_validated=False, initial_cash=initial_cash,
                   start_date=str(first_entry.date()), end_date=str(end.date()), ending_cash=cash,
                   ending_equity=float(ending) if np.isfinite(ending) else None,
                   ending_receivables=float(daily.receivables.iloc[-1]), open_positions=len(positions),
                   pending_sells=len(pending), completed_positions=len(completed),
                   fees=float(trades.fee.sum()), missing_valuation_days=int(daily.equity.isna().sum()),
                   reference_valuation_days=int(daily.reference_only_positions.gt(0).sum()),
                   total_return=float(ending / initial_cash - 1) if np.isfinite(ending) else None,
                   cagr=float((ending / initial_cash) ** (365.25 / elapsed) - 1)
                   if np.isfinite(ending) and elapsed > 0 else None,
                   max_drawdown_daily=float((daily.equity / daily.equity.cummax() - 1).min()) if covered else None,
                   profitable_hit_rate=float(completed.net_return.gt(0).mean()) if len(completed) else None,
                   fee_per_side=fee, max_participation=max_participation)
    return dict(summary=summary, daily=daily, trades=trades, completed=completed,
                orders=pd.DataFrame(orders, columns=['date', 'security_id', 'status']),
                events=pd.DataFrame(events), gaps=pd.DataFrame(gaps, columns=['date', 'security_id', 'reason']),
                holdings=pd.DataFrame(holdings))
