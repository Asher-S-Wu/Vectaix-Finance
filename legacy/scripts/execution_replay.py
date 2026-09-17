#!/usr/bin/env python3
"""Replay frozen historical picks against the downloaded QVeris daily tape."""
import bisect
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from project_paths import qveris_dir

BASE: Path
INITIAL_CASH = 1_000_000.0
PARTICIPATION = 0.01
FEES = (0.0015, 0.003, 0.005)
BENCHMARK = {'a': '000300.SH'}


def read_json(path):
    return json.loads(path.read_text())


def save_json(name, obj):
    (BASE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def save_csv(name, rows):
    if not rows:
        if (BASE / name).exists():
            (BASE / name).unlink()
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (BASE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def positive(value):
    result = number(value)
    return result if result is not None and result > 0 else None


def code_key(code):
    code = str(code).upper()
    if code.endswith('.HK') and code.split('.')[0].isdigit():
        return code.split('.')[0].zfill(5) + '.HK'
    return code


def flatten(data):
    for item in data:
        if isinstance(item, list):
            yield from flatten(item)
        elif isinstance(item, dict) and 'thscode' in item:
            yield item


def load_tape(requests):
    tape = {'raw': {}, 'adjusted': {}, 'benchmark': {}}
    tape['raw_request_observations'] = []
    evidence_file = BASE / 'confirmed_nontrading.json'
    tape['confirmed_nontrading'] = read_json(evidence_file) if evidence_file.exists() else []
    gaps, conflicts = [], []
    source_requests = requests + [{'id': 'benchmark_indices', 'file': 'raw/benchmark_indices.json', 'kind': 'benchmark'}]
    for request in source_requests:
        path = BASE / request['file']
        if not path.exists():
            gaps.append({'type': 'missing_request_file', 'request_id': request['id'], 'file': request['file']})
            continue
        result = read_json(path)['result']
        if result['status_code'] != 200 or not isinstance(result['data'], list):
            gaps.append({'type': 'failed_request', 'request_id': request['id'], 'status_code': result['status_code']})
            continue
        kind = request['kind']
        observed_codes = set()
        for row in flatten(result['data']):
            key = (code_key(row['thscode']), row['time'][:10])
            if positive(row.get('close')):
                observed_codes.add(key[0])
            previous = tape[kind].get(key)
            if previous:
                for field in ('close', 'volume', 'amount'):
                    old, new = number(previous.get(field)), number(row.get(field))
                    if old is not None and new is not None and not math.isclose(old, new, rel_tol=1e-8, abs_tol=1e-7):
                        conflicts.append({'type': 'conflicting_observation', 'code': key[0], 'date': key[1], 'kind': kind, 'field': field, 'previous': old, 'new': new})
            tape[kind][key] = row
        if kind == 'raw':
            tape['raw_request_observations'].extend(
                {'code': code_key(code), 'start_date': request['params']['startdate'],
                 'end_date': request['params']['enddate'], 'has_valid_price': code_key(code) in observed_codes}
                for code in request['symbols'])
    return tape, gaps + conflicts


def status_reason(market, raw, side):
    if raw is None:
        return 'missing_raw_row'
    trading = raw.get('ths_trading_status_stock')
    limit = raw.get('ths_up_and_down_status_stock')
    if market == 'a':
        if trading and ('停牌' in str(trading) or '退市' in str(trading)):
            return 'suspended'
        if trading not in (None, '', '交易', '正常交易', '新股上市'):
            return 'unparsed_trading_status:' + str(trading)
        if limit not in (None, '', '非涨跌停', '涨停', '跌停', '曾涨停', '曾跌停'):
            return 'unparsed_limit_status:' + str(limit)
        if side == 'buy' and limit == '涨停':
            return 'limit_up'
        if side == 'sell' and limit == '跌停':
            return 'limit_down'
    if positive(raw.get('close')) is None:
        return 'missing_raw_close'
    if positive(raw.get('volume')) is None:
        return 'zero_or_missing_volume'
    if positive(raw.get('amount')) is None:
        return 'zero_or_missing_amount'
    return None


def is_suspended(raw):
    return raw is not None and '停牌' in str(raw.get('ths_trading_status_stock', ''))


def confirmed_nontrading(tape, code, day, raw):
    if is_suspended(raw):
        return True
    return any(code_key(item['code']) == code and item['start_date'] <= day <= item['end_date']
               for item in tape['confirmed_nontrading'])


def perf(values, dates, initial=1.0):
    final = values[-1]
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    peak = initial
    drawdowns = []
    for value in values:
        peak = max(peak, value)
        drawdowns.append(value / peak - 1)
    daily_returns = [values[0] / initial - 1] + [b / a - 1 for a, b in zip(values, values[1:])]
    stdev = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
    return {'total_return': final / initial - 1,
            'cagr': (final / initial) ** (1 / years) - 1 if years > 0 else None,
            'max_drawdown': min(drawdowns),
            'annualized_volatility': stdev * math.sqrt(252),
            'sharpe_zero_rf': statistics.mean(daily_returns) / stdev * math.sqrt(252) if stdev else None}


def monthly_perf(rows, field):
    returns = [row[field] for row in rows]
    nav, peak, mdd = 1.0, 1.0, 0.0
    for value in returns:
        nav *= 1 + value
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
    benchmark_complete = all(row['benchmark_return'] is not None for row in rows)
    excess = [row[field] - row['benchmark_return'] for row in rows] if benchmark_complete else []
    return {'months': len(rows), 'arithmetic_annual_return': statistics.mean(returns) * 12,
            'compound_annual_return_month_count': nav ** (12 / len(rows)) - 1,
            'total_return': nav - 1, 'monthly_max_drawdown': mdd,
            'arithmetic_annual_excess_vs_price_index': statistics.mean(excess) * 12 if benchmark_complete else None,
            'monthly_win_vs_price_index': statistics.mean(value > 0 for value in excess) if benchmark_complete else None,
            'benchmark_comparison_complete': benchmark_complete}


def period_groups(targets, calendar):
    by_signal = defaultdict(list)
    for row in targets:
        by_signal[row['signal_date']].append(row)
    signals = sorted(by_signal)
    periods = []
    for index, signal in enumerate(signals):
        rows = sorted(by_signal[signal], key=lambda row: int(row['rank']))
        entry = calendar[bisect.bisect_right(calendar, signal)]
        if 'exit_date' in rows[0] and rows[0]['exit_date']:
            end = rows[0]['exit_date']
        elif index + 1 < len(signals):
            end = calendar[bisect.bisect_right(calendar, signals[index + 1])]
        else:
            end = '2026-09-01'
        if rows[0].get('entry_date') and rows[0]['entry_date'] != entry:
            raise ValueError(f"Entry calendar mismatch: {signal}: {rows[0]['entry_date']} != {entry}")
        periods.append({'signal_date': signal, 'entry_date': entry, 'exit_date': end,
                        'codes': [code_key(row['symbol']) for row in rows], 'rows': rows})
    return periods


def benchmark_returns(market, periods, tape):
    results, gaps = {}, []
    code = BENCHMARK[market]
    for period in periods:
        prices = []
        for day in (period['entry_date'], period['exit_date']):
            row = tape['benchmark'].get((code, day))
            price = positive(row.get('close')) if row else None
            if price is None:
                gaps.append({'type': 'missing_benchmark_close', 'market': market, 'code': code, 'date': day})
            prices.append(price)
        if all(prices):
            results[period['signal_date']] = prices[1] / prices[0] - 1
    return results, gaps


def ideal_replay(market, periods, tape, benchmark):
    months, stock_rows, gaps = [], [], []
    previous = None
    for period in periods:
        code_returns = []
        observable_returns = []
        for row, code in zip(period['rows'], period['codes']):
            entry, end = period['entry_date'], period['exit_date']
            start_row = tape['adjusted'].get((code, entry))
            end_row = tape['adjusted'].get((code, end))
            start_price = positive(start_row.get('close')) if start_row else None
            end_price = positive(end_row.get('close')) if end_row else None
            if start_price is None or end_price is None:
                for day, price in ((entry, start_price), (end, end_price)):
                    if price is None:
                        gaps.append({'type': 'missing_ideal_endpoint', 'market': market, 'code': code, 'date': day})
                continue
            value = end_price / start_price - 1
            code_returns.append(value)
            signal_row = tape['raw'].get((code, period['signal_date']))
            signal_observable = bool(signal_row and positive(signal_row.get('close')) and positive(signal_row.get('volume')))
            observable_returns.append(value if signal_observable else 0.0)
            detail = {'market': market, 'signal_date': period['signal_date'], 'entry_date': entry, 'exit_date': end,
                      'code': code, 'rank': row['rank'], 'entry_adjusted_close': start_price,
                      'exit_adjusted_close': end_price, 'qveris_gross_return': value}
            detail['observable_signal_bar'] = signal_observable
            detail['original_label_start'] = row['original_label_start']
            detail['original_label_end'] = row['original_label_end']
            if row.get('original_return'):
                detail['original_fwd_ret'] = float(row['original_return'])
                detail['return_difference'] = value - detail['original_fwd_ret']
            stock_rows.append(detail)
        cur = set(period['codes'])
        turnover = 1.0 if previous is None else 1 - len(previous & cur) / len(cur)
        previous = cur
        if len(code_returns) == len(cur):
            months.append({'market': market, 'signal_date': period['signal_date'], 'entry_date': period['entry_date'],
                           'exit_date': period['exit_date'], 'stocks': len(cur), 'turnover': turnover,
                           'gross_return': statistics.mean(code_returns), 'net_return': statistics.mean(code_returns) - 0.003 * turnover,
                           'signal_observable_gross_return': statistics.mean(observable_returns),
                           'signal_observable_net_same_cost': statistics.mean(observable_returns) - 0.003 * turnover,
                           'benchmark_return': benchmark.get(period['signal_date'])})
    return months, stock_rows, gaps


def account_replay(market, periods, calendar, tape, fee, benchmark):
    start, end = periods[0]['entry_date'], periods[-1]['exit_date']
    days = [day for day in calendar if start <= day <= end]
    at_entry = {period['entry_date']: period for period in periods}
    price_history = defaultdict(list)
    for (code, day), row in tape['adjusted'].items():
        price = positive(row.get('close'))
        if price:
            price_history[code].append((day, price))
    for history in price_history.values():
        history.sort()
    last_price, last_price_day, last_trade_day = {}, {}, {}
    for code, history in price_history.items():
        older = [item for item in history if item[0] < start]
        if older:
            last_price_day[code], last_price[code] = older[-1]
            last_trade_day[code] = last_price_day[code]
    positions = defaultdict(float)
    targets, unknown_targets = {}, {}
    cash, fees_paid, reserved_cash = INITIAL_CASH, 0.0, 0.0
    daily, trades, blocks, gaps, stale = [], [], [], [], []
    signal_exclusions = []
    before_values = {}
    current_signal = None
    day_indexes = {day: index for index, day in enumerate(calendar)}
    first_bench = tape['benchmark'].get((BENCHMARK[market], start))
    first_bench_close = positive(first_bench.get('close')) if first_bench else None
    for day in days:
        relevant = set(positions)
        if day not in at_entry:
            relevant |= {code for code, target in targets.items() if target > 1e-9} | set(unknown_targets)
        if day in at_entry:
            for code in at_entry[day]['codes']:
                signal_raw = tape['raw'].get((code, at_entry[day]['signal_date']))
                if signal_raw and positive(signal_raw.get('close')) and positive(signal_raw.get('volume')):
                    relevant.add(code)
        for code in sorted(relevant):
            raw = tape['raw'].get((code, day))
            adjusted = tape['adjusted'].get((code, day))
            price = positive(adjusted.get('close')) if adjusted else None
            active = True
            nontrading = confirmed_nontrading(tape, code, day, raw)
            if raw is None and active and not nontrading:
                gaps.append({'type': 'missing_daily_raw', 'market': market, 'code': code, 'date': day})
            if price:
                last_price[code], last_price_day[code] = price, day
                if raw and positive(raw.get('volume')) and not nontrading:
                    last_trade_day[code] = day
            elif active and not nontrading:
                gaps.append({'type': 'missing_daily_adjusted_close', 'market': market, 'code': code, 'date': day})
            if market == 'a' and raw:
                reason = status_reason(market, raw, 'buy')
                if reason and reason.startswith('unparsed_'):
                    gaps.append({'type': reason, 'market': market, 'code': code, 'date': day})
        invested = sum(units * last_price[code] for code, units in positions.items() if units > 1e-10)
        before_nav = cash + invested
        before_values[day] = before_nav
        if day in at_entry:
            period = at_entry[day]
            current_signal = period['signal_date']
            selected = period['codes']
            current_values = {code: units * last_price[code] for code, units in positions.items() if units > 1e-10}
            n = len(selected)
            eligible = []
            for code in selected:
                signal_raw = tape['raw'].get((code, period['signal_date']))
                observable = signal_raw and positive(signal_raw.get('close')) and positive(signal_raw.get('volume'))
                if observable:
                    eligible.append(code)
                else:
                    request_has_any_price = any(item['code'] == code and item['start_date'] <= current_signal <= item['end_date']
                                                and item['has_valid_price'] for item in tape['raw_request_observations'])
                    if not request_has_any_price and not confirmed_nontrading(tape, code, current_signal, signal_raw):
                        gaps.append({'type': 'entire_symbol_history_unavailable_at_signal', 'market': market,
                                     'code': code, 'date': current_signal})
                    signal_exclusions.append({'market': market, 'fee_per_side': fee, 'signal_date': current_signal,
                                              'code': code, 'reason': 'no_observable_signal_bar',
                                              'source_bar_status': 'omitted_row_in_completed_request' if signal_raw is None else 'returned_row_without_positive_close_and_volume',
                                              'request_contains_other_observed_prices': request_has_any_price,
                                              'reserved_target_weight': 1 / n})
            low, high = 0.0, before_nav
            for _ in range(60):
                after = (low + high) / 2
                estimated_trade = sum(abs(after / n - current_values.get(code, 0)) for code in eligible)
                estimated_trade += sum(value for code, value in current_values.items() if code not in eligible)
                if after + fee * estimated_trade > before_nav:
                    high = after
                else:
                    low = after
            value_per_target = (low + high) / 2 / n
            reserved_cash = value_per_target * (n - len(eligible))
            targets = {code: 0.0 for code in positions}
            unknown_targets = {}
            for code in eligible:
                if code in last_price:
                    targets[code] = value_per_target / last_price[code]
                else:
                    unknown_targets[code] = value_per_target
        for code, allocation in list(unknown_targets.items()):
            adjusted = tape['adjusted'].get((code, day))
            if adjusted and positive(adjusted.get('close')):
                targets[code] = allocation / float(adjusted['close'])
                del unknown_targets[code]
        candidates = {'sell': [], 'buy': []}
        for code, target in sorted(targets.items()):
            units = positions.get(code, 0)
            delta = target - units
            if abs(delta) < 1e-9:
                continue
            side = 'buy' if delta > 0 else 'sell'
            raw = tape['raw'].get((code, day))
            adjusted = tape['adjusted'].get((code, day))
            price = positive(adjusted.get('close')) if adjusted else None
            reason = status_reason(market, raw, side)
            if price is None:
                reason = 'missing_adjusted_close'
            if confirmed_nontrading(tape, code, day, raw):
                reason = 'confirmed_nontrading'
            if reason:
                blocks.append({'market': market, 'fee_per_side': fee, 'signal_date': current_signal, 'date': day,
                               'code': code, 'side': side, 'reason': reason, 'remaining_units': abs(delta)})
                continue
            requested = abs(delta) * price
            capacity = float(raw['amount']) * PARTICIPATION
            gross = min(requested, capacity)
            if requested > capacity + 1e-8:
                blocks.append({'market': market, 'fee_per_side': fee, 'signal_date': current_signal, 'date': day,
                               'code': code, 'side': side, 'reason': 'daily_amount_capacity',
                               'requested_notional': requested, 'capacity_notional': capacity})
            candidates[side].append((code, price, gross, capacity, float(raw['close'])))
        today_gross = today_fee = 0.0
        for side in ('sell', 'buy'):
            gross_total = sum(item[2] for item in candidates[side])
            scale = min(1.0, max(0.0, cash - reserved_cash) / (gross_total * (1 + fee))) if side == 'buy' and gross_total else 1.0
            for code, price, requested_gross, capacity, raw_close in candidates[side]:
                gross = requested_gross * scale
                if side == 'buy' and requested_gross - gross > 1.0:
                    blocks.append({'market': market, 'fee_per_side': fee, 'signal_date': current_signal, 'date': day,
                                   'code': code, 'side': side, 'reason': 'insufficient_unreserved_cash',
                                   'requested_notional': requested_gross, 'funded_notional': gross})
                if gross < 1e-8:
                    continue
                units = gross / price
                cost = gross * fee
                if side == 'sell':
                    positions[code] -= units
                    cash += gross - cost
                else:
                    positions[code] += units
                    cash -= gross + cost
                if abs(positions[code]) < 1e-8:
                    del positions[code]
                trades.append({'market': market, 'fee_per_side': fee, 'signal_date': current_signal, 'date': day,
                               'code': code, 'side': side, 'adjusted_units': units, 'adjusted_close': price,
                               'raw_close': raw_close, 'gross_notional': gross, 'fee_paid': cost,
                               'cash_after_trade': cash, 'daily_capacity_notional': capacity})
                today_gross += gross
                today_fee += cost
        fees_paid += today_fee
        invested = sum(units * last_price[code] for code, units in positions.items())
        nav = cash + invested
        if not math.isclose(nav, before_nav - today_fee, rel_tol=1e-9, abs_tol=1e-5):
            raise ArithmeticError(f'Cash/holdings accounting mismatch: {market} {day}')
        if cash < -1e-6 or any(units < -1e-8 for units in positions.values()):
            raise ArithmeticError(f'Negative cash/holdings: {market} {day}')
        stale_value, maximum_age = 0.0, 0
        for code, units in positions.items():
            quote_day = last_price_day[code]
            age = day_indexes[day] - bisect.bisect_right(calendar, quote_day) + 1
            trade_day = last_trade_day.get(code)
            trade_age = day_indexes[day] - bisect.bisect_right(calendar, trade_day) + 1 if trade_day else None
            if age > 0 or (trade_age is not None and trade_age > 0):
                value = units * last_price[code]
                stale_value += value
                maximum_age = max(maximum_age, age, trade_age or 0)
                stale.append({'market': market, 'fee_per_side': fee, 'date': day, 'code': code,
                              'market_value': value, 'last_quote_date': quote_day, 'missing_quote_trading_days': age,
                              'last_positive_volume_quote_date': trade_day, 'nontrading_quote_days': trade_age})
        pending = sum(abs(target - positions.get(code, 0)) * last_price.get(code, 0) > 1.0 for code, target in targets.items()) + len(unknown_targets)
        bench = tape['benchmark'].get((BENCHMARK[market], day))
        bench_close = positive(bench.get('close')) if bench else None
        if bench_close is None:
            gaps.append({'type': 'missing_daily_benchmark', 'market': market, 'code': BENCHMARK[market], 'date': day})
        daily.append({'market': market, 'fee_per_side': fee, 'date': day, 'signal_date': current_signal,
                      'equity_before_trades': before_nav, 'cash': cash, 'holdings_value': invested,
                      'equity': nav, 'net_value': nav / INITIAL_CASH, 'cash_weight': cash / nav,
                      'signal_exclusion_reserved_cash': reserved_cash,
                      'trade_gross': today_gross, 'fees_paid': today_fee, 'held_stocks': len(positions),
                      'pending_orders': pending, 'stale_market_value': stale_value, 'max_stale_trading_days': maximum_age,
                      'benchmark_close': bench_close, 'benchmark_nav': bench_close / first_bench_close if bench_close and first_bench_close else None})
    months = []
    for period in periods:
        months.append({'market': market, 'fee_per_side': fee, 'signal_date': period['signal_date'],
                           'entry_date': period['entry_date'], 'exit_date': period['exit_date'],
                           'net_return': (daily[-1]['equity'] if period['exit_date'] == end else before_values[period['exit_date']]) / before_values[period['entry_date']] - 1,
                           'benchmark_return': benchmark.get(period['signal_date'])})
    metrics = perf([row['equity'] for row in daily], days, INITIAL_CASH)
    monthly_end_equity = INITIAL_CASH * math.prod(1 + row['net_return'] for row in months)
    if not math.isclose(monthly_end_equity, daily[-1]['equity'], rel_tol=1e-9, abs_tol=1e-5):
        raise ArithmeticError(f'Monthly/daily equity reconciliation failed: {market} {fee}')
    metrics.update({'initial_cash': INITIAL_CASH, 'final_equity': daily[-1]['equity'], 'total_fees_paid': fees_paid,
                    'mean_cash_weight': statistics.mean(row['cash_weight'] for row in daily),
                    'trade_count': len(trades), 'total_gross_traded': sum(row['gross_notional'] for row in trades),
                    'blocked_order_days_by_reason': dict(sorted({reason: sum(row['reason'] == reason for row in blocks) for reason in set(row['reason'] for row in blocks)}.items())),
                    'days_with_stale_valuation': sum(row['stale_market_value'] > 0 for row in daily),
                    'max_stale_trading_days': max(row['max_stale_trading_days'] for row in daily),
                    'final_pending_orders': daily[-1]['pending_orders'], 'final_cash_weight': daily[-1]['cash_weight']})
    metrics['accounting_audit'] = {'daily_cash_holdings_fee_identity_passed': True,
                                   'negative_cash_or_holdings_observed': False,
                                   'monthly_compound_matches_daily_final_equity': True,
                                   'daily_records': len(daily)}
    metrics['signal_excluded_stock_months'] = len(signal_exclusions)
    if months:
        metrics['monthly'] = monthly_perf(months, 'net_return')
        metrics['latest_12_months'] = monthly_perf(months[-12:], 'net_return')
    if all(row['benchmark_nav'] is not None for row in daily):
        metrics['benchmark_price_index'] = perf([row['benchmark_nav'] for row in daily], days)
    return metrics, daily, months, trades, blocks, gaps, stale, signal_exclusions


def main():
    global BASE
    parser = argparse.ArgumentParser(description='使用已下载的 QVeris 行情复算指定市场的历史账户。')
    parser.add_argument('--market', choices=('a',), required=True, help='a 为 A 股')
    parser.add_argument('--directory', type=Path,
                        help='指定行情及回放结果目录；默认使用 backtests/a/qveris')
    args = parser.parse_args()
    BASE = (args.directory if args.directory is not None else qveris_dir(args.market)).resolve()
    markets = ('a',)
    requests = read_json(BASE / 'requests.json')
    requests = [request for request in requests if request['market'] == args.market]
    tape, load_gaps = load_tape(requests)
    if load_gaps:
        save_json('coverage_needed.json', load_gaps)
        save_json('replay_summary.json', {'status': 'incomplete_download', 'gap_count': len(load_gaps), 'performance_withheld': True})
        print(json.dumps({'status': 'incomplete_download', 'gap_count': len(load_gaps)}))
        return
    with (BASE / 'targets.csv').open() as stream:
        targets = list(csv.DictReader(stream))
    gaps = []
    all_daily, all_months, all_trades, all_blocks, all_stale = [], [], [], [], []
    all_signal_exclusions = []
    ideal_months, ideal_stocks = [], []
    summary = {'status': 'complete', 'source': 'QVeris downloaded historical daily quotes; monthly target selections supplied in targets.csv',
               'assumptions': {'initial_cash_each_market': INITIAL_CASH, 'currency': {'a': 'CNY'},
                               'execution': 'market-calendar T+1 closing price; fixed target units until next monthly signal',
                               'portfolio': 'fractional adjusted units, dividend-reinvestment return convention; not actual-share tax-account bookkeeping',
                               'participation_cap': PARTICIPATION, 'fees_per_side': list(FEES),
                               'rebalance': 'sell first, then proportionally allocate available cash to executable buys',
                               'blocked_orders': 'continue original unfilled units on following trading days until next signal',
                               'end_date': '2026-09-01', 'terminal_liquidation': False,
                               'round_lots': False, 'closing_auction_capacity': 'not separately modeled; 1% of full-day amount is only a proxy',
                               'benchmark': 'unadjusted price indices; stock returns include QVeris adjustment',
                               'selection_bias': 'target selections retain the source model universe and historical model-selection biases'},
               'markets': {}}
    for market in markets:
        calendar = read_json(BASE / 'raw' / 'calendar_a.json')['result']['data']['time']
        periods = period_groups([row for row in targets if row['market'] == market], calendar)
        benchmark, bench_gaps = benchmark_returns(market, periods, tape)
        ideal, stock_rows, ideal_gaps = ideal_replay(market, periods, tape, benchmark)
        ideal_months.extend(ideal)
        ideal_stocks.extend(stock_rows)
        market_gaps = list(bench_gaps)
        market_result = {'start_date': periods[0]['entry_date'], 'end_date': periods[-1]['exit_date'], 'months': len(periods), 'scenarios': {}}
        if len(ideal) == len(periods):
            market_result['ideal_equal_weight'] = monthly_perf(ideal, 'net_return')
            market_result['ideal_equal_weight']['latest_12_months'] = monthly_perf(ideal[-12:], 'net_return')
            market_result['ideal_signal_observable_same_cost'] = monthly_perf(ideal, 'signal_observable_net_same_cost')
        else:
            market_result['ideal_equal_weight'] = {'status': 'missing_endpoint', 'performance_withheld': True}
        for fee in FEES:
            metrics, daily, months, trades, blocks, scenario_gaps, stale, exclusions = account_replay(market, periods, calendar, tape, fee, benchmark)
            account_gaps = [gap for gap in scenario_gaps if 'benchmark' not in gap['type']]
            if account_gaps:
                market_result['scenarios'][str(fee)] = {'status': 'data_gaps', 'performance_withheld': True,
                                                      'account_gap_count': len(account_gaps)}
            else:
                metrics['status'] = 'complete_account'
                market_result['scenarios'][str(fee)] = metrics
            for dataset in (daily, months, trades, blocks, stale, exclusions):
                for row in dataset:
                    row['account_evidence_status'] = 'data_gaps' if account_gaps else 'complete_account'
            all_daily.extend(daily)
            all_months.extend(months)
            all_trades.extend(trades)
            all_blocks.extend(blocks)
            all_stale.extend(stale)
            all_signal_exclusions.extend(exclusions)
            market_gaps.extend(scenario_gaps)
        if market_gaps or ideal_gaps:
            market_result['status'] = 'some_evidence_incomplete'
            summary['status'] = 'data_gaps'
        summary['markets'][market] = market_result
        gaps.extend(market_gaps + ideal_gaps)
    unique = {json.dumps(gap, sort_keys=True): gap for gap in gaps}
    gaps = sorted(unique.values(), key=lambda row: (row.get('market', ''), row.get('code', ''), row.get('date', ''), row['type']))
    summary['gap_count'] = len(gaps)
    execution_gaps = [gap for gap in gaps if gap['type'] != 'missing_ideal_endpoint' and 'benchmark' not in gap['type']]
    summary['execution_gap_count'] = len(execution_gaps)
    summary['ideal_endpoint_gap_count'] = sum(gap['type'] == 'missing_ideal_endpoint' for gap in gaps)
    summary['execution_status'] = 'data_gaps' if execution_gaps else 'complete'
    if not execution_gaps and gaps:
        summary['status'] = 'complete_execution_incomplete_ideal_comparison'
    save_json('coverage_needed.json', gaps)
    save_json('replay_summary.json', summary)
    save_csv('daily_accounts.csv', all_daily)
    save_csv('monthly_accounts.csv', all_months)
    save_csv('trades.csv', all_trades)
    save_csv('execution_blocks.csv', all_blocks)
    save_csv('stale_valuations.csv', all_stale)
    save_csv('signal_exclusions.csv', all_signal_exclusions)
    save_csv('ideal_monthly.csv', ideal_months)
    save_csv('ideal_stock_returns.csv', ideal_stocks)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
