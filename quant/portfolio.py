from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd


INITIAL_CAPITAL = 1_000_000.0
SLIPPAGE = 0.001


def transaction_fee(notional: float, date: pd.Timestamp, stamp_duty_exempt_from: pd.Timestamp | None) -> float:
    if notional <= 0:
        return 0.0
    stamp_rate = 0.0013 if pd.Timestamp("2021-08-01") <= date < pd.Timestamp("2023-11-17") else 0.001
    exempt = stamp_duty_exempt_from is not None and date >= stamp_duty_exempt_from
    stamp = 0.0 if exempt else float(ceil(notional * stamp_rate))
    broker = max(3.0, notional * 0.0003)
    sfc = notional * 0.000027
    afrc = notional * 0.0000015 if date >= pd.Timestamp("2022-01-01") else 0.0
    exchange = notional * (0.0000565 if date >= pd.Timestamp("2023-01-01") else 0.00005)
    trading_tariff = 0.0 if date >= pd.Timestamp("2023-01-01") else 0.5
    settlement = notional * 0.000042 if date >= pd.Timestamp("2025-06-30") else min(100.0, max(2.0, notional * 0.00002))
    return stamp + broker + sfc + afrc + exchange + trading_tariff + settlement


@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: list[str]
    opens: np.ndarray
    closes: np.ndarray
    raw_closes: np.ndarray
    tradable: np.ndarray
    benchmark: str
    corporate_actions: list[dict]
    stamp_exemptions: list[pd.Timestamp | None]

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, calendar: pd.DatetimeIndex, benchmark: str, corporate_actions: list[dict], stamp_duty_exempt_from: dict[str, str | None]) -> "Market":
        symbols = sorted(prices.symbol.unique())
        if set(symbols) != set(stamp_duty_exempt_from):
            raise ValueError("印花税豁免元数据必须覆盖全部证券。")
        exemptions = [pd.Timestamp(stamp_duty_exempt_from[symbol]) if stamp_duty_exempt_from[symbol] is not None else None for symbol in symbols]
        def matrix(column: str) -> np.ndarray:
            return prices.pivot(index="date", columns="symbol", values=column).reindex(
                index=calendar, columns=symbols
            ).to_numpy(dtype=float)
        opens, closes, volumes = matrix("research_open"), matrix("research_close"), matrix("volume")
        tradable = (volumes > 0) & np.isfinite(opens) & np.isfinite(closes)
        return cls(calendar, symbols, opens, closes, matrix("close"), tradable, benchmark, corporate_actions, exemptions)


@dataclass
class Simulation:
    equity: pd.Series
    daily_accounting: pd.DataFrame
    trades: list[dict]
    holdings: list[dict]
    stale_marks: int
    blocked_orders: list[dict]
    turnover: float
    total_fees: float
    slippage_cost: float
    max_identity_residual: float
    corporate_events: list[dict]


def simulate(
    market: Market,
    signals: dict[pd.Timestamp, dict[str, float]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    slippage: float = SLIPPAGE,
    record_details: bool = False,
) -> Simulation:
    """Trade next opens, account in dividend-adjusted synthetic units and cash."""
    if start not in market.dates or end not in market.dates or start >= end:
        raise ValueError("回测起止日期必须是有效且不同的 XHKG 正式交易日。")
    lookup = {symbol: i for i, symbol in enumerate(market.symbols)}
    first = int(market.dates.get_loc(start))
    last = int(market.dates.get_loc(end))
    units = np.zeros(len(market.symbols), dtype=float)
    marks = np.full(len(market.symbols), np.nan)
    raw_marks = np.full(len(market.symbols), np.nan)
    cash = INITIAL_CAPITAL
    events: dict[int, tuple[pd.Timestamp, dict[str, float]]] = {}
    for signal, targets in signals.items():
        if signal not in market.dates:
            raise ValueError(f"信号 {signal} 不是交易日。")
        trade_index = int(market.dates.get_loc(signal)) + 1
        if first < trade_index <= last:
            if any(weight < 0 for weight in targets.values()) or sum(targets.values()) > 1 + 1e-10:
                raise ValueError("目标权重包含负数或超过本金。")
            events[trade_index] = (signal, targets)
    rows: list[dict] = []
    trades: list[dict] = []
    holdings: list[dict] = []
    blocked: list[dict] = []
    stale_marks = 0
    traded_value = 0.0
    total_fees = 0.0
    slippage_cost = 0.0
    max_identity_residual = 0.0
    receivables: dict[str, float] = {}
    corporate_events: list[dict] = []
    for day in range(first, last + 1):
        date = market.dates[day]
        available = market.tradable[day]
        for action in market.corporate_actions:
            symbol = action["symbol"]
            index = lookup[symbol]
            if date == pd.Timestamp(action["effectiveDate"]) and units[index] > 1e-10:
                quantity = float(units[index])
                amount = quantity * action["adjustedCashPerUnit"]
                receivables[symbol] = amount
                units[index] = 0.0
                corporate_events.append({
                    "date": _date_string(date), "symbol": symbol,
                    "event": "cash_delisting_receivable", "amount": amount,
                    "adjustedUnits": quantity, "cashPerShare": action["cashPerShare"],
                })
            if date == pd.Timestamp(action["paymentDate"]) and symbol in receivables:
                amount = receivables.pop(symbol)
                cash += amount
                corporate_events.append({
                    "date": _date_string(date), "symbol": symbol,
                    "event": "cash_delisting_payment", "amount": amount,
                    "cashAfter": cash,
                })
        if day in events:
            signal, targets = events[day]
            selected = np.zeros(len(units), dtype=float)
            for symbol, weight in targets.items():
                selected[lookup[symbol]] = weight
            held = units > 1e-10
            open_marks = marks.copy()
            open_marks[available] = market.opens[day, available]
            if np.isnan(open_marks[held]).any():
                raise ValueError(f"{date.date()} 持仓缺少可核算价格。")
            open_equity = cash + sum(receivables.values()) + float(np.dot(units[held], open_marks[held]))
            desired = np.zeros(len(units), dtype=float)
            desired[available] = open_equity * selected[available] / market.opens[day, available]
            for index in np.where(~available & ((units > 1e-10) | (selected > 0)))[0]:
                blocked.append({
                    "date": date.date().isoformat(),
                    "signalDate": signal.date().isoformat(),
                    "symbol": market.symbols[index],
                    "reason": "当日无有效成交，订单未执行，原有份额保留至下一次月度调仓。",
                })

            def execute(index: int, quantity: float, side: str) -> None:
                nonlocal cash, traded_value, total_fees, slippage_cost
                if quantity <= 1e-10:
                    return
                reference = float(market.opens[day, index])
                execution = reference * (1 + slippage if side == "buy" else 1 - slippage)
                notional = quantity * execution
                fee = transaction_fee(notional, date, market.stamp_exemptions[index])
                if side == "buy":
                    cash -= notional + fee
                    units[index] += quantity
                else:
                    cash += notional - fee
                    units[index] -= quantity
                    if abs(units[index]) < 1e-10:
                        units[index] = 0.0
                traded_value += notional
                total_fees += fee
                slippage_cost += quantity * reference * slippage
                if record_details:
                    trades.append({
                        "date": date.date().isoformat(),
                        "signalDate": signal.date().isoformat(),
                        "symbol": market.symbols[index],
                        "side": side,
                        "adjustedUnits": quantity,
                        "referencePrice": reference,
                        "executionPrice": execution,
                        "notional": notional,
                        "fees": fee,
                        "slippageCost": quantity * reference * slippage,
                        "cashAfter": cash,
                    })

            sells = np.where(available & (units > desired + 1e-10))[0]
            sells = sorted(sells, key=lambda index: -(units[index] - desired[index]) * market.opens[day, index])
            for index in sells:
                execute(int(index), float(units[index] - desired[index]), "sell")
            buy_indices = np.where(available & (desired > units + 1e-10))[0]
            requested = desired[buy_indices] - units[buy_indices]

            def buy_cost(scale: float) -> float:
                total = 0.0
                for index, quantity in zip(buy_indices, requested * scale):
                    if quantity <= 1e-10:
                        continue
                    notional = quantity * market.opens[day, index] * (1 + slippage)
                    total += notional + transaction_fee(notional, date, market.stamp_exemptions[index])
                return float(total)

            if cash < -1e-6:
                raise ValueError(f"{date.date()} 卖出费用造成现金不足；没有启用借款。")
            scale = 1.0
            if buy_cost(scale) > cash:
                lower, upper = 0.0, 1.0
                for _ in range(48):
                    midpoint = (lower + upper) / 2
                    if buy_cost(midpoint) <= cash:
                        lower = midpoint
                    else:
                        upper = midpoint
                scale = lower
            for index, quantity in zip(buy_indices, requested * scale):
                execute(int(index), float(quantity), "buy")
            if cash < -1e-6 or (units < -1e-10).any():
                raise ValueError(f"{date.date()} 出现负现金或负持仓，会计检查失败。")
        marks[available] = market.closes[day, available]
        raw_marks[available] = market.raw_closes[day, available]
        held = units > 1e-10
        if np.isnan(marks[held]).any():
            raise ValueError(f"{date.date()} 持仓缺少最近实际成交收盘价。")
        stale = int((held & ~available).sum())
        stale_marks += stale
        asset_value = float(np.dot(units[held], marks[held]))
        receivable_value = float(sum(receivables.values()))
        total_equity = cash + asset_value + receivable_value
        identity_residual = abs(total_equity - (cash + receivable_value + sum(float(units[i] * marks[i]) for i in np.where(held)[0])))
        max_identity_residual = max(max_identity_residual, identity_residual)
        if not np.isfinite(total_equity) or total_equity <= 0 or identity_residual > 1e-5:
            raise ValueError(f"{date.date()} 净值会计恒等式失败。")
        rows.append({
            "date": date,
            "cash": cash,
            "assetValue": asset_value,
            "receivables": receivable_value,
            "equity": total_equity,
            "stalePositions": stale,
            "identityResidual": identity_residual,
        })
        if record_details and (day == last or (day + 1 <= last and market.dates[day + 1].month != date.month)):
            for index in np.where(held)[0]:
                holdings.append({
                    "date": date.date().isoformat(),
                    "symbol": market.symbols[index],
                    "adjustedUnits": float(units[index]),
                    "price": float(marks[index]),
                    "rawPrice": float(raw_marks[index]),
                    "value": float(units[index] * marks[index]),
                    "weight": float(units[index] * marks[index] / total_equity),
                    "stale": bool(not available[index]),
                })
    accounting = pd.DataFrame(rows).set_index("date")
    return Simulation(
        equity=accounting.equity,
        daily_accounting=accounting,
        trades=trades,
        holdings=holdings,
        stale_marks=stale_marks,
        blocked_orders=blocked,
        turnover=traded_value / INITIAL_CAPITAL,
        total_fees=total_fees,
        slippage_cost=slippage_cost,
        max_identity_residual=max_identity_residual,
        corporate_events=corporate_events,
    )


def _date_string(date: pd.Timestamp) -> str:
    return date.date().isoformat()
