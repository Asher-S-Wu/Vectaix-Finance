from __future__ import annotations

import numpy as np
import pandas as pd

from .data import research_calendar


FACTOR_NAMES = {
    'momentum_12_1': '12—1 月动量',
    'momentum_6_1': '6—1 月动量',
    'reversal_1': '1 月反转',
    'low_volatility': '60 日低波动',
    'low_downside': '60 日低下行波动',
    'trend': '半年趋势',
    'liquidity': '60 日成交额',
    'low_beta': '60 日低市场敏感度',
}
FACTOR_KEYS = tuple(FACTOR_NAMES)
LABEL_HORIZON = 21


def build_factors(prices: pd.DataFrame, stock_symbols: list[str], benchmark: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Daily time-series features; all windows end at the observation close."""
    calendar = research_calendar()
    market = prices.loc[prices.symbol == benchmark].set_index('date').reindex(calendar)
    market_returns = market.research_close.pct_change(fill_method=None)
    frames = []
    for symbol in stock_symbols:
        rows = prices.loc[prices.symbol == symbol].set_index('date').reindex(calendar)
        close = rows.research_close.where(rows.volume > 0)
        returns = close.pct_change(fill_method=None)
        factors = pd.DataFrame(index=calendar)
        factors['momentum_12_1'] = close.shift(21) / close.shift(252) - 1
        factors['momentum_6_1'] = close.shift(21) / close.shift(126) - 1
        factors['reversal_1'] = -(close / close.shift(21) - 1)
        factors['low_volatility'] = -returns.rolling(60).std()
        factors['low_downside'] = -np.sqrt(returns.clip(upper=0).pow(2).rolling(60).mean())
        factors['trend'] = close / close.rolling(126).mean() - 1
        factors['liquidity'] = np.log(rows.turnover.where(rows.volume > 0).rolling(60).mean())
        factors['low_beta'] = -returns.rolling(60).cov(market_returns) / market_returns.rolling(60).var()
        factors['annualVolatility'] = -factors.low_volatility * np.sqrt(252)
        eligible = (rows.close >= 1) & (rows.volume > 0) & (rows.turnover.rolling(60).mean() >= 10_000_000)
        factors = factors.loc[eligible].replace([np.inf, -np.inf], np.nan).dropna()
        factors['symbol'] = symbol
        frames.append(factors.rename_axis('date').reset_index())
    result = pd.concat(frames, ignore_index=True)
    if set(result.symbol) != set(stock_symbols):
        raise ValueError('有指定股票无法形成完整因子，停止本轮研究。')
    return result.sort_values(['symbol', 'date']).reset_index(drop=True), calendar


def build_labels(factors: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Next-open to 21 sessions later open; no filling missing endpoints."""
    frames = []
    for symbol in factors.symbol.unique():
        rows = prices.loc[prices.symbol == symbol].set_index('date').reindex(calendar)
        opens = rows.research_open.where(rows.volume > 0)
        dates = pd.Series(calendar, index=calendar)
        labels = pd.DataFrame({
            'label_start': dates.shift(-1),
            'label_end': dates.shift(-LABEL_HORIZON - 1),
            'target': opens.shift(-LABEL_HORIZON - 1) / opens.shift(-1) - 1,
        }).dropna()
        labels['symbol'] = symbol
        frames.append(labels.rename_axis('date').reset_index())
    return factors.merge(pd.concat(frames, ignore_index=True), on=['date', 'symbol'], how='inner', validate='one_to_one')
