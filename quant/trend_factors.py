from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FACTOR_NAMES, build_factors


TREND_FACTOR_NAMES = {
    **FACTOR_NAMES,
    'momentum_3': '季度动量',
    'relative_momentum_1': '月度相对大盘强弱',
    'relative_momentum_3': '季度相对大盘强弱',
    'relative_momentum_6': '半年相对大盘强弱',
    'short_trend': '月度趋势',
    'trend_acceleration': '短长趋势差',
    'distance_high': '距一年高点',
    'drawdown_3': '季度回撤',
    'rsi_14': '两周涨跌力量',
    'volatility_ratio': '短长波动比',
    'downside_share': '下跌波动占比',
    'residual_volatility': '扣除大盘后的波动',
    'range_volatility': '日内高低价波动',
    'volume_shock': '近期成交量变化',
    'turnover_change': '成交额趋势',
    'signed_turnover': '涨跌成交额倾向',
    'overnight_gap': '隔夜价格变化',
    'intraday_return': '日内价格变化',
    'market_momentum_1': '大盘月度动量',
    'market_momentum_3': '大盘季度动量',
    'market_trend': '大盘半年趋势',
    'market_volatility': '大盘波动',
    'bond_momentum_1': '美债基金月度趋势',
    'bond_momentum_3': '美债基金季度趋势',
    'bond_trend': '美债基金半年趋势',
    'bond_volatility': '美债基金波动',
}
TREND_FACTOR_KEYS = tuple(TREND_FACTOR_NAMES)


def build_trend_factors(prices: pd.DataFrame, symbols: list[str], benchmark: str, macro_prices: pd.DataFrame):
    base, calendar = build_factors(prices, symbols, benchmark)
    market = prices.loc[prices.symbol == benchmark].set_index('date').reindex(calendar)
    market_close = market.research_close.where(market.volume > 0)
    market_return = market_close.pct_change(fill_method=None)
    frames = []
    for symbol in symbols:
        rows = prices.loc[prices.symbol == symbol].set_index('date').reindex(calendar)
        close = rows.research_close.where(rows.volume > 0)
        returns = close.pct_change(fill_method=None)
        turnover = rows.turnover.where(rows.volume > 0)
        volume = rows.volume.where(rows.volume > 0)
        frame = pd.DataFrame(index=calendar)
        frame['momentum_3'] = close / close.shift(63) - 1
        for period, sessions in ((1, 21), (3, 63), (6, 126)):
            frame[f'relative_momentum_{period}'] = (
                close / close.shift(sessions) - market_close / market_close.shift(sessions)
            )
        frame['short_trend'] = close / close.rolling(21).mean() - 1
        frame['trend_acceleration'] = close.rolling(21).mean() / close.rolling(126).mean() - 1
        frame['distance_high'] = close / close.rolling(252).max() - 1
        frame['drawdown_3'] = close / close.rolling(63).max() - 1
        gain = returns.clip(lower=0).rolling(14).mean()
        loss = -returns.clip(upper=0).rolling(14).mean()
        frame['rsi_14'] = gain / (gain + loss)
        frame['volatility_ratio'] = returns.rolling(21).std() / returns.rolling(126).std()
        frame['downside_share'] = returns.clip(upper=0).pow(2).rolling(60).sum() / returns.pow(2).rolling(60).sum()
        covariance = returns.rolling(60).cov(market_return)
        residual_variance = returns.rolling(60).var() - covariance.pow(2) / market_return.rolling(60).var()
        frame['residual_volatility'] = np.sqrt(residual_variance.clip(lower=0))
        frame['range_volatility'] = np.sqrt(np.log(rows.high / rows.low).pow(2).rolling(21).mean() / (4 * np.log(2)))
        frame['volume_shock'] = volume.rolling(5).mean() / volume.rolling(60).mean() - 1
        frame['turnover_change'] = turnover.rolling(21).mean() / turnover.rolling(126).mean() - 1
        frame['signed_turnover'] = (np.sign(returns) * turnover).rolling(21).sum() / turnover.rolling(21).sum()
        frame['overnight_gap'] = (rows.research_open / close.shift(1) - 1).rolling(5).mean()
        frame['intraday_return'] = (close / rows.research_open - 1).rolling(5).mean()
        frame['market_momentum_1'] = market_close / market_close.shift(21) - 1
        frame['market_momentum_3'] = market_close / market_close.shift(63) - 1
        frame['market_trend'] = market_close / market_close.rolling(126).mean() - 1
        frame['market_volatility'] = market_return.rolling(60).std()
        frame['symbol'] = symbol
        frames.append(frame.rename_axis('date').reset_index())
    result = base.merge(pd.concat(frames, ignore_index=True), on=['date', 'symbol'], validate='one_to_one')
    bond = macro_prices.set_index('date')
    bond_close = bond.adjClose
    macro = pd.DataFrame({
        'bond_momentum_1': bond_close / bond_close.shift(21) - 1,
        'bond_momentum_3': bond_close / bond_close.shift(63) - 1,
        'bond_trend': bond_close / bond_close.rolling(126).mean() - 1,
        'bond_volatility': bond_close.pct_change(fill_method=None).rolling(21).std(),
    }).rename_axis('sourceDate').reset_index()
    # Match a real US close that has already happened in Hong Kong time.
    # The complete XNYS series is verified before this join; no US observation is filled.
    aligned = pd.merge_asof(pd.DataFrame({'date': calendar}), macro,
                            left_on='date', right_on='sourceDate', direction='backward', allow_exact_matches=False)
    aligned['sourceAgeDays'] = (aligned.date - aligned.sourceDate).dt.days
    result = result.merge(aligned, on='date', validate='many_to_one')
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=list(TREND_FACTOR_KEYS))
    if not (result.sourceDate < result.date).all():
        raise ValueError('美国行情尚未收盘，不能用于该香港预测日。')
    if set(result.symbol) != set(symbols):
        raise ValueError('扩展因子无法覆盖四只股票，停止训练。')
    return result.sort_values(['symbol', 'date']).reset_index(drop=True), calendar
