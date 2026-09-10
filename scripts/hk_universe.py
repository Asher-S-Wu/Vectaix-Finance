"""按历史季度名单、成交额和行情年限筛选港股候选池。"""
import numpy as np
import pandas as pd

from project_paths import data_dir

TARGET_SIZE = 500
MIN_DAILY_AMOUNT_HKD = 10_000_000
MIN_HISTORY_YEARS = 2
MIN_HISTORY_SESSIONS = 480


def load_memberships():
    return pd.read_csv(data_dir('hk2') / 'reference/hsci_memberships.csv',
                       parse_dates=['as_of'], dtype={'code': str})


def quarterly_pool(close, amt, volume, memberships, signal_date,
                   target_size=TARGET_SIZE):
    quarter_start = pd.Timestamp(signal_date).to_period('Q').start_time
    previous = close.index[close.index < quarter_start]
    if previous.empty:
        raise ValueError(f'缺少 {quarter_start.date()} 之前的季度行情')
    as_of = previous[-1]
    members = memberships.loc[memberships.as_of.eq(as_of)].copy()
    if members.empty:
        raise ValueError(f'缺少季度 {as_of.date()} 的恒生综合指数历史名单')
    if members.code.duplicated().any():
        raise ValueError(f'季度 {as_of.date()} 名单包含重复股票')
    members = members.sort_values(['market_cap_hkd', 'code'], ascending=[False, True])
    codes = members.code.tolist()
    prices = close.loc[:as_of].reindex(columns=codes)
    amounts = amt.loc[:as_of].reindex(columns=codes)
    volumes = volume.loc[:as_of].reindex(columns=codes)
    valid = prices.gt(0) & amounts.gt(0) & volumes.gt(0)
    first = prices.where(valid).apply(lambda x: x.first_valid_index())
    history = (pd.to_datetime(first).le(as_of - pd.DateOffset(years=MIN_HISTORY_YEARS))
               & valid.sum().ge(MIN_HISTORY_SESSIONS))
    daily_amount = amounts.rolling(60, min_periods=60).mean().iloc[-1]
    members['data_available'] = members.code.isin(close.columns)
    members['history_ok'] = members.code.map(history)
    members['average_daily_amount_hkd'] = members.code.map(daily_amount)
    members['liquidity_ok'] = members.average_daily_amount_hkd.ge(MIN_DAILY_AMOUNT_HKD)
    members['trading_at_review'] = members.code.map(valid.iloc[-1])
    members['eligible'] = (members.data_available & members.history_ok
                           & members.liquidity_ok & members.trading_at_review
                           & members.market_cap_hkd.gt(0))
    members['selected'] = False
    members.loc[members.loc[members.eligible].head(target_size).index, 'selected'] = True
    members['effective_from'] = quarter_start
    return members.reset_index(drop=True)


def filter_candidates(frame, close, amt, volume, memberships, required_factors=()):
    selected = []
    pools = {}
    for date, group in frame.groupby('date', sort=True):
        quarter = pd.Timestamp(date).to_period('Q')
        if quarter not in pools:
            pools[quarter] = quarterly_pool(close, amt, volume, memberships, date)
        pool = pools[quarter]
        codes = pool.loc[pool.selected, 'code']
        trading = close.loc[date].gt(0) & amt.loc[date].gt(0) & volume.loc[date].gt(0)
        mask = group.code.isin(codes) & group.code.isin(trading.index[trading])
        if required_factors:
            mask &= complete_factors(group, required_factors)
        selected.append(group.loc[mask])
    return pd.concat(selected, ignore_index=True)


def complete_factors(frame, features):
    complete = frame[list(features)].replace([np.inf, -np.inf], np.nan).notna()
    # 亏损公司的正市盈率倒数无定义；该因子允许为空。
    if 'ep_ttm' in features and 'pe_ttm' in frame:
        complete['ep_ttm'] |= frame.pe_ttm.le(0)
    return complete.all(axis=1)


def require_full_portfolio(frame, size):
    if len(frame) < size:
        raise ValueError(f'合格股票只有 {len(frame)} 只，不能生成 {size} 只的完整组合')
