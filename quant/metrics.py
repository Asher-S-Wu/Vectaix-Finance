from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance(values: pd.Series, benchmark: pd.Series | None = None) -> dict:
    values = values.sort_index().astype(float)
    if len(values) < 20 or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError("净值数据不足 20 天，或包含无效资产价值。")
    returns = values.pct_change(fill_method=None).iloc[1:]
    years = (values.index[-1] - values.index[0]).days / 365.25
    if years <= 0:
        raise ValueError("净值日期跨度必须大于零。")
    total_return = float(values.iloc[-1] / values.iloc[0] - 1)
    cagr = float((1 + total_return) ** (1 / years) - 1)
    volatility = float(returns.std(ddof=1) * math.sqrt(252))
    daily_std = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / daily_std * math.sqrt(252)) if daily_std > 0 else None
    max_drawdown = float((values / values.cummax() - 1).min())
    annual_excess = None
    if benchmark is not None:
        aligned = benchmark.reindex(values.index)
        if aligned.isna().any() or (aligned <= 0).any():
            raise ValueError("比较基准缺少对应交易日的有效净值。")
        benchmark_cagr = float((aligned.iloc[-1] / aligned.iloc[0]) ** (1 / years) - 1)
        annual_excess = cagr - benchmark_cagr
    return {
        "totalReturn": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "maxDrawdown": max_drawdown,
        "annualExcess": annual_excess,
        "observations": len(returns),
        "start": values.index[0].date().isoformat(),
        "end": values.index[-1].date().isoformat(),
    }


def period_returns(equity: pd.DataFrame, frequency: str) -> list[dict]:
    periods = equity.groupby(equity.index.to_period(frequency)).last()
    changes = periods.pct_change(fill_method=None)
    changes.iloc[0] = periods.iloc[0] / equity.iloc[0] - 1
    records = []
    for period, values in changes.iterrows():
        item = {"date": str(period)}
        if frequency == "Y":
            item["year"] = int(period.year)
        item.update({key: float(values[key]) for key in equity.columns})
        item["excess"] = item["strategy"] - item["benchmark"]
        records.append(item)
    return records
