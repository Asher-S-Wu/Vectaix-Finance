#!/usr/bin/env python3
"""factor_lib.py — 因子库计算模块(供面板构建与训练共用)
输入: wide DataFrame (index=trade_date, columns=ticker): close, amt, volume
      valuation: dict indicator -> wide DataFrame (稀疏采样, 已 asof 对齐到交易日)
输出: 每个调仓日(月末)的截面因子长表 + 次月收益标签
"""
import numpy as np
import pandas as pd

TRADING_DAYS = {"1m": 20, "3m": 60, "6m": 120, "12m": 240}

def month_ends(dates):
    s = pd.Series(pd.to_datetime(dates))
    return s.groupby(s.dt.to_period("M")).max().tolist()

def compute_factors(close, amt, volume, val_wide=None, eval_dates=None):
    """在指定评估日(默认每个月末交易日)计算截面因子。返回 long DataFrame: date, code, f1..fk"""
    ret = close.pct_change(fill_method=None)
    logret = np.log(close / close.shift(1))
    me_dates = month_ends(close.index) if eval_dates is None else list(eval_dates)

    # 预计算 rolling 量
    mom_1m = close / close.shift(20) - 1
    mom_3m = close / close.shift(60) - 1
    mom_6m_s1 = close.shift(20) / close.shift(120) - 1   # 6个月动量跳过近1月
    mom_12m_s1 = close.shift(20) / close.shift(240) - 1  # 12个月动量跳过近1月
    rev_5d = close / close.shift(5) - 1
    vol_20 = logret.rolling(20).std()
    vol_60 = logret.rolling(60).std()
    max_20 = ret.rolling(20).max()
    amihud_20 = (ret.abs() / (amt + 1)).rolling(20).mean() * 1e8
    amt_log_20 = np.log(amt.rolling(20).mean() + 1)
    amt_ratio = amt.rolling(20).mean() / (amt.rolling(120).mean() + 1)
    corr_pv_20 = close.rolling(20).corr(volume)
    high_52w = close / close.rolling(240, min_periods=120).max()
    skew_60 = ret.rolling(60).skew()
    downside_vol = logret.clip(upper=0).pow(2).rolling(60).mean().pow(0.5)
    # 距250日低点
    low_52w = close / close.rolling(240, min_periods=120).min()

    # --- v2 新增因子 ---
    ma60 = close.rolling(60).mean()
    bias_60 = close / (ma60 + 1e-12) - 1                     # 60日乖离率
    ma20 = close.rolling(20).mean()
    bias_20 = close / (ma20 + 1e-12) - 1                     # 20日乖离率
    vol_cv_20 = volume.rolling(20).std() / (volume.rolling(20).mean() + 1)  # 量能稳定性

    frames = {
        "mom_1m": mom_1m, "mom_3m": mom_3m, "mom_6m_s1": mom_6m_s1, "mom_12m_s1": mom_12m_s1,
        "rev_5d": rev_5d, "vol_20": vol_20, "vol_60": vol_60, "max_20": max_20,
        "amihud_20": amihud_20, "amt_log_20": amt_log_20, "amt_ratio": amt_ratio,
        "corr_pv_20": corr_pv_20, "high_52w": high_52w, "skew_60": skew_60,
        "downside_vol": downside_vol, "low_52w": low_52w,
        "bias_60": bias_60, "bias_20": bias_20, "vol_cv_20": vol_cv_20,
    }
    if val_wide:
        if "turnover_proxy" in val_wide:
            tp = val_wide["turnover_proxy"]
            frames["turn_chg"] = tp / (tp.rolling(60).mean() + 1e-12)   # 换手率相对60日均值变化
            frames["turn_20"] = tp.rolling(20).mean()                    # 20日平均换手
        frames.update(val_wide)

    recs = []
    for d in me_dates:
        row = {"date": d}
        for name, fdf in frames.items():
            if d in fdf.index:
                row[name] = fdf.loc[d]
        recs.append(row)
    # 组装
    out = None
    for name in frames:
        rows = []
        for r in recs:
            v = r.get(name)
            if isinstance(v, pd.Series):
                rows.append(pd.DataFrame({"date": r["date"], "code": v.index, name: v.values}))
        f_long = pd.concat(rows)
        out = f_long if out is None else out.merge(f_long, on=["date", "code"], how="outer")
    return out

def forward_labels(close, me_dates=None):
    """标签: 信号日=月末T, 买入=T后第1个交易日收盘, 卖出=次月末后第1个交易日收盘"""
    if me_dates is None:
        me_dates = month_ends(close.index)
    idx = close.index
    nxt = {d: idx[idx.get_loc(d) + 1] if idx.get_loc(d) + 1 < len(idx) else None for d in me_dates}
    rows = []
    for i in range(len(me_dates) - 1):
        t0, t1 = nxt.get(me_dates[i]), nxt.get(me_dates[i + 1])
        if t0 is None or t1 is None:
            continue
        r = close.loc[t1] / close.loc[t0] - 1
        rows.append(pd.DataFrame({"date": me_dates[i], "code": r.index, "fwd_ret": r.values,
                                  "label_start": t0, "label_end": t1}))
    return pd.concat(rows)
