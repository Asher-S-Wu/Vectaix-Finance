#!/usr/bin/env python3
"""build_panel.py <market:a|hk2> — 合并批次行情与估值，生成训练面板。"""
import sys, os, glob
import numpy as np
import pandas as pd
from hk_market_data import filter_hk_prices, filter_hk_valuation, hk_calendar, hk_listing_dates
from project_paths import data_dir

market = sys.argv[1]
is_hk = market == "hk2"
pxdir = data_dir(market) / "raw/prices"
valdir = data_dir(market) / "raw/valuations"

# 1) 行情合并
frames = []
price_files = (sorted((data_dir(market) / 'raw/qveris_prices').glob('*.csv'))
               if is_hk else sorted(glob.glob(f"{pxdir}/batch_*.csv")))
for f in price_files:
    d = pd.read_csv(f, dtype={"wind_code": str})
    frames.append(d)
px = pd.concat(frames).drop_duplicates(["trade_date", "wind_code"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
if is_hk:
    px = filter_hk_prices(px)
px = px.sort_values("trade_date")
print("行情:", px.shape, "股票数:", px.wind_code.nunique(),
      "区间:", px.trade_date.min().date(), "->", px.trade_date.max().date())

close = px.pivot(index="trade_date", columns="wind_code", values="close")
amt = px.pivot(index="trade_date", columns="wind_code", values="amt")
vol = px.pivot(index="trade_date", columns="wind_code", values="volume")
if is_hk:
    dates = hk_calendar()
    close, amt, vol = (frame.reindex(dates) for frame in (close, amt, vol))
    close.index.name = amt.index.name = vol.index.name = "trade_date"

# 港股按每个历史季度检查两年行情，面板不按全时期样本数删列。
if not is_hk:
    enough = close.notna().sum() >= 300
    close, amt, vol = close.loc[:, enough], amt.loc[:, enough], vol.loc[:, enough]
print("有效股票:", close.shape[1])

# 2) 估值: 半月频采样 asof 对齐到交易日
val_wide = {}
uni = pd.read_csv(data_dir(market) / "reference/universe.csv", dtype=str)
if market == "a":
    inds = {"总市值": "mktcap", "市盈率(TTM)": "pe_ttm", "市净率": "pb", "股息率": "dv"}
    code_map = {w: w[:6] for w in uni.windcode}
else:
    inds = {"总市值": "mktcap", "市盈率(TTM)": "pe_ttm", "市净率": "pb", "股息率": "dv"}
    code_map = {w: w.split(".")[0].zfill(5) for w in uni.windcode}

per_ind = {}
for f in glob.glob(f"{valdir}/*.csv"):
    sym = os.path.basename(f)[:-4]
    d = pd.read_csv(f)
    d["date"] = pd.to_datetime(d["date"])
    if is_hk:
        d = filter_hk_valuation(d, sym.zfill(5) + ".HK")
    d = d.set_index("date")
    for ind_cn, ind_en in inds.items():
        if ind_cn in d.columns:
            per_ind.setdefault(ind_en, {})[sym] = d[ind_cn]
for ind_en, series_map in per_ind.items():
    if not series_map:
        continue
    wide = pd.DataFrame(series_map).sort_index()
    # 对齐到交易日: reindex + ffill
    if is_hk:
        wide = wide.reindex(wide.index.union(close.index)).sort_index().ffill().reindex(close.index)
    else:
        wide = wide.reindex(close.index).ffill()
    # 列名映射回 windcode
    inv = {v: k for k, v in code_map.items()}
    wide = wide.rename(columns={c: inv.get(c, c) for c in wide.columns})
    wide = wide.reindex(columns=close.columns)
    if is_hk:
        for code, listing_date in hk_listing_dates().items():
            if code in wide.columns:
                wide.loc[wide.index < listing_date, code] = np.nan
    val_wide[ind_en] = wide
    print(f"估值 {ind_en}: 覆盖率 {wide.notna().mean().mean():.1%}")

# 派生估值因子
derived = {}
if "pe_ttm" in val_wide:
    ep = 1.0 / val_wide["pe_ttm"].where(val_wide["pe_ttm"] > 0)
    derived["ep_ttm"] = ep
if "pb" in val_wide:
    derived["bp"] = 1.0 / val_wide["pb"].where(val_wide["pb"] > 0)
if "dv" in val_wide:
    derived["dv"] = val_wide["dv"]
if "mktcap" in val_wide:
    derived["size"] = -np.log(val_wide["mktcap"] + 1)
    # 换手率代理 = 成交额/总市值
    derived["turnover_proxy"] = amt / (val_wide["mktcap"] * 1e8 + 1)
val_wide.update(derived)

panel = {"close": close, "amt": amt, "volume": vol}
# 港股使用普通 pickle；A 股保留原有压缩格式。
panel = {k: v.astype(np.float32) for k, v in panel.items()}
val_wide = {k: v.astype(np.float32) for k, v in val_wide.items()}
processed = data_dir(market) / "processed"
processed.mkdir(parents=True, exist_ok=True)
pd.to_pickle(panel, processed / "prices.pkl", compression=None if is_hk else "gzip")
pd.to_pickle(val_wide, processed / "valuations.pkl", compression=None if is_hk else "gzip")
print("SAVED", processed / "prices.pkl", processed / "valuations.pkl")
