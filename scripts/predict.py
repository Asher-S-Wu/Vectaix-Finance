#!/usr/bin/env python3
"""predict.py <market:a|hk2> [--as-of YYYY-MM-DD] [--output FILE] — 用活体模型生成选股信号
与回测口径一致:
  a   : 28因子 LGB集成, Top30
  hk2 : 28因子 LGB集成 + high_52w混合(w=0.5) + 信号平滑(0.5), Top15
默认输出: models/a/signals.csv 或 models/hk/signals.csv (全池打分降序)"""
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from project_paths import ROOT, data_dir, model_dir, backtest_dir
from factor_lib import compute_factors, month_ends
from portfolio_scores import blend_scores, smooth_scores
from hk_universe import filter_candidates, load_memberships, require_full_portfolio

class LGBEnsemble:  # 与 train_backtest.py 中定义同名同构, 供 unpickle 使用
    def __init__(self, params, seeds=(42, 7, 2024)):
        self.params, self.seeds = params, seeds
    def fit(self, X, y, w=None):
        self.models_ = []
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


parser = argparse.ArgumentParser(description="生成 A 股或港股的最新完整选股名单")
parser.add_argument("market", choices=("a", "hk2"), help="a 为 A 股，hk2 为港股")
parser.add_argument("--output", help="输出文件，默认保存到对应市场的 models 目录")
parser.add_argument("--as-of", type=lambda value: pd.Timestamp(datetime.strptime(value, "%Y-%m-%d")),
                    help="信号日期 YYYY-MM-DD，必须存在于行情中；默认使用最后一个行情日期")
args = parser.parse_args()
market = args.market
output_csv = ROOT / args.output if args.output is not None else model_dir(market) / "signals.csv"

CONF = {
    "a":   {"smooth": 0.0, "blend_factor": None, "blend_w": 0.0, "topn": 30},
    "hk2": {"smooth": 0.5, "blend_factor": "high_52w", "blend_w": 0.5, "topn": 15},
}[market]

bundle = pd.read_pickle(model_dir(market) / "model.pkl")
model, feats = bundle["model"], bundle["features"]
compression = "gzip" if market == "a" else None
px = pd.read_pickle(data_dir(market) / "processed/prices.pkl", compression=compression)
val = pd.read_pickle(data_dir(market) / "processed/valuations.pkl", compression=compression)
last_day = args.as_of if args.as_of is not None else px["close"].index[-1]
if last_day not in px["close"].index:
    parser.error(f"指定信号日期 {last_day.date()} 不在 {market} 行情中")
px = {key: frame.loc[:last_day] for key, frame in px.items()}
val = {key: frame.loc[:last_day] for key, frame in val.items()}
close, amt, vol = px["close"], px["amt"], px["volume"]

evals = [last_day]

f = compute_factors(close, amt, vol, val_wide=val, eval_dates=evals)
# 只在信号日有实际成交的股票中计算截面，避免缺失行情参与打分。
tradable = close.gt(0) & amt.gt(0) & vol.gt(0)
eligible = pd.concat([
    pd.DataFrame({"date": d, "code": tradable.columns[tradable.loc[d]]})
    for d in evals
], ignore_index=True)
f = f.merge(eligible, on=["date", "code"], how="inner")
if market == 'hk2':
    f = filter_candidates(f, close, amt, vol, load_memberships(), feats)
    require_full_portfolio(f, CONF['topn'])
raw_factors = f.copy()

def cs_norm(g):
    for c in feats:
        x = g[c]; med = x.median(); mad = (x - med).abs().median()
        if mad > 0: x = x.clip(med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        sd = x.std()
        g[c] = (x - x.mean()) / sd if sd > 0 else 0.0
    return g
f = pd.concat([cs_norm(g.copy()) for _, g in f.groupby("date")], ignore_index=True)
f["pred"] = model.predict(f[feats])

f = blend_scores(f[["date", "code", "pred"]], raw_factors,
                 CONF["blend_factor"], CONF["blend_w"], normalize_predictions=True)
if CONF["smooth"] > 0:
    historical = pd.read_pickle(backtest_dir(market) / "predictions.pkl")
    completed_months = {d for d in month_ends(close.index) if d < last_day}
    historical = historical[historical["date"].isin(completed_months)]
    expected_dates = {d for d in completed_months if historical["date"].min() <= d}
    missing_dates = expected_dates - set(historical["date"])
    if historical.empty or missing_dates:
        raise ValueError(f"缺少当期模型的历史预测，不能生成平滑信号: {sorted(missing_dates)}")
    historical_factors = pd.read_pickle(data_dir(market) / "processed/factors.pkl")
    historical = blend_scores(historical[["date", "code", "pred"]], historical_factors,
                              CONF["blend_factor"], CONF["blend_w"], normalize_predictions=True)
    f = pd.concat([historical, f], ignore_index=True)
f = smooth_scores(f, CONF["smooth"])
cur = f[f["date"] == last_day].dropna(subset=["score"]).copy()
if market == 'hk2':
    require_full_portfolio(cur, CONF['topn'])
cur = cur.sort_values("score", ascending=False, kind="stable")

# 名称映射
u = pd.read_csv(data_dir(market) / "reference/universe.csv", dtype=str)
name_column = "成分券名称" if market == "a" else "name"
name_map = dict(zip(u.windcode, u[name_column]))
cur["name"] = cur["code"].map(name_map)
n = CONF["topn"]
cur["date"] = pd.Timestamp(last_day).strftime("%Y-%m-%d")
cur["market"] = market
cur["rank"] = np.arange(1, len(cur) + 1)
cur["selected"] = cur["rank"] <= n
cur["target_weight"] = np.where(cur["selected"], 1.0 / n, 0.0)
columns = ["date", "market", "rank", "code", "name", "score", "selected", "target_weight"]
cur[columns].to_csv(output_csv, index=False)

print(f"市场: {market}  信号日期: {last_day.date()}  模型训练于: {bundle['trained_at']}")
print(f"打分股票数: {len(cur)}  完整组合: 前 {n} 只股票，每只目标权重 {1.0 / n:.2%}")
print("\n=== 排名前 10 ===")
print(cur.head(10)[["rank", "code", "name", "score", "selected", "target_weight"]].to_string(index=False))
print("\n=== 排名后 10 ===")
print(cur.tail(10)[["rank", "code", "name", "score", "selected", "target_weight"]].to_string(index=False))
print(f"\n完整榜单: {output_csv}")
print(f"完整组合见榜单中 selected 为 True 的前 {n} 行。")
