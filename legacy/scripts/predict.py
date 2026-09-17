#!/usr/bin/env python3
"""predict.py a [--as-of YYYY-MM-DD] [--output FILE] — 用现有 A 股模型生成选股信号。"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from project_paths import ROOT, data_dir, model_dir
from factor_lib import compute_factors
from portfolio_scores import zscore


class LGBEnsemble:  # 与 train_backtest.py 中定义同名同构，供 unpickle 使用
    def __init__(self, params, seeds=(42, 7, 2024)):
        self.params, self.seeds = params, seeds

    def fit(self, X, y, w=None):
        self.models_ = []
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


parser = argparse.ArgumentParser(description="生成 A 股最新完整选股名单")
parser.add_argument("market", choices=("a",), help="a 为 A 股")
parser.add_argument("--output", help="输出文件，默认保存到 models/a/signals.csv")
parser.add_argument("--as-of", type=lambda value: pd.Timestamp(datetime.strptime(value, "%Y-%m-%d")),
                    help="信号日期 YYYY-MM-DD，必须存在于行情中；默认使用最后一个行情日期")
args = parser.parse_args()
output_csv = ROOT / args.output if args.output is not None else model_dir("a") / "signals.csv"
TOPN = 30

bundle = pd.read_pickle(model_dir("a") / "model.pkl")
model, feats = bundle["model"], bundle["features"]
px = pd.read_pickle(data_dir("a") / "processed/prices.pkl", compression="gzip")
val = pd.read_pickle(data_dir("a") / "processed/valuations.pkl", compression="gzip")
last_day = args.as_of if args.as_of is not None else px["close"].index[-1]
if last_day not in px["close"].index:
    parser.error(f"指定信号日期 {last_day.date()} 不在 A 股行情中")
px = {key: frame.loc[:last_day] for key, frame in px.items()}
val = {key: frame.loc[:last_day] for key, frame in val.items()}
close, amt, vol = px["close"], px["amt"], px["volume"]

f = compute_factors(close, amt, vol, val_wide=val, eval_dates=[last_day])
tradable = close.gt(0) & amt.gt(0) & vol.gt(0)
eligible = pd.DataFrame({"date": last_day, "code": tradable.columns[tradable.loc[last_day]]})
f = f.merge(eligible, on=["date", "code"], how="inner")


def cs_norm(g):
    for c in feats:
        x = g[c]
        med = x.median()
        mad = (x - med).abs().median()
        if mad > 0:
            x = x.clip(med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        sd = x.std()
        g[c] = (x - x.mean()) / sd if sd > 0 else 0.0
    return g


f = cs_norm(f.copy())
f["pred"] = model.predict(f[feats])
f["score"] = f.groupby("date")["pred"].transform(zscore)
cur = f.dropna(subset=["score"]).sort_values("score", ascending=False, kind="stable").copy()
universe = pd.read_csv(data_dir("a") / "reference/universe.csv", dtype=str)
cur["name"] = cur["code"].map(dict(zip(universe.windcode, universe["成分券名称"])))
cur["date"] = pd.Timestamp(last_day).strftime("%Y-%m-%d")
cur["market"] = "a"
cur["rank"] = np.arange(1, len(cur) + 1)
cur["selected"] = cur["rank"] <= TOPN
cur["target_weight"] = np.where(cur["selected"], 1.0 / TOPN, 0.0)
columns = ["date", "market", "rank", "code", "name", "score", "selected", "target_weight"]
output_csv.parent.mkdir(parents=True, exist_ok=True)
cur[columns].to_csv(output_csv, index=False)

print(f"市场: a  信号日期: {last_day.date()}  模型训练于: {bundle['trained_at']}")
print(f"打分股票数: {len(cur)}  完整组合: 前 {TOPN} 只股票，每只目标权重 {1.0 / TOPN:.2%}")
print(f"完整榜单: {output_csv}")
