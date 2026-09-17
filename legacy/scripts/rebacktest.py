#!/usr/bin/env python3
"""rebacktest.py a [config_json] — 复用现用 A 股模型的已存预测重跑组合回测。"""
import argparse, json
import numpy as np
import pandas as pd
from factor_lib import forward_labels
from project_paths import data_dir, backtest_dir
from pathlib import Path

parser = argparse.ArgumentParser(description="复用 A 股现用模型的历史预测生成组合回测")
parser.add_argument("market", choices=("a",), help="a 为 A 股")
parser.add_argument("config_json", nargs="?", type=json.loads, default={}, help="组合参数 JSON")
args = parser.parse_args()
market, cfg = args.market, args.config_json
TOPN = cfg.get("topn", 30)
LIQ_GATE = cfg.get("liq_gate", 0.0)
COST = cfg.get("cost", 0.003)
results = backtest_dir(market)
output = Path(cfg['output_dir']) if 'output_dir' in cfg else results

pred = pd.read_pickle(results / "predictions.pkl")


# 先处理全部可预测日期，再只评价已到卖出日的持有期。
px = pd.read_pickle(data_dir(market) / "processed/prices.pkl", compression="gzip")
labels = forward_labels(px['close'])
realized = labels[["date", "label_end"]].drop_duplicates()
realized = realized.loc[realized["label_end"].le(px["close"].index[-1]), "date"]
pred = pred[pred["date"].isin(realized)]
if 'start_date' in cfg:
    pred = pred[pred.date.ge(pd.Timestamp(cfg['start_date']))]
if pred.empty:
    raise ValueError('没有满足股票池规则的完整回测月份')

bench = pred.groupby("date")["fwd_ret"].mean()
recs = []
selections = []
prev_hold = None
for d, g in pred.groupby("date"):
    gsel = g
    if LIQ_GATE > 0:
        thr = g["amt_log_20"].quantile(LIQ_GATE)
        gsel = g[g["amt_log_20"] >= thr]
    top = gsel.nlargest(TOPN, "pred")
    selected = top.copy()
    selected["target_weight"] = 1.0 / TOPN
    selected["rank"] = np.arange(1, len(selected) + 1)
    selections.append(selected)
    cur = set(top["code"])
    turn = 1.0 if prev_hold is None else 1 - len(cur & prev_hold) / TOPN
    prev_hold = cur
    cost = turn * COST
    r_top = top["fwd_ret"].mean() - cost if top["fwd_ret"].notna().all() else np.nan
    r_b = bench.loc[d]
    r_med = g["fwd_ret"].median()
    recs.append({"date": d, "r_top": r_top, "r_bench": r_b, "excess": r_top - r_b,
                 "hit_rate": (top["fwd_ret"] > r_b).mean(),
                 "hit_rate_med": (top["fwd_ret"] > r_med).mean(),
                 "turnover": turn,
                 "selected_stocks": len(top), "missing_return_stocks": int(top["fwd_ret"].isna().sum()),
                 "is_holdout": bool(g["is_holdout"].iloc[0]),
                 "ic": g["pred"].corr(g["fwd_ret"], method="spearman")})
bt = pd.DataFrame(recs).set_index("date")
output.mkdir(parents=True, exist_ok=True)
bt.to_pickle(output / "returns.pkl")
pd.concat(selections, ignore_index=True).to_csv(output / "targets.csv", index=False)

def metrics_of(sub):
    if sub["r_top"].isna().any():
        return {"status": "missing_execution_prices", "oos_months": int(len(sub)),
                "missing_return_months": int(sub["r_top"].isna().sum()),
                "missing_return_stock_months": int(sub["missing_return_stocks"].sum())}
    nav_t = (1 + sub["r_top"]).cumprod()
    nav_b = (1 + sub["r_bench"]).cumprod()
    ex = sub["excess"]
    n = len(sub)
    return {
        "ic_mean": float(sub["ic"].mean()),
        "icir": float(sub["ic"].mean() / (sub["ic"].std() + 1e-12)),
        "ic_pos_ratio": float((sub["ic"] > 0).mean()),
        "ann_excess": float(ex.mean() * 12),
        "info_ratio": float(ex.mean() / (ex.std() + 1e-12) * np.sqrt(12)),
        "monthly_win": float((sub["r_top"] > sub["r_bench"]).mean()),
        "hit_rate": float(sub["hit_rate"].mean()),
        "hit_rate_med": float(sub["hit_rate_med"].mean()),
        "turnover_ann": float(sub["turnover"].mean() * 12),
        "oos_months": int(n),
        "ann_ret_top": float(sub["r_top"].mean() * 12),
        "ann_ret_bench": float(sub["r_bench"].mean() * 12),
        "total_return_top": float(nav_t.iloc[-1] - 1),
        "compound_annual_return_top": float(nav_t.iloc[-1] ** (12 / n) - 1),
        "max_dd_top": float((nav_t / nav_t.cummax().clip(lower=1) - 1).min()),
        "max_dd_bench": float((nav_b / nav_b.cummax().clip(lower=1) - 1).min()),
    }

m_all = metrics_of(bt)
m_ho = metrics_of(bt[bt["is_holdout"]])
json.dump(m_all, open(output / "metrics.json", "w"), indent=2)
json.dump(m_ho, open(output / "metrics_recent.json", "w"), indent=2)
print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m_all.items()}, ensure_ascii=False, indent=1))
print("最后12个月:", json.dumps(m_ho, ensure_ascii=False))
