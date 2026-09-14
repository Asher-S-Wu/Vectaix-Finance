#!/usr/bin/env python3
"""汇总现有 A 股回测、成交模拟与选股信号，更新报告和净值图。"""
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_metrics import THRESHOLDS as RULES
from project_paths import ROOT as BASE, REPORTS, data_dir, model_dir, backtest_dir, qveris_dir

MARKET = "a"
TOPN = 30


def file_link(path, label=None):
    return f"[{label or Path(path).name}]({BASE / path})"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def passes(value, rule):
    return value >= rule["min"] if "min" in rule else value <= rule["max"]


def threshold_text(key):
    rule = RULES[key]
    value = rule["min"] if "min" in rule else rule["max"]
    operator = "≥" if "min" in rule else "≤"
    text = f"{value:.0%}" if key in {"ic_pos_ratio", "ann_excess", "monthly_win", "hit_rate_med"} else f"{value:g}"
    return operator + text


def assessment(metrics):
    failed = [key for key, rule in RULES.items() if not passes(metrics[key], rule)]
    if not failed:
        return f"{len(RULES)}/{len(RULES)} 项通过，达到现有门槛。"
    details = "；".join(f"{RULES[key]['desc']}未达 {threshold_text(key)}" for key in failed)
    return f"{len(RULES) - len(failed)}/{len(RULES)} 项通过，未达到全部门槛。{details}。"


def metrics_table(metrics, latest):
    rows = [
        ("ic_mean", "月度排名相关性 IC", ".4f"), ("icir", "IC 均值／标准差", ".3f"),
        ("ic_pos_ratio", "IC 为正的月份占比", ".2%"),
        ("ann_excess", "算术年化超额（扣费后，相对池内等权）", ".2%"),
        ("info_ratio", "超额信息比率", ".3f"), ("monthly_win", "相对池内等权基准的月胜率", ".2%"),
        ("hit_rate_med", "选股命中率（跑赢池中位数）", ".2%"), ("hit_rate", "选股命中率（跑赢池均值，仅参考）", ".2%"),
        ("turnover_ann", "年化换手（倍）", ".2f"), ("ann_ret_top", "Top30 算术年化收益", ".2%"),
        ("ann_ret_bench", "池内等权基准算术年化收益", ".2%"), ("max_dd_top", "组合月末净值最大回撤", ".2%"),
        ("max_dd_bench", "池内等权基准月末净值最大回撤", ".2%"), ("oos_months", "历史滚动预测月份数", "d"),
    ]
    values = []
    for key, label, spec in rows:
        threshold = threshold_text(key) if key in RULES else "—"
        result = "通过" if key in RULES and passes(metrics[key], RULES[key]) else ("未通过" if key in RULES else "参考")
        values.append([label, format(metrics[key], spec), format(latest[key], spec), threshold, result])
    return markdown_table(["指标", "全部历史滚动预测", "最后12个月（已参与历史选型）", "现有 v2 门槛", "全部历史结果"], values)


def equity_plot(bt):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(bt.index, (1 + bt["r_top"]).cumprod(), label="Top30 portfolio after costs", lw=1.8, color="#8B4513")
    axes[0].plot(bt.index, (1 + bt["r_bench"]).cumprod(), label="Equal-weight stock pool", lw=1.4, color="#999999")
    axes[0].set_title("A-share: monthly endpoint backtest")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].bar(bt.index, bt["excess"], color=["#B03A2E" if value < 0 else "#7D8C7C" for value in bt["excess"]], width=20)
    axes[1].set_title("Monthly excess return versus the stock pool"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS / "equity_a.png", dpi=110); plt.close(fig)


bt = pd.read_pickle(backtest_dir(MARKET) / "returns.pkl")
metrics = json.loads((backtest_dir(MARKET) / "metrics.json").read_text())
latest = json.loads((backtest_dir(MARKET) / "metrics_recent.json").read_text())
ic = pd.read_csv(backtest_dir(MARKET) / "factor_ic.csv")
panel = pd.read_pickle(data_dir(MARKET) / "processed/prices.pkl", compression="gzip")
signals = pd.read_csv(model_dir(MARKET) / "signals.csv", dtype={"code": str}, parse_dates=["date"])
replay = json.loads((BASE / "backtests/a/qveris/replay_summary.json").read_text())
replay_market = replay["markets"]["a"]
for account in replay_market["scenarios"].values():
    if account["status"] != "complete_account":
        raise ValueError("A股成交账户不完整，不能展示其绩效。")
if signals["date"].nunique() != 1 or set(signals["market"]) != {MARKET}:
    raise ValueError("A股选股名单日期或市场不一致。")

account = replay_market["scenarios"]["0.0015"]
signal_day = signals["date"].iloc[0]
source_day = panel["close"].index[-1]
valid = panel["close"].loc[signal_day].gt(0) & panel["amt"].loc[signal_day].gt(0) & panel["volume"].loc[signal_day].gt(0)
source_valid = panel["close"].loc[source_day].gt(0) & panel["amt"].loc[source_day].gt(0) & panel["volume"].loc[source_day].gt(0)
status = "逐日成交账户完整；理想等权端点比较完整。" if not replay["ideal_endpoint_gap_count"] else f"逐日成交账户完整；理想等权端点比较缺少 {replay['ideal_endpoint_gap_count']} 个端点，其完整绩效不展示。"

model_table = markdown_table(["模型与数据项目", "内容"], [
    ["模型结构", "28个输入因子，3个随机种子的 LightGBM 集成"], ["组合规模与调仓", "前30只，等权，每月调仓"],
    ["股票池数量", panel["close"].shape[1]], ["原始行情最后日期", f"{source_day:%Y-%m-%d}"],
    ["原始最后日期有效成交股票数", int(source_valid.sum())], ["本次选股计算日期", f"{signal_day:%Y-%m-%d}"],
    ["当日有效成交股票数", int(valid.sum())], ["有最终分数的股票数", len(signals)],
    ["输入因子数量／有历史IC统计的数量", f"{len(ic)}／{int(ic['ic_mean'].notna().sum())}"], ["验收结论", assessment(metrics)],
    ["回放证据状态", status],
])
costs = markdown_table(["每边假设综合费用", "复合年化收益", "每日最大回撤", "期末账户价值（人民币）"], [
    [f"{float(fee):.2%}", f"{s['cagr']:.2%}", f"{s['max_drawdown']:.2%}", f"{s['final_equity']:,.0f}"]
    for fee, s in replay_market["scenarios"].items()
])
account_table = markdown_table(["成交模拟指标", "结果"], [
    ["初始资金", f"{account['initial_cash']:,.0f} 人民币"], ["期末账户价值", f"{account['final_equity']:,.0f} 人民币"],
    ["扣费后累计收益", f"{account['total_return']:.2%}"], ["按实际天数计算的复合年化收益", f"{account['cagr']:.2%}"],
    ["每日账户净值最大回撤", f"{account['max_drawdown']:.2%}"], ["年化波动率", f"{account['annualized_volatility']:.2%}"],
    ["夏普比率（无风险收益设为0）", f"{account['sharpe_zero_rf']:.3f}"], ["相对沪深300价格指数的月胜率", f"{account['monthly']['monthly_win_vs_price_index']:.2%}"],
    ["累计费用", f"{account['total_fees_paid']:,.2f} 人民币"], ["平均现金占比", f"{account['mean_cash_weight']:.2%}"],
    ["成交笔数", account["trade_count"]], ["期末未完成订单数", account["final_pending_orders"]],
])
picks = markdown_table(["排名", "股票代码", "名称", "最终分数", "是否入选", "目标权重"], [
    [r.rank, r.code, r.name, f"{r.score:.4f}", "是" if r.selected else "否", f"{r.target_weight:.2%}"]
    for r in signals.head(10).itertuples(index=False)
])
factor_rows = [[r.factor, f"{r.ic_mean:.4f}", f"{r.icir:.3f}", f"{r.ic_pos:.2%}"] for r in ic.head(10).itertuples(index=False)]
files = markdown_table(["内容", "文件"], [
    ["现用模型", file_link("models/a/model.pkl")], ["完整选股名单", file_link("models/a/signals.csv")],
    ["因子重要性", file_link("models/a/importance.csv")], ["历史预测", file_link("backtests/a/predictions.pkl")],
    ["月度回测指标", file_link("backtests/a/metrics.json")], ["最后12个月指标", file_link("backtests/a/metrics_recent.json")],
    ["单因子历史统计", file_link("backtests/a/factor_ic.csv")], ["QVeris成交模拟汇总", file_link("backtests/a/qveris/replay_summary.json")],
    ["QVeris取数记录", file_link("backtests/a/qveris/requests.json")], ["逐日账户", file_link(qveris_dir(MARKET) / "daily_accounts.csv")],
    ["逐笔成交", file_link(qveris_dir(MARKET) / "trades.csv")],
])

equity_plot(bt)
report = f"""# A股多因子量化模型报告

生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}

## 模型与数据

{model_table}

评分规则：按模型预测分数排序，等权选取前30只；不额外混合因子或平滑分数。

## 按成交约束模拟的账户结果

QVeris 回放区间为 {replay_market['start_date']} 至 {replay_market['end_date']}，共 {replay_market['months']} 个调仓月份；以下主结果采用买卖每边各0.15%的综合费用、单日成交额1%的参与上限。

{account_table}

证据状态：{status}

### 费用敏感性

{costs}

## 月度回测与现有门槛

月度持有期端点回测的信号区间为 {bt.index.min():%Y-%m-%d} 至 {bt.index.max():%Y-%m-%d}。按换手率乘以双边0.3%扣费，基准为当期股票池等权收益；统计口径与逐日成交账户分开。

{metrics_table(metrics, latest)}

![A股月度端点回测净值]({REPORTS / 'equity_a.png'})

## 最新选股预览（前10名）

计算日期：{signal_day:%Y-%m-%d}。完整组合为前30名，见{file_link('models/a/signals.csv', '完整名单')}中入选标记为真的股票。

{picks}

## 因子历史表现（前10项）

{markdown_table(['因子', 'IC 均值', 'IC 均值／标准差', 'IC 为正占比'], factor_rows)}

## 文件与使用

{files}

使用现有模型生成指定日期的名单，不触发训练：

```sh
.venv/bin/python scripts/predict.py a --as-of {signal_day:%Y-%m-%d}
```
"""
(REPORTS / "report.md").write_text(report, encoding="utf-8")
print("A股报告已生成：", REPORTS / "report.md")
