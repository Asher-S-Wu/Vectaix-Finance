#!/usr/bin/env python3
"""make_report.py — 汇总当前回测、成交模拟与选股信号，更新现有报告和净值图。"""
import json
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import ROOT as BASE, REPORTS, MARKET_FOLDERS, data_dir, model_dir, backtest_dir, qveris_dir
from check_metrics import THRESHOLDS as RULES
TOPN = {"a": 30, "hk2": 15}


def load(market):
    bt = pd.read_pickle(backtest_dir(market) / "returns.pkl")
    metrics = json.loads((backtest_dir(market) / "metrics.json").read_text())
    latest = json.loads((backtest_dir(market) / "metrics_recent.json").read_text())
    ic = pd.read_csv(backtest_dir(market) / "factor_ic.csv")
    return bt, metrics, latest, ic


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


def metrics_table(market, metrics, latest):
    rows = [
        ("ic_mean", "月度排名相关性 IC", ".4f"),
        ("icir", "IC 均值／标准差", ".3f"),
        ("ic_pos_ratio", "IC 为正的月份占比", ".2%"),
        ("ann_excess", "算术年化超额（扣费后，相对池内等权）", ".2%"),
        ("info_ratio", "超额信息比率", ".3f"),
        ("monthly_win", "相对池内等权基准的月胜率", ".2%"),
        ("hit_rate_med", "选股命中率（跑赢池中位数）", ".2%"),
        ("hit_rate", "选股命中率（跑赢池均值，仅参考）", ".2%"),
        ("turnover_ann", "年化换手（倍）", ".2f"),
        ("ann_ret_top", f"Top{TOPN[market]} 算术年化收益", ".2%"),
        ("ann_ret_bench", "池内等权基准算术年化收益", ".2%"),
        ("max_dd_top", "组合月末净值最大回撤", ".2%"),
        ("max_dd_bench", "池内等权基准月末净值最大回撤", ".2%"),
        ("oos_months", "历史滚动预测月份数", "d"),
    ]
    output = []
    for key, label, spec in rows:
        threshold = threshold_text(key) if key in RULES else "—"
        result = ("通过" if passes(metrics[key], RULES[key]) else "未通过") if key in RULES else "参考"
        output.append([label, format(metrics[key], spec), format(latest[key], spec), threshold, result])
    return markdown_table(["指标", "全部历史滚动预测", "最后12个月（已参与历史选型）", "现有 v2 门槛", "全部历史结果"], output)


def equity_plot(bt, market):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(bt.index, (1 + bt["r_top"]).cumprod(), label=f"Top{TOPN[market]} portfolio after costs", lw=1.8, color="#8B4513")
    axes[0].plot(bt.index, (1 + bt["r_bench"]).cumprod(), label="Equal-weight stock pool", lw=1.4, color="#999999")
    axes[0].set_title(f"{'Hong Kong' if market == 'hk2' else 'A-share'}: monthly endpoint backtest")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].bar(bt.index, bt["excess"], color=["#B03A2E" if value < 0 else "#7D8C7C" for value in bt["excess"]], width=20)
    axes[1].set_title("Monthly excess return versus the stock pool")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / f"equity_{MARKET_FOLDERS[market]}.png", dpi=110)
    plt.close(fig)


def factor_table(ic):
    rows = [[row.factor, f"{row.ic_mean:.4f}", f"{row.icir:.3f}", f"{row.ic_pos:.2%}"]
            for row in ic.head(10).itertuples(index=False)]
    return markdown_table(["因子", "IC 均值", "IC 均值／标准差", "IC 为正占比"], rows)


MARKETS = {
    "a": {
        "name": "A股", "currency": "人民币", "benchmark": "沪深300价格指数",
        "replay_key": "a", "replay_dir": "backtests/a/qveris",
        "replay_file": "backtests/a/qveris/replay_summary.json",
        "scoring": "按模型预测分数排序，等权选取前30只；不额外混合因子或平滑分数。",
        "data_note": "此次以9月8日的有效截面替换依据不完整9月9日行情生成的旧名单：793行更新为781行，排序随之更新。原始行情保留。",
        "valuation_note": "部分已核实停牌的持仓缺少当日报价，使用证据支持的最近有效报价估值；相关天数随账户结果披露。",
        "factor_note": "旧模型中 downside_vol 未实际用于任何一棵树的分裂，对应历史 IC 统计为空；本次不重新训练。",
    },
    "hk2": {
        "name": "港股", "currency": "港元", "benchmark": "恒生价格指数",
        "replay_key": "hk", "replay_dir": "backtests/hk/qveris",
        "replay_file": "backtests/hk/qveris/replay_summary.json",
        "scoring": "在当季合格股票池内，预测分数与52周高点因子分别标准化后按1与0.5混合，再将本期与上一个月的混合分数各取一半，等权选取前15只。缺少上月分数的股票不入选。",
        "data_note": "港股行情已按交易日历及核实的证券实体边界清理；计算日期由生成名单时明确指定。",
        "valuation_note": "港股记录中的无成交持仓均有当日有效报价，使用当日报价估值；零成交量没有被当作缺失价格。",
        "factor_note": "downside_vol 已修复并参与当前港股模型训练，28项因子都有历史 IC 统计。",
    },
}


def file_link(path, label=None):
    return f"[{label or Path(path).name}]({BASE / path})"


def assessment(metrics):
    failed = [key for key, rule in RULES.items() if not passes(metrics[key], rule)]
    result = f"{len(RULES) - len(failed)}/{len(RULES)} 项通过"
    if not failed:
        return result + "，达到现有门槛。"
    details = "；".join(f"{RULES[key]['desc']}未达 {threshold_text(key)}" for key in failed)
    return result + "，未达到全部门槛。" + details + "。"


def read_market(market):
    cfg = MARKETS[market]
    bt, metrics, latest, ic = load(market)
    panel = pd.read_pickle(data_dir(market) / "processed/prices.pkl", compression="gzip" if market == "a" else None)
    signals = pd.read_csv(model_dir(market) / "signals.csv", dtype={"code": str}, parse_dates=["date"])
    if market == 'hk2':
        return dict(cfg=cfg, bt=bt, metrics=metrics, latest=latest, ic=ic, panel=panel,
                    signals=signals, date=signals['date'].iloc[0])
    replay = json.loads((BASE / cfg["replay_file"]).read_text())
    replay_market = replay["markets"][cfg["replay_key"]]
    for account in replay_market["scenarios"].values():
        if account["status"] != "complete_account":
            raise ValueError(f"{cfg['name']}成交账户不完整，不能展示其绩效。")
    if signals["date"].nunique() != 1 or set(signals["market"]) != {market}:
        raise ValueError(f"{cfg['name']}选股名单日期或市场不一致。")
    return dict(cfg=cfg, bt=bt, metrics=metrics, latest=latest, ic=ic, panel=panel,
                signals=signals, date=signals["date"].iloc[0], replay=replay,
                replay_market=replay_market, account=replay_market["scenarios"]["0.0015"])


def account_table(data):
    account, cfg = data["account"], data["cfg"]
    bench, currency = cfg["benchmark"], cfg["currency"]
    return markdown_table(["成交模拟指标", "结果"], [
        ["初始资金", f"{account['initial_cash']:,.0f} {currency}"],
        ["期末账户价值", f"{account['final_equity']:,.0f} {currency}"],
        ["扣费后累计收益", f"{account['total_return']:.2%}"],
        ["按实际天数计算的复合年化收益", f"{account['cagr']:.2%}"],
        ["每日账户净值最大回撤", f"{account['max_drawdown']:.2%}"],
        ["年化波动率", f"{account['annualized_volatility']:.2%}"],
        ["夏普比率（无风险收益设为0）", f"{account['sharpe_zero_rf']:.3f}"],
        [f"相对{bench}的月胜率", f"{account['monthly']['monthly_win_vs_price_index']:.2%}"],
        [f"{bench}复合年化收益", f"{account['benchmark_price_index']['cagr']:.2%}"],
        [f"{bench}每日最大回撤", f"{account['benchmark_price_index']['max_drawdown']:.2%}"],
        ["累计费用", f"{account['total_fees_paid']:,.2f} {currency}"],
        ["平均现金占比", f"{account['mean_cash_weight']:.2%}"],
        ["成交笔数", account["trade_count"]],
        ["期末未完成订单数", account["final_pending_orders"]],
        ["持仓存在无成交的天数", account["days_with_stale_valuation"]],
        ["单只持仓最长连续无成交交易日数", account["max_stale_trading_days"]],
        ["信号日排除的股票月份数", account["signal_excluded_stock_months"]],
        ["账户执行数据缺项数", data["replay"]["execution_gap_count"]],
        ["理想等权比较缺失端点数", data["replay"]["ideal_endpoint_gap_count"]],
    ])


def hk_section(data):
    audit = pd.read_csv(data_dir('hk2') / 'reference/universe_audit.csv')
    picks = markdown_table(['排名', '股票代码', '名称', '分数', '目标权重'], [
        [r['rank'], r['code'], r['name'], f"{r['score']:.4f}", f"{r['target_weight']:.2%}"]
        for r in data['signals'].head(15).to_dict('records')
    ])
    metrics, latest = data['metrics'], data['latest']
    return f"""## 港股

### 股票池与数据

以当季度恒生综合指数名单为基础，按上一季度末的数据筛选，最多保留500只。要求至少两年且480个有效交易日的行情，近60个交易日日均成交额不低于1000万港元；评分当天还需有成交且因子齐全。

当前指数名单有{len(audit)}条记录，{int(audit.selected.sum())}只满足季度筛选条件，{len(data['signals'])}只具备完整评分条件，实际组合为15只、每只目标权重6.67%。历史行情面板覆盖{data['panel']['close'].shape[1]}只股票，包括已调出指数的股票。

行情来自QVeris，使用复权收盘价与实际成交额。历史季度名单覆盖2023年末至2026年二季度；新方案从2024年开始评价，首个预测月份用于分数平滑。更早行情用于当时已知候选股的训练。个别历史证券及两条缺少代码的指数记录仍有数据缺口，未用其他股票代替。

评分规则：{data['cfg']['scoring']}

### 新股票池回测

信号区间为{data['bt'].index.min():%Y-%m-%d}至{data['bt'].index.max():%Y-%m-%d}，共{len(data['bt'])}个完整持有期。累计收益{metrics['total_return_top']:.2%}，复合年化收益{metrics['compound_annual_return_top']:.2%}，月度端点最大回撤{metrics['max_dd_top']:.2%}。最近12个月累计收益{latest['total_return_top']:.2%}。

每月等权调整15只持仓，费用按换手比例乘以双边0.3%估算。买卖端点没有成交量时不确认该笔收益。这是持有期端点回测，尚未对新名单重跑逐日订单及整手成交模拟。

{assessment(metrics)}

{metrics_table('hk2', metrics, latest)}

![港股月度回测净值]({REPORTS / 'equity_hk.png'})

### 最新15只模型持仓

计算日期：{data['date']:%Y-%m-%d}。分数表示模型排序，不是预期收益率。

{picks}

### 数据与完整名单

{file_link('data/hk/reference/universe_eligible.csv', '当前可评分股票')}、{file_link('data/hk/reference/universe_audit.csv', '筛选及缺口明细')}、{file_link('models/hk/signals.csv', '完整评分')}、{file_link('backtests/hk/targets.csv', '历史持仓')}。
"""


def market_section(market, data):
    if market == 'hk2':
        return hk_section(data)
    cfg, account, replay_market = data["cfg"], data["account"], data["replay_market"]
    signals, ic, signal_day = data["signals"], data["ic"], data["date"]
    panel = data["panel"]
    source_day = panel["close"].index[-1]
    valid = panel["close"].loc[signal_day].gt(0) & panel["amt"].loc[signal_day].gt(0) & panel["volume"].loc[signal_day].gt(0)
    source_valid = panel["close"].loc[source_day].gt(0) & panel["amt"].loc[source_day].gt(0) & panel["volume"].loc[source_day].gt(0)
    status = "逐日成交账户完整；理想等权端点比较完整。"
    if data["replay"]["ideal_endpoint_gap_count"]:
        status = f"逐日成交账户完整；理想等权端点比较缺少 {data['replay']['ideal_endpoint_gap_count']} 个端点，其完整绩效不展示。"
    model_table = markdown_table(["模型与数据项目", "内容"], [
        ["模型结构", "28个输入因子，3个随机种子的 LightGBM 集成"],
        ["组合规模与调仓", f"前{TOPN[market]}只，等权，每月调仓"],
        ["股票池数量", panel["close"].shape[1]],
        ["原始行情最后日期", f"{source_day:%Y-%m-%d}"],
        ["原始最后日期有效成交股票数", int(source_valid.sum())],
        ["本次选股计算日期", f"{signal_day:%Y-%m-%d}"],
        ["当日有效成交股票数", int(valid.sum())],
        ["有最终分数的股票数", len(signals)],
        ["输入因子数量／有历史IC统计的数量", f"{len(ic)}／{int(ic['ic_mean'].notna().sum())}"],
        ["验收结论", assessment(data["metrics"])],
        ["回放证据状态", status],
    ])
    costs = markdown_table(["每边假设综合费用", "复合年化收益", "每日最大回撤", f"期末账户价值（{cfg['currency']}）"], [
        [f"{float(fee):.2%}", f"{s['cagr']:.2%}", f"{s['max_drawdown']:.2%}", f"{s['final_equity']:,.0f}"]
        for fee, s in replay_market["scenarios"].items()
    ])
    picks = markdown_table(["排名", "股票代码", "名称", "最终分数", "是否入选", "目标权重"], [
        [r.rank, r.code, r.name, f"{r.score:.4f}", "是" if r.selected else "否", f"{r.target_weight:.2%}"]
        for r in signals.head(10).itertuples(index=False)
    ])
    files = markdown_table(["内容", "文件"], [
        ["现用模型", file_link(model_dir(market) / "model.pkl")],
        ["完整选股名单", file_link(model_dir(market) / "signals.csv")],
        ["因子重要性", file_link(model_dir(market) / "importance.csv")],
        ["历史预测", file_link(backtest_dir(market) / "predictions.pkl")],
        ["月度回测指标", file_link(backtest_dir(market) / "metrics.json")],
        ["最后12个月指标", file_link(backtest_dir(market) / "metrics_recent.json")],
        ["单因子历史统计", file_link(backtest_dir(market) / "factor_ic.csv")],
        ["QVeris成交模拟汇总", file_link(cfg["replay_file"])],
        ["QVeris取数记录", file_link(f"{cfg['replay_dir']}/requests.json")],
        ["逐日账户", file_link(qveris_dir(market) / "daily_accounts.csv")],
        ["逐笔成交", file_link(qveris_dir(market) / "trades.csv")],
    ])
    return f"""## {cfg['name']}

### 模型与数据

{model_table}

评分规则：{cfg['scoring']}

行情情况：{cfg['data_note']}

因子情况：{cfg['factor_note']}

### 按成交约束模拟的账户结果

QVeris 回放区间为 {replay_market['start_date']} 至 {replay_market['end_date']}，共 {replay_market['months']} 个调仓月份；币种为{cfg['currency']}，基准为{cfg['benchmark']}。以下主结果采用买卖每边各0.15%的综合费用、单日成交额1%的参与上限。

{account_table(data)}

证据状态：{status}“理想等权比较”是单独核对买入、卖出端点的计算，其缺项与逐日成交账户分别统计。

无成交持仓估值：{cfg['valuation_note']}

最后12个月账户累计收益为 {account['latest_12_months']['total_return']:.2%}，相对{cfg['benchmark']}的月胜率为 {account['latest_12_months']['monthly_win_vs_price_index']:.2%}。该时间段已参与历史方案选择，不属于新的独立留出检验。

### 费用敏感性

保持同一持仓方案，只提高假设费用：

{costs}

### 月度回测与现有门槛

月度持有期端点回测的信号区间为 {data['bt'].index.min():%Y-%m-%d} 至 {data['bt'].index.max():%Y-%m-%d}。按换手率乘以双边0.3%扣费，基准为当期股票池等权收益；统计口径与逐日成交账户分开。

{metrics_table(market, data['metrics'], data['latest'])}

![{cfg['name']}月度端点回测净值]({REPORTS / f'equity_{MARKET_FOLDERS[market]}.png'})

### 最新选股预览（前10名）

计算日期：{signal_day:%Y-%m-%d}。两市场均展示前10名；完整组合为前{TOPN[market]}名，见{file_link(model_dir(market) / 'signals.csv', '完整名单')}中入选标记为真的股票。目标权重为组合目标，实际成交受现金和可交易性约束；分数表示排序，不是预期收益率。

{picks}

### 因子历史表现（前10项）

按历史 IC 绝对值排列，属于全历史描述性统计；因子数量和缺失情况见上方模型与数据表。

{factor_table(ic)}

### 文件与使用

{files}

在项目目录运行以下命令，使用现有模型生成指定日期的名单，不触发训练：

```sh
.venv/bin/python scripts/predict.py {market} --as-of {signal_day:%Y-%m-%d}
```
"""


results = {market: read_market(market) for market in MARKETS}
if len({data["date"] for data in results.values()}) != 1:
    raise ValueError("A股和港股选股计算日期不一致，请先明确同一日期生成名单。")
summary_rows = []
for label, value in [
    ("选股计算日期", lambda d: f"{d['date']:%Y-%m-%d}"),
    ("有分数的股票数量", lambda d: str(len(d['signals']))),
    ("组合持股数", lambda d: str(TOPN[d['signals']['market'].iloc[0]])),
    ("月度回测信号区间", lambda d: f"{d['bt'].index.min():%Y-%m-%d} — {d['bt'].index.max():%Y-%m-%d}"),
    ("月度回测复合年化收益", lambda d: f"{(1 + d['bt'].r_top).prod() ** (12 / len(d['bt'])) - 1:.2%}"),
    ("月度端点最大回撤", lambda d: f"{d['metrics']['max_dd_top']:.2%}"),
    ("现有门槛", lambda d: assessment(d['metrics'])),
]:
    summary_rows.append([label, *[value(data) for data in results.values()]])
summary = markdown_table(["项目", "A股", "港股"], summary_rows)
parts = [f"""# A股与港股多因子量化模型报告

生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}

quant_model 同时支持A股和港股。港股使用按季度筛选的新股票池并已重新训练；A股使用其现有模型。

## 双市场概览

{summary}

上表统一展示月度持有期端点回测，区间和股票池不同。A股章节另列原有逐日成交模拟；港股章节展示新股票池的回测结果。

## 共用计算与阅读口径

两市场均展示月度回测与现有评价门槛。A股持仓30只，港股持仓15只。

选股名单统一包含计算日期、市场、排名、股票代码、名称、最终分数、是否入选和目标权重。以下名单是现在用保留的模型、截至指定日期的行情重新计算的结果，不能当作该历史日期实时发布过的信号。两边都只在指定日价格、成交量、成交额为正的股票中计算截面。

A股章节的逐日成交账户采用月末信号后首个市场交易日收盘价，先卖后买；未约束整手交易，也未单独模拟收盘竞价容量。该账户的指数基准与复权股票收益的股息口径不同。

月度表中“算术年化”是月平均收益或超额乘以12，不等于复合年化；“月末净值最大回撤”只查看月度端点，不等于逐日账户回撤。两个市场沿用同一套9项门槛，没有调整达标要求。全部历史和最后12个月已被查看或参与选型，不能替代新的独立检验。
"""]
for market, data in results.items():
    equity_plot(data["bt"], market)
    parts.append(market_section(market, data))
parts.append(f"""## 共用文件与数据范围

{file_link('scripts/factor_lib.py', '因子计算')}、{file_link('scripts/train_backtest.py', '训练')}、{file_link('scripts/predict.py', '预测')}、{file_link('scripts/rebacktest.py', '月度复算')}、{file_link('scripts/execution_replay.py', '成交模拟')}、{file_link('scripts/check_metrics.py', '指标判定')}和{file_link('scripts/make_report.py', '报告生成')}供两个市场共用。

港股训练使用当季已知候选股的历史行情和估值数据，并按实际季度名单回测；残余历史数据缺口已披露。A股继续沿用原有股票池。报告从各自保存的历史预测和回测读取结果。

更新两市场名单后，同步生成本报告及两张净值图：

```sh
.venv/bin/python scripts/make_report.py
```
""")
(REPORTS / "report.md").write_text("\n".join(parts), encoding="utf-8")
print("双市场报告已生成：", REPORTS / "report.md")
