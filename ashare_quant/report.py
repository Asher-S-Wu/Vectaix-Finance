"""Provenance-linked prediction and portfolio research reports; no return promises."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import DATA,RESULTS,MODELS
from .dataset import write_json


def benchmark_curves(source,dates):
    dates=pd.DatetimeIndex(dates).sort_values()
    output=pd.DataFrame({'date':dates});summaries={}
    for code,group in source.groupby('ts_code'):
        group=group.copy();group['date']=pd.to_datetime(group.trade_date,format='%Y%m%d')
        if group.date.duplicated().any():raise ValueError('duplicate benchmark dates')
        price=group.set_index('date').close.reindex(dates)
        missing=int(price.isna().sum())
        curve=price/price.iloc[0] if len(price) and price.iloc[0]>0 else price*np.nan
        output[code]=curve.to_numpy()
        valid=not missing and len(curve)>1
        total=float(curve.iloc[-1]-1) if valid else None
        days=(dates[-1]-dates[0]).days if len(dates)>1 else 0
        summaries[code]=dict(status='complete' if valid else 'incomplete',missing_sessions=missing,
            total_return=total,cagr=float(curve.iloc[-1]**(365.25/days)-1) if valid and days else None,
            max_drawdown=float((curve/curve.cummax()-1).min()) if valid else None,
            basis='price index, excluding dividends and trading costs')
    return output,summaries


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _pct(value):
    return '不可用' if value is None else f'{value:.2%}'


def build_report(data_root=DATA,results_root=RESULTS,model_root=MODELS):
    data_root,results_root,model_root=map(Path,[data_root,results_root,model_root])
    if not (results_root/'training_status.json').is_file():raise ValueError('No completed real training evidence')
    status=_load_json(results_root/'training_status.json')
    if status.get('status') not in ['trained','complete']:raise ValueError('Real training is not complete')
    manifest=_load_json(data_root/'feature_manifest.json');protocol=_load_json(results_root/'protocol.json')
    frozen=_load_json(results_root/'frozen_architecture.json');development=_load_json(results_root/'development_summary.json')
    confirmation=_load_json(results_root/'confirmation_summary.json');active=_load_json(model_root/'active.json')
    execution_path=results_root/'execution_summary.json';execution=_load_json(execution_path) if execution_path.exists() else {'status':'pending'}
    reference_path=results_root/'reference_summary.json';reference=_load_json(reference_path) if reference_path.exists() else {'status':'pending'}
    benchmark_path=results_root/'benchmark_summary.json';benchmarks=_load_json(benchmark_path) if benchmark_path.exists() else {}
    action_path=results_root/'corporate_actions_status.json';actions=_load_json(action_path) if action_path.exists() else {'collection_status':'not_collected'}
    lines=['# A股多因子模型训练与确认报告','',f"数据截止：{manifest['data_as_of']}；币种：CNY；研究模型：{active['model_version']}",'',
        '## 实际完成内容',f"- 真实历史证券 {manifest['securities']:,} 个身份，因子行 {manifest['rows']:,}，可用训练候选行 {manifest['eligible_rows']:,}",
        f"- 基础输入 {len(manifest['features'])} 个；每天最多 {protocol['training_stocks_per_date']} 只确定性抽样训练，日度预测覆盖全部可用身份",
        '- 四类候选：固定因子、Ridge、Small LightGBM、Large LightGBM；1/5/20/60交易日期限，独立概率/收益/区间/评分与解释',
        f"- 2023-01-01之前训练标签，2023独立校准，模型截止 {protocol['training_as_of']}；2024只用于架构选择，2025起为确认集",
        f"- 冻结选择：{frozen['selected_kind']}；规则：{frozen['selection_rule']}；未使用确认集选型",
        '- 最新重训模型与冻结确认模型是两个独立文件；最新模型不能用于宣称历史无泄漏表现',
        '- 这是本次未参与选型的历史确认区间，不是已经发生的实盘记录；模拟初始资金不代表用户实际资产','',
        '## 开发集比较（到2024-12-31已成熟的20日标签）','',
        '| 候选 | 全可用池日Rank IC | 同池日Rank IC（选型） | IC日期 | 概率Brier | 区间覆盖 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind,result in development.items():
        h=result['horizons']['20'];lines.append(f"| {kind} | {h['rank_ic_mean']} | {result.get('common_score_universe',h).get('rank_ic_mean')} | {h['ic_dates']} | {h['brier']} | {_pct(h['interval_coverage'])} |")
    lines+=['','## 未参与选型的确认集预测质量','','| 期限 | 成熟标签 | 评分可用 | 日Rank IC | IC 95%区间 | 方向准确率 | Brier改善 | 区间覆盖 |',
            '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for horizon,h in confirmation['horizons'].items():
        lines.append(f"| {horizon} | {h['mature_labels']:,} | {h['score_available']:,} | {h['rank_ic_mean']} | {h['ic_ci_lower']}, {h['ic_ci_upper']} | {_pct(h['direction_accuracy'])} | {_pct(h['brier_skill'])} | {_pct(h['interval_coverage'])} |")
    lines+=['','置信区间使用60交易日循环块自助法，2000次；概率ECE采用10个等宽分箱。日标签有重叠，不把证券行数视为独立样本数量。',
            '','## 组合与执行审计','',f"实际股数账户回放状态：{execution.get('status',execution.get('accounting_valid'))}",
            '实际股数回放使用原始价格、A股整股/起点约束、T+1、实际每日涨跌停、停牌、成交额参与率、日期相关费用。未核实企业行动不伪造现金、股份或退市退出。',
            '', f"股息证据采集状态：{actions.get('collection_status')}。已接受的厂商最终实施计划仅经过数值/日期一致性校验，未逐项核对发行人原文，不代表个人实际税后到账；现金股息按20%先行预留税额。送股税项、差异化分红、版本冲突及无法核实事件仍保留为未解决限制。",
            '', '方法对照另提供复权研究单位回放与成本压力情景。它不是可执行券商账户，不代表实际税后股息或整手交易；两套口径不得混用。','']
    if isinstance(reference,dict):
        for key,value in reference.items():
            if isinstance(value,dict):lines.append(f"- {key}: 累计 {_pct(value.get('total_return'))}；年化 {_pct(value.get('cagr'))}；最大回撤 {_pct(value.get('max_drawdown_daily',value.get('max_drawdown')))}")
    lines+=['','基准均为价格指数，不含股息和交易费用；同一日期网格，无缺失值前向填充。']
    for code,value in benchmarks.items():lines.append(f"- {code}: 累计 {_pct(value.get('total_return'))}；年化 {_pct(value.get('cagr'))}；最大回撤 {_pct(value.get('max_drawdown'))}")
    lines+=['','## 数据、限制与可复现性','',*['- '+s for s in manifest['limitations']],
        '- 当期股票名称/行业仅展示，不回填为历史模型输入；同时保留在市与退市身份',
        f"- 不连续复权参考断点 {len(manifest.get('adjustment_reference_issues',[]))} 处：保留原始记录、重新积累历史，标签不跨未核实断点",
        '- 所有原始分区记录官方端点、参数、抓取时间、行数与SHA256；训练样本及标准化数据哈希在拟合前验证',
        '- 收益预测是CNY来源复权价格收益。财报公告时点适配器可用，但此次37因子基线没有加入未经版本完整性审核的财务字段',
        '- 当前研究快照 research_ready=true；eligible=false，execution_validated=false，不能理解为收益达标或实盘验证',
        '- 港股源码、展示结果和路径保持独立；原源码分发未包含港股完整训练数据/权重',
        '', '## 主要运行入口', '',
        '- python -m ashare_quant.collect --oldest-first：带来源核验的断点续采',
        '- python -m ashare_quant.pipeline --skip-collection：从已完整缓存依次标准化、构建因子、训练、确认回放和报告',
        '- python -m ashare_quant.api --host 127.0.0.1 --port 8001：独立A股研究API，需要环境变量 ASHARE_QUANT_API_KEY',
        '', '详见同目录 JSON 指标、完整预测 Parquet、交易/现金流水及独立模型快照。']
    # Report rendering uses no credentials or private environment metadata.
    directory=results_root/'report';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'report.zh-CN.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary=dict(status='complete' if execution_path.exists() and reference_path.exists() else 'training_complete_replay_pending',
        selected_kind=frozen['selected_kind'],data_as_of=manifest['data_as_of'],forecast_snapshot=active['forecast_path'],
        research_ready=True,performance_accepted=False,execution_validated=False,report=str(path),
        confirmation=confirmation,execution=execution,reference=reference,benchmarks=benchmarks,corporate_actions=actions)
    write_json(directory/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--results-root',type=Path,default=RESULTS);p.add_argument('--model-root',type=Path,default=MODELS)
    a=p.parse_args();print(json.dumps(build_report(a.data_root,a.results_root,a.model_root),ensure_ascii=False))
