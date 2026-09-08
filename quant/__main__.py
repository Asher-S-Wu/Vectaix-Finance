from __future__ import annotations

import argparse
import json
import math
import sys


def _number(value: object, *, percent: bool = False) -> str:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('指标必须是有限数字')
    return f'{value * 100:.2f}%' if percent else f'{value:.2f}'


def _report_text(report: dict) -> str:
    from .runner import RUNS

    try:
        rows = report['stockReports']
        if not isinstance(rows, list) or not rows:
            raise ValueError('报告没有股票结果')
        lines = [
            'Vectaix · 本地量化研究',
            f"数据截至：{report['summary']['dataEnd']}",
            f"研究记录：{report['run']['evaluationNote']}",
            '',
        ]
        for row in rows:
            stock, period = row['stock'], row['holdout']
            metrics, baseline = row['metrics']['holdout'], row['metrics']['baseline']
            passed = row['summary']['statisticalPass']
            if type(passed) is not bool:
                raise ValueError('验收结论格式无效')
            sharpe = '无定义' if metrics['sharpe'] is None else _number(metrics['sharpe'])
            lines.extend([
                f"{stock['name']}（{stock['symbol']}）",
                f"  验收区间：{period['start']} 至 {period['end']}",
                f"  模型年化：{_number(metrics['cagr'], percent=True)}；长期持有年化：{_number(baseline['cagr'], percent=True)}",
                f"  模型最大回撤：{_number(metrics['maxDrawdown'], percent=True)}；夏普比率：{sharpe}",
                f"  验收结论：{'历史指标达标' if passed else '未通过预定验收'}",
                '',
            ])
        lines.extend([
            '以上为含股息并扣除交易成本的历史研究，不是实际账户收益或未来收益保证。',
            f"结果目录：{RUNS / report['run']['id']}",
        ])
        return '\n'.join(lines)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('研究报告内容不完整或指标格式错误，无法展示；现有文件保持原样。') from exc


def _trend_report_text(report: dict) -> str:
    from .trend_training import TREND_RUNS

    lines = [
        'Vectaix · 趋势概率训练',
        f"行情截至：{report['summary']['dataEnd']}；预测跨度：21个交易日；因子数：{report['summary']['factorCount']}",
        report['run']['evaluationNote'], '',
    ]
    for row in report['stockReports']:
        stock, metrics, forecast = row['stock'], row['evaluation'], row['latestForecast']
        interval = forecast['referenceReturnInterval']
        lines.extend([
            f"{stock['name']}（{stock['symbol']}） · {row['selection']['selectedName']}",
            f"  基于{forecast['asOf']}收盘信息：{forecast['direction']}，上涨概率 {_number(forecast['upProbability'], percent=True)}",
            f"  预测收益区间：{forecast['returnStart']}开盘至{forecast['returnEnd']}开盘（21个交易日）",
            f"  21日收益参考区间：{_number(interval['lower'], percent=True)} 至 {_number(interval['upper'], percent=True)}（目标覆盖80%，不保证）",
            f"  后段历史命中率：{_number(metrics['accuracy'], percent=True)}，共{metrics['n']}次；原收益模型：{_number(metrics['oldAccuracy'], percent=True)}",
            f"  概率误差：{metrics['brier']:.4f}；八因子对照：{metrics['coreBrier']:.4f}；历史涨频对照：{metrics['baseBrier']:.4f}（越低越好）",
            f"  校准记录：{forecast['calibrationSamples']}次；证据状态：{row['evidence']['status']}", '',
        ])
    lines.extend([
        '本轮完成了模型训练；现有样本尚不能证明四只股票都获得稳定的高置信预测能力。',
        f"结果目录：{TREND_RUNS / report['run']['id']}",
    ])
    return '\n'.join(lines)


def _lab_report_text(report: dict) -> str:
    from .lab_training import LAB_RUNS

    summary = report['summary']
    lines = [
        'Vectaix · 同行、跨市场与业绩事件研究',
        f"行情截至：{summary['dataEnd']}；学习股票：{summary['learningStocks']}只；因子库：{summary['factorCount']}项；官方业绩事件：{summary['earningsEventCount']}条；候选：{summary['candidateCount']}个",
        report['run']['evaluationNote'], '',
    ]
    for row in report['stockReports']:
        stock, metrics, recent = row['stock'], row['evaluation'], row['recentEvaluation']
        balanced = '不可用' if metrics['balancedAccuracy'] is None else _number(metrics['balancedAccuracy'], percent=True)
        skill = '不可用' if metrics['brierSkill'] is None else _number(metrics['brierSkill'], percent=True)
        lines.extend([
            f"{stock['name']}（{stock['symbol']}）：{row['status']}",
            f"  逐年向前评估：{metrics['start']} 至 {metrics['end']}；{metrics['n']}次预测，最多可选{metrics['greedyNonOverlappingCount']}个不重叠收益区间",
            f"  方向命中率：{_number(metrics['accuracy'], percent=True)}；始终看涨：{_number(metrics['alwaysUpAccuracy'], percent=True)}；涨跌平衡命中率：{balanced}",
            f"  相对历史涨频的概率误差改善：{skill}",
            f"  2025年后：命中率{_number(recent['accuracy'], percent=True)}，始终看涨{_number(recent['alwaysUpAccuracy'], percent=True)}",
            f"  同日期价格模型命中率：{_number(row['priceOnlyEvaluation']['accuracy'], percent=True)}；加入公告候选后的概率误差变化：{row['eventComparison']['brierImprovement']:+.4f}（正数为改善）",
            f"  扣费年化：{_number(row['trading']['strategy']['cagr'], percent=True)}；同波动目标持有：{_number(row['trading']['riskMatchedBuyHold']['cagr'], percent=True)}",
            f"  截至{row['latestForecast']['asOf']}信息的上涨概率：{_number(row['latestForecast']['upProbability'], percent=True)}（{row['latestForecast']['returnStart']}开盘至{row['latestForecast']['returnEnd']}开盘）",
            f"  预测性质：{row['latestForecast']['interpretation']}",
            f"  最新年度配置的历史选型资格：{'通过' if row['currentSelectionQualified'] else '未通过'}",
            f"  优秀门槛：通过{sum(check['passed'] for check in row['checks'])}/{len(row['checks'])}项", '',
        ])
    lines.extend([
        f"达到全部历史优秀门槛：{summary['historicalExcellentCount']}/4；尚未完成模型冻结后的未来验证。",
        f"结果目录：{LAB_RUNS / report['run']['id']}",
    ])
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog='python -m quant', description='纯本地港股多因子研究工具')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', help='校验已有 Qveris 行情并准备研究数据')
    sub.add_parser('research', help='按冻结规则逐只训练、回测并保存结果')
    sub.add_parser('report', help='只读取并展示最新研究报告，不启动训练')
    sub.add_parser('train-trends', help='扩展因子、训练并校准四只股票的趋势概率')
    sub.add_parser('trend-report', help='查看最近一次趋势概率训练结果')
    sub.add_parser('train-lab', help='结合同行与跨市场数据，训练并逐年筛选趋势模型')
    sub.add_parser('lab-report', help='查看同行与跨市场联合训练结果')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            from .data import prepare_dataset
            from .runner import DATA

            manifest = prepare_dataset(DATA)
            print(json.dumps({key: manifest[key] for key in ('provider', 'stockCount', 'rowCount', 'startDate', 'endDate', 'sha256')}, ensure_ascii=False, indent=2))
        elif args.command == 'research':
            from .runner import execute_research

            report = execute_research(lambda message, value: print(f'{value:6.1%}  {message}', flush=True))
            print(_report_text(report))
        elif args.command == 'report':
            from .runner import read_report

            print(_report_text(read_report()))
        elif args.command == 'train-trends':
            from .trend_training import execute_trend_training

            report = execute_trend_training(lambda message, value: print(f'{value:6.1%}  {message}', flush=True))
            print(_trend_report_text(report))
        elif args.command == 'trend-report':
            from .trend_training import read_trend_report

            print(_trend_report_text(read_trend_report()))
        elif args.command == 'train-lab':
            from .lab_training import execute_lab_training

            report = execute_lab_training(lambda message, value: print(f'{value:6.1%}  {message}', flush=True))
            print(_lab_report_text(report))
        elif args.command == 'lab-report':
            from .lab_training import read_lab_report

            print(_lab_report_text(read_lab_report()))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
