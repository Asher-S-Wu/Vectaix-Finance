from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as exchange
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import DATA_START, OFFICIAL_CLOSURES
from .lab_data import build_lab_dataset
from .lab_evaluation import forecast_metrics, trading_metrics
from .lab_events import add_event_factors
from .lab_models import BASE_CANDIDATE_IDS, CANDIDATE_IDS, HORIZON, calibrate_walk_forward, model_specification, raw_walk_forward
from .portfolio import Market
from .runner import DATA, ROOT, RUNS


LAB_RUNS = RUNS / 'lab'
OUTER_START_YEAR = 2019
EXCELLENCE_RULES = {
    'minimumAccuracy': 0.65, 'minimumBalancedAccuracy': 0.60,
    'minimumUpRecall': 0.55, 'minimumDownRecall': 0.55,
    'minimumAccuracyExcessAlwaysUp': 0.02, 'minimumBrierSkill': 0.05,
    'minimumNonOverlappingIntervals': 36, 'minimumPositiveYearFraction': 0.60,
    'minimumHighConfidenceNonOverlappingIntervals': 12, 'minimumHighConfidenceAccuracy': 0.65,
    'minimumNetSharpe': 0.80, 'minimumMaxDrawdown': -0.25,
    'minimumRiskMatchedAnnualExcess': 0.02,
    'positiveBrierBootstrapLowerBoundRequired': True, 'positiveDoubleSlippageExcessRequired': True,
    'independentFutureObservationRequired': True,
}


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _iso(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def read_lab_report() -> dict:
    pointer = json.loads((LAB_RUNS / 'latest.json').read_text(encoding='utf-8'))
    run_id = pointer['id']
    if not isinstance(run_id, str) or not run_id or run_id in {'.', '..'} or Path(run_id).name != run_id:
        raise ValueError('共享训练批次编号无效。')
    report = json.loads((LAB_RUNS / run_id / 'report.json').read_text(encoding='utf-8'))
    if report['schemaVersion'] != 2 or report['run']['id'] != run_id:
        raise ValueError('当前报告不是包含公告对照的第二版格式，或报告批次不一致；原始报告文件保留。')
    return report


def _inner_score(rows: pd.DataFrame) -> dict:
    year_differences = []
    for _, group in rows.groupby(['symbol', rows.date.dt.year]):
        if len(group) < 24:
            continue
        truth = (group.target.to_numpy() > 0).astype(float)
        difference = np.mean((group.probability.to_numpy() - truth) ** 2 - (group.base_probability.to_numpy() - truth) ** 2)
        year_differences.append(float(difference))
    balanced, downside, symbols = [], [], []
    for symbol, group in rows.groupby('symbol'):
        truth = group.target.to_numpy() > 0
        hit = (group.probability.to_numpy() >= 0.5) == truth
        if len(group) < 24 or not truth.any() or truth.all():
            continue
        balanced.append(float((hit[truth].mean() + hit[~truth].mean()) / 2))
        downside.append(float(hit[~truth].mean()))
        symbols.append(symbol)
    if not year_differences or not balanced:
        raise ValueError('内层选型缺少足够的跨股票、跨年度成熟预测。')
    mean_difference, instability = float(np.mean(year_differences)), float(np.std(year_differences))
    mean_balanced, worst_downside = float(np.mean(balanced)), float(min(downside))
    penalty = 0.5 * max(0.0, 0.52 - mean_balanced) + 0.1 * max(0.0, 0.35 - worst_downside)
    return {
        'score': mean_difference + 0.5 * instability + penalty,
        'meanBrierDifference': mean_difference, 'yearStockBrierStd': instability,
        'meanBalancedAccuracy': mean_balanced, 'worstStockDownRecall': worst_downside,
        'qualificationPenalty': penalty, 'stockYearGroups': len(year_differences),
        'stocks': symbols, 'forecastRows': len(rows), 'distinctForecastDates': int(rows.date.nunique()),
        'researchQualificationPassed': mean_difference < 0 and mean_balanced >= 0.52 and worst_downside >= 0.35,
    }


def select_annual_configurations(predictions: pd.DataFrame, last_year: int, target_symbols: list[str], candidate_ids: tuple[str, ...]):
    selections, outer = [], []
    for year in range(OUTER_START_YEAR, last_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        inside = predictions.loc[(predictions.date >= cutoff - pd.DateOffset(years=3)) & (predictions.label_end < cutoff)
                                 & predictions.symbol.isin(target_symbols)]
        candidates = []
        keys = None
        for candidate in candidate_ids:
            current = inside.loc[inside.candidate == candidate].sort_values(['date', 'symbol'])
            current_keys = list(zip(current.date, current.symbol))
            if keys is None:
                keys = current_keys
            elif current_keys != keys:
                raise ValueError('内层候选未使用相同日期和证券。')
            candidates.append({'candidate': candidate, **_inner_score(current)})
        winner = min(candidates, key=lambda row: (row['score'], candidate_ids.index(row['candidate'])))
        selections.append({
            'year': year, 'selectedCandidate': winner['candidate'],
            'innerStart': _iso(cutoff - pd.DateOffset(years=3)),
            'innerLastMaturedLabelEnd': _iso(inside.label_end.max()),
            'selectionInformationCutoffExclusive': _iso(cutoff),
            'scope': '按原目标股的历史预测选择共同配置；同行只供训练与共享校准，各目标股单独验收。',
            'researchQualificationPassed': winner['researchQualificationPassed'],
            'candidates': candidates,
        })
        current = predictions.loc[(predictions.date.dt.year == year) & (predictions.candidate == winner['candidate'])].copy()
        current['selection_year'] = year
        outer.append(current)
    return pd.concat(outer, ignore_index=True).sort_values(['symbol', 'date']).reset_index(drop=True), selections


def _excellent_checks(metrics: dict, trading: dict, yearly: list[dict], high_metrics: dict | None) -> list[dict]:
    rule = EXCELLENCE_RULES
    complete_years = [row for row in yearly if row['completeCalendarYear']]
    positive_fraction = sum(row['metrics']['brierImprovement'] > 0 for row in complete_years) / len(complete_years) if complete_years else None
    values = [
        ('accuracy', metrics['accuracy'], rule['minimumAccuracy']),
        ('balanced_accuracy', metrics['balancedAccuracy'], rule['minimumBalancedAccuracy']),
        ('up_recall', metrics['upRecall'], rule['minimumUpRecall']),
        ('down_recall', metrics['downRecall'], rule['minimumDownRecall']),
        ('accuracy_excess_always_up', metrics['accuracyExcessAlwaysUp'], rule['minimumAccuracyExcessAlwaysUp']),
        ('probability_skill', metrics['brierSkill'], rule['minimumBrierSkill']),
        ('non_overlapping_history', metrics['greedyNonOverlappingCount'], rule['minimumNonOverlappingIntervals']),
        ('positive_year_fraction', positive_fraction, rule['minimumPositiveYearFraction']),
        ('high_confidence_history', high_metrics['greedyNonOverlappingCount'] if high_metrics is not None else 0, rule['minimumHighConfidenceNonOverlappingIntervals']),
        ('high_confidence_accuracy', high_metrics['accuracy'] if high_metrics is not None else None, rule['minimumHighConfidenceAccuracy']),
        ('net_sharpe', trading['strategy']['sharpe'], rule['minimumNetSharpe']),
        ('max_drawdown', trading['strategy']['maxDrawdown'], rule['minimumMaxDrawdown']),
        ('risk_matched_annual_excess', trading['strategy']['annualExcess'], rule['minimumRiskMatchedAnnualExcess']),
    ]
    checks = [{'key': key, 'value': value, 'threshold': threshold,
               'passed': value is not None and value >= threshold} for key, value, threshold in values]
    interval = metrics['bootstrap95']['brierImprovement']
    lower = interval[0] if interval is not None else None
    checks.extend([
        {'key': 'brier_improvement_interval', 'value': lower, 'threshold': 0, 'passed': lower is not None and lower > 0},
        {'key': 'double_slippage_excess', 'value': trading['doubleSlippage']['annualExcess'], 'threshold': 0,
         'passed': trading['doubleSlippage']['annualExcess'] > 0},
    ])
    return checks


def execute_lab_training(progress):
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / '.research.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('另一个量化研究仍在运行。') from exc
        try:
            with threadpool_limits(limits=1):
                return _execute_locked(progress)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _execute_locked(progress):
    progress('校验十二只股票、市场环境及官方业绩公告数据', 0.01)
    dataset = add_event_factors(build_lab_dataset(DATA), DATA)
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    destination = LAB_RUNS / run_id
    destination.mkdir(parents=True)
    source_files = sorted((ROOT / 'quant').glob('*.py')) + [ROOT / 'requirements.txt']
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    runtime = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'exchange-calendars', 'joblib', 'threadpoolctl')}
    record = {'id': run_id, 'createdAt': now.isoformat(), 'sourceHashes': source_hashes,
              'dataset': dataset['provenance'], 'runtime': runtime, 'python': sys.version,
              'independentValidation': False,
              'evaluationNote': '扩大数据后的探索性历史研究；全部历史已经被查看，年度时间隔离不构成真正未来验证。'}
    specification = {
        **model_specification(), 'featureGroups': dataset['feature_groups'], 'featureNames': dataset['feature_names'],
        'outerStartYear': OUTER_START_YEAR, 'innerYears': 3,
        'selection': '每年开始前，仅用原目标股此前三年已成熟预测；股票年度等权Brier差+0.5倍年度差异标准差，再加低平衡命中和低下跌召回惩罚。同行只供训练与校准。全年固定候选类型，滚动训练参数。',
        'selectionPenalty': {'balancedAccuracyFloor': 0.52, 'balancedPenaltyWeight': 0.5,
                             'worstStockDownRecallFloor': 0.35, 'downPenaltyWeight': 0.1,
                             'minimumForecastsPerStockYear': 24},
        'latestRefit': '数据截止日另行重训最新模型；该日最新预测单独封存，不覆盖历史定期预测。',
        'excellenceRules': EXCELLENCE_RULES,
        'ruleStatus': '项目事先设定的研究门槛，不是行业统一标准；不会因本轮结果改变门槛。',
        'trading': {'entryProbability': 0.55, 'annualVolatilityTarget': 0.15, 'signalEverySessions': 5,
                    'execution': '次日开盘，旧引擎真实历史费用，单股与现金'},
        'uncertainty': {'overlappingLabels': True, 'bootstrapBlockForecasts': 6, 'replications': 2000, 'seed': 42},
    }
    _write(destination / 'provenance.json', record)
    _write(destination / 'specification.json', specification)
    snapshot = destination / 'reproduction'
    for relative, digest in dataset['provenance']['fileHashes'].items():
        source, target = DATA / relative, snapshot / 'data' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('数据复制期间源文件发生变化。')
    for source in source_files:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dataset['factors'].to_csv(destination / 'factors.csv', index=False, date_format='%Y-%m-%d')
    dataset['labels'].to_csv(destination / 'labels.csv', index=False, date_format='%Y-%m-%d')
    progress(f'已冻结{len(CANDIDATE_IDS)}个配置，开始逐历史时点训练', 0.03)
    raw, latest_raw, models, fit_audits = raw_walk_forward(
        dataset, destination, lambda done, total: progress(f'模型训练：{done}/{total} 个历史重训时点', 0.03 + 0.67 * done / total),
    )
    raw.to_csv(destination / 'raw-predictions.csv', index=False, date_format='%Y-%m-%d')
    latest_raw.to_csv(destination / 'latest-raw-predictions.csv', index=False, date_format='%Y-%m-%d')
    joblib.dump(models, destination / 'raw-models.joblib')
    _write(destination / 'fit-audit.json', {'fits': fit_audits})
    calibrated, latest, calibrators, calibration_audits = calibrate_walk_forward(
        raw, latest_raw, lambda done, total: progress(f'历史概率校准：{done}/{total} 个模型', 0.70 + 0.17 * done / total),
    )
    calibrated.to_csv(destination / 'candidate-predictions.csv', index=False, date_format='%Y-%m-%d')
    _write(destination / 'calibration-audit.json', {'calibrations': calibration_audits})
    end = dataset['calendar'][-1]
    chosen, selections = select_annual_configurations(calibrated, end.year, dataset['target_symbols'], CANDIDATE_IDS)
    price_chosen, price_selections = select_annual_configurations(calibrated, end.year, dataset['target_symbols'], BASE_CANDIDATE_IDS)
    _write(destination / 'selection.json', {'selections': selections})
    chosen.to_csv(destination / 'selected-predictions.csv', index=False, date_format='%Y-%m-%d')
    _write(destination / 'price-only-selection.json', {'selections': price_selections})
    price_chosen.to_csv(destination / 'price-only-selected-predictions.csv', index=False, date_format='%Y-%m-%d')
    current_candidate = selections[-1]['selectedCandidate']
    future_calendar = exchange.get_calendar('XHKG', start=DATA_START, end=end + pd.Timedelta(days=90))
    future = future_calendar.sessions.difference(pd.DatetimeIndex(OFFICIAL_CLOSURES))
    end_position = future.get_loc(end)
    target_prices = dataset['prices'].loc[dataset['prices'].symbol.isin(dataset['target_symbols'] + ['02800'])]
    securities = dataset['universe']['stocks'] + [dataset['universe']['benchmark']]
    market = Market.from_prices(target_prices, dataset['calendar'], '02800', dataset['universe']['corporateActions'],
                                {stock['symbol']: stock['stampDutyExemptFrom'] for stock in securities})
    reports = []
    for index, stock in enumerate(dataset['universe']['stocks']):
        progress(f'{stock["name"]}：核算历史预测与扣费交易表现', 0.88 + index * 0.025)
        all_rows = chosen.loc[chosen.symbol == stock['symbol']].sort_values('date')
        matured = all_rows.loc[all_rows.target.notna() & (all_rows.label_end <= end)]
        metrics = forecast_metrics(matured)
        recent = forecast_metrics(matured.loc[matured.date >= pd.Timestamp('2025-01-01')])
        price_rows = price_chosen.loc[price_chosen.symbol == stock['symbol']].sort_values('date')
        price_matured = price_rows.loc[price_rows.target.notna() & (price_rows.label_end <= end)]
        if not list(zip(matured.date, matured.symbol)) == list(zip(price_matured.date, price_matured.symbol)):
            raise ValueError('公告模型与价格模型的评估日期不一致。')
        price_metrics = forecast_metrics(price_matured)
        price_recent = forecast_metrics(price_matured.loc[price_matured.date >= pd.Timestamp('2025-01-01')])
        yearly = []
        for year, rows in matured.groupby(matured.date.dt.year):
            year_calendar = dataset['calendar'][dataset['calendar'].year == year]
            complete = year < end.year and rows.date.min() <= year_calendar[4] and rows.date.max() >= year_calendar[-6]
            yearly.append({'year': int(year), 'completeCalendarYear': bool(complete), 'metrics': forecast_metrics(rows)})
        high_rows = matured.loc[np.maximum(matured.probability, 1 - matured.probability) >= 0.65]
        high_metrics = forecast_metrics(high_rows, compute_bootstrap=False) if not high_rows.empty else None
        trading = trading_metrics(market, all_rows, stock['symbol'], all_rows.date.min(), end, dataset['factors'])
        checks = _excellent_checks(metrics, trading, yearly, high_metrics)
        historical_excellent = all(check['passed'] for check in checks)
        last = latest.loc[(latest.symbol == stock['symbol']) & (latest.candidate == current_candidate)]
        if len(last) != 1:
            raise ValueError('最新模型未对指定股票生成唯一预测。')
        probability = float(last.probability.iloc[0])
        expert, calibration_mode = current_candidate.split(':')
        generated_at = datetime.now(timezone.utc)
        window_started = pd.Timestamp(generated_at) >= future_calendar.session_open(future[end_position + 1])
        forecast = {'asOf': _iso(end), 'returnStart': _iso(future[end_position + 1]),
                    'returnEnd': _iso(future[end_position + HORIZON + 1]), 'horizonTradingDays': HORIZON,
                    'candidate': current_candidate, 'upProbability': probability,
                    'direction': '偏向上涨' if probability >= 0.5 else '偏向下跌或持平',
                    'calibrationMode': calibration_mode, 'calibration': {**calibrators[expert], 'applied': calibration_mode == 'monotone'},
                    'generatedAt': generated_at.isoformat(), 'targetWindowAlreadyStarted': bool(window_started),
                    'prospectiveAtIssue': not bool(window_started),
                    'interpretation': '基于较早信息截面的重建预测，生成时目标窗口已开始，不能视作当时提前发出的预测。' if window_started else '生成时目标窗口尚未开始；模型本身仍处于研究状态。',
                    'futureOutcomeObserved': False}
        reports.append({'stock': stock, 'evaluation': metrics, 'recentEvaluation': recent,
                        'priceOnlyEvaluation': price_metrics, 'priceOnlyRecentEvaluation': price_recent,
                        'eventComparison': {'sameForecastDates': True,
                                            'accuracyChange': metrics['accuracy'] - price_metrics['accuracy'],
                                            'brierImprovement': price_metrics['brier'] - metrics['brier'],
                                            'recentAccuracyChange': recent['accuracy'] - price_recent['accuracy'],
                                            'recentBrierImprovement': price_recent['brier'] - recent['brier']},
                        'yearlyEvaluation': yearly, 'highConfidenceEvaluation': high_metrics,
                        'trading': trading, 'latestForecast': forecast, 'checks': checks,
                        'historicalExcellent': historical_excellent, 'independentValidation': False,
                        'currentSelectionQualified': selections[-1]['researchQualificationPassed'],
                        'status': '逐年系统达到历史门槛，尚未证明当前模型优秀' if historical_excellent else '未通过优秀模型门槛'})
    bundle = {'models': models, 'calibrators': calibrators, 'selectedCandidate': current_candidate,
              'asOf': _iso(end), 'featureGroups': dataset['feature_groups'], 'specification': specification,
              'sectors': {stock['symbol']: stock['sector'] for stock in dataset['universe']['stocks'] + dataset['universe']['peerStocks']}}
    joblib.dump(bundle, destination / 'models.joblib')
    latest.to_csv(destination / 'latest-predictions.csv', index=False, date_format='%Y-%m-%d')
    for relative, digest in dataset['provenance']['fileHashes'].items():
        if hashlib.sha256((DATA / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('训练期间输入文件已变化，本批次不能标记完成。')
    for relative, digest in source_hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('训练期间源码已变化，本批次不能标记完成。')
    report = {'schemaVersion': 2, 'run': {**record, 'completedAt': datetime.now(timezone.utc).isoformat()},
              'summary': {'stocks': 4, 'learningStocks': 12, 'factorCount': len(dataset['feature_names']),
                          'candidateCount': len(CANDIDATE_IDS), 'dataEnd': _iso(end),
                          'earningsEventCount': sum(item['eventCount'] for item in dataset['provenance']['events']['securities']),
                          'priceOnlyCandidateCount': len(BASE_CANDIDATE_IDS),
                          'historicalExcellentCount': sum(row['historicalExcellent'] for row in reports),
                          'independentValidation': False, 'liveReady': False},
              'specification': specification, 'annualSelections': selections, 'stockReports': reports,
              'priceOnlyAnnualSelections': price_selections,
              'limitations': [
                  '这是扩大数据后的探索性历史重演，所有历史均已被研究查看；无法把反复训练后的优胜结果当成新的独立验证。',
                  '每五个交易日预测未来二十一个交易日，标签有重叠；报告另列互不重叠区间数量并用连续区块估算不确定性。',
                  '同行按当前业务关系指定，存在事后选样，十二只股票的同日表现也相互相关；股票行数不是独立市场次数。',
                  '美国行业ETF仅作环境信息，不等同于港股行业指数；财报公布时点不可信的数据仍排除。',
                  '官方业绩公告仅提供已公布事件及其后价格反应，不包含尚未核验的利润、营收或财报预期差数值；新增与价格候选在同日期比较。',
                  '共享校准保持原始排序，但不能给无信息信号制造预测能力；原始与校准版本均参与事先固定的配置比较。',
                  '历史优秀衡量逐年选择系统的全期表现，不代表近期或最新年度候选已达标；近期成绩与当年选型资格另列。',
                  '每条预测同时记录信息截止日、实际生成时间和目标窗口；生成前已经开始的窗口属于重建研究，不能用作真实前瞻验收。',
                  '优秀门槛在训练前保存；没有为达到门槛而更改预测跨度、评估日期、费用或概率数值。',
                  '策略为复权研究单位的单股与现金模拟，不是实际账户股数或下单指令；未模拟整手限制与实际股息到账。',
              ]}
    _write(destination / 'report.json', report)
    pending = LAB_RUNS / f'latest.{run_id}.pending.json'
    _write(pending, {'id': run_id})
    pending.replace(LAB_RUNS / 'latest.json')
    progress('已保存全部候选、逐年选择、交易评估和可复现模型', 1.0)
    return report
