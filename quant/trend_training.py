from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import exchange_calendars as exchange
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import DATA_START, OFFICIAL_CLOSURES, verify_snapshot
from .engine import _read_inputs
from .factors import LABEL_HORIZON, build_labels
from .macro_data import read_macro_snapshot
from .runner import DATA, ROOT, RUNS, read_report
from .trend_factors import TREND_FACTOR_NAMES, build_trend_factors
from .trend_metrics import probability_metrics
from .trend_models import (
    CANDIDATES, CANDIDATE_NAMES, INTERVAL_COVERAGE, MAX_CALIBRATION,
    MIN_CALIBRATION, MIN_TRAINING, TRAINING_YEARS, walk_forward,
)


SELECTION_END = pd.Timestamp('2024-12-31')
EVALUATION_START = pd.Timestamp('2025-01-01')
MIN_SELECTION = 12
TREND_RUNS = RUNS / 'trends'


def _iso(date) -> str:
    return pd.Timestamp(date).date().isoformat()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_trend_report() -> dict:
    pointer = json.loads((TREND_RUNS / 'latest.json').read_text(encoding='utf-8'))
    run_id = pointer['id']
    if not isinstance(run_id, str) or not run_id or run_id in {'.', '..'} or Path(run_id).name != run_id:
        raise ValueError('趋势训练批次编号无效。')
    report = json.loads((TREND_RUNS / run_id / 'report.json').read_text(encoding='utf-8'))
    if report['schemaVersion'] != 1 or report['run']['id'] != run_id:
        raise ValueError('趋势报告版本或批次不一致。')
    return report


def execute_trend_training(progress: Callable[[str, float], None]) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / '.research.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('已有本地研究正在运行。') from exc
        try:
            with threadpool_limits(limits=1):
                return _train_locked(progress)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _train_locked(progress: Callable[[str, float], None]) -> dict:
    prices, universe, manifest = _read_inputs(DATA)
    previous = read_report()
    previous_stocks = {r['stock']['symbol']: r for r in previous['stockReports']}
    macro_prices, macro_manifest = read_macro_snapshot(DATA)
    factors, calendar = build_trend_factors(prices, [s['symbol'] for s in universe['stocks']], universe['benchmark']['symbol'], macro_prices)
    labels = build_labels(factors, prices, calendar)
    future_calendar = exchange.get_calendar('XHKG', start=DATA_START, end=calendar[-1] + pd.Timedelta(days=90)).sessions
    future_calendar = future_calendar.difference(pd.DatetimeIndex(OFFICIAL_CLOSURES))
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    destination = TREND_RUNS / run_id
    destination.mkdir(parents=True)
    sources = sorted((ROOT / 'quant').glob('*.py'))
    runtime = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'exchange-calendars', 'joblib', 'threadpoolctl')}
    runtime['python'] = sys.version
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    record = {
        'id': run_id, 'createdAt': now.isoformat(), 'datasetSha256': manifest['sha256'],
        'sourceHashes': hashes, 'runtime': runtime, 'baselineRunId': previous['run']['id'],
        'baselineReportSha256': hashlib.sha256((RUNS / previous['run']['id'] / 'report.json').read_bytes()).hexdigest(),
        'macroDatasetSha256': macro_manifest['sha256'], 'macroManifestSha256': macro_manifest['manifestSha256'],
        'independentValidation': False, 'historicalDataAlreadyExaminedThrough': previous['summary']['dataEnd'],
        'evaluationNote': '继续训练的探索性历史比较；这些历史已用于上一轮研究，不能当作新的独立验收。',
    }
    specification = {
        'horizonTradingDays': LABEL_HORIZON, 'target': '次日开盘起未来21个交易日复权收益大于零',
        'candidateIds': CANDIDATES, 'factorNames': TREND_FACTOR_NAMES,
        'maximumTrainingYears': TRAINING_YEARS, 'minimumMaturedTrainingRows': MIN_TRAINING,
        'dailyTrainingWeight': 1 / LABEL_HORIZON, 'predictionSpacingTradingDays': LABEL_HORIZON,
        'calibration': {'method': '在更早的成熟样本外预测上拟合正则化逻辑校准', 'minimumSamples': MIN_CALIBRATION, 'maximumSamples': MAX_CALIBRATION},
        'selectionLabelEndThrough': _iso(SELECTION_END), 'minimumSelectionSamples': MIN_SELECTION,
        'selectionCriterion': '只用2024年底前已成熟的校准预测，选择平均概率平方误差最低的候选；同分按固定候选顺序。',
        'evaluationSignalStart': _iso(EVALUATION_START), 'evaluationIndependent': False,
        'evaluationUpdateRule': '候选类型冻结；每个预测日按固定规则，仅用当时已成熟的标签重新训练并更新校准。',
        'referenceIntervalTargetCoverage': INTERVAL_COVERAGE,
        'modelParameters': {
            'logisticC': 1.0, 'calibrationC': 0.5,
            'treeIterations': 80, 'treeLearningRate': 0.05, 'treeMaximumDepth': 2,
            'treeMaximumLeaves': 7, 'treeMinimumLeafRows': 63, 'treeL2': 1.0,
            'treeEarlyStopping': False, 'ensembleLinearWeight': 0.5,
            'returnRidgeAlpha': 10.0, 'seed': 42,
        },
        'sources': [
            'https://scikit-learn.org/stable/modules/calibration.html',
            'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html',
        ],
    }
    _write(destination / 'provenance.json', record)
    _write(destination / 'specification.json', specification)
    # Preserve executable sources and the exact inputs, not only their hashes.
    snapshot = destination / 'reproduction'
    snapshot_files = sources + [ROOT / 'requirements.txt'] + [DATA / name for name in (
        'prices.csv', 'universe.json', 'review.json', 'manifest.json', 'macro.json', macro_manifest['file'],
    )] + [RUNS / previous['run']['id'] / 'report.json']
    for source in snapshot_files:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _write(snapshot / 'runs' / 'latest.json', {'id': previous['run']['id']})
    selections, stock_results, final_models = [], [], {}
    all_predictions = []
    for stock_index, stock in enumerate(universe['stocks']):
        symbol = stock['symbol']
        sf = factors.loc[factors.symbol == symbol]
        sl = labels.loc[labels.symbol == symbol]
        def update(done, total):
            progress(f'{stock["name"]}：已完成 {done}/{total} 个历史预测时点', 0.03 + 0.86 * (stock_index + done / total) / 4)
        predictions, bundle = walk_forward(sf, sl, calendar, previous_stocks[symbol]['selection']['alpha'], progress=update)
        valid = predictions.loc[predictions.scheduled & predictions.probability.notna() & predictions.target.notna()]
        development = valid.loc[valid.label_end <= SELECTION_END]
        candidates = []
        for candidate in CANDIDATES:
            rows = development.loc[development.candidate == candidate]
            if len(rows) < MIN_SELECTION:
                raise ValueError(f'{stock["name"]} 的候选 {candidate} 只有 {len(rows)} 次成熟选型预测，少于 {MIN_SELECTION} 次。')
            metrics = probability_metrics(rows)
            candidates.append({'id': candidate, 'name': CANDIDATE_NAMES[candidate], 'metrics': metrics})
        winner = min(candidates, key=lambda c: (c['metrics']['brier'], CANDIDATES.index(c['id'])))
        selection = {'symbol': symbol, 'selectedId': winner['id'], 'selectedName': winner['name'],
                     'selectionSamples': winner['metrics']['n'], 'selectionEnd': _iso(SELECTION_END),
                     'lockedAt': datetime.now(timezone.utc).isoformat(), 'candidates': candidates}
        selections.append(selection)
        bundle['selected_id'] = winner['id']
        final_models[symbol] = bundle
        stock_results.append((stock, predictions, valid, selection))
        all_predictions.append(predictions)
    # All four decisions are persisted before calculating comparison-period metrics.
    _write(destination / 'selection.json', {'selections': selections})
    reports = []
    for stock, predictions, valid, selection in stock_results:
        evaluated = valid.loc[(valid.date >= EVALUATION_START) & (valid.candidate == selection['selectedId'])]
        metrics = probability_metrics(evaluated)
        selected = predictions.loc[predictions.candidate == selection['selectedId']]
        last = selected.sort_values('date').iloc[-1]
        if pd.isna(last.probability) or pd.isna(last.lower_return):
            raise ValueError(f'{stock["name"]} 的最新模型尚未积累足够校准记录。')
        probability = float(last.probability)
        signal_index = future_calendar.get_loc(last.date)
        agreement = predictions.loc[predictions.date == last.date, ['candidate', 'probability']]
        core_metrics = probability_metrics(valid.loc[(valid.date >= EVALUATION_START) & (valid.candidate == 'core_logistic')])
        forecast = {
            'asOf': _iso(last.date), 'horizonTradingDays': LABEL_HORIZON,
            'returnStart': _iso(future_calendar[signal_index + 1]),
            'returnEnd': _iso(future_calendar[signal_index + LABEL_HORIZON + 1]),
            'upProbability': probability, 'downOrFlatProbability': 1 - probability,
            'direction': '偏向上涨' if probability >= 0.5 else '偏向下跌或持平',
            'directionProbability': max(probability, 1 - probability),
            'expectedReturn': float(last.expected_return),
            'referenceReturnInterval': {'lower': float(last.lower_return), 'upper': float(last.upper_return),
                                        'targetCoverage': INTERVAL_COVERAGE, 'guaranteedCoverage': False},
            'calibrationSamples': int(last.calibration_samples), 'trainingRows': int(last.training_samples),
            'trainingStart': _iso(last.training_start), 'trainingLastLabelEnd': _iso(last.training_last_label_end),
            'calibrationLastLabelEnd': _iso(last.calibration_last_label_end),
            'candidateProbabilities': {row.candidate: float(row.probability) for row in agreement.itertuples()},
            'futureOutcomeObserved': False,
            'interpretation': '概率是模型估计；当前历史样本不足以证明高置信度，收益区间按过去误差估算。',
        }
        reports.append({
            'stock': stock, 'selection': selection, 'evaluation': metrics,
            'coreEvaluation': core_metrics, 'latestForecast': forecast,
            'evidence': {'independentValidation': False, 'liveReady': False,
                         'status': '等待未来验证',
                         'expandedFactorsSelected': selection['selectedId'] != 'core_logistic',
                         'historicalBrierImprovedOverCore': metrics['brier'] < core_metrics['brier'],
                         'probabilitySkillOverBaseRate': metrics['brierSkill']},
        })
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(destination / 'predictions.csv', index=False, date_format='%Y-%m-%d')
    factors.to_csv(destination / 'factors.csv', index=False, date_format='%Y-%m-%d')
    labels.to_csv(destination / 'labels.csv', index=False, date_format='%Y-%m-%d')
    joblib.dump(final_models, destination / 'models.joblib')
    # Detect a concurrent data/source edit instead of assigning an incorrect fingerprint.
    if verify_snapshot(DATA)['sha256'] != manifest['sha256']:
        raise ValueError('训练期间行情已变化，本轮不能标记完成。')
    _, final_macro = read_macro_snapshot(DATA)
    if final_macro['sha256'] != macro_manifest['sha256'] or final_macro['manifestSha256'] != macro_manifest['manifestSha256']:
        raise ValueError('训练期间宏观数据已变化，本轮不能标记完成。')
    if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        raise ValueError('训练期间程序已修改，本轮不能标记完成。')
    report = {
        'schemaVersion': 1, 'run': {**record, 'completedAt': datetime.now(timezone.utc).isoformat()},
        'summary': {'stocks': len(reports), 'factorCount': len(TREND_FACTOR_NAMES), 'dataEnd': _iso(calendar[-1]),
                    'predictionHorizonTradingDays': LABEL_HORIZON, 'independentValidation': False, 'liveReady': False},
        'specification': specification, 'stockReports': reports,
        'macroData': macro_manifest,
        'limitations': [
            '所有历史已被先前研究查看；时间隔离的重演仍不等于真正未见过的未来验证。',
            '每21个交易日预测一次，避免收益区间重叠；相邻市场阶段仍可能相关，样本数不是独立实验次数。',
            '逐日训练收益标签重叠，每行权重为1/21；概率校准只使用更早预测且已成熟的非重叠收益。',
            '药明康德上市历史较短，可用于选模型和校准的样本较少，不能据此宣称稳定优势。',
            '概率误差与涨跌命中率不等于扣费交易收益，本轮没有产生下单指令或实盘仓位。',
            '收益均为含股息复权收益，参考区间不是保本区间；非平稳市场下不保证目标覆盖率。',
            '本轮因子来自经审查的Qveris/FMP日线；财报公布时间不可信的数据不进入历史训练。',
            '美债环境以IEF基金价格表示，不是利率数值；只使用美国日期严格早于香港预测日的真实收盘行情。',
            '当前四只股票由现有持仓指定，存在事后选样；行情截至2026-08-31，预测以该日为起点。',
        ],
    }
    _write(destination / 'report.json', report)
    pending = TREND_RUNS / f'latest.{run_id}.pending.json'
    _write(pending, {'id': run_id})
    pending.replace(TREND_RUNS / 'latest.json')
    progress('四只股票训练完成，概率评估和模型已保存', 1.0)
    return report
