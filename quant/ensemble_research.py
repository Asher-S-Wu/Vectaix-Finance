from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import research_calendar
from .engine import _read_inputs
from .lab_evaluation import forecast_metrics, trading_metrics
from .lab_training import EXCELLENCE_RULES, OUTER_START_YEAR, _excellent_checks
from .portfolio import Market
from .runner import ROOT, RUNS


SOURCES = {
    'OLD': ('lab/20260908T093918232471Z', 40, 'selected-predictions.csv'),
    'NEW': ('fundamental/20260908T111451178392Z', 16, 'ALL-selected-predictions.csv'),
}
TARGETS = ('00883', '02359', '00003', '00939')
MATCH_COLUMNS = ['date', 'symbol', 'label_start', 'label_end', 'target', 'base_probability']
INNER_YEARS = 3
HALF_LIFE = 252
ETA = 1.0
OBSERVATION_WEIGHT = 5 / 21
PRIOR_SHRINKAGE = 0.5


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _iso(value):
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _dependencies() -> list[Path]:
    pending, seen = ['ensemble_research'], set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        path = ROOT / 'quant' / f'{module}.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        seen.add(module)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                pending.append(node.module)
    return [ROOT / 'quant' / '__init__.py', *[ROOT / 'quant' / f'{name}.py' for name in sorted(seen)],
            ROOT / 'requirements.txt']


def _freeze(destination: Path, run_id: str) -> tuple[dict, dict, dict]:
    specifications = {name: json.loads((RUNS / info[0] / 'specification.json').read_bytes())
                      for name, info in SOURCES.items()}
    inputs, candidate_ids = {}, []
    for name, (relative, count, selected) in SOURCES.items():
        spec = specifications[name]
        if (spec['candidateCount'] != count or len(spec['candidates']) != count
                or len(set(spec['candidates'])) != count or spec['excellenceRules'] != EXCELLENCE_RULES
                or spec['outerStartYear'] != OUTER_START_YEAR):
            raise ValueError('来源候选数、优秀门槛或评估起点与预先约定不一致。')
        candidate_ids.extend(f'{name}/{candidate}' for candidate in spec['candidates'])
        names = ['candidate-predictions.csv', selected, 'specification.json', 'provenance.json',
                 'report.json', 'factors.csv', 'labels.csv',
                 *[f'reproduction/data/{file}' for file in ('prices.csv', 'universe.json', 'manifest.json', 'review.json')],
                 *[f'reproduction/quant/{file}' for file in ('lab_evaluation.py', 'portfolio.py', 'metrics.py')]]
        inputs[name] = {file: _digest(RUNS / relative / file) for file in names}
    source_hashes = {str(path.relative_to(ROOT)): _digest(path) for path in _dependencies()}
    specification = {
        'experimentId': 'fixed_probability_combination_56', 'frozenAt': datetime.now(timezone.utc).isoformat(),
        'sourceRuns': {name: value[0] for name, value in SOURCES.items()}, 'candidateIds': candidate_ids,
        'candidateCount': 56, 'policies': ['ADAPTIVE', 'STATIC'], 'primaryPolicy': 'ADAPTIVE',
        'policySelection': False, 'candidateRefitting': False, 'modelsRefitted': 0,
        'prior': {'OLD': {'totalMass': 0.5, 'candidateCount': 40}, 'NEW': {'totalMass': 0.5, 'candidateCount': 16}},
        'staticRule': '每个系统总先验质量0.5，系统内部候选等权；组合为先验加权概率之和。',
        'adaptive': {'lookbackYears': INNER_YEARS, 'lookbackDate': 'historical signal date',
                     'maturityRule': 'historical label_end strictly before current signal date',
                     'decayAge': 'number of actual XHKG sessions from historical label_end to current signal date',
                     'halfLifeTradingDays': HALF_LIFE, 'observationWeight': OBSERVATION_WEIGHT,
                     'loss': 'sum((5/21)*2**(-age/252)*(candidate_probability-(target>0))**2)',
                     'lossNormalized': False, 'eta': ETA,
                     'posterior': 'prior*exp(-eta*cumulative_weighted_loss), normalized across all 56 candidates',
                     'finalWeight': '0.5*prior+0.5*posterior', 'priorShrinkage': PRIOR_SHRINKAGE,
                     'noMaturedObservations': 'empty sum equals zero; the same formula yields the prior',
                     'trainingStocks': 'only the predicted stock; no other stock outcomes enter its weights'},
        'outerStartYear': OUTER_START_YEAR, 'targets': TARGETS,
        'forecastUse': 'Use saved candidate OOF probabilities, including their original unavailable future labels; do not refit or substitute latest full-history models.',
        'comparisons': ['OLD selected annual system', 'NEW ALL selected annual system'],
        'comparisonUse': 'background comparison on identical dates only; selected strategies never enter weighting or policy choice',
        'validation': 'exact same candidate dates, labels, baseline probabilities and target volatility; missing source rows stop the experiment',
        'excellenceRules': EXCELLENCE_RULES, 'independentFutureObservationRequired': True,
        'independentValidation': False, 'liveReady': False,
        'researchNote': '来源历史均已研究，本次只冻结两种组合结构；全部结果仍是探索性历史比较，不能保证65%命中率或宣称独立验收。',
        'sourceHashes': source_hashes, 'inputHashes': inputs,
    }
    provenance = {'id': run_id, 'createdAt': specification['frozenAt'], 'sourceHashes': source_hashes,
                  'inputHashes': inputs, 'python': sys.version,
                  'runtime': {name: importlib.metadata.version(name) for name in
                              ('numpy', 'pandas', 'scipy', 'scikit-learn', 'exchange-calendars', 'joblib', 'threadpoolctl')},
                  'independentValidation': False, 'liveReady': False}
    _write(destination / 'specification.json', specification)
    _write(destination / 'provenance.json', provenance)
    for relative, digest in source_hashes.items():
        target = destination / 'reproduction' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if _digest(target) != digest:
            raise ValueError('冻结依赖复制期间发生变化。')
    for name, mapping in inputs.items():
        for relative, digest in mapping.items():
            target = destination / 'reproduction' / 'inputs' / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNS / SOURCES[name][0] / relative, target)
            if _digest(target) != digest:
                raise ValueError('冻结预测输入复制期间发生变化。')
    return specification, provenance, specifications


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'symbol': str}, float_precision='round_trip')
    for name in ('date', 'label_start', 'label_end', 'fit_date', 'training_last_label_end'):
        frame[name] = pd.to_datetime(frame[name], errors='raise')
    frame = frame.loc[frame.symbol.isin(TARGETS)].copy()
    values = frame[['probability', 'base_probability']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f'来源概率不是有限的0至1数值：{path}')
    if not ((frame.training_last_label_end < frame.fit_date) & (frame.fit_date <= frame.date)).all():
        raise ValueError(f'来源预测的模型训练成熟边界不正确：{path}')
    return frame


def _same(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], context: str) -> None:
    left, right = left[columns].reset_index(drop=True), right[columns].reset_index(drop=True)
    if left.equals(right):
        return
    keys = ['symbol', 'date']
    lkeys, rkeys = set(map(tuple, left[keys].to_numpy())), set(map(tuple, right[keys].to_numpy()))
    difference = {'context': context, 'leftRows': len(left), 'rightRows': len(right),
                  'leftOnlyDates': [(s, _iso(d)) for s, d in sorted(lkeys-rkeys)],
                  'rightOnlyDates': [(s, _iso(d)) for s, d in sorted(rkeys-lkeys)], 'valueDifferences': []}
    merged = left.merge(right, on=keys, suffixes=('_left', '_right'), validate='one_to_one')
    for column in set(columns)-set(keys):
        a, b = merged[f'{column}_left'], merged[f'{column}_right']
        unequal = ~((a == b) | (a.isna() & b.isna()))
        for index in merged.index[unequal]:
            difference['valueDifferences'].append({'symbol': merged.at[index, 'symbol'],
                                                  'date': _iso(merged.at[index, 'date']), 'column': column,
                                                  'left': str(a.at[index]), 'right': str(b.at[index])})
    raise ValueError(json.dumps(difference, ensure_ascii=False))


def _load_and_audit(destination: Path, specifications: dict) -> tuple:
    frames, controls, matrices, references, volatility = {}, {}, [], {}, {}
    inputs = destination / 'reproduction' / 'inputs'
    for name, (_, _, selected) in SOURCES.items():
        frame = _read_predictions(inputs / name / 'candidate-predictions.csv')
        candidates = specifications[name]['candidates']
        if set(frame.candidate) != set(candidates) or frame.duplicated(['symbol', 'date', 'candidate']).any():
            raise ValueError('来源四股候选集合缺失或存在重复预测。')
        reference = frame.loc[frame.candidate == candidates[0]].sort_values(['symbol', 'date']).reset_index(drop=True)
        for candidate in candidates:
            rows = frame.loc[frame.candidate == candidate].sort_values(['symbol', 'date'])
            _same(reference, rows, MATCH_COLUMNS, f'{name}/{candidate}')
        control = _read_predictions(inputs / name / selected).sort_values(['symbol', 'date'])
        if control.duplicated(['symbol', 'date']).any():
            raise ValueError('来源已选系统存在重复股票日期。')
        outer_reference = reference.loc[reference.date.dt.year >= OUTER_START_YEAR]
        _same(outer_reference, control, MATCH_COLUMNS, f'{name}/selected date coverage')
        selected_source = frame.merge(control[['symbol', 'date', 'candidate']], on=['symbol', 'date', 'candidate'], validate='one_to_one').sort_values(['symbol', 'date'])
        _same(control, selected_source, MATCH_COLUMNS+['probability'], f'{name}/selected probability origin')
        frame['candidate'] = name + '/' + frame.candidate
        frames[name], references[name], controls[name] = frame, reference, control
        matrices.append(frame.pivot(index=['symbol', 'date'], columns='candidate', values='probability'))
        vol = pd.read_csv(inputs / name / 'factors.csv', usecols=['symbol', 'date', 'annualVolatility'],
                          dtype={'symbol': str}, parse_dates=['date'], float_precision='round_trip')
        volatility[name] = reference[['symbol', 'date']].merge(vol, on=['symbol', 'date'], validate='one_to_one').sort_values(['symbol', 'date'])
        _same(reference, volatility[name], ['symbol', 'date'], f'{name}/volatility coverage')
        for file in ('lab_evaluation.py', 'portfolio.py', 'metrics.py'):
            if _digest(ROOT / 'quant' / file) != _digest(inputs / name / 'reproduction/quant' / file):
                raise ValueError(f'当前评估代码与来源冻结代码不一致：{name}/{file}')
    _same(references['OLD'], references['NEW'], MATCH_COLUMNS, 'OLD versus NEW common OOF')
    _same(volatility['OLD'], volatility['NEW'], ['symbol', 'date', 'annualVolatility'], 'OLD versus NEW volatility')
    for file in ('prices.csv', 'universe.json', 'manifest.json', 'review.json'):
        if _digest(inputs / 'OLD/reproduction/data' / file) != _digest(inputs / 'NEW/reproduction/data' / file):
            raise ValueError(f'两源交易原始输入不一致：{file}')
    reference = references['OLD'][MATCH_COLUMNS].copy()
    matrix = pd.concat(matrices, axis=1).loc[pd.MultiIndex.from_frame(reference[['symbol', 'date']])]
    candidate_ids = [f'{name}/{candidate}' for name in SOURCES for candidate in specifications[name]['candidates']]
    matrix = matrix[candidate_ids]
    if matrix.isna().any().any() or len(matrix.columns) != 56:
        raise ValueError('组合矩阵必须每个股票日期完整包含56个真实候选概率。')
    calendar = research_calendar()
    if not reference.date.isin(calendar).all():
        raise ValueError('来源信号含非真实港股交易日。')
    known = reference.target.notna()
    if not ((reference.loc[known, 'date'] < reference.loc[known, 'label_start'])
            & (reference.loc[known, 'label_start'] < reference.loc[known, 'label_end'])
            & (reference.loc[known, 'label_end'] <= calendar[-1])).all():
        raise ValueError('来源成熟标签区间不正确。')
    audit = {'passed': True, 'strictCandidateDateLabelBaselineEquality': True, 'sameSelectedSystemDates': True,
             'sameOriginalMarketData': True, 'sameSignalVolatility': True, 'candidateCount': 56,
             'stocks': {symbol: {'allOOFRows': len(group), 'firstSignal': _iso(group.date.min()),
                                 'lastSignal': _iso(group.date.max()),
                                 'outerRows': int((group.date.dt.year >= OUTER_START_YEAR).sum())}
                        for symbol, group in reference.groupby('symbol')}}
    _write(destination / 'input-audit.json', audit)
    return reference, matrix, controls, calendar, volatility['OLD'], audit


def _combine(reference: pd.DataFrame, matrix: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple:
    ids = list(matrix.columns)
    prior = np.asarray([0.5/SOURCES[name.split('/')[0]][1] for name in ids])
    if not np.isclose(prior.sum(), 1, rtol=0, atol=1e-14):
        raise ValueError('组合先验质量不为一。')
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    outputs, weights, audits = [], [], []
    for symbol, stock in reference.groupby('symbol', sort=False):
        stock = stock.sort_values('date').reset_index(drop=True)
        probabilities = matrix.loc[symbol].loc[stock.date].to_numpy(dtype=float)
        for index, row in enumerate(stock.itertuples(index=False)):
            cutoff = row.date-pd.DateOffset(years=INNER_YEARS)
            mature = (stock.date >= cutoff) & (stock.label_end < row.date) & stock.target.notna()
            history = stock.loc[mature]
            ages = positions.loc[row.date]-positions.reindex(history.label_end).to_numpy(dtype=float)
            if not np.isfinite(ages).all() or (ages <= 0).any():
                raise ValueError('成熟损失的交易日年龄无效或读取了尚未成熟标签。')
            observation_weights = OBSERVATION_WEIGHT * np.exp2(-ages/HALF_LIFE)
            errors = (probabilities[mature.to_numpy()]-(history.target.to_numpy() > 0)[:, None])**2
            losses = np.sum(observation_weights[:, None]*errors, axis=0)
            posterior = prior*np.exp(-ETA*losses)
            posterior /= posterior.sum()
            adaptive = PRIOR_SHRINKAGE*prior+(1-PRIOR_SHRINKAGE)*posterior
            static_probability = float(np.dot(prior, probabilities[index]))
            adaptive_probability = float(np.dot(adaptive, probabilities[index]))
            if not np.isclose(adaptive.sum(), 1, rtol=0, atol=1e-14) or not np.isfinite(adaptive).all():
                raise ValueError('组合权重无效。')
            base = row._asdict()
            for policy, probability in [('STATIC', static_probability), ('ADAPTIVE', adaptive_probability)]:
                outputs.append({**base, 'policy': policy, 'probability': probability})
            for candidate, p, loss, q, w in zip(ids, prior, losses, posterior, adaptive, strict=True):
                weights.append({'symbol': symbol, 'date': row.date, 'candidate': candidate, 'prior': float(p),
                                'cumulativeWeightedLoss': float(loss), 'posterior': float(q),
                                'STATIC': float(p), 'ADAPTIVE': float(w)})
            audits.append({'symbol': symbol, 'date': _iso(row.date), 'historyWindowStartInclusive': _iso(cutoff),
                           'labelEndCutoffExclusive': _iso(row.date), 'maturedForecastCount': len(history),
                           'weightedMaturedMass': float(observation_weights.sum()),
                           'lastMaturedSignal': _iso(history.date.max()), 'lastMaturedLabelEnd': _iso(history.label_end.max()),
                           'oldSystemAdaptiveWeight': float(adaptive[:40].sum()),
                           'newSystemAdaptiveWeight': float(adaptive[40:].sum()),
                           'informationRulePassed': True,
                           'effectiveCountNote': '权重总量是重叠校正及时间衰减后的损失质量，不是统计独立样本数。'})
    return pd.DataFrame(outputs), pd.DataFrame(weights), audits


def _evaluate(rows, market, symbol, calendar, volatility) -> dict:
    end = calendar[-1]
    matured = rows.loc[rows.target.notna() & (rows.label_end <= end)]
    metrics = forecast_metrics(matured)
    recent = forecast_metrics(matured.loc[matured.date >= pd.Timestamp('2025-01-01')])
    yearly = []
    for year, group in matured.groupby(matured.date.dt.year):
        year_calendar = calendar[calendar.year == year]
        complete = year < end.year and group.date.min() <= year_calendar[4] and group.date.max() >= year_calendar[-6]
        yearly.append({'year': int(year), 'completeCalendarYear': bool(complete), 'metrics': forecast_metrics(group)})
    high_rows = matured.loc[np.maximum(matured.probability, 1-matured.probability) >= 0.65]
    high = forecast_metrics(high_rows, compute_bootstrap=False) if not high_rows.empty else None
    trading = trading_metrics(market, rows, symbol, rows.date.min(), end, volatility)
    checks = _excellent_checks(metrics, trading, yearly, high)
    return {'evaluation': metrics, 'recentEvaluation': recent, 'yearlyEvaluation': yearly,
            'highConfidenceEvaluation': high, 'trading': trading, 'checks': checks,
            'historicalExcellent': all(check['passed'] for check in checks),
            'independentValidation': False, 'independentFutureObservationPassed': False, 'liveReady': False}


def execute() -> dict:
    folder = RUNS / 'ensemble'
    folder.mkdir(exist_ok=True)
    with (folder / '.lock').open('a') as guard, threadpool_limits(limits=1):
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = folder / run_id
        destination.mkdir()
        specification, provenance, originals = _freeze(destination, run_id)
        print(f'已在结果计算前冻结两个固定政策：{run_id}', flush=True)
        try:
            reference, matrix, controls, calendar, volatility, audit = _load_and_audit(destination, originals)
        except Exception as exc:
            failure = {'status': 'input_validation_blocked', 'error': str(exc), 'evaluationPerformed': False,
                       'specification': specification, 'independentValidation': False, 'liveReady': False}
            _write(destination / 'report.json', failure)
            print(json.dumps({'runId': run_id, **failure}, ensure_ascii=False), flush=True)
            return failure
        print('四股56个候选日期、标签、涨频基准及交易输入严格核对通过', flush=True)
        predictions, weights, weight_audit = _combine(reference, matrix, calendar)
        predictions.to_csv(destination / 'all-combination-predictions.csv', index=False, date_format='%Y-%m-%d')
        weights.to_csv(destination / 'candidate-weights.csv', index=False, date_format='%Y-%m-%d')
        _write(destination / 'weight-audit.json', {'signals': weight_audit, 'candidateWeightsPerSignal': 56})
        outer = predictions.loc[predictions.date.dt.year >= OUTER_START_YEAR]
        outer.to_csv(destination / 'outer-predictions.csv', index=False, date_format='%Y-%m-%d')
        inputs = destination / 'reproduction/inputs/OLD'
        prices, universe, _ = _read_inputs(inputs / 'reproduction/data')
        securities = universe['stocks']+[universe['benchmark']]
        market = Market.from_prices(prices, calendar, universe['benchmark']['symbol'], universe['corporateActions'],
                                    {item['symbol']: item['stampDutyExemptFrom'] for item in securities})
        reports = []
        for symbol in TARGETS:
            print(f'{symbol}：按原15条历史门槛核算固定组合及两源背景对照', flush=True)
            background = {name: _evaluate(frame.loc[frame.symbol == symbol].sort_values('date'), market, symbol, calendar, volatility)
                          for name, frame in controls.items()}
            policies = {}
            for policy in ('ADAPTIVE', 'STATIC'):
                rows = outer.loc[(outer.symbol == symbol) & (outer.policy == policy)].sort_values('date')
                _same(rows, controls['OLD'].loc[controls['OLD'].symbol == symbol], MATCH_COLUMNS, f'{symbol}/{policy}/outer coverage')
                result = _evaluate(rows, market, symbol, calendar, volatility)
                latest = rows.iloc[-1]
                result['latestSavedOOFSignal'] = {'asOf': _iso(latest.date), 'upProbability': float(latest.probability),
                                                'generatedAt': datetime.now(timezone.utc).isoformat(),
                                                'futureOutcomeObserved': False, 'prospectiveAtIssue': False,
                                                'interpretation': '较早信息截面的已存OOF组合；并非当时提前发出的预测，也不是新重训模型。'}
                policies[policy] = result
            reports.append({'symbol': symbol, 'primaryPolicy': 'ADAPTIVE', 'policies': policies,
                            'selectedSourceBackgroundOnly': background, 'sameForecastDates': True})
        for relative, digest in provenance['sourceHashes'].items():
            if _digest(ROOT / relative) != digest:
                raise ValueError('组合研究期间实际依赖发生变化。')
        for name, mapping in provenance['inputHashes'].items():
            for relative, digest in mapping.items():
                if _digest(RUNS / SOURCES[name][0] / relative) != digest:
                    raise ValueError('组合研究期间原预测源发生变化。')
        report = {'schemaVersion': 1, 'status': 'complete', 'run': {**provenance, 'completedAt': datetime.now(timezone.utc).isoformat()},
                  'specification': specification, 'inputAudit': audit, 'stockReports': reports,
                  'summary': {'primaryPolicy': 'ADAPTIVE', 'policiesEvaluated': 2, 'policySelection': False,
                              'historicalExcellentByPolicy': {policy: sum(row['policies'][policy]['historicalExcellent'] for row in reports)
                                                              for policy in ('ADAPTIVE', 'STATIC')},
                              'modelsRefitted': 0, 'independentValidation': False, 'liveReady': False}}
        _write(destination / 'report.json', report)
        _write(folder / 'latest.json', {'id': run_id})
        print(json.dumps({'runId': run_id, **report['summary']}, ensure_ascii=False), flush=True)
        return report


if __name__ == '__main__':
    execute()
