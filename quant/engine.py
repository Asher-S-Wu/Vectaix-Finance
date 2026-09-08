from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import complete_month_ends, research_calendar, verify_snapshot
from .factors import FACTOR_KEYS, FACTOR_NAMES, build_factors, build_labels
from .metrics import performance, period_returns
from .portfolio import INITIAL_CAPITAL, SLIPPAGE, Market, simulate


DEVELOPMENT_START = pd.Timestamp("2017-01-01")
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
ALPHAS = (10.0, 100.0, 1000.0)
WINDOW_YEARS = 3
MIN_TRAINING_SAMPLES = 504
RETURN_HURDLE = 0.005
TARGET_VOLATILITY = 0.15


def _iso(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _read_inputs(data_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    manifest = verify_snapshot(data_dir)
    required = {"date", "symbol", "open", "high", "low", "close", "adjusted_open", "adjusted_close", "volume", "turnover"}
    prices = pd.read_csv(data_dir / "prices.csv", dtype={"symbol": str}, parse_dates=["date"])
    if not required.issubset(prices.columns):
        raise ValueError(f"行情缺少字段：{sorted(required - set(prices.columns))}")
    if prices[list(required)].isna().any().any():
        raise ValueError("行情包含空值，必须在数据源端明确解决后再运行。")
    if prices.duplicated(["date", "symbol"]).any():
        raise ValueError("同一股票同一天出现重复行情。")
    numeric_columns = sorted(required - {"date", "symbol"})
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="raise")
        if not np.isfinite(prices[column].to_numpy()).all():
            raise ValueError(f"行情字段 {column} 存在无效数字。")
    if (prices[["open", "high", "low", "close", "adjusted_open", "adjusted_close"]] <= 0).any().any():
        raise ValueError("原始价格及复权开盘、收盘价必须全部大于零。")
    if (prices[["volume", "turnover"]] < 0).any().any():
        raise ValueError("成交量和成交额不能为负。")
    invalid_ohlc = (
        (prices.high < prices[["open", "close", "low"]].max(axis=1) - 1e-6)
        | (prices.low > prices[["open", "close", "high"]].min(axis=1) + 1e-6)
    )
    if invalid_ohlc.any():
        raise ValueError(f"有 {int(invalid_ohlc.sum())} 行原始开高低收价格矛盾。")
    universe = json.loads((data_dir / "universe.json").read_text(encoding="utf-8"))
    calendar = research_calendar()
    if not prices.date.isin(calendar).all():
        raise ValueError("已封存行情包含非 XHKG 正式交易日记录。")
    stocks = universe["stocks"]
    symbols = [item["symbol"] for item in stocks]
    if len(symbols) != len(set(symbols)) or len(symbols) != 4:
        raise ValueError("股票池必须包含本轮指定的四个互不重复证券。")
    benchmark = universe["benchmark"]["symbol"]
    expected = set(symbols) | {benchmark}
    if set(prices.symbol) != expected:
        raise ValueError("行情中的证券与 universe.json 股票池不一致。")
    for item in stocks:
        if not item["name"] or not item["sector"]:
            raise ValueError("股票池必须提供每只证券的名称及行业。")
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    for action in universe["corporateActions"]:
        if action["type"] != "cash_delisting":
            raise ValueError("尚未实现该类公司行动，不能继续模拟。")
        effective = pd.Timestamp(action["effectiveDate"])
        payment = pd.Timestamp(action["paymentDate"])
        if payment < effective or action["cashPerShare"] <= 0:
            raise ValueError("退市现金方案的日期或支付金额无效。")
        reference = prices.loc[
            (prices.symbol == action["symbol"]) & (prices.date < effective) & (prices.volume > 0)
        ]
        if reference.empty:
            raise ValueError("现金退市证券缺少生效前有效行情。")
        final = reference.iloc[-1]
        action["adjustedCashPerUnit"] = float(action["cashPerShare"] * final.adjusted_close / final.close)
        action["referenceDate"] = _iso(final.date)
        prices = prices.loc[~((prices.symbol == action["symbol"]) & (prices.date >= effective))].copy()
    prices["research_close"] = prices.adjusted_close
    prices["research_open"] = prices.adjusted_open
    return prices, universe, manifest


def _training(labels: pd.DataFrame, signal: pd.Timestamp) -> pd.DataFrame:
    return labels.loc[(labels.date >= signal - pd.DateOffset(years=WINDOW_YEARS)) & (labels.label_end < signal)]


def _fit_predictions(factors: pd.DataFrame, labels: pd.DataFrame, dates: list[pd.Timestamp], alpha: float):
    predictions, coefficients = [], []
    for signal in dates:
        current = factors.loc[factors.date == signal]
        if len(current) != 1:
            raise ValueError(f'{_iso(signal)} 指定股票没有唯一的有效月末因子，停止研究。')
        train = _training(labels, signal)
        if len(train) < MIN_TRAINING_SAMPLES:
            raise ValueError(f'{_iso(signal)} 成熟训练样本不足 {MIN_TRAINING_SAMPLES} 条。')
        # Each scaler is fitted on this stock's matured training window only.
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[list(FACTOR_KEYS)], train.target)
        score = float(model.predict(current[list(FACTOR_KEYS)])[0])
        if not np.isfinite(score):
            raise ValueError('模型预测不是有限数字。')
        volatility = float(current.annualVolatility.iloc[0])
        if volatility <= 0:
            raise ValueError('观测波动率必须大于零。')
        weight = min(1.0, TARGET_VOLATILITY / volatility) if score > RETURN_HURDLE else 0.0
        predictions.append({'date': signal, 'symbol': str(current.symbol.iloc[0]), 'score': score, 'targetWeight': weight})
        ridge = model.named_steps['ridge']
        training = {
            'firstLabelDate': _iso(train.date.min()), 'lastMaturedLabelEnd': _iso(train.label_end.max()),
            'samples': len(train), 'months': int(train.date.dt.to_period('M').nunique()),
            'refitFrequency': '每月，逐只运行', 'target': '次日开盘起未来 21 个交易日的复权收益',
        }
        coefficients.append({'date': _iso(signal), 'alpha': alpha, 'windowYears': WINDOW_YEARS,
                             **training, 'intercept': float(ridge.intercept_),
                             **{key: float(value) for key, value in zip(FACTOR_KEYS, ridge.coef_)}})
    if not predictions:
        raise ValueError('没有可训练的月份。')
    return pd.DataFrame(predictions), coefficients, model, training


def _signals(predictions: pd.DataFrame) -> dict:
    return {row.date: {row.symbol: row.targetWeight} for row in predictions.itertuples()}


def _equity_rows(frame: pd.DataFrame) -> list[dict]:
    return [{'date': _iso(date), **{key: float(value) for key, value in row.items()}} for date, row in frame.iterrows()]


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 10 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = float(left.rank().corr(right.rank()))
    return value if np.isfinite(value) else None


def _candidate_key(candidate: dict):
    sharpe = candidate['metrics']['sharpe']
    # All-cash series have undefined Sharpe and sort after measurable candidates.
    return (sharpe is None, -sharpe if sharpe is not None else 0, -candidate['metrics']['cagr'], candidate['id'])


def run_research(data_dir: Path, output_dir: Path, progress: Callable[[str, float], None]) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, universe, manifest = _read_inputs(data_dir)
    stocks = universe['stocks']
    benchmark_symbol = universe['benchmark']['symbol']
    factors, calendar = build_factors(prices, [s['symbol'] for s in stocks], benchmark_symbol)
    labels = build_labels(factors, prices, calendar)
    months = complete_month_ends(calendar)
    dev_end = calendar[calendar <= DEVELOPMENT_END][-1]
    end = calendar[-1]
    if end < pd.Timestamp('2025-12-31'):
        raise ValueError('独立验收至少需要覆盖 2024 和 2025 两个完整年度。')
    market = Market.from_prices(prices, calendar, benchmark_symbol, universe['corporateActions'], {
        s['symbol']: s['stampDutyExemptFrom'] for s in stocks + [universe['benchmark']]
    })
    specification = {
        'model': '逐股时间序列标准化 Ridge；不进行四股横截面排序',
        'alphas': ALPHAS, 'windowYears': WINDOW_YEARS, 'factorKeys': FACTOR_KEYS,
        'labelTradingDays': 21, 'minimumTrainingSamples': MIN_TRAINING_SAMPLES,
        'returnHurdle': RETURN_HURDLE, 'annualVolatilityTarget': TARGET_VOLATILITY,
        'maximumStockWeight': 1.0, 'cashInterest': 0, 'workers': 1,
        'earliestDevelopmentStart': _iso(DEVELOPMENT_START), 'developmentEnd': _iso(dev_end),
        'holdoutStart': _iso(HOLDOUT_START),
        'criterion': '每股分别选开发期扣费后日夏普最高者；同分按年化收益和编号排序；全现金夏普无定义，排在有定义候选之后。',
        'acceptance': {'holdoutSharpeMinimum': 0.7, 'annualExcessOverSameStockMinimumExclusive': 0,
                       'holdoutMaxDrawdownMinimum': -0.25, 'doubleSlippageExcessMinimumExclusive': 0,
                       'minimumHoldoutMonths': 24, 'futureObservationRequired': True},
    }
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    selections, prepared, all_candidates = [], [], []
    candidate_equities, candidate_coefficients = [], []
    # Freeze every stock's development winner before opening any stock holdout.
    for stock_index, stock in enumerate(stocks):
        symbol = stock['symbol']
        progress(f'{stock_index + 1}/4 · {stock["name"]}：开发期训练', 0.05 + stock_index * 0.12)
        sf, sl = factors.loc[factors.symbol == symbol], labels.loc[labels.symbol == symbol]
        eligible_starts = [d for d in months if d >= pd.Timestamp('2016-12-01') and d <= dev_end
                           and d in set(sf.date) and len(_training(sl, d)) >= MIN_TRAINING_SAMPLES]
        if not eligible_starts:
            raise ValueError(f'{symbol} 无足量历史训练区间。')
        anchor = eligible_starts[0]
        if (dev_end - anchor).days < 540:
            raise ValueError(f'{symbol} 开发期短于 18 个月，不能选择模型。')
        dates = [pd.Timestamp(d) for d in months if anchor <= d <= end]
        missing = sorted(set(dates) - set(sf.date))
        if missing:
            raise ValueError(f'{symbol} 缺少完整月末因子：{[_iso(d) for d in missing]}')
        dev_dates = [d for d in dates if d <= dev_end]
        dev_benchmark = simulate(market, {anchor: {benchmark_symbol: 1.0}}, anchor, dev_end)
        candidates = []
        for alpha in ALPHAS:
            predictions, coefficients, _, _ = _fit_predictions(sf, sl, dev_dates, alpha)
            simulation = simulate(market, _signals(predictions), anchor, dev_end)
            metrics = performance(simulation.equity, dev_benchmark.equity)
            joined = predictions.merge(sl.loc[sl.label_end <= dev_end, ['date', 'target']], on='date', validate='one_to_one')
            annual = period_returns(pd.DataFrame({'strategy': simulation.equity, 'benchmark': dev_benchmark.equity}), 'Y')
            annual = [r for r in annual if r['year'] >= calendar[calendar.get_loc(anchor) + 1].year]
            candidate_id = f'{symbol}-ridge-a{int(alpha)}'
            candidate = {'id': candidate_id, 'symbol': symbol, 'alpha': alpha, 'windowYears': WINDOW_YEARS,
                         'topN': 1, 'selected': False, 'score': metrics['sharpe'], 'metrics': metrics,
                         'positiveYears': sum(r['excess'] > 0 for r in annual), 'totalYears': len(annual),
                         'meanIC': _correlation(joined.score, joined.target), 'annualReturns': annual,
                         'fees': simulation.total_fees, 'slippageCost': simulation.slippage_cost}
            candidates.append(candidate)
            candidate_equities.append(simulation.daily_accounting.assign(symbol=symbol, candidate=candidate_id))
            candidate_coefficients.extend({'symbol': symbol, 'candidate': candidate_id, **r} for r in coefficients)
        candidates.sort(key=_candidate_key)
        winner = candidates[0]
        winner['selected'] = True
        selection = {'symbol': symbol, 'criterion': specification['criterion'], 'selectedId': winner['id'],
                     'alpha': winner['alpha'], 'lockedAt': datetime.now(timezone.utc).isoformat(),
                     'developmentStart': _iso(calendar[calendar.get_loc(anchor) + 1]), 'developmentEnd': _iso(dev_end),
                     'holdoutStart': _iso(HOLDOUT_START), 'evaluatedCandidates': len(candidates), 'specificationHash': spec_hash}
        selections.append(selection)
        all_candidates.extend(candidates)
        prepared.append((stock, sf, sl, anchor, dates, candidates, selection))
    (output_dir / 'selection.json').write_text(json.dumps({'specification': specification, 'selections': selections}, ensure_ascii=False, indent=2, allow_nan=False))
    with (output_dir / 'holdout-opened.json').open('x') as marker:
        json.dump({'specificationHash': spec_hash, 'datasetSha256': manifest['sha256'],
                   'openedAt': datetime.now(timezone.utc).isoformat()}, marker)
    reports, exports, final_models, coefficient_exports, equity_exports = [], [], {}, [], []
    for index, (stock, sf, sl, anchor, dates, candidates, selection) in enumerate(prepared):
        symbol = stock['symbol']
        progress(f'{index + 1}/4 · {stock["name"]}：固定参数独立验收', 0.55 + index * 0.1)
        predictions, coefficients, model, training = _fit_predictions(sf, sl, dates, selection['alpha'])
        signals = _signals(predictions)
        strategy = simulate(market, signals, anchor, end, record_details=True)
        baseline = simulate(market, {anchor: {symbol: 1.0}}, anchor, end, record_details=True)
        benchmark = simulate(market, {anchor: {benchmark_symbol: 1.0}}, anchor, end, record_details=True)
        stress = simulate(market, signals, anchor, end, slippage=SLIPPAGE * 2, record_details=True)
        equity = pd.DataFrame({'strategy': strategy.equity, 'baseline': baseline.equity,
                               'benchmark': benchmark.equity, 'stress': stress.equity})
        if equity.isna().any().any():
            raise ValueError('策略和对照净值没有严格对齐。')
        dev, holdout = equity.loc[:dev_end], equity.loc[dev_end:]
        dm, hm = performance(dev.strategy, dev.benchmark), performance(holdout.strategy, holdout.benchmark)
        bm, sm = performance(holdout.baseline, holdout.benchmark), performance(holdout.stress, holdout.benchmark)
        same_stock_excess, stressed_excess = hm['cagr'] - bm['cagr'], sm['cagr'] - bm['cagr']
        holdout_months = len(holdout.loc[holdout.index >= HOLDOUT_START].index.to_period('M').unique())
        checks = [
            {'key': 'holdout_sharpe', 'label': '验收期净夏普至少 0.70', 'passed': hm['sharpe'] is not None and hm['sharpe'] >= .7, 'value': hm['sharpe'], 'threshold': .7},
            {'key': 'same_stock_excess', 'label': '年化收益超过长期持有本股', 'passed': same_stock_excess > 0, 'value': same_stock_excess, 'threshold': 0},
            {'key': 'drawdown', 'label': '验收期最大回撤不超过 25%', 'passed': hm['maxDrawdown'] >= -.25, 'value': hm['maxDrawdown'], 'threshold': -.25},
            {'key': 'stress', 'label': '双倍滑点后仍超过长期持有本股', 'passed': stressed_excess > 0, 'value': stressed_excess, 'threshold': 0},
            {'key': 'history', 'label': '独立验收至少 24 个月', 'passed': holdout_months >= 24, 'value': holdout_months, 'threshold': 24},
            {'key': 'accounting', 'label': '每日现金与持仓账目一致', 'passed': strategy.max_identity_residual <= 1e-5, 'value': strategy.max_identity_residual, 'threshold': 1e-5},
            {'key': 'future_observation', 'label': '完成模型建立后的未来实测', 'passed': False, 'value': '尚未开展', 'threshold': '真实未来的独立观察记录'},
        ]
        passed = all(c['passed'] for c in checks if c['key'] != 'future_observation')
        last = predictions.iloc[-1]
        latest_holdings = [{**r, 'name': stock['name'], 'sector': stock['sector']} for r in strategy.holdings if r['date'] == _iso(end)]
        matured = sl.loc[(sl.date >= anchor) & (sl.label_end <= dev_end)]
        normalized = equity / INITIAL_CAPITAL * 100
        raw_rows = prices.loc[prices.symbol == symbol].sort_values('date')
        report = {
            'schemaVersion': 2, 'stock': stock,
            'summary': {'status': 'research_pass' if passed else 'research_failed',
                        'statusLabel': '历史指标达标，尚未完成未来实测' if passed else '未通过预定验收',
                        'statisticalPass': passed, 'liveReady': False, 'selectedCandidate': selection['selectedId'],
                        'initialCapital': INITIAL_CAPITAL, 'asOf': _iso(end), 'dataStart': _iso(raw_rows.date.min()),
                        'dataEnd': _iso(raw_rows.date.max()), 'finalEquity': float(strategy.equity.iloc[-1]),
                        'cash': float(strategy.daily_accounting.cash.iloc[-1]),
                        'cashWeight': float(strategy.daily_accounting.cash.iloc[-1] / strategy.equity.iloc[-1]),
                        'latestRawPrice': float(raw_rows.close.iloc[-1]), 'sameStockAnnualExcess': same_stock_excess},
            'metrics': {'development': dm, 'holdout': hm, 'full': performance(equity.strategy, equity.benchmark),
                        'baseline': bm, 'benchmark': performance(holdout.benchmark), 'stress': sm},
            'development': {'start': selection['developmentStart'], 'end': _iso(dev_end), 'metrics': dm,
                            'baselineMetrics': performance(dev.baseline, dev.benchmark),
                            'benchmarkMetrics': performance(dev.benchmark), 'stressMetrics': performance(dev.stress, dev.benchmark)},
            'holdout': {'start': _iso(HOLDOUT_START), 'end': _iso(end), 'metrics': hm,
                        'baselineMetrics': bm, 'benchmarkMetrics': performance(holdout.benchmark), 'stressMetrics': sm},
            'selection': selection, 'candidates': candidates,
            'equity': _equity_rows(normalized.groupby(normalized.index.to_period('M')).tail(1)),
            'dailyEquity': _equity_rows(normalized),
            'factors': [{'key': key, 'name': FACTOR_NAMES[key], 'coefficient': coefficients[-1][key],
                         'meanIC': _correlation(matured[key], matured.target)} for key in FACTOR_KEYS],
            'coefficients': coefficients, 'training': training,
            'latestSignal': {'date': _iso(last.date), 'executionDate': None,
                             'nextExecution': '下一交易日开盘；该日不在本轮数据内，信号尚未执行',
                             'positions': [{'symbol': symbol, 'name': stock['name'], 'sector': stock['sector'],
                                            'score': float(last.score), 'targetWeight': float(last.targetWeight)}],
                             'cashWeight': 1 - float(last.targetWeight)},
            'holdings': latest_holdings, 'monthlyReturns': period_returns(equity, 'M'), 'annualReturns': period_returns(equity, 'Y'),
            'checks': checks,
            'dataQuality': {'source': manifest['provider'], 'staticUniverse': True, 'universeSize': 1,
                            'priceRows': len(raw_rows), 'calendarDays': len(calendar), 'missingSignalMonths': 0,
                            'accounting': '复权研究单位', 'staleMarks': strategy.stale_marks},
            'execution': {'totalFees': strategy.total_fees, 'slippageCost': strategy.slippage_cost,
                          'tradeCount': len(strategy.trades), 'maxAccountingResidual': strategy.max_identity_residual,
                          'blockedOrders': strategy.blocked_orders},
            'methodology': [
                '用户指定四只股票，分别训练自己的模型与报告；单股策略只持有该股票和现金，不比较四股横截面排名。截图中的股数和成本只供展示，不进入特征、训练标签或模型选择。',
                '使用八类价格与成交量因子；每个观测日只计算截至该日收盘的信息。原始股价至少 1 港元，60 日平均估算成交额至少 1000 万港元。特征在每次训练窗口内单独标准化。',
                '训练目标为次日开盘至 21 个交易日后开盘的含股息复权收益。采用过去三年的日样本，每次至少 504 条成熟标签，标签结束日期必须严格早于当月信号日。',
                '每股仅比较三个固定惩罚强度 10、100、1000。按开发期扣费后夏普选择；四股参数全部落盘锁定后，才打开 2024 年以后的共同验收区间。验收阶段允许按冻结规则逐月更新系数。',
                '月末预测 21 日收益超过 0.5% 才持股；仓位为 15% 年化波动目标除以过去 60 日年化波动，最多 100%，其余现金。没有做空、融资或现金利息。15% 是历史波动目标，不能保证未来波动。',
                '在真正月末收盘后生成信号，下一交易日开盘执行。每股独立用 100 万港元比较策略、长期持有本股、盈富基金和双倍滑点策略，使用相同起点和正式交易日历。',
                '逐笔收取按日期变化的印花税、交易所费用、SFC / AFRC 征费和交收费；经纪费万分之三且最低 3 港元。常规单边滑点 0.1%，压力为 0.2%。',
                '买卖和记账使用含股息的复权研究单位，不重复发放股息；按现金与持仓逐日核算。缺失及零成交量记录不可执行订单，已有持仓按最近实际成交价估值并记录。',
                '2024 年验收从 2023 年末收盘净值开始，保留当时持仓及首日调仓费用。同期长期持有本股是主要验收对照，盈富基金是额外市场参照。',
            ],
            'limitations': [
                '四只股票是现在按持仓指定的研究对象，存在事后选样偏差；结论只描述这些股票，不能代表全港股或各行业股票的普遍表现。',
                '21 日收益标签相互重叠，504 个日样本并不等于 504 次独立实验；开发期尤其药明康德较短，容易受单一市场阶段影响。',
                '这是历史资料中的隔离验收，并非模型建立后真实发生的未来检验。未达标就记录未达标，不反复用验收结果挑参数。',
                '仅含价格和成交量因素，尚未使用当时发布的财报、估值、盈利增长或油价等公司业务变量。',
                '全复权数据由 Qveris / FMP 提供，已做交易日、格式及关键除权核验；仍有供应商误差可能。没有逐笔成交、历史盘口和完整实际股息到账记录。',
                '复权研究单位可以细分；没有真实整手交易、账户税务和券商股息规则，不能直接当作截图账户的买卖股数。截图价格日期未知，模型行情截至 2026-08-31，不是实时买卖建议。',
            ],
        }
        reports.append(report)
        final_models[symbol] = {'model': model, 'selection': selection, 'factorKeys': FACTOR_KEYS, 'training': training}
        coefficient_exports.extend({'symbol': symbol, **row} for row in coefficients)
        accounting = strategy.daily_accounting.rename(columns={'equity': 'strategy'}).join(equity[['baseline', 'benchmark', 'stress']])
        equity_exports.append(accounting.assign(symbol=symbol))
        for label, simulation in (('strategy', strategy), ('baseline', baseline), ('benchmark', benchmark), ('stress', stress)):
            exports.append((symbol, label, simulation))
    pd.concat(equity_exports).to_csv(output_dir / 'equity.csv', index_label='date')
    pd.DataFrame([{'modelSymbol': symbol, 'portfolio': label, **r} for symbol, label, sim in exports for r in sim.trades]).to_csv(output_dir / 'trades.csv', index=False)
    pd.DataFrame([{'modelSymbol': symbol, 'portfolio': label, **r} for symbol, label, sim in exports for r in sim.holdings]).to_csv(output_dir / 'holdings.csv', index=False)
    pd.DataFrame([{key: c[key] for key in ('symbol', 'id', 'alpha', 'selected', 'meanIC')} | c['metrics'] for c in all_candidates]).to_csv(output_dir / 'candidates.csv', index=False)
    pd.DataFrame(coefficient_exports).to_csv(output_dir / 'coefficients.csv', index=False)
    pd.concat(candidate_equities).to_csv(output_dir / 'candidate_equity.csv', index_label='date')
    pd.DataFrame(candidate_coefficients).to_csv(output_dir / 'candidate_coefficients.csv', index=False)
    joblib.dump(final_models, output_dir / 'models.joblib')
    result = {'schemaVersion': 2, 'summary': {'stocks': len(reports), 'passed': sum(r['summary']['statisticalPass'] for r in reports),
                                           'liveReady': False, 'dataEnd': _iso(end)}, 'stockReports': reports}
    (output_dir / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    progress('四只股票已逐一完成研究，保存独立报告和账目', 1.0)
    return result
