from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .metrics import performance
from .portfolio import Market, SLIPPAGE, Simulation, simulate


BOOTSTRAP_BLOCK_LENGTH = 6
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_SEED = 42
CONFIDENCE_THRESHOLD = 0.65
SIGNAL_SPACING = 5
ENTRY_PROBABILITY = 0.55
TARGET_VOLATILITY = 0.15


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"评估数据缺少字段：{sorted(missing)}")
    if frame.empty:
        raise ValueError("评估数据不能为空。")


def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"{column} 不能包含缺失日期。")
    return values


def _numbers(frame: pd.DataFrame, column: str, *, probability: bool = False) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} 必须全部为有限数字。")
    if probability and ((values < 0) | (values > 1)).any():
        raise ValueError(f"{column} 必须在 0 至 1 之间。")
    return values


def _ordered_dates(dates: pd.Series) -> None:
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("评估数据必须按唯一预测日期从早到晚排列。")


def _bootstrap_intervals(actual: np.ndarray, hit: np.ndarray, brier_gain: np.ndarray) -> dict:
    count = len(actual)
    result = {
        "accuracyExcessAlwaysUp": None,
        "brierImprovement": None,
        "balancedAccuracy": None,
        "blockLength": BOOTSTRAP_BLOCK_LENGTH,
        "replications": BOOTSTRAP_REPLICATIONS,
        "seed": BOOTSTRAP_SEED,
        "available": count >= BOOTSTRAP_BLOCK_LENGTH,
        "balancedAccuracyValidReplications": 0,
        "method": "按日期顺序抽取连续六次预测组成的移动区块，不环绕；拼接后截取至原样本数。",
        "definition": "命中率差为模型减始终看涨；Brier 改善为涨频基准误差减模型误差；二者正数均表示模型更好。",
        "limitation": "六次预测约跨三十个交易日，仅近似保留重叠标签和短期依赖；不能消除反复查看历史、选型和多重比较偏差，不代表独立验收。",
        "balancedAccuracyNote": "平衡命中率只使用同时包含上涨和未上涨的重采样；有效次数另列，单类原样本不可用。",
    }
    if count < BOOTSTRAP_BLOCK_LENGTH:
        result["unavailableReason"] = "预测不足六次，不能按预定区块长度计算区间。"
        return result
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks_per_draw = (count + BOOTSTRAP_BLOCK_LENGTH - 1) // BOOTSTRAP_BLOCK_LENGTH
    starts = rng.integers(
        0, count - BOOTSTRAP_BLOCK_LENGTH + 1,
        size=(BOOTSTRAP_REPLICATIONS, blocks_per_draw),
    )
    indices = (starts[:, :, None] + np.arange(BOOTSTRAP_BLOCK_LENGTH)).reshape(
        BOOTSTRAP_REPLICATIONS, -1,
    )[:, :count]

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    accuracy_gain = hit.astype(float) - actual.astype(float)
    result["accuracyExcessAlwaysUp"] = interval(accuracy_gain[indices].mean(axis=1))
    result["brierImprovement"] = interval(brier_gain[indices].mean(axis=1))
    if actual.any() and (~actual).any():
        sampled_actual, sampled_hit = actual[indices], hit[indices]
        up_count = sampled_actual.sum(axis=1)
        down_count = count - up_count
        valid = (up_count > 0) & (down_count > 0)
        valid_count = int(valid.sum())
        result["balancedAccuracyValidReplications"] = valid_count
        if valid_count >= 2:
            up_recall = (sampled_hit & sampled_actual).sum(axis=1)[valid] / up_count[valid]
            down_recall = (sampled_hit & ~sampled_actual).sum(axis=1)[valid] / down_count[valid]
            result["balancedAccuracy"] = interval((up_recall + down_recall) / 2)
    return result


def forecast_metrics(frame: pd.DataFrame, *, compute_bootstrap: bool = True) -> dict:
    """Evaluate ordered forecasts; disable bootstrap for sparse filtered groups."""
    _require_columns(frame, {
        "date", "label_start", "label_end", "target", "probability", "base_probability",
    })
    if "symbol" in frame.columns and (frame.symbol.isna().any() or frame.symbol.nunique() != 1):
        raise ValueError("概率评估必须只包含一只股票。")
    dates = _dates(frame, "date")
    starts, ends = _dates(frame, "label_start"), _dates(frame, "label_end")
    _ordered_dates(dates)
    if not ((dates.to_numpy() < starts.to_numpy()) & (starts.to_numpy() < ends.to_numpy())).all():
        raise ValueError("收益区间必须在预测日之后开始，并在开始之后结束。")
    actual = _numbers(frame, "target") > 0
    probability = _numbers(frame, "probability", probability=True)
    base = _numbers(frame, "base_probability", probability=True)
    count = len(frame)
    hit = (probability >= 0.5) == actual
    errors, base_errors = (probability - actual) ** 2, (base - actual) ** 2
    brier, base_brier = float(errors.mean()), float(base_errors.mean())
    up_count, down_count = int(actual.sum()), int((~actual).sum())
    up_recall = float(hit[actual].mean()) if up_count else None
    down_recall = float(hit[~actual].mean()) if down_count else None
    both_classes = up_count > 0 and down_count > 0
    high = np.maximum(probability, 1 - probability) >= CONFIDENCE_THRESHOLD
    high_count = int(high.sum())
    bins = []
    calibration_error = 0.0
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= lower) & ((probability <= upper) if index == 4 else (probability < upper))
        bin_count = int(mask.sum())
        mean_probability = float(probability[mask].mean()) if bin_count else None
        actual_frequency = float(actual[mask].mean()) if bin_count else None
        if bin_count:
            calibration_error += bin_count / count * abs(mean_probability - actual_frequency)
        bins.append({
            "lower": lower, "upper": upper, "upperInclusive": index == 4,
            "count": bin_count, "meanProbability": mean_probability,
            "actualFrequency": actual_frequency,
        })
    intervals = pd.DataFrame({"start": starts.to_numpy(), "end": ends.to_numpy()}).sort_values(
        ["end", "start"], kind="stable",
    )
    last_end = None
    disjoint_count = 0
    for interval in intervals.itertuples(index=False):
        if last_end is None or interval.start > last_end:
            disjoint_count += 1
            last_end = interval.end
    return {
        "n": count,
        "start": dates.iloc[0].date().isoformat(),
        "end": dates.iloc[-1].date().isoformat(),
        "labelStart": starts.min().date().isoformat(),
        "labelEnd": ends.max().date().isoformat(),
        "accuracy": float(hit.mean()),
        "alwaysUpAccuracy": float(actual.mean()),
        "frequencyAccuracy": float(((base >= 0.5) == actual).mean()),
        "accuracyExcessAlwaysUp": float(hit.mean() - actual.mean()),
        "brier": brier,
        "baseBrier": base_brier,
        "brierImprovement": base_brier - brier,
        "brierSkill": 1 - brier / base_brier if base_brier > 0 else None,
        "brierSkillNote": "相对预测当时已知的上涨频率；基准误差为零时不定义相对改善比例。",
        "balancedAccuracy": (up_recall + down_recall) / 2 if both_classes else None,
        "upRecall": up_recall,
        "downRecall": down_recall,
        "upCount": up_count,
        "downCount": down_count,
        "auc": float(roc_auc_score(actual, probability)) if both_classes else None,
        "classMetricsAvailable": both_classes,
        "classMetricsUnavailableReason": None if both_classes else "真实结果只有一个类别，平衡命中率和 AUC 不可用；缺少类别的召回率也不可用。",
        "logLoss": float(log_loss(actual, probability, labels=[False, True])),
        "logLossNote": "对数损失按 sklearn 的机器精度裁剪端点概率，以计算有限值。",
        "ece": float(calibration_error),
        "eceNote": "五个固定等宽概率区间的加权校准误差，只作描述，不能证明预测能力。",
        "calibrationBins": bins,
        "highConfidence": {
            "threshold": CONFIDENCE_THRESHOLD,
            "count": high_count,
            "accuracy": float(hit[high].mean()) if high_count else None,
            "coverage": high_count / count,
            "meanConfidence": float(np.maximum(probability[high], 1 - probability[high]).mean()) if high_count else None,
            "definition": "上涨或未上涨方向的概率至少为 65%，只表示模型输出分组。",
        },
        "greedyNonOverlappingCount": disjoint_count,
        "nonOverlappingNote": "按收益结束日期优先贪心选取，后一收益开始日必须严格晚于前一结束日；区间不重叠仍不等于统计独立。",
        "bootstrap95": _bootstrap_intervals(actual, hit, base_errors - errors) if compute_bootstrap else {
            "accuracyExcessAlwaysUp": None,
            "brierImprovement": None,
            "balancedAccuracy": None,
            "available": False,
            "replications": 0,
            "balancedAccuracyValidReplications": 0,
            "unavailableReason": "本次评估已明确关闭区块重采样；稀疏筛选后的预测不保留完整五日时间网格，不能把连续六行解释为约三十个交易日并据此给出区间。",
        },
        "independentEvaluation": False,
        "researchNote": "每五个交易日预测未来二十一个交易日，标签存在重叠；本结果是探索性历史研究，不能视为独立验收。",
    }


def _simulation_metrics(simulation: Simulation, benchmark: Simulation) -> dict:
    return {
        **performance(simulation.equity, benchmark.equity),
        "totalFees": float(simulation.total_fees),
        "slippageCost": float(simulation.slippage_cost),
        "tradeCount": len(simulation.trades),
        "turnover": float(simulation.turnover),
        "blockedOrderCount": len(simulation.blocked_orders),
        "staleMarks": simulation.stale_marks,
        "maxIdentityResidual": float(simulation.max_identity_residual),
        "endingEquity": float(simulation.equity.iloc[-1]),
    }


def trading_metrics(
    market: Market,
    predictions: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    volatility: pd.DataFrame,
) -> dict:
    """Compare fixed long/cash rules through the existing next-open simulator."""
    _require_columns(predictions, {"date", "probability"})
    _require_columns(volatility, {"date", "symbol", "annualVolatility"})
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in market.dates or end not in market.dates or start >= end:
        raise ValueError("交易评估起止日期必须是有效且不同的港股交易日。")
    if symbol not in market.symbols:
        raise ValueError(f"交易市场不包含股票 {symbol}。")
    if "symbol" in predictions.columns and (predictions.symbol.isna().any() or set(predictions.symbol) != {symbol}):
        raise ValueError("交易预测必须只包含指定股票。")
    dates = _dates(predictions, "date")
    _ordered_dates(dates)
    probabilities = _numbers(predictions, "probability", probability=True)
    selected = pd.DataFrame({"date": dates.to_numpy(), "probability": probabilities})
    selected = selected.loc[(selected.date >= start) & (selected.date < end)].copy()
    if selected.empty:
        raise ValueError("指定交易区间没有可在下一开盘执行的预测。")
    anchor = pd.Timestamp(selected.date.iloc[0])
    if anchor not in market.dates:
        raise ValueError("首个预测日期不是市场交易日。")
    expected_dates = market.dates[(market.dates >= anchor) & (market.dates < end)][::SIGNAL_SPACING]
    if not pd.DatetimeIndex(selected.date).equals(expected_dates):
        raise ValueError("交易预测必须从共同起点开始，每五个正式交易日提供一次，完整覆盖回测区间。")
    stock_volatility = volatility.loc[volatility.symbol == symbol, ["date", "annualVolatility"]].copy()
    if stock_volatility.empty:
        raise ValueError("指定股票没有波动率数据。")
    stock_volatility["date"] = _dates(stock_volatility, "date")
    if stock_volatility.date.duplicated().any():
        raise ValueError("股票波动率存在重复日期。")
    selected = selected.merge(stock_volatility, on="date", how="left", validate="one_to_one")
    annual_volatility = _numbers(selected, "annualVolatility")
    if (annual_volatility <= 0).any():
        raise ValueError("预测日年化波动率必须严格大于零。")
    risk_weights = np.minimum(1.0, TARGET_VOLATILITY / annual_volatility)
    strategy_signals, risk_signals = {}, {}
    for row, risk_weight in zip(selected.itertuples(index=False), risk_weights):
        risk_signals[row.date] = {symbol: float(risk_weight)}
        strategy_signals[row.date] = {symbol: float(risk_weight)} if row.probability >= ENTRY_PROBABILITY else {}
    risk_hold = simulate(market, risk_signals, anchor, end, record_details=True)
    strategy = simulate(market, strategy_signals, anchor, end, record_details=True)
    buy_hold = simulate(market, {anchor: {symbol: 1.0}}, anchor, end, record_details=True)
    stress_risk = simulate(market, risk_signals, anchor, end, slippage=2 * SLIPPAGE, record_details=True)
    stress = simulate(market, strategy_signals, anchor, end, slippage=2 * SLIPPAGE, record_details=True)
    return {
        "symbol": symbol,
        "requestedStart": start.date().isoformat(),
        "anchor": anchor.date().isoformat(),
        "end": end.date().isoformat(),
        "lastSignal": selected.date.iloc[-1].date().isoformat(),
        "signalCount": len(selected),
        "investedSignalCount": int((selected.probability >= ENTRY_PROBABILITY).sum()),
        "rules": {
            "entryProbability": ENTRY_PROBABILITY,
            "targetAnnualVolatility": TARGET_VOLATILITY,
            "maximumWeight": 1.0,
            "signalSpacingTradingDays": SIGNAL_SPACING,
            "slippagePerSide": SLIPPAGE,
            "stressSlippagePerSide": 2 * SLIPPAGE,
            "execution": "信号日收盘后确定目标仓位，由既有模拟器于下一正式交易日开盘执行。",
            "position": "上涨概率至少 55% 时，目标仓位为 1 与 15% 除以当日年化波动率的较小值；其余目标为空仓。",
            "baseline": "同波动目标基准每五日调整本股仓位；普通持有基准只在共同起点发出一次满仓买入信号。",
        },
        "strategy": _simulation_metrics(strategy, risk_hold),
        "riskMatchedBuyHold": _simulation_metrics(risk_hold, risk_hold),
        "buyHold100": _simulation_metrics(buy_hold, risk_hold),
        "doubleSlippage": _simulation_metrics(stress, stress_risk),
        "doubleSlippageRiskMatchedBuyHold": _simulation_metrics(stress_risk, stress_risk),
        "annualExcessDefinition": "策略净年化收益减同波动目标持有基准净年化收益；双倍滑点策略对照同为双倍滑点的基准。仓位规则相同不代表实现波动率完全一致。",
        "independentEvaluation": False,
        "researchNote": "阈值和仓位规则固定，所有策略使用相同起点及终点。净值含费用与滑点；期末持仓按收盘价计值，未强制卖出。夏普比率以零现金收益计算。本结果属于探索性历史研究。",
    }
