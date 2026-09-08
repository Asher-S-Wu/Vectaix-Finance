from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


CONFIDENCE_THRESHOLD = 0.65
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_REPLICATIONS = 2000
BOOTSTRAP_SEED = 42


def _wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    frequency = successes / count
    denominator = 1 + z * z / count
    center = (frequency + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(frequency * (1 - frequency) / count + z * z / (4 * count * count)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _probabilities(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{column} 必须全部为 0 至 1 之间的有限概率。")
    return values


def _brier_difference_interval(differences: np.ndarray) -> list[float] | None:
    count = len(differences)
    if count < BOOTSTRAP_BLOCK_LENGTH:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks_per_draw = (count + BOOTSTRAP_BLOCK_LENGTH - 1) // BOOTSTRAP_BLOCK_LENGTH
    starts = rng.integers(
        0, count - BOOTSTRAP_BLOCK_LENGTH + 1,
        size=(BOOTSTRAP_REPLICATIONS, blocks_per_draw),
    )
    indices = (starts[:, :, None] + np.arange(BOOTSTRAP_BLOCK_LENGTH)).reshape(BOOTSTRAP_REPLICATIONS, -1)
    draw_means = differences[indices[:, :count]].mean(axis=1)
    return [float(value) for value in np.quantile(draw_means, [0.025, 0.975])]


def probability_metrics(frame: pd.DataFrame, probability_column: str = "probability") -> dict:
    """Score chronological forecasts whose realized return intervals do not overlap."""
    required = {
        "date", "label_start", "label_end", "target", probability_column,
        "raw_probability", "base_probability", "core_probability", "old_direction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"概率评估缺少字段：{sorted(missing)}")
    count = len(frame)
    if count < 1:
        raise ValueError("概率评估至少需要一条已成熟的预测。")
    dates = pd.to_datetime(frame["date"], errors="raise")
    starts = pd.to_datetime(frame["label_start"], errors="raise")
    ends = pd.to_datetime(frame["label_end"], errors="raise")
    if dates.isna().any() or starts.isna().any() or ends.isna().any():
        raise ValueError("预测日期及收益区间日期不能缺失。")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("概率评估必须按唯一预测日期从早到晚排列。")
    if not ((dates.to_numpy() < starts.to_numpy()) & (starts.to_numpy() < ends.to_numpy())).all():
        raise ValueError("收益区间必须在预测日之后开始，并在开始之后结束。")
    if count > 1 and (ends.to_numpy()[:-1] > starts.to_numpy()[1:]).any():
        raise ValueError("概率评估包含重叠收益区间，不能计为互不重叠样本。")
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("已成熟收益必须全部为有限数字。")
    actual = target > 0
    probabilities = {
        column: _probabilities(frame, column)
        for column in {probability_column, "raw_probability", "base_probability", "core_probability"}
    }
    probability = probabilities[probability_column]
    base = probabilities["base_probability"]
    core = probabilities["core_probability"]
    old_direction = frame["old_direction"].to_numpy()
    if any(not isinstance(value, (bool, np.bool_)) for value in old_direction):
        raise ValueError("old_direction 必须全部为布尔值，表示旧模型是否预测上涨。")
    hit = (probability >= 0.5) == actual
    errors = (probability - actual) ** 2
    base_errors = (base - actual) ** 2
    core_errors = (core - actual) ** 2
    brier = float(errors.mean())
    base_brier = float(base_errors.mean())
    high_mask = np.maximum(probability, 1 - probability) >= CONFIDENCE_THRESHOLD
    high_count = int(high_mask.sum())
    high_successes = int(hit[high_mask].sum())
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
            "lower": float(lower), "upper": float(upper), "upperInclusive": index == 4,
            "count": bin_count, "meanProbability": mean_probability, "actualFrequency": actual_frequency,
        })
    result = {
        "n": count,
        "start": dates.iloc[0].date().isoformat(),
        "end": dates.iloc[-1].date().isoformat(),
        "firstLabelStart": starts.iloc[0].date().isoformat(),
        "lastLabelEnd": ends.iloc[-1].date().isoformat(),
        "actualUpFrequency": float(actual.mean()),
        "accuracy": float(hit.mean()),
        "accuracyWilson95": _wilson(int(hit.sum()), count),
        "brier": brier,
        "rawBrier": float(((probabilities["raw_probability"] - actual) ** 2).mean()),
        "logLoss": float(log_loss(actual, probability, labels=[False, True])),
        "baseBrier": base_brier,
        "baseAccuracy": float(((base >= 0.5) == actual).mean()),
        "coreBrier": float(core_errors.mean()),
        "oldAccuracy": float((old_direction == actual).mean()),
        "brierSkill": 1 - brier / base_brier if base_brier > 0 else None,
        "brierSkillNote": "相对预测当时已知的无条件上涨频率；基准误差为零时该比值无定义。",
        "highConfidence": {
            "threshold": CONFIDENCE_THRESHOLD,
            "definition": "预测上涨或下跌的方向概率至少为 65%；只是概率分组，不代表已证实可靠。",
            "count": high_count,
            "accuracy": high_successes / high_count if high_count else None,
            "accuracyWilson95": _wilson(high_successes, high_count) if high_count else None,
            "coverage": high_count / count,
        },
        "calibrationBins": bins,
        "ece": float(calibration_error),
        "eceLabel": "5 个固定概率区间的加权平均校准误差；小样本下仅作描述，不能代替预测能力评价。",
        "intervalCoverage": None,
        "meanIntervalWidth": None,
        "coreBrierDifference": float((errors - core_errors).mean()),
        "coreBrierDifferenceBootstrap95": _brier_difference_interval(errors - core_errors),
        "brierDifferenceBootstrap": {
            "method": "按时间顺序进行配对移动区块重采样，连续区块不环绕；尾部截取至原样本数。",
            "blockLength": BOOTSTRAP_BLOCK_LENGTH,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
            "available": count >= BOOTSTRAP_BLOCK_LENGTH,
            "definition": "当前模型 Brier 减去同日 8 因子概率模型 Brier；负数表示当前模型误差较小。",
            "limitation": "区间仅近似保留短期依赖；样本少时不稳定，不能消除模型选择、历史反复查看或事后改方案的偏差。",
        },
        "sampleNote": "收益区间互不重叠不等于统计独立；Wilson 区间按二项样本计算，时间相关会降低其可靠性。四股结果不能直接合并为独立样本。",
        "selectionBiasNote": "本轮是探索性历史比较；选型期结果存在选择偏差，2025 年后的历史也已被此前研究查看，均不是新的独立验收。",
    }
    interval_columns = {"lower_return", "upper_return"}
    present_intervals = interval_columns & set(frame.columns)
    if present_intervals and present_intervals != interval_columns:
        raise ValueError("收益区间必须同时提供 lower_return 和 upper_return。")
    if present_intervals:
        lower = pd.to_numeric(frame["lower_return"], errors="raise").to_numpy(dtype=float)
        upper = pd.to_numeric(frame["upper_return"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
            raise ValueError("收益区间端点必须有限且下界不大于上界。")
        result["intervalCoverage"] = float(((target >= lower) & (target <= upper)).mean())
        result["meanIntervalWidth"] = float((upper - lower).mean())
    return result
