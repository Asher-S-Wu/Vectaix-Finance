"""可复用的预测、组合与发布资格评价。"""

from __future__ import annotations

from math import ceil, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .prediction_tasks import (MODEL_REJECTION_STATUSES, TASK_COLUMNS, TASK_STATUS_COLUMNS,
                               aggregate_task_status, validate_task_outputs)


HORIZONS = (1, 5, 20, 60)
PREDICTION_COLUMNS = (
    "date", "security_id", "horizon", "fwd_return", "label_end", "score",
    "probability_up", "q10", "q50", "q90", "baseline_probability",
    "baseline_q10", "baseline_q50", "baseline_q90", "expected_return", "status",
    *TASK_STATUS_COLUMNS.values(),
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} 缺少必需字段: {', '.join(missing)}")


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _plain(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _ece(actual_up: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    order = np.argsort(probabilities, kind="stable")
    bins = []
    weighted_error = 0.0
    for indexes in np.array_split(order, min(10, len(order))):
        if not len(indexes):
            continue
        mean_probability = float(np.mean(probabilities[indexes]))
        observed_rate = float(np.mean(actual_up[indexes]))
        error = abs(mean_probability - observed_rate)
        weighted_error += len(indexes) * error
        bins.append({
            "count": int(len(indexes)),
            "mean_probability": mean_probability,
            "observed_up_rate": observed_rate,
            "absolute_error": error,
        })
    return float(weighted_error / len(order)), bins


def _block_bootstrap_mean(values: np.ndarray) -> tuple[float | None, float | None]:
    block_length = 60
    if len(values) < block_length:
        return None, None
    rng = np.random.default_rng(20260910)
    positions = np.arange(len(values))
    estimates = np.empty(2000, dtype=float)
    block_offsets = np.arange(block_length)
    blocks_needed = ceil(len(values) / block_length)
    for iteration in range(2000):
        starts = rng.integers(0, len(values), size=blocks_needed)
        sampled = np.concatenate([values[(start + block_offsets) % len(values)] for start in starts])
        estimates[iteration] = sampled[:len(values)].mean()
    lower, upper = np.quantile(estimates, [.025, .975])
    return float(lower), float(upper)


def _prediction_horizon_metrics(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    horizon_rows = frame.loc[frame["horizon"] == horizon].copy()
    observed = np.isfinite(horizon_rows['fwd_return'])
    valid = horizon_rows.loc[horizon_rows.status.eq('ok') & observed, ['date']]
    task_valid = {task: horizon_rows.loc[horizon_rows[column].eq('ok') & observed,
                                        ['date', 'fwd_return', *TASK_COLUMNS[task]]]
                  for task, column in TASK_STATUS_COLUMNS.items()}

    daily_ic = []
    for date, group in task_valid['score'].groupby("date", sort=True):
        if len(group) < 20 or group["score"].nunique() < 2 or group["fwd_return"].nunique() < 2:
            continue
        coefficient = group["score"].corr(group["fwd_return"], method="spearman")
        if _finite_number(coefficient):
            daily_ic.append((pd.Timestamp(date), float(coefficient)))
    ic_values = np.array([value for _, value in daily_ic], dtype=float)
    ci_lower, ci_upper = _block_bootstrap_mean(ic_values)
    annual_ic: dict[str, float] = {}
    if daily_ic:
        ic_frame = pd.DataFrame(daily_ic, columns=["date", "ic"])
        annual_ic = {
            str(int(year)): float(value)
            for year, value in ic_frame.groupby(ic_frame["date"].dt.year)["ic"].mean().items()
        }
    positive_year_ratio = (
        float(np.mean(np.array(list(annual_ic.values())) > 0)) if annual_ic else None
    )

    probabilities_valid = task_valid['probability_up']
    if probabilities_valid.empty:
        brier_model = brier_baseline = brier_skill = None
        ece = None
        calibration_bins: list[dict[str, Any]] = []
    else:
        actual = probabilities_valid["fwd_return"].to_numpy(float)
        actual_up = (actual > 0).astype(float)
        probabilities = probabilities_valid["probability_up"].to_numpy(float)
        baseline_probabilities = probabilities_valid["baseline_probability"].to_numpy(float)
        brier_model = float(np.mean((probabilities - actual_up) ** 2))
        brier_baseline = float(np.mean((baseline_probabilities - actual_up) ** 2))
        brier_skill = 1.0 - brier_model / brier_baseline if brier_baseline > 0 else None
        ece, calibration_bins = _ece(actual_up, probabilities)
    intervals_valid = task_valid['intervals']
    if intervals_valid.empty:
        coverage = pinball_model = pinball_baseline = None
    else:
        actual = intervals_valid['fwd_return'].to_numpy(float)
        coverage = float(np.mean(
            (actual >= intervals_valid["q10"].to_numpy(float))
            & (actual <= intervals_valid["q90"].to_numpy(float))
        ))
        pinball_model = float(np.mean([
            _pinball(actual, intervals_valid[column].to_numpy(float), quantile)
            for column, quantile in (("q10", .1), ("q50", .5), ("q90", .9))
        ]))
        pinball_baseline = float(np.mean([
            _pinball(actual, intervals_valid[column].to_numpy(float), quantile)
            for column, quantile in (
                ("baseline_q10", .1), ("baseline_q50", .5), ("baseline_q90", .9),
            )
        ]))

    return {
        "horizon": horizon,
        **mature_label_coverage(horizon_rows),
        "valid_samples": int(len(valid)),
        "task_valid_samples": {task: int(len(values)) for task, values in task_valid.items()},
        "valid_dates": int(valid["date"].nunique()),
        "years": sorted(int(year) for year in valid["date"].dt.year.unique()),
        "ic": {
            "daily_count": int(len(ic_values)),
            "minimum_cross_section_size": 20,
            "mean": float(ic_values.mean()) if len(ic_values) else None,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bootstrap_seed": 20260910,
            "bootstrap_samples": 2000,
            "block_length_trading_days": 60,
            "annual_mean": annual_ic,
            "positive_year_ratio": positive_year_ratio,
        },
        "brier": {
            "model": brier_model,
            "baseline": brier_baseline,
            "skill": brier_skill,
            "skill_definition": "1 - model_brier / baseline_brier",
            "baseline_definition": "predictions.baseline_probability",
        },
        "calibration": {
            "ece": ece,
            "threshold": .05,
            "passes": ece is not None and ece <= .05,
            "binning": "10 equal-frequency bins (or one bin per sample when fewer than 10)",
            "bins": calibration_bins,
        },
        "interval": {
            "coverage": coverage,
            "coverage_range": [.75, .85],
            "coverage_pass": coverage is not None and .75 <= coverage <= .85,
            "pinball_model": pinball_model,
            "pinball_baseline": pinball_baseline,
            "pinball_improved": (
                pinball_model is not None and pinball_baseline is not None
                and pinball_model < pinball_baseline
            ),
            "baseline_definition": "predictions.baseline_q10/baseline_q50/baseline_q90",
        },
    }


def mature_label_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """已到期限的预测必须区分可观测收益与尚未解决的收益缺口。"""
    total = len(frame)
    observed = int(np.isfinite(frame['fwd_return']).sum())
    return {
        'matured_rows': total,
        'matured_label_rows': observed,
        'missing_matured_label_rows': total - observed,
        'matured_label_coverage': observed / total if total else None,
        'label_coverage_complete': total > 0 and observed == total,
    }


def prediction_status_coverage(frame: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    """在完整预测清单中区分输入不足与模型已有输入后的输出失败。"""
    _require_columns(frame, ('status', 'label_end', *TASK_STATUS_COLUMNS.values()), 'predictions')
    aggregate = aggregate_task_status(frame)
    if not aggregate.eq(frame.status).all():
        raise ValueError('整体状态与独立任务状态不一致')
    task_coverage = {}
    for task, column in TASK_STATUS_COLUMNS.items():
        states = frame[column]
        task_ready = states.ne('insufficient_model_inputs')
        task_rejected = states.isin(MODEL_REJECTION_STATUSES)
        eligible_count = int(task_ready.sum())
        failure_count = int(task_rejected.sum())
        task_coverage[task] = {
            'total_forecast_rows': len(frame),
            'scored_forecast_rows': int(states.eq('ok').sum()),
            'excluded_status_rows': int(states.ne('ok').sum()),
            'input_unavailable_rows': int(states.eq('insufficient_model_inputs').sum()),
            'input_eligible_forecast_rows': eligible_count,
            'model_rejected_rows': failure_count,
            'matured_model_rejected_rows': int((task_rejected & frame.label_end.le(cutoff)).sum()),
            'model_rejection_rate': failure_count / eligible_count if eligible_count else None,
            'model_output_coverage_complete': eligible_count > 0 and failure_count == 0,
            'status_counts': {str(key): int(value) for key, value in states.value_counts().items()},
        }
    states = frame[list(TASK_STATUS_COLUMNS.values())]
    rejected = states.isin(MODEL_REJECTION_STATUSES).any(axis=1)
    ready = states.ne('insufficient_model_inputs').any(axis=1)
    any_scored = states.eq('ok').any(axis=1)
    all_scored = states.eq('ok').all(axis=1)
    eligible = int(ready.sum())
    failures = int(rejected.sum())
    return {
        'total_forecast_rows': len(frame),
        'scored_forecast_rows': int(frame.status.eq('ok').sum()),
        'excluded_status_rows': int(frame.status.ne('ok').sum()),
        'input_unavailable_rows': int((~ready).sum()),
        'partially_input_unavailable_rows': int((ready & states.eq('insufficient_model_inputs').any(axis=1)).sum()),
        'any_task_scored_forecast_rows': int(any_scored.sum()),
        'partially_scored_forecast_rows': int((any_scored & ~all_scored).sum()),
        'input_eligible_forecast_rows': eligible,
        'model_rejected_rows': failures,
        'matured_model_rejected_rows': int((rejected & frame.label_end.le(cutoff)).sum()),
        'model_rejection_rate': failures / eligible if eligible else None,
        'model_output_coverage_complete': all(value['model_output_coverage_complete'] for value in task_coverage.values()),
        'status_counts': {str(key): int(value) for key, value in frame.status.value_counts().items()},
        'task_coverage': task_coverage,
    }


def evaluate_predictions(predictions: pd.DataFrame, as_of: Any) -> dict[str, Any]:
    """评价截至 ``as_of`` 已成熟的预测；未成熟标签完全不参与指标。"""
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions")
    frame = predictions.copy()
    missing_label_end = frame["label_end"].isna()
    empty_return = frame["fwd_return"].isna()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["label_end"] = pd.to_datetime(frame["label_end"], errors="coerce")
    if frame["date"].isna().any() or (frame["label_end"].isna() & ~missing_label_end).any():
        raise ValueError("date 和非空 label_end 必须是有效日期")
    if (missing_label_end & ~empty_return).any():
        raise ValueError("缺少 label_end 的预测不能提供 fwd_return")
    try:
        cutoff = pd.Timestamp(as_of)
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of 必须是有效日期") from exc
    numeric_columns = [column for column in PREDICTION_COLUMNS if column not in {
        "date", "security_id", "label_end", "status", *TASK_STATUS_COLUMNS.values(),
    }]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    validate_task_outputs(frame)
    if (~frame["horizon"].isin(HORIZONS)).any():
        raise ValueError("horizon 只允许 1、5、20、60")

    input_eligible = frame[list(TASK_STATUS_COLUMNS.values())].ne('insufficient_model_inputs').any(axis=1)
    mature = frame.loc[input_eligible & frame["label_end"].le(cutoff)].copy()
    coverage = mature_label_coverage(mature)
    statuses = prediction_status_coverage(frame, cutoff)
    return {
        "valid": coverage['label_coverage_complete'] and statuses['model_output_coverage_complete'],
        "as_of": cutoff.isoformat(),
        "input_rows": int(len(frame)),
        **coverage,
        **statuses,
        'task_valid_samples': {task: int((mature[column].eq('ok') & np.isfinite(mature.fwd_return)).sum())
                               for task, column in TASK_STATUS_COLUMNS.items()},
        "excluded_unmatured_rows": int((input_eligible & frame["label_end"].gt(cutoff)).sum()),
        "pending_label_end_rows": int((input_eligible & missing_label_end).sum()),
        "label_rule": "Only label_end <= as_of is evaluated",
        "evaluation_scope": "Each task is evaluated using its own status=ok predictions and finite mature returns. Label coverage includes every matured row with inputs for at least one task, including other tasks' input gaps and model rejections. Any model rejection prevents complete capability acceptance. Rows with no task inputs and task-specific availability remain separately visible. Full-market and terminal-return coverage require separate evidence.",
        "baseline_definition": {
            "probability": "baseline_probability supplied with each prediction",
            "quantiles": "baseline_q10, baseline_q50 and baseline_q90 supplied with each prediction",
        },
        "horizons": {horizon: {**_prediction_horizon_metrics(mature, horizon),
                                **prediction_status_coverage(frame.loc[frame.horizon.eq(horizon)], cutoff)}
                     for horizon in HORIZONS},
    }


def _equity_series(frame: pd.DataFrame, name: str) -> pd.Series:
    _require_columns(frame, ("date", "equity"), name)
    data = frame[["date", "equity"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError(f"{name} 日期无效或重复")
    if data.empty or (~np.isfinite(data["equity"])).any() or (data["equity"] <= 0).any():
        raise ValueError(f"{name} 每日净值必须为有限正数")
    return data.sort_values("date").set_index("date")["equity"].astype(float)


def _require_same_dates(left: pd.Series, right: pd.Series, names: str) -> None:
    if not left.index.equals(right.index):
        raise ValueError(f"{names} 日期集合必须完全一致，不能内连接丢弃日期")


def _annualized_growth(equity: pd.Series) -> float | None:
    if len(equity) < 2:
        return None
    return float((equity.iloc[-1] / equity.iloc[0]) ** (252 / (len(equity) - 1)) - 1)


def _excess_metrics(portfolio: pd.Series, benchmark: pd.Series) -> tuple[float | None, float | None]:
    excess = portfolio.pct_change().iloc[1:] - benchmark.pct_change().iloc[1:]
    if excess.empty:
        return None, None
    annual_excess = float(excess.mean() * 252)
    standard_deviation = float(excess.std(ddof=1))
    information_ratio = (
        float(excess.mean() / standard_deviation * sqrt(252))
        if _finite_number(standard_deviation) and standard_deviation > 0 else None
    )
    return annual_excess, information_ratio


def _positive_excess_year_ratio(portfolio: pd.Series, benchmark: pd.Series) -> float | None:
    daily = pd.DataFrame({
        "portfolio": portfolio.pct_change(),
        "benchmark": benchmark.pct_change(),
    }).dropna()
    if daily.empty:
        return None
    annual = daily.groupby(daily.index.year).apply(
        lambda group: (1 + group["portfolio"]).prod() - (1 + group["benchmark"]).prod(),
        include_groups=False,
    )
    return float((annual > 0).mean())


def evaluate_portfolio(
    daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    stress_daily: pd.DataFrame,
    stress_benchmark_daily: pd.DataFrame,
    execution_summary: dict[str, Any],
) -> dict[str, Any]:
    """评价组合，并把成交与公司行动证据纳入有效性判定。"""
    portfolio = _equity_series(daily, "daily")
    benchmark = _equity_series(benchmark_daily, "benchmark_daily")
    stress = _equity_series(stress_daily, "stress_daily")
    stress_benchmark = _equity_series(stress_benchmark_daily, "stress_benchmark_daily")
    _require_same_dates(portfolio, benchmark, "daily 与 benchmark_daily")
    _require_same_dates(stress, stress_benchmark, "stress_daily 与 stress_benchmark_daily")

    running_peak = portfolio.cummax()
    max_drawdown = float((portfolio / running_peak - 1).min())
    annual_excess, information_ratio = _excess_metrics(portfolio, benchmark)
    stress_annual_excess, _ = _excess_metrics(stress, stress_benchmark)
    execution_summary = execution_summary if isinstance(execution_summary, dict) else {}
    coverage = execution_summary.get("corporate_action_coverage")
    coverage_complete = coverage is True or (
        _finite_number(coverage) and float(coverage) == 1.0
    )
    execution_complete = (
        execution_summary.get("status") in {"complete", "ok"}
        and execution_summary.get("gap_count") == 0
        and coverage_complete
    )
    return {
        "portfolio_valid": bool(execution_complete),
        "valid_samples": int(len(portfolio)),
        "first_date": portfolio.index[0].date().isoformat(),
        "last_date": portfolio.index[-1].date().isoformat(),
        "years": sorted(int(year) for year in portfolio.index.year.unique()),
        "cagr": _annualized_growth(portfolio),
        "max_drawdown": max_drawdown,
        "information_ratio": information_ratio,
        "ann_excess": annual_excess,
        "ann_excess_definition": "mean daily portfolio return minus benchmark return, multiplied by 252",
        "positive_year_ratio": _positive_excess_year_ratio(portfolio, benchmark),
        "stress_ann_excess": stress_annual_excess,
        "stress_definition": "mean daily stress portfolio return minus stress benchmark return, multiplied by 252",
        "execution_complete": bool(execution_complete),
        "execution_summary": execution_summary,
    }


def _lookup(mapping: Any, *path: Any) -> tuple[bool, Any]:
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return False, None
        if key in value:
            value = value[key]
        elif isinstance(key, int) and str(key) in value:
            value = value[str(key)]
        else:
            return False, None
    return True, value


def monthly_confirmation_evidence(predictions: dict[str, Any], development_revision: Any) -> dict[str, Any]:
    """核对实际月度运行记录，静态或不完整预测只能作为诊断。"""
    reasons = []
    revision = predictions.get('model_revision')
    if not revision or revision != development_revision or revision != predictions.get('frozen_model_revision'):
        reasons.append('开发、冻结和确认的模型方案版本不一致或缺失')
    if predictions.get('formal_confirmation') is not True:
        reasons.append('缺少正式确认运行')
    if predictions.get('parameter_selection_used') is not False:
        reasons.append('缺少未用于参数挑选的确认声明')
    if predictions.get('retrain_frequency') != 'monthly':
        reasons.append('确认未按月重训')
    cutoff = pd.to_datetime(predictions.get('as_of'), errors='coerce')
    expected = ([str(p) for p in pd.period_range('2024-01', cutoff, freq='M')]
                if pd.notna(cutoff) and cutoff >= pd.Timestamp('2024-01-01') else [])
    if not expected:
        reasons.append('确认截止日期无效')
    records = predictions.get('monthly_model_records')
    if not isinstance(records, list):
        records = []
        reasons.append('缺少逐月模型运行记录')
    months = [record.get('forecast_month') for record in records if isinstance(record, dict)]
    if months != expected:
        reasons.append('逐月记录未完整覆盖2024年1月至确认截止日期')
    heads = predictions.get('frozen_heads')
    calibration = predictions.get('frozen_calibration_parameters')
    if not isinstance(heads, dict) or not heads or not isinstance(calibration, dict) or not calibration:
        reasons.append('冻结的任务配置或校准参数缺失')
    for record in records:
        if not isinstance(record, dict):
            reasons.append('月度模型记录无效')
            continue
        month = record.get('forecast_month')
        if month not in expected:
            continue
        trained = pd.to_datetime(record.get('trained_as_of'), errors='coerce')
        start = pd.Period(month, freq='M').start_time
        if pd.isna(trained) or trained >= start:
            reasons.append(f'{month}: 模型训练截止不早于预测月')
        if record.get('model_revision') != revision or record.get('heads') != heads:
            reasons.append(f'{month}: 模型方案版本或任务配置不等于冻结方案')
        components = record.get('component_calibration_parameters')
        if (not isinstance(components, dict) or not components
                or any(value != calibration for value in components.values())):
            reasons.append(f'{month}: 实际模型校准参数与冻结方案不一致或缺失')
        if record.get('horizons') != list(HORIZONS):
            reasons.append(f'{month}: 未覆盖四个预测期限')
        rows = record.get('forecast_rows')
        if not _finite_number(rows) or rows <= 0 or record.get('prediction_coverage_complete') is not True:
            reasons.append(f'{month}: 预测范围不完整或缺少核对记录')
    return {'valid': not reasons, 'model_revision': revision, 'expected_months': len(expected),
            'recorded_months': len(records), 'reasons': reasons}


def release_decision(
    development: dict[str, Any],
    confirmation: dict[str, Any],
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    """按固定门槛判断是否具备发布资格；证据缺失一律失败。"""
    checks: dict[str, dict[str, Any]] = {}
    failed_reasons: list[str] = []

    def check(name: str, path: tuple[Any, ...], source: Any, predicate, requirement: str) -> None:
        present, actual = _lookup(source, *path)
        passed = bool(present and predicate(actual))
        checks[name] = {
            "passed": passed,
            "present": present,
            "actual": actual if present else None,
            "requirement": requirement,
        }
        if not passed:
            failed_reasons.append(
                f"{name}: {'证据缺失' if not present else f'实际值 {actual!r}'}；要求 {requirement}"
            )

    finite = lambda value: _finite_number(value)
    for period_name, period in (("development", development), ("confirmation", confirmation)):
        check(f"{period_name}.predictions.valid", ("predictions", "valid"), period,
              lambda value: value is True, "True")
        check(f"{period_name}.predictions.label_coverage_complete",
              ("predictions", "label_coverage_complete"), period,
              lambda value: value is True, "所有已成熟预测均有可核验收益标签")
        check(f"{period_name}.predictions.missing_matured_label_rows",
              ("predictions", "missing_matured_label_rows"), period,
              lambda value: finite(value) and value == 0, "== 0")
        check(f"{period_name}.predictions.model_output_coverage_complete",
              ("predictions", "model_output_coverage_complete"), period,
              lambda value: value is True, "已有模型输入的预测无模型输出失败")
        check(f"{period_name}.predictions.model_rejected_rows",
              ("predictions", "model_rejected_rows"), period,
              lambda value: finite(value) and value == 0, "== 0")
        check(f"{period_name}.predictions.h20.ic_lower", ("predictions", "horizons", 20, "ic", "ci_lower"),
              period, lambda value: finite(value) and float(value) > 0, "> 0")
        check(f"{period_name}.predictions.h20.positive_year_ratio",
              ("predictions", "horizons", 20, "ic", "positive_year_ratio"), period,
              lambda value: finite(value) and float(value) >= .60, ">= 0.60")
        for horizon in HORIZONS:
            prefix = f"{period_name}.predictions.h{horizon}"
            check(f"{prefix}.brier_skill", ("predictions", "horizons", horizon, "brier", "skill"),
                  period, lambda value: finite(value) and float(value) > 0, "> 0")
            check(f"{prefix}.ece", ("predictions", "horizons", horizon, "calibration", "ece"),
                  period, lambda value: finite(value) and float(value) <= .05, "<= 0.05")
            check(f"{prefix}.coverage", ("predictions", "horizons", horizon, "interval", "coverage"),
                  period, lambda value: finite(value) and .75 <= float(value) <= .85, "0.75 至 0.85")
            present_model, model_loss = _lookup(period, "predictions", "horizons", horizon, "interval", "pinball_model")
            present_base, baseline_loss = _lookup(period, "predictions", "horizons", horizon, "interval", "pinball_baseline")
            name = f"{prefix}.pinball_improved"
            passed = bool(
                present_model and present_base and finite(model_loss) and finite(baseline_loss)
                and float(model_loss) < float(baseline_loss)
            )
            checks[name] = {
                "passed": passed,
                "present": present_model and present_base,
                "actual": {"model": model_loss, "baseline": baseline_loss},
                "requirement": "model < baseline",
            }
            if not passed:
                failed_reasons.append(f"{name}: 必需证据缺失或模型损失未低于基准")

        portfolio_checks = (
            ("portfolio_valid", lambda value: value is True, "True"),
            ("ann_excess", lambda value: finite(value) and float(value) > 0, "> 0"),
            ("information_ratio", lambda value: finite(value) and float(value) >= .5, ">= 0.50"),
            ("max_drawdown", lambda value: finite(value) and float(value) >= -.30, ">= -0.30"),
            ("positive_year_ratio", lambda value: finite(value) and float(value) >= .60, ">= 0.60"),
            ("stress_ann_excess", lambda value: finite(value) and float(value) > 0, "> 0"),
            ("execution_complete", lambda value: value is True, "True"),
        )
        for field, predicate, requirement in portfolio_checks:
            check(f"{period_name}.portfolio.{field}", ("portfolio", field), period,
                  predicate, requirement)

    _, predictions = _lookup(confirmation, 'predictions')
    _, revision = _lookup(development, 'predictions', 'model_revision')
    monthly = monthly_confirmation_evidence(predictions if isinstance(predictions, dict) else {}, revision)
    check('confirmation.predictions.monthly_frozen_validation', ('evidence',), {'evidence': monthly},
          lambda value: value['valid'], '完整月度重训、相同冻结配置及四期限预测记录')

    audit_checks = (
        ("critical_gap_count", lambda value: value == 0, "== 0"),
        ("future_leakage_detected", lambda value: value is False, "False"),
        ("historical_lot_coverage_complete", lambda value: value is True, "True"),
        ("corporate_action_cash_coverage_complete", lambda value: value is True, "True"),
        ("terminal_return_coverage_complete", lambda value: value is True, "True"),
    )
    for field, predicate, requirement in audit_checks:
        check(f"data_audit.{field}", (field,), data_audit, predicate, requirement)

    return {
        "eligible": not failed_reasons,
        "checks": checks,
        "failed_reasons": failed_reasons,
        "policy": "All fixed checks must pass in both development and confirmation periods; thresholds are not adjusted automatically.",
    }
