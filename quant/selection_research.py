from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import research_calendar
from .engine import _read_inputs
from .lab_evaluation import forecast_metrics, trading_metrics
from .lab_training import EXCELLENCE_RULES, _excellent_checks
from .portfolio import Market
from .runner import ROOT, RUNS


SOURCE_RUN_ID = "20260908T093918232471Z"
POLICY_ID = "target_weight_3_other_targets_1"
TARGET_WEIGHT = 3.0
OTHER_WEIGHT = 1.0
INNER_YEARS = 3
MINIMUM_GROUP_FORECASTS = 24
DEPENDENCIES = (
    "__init__.py", "selection_research.py", "data.py", "engine.py", "factors.py",
    "lab_binning.py", "lab_data.py", "lab_events.py", "lab_evaluation.py",
    "lab_models.py", "lab_training.py", "macro_data.py", "metrics.py",
    "portfolio.py", "runner.py", "trend_factors.py",
)
SOURCE_INPUTS = (
    "report.json", "specification.json", "candidate-predictions.csv",
    "selected-predictions.csv", "factors.csv", "reproduction/data/prices.csv",
    "reproduction/data/universe.json", "reproduction/data/manifest.json",
    "reproduction/data/review.json",
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _candidate_statistics(rows: pd.DataFrame) -> dict:
    year_groups = []
    for (symbol, year), group in rows.groupby(["symbol", rows.date.dt.year]):
        if len(group) < MINIMUM_GROUP_FORECASTS:
            continue
        truth = (group.target.to_numpy() > 0).astype(float)
        difference = np.mean(
            (group.probability.to_numpy() - truth) ** 2
            - (group.base_probability.to_numpy() - truth) ** 2,
        )
        year_groups.append({
            "symbol": symbol, "year": int(year), "n": len(group),
            "brierDifference": float(difference),
        })
    stock_groups = []
    for symbol, group in rows.groupby("symbol"):
        truth = group.target.to_numpy() > 0
        if len(group) < MINIMUM_GROUP_FORECASTS or not truth.any() or truth.all():
            continue
        hit = (group.probability.to_numpy() >= 0.5) == truth
        up_recall, down_recall = float(hit[truth].mean()), float(hit[~truth].mean())
        stock_groups.append({
            "symbol": symbol, "n": len(group), "upRecall": up_recall,
            "downRecall": down_recall, "balancedAccuracy": (up_recall + down_recall) / 2,
        })
    if not year_groups or not stock_groups:
        raise ValueError("加权选型缺少足够的股票年度或双向成熟预测。")
    return {
        "stockYears": year_groups, "stocks": stock_groups, "forecastRows": len(rows),
        "distinctForecastDates": int(rows.date.nunique()),
        "firstForecastDate": _iso(rows.date.min()),
        "lastMaturedLabelEnd": _iso(rows.label_end.max()),
    }


def _weighted_score(statistics: dict, target_symbol: str) -> dict:
    year_groups, stock_groups = statistics["stockYears"], statistics["stocks"]
    differences = np.asarray([group["brierDifference"] for group in year_groups])
    year_weights = np.asarray([
        TARGET_WEIGHT if group["symbol"] == target_symbol else OTHER_WEIGHT
        for group in year_groups
    ])
    stock_weights = np.asarray([
        TARGET_WEIGHT if group["symbol"] == target_symbol else OTHER_WEIGHT
        for group in stock_groups
    ])
    mean_difference = float(np.average(differences, weights=year_weights))
    instability = float(np.sqrt(np.average((differences - mean_difference) ** 2, weights=year_weights)))
    mean_balanced = float(np.average(
        [group["balancedAccuracy"] for group in stock_groups], weights=stock_weights,
    ))
    worst_downside = float(min(group["downRecall"] for group in stock_groups))
    penalty = 0.5 * max(0.0, 0.52 - mean_balanced) + 0.1 * max(0.0, 0.35 - worst_downside)
    return {
        "score": mean_difference + 0.5 * instability + penalty,
        "weightedMeanBrierDifference": mean_difference,
        "weightedYearStockBrierStd": instability,
        "weightedMeanBalancedAccuracy": mean_balanced,
        "worstStockDownRecall": worst_downside,
        "qualificationPenalty": penalty,
        "researchQualificationPassed": mean_difference < 0 and mean_balanced >= 0.52 and worst_downside >= 0.35,
        "stockYearWeights": [
            {"symbol": group["symbol"], "year": group["year"], "weight": float(weight)}
            for group, weight in zip(year_groups, year_weights)
        ],
        "stockWeights": [
            {"symbol": group["symbol"], "weight": float(weight)}
            for group, weight in zip(stock_groups, stock_weights)
        ],
    }


def _select(predictions: pd.DataFrame, candidate_ids: tuple[str, ...], targets: list[str], first_year: int, last_year: int):
    selections, outputs, statistics_audit = [], [], []
    ranks = {candidate: index for index, candidate in enumerate(candidate_ids)}
    for year in range(first_year, last_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        inside = predictions.loc[
            (predictions.date >= cutoff - pd.DateOffset(years=INNER_YEARS))
            & (predictions.label_end < cutoff) & predictions.symbol.isin(targets)
        ]
        summaries = {}
        reference_keys = None
        for candidate in candidate_ids:
            rows = inside.loc[inside.candidate == candidate].sort_values(["date", "symbol"])
            keys = list(zip(rows.date, rows.symbol, rows.label_start, rows.label_end, rows.target))
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise ValueError("加权选型候选未使用完全相同的成熟日期、证券和标签。")
            summaries[candidate] = _candidate_statistics(rows)
            statistics_audit.append({"selectionYear": year, "candidate": candidate, **summaries[candidate]})
        for symbol in targets:
            scored = [{"candidate": candidate, **_weighted_score(summaries[candidate], symbol)} for candidate in candidate_ids]
            winner = min(scored, key=lambda item: (item["score"], ranks[item["candidate"]]))
            selections.append({
                "symbol": symbol, "year": year, "selectedCandidate": winner["candidate"],
                "innerStart": _iso(cutoff - pd.DateOffset(years=INNER_YEARS)),
                "innerLastMaturedLabelEnd": _iso(inside.label_end.max()),
                "selectionInformationCutoffExclusive": _iso(cutoff),
                "targetMaturedForecastsPerCandidate": int(
                    ((inside.symbol == symbol) & (inside.candidate == candidate_ids[0])).sum(),
                ),
                "researchQualificationPassed": winner["researchQualificationPassed"],
                "candidates": scored,
            })
            current = predictions.loc[
                (predictions.symbol == symbol) & (predictions.date.dt.year == year)
                & (predictions.candidate == winner["candidate"])
            ].copy()
            current["selection_year"] = year
            current["selection_policy"] = POLICY_ID
            outputs.append(current)
    selected = pd.concat(outputs, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    if selected.duplicated(["symbol", "date"]).any():
        raise ValueError("独立年度选型产生重复股票预测日期。")
    return selected, selections, statistics_audit


def _evaluate(rows: pd.DataFrame, market: Market, symbol: str, calendar: pd.DatetimeIndex, volatility: pd.DataFrame) -> dict:
    end = calendar[-1]
    matured = rows.loc[rows.target.notna() & (rows.label_end <= end)]
    metrics = forecast_metrics(matured)
    recent = forecast_metrics(matured.loc[matured.date >= pd.Timestamp("2025-01-01")])
    yearly = []
    for year, group in matured.groupby(matured.date.dt.year):
        year_calendar = calendar[calendar.year == year]
        complete = year < end.year and group.date.min() <= year_calendar[4] and group.date.max() >= year_calendar[-6]
        yearly.append({"year": int(year), "completeCalendarYear": bool(complete), "metrics": forecast_metrics(group)})
    high_rows = matured.loc[np.maximum(matured.probability, 1 - matured.probability) >= 0.65]
    high_metrics = forecast_metrics(high_rows, compute_bootstrap=False) if not high_rows.empty else None
    trading = trading_metrics(market, rows, symbol, rows.date.min(), end, volatility)
    checks = _excellent_checks(metrics, trading, yearly, high_metrics)
    return {
        "evaluation": metrics, "recentEvaluation": recent, "yearlyEvaluation": yearly,
        "highConfidenceEvaluation": high_metrics, "trading": trading, "checks": checks,
        "historicalExcellent": all(check["passed"] for check in checks),
        "independentValidation": False, "liveReady": False,
    }


def _same_metric(actual, expected, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"原系统重算与封存报告不一致：{label}")
    elif not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"原系统重算与封存报告不一致：{label}")


def _run(progress) -> dict:
    source_run = ROOT / "runs" / "lab" / SOURCE_RUN_ID
    original = json.loads((source_run / "report.json").read_text(encoding="utf-8"))
    original_spec = json.loads((source_run / "specification.json").read_text(encoding="utf-8"))
    if original["run"]["id"] != SOURCE_RUN_ID or original["schemaVersion"] != 2:
        raise ValueError("选型实验要求指定的已完成公告研究第二版报告。")
    if EXCELLENCE_RULES != original_spec["excellenceRules"]:
        raise ValueError("当前优秀门槛与来源研究不一致，不能比较。")
    candidates = tuple(original_spec["candidates"])
    if len(candidates) != 40 or len(set(candidates)) != 40:
        raise ValueError("来源必须包含封存的四十个互不重复候选。")
    targets = [row["stock"]["symbol"] for row in original["stockReports"]]
    if set(targets) != {"00883", "02359", "00003", "00939"}:
        raise ValueError("选型股票必须为来源报告的原四只股票。")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    destination = RUNS / "selection" / run_id
    destination.mkdir(parents=True)
    sources = [ROOT / "quant" / name for name in DEPENDENCIES] + [ROOT / "requirements.txt"]
    source_hashes = {str(path.relative_to(ROOT)): _digest(path) for path in sources}
    input_hashes = {relative: _digest(source_run / relative) for relative in SOURCE_INPUTS}
    runtime = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "exchange-calendars", "joblib", "threadpoolctl",
    )}
    specification = {
        "policyId": POLICY_ID, "sourceRunId": SOURCE_RUN_ID, "candidateIds": candidates,
        "candidateRefitting": False, "policyCount": 1, "targetStockWeight": TARGET_WEIGHT,
        "otherOriginalStockWeight": OTHER_WEIGHT, "peerStocksEnterSelection": False,
        "innerYears": INNER_YEARS, "outerStartYear": original_spec["outerStartYear"],
        "informationRule": "每年年初只读取此前三年且 label_end 严格早于当年一月一日的成熟预测。",
        "score": "股票年度 Brier 差按目标股三、其余原股一加权均值，加同权重总体标准差的0.5倍，再加原惩罚；各股票平衡命中率也按三比一加权。",
        "downRecallPenalty": "最差股票下跌召回率仍取原始最小值，不改为加权均值。",
        "minimumForecastsPerStockYear": MINIMUM_GROUP_FORECASTS,
        "minimumForecastsPerStockRecall": MINIMUM_GROUP_FORECASTS,
        "singleClassStockRecall": "该股票不进入双向召回汇总，与来源评分规则一致。",
        "targetWithoutMaturedHistory": "评分集合只含实际存在且达原样本资格的股票记录；没有目标股记录时不生成该股票统计。",
        "instabilityCoefficient": 0.5,
        "penalty": {"balancedFloor": 0.52, "balancedWeight": 0.5, "worstDownRecallFloor": 0.35, "downWeight": 0.1},
        "tieBreak": "按封存四十个候选的原始顺序选择。",
        "annualRule": "每股独立选型，全年固定候选；直接采用来源研究已生成的逐时点样本外预测。",
        "comparison": "与来源报告共同年度选型使用完全相同股票日期、标签、涨频基准、波动率、交易起点和费用。",
        "excellenceRules": original_spec["excellenceRules"],
        "independentFutureObservationRequired": True,
        "sourceHashes": source_hashes, "inputHashes": input_hashes,
        "frozenAt": now.isoformat(),
        "researchNote": "全部来源历史均已查看；这是单个预先冻结的选型结构实验，不能消除历史反复研究偏差，也不会替换原部署模型。",
    }
    provenance = {
        "id": run_id, "sourceRunId": SOURCE_RUN_ID, "createdAt": now.isoformat(),
        "sourceHashes": source_hashes, "inputHashes": input_hashes,
        "runtime": runtime, "python": sys.version, "independentValidation": False,
    }
    _write(destination / "specification.json", specification)
    _write(destination / "provenance.json", provenance)
    snapshot = destination / "reproduction"
    for source in sources:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _digest(target) != source_hashes[str(source.relative_to(ROOT))]:
            raise ValueError("选型实验源文件复制期间发生变化。")
    frozen_source_run = snapshot / "runs" / "lab" / SOURCE_RUN_ID
    for relative in SOURCE_INPUTS:
        target = frozen_source_run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_run / relative, target)
        if _digest(target) != input_hashes[relative]:
            raise ValueError("来源研究输入复制期间发生变化。")
    progress(f"已冻结唯一选型政策与源指纹：{run_id}")
    date_columns = ["date", "label_start", "label_end"]
    predictions = pd.read_csv(
        frozen_source_run / "candidate-predictions.csv", dtype={"symbol": str},
        parse_dates=date_columns, float_precision="round_trip",
    )
    baseline = pd.read_csv(
        frozen_source_run / "selected-predictions.csv", dtype={"symbol": str},
        parse_dates=date_columns, float_precision="round_trip",
    )
    if set(predictions.candidate) != set(candidates) or predictions.duplicated(["candidate", "date", "symbol"]).any():
        raise ValueError("来源候选集合或唯一预测键无效。")
    values = predictions[["probability", "base_probability"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("来源预测概率必须有限且位于零与一之间。")
    prices, universe, _ = _read_inputs(frozen_source_run / "reproduction" / "data")
    calendar = research_calendar()
    if _iso(calendar[-1]) != original["summary"]["dataEnd"]:
        raise ValueError("交易日历截止日与来源研究不一致。")
    securities = universe["stocks"] + [universe["benchmark"]]
    market = Market.from_prices(
        prices, calendar, universe["benchmark"]["symbol"], universe["corporateActions"],
        {stock["symbol"]: stock["stampDutyExemptFrom"] for stock in securities},
    )
    volatility = pd.read_csv(
        frozen_source_run / "factors.csv", usecols=["date", "symbol", "annualVolatility"],
        dtype={"symbol": str}, parse_dates=["date"], float_precision="round_trip",
    )
    weighted, selections, selection_statistics = _select(
        predictions, candidates, targets, original_spec["outerStartYear"], calendar[-1].year,
    )
    weighted.to_csv(destination / "selected-predictions.csv", index=False, date_format="%Y-%m-%d")
    _write(destination / "selection.json", {"selections": selections})
    _write(destination / "selection-statistics.json", {"statistics": selection_statistics})
    reports = []
    matching_columns = ["date", "symbol", "label_start", "label_end", "target", "base_probability"]
    for original_row in original["stockReports"]:
        stock = original_row["stock"]
        symbol = stock["symbol"]
        progress(f"核算{stock['name']}的同日期预测与交易对照")
        current = weighted.loc[weighted.symbol == symbol].sort_values("date").reset_index(drop=True)
        control = baseline.loc[baseline.symbol == symbol].sort_values("date").reset_index(drop=True)
        if not current[matching_columns].equals(control[matching_columns]):
            raise ValueError(f"{symbol} 的新政策与原报告未覆盖完全相同的日期和真实标签。")
        evaluated = _evaluate(current, market, symbol, calendar, volatility)
        baseline_evaluated = _evaluate(control, market, symbol, calendar, volatility)
        for section in ("evaluation", "recentEvaluation"):
            for key in ("n", "accuracy", "balancedAccuracy", "brier", "baseBrier", "brierSkill"):
                _same_metric(baseline_evaluated[section][key], original_row[section][key], f"{symbol}/{section}/{key}")
            for key in ("start", "end", "labelEnd"):
                if baseline_evaluated[section][key] != original_row[section][key]:
                    raise ValueError("原系统重算日期与原报告不一致。")
        for key in ("cagr", "sharpe", "maxDrawdown", "annualExcess", "totalFees", "tradeCount"):
            _same_metric(baseline_evaluated["trading"]["strategy"][key], original_row["trading"]["strategy"][key], f"{symbol}/trading/{key}")
        original_choices = {item["year"]: item["selectedCandidate"] for item in original["annualSelections"]}
        annual = [{
            "year": item["year"], "weightedCandidate": item["selectedCandidate"],
            "commonCandidate": original_choices[item["year"]],
            "different": item["selectedCandidate"] != original_choices[item["year"]],
            "targetMaturedForecastsPerCandidate": item["targetMaturedForecastsPerCandidate"],
        } for item in selections if item["symbol"] == symbol]
        before, after = baseline_evaluated["evaluation"], evaluated["evaluation"]
        recent_before, recent_after = baseline_evaluated["recentEvaluation"], evaluated["recentEvaluation"]
        reports.append({
            "stock": stock, **evaluated, "commonSelection": baseline_evaluated,
            "sameForecastDates": True, "sourceReportReproduced": True, "annualSelections": annual,
            "comparison": {
                "accuracyChange": after["accuracy"] - before["accuracy"],
                "balancedAccuracyChange": after["balancedAccuracy"] - before["balancedAccuracy"],
                "brierImprovement": before["brier"] - after["brier"],
                "recentAccuracyChange": recent_after["accuracy"] - recent_before["accuracy"],
                "recentBrierImprovement": recent_before["brier"] - recent_after["brier"],
                "netCagrChange": evaluated["trading"]["strategy"]["cagr"] - baseline_evaluated["trading"]["strategy"]["cagr"],
            },
        })
    for relative, digest in input_hashes.items():
        if _digest(source_run / relative) != digest:
            raise ValueError("选型研究期间来源输入发生变化，不能标记完成。")
    for relative, digest in source_hashes.items():
        if _digest(ROOT / relative) != digest:
            raise ValueError("选型研究期间实际依赖源码发生变化，不能标记完成。")
    report = {
        "schemaVersion": 1, "run": {**provenance, "completedAt": datetime.now(timezone.utc).isoformat()},
        "specification": specification,
        "summary": {
            "policyId": POLICY_ID, "stocks": len(targets), "candidateCount": len(candidates),
            "historicalExcellentCount": sum(row["historicalExcellent"] for row in reports),
            "independentValidation": False, "liveReady": False, "modelsRefitted": 0,
        },
        "stockReports": reports,
        "limitations": [
            "这是查看原历史之后提出的单一结构实验；年初信息约束不消除研究者已经查看历史带来的偏差。",
            "只改变选型汇总权重，原模型训练、概率、费用、预测跨度及优秀门槛保持不变。",
            "每股每年只按当时成熟记录选择一次；没有用该年实际表现挑当年候选。",
            "完整未来独立观测仍缺失；达到历史门槛也不能自动认定优秀或上线。",
        ],
    }
    _write(destination / "report.json", report)
    progress(f"研究完成：{destination / 'report.json'}")
    return report


def execute_selection_research(progress=print) -> dict:
    directory = RUNS / "selection"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".research.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("另一项选型结构研究正在运行。") from exc
        try:
            with threadpool_limits(limits=1):
                return _run(progress)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    result = execute_selection_research()
    for item in result["stockReports"]:
        before, after = item["commonSelection"]["evaluation"], item["evaluation"]
        print(
            f"{item['stock']['name']}：命中率 {before['accuracy']:.1%} → {after['accuracy']:.1%}；"
            f"涨跌平衡 {before['balancedAccuracy']:.1%} → {after['balancedAccuracy']:.1%}；"
            f"Brier改善 {item['comparison']['brierImprovement']:+.5f}；"
            f"门槛通过 {sum(check['passed'] for check in item['checks'])}/{len(item['checks'])}",
        )
