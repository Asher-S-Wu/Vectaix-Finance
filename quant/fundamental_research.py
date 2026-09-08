from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as exchange
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import DATA_START, OFFICIAL_CLOSURES
from .hkma_data import RATE_FEATURE_NAMES, RATE_KEYS, build_hibor_factors
from .lab_binning import install_batch_binning
from .lab_data import build_lab_dataset
from .lab_evaluation import forecast_metrics, trading_metrics
from .lab_events import add_event_factors
from .lab_fundamentals import FUNDAMENTAL_FEATURE_NAMES, FUNDAMENTAL_KEYS, add_fundamental_factors
from .lab_models import (
    CALIBRATION_YEARS, HORIZON, MIN_CALIBRATION_DATES, MIN_TRAINING, MODEL_WORKERS,
    PREDICTION_STRIDE, REFIT_STRIDE, TRAINING_YEARS, Expert, _fit, _probability,
    apply_calibrator, fit_monotone_calibrator,
)
from .lab_training import EXCELLENCE_RULES, OUTER_START_YEAR, _excellent_checks, select_annual_configurations
from .portfolio import Market
from .runner import DATA, ROOT, RUNS


GROUPS = ("P", "F", "R", "FR")
GROUP_NAMES = {
    "P": "价格与已公布业绩事件控制组",
    "F": "控制组加财务增长",
    "R": "控制组加港元拆息",
    "FR": "控制组加财务增长及港元拆息",
}
EXPERTS = tuple(
    expert for group in GROUPS for expert in (
        Expert(f"{group}_local_logistic_01", "local", group, "logistic", 0.1),
        Expert(f"{group}_pool_tree_7", "pool", group, "tree", 7),
    )
)
EXPERT_IDS = tuple(expert.id for expert in EXPERTS)
CANDIDATE_IDS = tuple(f"{expert}:{mode}" for expert in EXPERT_IDS for mode in ("raw", "monotone"))
GROUP_CANDIDATES = {
    group: tuple(f"{expert.id}:{mode}" for expert in EXPERTS if expert.features == group for mode in ("raw", "monotone"))
    for group in GROUPS
}
DEPENDENCIES = (
    "__init__.py", "fundamental_research.py", "data.py", "engine.py", "factors.py",
    "hkma_data.py", "lab_binning.py", "lab_data.py", "lab_events.py", "lab_fundamentals.py",
    "lab_evaluation.py", "lab_models.py", "lab_training.py", "macro_data.py", "metrics.py",
    "portfolio.py", "runner.py", "trend_factors.py",
)
MATCH_COLUMNS = ("date", "symbol", "label_start", "label_end", "target", "base_probability")


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def build_research_dataset(data_dir: Path) -> dict:
    """Build one common four-stock cohort for all four fixed feature groups."""
    dataset = add_fundamental_factors(add_event_factors(build_lab_dataset(data_dir), data_dir), data_dir)
    rates, rate_manifest = build_hibor_factors(data_dir, dataset["calendar"])
    if rates.date.duplicated().any() or not (rates.hiborSourceDate < rates.date).all():
        raise ValueError("拆息必须为唯一信号日期，且对应定盘日期严格更早。")
    targets = list(dataset["target_symbols"])
    if set(targets) != {"00883", "02359", "00003", "00939"} or dataset["peer_symbols"]:
        raise ValueError("本轮财务与拆息研究只接受原四只股票，不使用同业训练行。")
    base = tuple(dataset["feature_groups"]["event_compact"])
    if len(base) != 18 or len(FUNDAMENTAL_KEYS) != 4 or len(RATE_KEYS) != 4:
        raise ValueError("固定实验必须为18项控制因子、4项财务因子及4项拆息因子。")
    groups = {
        "P": base, "F": (*base, *FUNDAMENTAL_KEYS), "R": (*base, *RATE_KEYS),
        "FR": (*base, *FUNDAMENTAL_KEYS, *RATE_KEYS),
    }
    if len(set(groups["FR"])) != 26:
        raise ValueError("固定四组因子键不能重复。")
    frames, coverage = {}, []
    for name in ("factors", "labels"):
        original = dataset[name]
        if original.duplicated(["date", "symbol"]).any() or set(original.symbol) != set(targets):
            raise ValueError("四股共同样本的日期键或股票池无效。")
        merged = original.merge(rates, on="date", how="inner", validate="many_to_one")
        if not np.isfinite(merged[list(groups["FR"])].to_numpy(dtype=float)).all():
            raise ValueError("四组共同样本仍含不可计算因子，不能补值或分组改变样本。")
        if set(merged.symbol) != set(targets):
            raise ValueError("财务与拆息的共同样本未覆盖原四股。")
        frames[name] = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
        for symbol in targets:
            before = original.loc[original.symbol == symbol]
            after = merged.loc[merged.symbol == symbol]
            excluded = before.loc[~before.date.isin(after.date), "date"]
            coverage.append({
                "table": name, "symbol": symbol, "beforeRateAlignment": len(before),
                "commonRows": len(after), "start": _iso(after.date.min()), "end": _iso(after.date.max()),
                "excludedRateDates": excluded.dt.strftime("%Y-%m-%d").tolist(),
            })
    factor_values = frames["factors"].set_index(["date", "symbol"])[list(groups["FR"])]
    label_values = frames["labels"].set_index(["date", "symbol"])[list(groups["FR"])]
    if not label_values.equals(factor_values.loc[label_values.index]):
        raise ValueError("标签训练行与同日预测因子不一致。")
    hashes = dict(dataset["provenance"]["fileHashes"])
    for relative, digest in rate_manifest["fileHashes"].items():
        if relative in hashes and hashes[relative] != digest:
            raise ValueError("拆息与原数据来源指纹冲突。")
        hashes[relative] = digest
    names = {**dataset["feature_names"], **FUNDAMENTAL_FEATURE_NAMES, **RATE_FEATURE_NAMES}
    prices = dataset["prices"].loc[dataset["prices"].symbol.isin(targets + ["02800"])].copy()
    provenance = {
        **dataset["provenance"], "fileHashes": hashes,
        "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "peerSymbols": [], "sourcePeerSymbols": dataset["provenance"]["sourcePeerSymbols"],
        "targetSymbols": targets, "priceRows": len(prices),
        "factorRows": len(frames["factors"]), "labelRows": len(frames["labels"]),
        "featureCount": 26, "featureGroupCounts": {key: len(value) for key, value in groups.items()},
        "commonSampleCoverage": coverage, "hibor": rate_manifest,
        "sampleRule": "四组使用同一四股完整因子行集合；即使控制组不输入财务或拆息，其训练、校准与评估行也保持相同。",
        "sourcePeerDataRole": "沿用原数据构造与证据校验过程读取同业来源；同业价格行、财务、训练、校准及选型均不进入本实验。",
    }
    return {
        **dataset, **frames, "prices": prices, "peer_symbols": [],
        "universe": {**dataset["universe"], "peerStocks": []},
        "feature_groups": groups, "feature_names": {key: names[key] for key in groups["FR"]},
        "provenance": provenance,
    }


def _fit_predict(train: pd.DataFrame, current: pd.DataFrame, feature_groups: dict):
    counts = train.groupby("symbol").size()
    eligible = counts.index[counts >= MIN_TRAINING].tolist()
    observed = current.loc[current.symbol.isin(eligible)].copy()
    training = train.loc[train.symbol.isin(eligible)].copy()
    if observed.empty:
        raise ValueError("当前历史时点没有达到504条成熟训练记录的目标股票。")

    def fit_expert(expert: Expert):
        columns = list(feature_groups[expert.features])
        bundle = {"specification": asdict(expert), "columns": columns, "models": {}}
        probabilities = pd.Series(index=observed.index, dtype=float)
        scopes = [(symbol, [symbol]) for symbol in observed.symbol.unique()] if expert.scope == "local" else [("all", eligible)]
        for key, symbols in scopes:
            current_group = observed.loc[observed.symbol.isin(symbols)]
            train_group = training.loc[training.symbol.isin(symbols)]
            model = _fit(expert, train_group, columns)
            probabilities.loc[current_group.index] = _probability(expert, model, current_group, columns)
            bundle["models"][key] = model
        if probabilities.isna().any():
            raise ValueError("固定专家没有覆盖共同预测行。")
        output = observed[["date", "symbol"]].assign(expert=expert.id, raw_probability=probabilities.to_numpy())
        return expert.id, bundle, output

    bundles, outputs = {}, []
    with ThreadPoolExecutor(max_workers=MODEL_WORKERS, initializer=threadpool_limits, initargs=(1,)) as executor:
        for expert, bundle, output in executor.map(fit_expert, EXPERTS):
            bundles[expert] = bundle
            outputs.append(output)
    audit = {
        "commonTrainingRows": len(training), "eligibleSymbols": eligible,
        "trainingRowsByStock": {symbol: int(counts[symbol]) for symbol in eligible},
        "distinctTrainingSignalDates": int(training.date.nunique()),
        "distinctFinancialDisclosuresByStock": {
            symbol: int(group.resultsEventId.nunique()) for symbol, group in training.groupby("symbol")
        },
        "lastMaturedLabelEnd": _iso(training.label_end.max()),
        "currentRowsPerExpert": len(observed), "expertCount": len(EXPERTS),
        "trainingKeysSha256": hashlib.sha256(training[["date", "symbol", "label_end"]].to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest(),
    }
    return pd.concat(outputs, ignore_index=True), bundles, audit


def _add_base_frequency(raw: pd.DataFrame, train: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    historical = train.loc[train.date.isin(calendar[::HORIZON])]
    rates = historical.groupby("symbol").target.agg(["size", lambda values: int((values > 0).sum())])
    rates.columns = ["observations", "up"]
    rates["base_probability"] = (rates.up + 1) / (rates.observations + 2)
    output = raw.merge(rates[["base_probability"]], left_on="symbol", right_index=True, how="left", validate="many_to_one")
    if output.base_probability.isna().any():
        raise ValueError("共同训练样本不足以计算该股票的固定涨频基准。")
    return output


def _raw_walk_forward(dataset: dict, destination: Path, progress):
    install_batch_binning()
    factors, labels, calendar = dataset["factors"], dataset["labels"], dataset["calendar"]
    forecast_dates = calendar[::PREDICTION_STRIDE]
    refits = [date for date in calendar[::REFIT_STRIDE] if date >= pd.Timestamp("2014-01-01")]
    records, audits = [], []
    for index, signal in enumerate(refits):
        stop = refits[index + 1] if index + 1 < len(refits) else calendar[-1] + pd.Timedelta(days=1)
        current = factors.loc[(factors.date >= signal) & (factors.date < stop) & factors.date.isin(forecast_dates)]
        if current.empty:
            continue
        train = labels.loc[(labels.date >= signal - pd.DateOffset(years=TRAINING_YEARS)) & (labels.label_end < signal)]
        raw, _, audit = _fit_predict(train, current, dataset["feature_groups"])
        raw = _add_base_frequency(raw, train, calendar)
        raw["fit_date"] = signal
        raw["training_last_label_end"] = train.label_end.max()
        raw = raw.merge(labels[["date", "symbol", "label_start", "label_end", "target"]],
                        on=["date", "symbol"], how="left", validate="many_to_one")
        records.append(raw)
        audits.append({"date": _iso(signal), "nextRefitExclusive": _iso(stop), **audit})
        if index % 6 == 0 or index == len(refits) - 1:
            pd.concat(records, ignore_index=True).to_csv(destination / "raw-predictions.csv", index=False, date_format="%Y-%m-%d")
            progress(f"固定八专家训练：{index + 1}/{len(refits)} 个历史时点")
    if not records:
        raise ValueError("本轮没有完成任何原始样本外预测。")
    history = pd.concat(records, ignore_index=True).sort_values(["expert", "date", "symbol"]).reset_index(drop=True)
    signal = calendar[-1]
    train = labels.loc[(labels.date >= signal - pd.DateOffset(years=TRAINING_YEARS)) & (labels.label_end < signal)]
    current = factors.loc[factors.date == signal]
    if set(current.symbol) != set(dataset["target_symbols"]):
        raise ValueError("数据截止日未覆盖全部四股的完整财务及拆息因子。")
    latest, models, latest_audit = _fit_predict(train, current, dataset["feature_groups"])
    if set(latest.symbol) != set(dataset["target_symbols"]):
        raise ValueError("最新模型未达到四股各自的最小训练样本要求。")
    latest = _add_base_frequency(latest, train, calendar)
    latest["fit_date"] = signal
    latest["training_last_label_end"] = train.label_end.max()
    return history, latest, models, audits, {"date": _iso(signal), **latest_audit}


def _calibrate(history: pd.DataFrame, latest_raw: pd.DataFrame, progress):
    rows, latest_rows, calibrators, audits = [], [], {}, []
    for index, expert in enumerate(EXPERT_IDS):
        past = history.loc[history.expert == expert].sort_values(["date", "symbol"])
        for signal, current in past.groupby("date", sort=True):
            matured = past.loc[(past.date >= signal - pd.DateOffset(years=CALIBRATION_YEARS)) & (past.label_end < signal)]
            if matured.date.nunique() < MIN_CALIBRATION_DATES:
                continue
            calibrator = fit_monotone_calibrator(matured)
            rows.extend([
                current.assign(candidate=f"{expert}:raw", probability=current.raw_probability),
                current.assign(candidate=f"{expert}:monotone", probability=apply_calibrator(calibrator, current.raw_probability)),
            ])
            audits.append({"expert": expert, "date": _iso(signal), **calibrator})
        current = latest_raw.loc[latest_raw.expert == expert]
        signal = current.date.iloc[0]
        matured = past.loc[(past.date >= signal - pd.DateOffset(years=CALIBRATION_YEARS)) & (past.label_end < signal)]
        if matured.date.nunique() < MIN_CALIBRATION_DATES:
            raise ValueError("最新候选没有足够的共同四股样本外校准记录。")
        calibrator = fit_monotone_calibrator(matured)
        calibrators[expert] = calibrator
        latest_rows.extend([
            current.assign(candidate=f"{expert}:raw", probability=current.raw_probability),
            current.assign(candidate=f"{expert}:monotone", probability=apply_calibrator(calibrator, current.raw_probability)),
        ])
        progress(f"四股共同历史概率校准：{index + 1}/{len(EXPERT_IDS)}")
    if not rows:
        raise ValueError("没有达到固定历史校准要求的候选预测。")
    return pd.concat(rows, ignore_index=True), pd.concat(latest_rows, ignore_index=True), calibrators, audits


def _same_prediction_keys(predictions: pd.DataFrame) -> None:
    if set(predictions.candidate) != set(CANDIDATE_IDS) or predictions.duplicated(["candidate", "date", "symbol"]).any():
        raise ValueError("新实验必须完整包含16个候选的唯一预测行。")
    reference = predictions.loc[predictions.candidate == CANDIDATE_IDS[0], list(MATCH_COLUMNS)].sort_values(["date", "symbol"]).reset_index(drop=True)
    for candidate in CANDIDATE_IDS[1:]:
        observed = predictions.loc[predictions.candidate == candidate, list(MATCH_COLUMNS)].sort_values(["date", "symbol"]).reset_index(drop=True)
        if not reference.equals(observed):
            raise ValueError("各组候选的日期、真实标签或涨频基准不一致。")


def _evaluate(rows: pd.DataFrame, market: Market, symbol: str, calendar: pd.DatetimeIndex, factors: pd.DataFrame) -> dict:
    end = calendar[-1]
    matured = rows.loc[rows.target.notna() & (rows.label_end <= end)]
    expected = calendar[::PREDICTION_STRIDE]
    expected = expected[(expected >= matured.date.min()) & (expected <= matured.date.max())]
    if not pd.DatetimeIndex(matured.date).equals(expected):
        raise ValueError("成熟评估记录存在缺口，不能把不等距样本用于固定六预测连续区块估计。")
    metrics = forecast_metrics(matured)
    recent = forecast_metrics(matured.loc[matured.date >= pd.Timestamp("2025-01-01")])
    yearly = []
    for year, group in matured.groupby(matured.date.dt.year):
        year_calendar = calendar[calendar.year == year]
        complete = year < end.year and group.date.min() <= year_calendar[4] and group.date.max() >= year_calendar[-6]
        yearly.append({"year": int(year), "completeCalendarYear": bool(complete), "metrics": forecast_metrics(group)})
    high_rows = matured.loc[np.maximum(matured.probability, 1 - matured.probability) >= 0.65]
    high = forecast_metrics(high_rows, compute_bootstrap=False) if not high_rows.empty else None
    trading = trading_metrics(market, rows, symbol, rows.date.min(), end, factors)
    checks = _excellent_checks(metrics, trading, yearly, high)
    return {
        "evaluation": metrics, "recentEvaluation": recent, "yearlyEvaluation": yearly,
        "highConfidenceEvaluation": high, "trading": trading, "checks": checks,
        "historicalExcellent": all(check["passed"] for check in checks),
        "independentValidation": False, "liveReady": False,
    }


def _specification(dataset: dict, source_hashes: dict, frozen_at: str) -> dict:
    return {
        "experimentId": "four_stock_financial_hibor_fixed_groups_v1", "frozenAt": frozen_at,
        "experts": [asdict(expert) for expert in EXPERTS], "candidateCount": len(CANDIDATE_IDS),
        "candidates": CANDIDATE_IDS, "groupCandidates": GROUP_CANDIDATES,
        "groupNames": GROUP_NAMES, "featureGroups": dataset["feature_groups"], "featureNames": dataset["feature_names"],
        "trainingSymbols": dataset["target_symbols"], "peerTrainingRows": 0,
        "financialDisclosuresByStock": {
            row["symbol"]: row["documents"] for row in dataset["provenance"]["fundamentalDocuments"]
        },
        "financialSampleCountRule": "公告数量按独立原公告计数；沿用同一已知财务值的日度行不当作新增财报，跨股同日也不是独立四倍市场观测。",
        "comparison": "四组相同训练行、预测行、校准成熟样本及评估日期；各组独立在其四个候选内年度选型，总系统在完整16候选内选型。",
        "controlGroup": "P含12项原compact因子与6项公告事件因子；是四股共同样本上重新训练的控制组，不是旧十二股系统。",
        "financialGrowth": "只取同份已公布公告的当期及同期比较数值；同比=2*(当期-同期)/(绝对当期+绝对同期)，双零不准入；归母利润/营业收入比例取同期差。",
        "financialTiming": "严格复用已验证公告的实际港股收盘后10分钟信号；最新公告数值不可计算时排除该公告对应行，不沿用更早财务。",
        "hiborTiming": dataset["provenance"]["hibor"]["timingRule"],
        "hiborVintageCaveat": dataset["provenance"]["hibor"]["vintageCaveat"],
        "trainingYears": TRAINING_YEARS, "minimumTrainingRowsPerStock": MIN_TRAINING,
        "horizonTradingDays": HORIZON, "predictionStride": PREDICTION_STRIDE, "refitStride": REFIT_STRIDE,
        "firstRefitNotBefore": "2014-01-01", "calendarAnchor": _iso(dataset["calendar"][0]),
        "modelWorkers": MODEL_WORKERS, "mathThreadsPerFit": 1,
        "logistic": {"C": 0.1, "maxIterations": 1000, "scope": "每股单独训练"},
        "tree": {"maxLeafNodes": 7, "maxDepth": 2, "iterations": 100, "learningRate": 0.03,
                 "minimumLeafRowsPerEligibleStock": 63, "l2Regularization": 1.0, "earlyStopping": False,
                 "seed": 42, "scope": "达到504条成熟样本的原目标股共同训练"},
        "trainingWeights": "本股每行1/21；共享树每个日期全部合格目标股票合计权重1/21。",
        "constantFeatures": "仅在每个线性模型训练窗口内剔除完全不变化的列，再按历史样本权重标准化。",
        "weightedBinning": "沿用固定scikit-learn 1.9.0的已核验批量分箱实现，不改权重或树模型。",
        "calibration": {"years": CALIBRATION_YEARS, "minimumDistinctForecastDates": MIN_CALIBRATION_DATES,
                        "scope": "只汇集原四股该专家的历史样本外预测；每日期等权。",
                        "slopeMinimum": 0, "identityPenalty": 1.0,
                        "rawEligibility": "raw与monotone都从相同校准历史门槛满足时开始进入候选。"},
        "baseProbability": "同一五年成熟训练行中按全局21日网格抽取本股标签，(上涨数+1)/(观测数+2)。",
        "outerStartYear": OUTER_START_YEAR, "innerYears": 3,
        "annualSelection": "年初只用此前三年且label_end严格早于当年一月一日的成熟样本；年度内固定候选类型。所有组与总系统均用原共同选型。",
        "score": "股票年度Brier差等权均值+0.5倍总体标准差+0.5*max(0,0.52-各股平衡命中均值)+0.1*max(0,0.35-最差股票下跌召回)。",
        "minimumForecastsPerStockYear": 24, "minimumForecastsPerStockRecall": 24,
        "tieBreak": "按各组及总候选清单的预先固定顺序。",
        "excellenceRules": EXCELLENCE_RULES, "independentFutureObservationRequired": True,
        "latestRefit": "截止日另外重训最新模型并单独保存，绝不覆盖历史定期预测。",
        "trading": {"entryProbability": 0.55, "annualVolatilityTarget": 0.15, "maximumPosition": 1.0,
                    "signalEverySessions": 5, "execution": "共用原下一开盘执行引擎、历史费用和双倍滑点压力对照"},
        "uncertainty": {"bootstrapBlockForecasts": 6, "replications": 2000, "seed": 42,
                        "irregularHighConfidenceBootstrap": False},
        "sourceHashes": source_hashes, "dataHashes": dataset["provenance"]["fileHashes"],
        "researchNote": "全部历史已经在前序研究中查看；这是预先固定16候选的新结构实验，不能把时间隔离重演称为真正独立验证。",
        "liveReady": False,
    }


def _latest_forecast(symbol: str, candidate: str, latest: pd.DataFrame, calibrators: dict,
                     calendar: pd.DatetimeIndex, future_calendar, future: pd.DatetimeIndex) -> dict:
    selected = latest.loc[(latest.symbol == symbol) & (latest.candidate == candidate)]
    if len(selected) != 1:
        raise ValueError("最新年度候选必须对本股产生唯一概率。")
    end = calendar[-1]
    position = future.get_loc(end)
    generated_at = datetime.now(timezone.utc)
    started = pd.Timestamp(generated_at) >= future_calendar.session_open(future[position + 1])
    probability = float(selected.probability.iloc[0])
    expert, mode = candidate.split(":")
    return {
        "asOf": _iso(end), "returnStart": _iso(future[position + 1]),
        "returnEnd": _iso(future[position + HORIZON + 1]), "horizonTradingDays": HORIZON,
        "candidate": candidate, "upProbability": probability,
        "direction": "偏向上涨" if probability >= 0.5 else "偏向下跌或持平",
        "calibrationMode": mode, "calibration": {**calibrators[expert], "applied": mode == "monotone"},
        "generatedAt": generated_at.isoformat(), "targetWindowAlreadyStarted": bool(started),
        "prospectiveAtIssue": not bool(started), "futureOutcomeObserved": False,
        "interpretation": "生成时目标窗口已开始，属于较早信息截面的重建预测。" if started else "生成时目标窗口尚未开始，模型仍处于探索研究状态。",
        "liveReady": False,
    }


def _execute_locked(progress) -> dict:
    source_files = [ROOT / "quant" / name for name in DEPENDENCIES] + [ROOT / "requirements.txt"]
    source_hashes = {str(path.relative_to(ROOT)): _digest(path) for path in source_files}
    progress("校验原四股财务、官方公告与六期限港元拆息，构造共同样本")
    dataset = build_research_dataset(DATA)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    destination = RUNS / "fundamental" / run_id
    destination.mkdir(parents=True)
    specification = _specification(dataset, source_hashes, now.isoformat())
    runtime = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "exchange-calendars", "joblib", "threadpoolctl",
    )}
    record = {"id": run_id, "createdAt": now.isoformat(), "sourceHashes": source_hashes,
              "dataset": dataset["provenance"], "runtime": runtime, "python": sys.version,
              "independentValidation": False, "liveReady": False,
              "evaluationNote": "所有历史已经被查看；财务与拆息的新增结构实验仍属探索性历史研究。"}
    _write(destination / "specification.json", specification)
    _write(destination / "provenance.json", record)
    snapshot = destination / "reproduction"
    for relative, digest in dataset["provenance"]["fileHashes"].items():
        source, target = DATA / relative, snapshot / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _digest(target) != digest:
            raise ValueError("冻结数据时源文件发生变化。")
    for source in source_files:
        relative = str(source.relative_to(ROOT))
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _digest(target) != source_hashes[relative]:
            raise ValueError("冻结源码时实际依赖文件发生变化。")
    dataset["factors"].to_csv(destination / "factors.csv", index=False, date_format="%Y-%m-%d")
    dataset["labels"].to_csv(destination / "labels.csv", index=False, date_format="%Y-%m-%d")
    progress(f"已冻结4组、8专家、16候选及源与数据指纹：{run_id}")
    raw, latest_raw, models, fit_audits, latest_fit_audit = _raw_walk_forward(dataset, destination, progress)
    raw.to_csv(destination / "raw-predictions.csv", index=False, date_format="%Y-%m-%d")
    latest_raw.to_csv(destination / "latest-raw-predictions.csv", index=False, date_format="%Y-%m-%d")
    joblib.dump(models, destination / "raw-models.joblib")
    _write(destination / "fit-audit.json", {"fits": fit_audits, "latestFit": latest_fit_audit})
    calibrated, latest, calibrators, calibration_audits = _calibrate(raw, latest_raw, progress)
    _same_prediction_keys(calibrated)
    calibrated.to_csv(destination / "candidate-predictions.csv", index=False, date_format="%Y-%m-%d")
    latest.to_csv(destination / "latest-predictions.csv", index=False, date_format="%Y-%m-%d")
    _write(destination / "calibration-audit.json", {"calibrations": calibration_audits})
    end = dataset["calendar"][-1]
    chosen, annual_selections = {}, {}
    for group, candidates in {**GROUP_CANDIDATES, "ALL": CANDIDATE_IDS}.items():
        selected, selections = select_annual_configurations(calibrated, end.year, dataset["target_symbols"], candidates)
        for selection in selections:
            selection["scope"] = "只按原四只股票的成熟历史预测选择共同年度配置；同业不进入训练、校准或选型。"
            selection["group"] = group
        chosen[group], annual_selections[group] = selected, selections
        selected.to_csv(destination / f"{group}-selected-predictions.csv", index=False, date_format="%Y-%m-%d")
    _write(destination / "selection.json", {"groupSelections": annual_selections})
    selected_candidates = {group: rows[-1]["selectedCandidate"] for group, rows in annual_selections.items()}
    bundle = {"models": models, "calibrators": calibrators, "selectedCandidates": selected_candidates,
              "asOf": _iso(end), "featureGroups": dataset["feature_groups"], "specification": specification,
              "trainingSymbols": dataset["target_symbols"], "liveReady": False, "independentValidation": False}
    joblib.dump(bundle, destination / "models.joblib")
    securities = dataset["universe"]["stocks"] + [dataset["universe"]["benchmark"]]
    market = Market.from_prices(dataset["prices"], dataset["calendar"], "02800", dataset["universe"]["corporateActions"],
                                {stock["symbol"]: stock["stampDutyExemptFrom"] for stock in securities})
    future_calendar = exchange.get_calendar("XHKG", start=DATA_START, end=end + pd.Timedelta(days=90))
    future = future_calendar.sessions.difference(pd.DatetimeIndex(OFFICIAL_CLOSURES))
    reports = []
    for stock in dataset["universe"]["stocks"]:
        symbol = stock["symbol"]
        progress(f"核算{stock['name']}五个年度系统的共同日期预测与扣费交易")
        evaluated = {}
        control = chosen["P"].loc[chosen["P"].symbol == symbol].sort_values("date").reset_index(drop=True)
        for group, selected in chosen.items():
            rows = selected.loc[selected.symbol == symbol].sort_values("date").reset_index(drop=True)
            if not rows[list(MATCH_COLUMNS)].equals(control[list(MATCH_COLUMNS)]):
                raise ValueError("各组年度系统没有使用完全相同的日期、标签及涨频基准。")
            result = _evaluate(rows, market, symbol, dataset["calendar"], dataset["factors"])
            result["latestForecast"] = _latest_forecast(symbol, selected_candidates[group], latest, calibrators,
                                                        dataset["calendar"], future_calendar, future)
            result["currentSelectionQualified"] = annual_selections[group][-1]["researchQualificationPassed"]
            evaluated[group] = result
        baseline = evaluated["P"]
        comparisons = {
            group: {
                "sameForecastDates": True, "controlGroup": "P",
                "accuracyChange": result["evaluation"]["accuracy"] - baseline["evaluation"]["accuracy"],
                "brierImprovement": baseline["evaluation"]["brier"] - result["evaluation"]["brier"],
                "recentAccuracyChange": result["recentEvaluation"]["accuracy"] - baseline["recentEvaluation"]["accuracy"],
                "recentBrierImprovement": baseline["recentEvaluation"]["brier"] - result["recentEvaluation"]["brier"],
                "netCagrChange": result["trading"]["strategy"]["cagr"] - baseline["trading"]["strategy"]["cagr"],
            } for group, result in evaluated.items() if group != "P"
        }
        disclosure_audit = next(row for row in dataset["provenance"]["fundamentalDocuments"] if row["symbol"] == symbol)
        reports.append({"stock": stock, **evaluated["ALL"], "financialDisclosureCoverage": disclosure_audit,
                        "groupReports": {group: evaluated[group] for group in GROUPS},
                        "comparisonsToControl": comparisons, "sameForecastDates": True,
                        "status": "通过历史门槛但尚无真正未来独立验证" if evaluated["ALL"]["historicalExcellent"] else "未通过优秀模型门槛"})
    for relative, digest in dataset["provenance"]["fileHashes"].items():
        if _digest(DATA / relative) != digest:
            raise ValueError("训练期间输入文件已改变，本批次不能标记完成。")
    for relative, digest in source_hashes.items():
        if _digest(ROOT / relative) != digest:
            raise ValueError("训练期间实际依赖源码已改变，本批次不能标记完成。")
    report = {
        "schemaVersion": 1, "run": {**record, "completedAt": datetime.now(timezone.utc).isoformat()},
        "specification": specification, "annualSelections": annual_selections, "stockReports": reports,
        "summary": {"stocks": 4, "learningStocks": 4, "factorCount": 26, "candidateCount": 16,
                    "groupCount": 4, "dataEnd": _iso(end),
                    "financialDisclosureCount": sum(row["documents"] for row in dataset["provenance"]["fundamentalDocuments"]),
                    "financialDisclosuresByStock": specification["financialDisclosuresByStock"],
                    "commonDailyFactorRows": len(dataset["factors"]), "commonDailyLabelRows": len(dataset["labels"]),
                    "historicalExcellentCount": sum(row["historicalExcellent"] for row in reports),
                    "groupHistoricalExcellentCounts": {group: sum(row["groupReports"][group]["historicalExcellent"] for row in reports) for group in GROUPS},
                    "independentValidation": False, "liveReady": False},
        "limitations": [
            "所有历史已经查看；预先固定新候选与逐年时间隔离仍不能消除反复研究带来的选择偏差。",
            "本次控制组也按共同财务及拆息可用行重新训练；不能将其直接称为旧十二股系统的原报告表现。",
            "财务增长只用同一公告当时提供的比较列；每股收益按该公告口径，不跨公告拼接或回写重述值。",
            "银行营业收入与其他行业营业额口径不同；归母利润占收入比例因子只使用同股同期变化，不是通用毛利率。",
            "低频财报在两次披露之间保持同一已知信息，日度训练行并不代表独立财务信息数量；共享股票也受同一市场影响。",
            "拆息固定滞后一个真实定盘日；官方历史序列为当前取得的版本，无法证明没有后续修正。",
            "21日标签重叠，连续六次预测区块估算不确定性；高置信筛选后不等距记录不做该区块估计。",
            "满仓和同波动目标基准均使用共同交易起点及费用；模拟不是实际账户下单或整手股数。",
            "固定15项历史门槛与未来独立验证要求未改变，历史通过仍不自动成为优秀或可上线模型。",
            "截止日最新模型单独保存；若生成时目标窗口已开始，最新预测只是重建，不能当作提前发出的预测。",
        ],
    }
    _write(destination / "report.json", report)
    pending = destination.parent / f"latest.{run_id}.pending.json"
    _write(pending, {"id": run_id})
    pending.replace(destination.parent / "latest.json")
    progress(f"财务与拆息结构研究完成：{destination / 'report.json'}")
    return report


def execute_fundamental_research(progress=print) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".research.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("另一个量化研究仍在运行，不能同时启动财务拆息实验。") from exc
        try:
            with threadpool_limits(limits=1):
                return _execute_locked(progress)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    result = execute_fundamental_research()
    for item in result["stockReports"]:
        print(f"{item['stock']['name']}：总系统命中率 {item['evaluation']['accuracy']:.1%}；"
              f"通过历史门槛 {sum(check['passed'] for check in item['checks'])}/{len(item['checks'])}；"
              "真正未来独立验证尚未完成")
