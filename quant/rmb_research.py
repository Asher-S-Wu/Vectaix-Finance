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
from .fundamental_research import _add_base_frequency, _evaluate, _latest_forecast
from .hkma_data import RATE_FEATURE_NAMES, RATE_KEYS, build_hibor_factors
from .lab_binning import install_batch_binning
from .lab_data import build_lab_dataset
from .lab_events import add_event_factors
from .lab_fundamentals import FUNDAMENTAL_FEATURE_NAMES, FUNDAMENTAL_KEYS, add_fundamental_factors
from .lab_models import (
    CALIBRATION_YEARS, HORIZON, MIN_CALIBRATION_DATES, MIN_TRAINING, MODEL_WORKERS,
    PREDICTION_STRIDE, REFIT_STRIDE, TRAINING_YEARS, Expert, _fit, _probability,
    apply_calibrator, fit_monotone_calibrator,
)
from .lab_training import EXCELLENCE_RULES, OUTER_START_YEAR, select_annual_configurations
from .portfolio import Market
from .runner import DATA, ROOT, RUNS
from .shibor_data import SHIBOR_FEATURE_NAMES, SHIBOR_KEYS, build_shibor_factors


GROUPS = ("RICH", "RMB")
GROUP_NAMES = {
    "RICH": "行业市场、公告、财务及港元拆息控制组",
    "RMB": "控制组加人民币拆息与港元减人民币一月利差",
}
SPREAD_KEY = "hkd_cny_1m_spread"
SPREAD_NAME = "严格早于信号日各自最近已知的港元减人民币一月定盘利差，年利率小数"
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
    "__init__.py", "rmb_research.py", "fundamental_research.py", "data.py", "engine.py", "factors.py",
    "hkma_data.py", "shibor_data.py", "lab_binning.py", "lab_data.py", "lab_events.py", "lab_fundamentals.py",
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
    """Preserve the full event-driver recipe and admit one common four-stock cohort."""
    dataset = add_fundamental_factors(add_event_factors(build_lab_dataset(data_dir), data_dir), data_dir)
    calendar = dataset["calendar"]
    hkd, hkd_manifest = build_hibor_factors(data_dir, calendar)
    cny, cny_manifest = build_shibor_factors(data_dir, calendar)
    for name, frame, source_column, keys in (
        ("港元", hkd, "hiborSourceDate", RATE_KEYS), ("人民币", cny, "shiborSourceDate", SHIBOR_KEYS),
    ):
        if (frame.date.duplicated().any() or not frame.date.isin(calendar).all()
                or not (frame[source_column] < frame.date).all()
                or not np.isfinite(frame[list(keys)].to_numpy(dtype=float)).all()):
            raise ValueError(f"{name}拆息必须是唯一交易信号日对应的更早真实定盘，数值不可缺失。")
    rates = hkd.merge(cny, on="date", how="inner", validate="one_to_one")
    rates[SPREAD_KEY] = rates.hibor_1m - rates.shibor_1m
    rates["crossCurrencySourceDateGapCalendarDays"] = (rates.hiborSourceDate - rates.shiborSourceDate).dt.days
    if (rates.empty or not (rates.hiborSourceDate < rates.date).all()
            or not (rates.shiborSourceDate < rates.date).all()
            or not np.isfinite(rates[SPREAD_KEY].to_numpy(dtype=float)).all()):
        raise ValueError("跨币种利差只能用两地各自已知的真实定盘，不生成当日或未来利率。")
    targets = list(dataset["target_symbols"])
    if set(targets) != {"00883", "02359", "00003", "00939"} or dataset["peer_symbols"]:
        raise ValueError("人民币结构研究只接受原四股，不接纳同行训练记录。")
    base = tuple(dataset["feature_groups"]["event_drivers"])
    if len(base) != 45 or tuple(map(len, (FUNDAMENTAL_KEYS, RATE_KEYS, SHIBOR_KEYS))) != (4, 4, 4):
        raise ValueError("固定配方必须包含45项行业与公告因子，以及各4项财务、港元和人民币因子。")
    rich = (*base, *FUNDAMENTAL_KEYS, *RATE_KEYS)
    groups = {"RICH": rich, "RMB": (*rich, *SHIBOR_KEYS, SPREAD_KEY)}
    if len(set(groups["RICH"])) != 53 or len(set(groups["RMB"])) != 58:
        raise ValueError("固定两组必须有53及58项互不重复输入。")
    frames, coverage = {}, []
    for name in ("factors", "labels"):
        original = dataset[name]
        if original.duplicated(["date", "symbol"]).any() or set(original.symbol) != set(targets):
            raise ValueError("原四股因子及标签日期必须唯一。")
        merged = original.merge(rates, on="date", how="inner", validate="many_to_one")
        if set(merged.symbol) != set(targets) or not np.isfinite(merged[list(groups["RMB"])].to_numpy(dtype=float)).all():
            raise ValueError("两组共同样本不能完整覆盖四股或仍有不可计算因子。")
        frames[name] = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
        for symbol in targets:
            before, after = original.loc[original.symbol == symbol], merged.loc[merged.symbol == symbol]
            excluded = before.loc[~before.date.isin(after.date), "date"]
            coverage.append({"table": name, "symbol": symbol, "beforeRateAlignment": len(before), "commonRows": len(after),
                             "start": _iso(after.date.min()), "end": _iso(after.date.max()),
                             "excludedRateDates": excluded.dt.strftime("%Y-%m-%d").tolist()})
    factor_values = frames["factors"].set_index(["date", "symbol"])[list(groups["RMB"])]
    label_values = frames["labels"].set_index(["date", "symbol"])[list(groups["RMB"])]
    if not label_values.equals(factor_values.loc[label_values.index]):
        raise ValueError("共同训练标签与同日58项输入不一致。")
    hashes = dict(dataset["provenance"]["fileHashes"])
    for manifest in (hkd_manifest, cny_manifest):
        for relative, digest in manifest["fileHashes"].items():
            if relative in hashes and hashes[relative] != digest:
                raise ValueError("来源文件在两个数据集中具有不同指纹。")
            hashes[relative] = digest
    names = {**dataset["feature_names"], **FUNDAMENTAL_FEATURE_NAMES, **RATE_FEATURE_NAMES,
             **SHIBOR_FEATURE_NAMES, SPREAD_KEY: SPREAD_NAME}
    prices = dataset["prices"].loc[dataset["prices"].symbol.isin(targets + ["02800"])].copy()
    provenance = {
        **dataset["provenance"], "fileHashes": hashes,
        "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "peerSymbols": [], "sourcePeerSymbols": dataset["provenance"]["sourcePeerSymbols"],
        "targetSymbols": targets, "priceRows": len(prices), "factorRows": len(frames["factors"]),
        "labelRows": len(frames["labels"]), "featureCount": 58,
        "featureGroupCounts": {key: len(value) for key, value in groups.items()},
        "commonSampleCoverage": coverage, "hibor": hkd_manifest, "shibor": cny_manifest,
        "sampleRule": "RICH和RMB使用同一完整58项四股样本，控制组也只在人民币因子可计算的共同训练行与预测日期参与。",
        "crossCurrencySpreadRule": "港元一月减人民币一月利率，均为年利率小数，各自来源日期严格早于香港信号日；两地最近定盘可以不是同一天。",
        "crossCurrencyMetadata": "两地来源日期、来源年龄及日期差只作审计信息，不作为额外模型输入。",
        "sourcePeerDataRole": "原十二股数据构造过程只提供来源校验及原特征配方，同行价格行不进入本轮训练、校准或选型。",
    }
    return {**dataset, **frames, "prices": prices, "peer_symbols": [],
            "universe": {**dataset["universe"], "peerStocks": []}, "feature_groups": groups,
            "feature_names": {key: names[key] for key in groups["RMB"]}, "provenance": provenance}


def _fit_predict(train: pd.DataFrame, current: pd.DataFrame, feature_groups: dict):
    counts = train.groupby("symbol").size()
    eligible = counts.index[counts >= MIN_TRAINING].tolist()
    observed = current.loc[current.symbol.isin(eligible)].copy()
    training = train.loc[train.symbol.isin(eligible)].copy()
    if observed.empty:
        raise ValueError("当前时点没有达到504条成熟训练记录的目标股票。")

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
            raise ValueError("固定专家没有覆盖全部共同预测行。")
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
        "distinctFinancialDisclosuresByStock": {symbol: int(group.resultsEventId.nunique()) for symbol, group in training.groupby("symbol")},
        "lastMaturedLabelEnd": _iso(training.label_end.max()),
        "currentRowsPerExpert": len(observed), "expertCount": len(EXPERTS),
        "trainingKeysSha256": hashlib.sha256(training[["date", "symbol", "label_end"]].to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest(),
    }
    return pd.concat(outputs, ignore_index=True), bundles, audit


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
            progress(f"固定四专家训练：{index + 1}/{len(refits)} 个历史时点")
    if not records:
        raise ValueError("本轮没有生成原始样本外预测。")
    history = pd.concat(records, ignore_index=True).sort_values(["expert", "date", "symbol"]).reset_index(drop=True)
    signal = calendar[-1]
    train = labels.loc[(labels.date >= signal - pd.DateOffset(years=TRAINING_YEARS)) & (labels.label_end < signal)]
    current = factors.loc[factors.date == signal]
    if set(current.symbol) != set(dataset["target_symbols"]):
        raise ValueError("截止日未覆盖全部四股的完整58项因子。")
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
        raise ValueError("本实验必须完整包含八个候选的唯一预测行。")
    reference = predictions.loc[predictions.candidate == CANDIDATE_IDS[0], list(MATCH_COLUMNS)].sort_values(["date", "symbol"]).reset_index(drop=True)
    for candidate in CANDIDATE_IDS[1:]:
        observed = predictions.loc[predictions.candidate == candidate, list(MATCH_COLUMNS)].sort_values(["date", "symbol"]).reset_index(drop=True)
        if not reference.equals(observed):
            raise ValueError("人民币与控制候选的日期、标签或涨频基准不一致。")


def _specification(dataset: dict, source_hashes: dict, frozen_at: str) -> dict:
    return {
        "experimentId": "four_stock_rich_context_rmb_increment_v1", "frozenAt": frozen_at,
        "hypothesis": "人民币融资环境与跨币种一月定盘利差，可能在已包含行业ETF、公告、财务及港元拆息的控制模型之外带来增量预测信息。",
        "experts": [asdict(expert) for expert in EXPERTS], "candidateCount": len(CANDIDATE_IDS),
        "candidates": CANDIDATE_IDS, "groupCandidates": GROUP_CANDIDATES, "groupNames": GROUP_NAMES,
        "featureGroups": dataset["feature_groups"], "featureNames": dataset["feature_names"],
        "trainingSymbols": dataset["target_symbols"], "peerTrainingRows": 0,
        "financialDisclosuresByStock": {row["symbol"]: row["documents"] for row in dataset["provenance"]["fundamentalDocuments"]},
        "financialSampleCountRule": "独立原公告按份计数，重复沿用低频信息的日度行不是独立财报样本，跨股同日也不是四倍独立市场观测。",
        "controlGroup": "RICH含原event_drivers45+4项财务+4项HIBOR，共53项；在与RMB相同的完整四股训练行上重新训练。",
        "incrementGroup": "RMB在RICH之外只增加固定四项SHIBOR与HIBOR一月减SHIBOR一月的利差，共58项。",
        "comparison": "RICH与RMB各在自己的四候选内独立年度选型；ALL在预先固定八候选内选型，三系统使用完全相同预测日期、标签和交易起点。",
        "financialRule": dataset["provenance"]["fundamentalRule"],
        "hiborTiming": dataset["provenance"]["hibor"]["timingRule"],
        "shiborTiming": dataset["provenance"]["shibor"]["timingRule"],
        "hiborVintageCaveat": dataset["provenance"]["hibor"]["vintageCaveat"],
        "shiborVintageCaveat": dataset["provenance"]["shibor"]["vintageCaveat"],
        "crossCurrencySpread": dataset["provenance"]["crossCurrencySpreadRule"],
        "rateMetadata": "来源日期、年龄及跨币种来源日期差仅审计，不增加任何模型特征或条件阈值。",
        "trainingYears": TRAINING_YEARS, "minimumTrainingRowsPerStock": MIN_TRAINING,
        "horizonTradingDays": HORIZON, "predictionStride": PREDICTION_STRIDE, "refitStride": REFIT_STRIDE,
        "firstRefitNotBefore": "2014-01-01", "calendarAnchor": _iso(dataset["calendar"][0]),
        "modelWorkers": MODEL_WORKERS, "mathThreadsPerFit": 1,
        "logistic": {"C": 0.1, "maxIterations": 1000, "scope": "每股单独训练"},
        "tree": {"maxLeafNodes": 7, "maxDepth": 2, "iterations": 100, "learningRate": 0.03,
                 "minimumLeafRowsPerEligibleStock": 63, "l2Regularization": 1.0, "earlyStopping": False,
                 "seed": 42, "scope": "达到504条成熟样本的原目标股共同训练"},
        "trainingWeights": "本股每行1/21；共享树每个日期全部合格目标股合计权重1/21。",
        "constantFeatures": "线性模型只在各自训练窗口内剔除完全恒定的列，再按历史训练权重标准化。",
        "weightedBinning": "沿用固定scikit-learn 1.9.0已核验批量分箱，不改变原权重或树模型。",
        "calibration": {"years": CALIBRATION_YEARS, "minimumDistinctForecastDates": MIN_CALIBRATION_DATES,
                        "scope": "只汇集原四股该专家的历史样本外预测，各日期合计等权。",
                        "slopeMinimum": 0, "identityPenalty": 1.0,
                        "rawEligibility": "raw与monotone都从同一校准成熟记录门槛满足后开始进入候选。"},
        "baseProbability": "同一五年成熟训练行中按全局21日网格抽取本股标签，(上涨数+1)/(观测数+2)。",
        "outerStartYear": OUTER_START_YEAR, "innerYears": 3,
        "annualSelection": "每年年初只用此前三年且label_end严格早于一月一日的成熟预测；全年固定共同候选类型，滚动更新拟合参数。",
        "score": "股票年度Brier差等权均值+0.5倍总体标准差+0.5*max(0,0.52-各股平衡命中均值)+0.1*max(0,0.35-最差股票下跌召回)。",
        "minimumForecastsPerStockYear": 24, "minimumForecastsPerStockRecall": 24,
        "tieBreak": "按各组及总候选清单的预先固定顺序。",
        "latestRefit": "数据截止日另外重训最新模型并单独保存，不覆盖历史定期预测。",
        "excellenceRules": EXCELLENCE_RULES, "independentFutureObservationRequired": True,
        "trading": {"entryProbability": 0.55, "annualVolatilityTarget": 0.15, "maximumPosition": 1.0,
                    "signalEverySessions": 5, "execution": "沿用下一开盘引擎、同一历史费用和双倍滑点压力对照"},
        "uncertainty": {"bootstrapBlockForecasts": 6, "replications": 2000, "seed": 42,
                        "irregularHighConfidenceBootstrap": False},
        "sourceHashes": source_hashes, "dataHashes": dataset["provenance"]["fileHashes"],
        "researchNote": "全部历史已经在前序研究中查看；本轮只执行预先固定的人民币增量假设，不能称为真正独立验证，也不能在看到结果后改因子或超参数。",
        "liveReady": False,
    }


def _execute_locked(progress) -> dict:
    source_files = [ROOT / "quant" / name for name in DEPENDENCIES] + [ROOT / "requirements.txt"]
    source_hashes = {str(path.relative_to(ROOT)): _digest(path) for path in source_files}
    progress("校验四股完整行业、公告、财务、港元及人民币环境来源，构造共同样本")
    dataset = build_research_dataset(DATA)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    destination = RUNS / "rmb" / run_id
    destination.mkdir(parents=True)
    specification = _specification(dataset, source_hashes, now.isoformat())
    runtime = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "exchange-calendars", "joblib", "threadpoolctl",
    )}
    record = {"id": run_id, "createdAt": now.isoformat(), "sourceHashes": source_hashes,
              "dataset": dataset["provenance"], "runtime": runtime, "python": sys.version,
              "independentValidation": False, "liveReady": False,
              "evaluationNote": "所有历史已经查看；固定的人民币增量研究仍属探索性历史重演。"}
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
    progress(f"已冻结RICH53/RMB58两组、四专家、八候选及数据源指纹：{run_id}")
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
            selection["scope"] = "只按原四只股票的成熟预测选择共同年度配置；同行不进入训练、校准或选型。"
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
        progress(f"核算{stock['name']}三个年度系统的同日期预测和扣费交易")
        control = chosen["RICH"].loc[chosen["RICH"].symbol == symbol].sort_values("date").reset_index(drop=True)
        evaluated = {}
        for group, selected in chosen.items():
            rows = selected.loc[selected.symbol == symbol].sort_values("date").reset_index(drop=True)
            if not rows[list(MATCH_COLUMNS)].equals(control[list(MATCH_COLUMNS)]):
                raise ValueError("两组与总系统的股票日期、标签或涨频基准不相同。")
            result = _evaluate(rows, market, symbol, dataset["calendar"], dataset["factors"])
            result["latestForecast"] = _latest_forecast(symbol, selected_candidates[group], latest, calibrators,
                                                        dataset["calendar"], future_calendar, future)
            result["currentSelectionQualified"] = annual_selections[group][-1]["researchQualificationPassed"]
            evaluated[group] = result
        baseline = evaluated["RICH"]
        comparisons = {
            group: {
                "sameForecastDates": True, "controlGroup": "RICH",
                "accuracyChange": result["evaluation"]["accuracy"] - baseline["evaluation"]["accuracy"],
                "brierImprovement": baseline["evaluation"]["brier"] - result["evaluation"]["brier"],
                "recentAccuracyChange": result["recentEvaluation"]["accuracy"] - baseline["recentEvaluation"]["accuracy"],
                "recentBrierImprovement": baseline["recentEvaluation"]["brier"] - result["recentEvaluation"]["brier"],
                "netCagrChange": result["trading"]["strategy"]["cagr"] - baseline["trading"]["strategy"]["cagr"],
            } for group, result in evaluated.items() if group != "RICH"
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
        "summary": {"stocks": 4, "learningStocks": 4, "factorCount": 58, "candidateCount": 8,
                    "groupCount": 2, "dataEnd": _iso(end),
                    "financialDisclosureCount": sum(row["documents"] for row in dataset["provenance"]["fundamentalDocuments"]),
                    "financialDisclosuresByStock": specification["financialDisclosuresByStock"],
                    "commonDailyFactorRows": len(dataset["factors"]), "commonDailyLabelRows": len(dataset["labels"]),
                    "historicalExcellentCount": sum(row["historicalExcellent"] for row in reports),
                    "groupHistoricalExcellentCounts": {group: sum(row["groupReports"][group]["historicalExcellent"] for row in reports) for group in GROUPS},
                    "independentValidation": False, "liveReady": False},
        "limitations": [
            "所有历史已经被查看；本轮固定候选与逐年时间隔离不能消除反复研究和选择偏差。",
            "RICH控制组在共同58项可计算四股样本上重新训练，不能与旧十二股或旧18项控制报告直接比较为同一个系统。",
            "人民币与港元各使用严格早于香港信号日的最近真实定盘，两者日期可以不同；利差是已知融资环境差，不是同一时刻可交易套利价差。",
            "人民币21/63阶变化在官方银行间原生定盘序列计算，保留官方补班周末；港股开市日不等于两地定盘日。",
            "两地利率均为当前取得的历史版本，没有逐日历史修订日志；响应时间不等于当时实际公布时刻。",
            "来源日期及年龄只供审计，不构成额外输入；未知缺失或未核验财务不会被填补。",
            "118份财报是低频信息源，沿用同一公告数值的日度行和同日跨股样本不是新增独立观测。",
            "21日收益标签有重叠，沿用六次预测区块；高置信筛选记录不等距时不计算该区块区间。",
            "原15项历史门槛、21日预测跨度、55%持仓阈值、历史费用与真正未来独立验证要求均保持不变。",
            "截止日最新模型单独保存；发行时目标窗口已经开始的预测只属于重建研究，不能作为真实提前预测。",
        ],
    }
    _write(destination / "report.json", report)
    pending = destination.parent / f"latest.{run_id}.pending.json"
    _write(pending, {"id": run_id})
    pending.replace(destination.parent / "latest.json")
    progress(f"人民币融资环境增量研究完成：{destination / 'report.json'}")
    return report


def execute_rmb_research(progress=print) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".research.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("另一个量化研究仍在运行，不能同时启动人民币结构实验。") from exc
        try:
            with threadpool_limits(limits=1):
                return _execute_locked(progress)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    result = execute_rmb_research()
    for item in result["stockReports"]:
        print(f"{item['stock']['name']}：总系统命中率 {item['evaluation']['accuracy']:.1%}；"
              f"通过历史门槛 {sum(check['passed'] for check in item['checks'])}/{len(item['checks'])}；"
              "真正未来独立验证尚未完成")
