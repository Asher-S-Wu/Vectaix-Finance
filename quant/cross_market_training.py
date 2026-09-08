from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .cross_market_factors import FEATURES, FEATURE_KEYS, build_cross_market_factors
from .cross_market_evaluation import evaluate
from .general_training import TREE_PARAMETERS, apply_calibrator, _logit
from .lab_binning import install_batch_binning


ROOT = Path(__file__).resolve().parent.parent
STREAMS = ("JOINT", "A_ONLY", "HK_ONLY")
IDENTITY = ["date", "market", "symbol", "issuer_id", "split", "signal_at", "label_start", "label_end",
            "label_status", "target", "forward_return", "current_list_status", "lifecycle_status"]


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(frame):
    if frame.empty or frame.duplicated(["date", "market", "issuer_id"]).any():
        raise ValueError("权重要求每个市场日期每家公司至多一份观察且非空。")
    counts = frame.groupby(["date", "market"])["symbol"].transform("size").to_numpy(float)
    markets = frame.groupby("date")["market"].transform("nunique").to_numpy(float)
    w = 1/(counts*markets)
    return w*len(w)/w.sum()


def calibrate(history, fit_date):
    selected = history.loc[history.split.eq("train") & history.date.ge(fit_date-pd.DateOffset(years=3))
                           & history.date.lt(fit_date) & history.label_end.lt(fit_date)
                           & history.label_status.eq("scorable")]
    info = {"rows": len(selected), "dates": int(selected.date.nunique()),
            "symbols": int(selected.symbol.nunique()), "fitDate": str(fit_date.date()),
            "maxLabelEnd": None if selected.empty else str(selected.label_end.max().date()),
            "holdoutRowsUsed": 0, "marketCounts": {str(k): int(v) for k, v in selected.market.value_counts().items()}}
    if selected.date.nunique() < 126 or selected.target.nunique() != 2:
        return {**info, "status": "unavailable", "reason": "requires_126_mature_dates_and_two_classes"}
    x, y = _logit(selected.raw_probability.to_numpy(float)), selected.target.to_numpy(float)
    w = weights(selected); w /= w.sum()
    def objective(parameters):
        slope, intercept = parameters
        z = slope*x+intercept
        residual = w*(expit(z)-y)
        return (float(np.dot(w, np.logaddexp(0, z)-y*z)+.01*((slope-1)**2+intercept**2)),
                np.array([np.dot(residual, x)+.02*(slope-1), residual.sum()+.02*intercept]))
    fit = minimize(objective, np.array([1., 0.]), method="L-BFGS-B", jac=True,
                   bounds=[(0., None), (None, None)], options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 3000})
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"跨市场校准未收敛：{fit.message}")
    return {**info, "status": "available", "slope": float(fit.x[0]), "intercept": float(fit.x[1]),
            "objective": float(fit.fun), "optimizerMessage": str(fit.message)}


def specification():
    return {"schemaVersion": 1, "experiment": "a-hk-shared-direction-v1", "features": FEATURES,
            "models": {stream: TREE_PARAMETERS for stream in STREAMS},
            "label": "Supplier multiplicatively adjusted next-local-session open to 22nd-local-session open direction; not total investment return",
            "predictionStride": 5, "refitStride": 63, "calendar": "Union of independently validated A and HK sessions; local rolling windows",
            "trainingYears": 5, "firstFitYear": 2014, "calibrationYears": 3, "calibrationMinimumDates": 126,
            "signalTime": "19:10 +08 historical information convention, no historical publication proof",
            "weights": "Equal date mass; within date equal observed market mass; within market equal issuer mass; mean row weight one",
            "splits": "Fixed identity-group hash modulo 5; known A/H links grouped; unresolved historical identity coverage explicitly blocks excellence",
            "calibration": "Own-stream raw out-of-time train-group forecasts only, past 3 years, labels matured strictly before fit; monotone Platt with .01 penalty",
            "transfer": "A_ONLY and HK_ONLY additionally predict the other market without fitting its labels; local calibration remains source-market-only",
            "baseline": "Market-specific date-equal up frequency from the same 5-year matured training-stock window; never holdout labels",
            "selection": "Three fixed streams, no parameter search or choosing a favourable period",
            "bootstrap": {"blockUnionPredictionDates": 12, "draws": 2000, "seed": 42},
            "candidateGatesPerMarket": {"brierSkillVersusFrequency": .05, "brierSkillVersusLocalHGB": .02,
                                       "bothPairedBrier95UpperBelow": 0, "balancedAccuracy95LowerAbove": .5,
                                       "eceAtMost": .05, "eligibleIssuanceAtLeast": .8,
                                       "maturedScorableRateAtLeast": .9, "matureDatesAtLeast": 126, "scorableGroupsAtLeast": 80},
            "gateMeaning": "Fixed research targets, not universal industry standards; cannot override unresolved data or missing prospective evidence",
            "historicalExplorationOnly": True, "oldHistoryPreviouslyExamined": True,
            "executionModelIncluded": False, "liveReady": False, "futureEvidenceAvailable": False}


def train(dataset, destination):
    frame = dataset["coverage"].merge(dataset["factors"], on=["date", "market", "symbol", "issuer_id"], validate="one_to_one")
    frame = frame.sort_values(["date", "market", "symbol"], ignore_index=True)
    calendar = dataset["calendar"]
    fit_dates = calendar[::63]; fit_dates = fit_dates[fit_dates >= "2014-01-01"]
    histories = {s: [] for s in STREAMS}; states = {}; audits = []; parts = []
    for index, fit_date in enumerate(fit_dates):
        stop = fit_dates[index+1] if index+1 < len(fit_dates) else calendar[-1]+pd.Timedelta(days=1)
        mature = frame.loc[frame.split.eq("train") & frame.date.ge(fit_date-pd.DateOffset(years=5))
                           & frame.date.lt(fit_date) & frame.label_end.lt(fit_date) & frame.label_status.eq("scorable")]
        current = frame.loc[frame.date.ge(fit_date) & frame.date.lt(stop)]
        bases = {}
        for market in ("A", "HK"):
            own = mature.loc[mature.market.eq(market)]
            if own.empty or own.target.nunique() != 2:
                raise ValueError(f"{fit_date} {market} 没有可训练的成熟两类数据。")
            bases[market] = float(np.average(own.target, weights=weights(own)))
        for stream in STREAMS:
            fit_frame = mature if stream == "JOINT" else mature.loc[mature.market.eq(stream.removesuffix("_ONLY"))]
            history = pd.concat(histories[stream], ignore_index=True) if histories[stream] else pd.DataFrame(columns=[*IDENTITY, "raw_probability"])
            calibration = calibrate(history, fit_date)
            w = weights(fit_frame)
            model = HistGradientBoostingClassifier(**TREE_PARAMETERS)
            model.fit(fit_frame[list(FEATURE_KEYS)].to_numpy(float), fit_frame.target.to_numpy(int), sample_weight=w)
            if not np.array_equal(model.classes_, [0, 1]):
                raise ValueError("模型缺失真实上涨/下跌类别。")
            output = current[IDENTITY].copy()
            output["model"] = stream; output["fit_id"] = fit_date.strftime("%Y%m%d"); output["model_fit_date"] = fit_date
            output["raw_probability"] = model.predict_proba(current[list(FEATURE_KEYS)].to_numpy(float))[:, 1]
            output["base_probability"] = output.market.map(bases)
            output["probability"] = np.nan; output["prediction_status"] = "calibration_unavailable"
            if calibration["status"] == "available":
                output["probability"] = apply_calibrator(calibration, output.raw_probability.to_numpy(float))
                output["prediction_status"] = "historical_reconstruction"
            # Local-stream calibration never learns from transfer-target labels.
            own_history = output if stream == "JOINT" else output.loc[output.market.eq(stream.removesuffix("_ONLY"))]
            histories[stream].append(own_history)
            audit = {"fitDate": str(fit_date.date()), "model": stream, "trainingRows": len(fit_frame),
                     "trainingDates": int(fit_frame.date.nunique()), "trainingGroups": int(fit_frame.issuer_id.nunique()),
                     "maxLabelEnd": str(fit_frame.label_end.max().date()), "holdoutRowsUsed": 0,
                     "marketRows": {str(k): int(v) for k, v in fit_frame.market.value_counts().items()},
                     "marketWeight": {str(k): float(w[fit_frame.market.eq(k).to_numpy()].sum()) for k in fit_frame.market.unique()},
                     "trainingGroupSha256": hashlib.sha256("\n".join(sorted(fit_frame.issuer_id.unique())).encode()).hexdigest(),
                     "baseProbabilities": bases, "calibration": calibration}
            states[f"{fit_date:%Y%m%d}:{stream}"] = {"model": model, "calibrator": calibration, "features": list(FEATURE_KEYS),
                                                    "fit_date": fit_date, "base_probabilities": bases, "audit": audit}
            audits.append(audit); parts.append(output)
        progress = {"completedFits": index+1, "totalFits": len(fit_dates), "lastFitDate": str(fit_date.date())}
        write(destination/"progress.json", progress); print(json.dumps(progress), flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated(["date", "market", "symbol", "model"]).any():
        raise ValueError("跨市场同模型发行重复。")
    latest_input = dataset["latest_coverage"].merge(dataset["factors"], on=["date", "market", "symbol", "issuer_id"], validate="one_to_one")
    latest_parts = []
    for stream in STREAMS:
        state = states[f"{fit_dates[-1]:%Y%m%d}:{stream}"]
        latest = latest_input[IDENTITY].copy(); latest["model"] = stream; latest["model_fit_date"] = fit_dates[-1]
        latest["raw_probability"] = state["model"].predict_proba(latest_input[list(FEATURE_KEYS)].to_numpy(float))[:, 1]
        latest["probability"] = np.nan
        if state["calibrator"]["status"] == "available":
            latest["probability"] = apply_calibrator(state["calibrator"], latest.raw_probability.to_numpy(float))
        latest["base_probability"] = latest.market.map(state["base_probabilities"])
        latest_parts.append(latest)
    return predictions, pd.concat(latest_parts, ignore_index=True), states, audits


def gates(reports, provenance):
    result = {}
    for market in ("A", "HK"):
        report = reports[market]["exploratory_heldout_2024_plus"]
        model = report["models"]["JOINT"]; m = model["metrics"]; c = model["coverage"]; p = report["jointVersusLocal"]
        ci = m.get("bootstrap95", {}); pci = p.get("bootstrap95", {})
        def above(value, threshold): return value is not None and value >= threshold
        def lower_above(interval, threshold): return interval is not None and interval[0] > threshold
        def upper_below(interval, threshold): return interval is not None and interval[1] < threshold
        checks = {"completeLocalComparisonCoverage": p.get("status") == "available" and p["jointOutsidePair"] == 0 and p["localOutsidePair"] == 0,
                  "brierSkillFrequency": above(m.get("brierSkill"), .05),
                  "brierSkillLocal": above(p.get("brierSkillVersusLocal"), .02),
                  "frequencyBrierInterval": upper_below(ci.get("brierMinusFrequency"), 0),
                  "localBrierInterval": upper_below(pci.get("brierMinusLocal"), 0),
                  "balancedAccuracyInterval": lower_above(ci.get("balancedAccuracy"), .5),
                  "calibration": m.get("ece") is not None and m["ece"] <= .05,
                  "issuanceCoverage": above(c["eligibleIssuanceRate"], .8),
                  "outcomeCoverage": above(c["maturedScorableRate"], .9),
                  "matureDates": above(m.get("dates"), 126), "scorableGroups": above(m.get("stocks"), 80)}
        result[market] = {"checks": checks, "passed": all(checks.values())}
    return {"perMarket": result, "historicalNumericGatesPassed": all(x["passed"] for x in result.values()),
            "excellent": False, "liveReady": False, "prospectiveEvidenceAvailable": False,
            "limitations": provenance["limitations"]}


def run():
    from .cross_market_data import load_cross_market_dataset
    install_batch_binning()
    destination = ROOT/"runs/cross-market"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    write(destination/"specification.json", specification())
    dataset = build_cross_market_factors(load_cross_market_dataset())
    joblib.dump(dataset, destination/"dataset.joblib", compress=3)
    write(destination/"data-quality.json", dataset["provenance"])
    source_dir = destination/"source"; source_dir.mkdir()
    sources = ["__init__.py", "cross_market_training.py", "cross_market_evaluation.py", "cross_market_factors.py", "cross_market_data.py",
               "cross_market_identity.py", "general_data.py", "general_training.py", "general_factors.py", "lab_binning.py", "data.py"]
    source_hashes = {}
    for name in sources:
        shutil.copy2(ROOT/"quant"/name, source_dir/name); source_hashes[name] = sha(source_dir/name)
    with threadpool_limits(limits=2):
        predictions, latest, states, audits = train(dataset, destination)
    predictions.to_csv(destination/"predictions.csv.gz", index=False)
    latest.to_csv(destination/"latest-predictions.csv", index=False)
    dataset["coverage"].to_csv(destination/"coverage.csv.gz", index=False)
    joblib.dump(states, destination/"models.joblib", compress=3)
    write(destination/"fit-audit.json", audits)
    reports = evaluate(dataset, predictions)
    for filename, expected in dataset["provenance"]["inputFileHashes"].items():
        if sha(filename) != expected:
            raise ValueError(f"训练期间原始来源发生改变：{filename}")
    for filename, expected in source_hashes.items():
        if sha(ROOT/"quant"/filename) != expected or sha(source_dir/filename) != expected:
            raise ValueError(f"训练期间模型源码发生改变：{filename}")
    report = {"schemaVersion": 1, "runId": destination.name, "experiment": specification(),
              "data": dataset["provenance"], "modelVersions": len(states), "latestStocks": int(latest.symbol.nunique()),
              "results": reports, "qualification": gates(reports, dataset["provenance"]),
              "sourceHashes": source_hashes, "sourceAndInputHashesUnchangedAtCompletion": True,
              "packageVersions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "exchange-calendars", "pypdf")}}
    write(destination/"report.json", report)
    write(destination/"artifacts.json", {p.name: sha(p) for p in destination.iterdir() if p.is_file()})
    print(str(destination), flush=True)
    return destination


if __name__ == "__main__":
    argparse.ArgumentParser(description="固定 A 股和港股共同模型研究").parse_args()
    run()
