from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .general_data import load_general_dataset
from .general_factors import FEATURE_KEYS, FEATURES, HORIZON, PREDICTION_STRIDE, build_general_factors
from .lab_binning import install_batch_binning


ROOT = Path(__file__).resolve().parent.parent
REFIT_STRIDE = 63
TRAINING_YEARS = 5
CALIBRATION_YEARS = 3
MIN_CALIBRATION_DATES = 126
BOOTSTRAP_BLOCK = 6
BOOTSTRAP_REPEATS = 2000
MODELS = ("hist_gradient_boosting", "logistic_baseline")
TREE_PARAMETERS = {
    "learning_rate": 0.05, "max_iter": 160, "max_leaf_nodes": 15,
    "max_depth": 4, "min_samples_leaf": 200, "l2_regularization": 10,
    "max_bins": 63, "early_stopping": False, "random_state": 42,
}
LOGISTIC_PARAMETERS = {"C": 0.1, "max_iter": 3000, "solver": "lbfgs", "class_weight": None}
DEPENDENCIES = ("__init__.py", "general_training.py", "general_factors.py", "general_data.py", "lab_binning.py")
IDENTITY_COLUMNS = ["date", "symbol", "split", "signal_at", "label_start", "label_end", "label_status", "target", "forward_return", "current_list_status", "lifecycle_status"]


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso(value) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def date_weights(frame: pd.DataFrame, *, normalize: bool = True) -> np.ndarray:
    if frame.empty:
        raise ValueError("日期权重不能作用于空样本。")
    weight = 1.0 / frame.groupby("date")["date"].transform("size").to_numpy(dtype=float)
    return weight * (len(weight) / weight.sum()) if normalize else weight


def _logit(probability: np.ndarray) -> np.ndarray:
    # Floating-point endpoint protection is numerical evaluation, not missing-data substitution.
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


def apply_calibrator(calibrator: dict, probability: np.ndarray) -> np.ndarray:
    return expit(calibrator["slope"] * _logit(probability) + calibrator["intercept"])


def fit_calibrator(history: pd.DataFrame, fit_date: pd.Timestamp, training_symbols: set[str]) -> dict:
    history = history.loc[
        history.symbol.isin(training_symbols) & history.date.ge(fit_date - pd.DateOffset(years=CALIBRATION_YEARS))
        & history.date.lt(fit_date) & history.label_end.lt(fit_date)
        & history.label_status.eq("scorable") & history.raw_probability.notna()
    ].sort_values(["date", "symbol"])
    info = {
        "fitDate": _iso(fit_date), "rows": len(history), "dates": int(history.date.nunique()),
        "symbols": int(history.symbol.nunique()), "start": _iso(history.date.min()),
        "end": _iso(history.date.max()), "maxLabelEnd": _iso(history.label_end.max()),
        "trainingSymbolsOnly": True, "penalty": 0.01,
    }
    if info["dates"] < MIN_CALIBRATION_DATES or history.target.nunique() != 2:
        return {**info, "status": "unavailable", "reason": "requires_126_mature_prediction_dates_and_both_classes"}
    x = _logit(history.raw_probability.to_numpy(dtype=float))
    y = history.target.to_numpy(dtype=float)
    w = date_weights(history)
    w /= w.sum()
    def objective(parameters):
        slope, intercept = parameters
        z = slope * x + intercept
        loss = np.dot(w, np.logaddexp(0, z) - y * z) + 0.01 * ((slope - 1) ** 2 + intercept ** 2)
        residual = w * (expit(z) - y)
        gradient = np.array([np.dot(residual, x) + 0.02 * (slope - 1), residual.sum() + 0.02 * intercept])
        return float(loss), gradient
    result = minimize(objective, np.array([1.0, 0.0]), method="L-BFGS-B", jac=True,
                      bounds=[(0.0, None), (None, None)], options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 3000})
    if not result.success or not np.isfinite(result.x).all() or result.x[0] < 0:
        raise ValueError(f"{fit_date.date()} 单调校准优化未收敛：{result.message}")
    return {**info, "status": "available", "slope": float(result.x[0]), "intercept": float(result.x[1]),
            "objective": float(result.fun), "optimizerMessage": str(result.message)}


def _fit_model(model_id: str, frame: pd.DataFrame):
    x = frame[list(FEATURE_KEYS)].to_numpy(dtype=float)
    y = frame.target.to_numpy(dtype=int)
    weight = date_weights(frame)
    if model_id == MODELS[0]:
        model = HistGradientBoostingClassifier(**TREE_PARAMETERS)
        model.fit(x, y, sample_weight=weight)
    elif model_id == MODELS[1]:
        model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(**LOGISTIC_PARAMETERS))])
        model.fit(x, y, scale__sample_weight=weight, model__sample_weight=weight)
    else:
        raise ValueError(f"未预先指定的模型：{model_id}")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("拟合后的类别必须为真实的下跌/上涨两类。")
    return model


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(frame[list(FEATURE_KEYS)].to_numpy(dtype=float))[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("模型生成了无效概率。")
    return probability


def train_walk_forward(dataset: dict, destination: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    calendar = dataset["calendar"]
    grid = dataset["prediction_dates"]
    training_symbols = set(dataset["training_symbols"])
    train_frame = dataset["coverage"].merge(dataset["factors"], on=["date", "symbol"], how="inner", validate="one_to_one")
    train_frame = train_frame.sort_values(["date", "symbol"], ignore_index=True)
    fit_dates = calendar[::REFIT_STRIDE]
    fit_dates = fit_dates[fit_dates >= pd.Timestamp("2014-01-01")]
    if fit_dates.empty:
        raise ValueError("没有从 2014 年开始的固定重训日期。")
    raw_parts, audits, persisted = [], [], {}
    empty_history = pd.DataFrame(columns=[*IDENTITY_COLUMNS, "model", "raw_probability"])
    for fit_index, fit_date in enumerate(fit_dates):
        next_fit = fit_dates[fit_index + 1] if fit_index + 1 < len(fit_dates) else calendar[-1] + pd.Timedelta(days=1)
        fit_id = fit_date.strftime("%Y%m%d")
        mature = train_frame.loc[
            train_frame.symbol.isin(training_symbols) & train_frame.date.ge(fit_date - pd.DateOffset(years=TRAINING_YEARS))
            & train_frame.date.lt(fit_date) & train_frame.label_end.lt(fit_date)
            & train_frame.label_status.eq("scorable")
        ]
        current = train_frame.loc[train_frame.date.ge(fit_date) & train_frame.date.lt(next_fit)]
        if mature.empty or mature.target.nunique() != 2:
            raise ValueError(f"{fit_id} 固定训练窗口没有真实两类成熟样本，不能补造模型。")
        if not mature.date.isin(grid).all() or not (mature.label_end < fit_date).all():
            raise ValueError("训练标签越过时间边界或使用非五日网格。")
        base_probability = float(np.average(mature.target.to_numpy(), weights=date_weights(mature)))
        shared_audit = {
            "fitId": fit_id, "fitDate": _iso(fit_date), "nextFitDate": _iso(next_fit),
            "trainingRows": len(mature), "trainingDates": int(mature.date.nunique()),
            "trainingSymbols": int(mature.symbol.nunique()), "minTrainingDate": _iso(mature.date.min()),
            "maxTrainingDate": _iso(mature.date.max()), "maxLabelEnd": _iso(mature.label_end.max()),
            "trainingSymbolsSha256": hashlib.sha256("\n".join(sorted(mature.symbol.unique())).encode()).hexdigest(),
            "holdoutRowsUsed": 0, "baseProbability": base_probability,
            "weightDefinition": "Equal total mass per prediction date, normalized to mean row weight 1",
        }
        for model_id in MODELS:
            history = pd.concat(raw_parts, ignore_index=True) if raw_parts else empty_history
            history = history.loc[history.model.eq(model_id)]
            calibrator = fit_calibrator(history, fit_date, training_symbols)
            fitted = _fit_model(model_id, mature)
            output = current[IDENTITY_COLUMNS].copy()
            output["model"], output["fit_id"] = model_id, fit_id
            output["model_fit_date"] = fit_date
            output["raw_probability"] = _predict(fitted, current) if len(current) else np.array([], dtype=float)
            output["base_probability"] = base_probability
            output["probability"] = np.nan
            output["prediction_status"] = "calibration_unavailable"
            if calibrator["status"] == "available":
                output["probability"] = apply_calibrator(calibrator, output.raw_probability.to_numpy())
                output["prediction_status"] = "issued_historical_reconstruction"
            raw_parts.append(output)
            state = {"model": fitted, "calibrator": calibrator, "base_probability": base_probability,
                     "features": list(FEATURE_KEYS), "fit_date": fit_date, "audit": shared_audit}
            persisted[f"{fit_id}:{model_id}"] = state
            audits.append({**shared_audit, "model": model_id, "calibration": calibrator,
                           "rawPredictions": len(output), "calibratedPredictions": int(output.probability.notna().sum())})
        _write(destination / "progress.json", {"completedFits": fit_index + 1, "totalFits": len(fit_dates),
                                                "lastFitDate": _iso(fit_date), "rawRows": sum(len(part) for part in raw_parts)})
        print(json.dumps({"completedFits": fit_index + 1, "totalFits": len(fit_dates), "date": _iso(fit_date)}, ensure_ascii=False), flush=True)
    raw = pd.concat(raw_parts, ignore_index=True).sort_values(["date", "symbol", "model"], ignore_index=True)
    if raw.duplicated(["date", "symbol", "model"]).any():
        raise ValueError("同一模型同一证券日期重复发行。")
    latest_input = dataset["latest_coverage"].merge(dataset["factors"], on=["date", "symbol"], how="inner", validate="one_to_one")
    latest_parts = []
    last_id = fit_dates[-1].strftime("%Y%m%d")
    for model_id in MODELS:
        state = persisted[f"{last_id}:{model_id}"]
        latest = latest_input[IDENTITY_COLUMNS].copy()
        latest["model"], latest["fit_id"] = model_id, last_id
        latest["model_fit_date"] = state["fit_date"]
        latest["raw_probability"] = _predict(state["model"], latest_input)
        latest["base_probability"] = state["base_probability"]
        latest["probability"] = np.nan
        latest["prediction_status"] = "calibration_unavailable"
        if state["calibrator"]["status"] == "available":
            latest["probability"] = apply_calibrator(state["calibrator"], latest.raw_probability.to_numpy())
            latest["prediction_status"] = "latest_historical_reconstruction"
        latest_parts.append(latest)
    return raw, pd.concat(latest_parts, ignore_index=True), persisted, audits


def _nonoverlapping(frame: pd.DataFrame) -> int:
    end, count = None, 0
    for row in frame[["label_start", "label_end"]].drop_duplicates().sort_values("label_end").itertuples(index=False):
        if end is None or row.label_start > end:
            count += 1
            end = row.label_end
    return count


def _cluster_draws(date_count: int) -> np.ndarray | None:
    if date_count < BOOTSTRAP_BLOCK:
        return None
    rng = np.random.default_rng(42)
    starts = rng.integers(0, date_count - BOOTSTRAP_BLOCK + 1,
                          size=(BOOTSTRAP_REPEATS, int(np.ceil(date_count / BOOTSTRAP_BLOCK))))
    return (starts[:, :, None] + np.arange(BOOTSTRAP_BLOCK)).reshape(BOOTSTRAP_REPEATS, -1)[:, :date_count]


def _interval(values: np.ndarray) -> list[float] | None:
    if not np.isfinite(values).all():
        return None
    return np.quantile(values, [0.025, 0.975]).astype(float).tolist()


def _ece(y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    assignment = np.minimum((probability * 5).astype(int), 4)
    return float(sum(abs(np.dot(weight[assignment == bucket], probability[assignment == bucket] - y[assignment == bucket])) for bucket in range(5)))


def probability_metrics(frame: pd.DataFrame, planned_dates: pd.DatetimeIndex, *, intervals: bool = True) -> dict:
    """Date-equal probability metrics; blocks retain the original calendar grid."""
    if frame.empty:
        return {"status": "unavailable", "reason": "no_scorable_calibrated_predictions", "rows": 0, "dates": 0, "stocks": 0}
    required = ["target", "probability", "base_probability"]
    if (frame.duplicated(["date", "symbol"]).any() or not frame.date.isin(planned_dates).all()
            or not np.isfinite(frame[required].to_numpy(dtype=float)).all()
            or not frame.target.isin([0, 1]).all()
            or not frame[["probability", "base_probability"]].ge(0).all().all()
            or not frame[["probability", "base_probability"]].le(1).all().all()):
        raise ValueError("概率评估输入须为有限、唯一且位于预定网格的真实两值标签/概率。")
    y, p, base = [frame[column].to_numpy(dtype=float) for column in required]
    predicted = p >= 0.5
    w = date_weights(frame, normalize=False)
    w /= w.sum()
    accuracy = float(np.dot(w, predicted == y))
    brier = float(np.dot(w, (p - y) ** 2))
    base_brier = float(np.dot(w, (base - y) ** 2))
    tp, tn = float(np.dot(w, (predicted & (y == 1)))), float(np.dot(w, (~predicted & (y == 0))))
    pos, neg = float(np.dot(w, y)), float(np.dot(w, 1 - y))
    both = pos > 0 and neg > 0
    balanced = (tp / pos + tn / neg) / 2 if both else None
    bins, ece = [], 0.0
    assignment = np.minimum((p * 5).astype(int), 4)
    for bucket in range(5):
        mask = assignment == bucket
        mass = float(w[mask].sum())
        mean_probability = float(np.dot(w[mask], p[mask]) / mass) if mass > 0 else None
        frequency = float(np.dot(w[mask], y[mask]) / mass) if mass > 0 else None
        if mass > 0:
            ece += mass * abs(mean_probability - frequency)
        bins.append({"lower": bucket / 5, "upper": (bucket + 1) / 5, "rows": int(mask.sum()),
                     "dateWeightedMass": mass, "meanProbability": mean_probability, "upFrequency": frequency})
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    base_clipped = np.clip(base, 1e-12, 1 - 1e-12)
    log_loss_values = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    base_log_loss_values = -(y * np.log(base_clipped) + (1 - y) * np.log1p(-base_clipped))
    base_ece = _ece(y, base, w)
    result = {
        "status": "available", "rows": len(frame), "dates": int(frame.date.nunique()),
        "stocks": int(frame.symbol.nunique()), "start": _iso(frame.date.min()), "end": _iso(frame.date.max()),
        "labelEnd": _iso(frame.label_end.max()), "plannedDates": len(planned_dates),
        "accuracy": accuracy, "alwaysUpAccuracy": pos,
        "frequencyAccuracy": float(np.dot(w, (base >= 0.5) == y)),
        "brier": brier, "baseBrier": base_brier, "brierSkill": 1 - brier / base_brier if base_brier > 0 else None,
        "balancedAccuracy": balanced, "upRecall": tp / pos if both else None, "downRecall": tn / neg if both else None,
        "auc": float(roc_auc_score(y, p, sample_weight=w)) if both else None,
        "classMetricsStatus": "available" if both else "unavailable_single_class",
        "logLoss": float(np.dot(w, log_loss_values)), "baseLogLoss": float(np.dot(w, base_log_loss_values)),
        "logLossMinusFrequency": float(np.dot(w, log_loss_values - base_log_loss_values)),
        "ece": float(ece), "baseEce": base_ece, "eceMinusFrequency": float(ece - base_ece), "calibrationBins": bins,
        "greedyNonOverlappingDateWindows": _nonoverlapping(frame),
        "sampleInterpretation": "Rows and dates overlap in outcome windows and share market factors; neither is an independent sample count.",
    }
    if not intervals:
        return result
    result["bootstrap95"] = {"status": "unavailable", "reason": "requires_6_scorable_dates", "blockDates": BOOTSTRAP_BLOCK}
    draws = _cluster_draws(len(planned_dates))
    if frame.date.nunique() < BOOTSTRAP_BLOCK or draws is None:
        return result
    daily = pd.DataFrame({
        "date": frame.date.to_numpy(), "accuracy": (predicted == y).astype(float),
        "accuracyMinusAlwaysUp": (predicted == y).astype(float) - y,
        "accuracyMinusFrequency": (predicted == y).astype(float) - ((base >= 0.5) == y),
        "brier": (p - y) ** 2, "brierMinusFrequency": (p - y) ** 2 - (base - y) ** 2,
        "tp": (predicted & (y == 1)).astype(float), "tn": (~predicted & (y == 0)).astype(float),
        "pos": y, "neg": 1 - y,
        "logLossMinusFrequency": log_loss_values - base_log_loss_values,
    }).groupby("date").mean().reindex(planned_dates)
    mass = daily.accuracy.notna().to_numpy(dtype=float)
    # Zero here denotes no contribution to a bootstrap sum, never an imputed observation.
    arrays = np.where(np.isfinite(daily.to_numpy()), daily.to_numpy(), 0.0)
    sums = arrays[draws].sum(axis=1)
    denominator = mass[draws].sum(axis=1)
    values = sums / np.where(denominator > 0, denominator, np.nan)[:, None]
    sampled_pos, sampled_neg = sums[:, 7], sums[:, 8]
    balanced_draw = (sums[:, 5] / np.where(sampled_pos > 0, sampled_pos, np.nan)
                     + sums[:, 6] / np.where(sampled_neg > 0, sampled_neg, np.nan)) / 2
    result["bootstrap95"] = {
        "status": "available", "method": "Noncircular moving blocks on original global five-session prediction grid, retaining whole date clusters",
        "blockDates": BOOTSTRAP_BLOCK, "draws": BOOTSTRAP_REPEATS, "seed": 42,
        "accuracy": _interval(values[:, 0]), "accuracyMinusAlwaysUp": _interval(values[:, 1]),
        "accuracyMinusFrequency": _interval(values[:, 2]), "brier": _interval(values[:, 3]),
        "brierMinusFrequency": _interval(values[:, 4]), "balancedAccuracy": _interval(balanced_draw) if both else None,
        "logLossMinusFrequency": _interval(values[:, 9]),
        "invalidDraws": int((~np.isfinite(values).all(axis=1) | ~np.isfinite(balanced_draw)).sum()),
        "nullIntervalMeaning": "That specific interval is unavailable; overall status does not make a null interval valid",
    }
    return result


def paired_metrics(primary: pd.DataFrame, baseline: pd.DataFrame, planned_dates: pd.DatetimeIndex) -> dict:
    left = primary[["date", "symbol", "target", "probability"]]
    right = baseline[["date", "symbol", "target", "probability"]]
    paired = left.merge(right, on=["date", "symbol"], suffixes=("_primary", "_baseline"), validate="one_to_one")
    if paired.empty:
        return {"status": "unavailable", "reason": "no_common_scorable_predictions"}
    if not paired.target_primary.equals(paired.target_baseline):
        raise ValueError("配对模型标签不一致。")
    y, p, b = [paired[column].to_numpy(dtype=float) for column in ("target_primary", "probability_primary", "probability_baseline")]
    pc, bc = np.clip(p, 1e-12, 1 - 1e-12), np.clip(b, 1e-12, 1 - 1e-12)
    values = pd.DataFrame({
        "date": paired.date.to_numpy(),
        "accuracyDifference": ((p >= 0.5) == y).astype(float) - ((b >= 0.5) == y),
        "brierDifference": (p - y) ** 2 - (b - y) ** 2,
        "logLossDifference": -(y * np.log(pc) + (1 - y) * np.log1p(-pc)) + (y * np.log(bc) + (1 - y) * np.log1p(-bc)),
    }).groupby("date").mean()
    result = {"status": "available", "rows": len(paired), "dates": len(values),
              "primaryRowsOutsidePair": len(primary) - len(paired), "baselineRowsOutsidePair": len(baseline) - len(paired),
              "differenceConvention": "Primary minus baseline; positive accuracy and negative Brier/log loss favour primary",
              **{column: float(value) for column, value in values.mean().items()}}
    weights = date_weights(paired, normalize=False)
    weights /= weights.sum()
    result["eceDifference"] = _ece(y, p, weights) - _ece(y, b, weights)
    draws = _cluster_draws(len(planned_dates))
    result["bootstrap95"] = {"status": "unavailable", "reason": "requires_6_scorable_dates"}
    if len(values) >= BOOTSTRAP_BLOCK and draws is not None:
        daily = values.reindex(planned_dates)
        mass = daily.iloc[:, 0].notna().to_numpy(dtype=float)
        arrays = np.where(np.isfinite(daily.to_numpy()), daily.to_numpy(), 0.0)
        count = mass[draws].sum(axis=1)
        means = arrays[draws].sum(axis=1) / np.where(count > 0, count, np.nan)[:, None]
        result["bootstrap95"] = {"status": "available", "blockDates": BOOTSTRAP_BLOCK, "draws": BOOTSTRAP_REPEATS, "seed": 42,
                                  **{column: _interval(means[:, index]) for index, column in enumerate(daily.columns)}}
    return result


def coverage_metrics(planned: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    issued = predictions.loc[predictions.probability.notna()]
    scorable = issued.loc[issued.label_status.eq("scorable")]
    correct = int(((scorable.probability >= 0.5) == scorable.target).sum())
    unknown = len(issued) - len(scorable)
    matured = issued.loc[issued.label_status.ne("pending_horizon")]
    matured_unknown = len(matured) - len(scorable)
    expected = planned.within_listing_lifecycle
    result = {
        "plannedRowsAllLifecycleStates": len(planned), "listedWindowPlannedRows": int(expected.sum()),
        "notYetListedRows": int(planned.lifecycle_status.eq("not_yet_listed").sum()),
        "alreadyDelistedRows": int(planned.lifecycle_status.eq("already_delisted").sum()),
        "featureAvailableRows": int(planned.feature_available.sum()),
        "listedWindowFeatureCoverage": float(planned.loc[expected, "feature_available"].mean()) if expected.any() else None,
        "rawPredictionRows": len(predictions), "issuedCalibratedRows": len(issued), "scorableIssuedRows": len(scorable),
        "unscorableIssuedRows": unknown, "labelScorableRate": len(scorable) / len(issued) if len(issued) else None,
        "calibrationUnavailableRows": int(predictions.probability.isna().sum()),
        "issuedLabelStatusCounts": {str(key): int(value) for key, value in issued.label_status.value_counts().items()},
        "listedFeatureStatusCounts": {str(key): int(value) for key, value in planned.loc[expected, "feature_status"].value_counts().items()},
        "missingOutcomeAccuracyBounds": {
            "weighting": "Simple counts of all issued calibrated forecasts; no missing outcome is filled",
            "knownCorrect": correct, "knownScorable": len(scorable), "unknown": unknown, "denominator": len(issued),
            "allUnknownWrong": correct / len(issued) if len(issued) else None,
            "allUnknownCorrect": (correct + unknown) / len(issued) if len(issued) else None,
        },
        "maturedIssuedAccuracyBounds": {
            "weighting": "Simple counts, excluding pending_horizon forecasts only; mature unknown outcomes retained",
            "knownCorrect": correct, "knownScorable": len(scorable), "unknown": matured_unknown,
            "denominator": len(matured), "pendingHorizonExcluded": len(issued) - len(matured),
            "allUnknownWrong": correct / len(matured) if len(matured) else None,
            "allUnknownCorrect": (correct + matured_unknown) / len(matured) if len(matured) else None,
        },
        "byCurrentListingStatus": {},
    }
    for status in sorted(planned.current_list_status.unique()):
        status_issued = issued.loc[issued.current_list_status.eq(status)]
        count = len(status_issued)
        unavailable = int(status_issued.label_status.ne("scorable").sum())
        result["byCurrentListingStatus"][status] = {
            "metadataTiming": "Current snapshot status, diagnostic only; not historical input",
            "stocks": int(planned.loc[planned.current_list_status.eq(status), "symbol"].nunique()),
            "issued": count, "unscorable": unavailable,
            "missingLabelRate": unavailable / count if count else None,
        }
    return result


def _distribution(records: list[dict], key: str) -> dict:
    values = np.asarray([record["metrics"][key] for record in records if record["metrics"].get(key) is not None], dtype=float)
    if not len(values):
        return {"availableStocks": 0, "allStocks": len(records)}
    return {"availableStocks": len(values), "allStocks": len(records), "weighting": "Each stock has equal weight",
            "mean": float(values.mean()), "median": float(np.median(values)), "minimum": float(values.min()),
            "maximum": float(values.max()), "p10": float(np.quantile(values, 0.1)), "p90": float(np.quantile(values, 0.9))}


def evaluate(dataset: dict, predictions: pd.DataFrame) -> dict:
    cohorts = (
        ("development_training_stocks_2019_2023", "train", "2019-01-01", "2023-12-31"),
        ("time_extrapolation_training_stocks_2024_plus", "train", "2024-01-01", "2026-08-31"),
        ("main_acceptance_permanent_holdout_stocks_2024_plus", "heldout", "2024-01-01", "2026-08-31"),
    )
    reports = {}
    for name, split, start, end in cohorts:
        planned = dataset["coverage"].loc[dataset["coverage"].split.eq(split) & dataset["coverage"].date.between(start, end)]
        dates = dataset["prediction_dates"][(dataset["prediction_dates"] >= start) & (dataset["prediction_dates"] <= end)]
        forecasts = predictions.loc[predictions.split.eq(split) & predictions.date.between(start, end)]
        report = {"start": start, "end": end, "split": split, "models": {}, "perYear": {}}
        scored = {}
        for model_id in MODELS:
            model_predictions = forecasts.loc[forecasts.model.eq(model_id)]
            scorable = model_predictions.loc[model_predictions.probability.notna() & model_predictions.label_status.eq("scorable")]
            scored[model_id] = scorable
            per_stock = []
            for symbol in sorted(planned.symbol.unique()):
                stock_planned = planned.loc[planned.symbol.eq(symbol)]
                stock_predictions = model_predictions.loc[model_predictions.symbol.eq(symbol)]
                stock_scorable = scorable.loc[scorable.symbol.eq(symbol)]
                per_stock.append({"symbol": symbol, "coverage": coverage_metrics(stock_planned, stock_predictions),
                                  "metrics": probability_metrics(stock_scorable, dates)})
            report["models"][model_id] = {
                "coverage": coverage_metrics(planned, model_predictions), "metrics": probability_metrics(scorable, dates),
                "perStock": per_stock, "stockEqualDistribution": {
                    key: _distribution(per_stock, key) for key in ("accuracy", "brier", "balancedAccuracy")},
            }
        report["primaryMinusLogistic"] = paired_metrics(scored[MODELS[0]], scored[MODELS[1]], dates)
        for year in sorted(planned.date.dt.year.unique()):
            year_dates = dates[dates.year == year]
            year_planned = planned.loc[planned.date.dt.year.eq(year)]
            report["perYear"][str(year)] = {}
            year_scored = {}
            for model_id in MODELS:
                year_predictions = forecasts.loc[forecasts.model.eq(model_id) & forecasts.date.dt.year.eq(year)]
                year_scored[model_id] = scored[model_id].loc[scored[model_id].date.dt.year.eq(year)]
                report["perYear"][str(year)][model_id] = {
                    "coverage": coverage_metrics(year_planned, year_predictions),
                    "metrics": probability_metrics(year_scored[model_id], year_dates),
                }
            report["perYear"][str(year)]["primaryMinusLogistic"] = paired_metrics(year_scored[MODELS[0]], year_scored[MODELS[1]], year_dates)
        reports[name] = report
    return reports


def specification() -> dict:
    return {
        "schemaVersion": 1, "experiment": "general-a-share-direction-v1", "hypothesis": "A fixed shared price/volume classifier can generalize across both future dates and permanently unseen stock identities",
        "stockCount": 537, "permanentHoldoutCount": 105, "features": FEATURES,
        "models": {MODELS[0]: TREE_PARAMETERS, MODELS[1]: LOGISTIC_PARAMETERS},
        "modelSelection": "None; report both fixed calibrated models without selecting a winner or tuning on acceptance outcomes",
        "calendarAnchor": "First official SSE/SZSE session of 2010", "predictionStride": PREDICTION_STRIDE,
        "refitStride": REFIT_STRIDE, "trainingYears": TRAINING_YEARS, "signalTime": "17:10 Asia/Shanghai",
        "horizonSessions": HORIZON, "label": "Adjusted next-session open to 22nd future-session open, positive return; all 22 source price dates required and no known S suspension",
        "trainingRows": "Training symbols only, original five-session grid, prior five years, label_end strictly before fit_date",
        "weighting": "Equal date total mass; training row weights normalized to mean one; scaler also fit with these training weights",
        "calibration": {"type": "monotone Platt", "slopeMinimum": 0, "years": CALIBRATION_YEARS,
                        "minimumMaturePredictionDates": MIN_CALIBRATION_DATES, "penalty": 0.01,
                        "loss": "Date-equal mean log loss + .01*((slope-1)^2+intercept^2)",
                        "sample": "Same model's raw out-of-fold predictions, training symbols only, label_end < fit_date",
                        "update": "Every 63 sessions together with model; frozen between fits; unavailable when requirements fail"},
        "frequencyBaseline": "Date-equal up frequency of exactly the same mature five-year model training sample, no smoothing",
        "rawOofStart": "2014-01-01", "development": "Training stocks only, 2019-2023",
        "acceptance": "2024-2026-08-31; report training-stock time extrapolation and permanent heldout-stock main acceptance separately",
        "heldoutDevelopmentScoring": False,
        "evaluationBoundary": "Cohorts use signal dates. Late-2023 development labels can end in 2024; development and acceptance do not have completely disjoint return windows. Permanent stock identities remain isolated from all fitting.",
        "bootstrap": {"blockPredictionDates": BOOTSTRAP_BLOCK, "draws": BOOTSTRAP_REPEATS, "seed": 42,
                      "circular": False, "clusters": "Whole original grid dates, keeping every stock from sampled dates"},
        "latest": "2026-08-31 with last scheduled 63-session model/calibrator; no special cutoff-date fit",
        "coverage": "Keep all planned symbol/date rows and all issued predictions including unknown labels; report lifecycle, source failure reasons, unscorable fractions and all-unknown-wrong/all-unknown-correct count bounds",
        "unknownFilled": False, "liveReady": False, "executionModelIncluded": False,
        "historicalCaveat": "Retrospective reconstruction from current-vintage downloaded snapshots. The broader historical market period has been seen during earlier HK research. This is not a newly issued forward validation or proof of historical source publication times.",
        "completionCriterion": "Data lineage, time and permanent-symbol isolation, coverage audits and persistence of both fixed models/calibrators/reports; no imposed accuracy or excellence target",
    }


def run(data_dir: Path = ROOT / "data/general-tushare", output_root: Path = ROOT / "runs/general") -> Path:
    install_batch_binning()
    dataset = build_general_factors(load_general_dataset(data_dir))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    source_root = destination / "source"
    source_root.mkdir()
    source_hashes = {}
    for filename in DEPENDENCIES:
        source = ROOT / "quant" / filename
        shutil.copy2(source, source_root / filename)
        source_hashes[f"quant/{filename}"] = _sha(source)
    snapshot_root = destination / "inputs"
    snapshot_root.mkdir()
    file_hashes = dataset["provenance"]["fileHashes"]
    for filename, digest in file_hashes.items():
        namespace, relative = filename.split("/", 1)
        source = Path(dataset["provenance"]["sourceRoots"][namespace]) / relative
        if _sha(source) != digest:
            raise ValueError(f"训练前来源指纹改变：{filename}")
        target = snapshot_root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha(target) != digest:
            raise ValueError(f"训练前来源快照指纹不一致：{filename}")
    dataset["factors"].to_csv(destination / "factors.csv", index=False, float_format="%.17g")
    dataset["coverage"].to_csv(destination / "coverage.csv", index=False, float_format="%.17g")
    dataset["latest_coverage"].to_csv(destination / "latest-coverage.csv", index=False, float_format="%.17g")
    dataset["stocks"].to_csv(destination / "stocks.csv", index=False)
    pd.DataFrame({"date": dataset["calendar"]}).to_csv(destination / "calendar.csv", index=False)
    _write(destination / "provenance.json", dataset["provenance"])
    recipe = specification()
    _write(destination / "specification.json", recipe)
    frozen = {
        "runId": run_id, "frozenAt": datetime.now(timezone.utc).isoformat(), "beforeTraining": True,
        "specificationSha256": _sha(destination / "specification.json"),
        "sourceHashes": source_hashes, "dataHashes": file_hashes,
        "artifactHashes": {name: _sha(destination / name) for name in ("factors.csv", "coverage.csv", "latest-coverage.csv", "stocks.csv", "calendar.csv", "provenance.json")},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "pythonVersion": sys.version,
    }
    _write(destination / "frozen-manifest.json", frozen)
    print(json.dumps({"runId": run_id, "status": "frozen_before_training", "factorRows": len(dataset["factors"]), "stocks": len(dataset["stocks"])}), flush=True)
    with threadpool_limits(limits=1):
        raw, latest, models, audits = train_walk_forward(dataset, destination)
    raw.to_csv(destination / "predictions.csv", index=False, float_format="%.17g")
    raw.drop(columns=["probability", "prediction_status"]).to_csv(destination / "raw-oof.csv", index=False, float_format="%.17g")
    latest.to_csv(destination / "latest.csv", index=False, float_format="%.17g")
    _write(destination / "fit-audit.json", audits)
    joblib.dump({"states": models, "features": list(FEATURE_KEYS), "specification": recipe, "frozenManifestSha256": _sha(destination / "frozen-manifest.json")}, destination / "models.joblib", compress=3)
    with threadpool_limits(limits=1):
        reports = evaluate(dataset, raw)
    for filename, digest in source_hashes.items():
        if _sha(ROOT / filename) != digest or _sha(source_root / Path(filename).name) != digest:
            raise ValueError(f"训练期间源码改变：{filename}")
    for filename, digest in file_hashes.items():
        namespace, relative = filename.split("/", 1)
        if (_sha(Path(dataset["provenance"]["sourceRoots"][namespace]) / relative) != digest
                or _sha(snapshot_root / filename) != digest):
            raise ValueError(f"训练期间原始来源或快照改变：{filename}")
    report = {
        "schemaVersion": 1, "runId": run_id, "completedAt": datetime.now(timezone.utc).isoformat(),
        "liveReady": False, "historicalCaveat": recipe["historicalCaveat"], "specification": recipe,
        "frozenManifestSha256": _sha(destination / "frozen-manifest.json"),
        "checks": {"sourceAndDataHashesMatch": True, "fixedTwoModels": True, "noHyperparameterSearch": True,
                   "permanentHoldoutRowsUsedForFitOrCalibration": 0, "allFitsHaveStrictlyMatureLabels": all(pd.Timestamp(row["maxLabelEnd"]) < pd.Timestamp(row["fitDate"]) for row in audits),
                   "fullPlannedCoverageRows": len(dataset["coverage"]), "rawPredictionRows": len(raw),
                   "calibratedPredictionRows": int(raw.probability.notna().sum()), "fitCount": len(audits) // len(MODELS)},
        "sourceCoverage": {"issues": dataset["provenance"]["issues"], "calendarSessions": len(dataset["calendar"]),
                           "stocks": len(dataset["stocks"]), "trainingStocks": len(dataset["training_symbols"]), "holdoutStocks": len(dataset["holdout_symbols"])},
        "cohorts": reports,
        "artifacts": {name: _sha(destination / name) for name in ("models.joblib", "raw-oof.csv", "predictions.csv", "latest.csv", "fit-audit.json")},
        "interpretation": "Report conditional direction performance together with missing-outcome bounds. Overlapping labels and common market factors are handled with date blocks, but this historical reconstruction is not an independent live record. No net strategy return or excellence claim is made.",
    }
    _write(destination / "report.json", report)
    print(json.dumps({"runId": run_id, "status": "complete", "report": str(destination / "report.json")}), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练固定 A 股通用方向模型并审计双重留出结果")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/general-tushare")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/general")
    args = parser.parse_args()
    run(args.data_dir, args.output_root)
