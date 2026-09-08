from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .cross_market_data import support
from .cross_market_factors import FEATURE_KEYS, _bar_flags, _features
from .cross_market_identity import load_cross_market_identity
from .general_training import apply_calibrator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_snapshot(run_dir: Path, snapshot_dir: Path) -> Path:
    """Apply the frozen JOINT model to independently supplied HK research snapshots."""
    run_dir, snapshot_dir = run_dir.resolve(), snapshot_dir.resolve()
    artifacts = json.loads((run_dir/"artifacts.json").read_text())
    for name in ("models.joblib", "report.json"):
        if sha(run_dir/name) != artifacts[name]:
            raise ValueError(f"冻结模型产物已改变：{name}")
    report = json.loads((run_dir/"report.json").read_text())
    model_states = joblib.load(run_dir/"models.joblib")
    state = max((v for key, v in model_states.items() if key.endswith(":JOINT")), key=lambda v: v["fit_date"])
    if state["features"] != list(FEATURE_KEYS) or state["calibrator"]["status"] != "available":
        raise ValueError("因子契约不符或没有可用概率校准，不能推断。")
    manifest = json.loads((snapshot_dir/"manifest.json").read_text())
    input_hashes = dict(manifest["inputHashes"])
    input_hashes[str(snapshot_dir/"manifest.json")] = sha(snapshot_dir/"manifest.json")
    for filename, expected in input_hashes.items():
        if sha(filename) != expected:
            raise ValueError(f"预测输入已改变：{filename}")
    stocks = pd.DataFrame(manifest["stocks"])
    identity = load_cross_market_identity(pd.DataFrame(columns=["ts_code", "market"]), stocks)
    stocks = identity["stocks"]
    calendar, benchmark, support_hashes = support()
    for filename, expected in support_hashes.items():
        if report["data"]["inputFileHashes"][filename] != expected:
            raise ValueError("预测基准或日历与训练冻结版本不同。")
    input_hashes.update(support_hashes)
    if state["fit_date"] > calendar[-1]:
        raise ValueError("模型拟合时间晚于预测数据截点。")
    requests = {(r["symbol"], r["mode"]): r for r in manifest["requests"]}
    predictions, feature_rows = [], []
    for stock in stocks.to_dict("records"):
        symbol = stock["ts_code"]; code = symbol.removesuffix(".HK")
        output = {"symbol": symbol, "name": stock["name"], "market": "HK", "as_of": str(calendar[-1].date()),
                  "model_fit_date": str(state["fit_date"].date()), "horizon_local_sessions": 21,
                  "status": "unavailable", "up_probability": None, "raw_probability": None}
        if stock["identity_status"] != "admitted":
            output["reason"] = stock["identity_reasons"]; predictions.append(output); continue
        price_receipt, factor_receipt = requests[(code, "raw")], requests[(code, "qfq-factor")]
        if (sha(snapshot_dir/price_receipt["rawFile"]) != price_receipt["sha256"]
                or sha(factor_receipt["sourceRawFile"]) != factor_receipt["rawSha256"]):
            raise ValueError(f"{symbol} 原始响应指纹不符。")
        input_hashes[factor_receipt["sourceRawFile"]] = factor_receipt["rawSha256"]
        raw = pd.read_csv(snapshot_dir/price_receipt["file"], float_precision="round_trip")
        events = pd.read_csv(snapshot_dir/factor_receipt["file"], float_precision="round_trip")
        raw["date"] = pd.to_datetime(raw.date, format="%Y-%m-%d").dt.as_unit("ns")
        events["date"] = pd.to_datetime(events.date, format="%Y-%m-%d").dt.as_unit("ns")
        if (raw.date.duplicated().any() or events.date.duplicated().any()
                or not events.qfq_factor.gt(0).all() or not np.isfinite(events.qfq_factor).all()):
            raise ValueError(f"{symbol} 原价日期或调整因子无效。")
        raw = raw.loc[raw.date.isin(calendar)].sort_values("date")
        if raw.date.lt(pd.Timestamp(stock["list_date"])).any() or pd.notna(stock["delist_date"]):
            raise ValueError(f"{symbol} 当前推断身份生命周期未核准。")
        raw = pd.merge_asof(raw, events.sort_values("date"), on="date", direction="backward")
        frame = raw.rename(columns={"amount": "turnover", "qfq_factor": "adj_factor"}).set_index("date").reindex(calendar)
        flags = _bar_flags(frame, "HK")
        flags["invalid_adjustment"] = ~np.isfinite(frame.adj_factor) | frame.adj_factor.le(0)
        valid = ~pd.DataFrame(flags).any(axis=1)
        factors, _ = _features(frame, valid, benchmark.set_index("date").close)
        latest = factors.loc[calendar[-1:]]
        if not np.isfinite(latest.to_numpy(float)).all():
            output["reason"] = "incomplete_or_invalid_source_window"; predictions.append(output); continue
        with threadpool_limits(limits=2):
            probability = state["model"].predict_proba(latest.to_numpy(float))[:, 1]
        calibrated = apply_calibrator(state["calibrator"], probability)
        output.update(status="historical_snapshot_prediction", raw_probability=float(probability[0]),
                      up_probability=float(calibrated[0]))
        predictions.append(output)
        feature_rows.append({"symbol": symbol, "date": str(calendar[-1].date()), **{k: float(latest.iloc[0][k]) for k in FEATURE_KEYS}})
    destination = run_dir/"external-hk-predictions.json"
    if destination.exists():
        raise ValueError("本次池外预测产物已存在，禁止覆盖已冻结发行记录。")
    for filename, expected in input_hashes.items():
        if sha(filename) != expected:
            raise ValueError(f"推断期间来源发生改变：{filename}")
    result = {"model": "JOINT", "trainingRun": run_dir.name, "modelSha256": artifacts["models.joblib"],
              "predictionSourceSha256": sha(Path(__file__)), "predictions": predictions, "features": feature_rows,
              "inputHashes": input_hashes,
              "interpretation": "Calibrated probability of positive supplier-adjusted next-open to 22nd-local-open price change; historical snapshot only, not a live order or proven prospective forecast",
              "modelSelectionUsedThesePredictions": False, "futureOutcomeEvaluated": False,
              "identity": identity["provenance"]}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用冻结共同模型计算池外港股预测")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(predict_snapshot(args.run, args.snapshot))
