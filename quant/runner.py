from __future__ import annotations

import hashlib
import fcntl
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"


def read_report() -> dict:
    try:
        pointer = json.loads((RUNS / "latest.json").read_text(encoding="utf-8"))
        run_id = pointer["id"]
        if not isinstance(run_id, str) or not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
            raise ValueError("研究批次编号无效")
        report = json.loads((RUNS / run_id / "report.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report["schemaVersion"] != 2 or report["run"]["id"] != run_id:
            raise ValueError("报告版本或研究批次不一致")
        return report
    except FileNotFoundError as exc:
        raise ValueError("找不到最新研究记录或报告；本次仅查看结果，不会启动训练。") from exc
    except (UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("研究记录或报告已损坏，无法读取；现有文件保持原样。") from exc


def execute_research(progress: Callable[[str, float], None]) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".research.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有另一个本地研究正在运行，请等它完成") from exc
        try:
            return _execute_locked(progress)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _execute_locked(progress: Callable[[str, float], None]) -> dict:
    from .engine import run_research
    from .data import verify_snapshot

    progress("校验本地行情与股票池", 0.01)
    manifest = verify_snapshot(DATA)
    sources = [ROOT / "quant" / name for name in ("engine.py", "factors.py", "portfolio.py", "metrics.py")]
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "exchange-calendars")}
    runtime["python"] = sys.version
    engine_digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources) + json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    prior = sorted(RUNS.glob("*/holdout-opened.json"))
    for marker in prior:
        previous = json.loads((marker.parent / "provenance.json").read_text())
        if previous["datasetSha256"] != manifest["sha256"] or previous["engineSha256"] != engine_digest:
            raise ValueError("这段独立历史已经被另一版模型或数据评估，不能再次用来选模型；新的实验需要重新划分未使用的验收区间")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    destination = RUNS / run_id
    destination.mkdir(parents=True)
    record = {"id": run_id, "createdAt": now.isoformat(), "datasetSha256": manifest["sha256"], "engineSha256": engine_digest, "runtime": runtime, "priorEvaluations": len(prior)}
    (destination / "provenance.json").write_text(json.dumps(record, indent=2))
    report = run_research(DATA, destination, progress)
    report["dataset"] = manifest
    report["run"] = {**record, "completedAt": datetime.now(timezone.utc).isoformat(), "evaluationNote": "首次完成独立期验收" if not prior else "独立历史已被评估；重复运行仅用于复现，不构成新的独立验证"}
    for stock_report in report["stockReports"]:
        stock_report["run"] = report["run"]
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    pointer = RUNS / f"latest.{run_id}.pending.json"
    pointer.write_text(json.dumps({"id": run_id}))
    pointer.replace(RUNS / "latest.json")
    progress("研究完成，已保存全部候选与独立验收结果", 1.0)
    return report
