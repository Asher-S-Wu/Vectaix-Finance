"""Public-release packaging checks; no vendor market inputs are required."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_report_builder_accepts_count_only_corporate_action_summary(tmp_path):
    pytest.importorskip("reportlab")
    source = ROOT / "delivery_report/report_data.json"
    data = json.loads(source.read_text())
    data["manifest"].pop("adjustment_reference_issues", None)
    execution = data["execution"]
    for name in ("unresolved_corporate_actions", "frozen_corporate_action_positions"):
        if name in execution:
            execution[name + "_count"] = len(execution.pop(name))
    summary = tmp_path / "public-report.json"
    summary.write_text(json.dumps(data))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "delivery_report/build_report_pdf.py"),
         "--input", str(summary), "--output", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "ashare_training_report.zh-CN.pdf").stat().st_size > 1000
    assert "仍有 5 个企业行动持仓冻结" in (
        tmp_path / "ashare_training_report.zh-CN.md"
    ).read_text()


def test_public_report_contains_only_aggregate_source_summary():
    data = json.loads((ROOT / "delivery_report/report_data.json").read_text())
    assert "adjustment_reference_issues" not in data["manifest"]
    assert "unresolved_corporate_actions" not in data["execution"]
    assert "frozen_corporate_action_positions" not in data["execution"]
    assert data["execution"]["frozen_corporate_action_positions_count"] == 5


def test_public_release_inventory_and_model_hashes():
    path = ROOT / "ASHARE_RELEASE.json"
    assert path.is_file(), "Public release needs its own inventory"
    manifest = json.loads(path.read_text())
    assert manifest["requires_user_market_data"] is True
    assert manifest["eligible"] is False
    assert manifest["execution_validated"] is False
    for name, record in manifest["files"].items():
        assert not name.startswith("data/")
        assert not name.endswith(".parquet")
        assert name != "models/cn/universal/active.json"
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["sha256"], name
    for name, digest in manifest["model_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
