"""Public A-share chart evidence and reproducibility checks."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'backtests/cn/universal'
FIGURES = ('confirmation_ic', 'model_selection', 'probability_skill', 'interval_coverage')


def showcase():
    assert (ROOT / 'scripts/build_ashare_showcase.py').is_file(), 'A-share charts need a public reproducible renderer'
    return importlib.import_module('scripts.build_ashare_showcase')


def test_evidence_uses_common_universe_and_frozen_confirmation():
    evidence = showcase().load_evidence(ROOT)
    common = json.loads((RESULTS / 'development_common_universe.json').read_text())
    confirmation = json.loads((RESULTS / 'confirmation_summary.json').read_text())
    for row in evidence['selection']:
        assert row['rank_ic_mean'] == common[row['model']]['rank_ic_mean']
        assert row['ic_dates'] == 222
    assert evidence['selected_model'] == 'linear'
    assert evidence['model_version'] == 'cn-linear-frozen-20231229-v1'
    assert evidence['confirmation_start'] == '2025-01-01'
    assert evidence['confirmation_end'] == '2026-09-30'
    for row in evidence['confirmation']:
        actual = confirmation['horizons'][str(row['horizon'])]
        for key in ('rank_ic_mean', 'ic_ci_lower', 'ic_ci_upper', 'ic_dates', 'brier',
                    'baseline_brier', 'brier_skill', 'interval_coverage', 'interval_available'):
            assert row[key] == actual[key]
        assert row['brier_skill'] < 0
        assert row['ic_ci_lower'] < row['rank_ic_mean'] < row['ic_ci_upper']


def test_renderer_rejects_confirmation_based_selection(tmp_path):
    module = showcase()
    for relative in module.SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    path = tmp_path / 'backtests/cn/universal/frozen_architecture.json'
    record = json.loads(path.read_text())
    record['confirmation_used'] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='development-only'):
        module.load_evidence(tmp_path)


def test_renderer_rejects_inconsistent_metric_arithmetic(tmp_path):
    module = showcase()
    for relative in module.SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    path = tmp_path / 'backtests/cn/universal/confirmation_summary.json'
    record = json.loads(path.read_text())
    record['horizons']['20']['brier_skill'] = 0.5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Brier skill'):
        module.load_evidence(tmp_path)


def test_build_is_reproducible_without_vendor_inputs(tmp_path):
    module = showcase()
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in module.SOURCE_FILES}
    for destination in (tmp_path / 'first', tmp_path / 'second'):
        result = subprocess.run([sys.executable, '-m', 'scripts.build_ashare_showcase',
                                 '--destination', str(destination)], cwd=ROOT,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        for name in FIGURES:
            path = destination / 'assets/cn' / f'{name}.png'
            assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
            assert path.stat().st_size > 20000
        summary = json.loads((destination / 'showcase/cn/summary.json').read_text())
        assert summary['source_sha256'] == before
        for table in (destination / 'showcase/cn').glob('*.csv'):
            assert b'\r' not in table.read_bytes(), 'Use repository LF line endings'
        rows = list(csv.DictReader((destination / 'showcase/cn/confirmation.csv').open()))
        assert [int(row['horizon']) for row in rows] == [1, 5, 20, 60]
        assert float(rows[2]['rank_ic_mean']) == pytest.approx(0.11112162752244786)
    for path in (tmp_path / 'first').rglob('*'):
        if path.is_file():
            relative = path.relative_to(tmp_path / 'first')
            assert path.read_bytes() == (tmp_path / 'second' / relative).read_bytes()
            assert path.read_bytes() == (ROOT / 'docs' / relative).read_bytes(), f'Stale public chart or table: {relative}'
    assert before == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in module.SOURCE_FILES}


@pytest.mark.parametrize('name', ['README.md', 'README.zh-CN.md', 'README.ja.md', 'README.ko.md'])
def test_every_readme_embeds_all_ashare_figures(name):
    text = (ROOT / name).read_text()
    section = text.split('<a id="ashare"></a>', 1)[1].split('<a id="hong-kong"></a>', 1)[0]
    for figure in FIGURES:
        assert re.search(r'!\[[^\]]+\]\(docs/assets/cn/' + figure + r'\.png\)', section)
        assert (ROOT / 'docs/assets/cn' / f'{figure}.png').is_file()
    assert 'python -m scripts.build_ashare_showcase' in text
    for path in re.findall(r'\]\((docs/(?:assets|showcase)/cn/[^)]+)\)', text):
        assert (ROOT / path).is_file(), path
