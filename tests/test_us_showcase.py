"""US aggregate chart contracts, defensive checks and deterministic regeneration."""
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
RESULTS = 'backtests/us/oef2015'
FIGURES = ('confirmation_ic', 'model_selection', 'probability_skill', 'interval_coverage')


def showcase():
    assert (ROOT / 'scripts/build_us_showcase.py').is_file(), 'US charts need an aggregate-only renderer'
    return importlib.import_module('scripts.build_us_showcase')


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def evidence_root(tmp_path):
    """Explicit synthetic contract fixture, never published as research results."""
    common = {kind: dict(rank_ic_mean=value, ic_dates=200, date_sha256='shared-dates',
                        common_score_rows=22000, common_mature_rows=18000)
              for kind, value in [('factor', -.01), ('linear', .02),
                                  ('lightgbm_small', .03), ('lightgbm_large', .01)]}
    records = {
        'protocol': dict(market='US', currency='USD', declared_cohort_count=101,
                         universe_date='2015-06-30', universe_known_by='2015-09-02',
                         development_start='2024-01-01', development_end='2024-12-31',
                         confirmation_start='2025-01-01', confirmation_end='2026-09-30',
                         horizons=[1, 5, 20, 60], survivorship_free=False, full_us_market=False,
                         bootstrap=dict(block_sessions=60, repetitions=2000, seed=42)),
        'development_common_universe': common,
        'frozen_architecture': dict(selected_kind='lightgbm_small', selection_statistic=.03,
                                   selection_set='development', confirmation_used=False,
                                   development_sha256=hashlib.sha256(json.dumps(common, sort_keys=True, allow_nan=False).encode()).hexdigest()),
        'source_audit': dict(declared_cohort=101, available_original_identities=89,
                             unavailable_original_identities=12, cohort_survivorship_free=False,
                             full_us_market=False, source_current_vintage=True,
                             start='2014-01-02', end='2026-09-30'),
        'training_status': dict(status='complete', selected_kind='lightgbm_small',
                                confirmation_used_for_selection=False, execution_validated=False, eligible=False),
        'latest_refit_summary': dict(kind='lightgbm_small', model_version='us-test-latest',
                                     as_of='2026-09-30', out_of_sample_evaluated=False,
                                     execution_validated=False, eligible=False),
        'confirmation_summary': dict(as_of='2026-09-30', horizons={str(h): dict(
            rank_ic_mean=.02, ic_ci_lower=-.03, ic_ci_upper=.06, ic_dates=400,
            ic_calendar_slots=400, predictions=45450, mature_labels=35000,
            missing_or_immature_outcomes=10450, score_available=35000,
            probability_available=34000, brier=.251, baseline_brier=.25,
            brier_skill=1 - .251/.25, interval_available=32000,
            interval_covered=25600, interval_coverage=.8) for h in [1, 5, 20, 60]})}
    for name, value in records.items():
        write(tmp_path, f'{RESULTS}/{name}.json', value)
    write(tmp_path, 'models/us/oef2015/frozen/lightgbm_small.json',
          dict(kind='lightgbm_small', model_version='us-test-frozen', market='US', currency='USD',
               protocol_sha256=hashlib.sha256((tmp_path/f'{RESULTS}/protocol.json').read_bytes()).hexdigest(),
               source_audit_sha256=hashlib.sha256((tmp_path/f'{RESULTS}/source_audit.json').read_bytes()).hexdigest()))
    return tmp_path


def mutate(root, name, callback):
    path = root / f'{RESULTS}/{name}.json'
    value = json.loads(path.read_text())
    callback(value)
    path.write_text(json.dumps(value))


def test_uses_saved_model_and_discloses_the_entire_cohort(evidence_root):
    evidence = showcase().load_evidence(evidence_root)
    assert evidence['selected_model'] == 'lightgbm_small'
    assert evidence['model_version'] == 'us-test-frozen'
    assert evidence['available_original_identities'] == 89
    assert evidence['unavailable_original_identities'] == 12
    assert evidence['declared_cohort'] == 101
    assert [row['horizon'] for row in evidence['confirmation']] == [1, 5, 20, 60]
    assert evidence['confirmation'][2]['ic_ci_lower'] == -.03
    assert len(evidence['source_sha256']) == 8


@pytest.mark.parametrize('name,change,error', [
    ('frozen_architecture', lambda x: x.update(confirmation_used=True), 'development-only'),
    ('frozen_architecture', lambda x: x.update(development_sha256='changed'), 'freeze hash'),
    ('frozen_architecture', lambda x: x.update(selection_statistic=.5), 'Selection statistic'),
    ('training_status', lambda x: x.update(status='running'), 'complete training'),
    ('latest_refit_summary', lambda x: x.update(out_of_sample_evaluated=True), 'latest refit'),
    ('confirmation_summary', lambda x: x['horizons']['20'].update(brier_skill=.5), 'Brier skill'),
    ('confirmation_summary', lambda x: x['horizons']['20'].update(interval_covered=10), 'Interval coverage'),
    ('confirmation_summary', lambda x: x['horizons']['20'].update(rank_ic_mean=float('nan')), 'Non-finite'),
    ('source_audit', lambda x: x.update(unavailable_original_identities=0), 'cohort counts'),
])
def test_rejects_inconsistent_evidence(evidence_root, name, change, error):
    mutate(evidence_root, name, change)
    with pytest.raises(ValueError, match=error):
        showcase().load_evidence(evidence_root)


def test_rejects_different_comparison_dates(evidence_root):
    mutate(evidence_root, 'development_common_universe',
           lambda x: x['factor'].update(date_sha256='different-dates'))
    common = json.loads((evidence_root/f'{RESULTS}/development_common_universe.json').read_text())
    mutate(evidence_root, 'frozen_architecture', lambda x: x.update(
        development_sha256=hashlib.sha256(json.dumps(common, sort_keys=True, allow_nan=False).encode()).hexdigest()))
    with pytest.raises(ValueError, match='same rows and dates'):
        showcase().load_evidence(evidence_root)


def test_synthetic_contract_renders_deterministically(evidence_root, tmp_path):
    module = showcase()
    for target in [tmp_path/'first', tmp_path/'second']:
        module.build(evidence_root, target)
    for path in (tmp_path/'first').rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (tmp_path/'second'/path.relative_to(tmp_path/'first')).read_bytes()
    for name in FIGURES:
        path = tmp_path/'first/assets/us'/f'{name}.png'
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert path.stat().st_size > 20000
    summary = json.loads((tmp_path/'first/showcase/us/summary.json').read_text())
    assert summary['survivorship_free'] is False
    assert summary['full_us_market'] is False
    assert summary['execution_validated'] is False
    assert summary['latest_refit_evaluated_out_of_sample'] is False


def test_real_artifacts_regenerate_without_vendor_inputs(tmp_path):
    module = showcase()
    evidence = module.load_evidence(ROOT)
    result = subprocess.run([sys.executable, '-m', 'scripts.build_us_showcase', '--destination', str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (ROOT/'docs'/path.relative_to(tmp_path)).read_bytes()
    summary = json.loads((tmp_path/'showcase/us/summary.json').read_text())
    assert all(summary['source_sha256'][name] == value for name, value in evidence['source_sha256'].items())
    assert len(summary['source_sha256']) == 10
    assert (tmp_path/'assets/us/conditional_replay.png').is_file()
    rows = list(csv.DictReader((tmp_path/'showcase/us/confirmation.csv').open()))
    assert [float(row['rank_ic_mean']) for row in rows] == [row['rank_ic_mean'] for row in evidence['confirmation']]


@pytest.mark.parametrize('name', ['README.md', 'README.zh-CN.md', 'README.ja.md', 'README.ko.md'])
def test_each_readme_has_consistent_upfront_us_evidence(name):
    text = (ROOT/name).read_text()
    section = text.split('<a id="us"></a>', 1)[1].split('<a id="ashare"></a>', 1)[0]
    assert '101' in section and '2015-06-30' in section and '2015-09-02' in section
    assert 'python -m scripts.build_us_showcase' in section
    evidence = showcase().load_evidence(ROOT)
    for row in evidence['confirmation']:
        assert f'{row["rank_ic_mean"]:.4f}' in section
        assert f'{row["brier_skill"]:+.2%}' in section
        assert f'{row["interval_coverage"]:.2%}' in section
    for figure in FIGURES:
        assert re.search(r'!\[[^\]]+\]\(docs/assets/us/'+figure+r'\.png\)', section)
    for path in re.findall(r'\]\((docs/(?:assets|showcase)/us/[^)]+)\)', section):
        assert (ROOT/path).is_file(), path


def test_replay_reconciles_saved_curves_and_summary():
    module = showcase()
    assert hasattr(module, 'load_replay'), 'Replay charts need independent aggregate validation'
    evidence = module.load_evidence(ROOT)
    replay = module.load_replay(ROOT, evidence)
    assert len(replay['curves']) == 437 * 4
    assert replay['metrics'][0]['total_return'] == pytest.approx(.16008797982445966)
    assert replay['metrics'][0]['mean_equity_exposure'] == pytest.approx(.9470, abs=.00005)
    assert replay['metrics'][2]['mean_equity_exposure'] == pytest.approx(.8383, abs=.00005)
    assert all(row['execution_validated'] is False for row in replay['metrics'])


def test_replay_rejects_altered_aggregate_curves(tmp_path):
    module = showcase()
    assert hasattr(module, 'load_replay'), 'Replay charts need independent aggregate validation'
    evidence = module.load_evidence(ROOT)
    for name in ['replay_summary.json', 'replay_curves.csv', 'source_audit.json']:
        target = tmp_path/RESULTS/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/RESULTS/name).read_bytes())
    with (tmp_path/RESULTS/'replay_curves.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='Replay curve hash'):
        module.load_replay(tmp_path, evidence)
