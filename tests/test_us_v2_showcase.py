"""Aggregate-only US v2 reporting, with fail-closed scientific labels."""
from __future__ import annotations
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = 'backtests/us/oef2015-rank-v2'
FILES = [f'{RESULTS}/{name}.json' for name in (
    'protocol', 'source_audit', 'annual_common_ic', 'development_selection',
    'nested_selection', 'frozen_architecture', 'reused_diagnostic_summary',
    'training_status', 'latest_refit_summary')]
FILES += ['models/us/oef2015-rank-v2/frozen/model.json']
FIGURES = ('rolling_selection', 'nested_selection', 'reused_diagnostic_ic',
           'calibration_checks', 'source_recovery')


def module():
    spec = importlib.util.find_spec('scripts.build_us_v2_showcase')
    assert spec is not None, 'A dedicated v2 aggregate renderer is required'
    return importlib.import_module('scripts.build_us_v2_showcase')


@pytest.fixture
def evidence_root(tmp_path):
    for name in FILES:
        dest = tmp_path/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    return tmp_path


def mutate(root, name, callback):
    path = root/f'{RESULTS}/{name}.json'
    value = json.loads(path.read_text())
    callback(value)
    path.write_text(json.dumps(value))


def refresh_freeze(root):
    mutate(root, 'frozen_architecture', lambda x: x.update(**{
        key: hashlib.sha256((root/f'{RESULTS}/{name}.json').read_bytes()).hexdigest()
        for key, name in [('annual_common_ic_sha256', 'annual_common_ic'),
                          ('selection_sha256', 'development_selection'),
                          ('nested_selection_sha256', 'nested_selection')]}))


def test_requires_dedicated_aggregate_renderer():
    assert callable(module().load_evidence)


def test_saved_evidence_is_reconciled(evidence_root):
    evidence = module().load_evidence(evidence_root)
    assert evidence['selected_model'] == 'ridge_rank_5y'
    assert evidence['selection_statistic'] == pytest.approx(.027149742617581567)
    assert evidence['mean_outer_ic'] == pytest.approx(.0008557292652002947)
    assert len(evidence['annual']) == 30
    assert len(evidence['selection']) == 6
    assert len(evidence['nested']) == 3
    assert len(evidence['diagnostic']) == 4
    assert evidence['recovered_quote_rows'] == 15284
    assert len(evidence['recovered']) == 12
    assert len(evidence['coverage']) == 101
    assert evidence['untouched_holdout'] is False
    assert len(evidence['source_sha256']) == 10


@pytest.mark.parametrize('name,change,error', [
    ('training_status', lambda x: x.update(status='running'), 'complete training'),
    ('training_status', lambda x: x.update(recent_period_used_for_selection=True), 'reused diagnostic'),
    ('reused_diagnostic_summary', lambda x: x.update(untouched_holdout=True), 'reused diagnostic'),
    ('latest_refit_summary', lambda x: x.update(out_of_sample_evaluated=True), 'latest refit'),
    ('frozen_architecture', lambda x: x.update(selection_statistic=.5), 'Selection statistic'),
    ('frozen_architecture', lambda x: x.update(selection_sha256='changed'), 'freeze hash'),
    ('reused_diagnostic_summary', lambda x: x['horizons']['20'].update(brier_skill=.5), 'Brier skill'),
    ('reused_diagnostic_summary', lambda x: x['horizons']['20'].update(interval_covered=1), 'Interval coverage'),
    ('reused_diagnostic_summary', lambda x: x['horizons']['20'].update(rank_ic_mean=float('nan')), 'Non-finite'),
    ('source_audit', lambda x: x.update(identities_with_some_history=100), 'cohort counts'),
])
def test_rejects_unsupported_claims(evidence_root, name, change, error):
    mutate(evidence_root, name, change)
    with pytest.raises(ValueError, match=error):
        module().load_evidence(evidence_root)


def test_rejects_different_candidate_samples(evidence_root):
    mutate(evidence_root, 'annual_common_ic', lambda x: x['2020']['factor_reference'].update(date_sha256='other'))
    refresh_freeze(evidence_root)
    with pytest.raises(ValueError, match='same rows and dates'):
        module().load_evidence(evidence_root)


def test_rejects_nested_selection_with_outer_year_leakage(evidence_root):
    mutate(evidence_root, 'nested_selection', lambda x: x['outer_path'][0]['inner_years'].append(2022))
    refresh_freeze(evidence_root)
    with pytest.raises(ValueError, match='strictly earlier'):
        module().load_evidence(evidence_root)


def test_rejects_altered_derived_selection_mean(evidence_root):
    mutate(evidence_root, 'development_selection', lambda x: x['ridge_rank_5y'].update(mean_annual_ic=.6))
    refresh_freeze(evidence_root)
    with pytest.raises(ValueError, match='annual metrics'):
        module().load_evidence(evidence_root)


def test_static_outputs_are_deterministic_and_aggregate_only(evidence_root, tmp_path):
    for name in ('first', 'second'):
        module().build(evidence_root, tmp_path/name)
    for path in (tmp_path/'first').rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (tmp_path/'second'/path.relative_to(tmp_path/'first')).read_bytes()
    for name in FIGURES:
        path = tmp_path/'first/assets/us/v2'/f'{name}.png'
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert path.stat().st_size > 20000
    summary = json.loads((tmp_path/'first/showcase/us/v2/summary.json').read_text())
    for key in ('untouched_holdout', 'survivorship_free', 'full_us_market',
                'terminal_outcomes_complete', 'execution_validated', 'eligible',
                'latest_refit_evaluated_out_of_sample'):
        assert summary[key] is False
    source = list(csv.DictReader((tmp_path/'first/showcase/us/v2/source_recovery.csv').open()))
    fox = next(r for r in source if r['security_id'] == 'FOXA')
    assert fox['supported_data_end'] == '2018-03-27'
    assert fox['last_original_trade_date'] == '2019-03-19'
    assert fox['has_source_gap_before_terminal'] == 'True'


def test_checked_in_figures_regenerate_without_vendor_inputs(tmp_path):
    module()
    command = subprocess.run([sys.executable, '-m', 'scripts.build_us_v2_showcase', '--destination', str(tmp_path)],
                             cwd=ROOT, capture_output=True, text=True)
    assert command.returncode == 0, command.stderr
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (ROOT/'docs'/path.relative_to(tmp_path)).read_bytes()


PRESERVED = {
    'README.md': ('082a4bf752a7fd3d661bd3d44d938fa1172560ca97d31e0f70a3779c0ea314cd', '9278c5c96e49f0aeaf6ae2cdef1964f231c9c76eec6fa3590653fff637428e7b'),
    'README.zh-CN.md': ('c016edde70210c356ee9a69bb79811613c25f3fdba878be7a662f845ae0ad51c', 'c1213c86aafa9bd78218af16d1809a06ebc69ec674fa963df39d66a74bfd92fc'),
    'README.ja.md': ('30cf5dab83e3a2c016cbdbe2f1f5bc2cad10cad8c9d5ef9915788a77a93729d0', 'b417b913ce5f46eeefb4ff8e68fcf20ec7b7e3efe0cfdeec92ddf397d29e205d'),
    'README.ko.md': ('3ac2a7820e6d5f1c2d68e4b16cd1bc1252bf475b953ec989bff754f1a070389e', '980843f1a527b0e1aa5df26c36d8ef4cd51302137f157cafa30ea5fa147988b8'),
}


@pytest.mark.parametrize('name', PRESERVED)
def test_readmes_preserve_initial_experiment_and_other_markets(name):
    content = (ROOT/name).read_text()
    assert '<a id="us-v1"></a>' in content, 'Keep a separate initial experiment anchor'
    initial = content.split('<a id="us-v1"></a>', 1)[1].split('<a id="ashare"></a>', 1)[0]
    other = content.split('<a id="ashare"></a>', 1)[1]
    assert hashlib.sha256(initial.encode()).hexdigest() == PRESERVED[name][0]
    assert hashlib.sha256(other.encode()).hexdigest() == PRESERVED[name][1]
    section = content.split('<a id="us"></a>', 1)[1].split('<a id="us-v1"></a>', 1)[0]
    assert 'REUSED DIAGNOSTIC' in section and 'ridge_rank_5y' in section
    assert 'python -m scripts.build_us_v2_showcase' in section
    for value in ['0.0271', '0.0009', '15,284', '101', '12']:
        assert value in section
    for name in FIGURES[:4]:
        assert re.search(r'!\[[^\]]+\]\(docs/assets/us/v2/'+name+r'\.png\)', section)
    for target in re.findall(r'\]\((docs/(?:assets|showcase|us)/us?/?.*?)\)', section):
        assert (ROOT/target).is_file()


def test_paired_and_replay_evidence_preserve_limits():
    evidence = module().load_evidence(ROOT)
    assert 'paired' in evidence, 'Include saved common-universe comparison separately'
    assert evidence['paired'][2]['second_run_ic'] != evidence['diagnostic'][2]['rank_ic_mean']
    assert evidence['paired'][2]['paired_ic_change'] == pytest.approx(.10046845925028751)
    assert 'replay' in evidence, 'Carry through terminal-aware portfolio status'
    for row in evidence['replay']:
        if row['strategy'] != 'spy_15bps':
            assert row['total_return'] is None
            assert row['valuation_gap_sessions'] == 274
            assert row['performance_scope'] == 'reference_only'


def test_replay_rejects_reference_returns_promoted_to_validated(evidence_root):
    for name in ('paired_recent_summary.json', 'replay_summary.json', 'replay_curves.csv'):
        shutil.copyfile(ROOT/RESULTS/name, evidence_root/RESULTS/name)
    mutate(evidence_root, 'replay_summary', lambda x: x['strategies']['selected_15bps'].update(total_return=.08125))
    with pytest.raises(ValueError, match='Unresolved portfolio'):
        module().load_evidence(evidence_root)


def test_chart_title_is_computed_from_saved_values():
    helper = getattr(module(), 'diagnostic_title', None)
    assert callable(helper), 'The diagnostic title must reflect saved values dynamically'
    assert helper([{'rank_ic_mean': .1, 'ic_ci_lower': -.1, 'ic_ci_upper': .2}]) == 'Positive point estimates; all intervals cross zero'
    assert helper([{'rank_ic_mean': -.1, 'ic_ci_lower': -.2, 'ic_ci_upper': -.01}]) == 'Frozen ranking on the reused diagnostic period'
