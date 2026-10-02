"""Disclosure-only evidence updates never promote or rewrite a research model."""
import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from test_ashare_service import cn_service, refresh_hashes


def helper():
    assert importlib.util.find_spec('ashare_quant.evidence') is not None, 'disclosure evidence helper not implemented'
    return importlib.import_module('ashare_quant.evidence').attach_confirmation_evidence


@pytest.fixture
def evidence_input(cn_service,tmp_path):
    active_path=cn_service.models/'active.json';active=json.loads(active_path.read_text())
    metadata_path=cn_service.models/active['metadata_path'];metadata=json.loads(metadata_path.read_text())
    metadata['kind']='linear';metadata_path.write_text(json.dumps(metadata));refresh_hashes(cn_service)
    source=tmp_path/'confirmation_summary.json'
    summary=dict(as_of=active['data_as_of'],horizons={str(h):dict(
        brier=.26,baseline_brier=.25,brier_skill=-.04,rank_ic_mean=.05,
        ic_ci_lower=.02,ic_ci_upper=.08,ic_dates=250,probability_available=10000,
        mature_labels=10100,score_available=10000) for h in (1,5,20,60)})
    source.write_text(json.dumps(summary))
    return cn_service,source,tmp_path/'disclosure_audit'


def test_attach_keeps_numeric_model_and_readiness_unchanged(evidence_input):
    service,source,audit=evidence_input;active_path=service.models/'active.json'
    before_active=active_path.read_bytes();before=json.loads(before_active)
    certificate_path=service.models/before['acceptance_path'];before_certificate=certificate_path.read_bytes()
    forecasts_before=(service.models/before['forecast_path']).read_bytes()
    metadata_before=(service.models/before['metadata_path']).read_bytes()
    result=helper()(service.models,source,'cn-linear-frozen-20231229-v1',audit)
    assert result['status']=='attached'
    active=json.loads(active_path.read_text());new_certificate_path=service.models/active['acceptance_path']
    certificate=json.loads(new_certificate_path.read_text())
    assert certificate_path.read_bytes()==before_certificate
    assert new_certificate_path!=certificate_path
    assert active['research_ready'] is True and active['eligible'] is False
    assert certificate['research_ready'] is True and certificate['eligible'] is False and certificate['execution_validated'] is False
    assert (service.models/before['forecast_path']).read_bytes()==forecasts_before
    assert (service.models/before['metadata_path']).read_bytes()==metadata_before
    assert active['artifact_sha256']['acceptance_path']==hashlib.sha256(new_certificate_path.read_bytes()).hexdigest()
    evidence=certificate['performance_evidence']
    assert evidence['frozen_model_version']=='cn-linear-frozen-20231229-v1'
    assert evidence['latest_refit_out_of_sample_evaluated'] is False
    assert all(evidence['horizons'][str(h)]['brier_skill']<0 for h in (1,5,20,60))
    folder=Path(result['audit_directory'])
    assert (folder/'active.before.json').read_bytes()==before_active
    assert (folder/'acceptance.before.json').read_bytes()==before_certificate
    assert json.loads((folder/'manifest.json').read_text())['status']=='complete'
    assert any('Brier' in warning and '最新' in warning for warning in service.status()['limitations'])
    assert any('Brier' in warning for warning in service.rank_market(20)['limitations'])
    assert any('Brier' in warning for warning in service.forecast_stock('600000.SH')['limitations'])
    assert any('Brier' in warning for warning in service.advise(pd.DataFrame(columns=['security_id','quantity']),10000.)['limitations'])


def test_repeat_is_idempotent_without_new_audit_or_artifact_write(evidence_input):
    service,source,audit=evidence_input;run=helper()
    run(service.models,source,'cn-frozen-v1',audit)
    active=(service.models/'active.json').read_bytes();paths=set(audit.rglob('*'))
    result=run(service.models,source,'cn-frozen-v1',audit)
    assert result['status']=='already_attached'
    assert (service.models/'active.json').read_bytes()==active and set(audit.rglob('*'))==paths


def test_invalid_acceptance_hash_is_not_replaced(evidence_input):
    service,source,audit=evidence_input;run=helper()
    active_path=service.models/'active.json';active=active_path.read_bytes();state=json.loads(active)
    certificate_path=service.models/state['acceptance_path'];certificate_path.write_bytes(certificate_path.read_bytes()+b' ')
    bad_certificate=certificate_path.read_bytes()
    with pytest.raises(ValueError,match='snapshot|hash|integrity'):
        run(service.models,source,'cn-frozen-v1',audit)
    assert active_path.read_bytes()==active and certificate_path.read_bytes()==bad_certificate
    assert not audit.exists()


@pytest.mark.parametrize('defect',['missing_horizon','positive_skill','inconsistent_skill','wrong_asof','bad_ic','empty_version','same_version'])
def test_bad_evidence_is_rejected_without_touching_snapshot(evidence_input,defect):
    service,source,audit=evidence_input;run=helper()
    version='cn-frozen-v1';evidence=json.loads(source.read_text())
    if defect=='missing_horizon':del evidence['horizons']['60']
    elif defect=='positive_skill':evidence['horizons']['1'].update(brier=.24,brier_skill=.04)
    elif defect=='inconsistent_skill':evidence['horizons']['1']['brier_skill']=-.8
    elif defect=='wrong_asof':evidence['as_of']='2024-06-02'
    elif defect=='bad_ic':evidence['horizons']['1']['rank_ic_mean']=5
    elif defect=='empty_version':version=''
    elif defect=='same_version':version='cn-test'
    source.write_text(json.dumps(evidence));before=(service.models/'active.json').read_bytes()
    with pytest.raises(ValueError):run(service.models,source,version,audit)
    assert (service.models/'active.json').read_bytes()==before and not audit.exists()


def test_checked_model_pickle_bytes_are_never_changed(evidence_input):
    service,source,audit=evidence_input;run=helper()
    # The updater hashes bytes only and never unpickles them.
    model=service.models/'snapshots/cn-test/model.pkl';model.write_bytes(b'self-produced-fixture-model-bytes')
    active_path=service.models/'active.json';active=json.loads(active_path.read_text())
    active.update(model_path='snapshots/cn-test/model.pkl',model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    active_path.write_text(json.dumps(active));before=model.read_bytes()
    run(service.models,source,'cn-frozen-v1',audit)
    assert model.read_bytes()==before


def test_readers_see_valid_old_snapshot_until_atomic_pointer_flip(evidence_input,monkeypatch):
    service,source,audit=evidence_input;run=helper()
    module=importlib.import_module('ashare_quant.evidence');original=module._atomic_bytes
    observations=[]
    def write(path,payload):
        if Path(path)==service.models/'active.json':
            observations.append(service.status()['limitations'])
        return original(path,payload)
    monkeypatch.setattr(module,'_atomic_bytes',write)
    run(service.models,source,'cn-frozen-v1',audit)
    assert len(observations)==1
    assert not any('Brier' in warning for warning in observations[0])
    assert any('Brier' in warning for warning in service.status()['limitations'])


def test_failed_pointer_update_preserves_readable_prior_snapshot(evidence_input,monkeypatch):
    service,source,audit=evidence_input;run=helper()
    module=importlib.import_module('ashare_quant.evidence');original=module._atomic_bytes
    before=(service.models/'active.json').read_bytes()
    def write(path,payload):
        if Path(path)==service.models/'active.json':raise OSError('simulated unavailable pointer write')
        return original(path,payload)
    monkeypatch.setattr(module,'_atomic_bytes',write)
    with pytest.raises(OSError):run(service.models,source,'cn-frozen-v1',audit)
    assert (service.models/'active.json').read_bytes()==before
    assert service.status()['research_ready'] is True
    assert not any('Brier' in warning for warning in service.status()['limitations'])


@pytest.fixture
def completed_run(evidence_input):
    service,source,audit=evidence_input
    results=source.parent/'results';results.mkdir();(results/'confirmation_summary.json').write_bytes(source.read_bytes())
    files={
        'pipeline_status.json':dict(status='complete'),
        'training_status.json':dict(status='trained',selected_kind='linear',latest_model_version='cn-test',frozen_model_version='cn-linear-frozen-v1'),
        'frozen_architecture.json':dict(selected_kind='linear',selection_set='development',confirmation_used=False),
        'protocol.json':dict(training_as_of='2023-12-29'),
    }
    for name,value in files.items():(results/name).write_text(json.dumps(value))
    (service.models/'frozen').mkdir()
    (service.models/'frozen/linear.json').write_text(json.dumps(dict(model_version='cn-linear-frozen-v1',kind='linear',as_of='2023-12-29T00:00:00')))
    return service,results


def test_completed_run_resolves_frozen_version_from_matching_records(completed_run):
    service,results=completed_run;module=importlib.import_module('ashare_quant.evidence')
    assert hasattr(module,'attach_completed_run_evidence'),'completed-run entrypoint missing'
    result=module.attach_completed_run_evidence(results,service.models)
    assert result['status']=='research_disclosure_attached'
    assert result['frozen_model_version']=='cn-linear-frozen-v1'
    saved=(results/'publication_status.json').read_bytes()
    assert json.loads(saved)==result
    module.attach_completed_run_evidence(results,service.models)
    assert (results/'publication_status.json').read_bytes()==saved


@pytest.mark.parametrize('defect',['pipeline_pending','frozen_mismatch','selection_from_confirmation','latest_mismatch','cutoff_mismatch'])
def test_completed_run_refuses_incomplete_or_inconsistent_identity(completed_run,defect):
    service,results=completed_run;module=importlib.import_module('ashare_quant.evidence')
    assert hasattr(module,'attach_completed_run_evidence'),'completed-run entrypoint missing'
    if defect=='pipeline_pending':path=results/'pipeline_status.json';value=dict(status='running')
    elif defect=='frozen_mismatch':path=service.models/'frozen/linear.json';value=dict(model_version='wrong',kind='linear',as_of='2023-12-29')
    elif defect=='selection_from_confirmation':path=results/'frozen_architecture.json';value=dict(selected_kind='linear',selection_set='confirmation',confirmation_used=True)
    elif defect=='latest_mismatch':path=results/'training_status.json';value=dict(status='trained',selected_kind='linear',latest_model_version='wrong',frozen_model_version='cn-linear-frozen-v1')
    else:path=service.models/'frozen/linear.json';value=dict(model_version='cn-linear-frozen-v1',kind='linear',as_of='2024-01-31')
    path.write_text(json.dumps(value));before=(service.models/'active.json').read_bytes()
    with pytest.raises(ValueError):module.attach_completed_run_evidence(results,service.models)
    assert (service.models/'active.json').read_bytes()==before
    assert not (results/'publication_status.json').exists()


def test_cli_attaches_completed_fixture_run(completed_run):
    import subprocess,sys
    service,results=completed_run
    result=subprocess.run([sys.executable,'-m','ashare_quant.evidence','--results-root',str(results),'--model-root',str(service.models)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout)
    assert output['status']=='research_disclosure_attached'
    assert any('Brier' in warning for warning in service.status()['limitations'])


def test_relative_model_and_results_paths_work_in_documented_cli(completed_run,monkeypatch):
    import os
    service,results=completed_run;module=importlib.import_module('ashare_quant.evidence')
    parent=results.parent;monkeypatch.chdir(parent)
    output=module.attach_completed_run_evidence(os.path.relpath(results,parent),os.path.relpath(service.models,parent))
    assert output['status']=='research_disclosure_attached'
    assert not Path(output['acceptance_path']).is_absolute()
