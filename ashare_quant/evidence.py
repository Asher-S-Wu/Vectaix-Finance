"""Attach observed frozen-confirmation limitations without changing fitted outputs.

This narrow post-training operation writes an immutable acceptance certificate
and atomically updates only its path/hash in active.json. It does not refit, recalibrate, rescore, promote a model,
or imply that the latest refit was evaluated out of sample.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from hk_quant.registry import publication_artifact
from .service import AShareQuantService, ModelNotPublished

PROBABILITY_WARNING = (
    '冻结 Ridge 模型的未参与选型确认集在1/5/20/60交易日期限上的概率 Brier skill 均为负，'
    '概率表现未超过基准；最新重训模型本身尚未经过样本外评估。'
)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _atomic_bytes(path, payload):
    """Write in the destination directory, fsync, then replace one file atomically."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',suffix='.pending',delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value, name, *, lower=None, upper=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f'{name} is outside the admissible range')
    return value


def attach_confirmation_evidence(model_root, confirmation_path, frozen_model_version, audit_root):
    """Attach the negative-Brier Ridge confirmation evidence to a research snapshot.

    Inputs are local, self-produced artifacts. Every horizon must have negative
    and internally consistent Brier skill; otherwise this specific disclosure
    would be false and the helper refuses it. An audit directory receives exact
    pre-change files and the evidence source before any replacement. Repeating
    the same attachment is read-only and idempotent. No model is unpickled.

    The caller, which owns training orchestration, supplies the verified frozen
    model version associated with confirmation_path. This helper cannot infer
    that model identity from an unlabeled metrics filename.
    """
    root, source, audit_root = (Path(value).resolve() for value in (model_root,confirmation_path,audit_root))
    active_path = root/'active.json'
    active_before = active_path.read_bytes()
    active = json.loads(active_before)
    try:
        checked, _ = AShareQuantService(models=root)._active()
    except ModelNotPublished as exc:
        raise ValueError('Current snapshot integrity/hash validation failed') from exc
    if active_path.read_bytes() != active_before:
        raise ValueError('Active snapshot changed during validation')
    if active.get('research_ready') is not True or active.get('eligible') is not False or checked['_certificate'].get('execution_validated') is not False:
        raise ValueError('Evidence attachment requires an unpromoted research snapshot')
    if checked['_metadata'].get('kind') != 'linear':
        raise ValueError('This Ridge disclosure requires the linear model family')
    if not isinstance(frozen_model_version,str) or not frozen_model_version.strip() or frozen_model_version == active['model_version']:
        raise ValueError('A distinct verified frozen model version is required')
    if 'model_path' in active:
        model_bytes = publication_artifact(root,active['model_path']).read_bytes()
        if _sha(model_bytes) != active.get('model_sha256'):
            raise ValueError('Current model hash validation failed')
    certificate_path = publication_artifact(root,active['acceptance_path'])
    certificate_before = certificate_path.read_bytes()
    if _sha(certificate_before) != active['artifact_sha256']['acceptance_path']:
        raise ValueError('Acceptance hash changed during validation')
    source_bytes = source.read_bytes()
    confirmation = json.loads(source_bytes)
    if confirmation.get('as_of') != active['data_as_of']:
        raise ValueError('Confirmation and snapshot as-of dates must match')
    horizons = confirmation.get('horizons',{})
    if set(horizons) != {'1','5','20','60'}:
        raise ValueError('Confirmation must contain exactly horizons 1, 5, 20 and 60')
    evidence_horizons = {}
    for horizon in ('1','5','20','60'):
        row = horizons[horizon]
        brier = _number(row.get('brier'),f'{horizon}:brier',lower=0,upper=1)
        baseline = _number(row.get('baseline_brier'),f'{horizon}:baseline_brier',lower=0,upper=1)
        skill = _number(row.get('brier_skill'),f'{horizon}:brier_skill')
        ic = _number(row.get('rank_ic_mean'),f'{horizon}:rank_ic_mean',lower=-1,upper=1)
        observations = _number(row.get('probability_available'),f'{horizon}:probability_available',lower=1)
        if baseline <= 0 or skill >= 0 or not math.isclose(skill,1-brier/baseline,rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError('All horizon Brier skills must be negative and consistent with their baseline')
        evidence_row = dict(brier=brier,baseline_brier=baseline,brier_skill=skill,rank_ic_mean=ic,
                            probability_available=int(observations))
        for field in ('ic_ci_lower','ic_ci_upper'):
            if row.get(field) is not None:
                evidence_row[field] = _number(row[field],f'{horizon}:{field}',lower=-1,upper=1)
        for field in ('ic_dates','mature_labels','score_available'):
            if row.get(field) is not None:
                evidence_row[field] = int(_number(row[field],f'{horizon}:{field}',lower=0))
        evidence_horizons[horizon] = evidence_row
    evidence = dict(schema_version='frozen-confirmation-disclosure-v1',frozen_model_version=frozen_model_version,
        latest_model_version=active['model_version'],confirmation_as_of=confirmation['as_of'],
        confirmation_sha256=_sha(source_bytes),confirmation_source_name=source.name,
        model_family='Ridge',selection_set='development',evaluation_set='frozen_confirmation',
        horizons=evidence_horizons,latest_refit_out_of_sample_evaluated=False,
        all_horizon_probability_brier_skill_negative=True)
    certificate = json.loads(certificate_before)
    if certificate.get('performance_evidence') == evidence and PROBABILITY_WARNING in certificate['limitations']:
        return dict(status='already_attached',model_version=active['model_version'],confirmation_sha256=_sha(source_bytes))
    if 'performance_evidence' in certificate and certificate['performance_evidence'] != evidence:
        raise ValueError('Different confirmation evidence is already attached; review it explicitly')
    certificate['performance_evidence'] = evidence
    certificate['limitations'] = list(dict.fromkeys([*certificate['limitations'],PROBABILITY_WARNING]))
    certificate.update(research_ready=True,eligible=False,execution_validated=False)
    certificate_after = _json_bytes(certificate)
    new_certificate_path = certificate_path.parent/f'acceptance-{_sha(certificate_after)}.json'
    new_relative_path = new_certificate_path.relative_to(root).as_posix()
    updated = {**active,'acceptance_path':new_relative_path,
               'artifact_sha256':{**active['artifact_sha256'],'acceptance_path':_sha(certificate_after)}}
    updated.update(research_ready=True,eligible=False,execution_validated=False)
    active_after = _json_bytes(updated)
    if active_path.read_bytes() != active_before or certificate_path.read_bytes() != certificate_before:
        raise ValueError('Snapshot changed before disclosure attachment')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = audit_root/f'confirmation-disclosure-{stamp}-{_sha(source_bytes)[:12]}'
    directory.mkdir(parents=True,exist_ok=False)
    # Exact before-images and source evidence are durable before replacement.
    for name,payload in [('active.before.json',active_before),('acceptance.before.json',certificate_before),
                         ('confirmation.source.json',source_bytes)]:
        _atomic_bytes(directory/name,payload)
    manifest = dict(status='prepared',change_type='post_training_disclosure_only',created_at=stamp,
        model_version=active['model_version'],frozen_model_version=frozen_model_version,
        confirmation_sha256=_sha(source_bytes),active_before_sha256=_sha(active_before),
        acceptance_before_sha256=_sha(certificate_before),active_after_sha256=_sha(active_after),
        acceptance_after_sha256=_sha(certificate_after),
        unchanged_forecast_sha256=active['artifact_sha256']['forecast_path'],
        unchanged_metadata_sha256=active['artifact_sha256']['metadata_path'],unchanged_model_sha256=active.get('model_sha256'))
    _atomic_bytes(directory/'manifest.json',_json_bytes(manifest))
    if new_certificate_path.exists():
        if new_certificate_path.read_bytes() != certificate_after:
            raise ValueError('Immutable certificate content-address collision')
    else:
        _atomic_bytes(new_certificate_path,certificate_after)
    if active_path.read_bytes() != active_before:
        raise ValueError('Active snapshot changed before the atomic pointer update')
    # The old certificate remains intact. Each reader sees a complete old or new snapshot.
    _atomic_bytes(active_path,active_after)
    manifest['status'] = 'complete'
    _atomic_bytes(directory/'manifest.json',_json_bytes(manifest))
    return dict(status='attached',model_version=active['model_version'],frozen_model_version=frozen_model_version,
                confirmation_sha256=_sha(source_bytes),audit_directory=str(directory),
                acceptance_path=new_relative_path,acceptance_sha256=_sha(certificate_after),research_ready=True,eligible=False,execution_validated=False)


def attach_completed_run_evidence(results_root=None, model_root=None):
    """Resolve the frozen identity from a completed local run and attach evidence."""
    from .paths import RESULTS, MODELS
    import pandas as pd
    results, root = Path(results_root or RESULTS), Path(model_root or MODELS)
    names = ('pipeline_status.json','training_status.json','frozen_architecture.json','protocol.json')
    records = {name:json.loads((results/name).read_text(encoding='utf-8')) for name in names}
    pipeline, training = records['pipeline_status.json'], records['training_status.json']
    selection, protocol = records['frozen_architecture.json'], records['protocol.json']
    active = json.loads((root/'active.json').read_text(encoding='utf-8'))
    if pipeline.get('status') != 'complete' or training.get('status') not in ('trained','complete'):
        raise ValueError('The pipeline and real training must be complete before publication disclosure')
    if selection.get('selected_kind') != 'linear' or training.get('selected_kind') != 'linear':
        raise ValueError('Completed run must select the Ridge/linear model')
    if selection.get('selection_set') != 'development' or selection.get('confirmation_used') is not False:
        raise ValueError('Frozen architecture must be selected on development without confirmation')
    frozen_path = root/'frozen/linear.json'
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    version = training.get('frozen_model_version')
    if not isinstance(version,str) or not version or frozen.get('model_version') != version or frozen.get('kind') != 'linear':
        raise ValueError('Frozen model metadata and training status identities do not match')
    if training.get('latest_model_version') != active.get('model_version'):
        raise ValueError('Latest active model does not match the completed training status')
    try:
        cutoff = pd.Timestamp(frozen['as_of'])
        expected = pd.Timestamp(protocol['training_as_of'])
        if pd.isna(cutoff) or pd.isna(expected) or cutoff.normalize() != expected.normalize():
            raise ValueError('Frozen model cutoff does not match the persisted protocol')
    except (KeyError,TypeError) as exc:
        raise ValueError('Frozen training cutoff evidence is missing') from exc
    result = attach_confirmation_evidence(root,results/'confirmation_summary.json',version,results/'publication_audit')
    updated = json.loads((root/'active.json').read_text(encoding='utf-8'))
    evidence_hashes = {name:_sha((results/name).read_bytes()) for name in names}
    evidence_hashes['frozen/linear.json'] = _sha(frozen_bytes)
    status = dict(status='research_disclosure_attached',change_type='post_training_disclosure_only',
        model_version=updated['model_version'],frozen_model_version=version,data_as_of=updated['data_as_of'],
        confirmation_sha256=result['confirmation_sha256'],acceptance_path=updated['acceptance_path'],
        acceptance_sha256=updated['artifact_sha256']['acceptance_path'],
        evidence_source_sha256=evidence_hashes,research_ready=True,eligible=False,execution_validated=False)
    path = results/'publication_status.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) == status:
        return status
    _atomic_bytes(path,_json_bytes(status))
    return status


def main():
    import argparse
    from .paths import RESULTS, MODELS
    parser = argparse.ArgumentParser(description='Attach frozen Ridge probability limitations after a completed report; never refit or promote')
    parser.add_argument('--results-root',type=Path,default=RESULTS)
    parser.add_argument('--model-root',type=Path,default=MODELS)
    args = parser.parse_args()
    print(json.dumps(attach_completed_run_evidence(args.results_root,args.model_root),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
