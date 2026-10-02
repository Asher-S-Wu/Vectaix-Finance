"""Refresh predictions using a frozen model; never auto-promote research candidates."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import pandas as pd
from hk_quant.registry import publication_artifact
from hk_quant.training import make_examples
from .paths import DATA,MODELS,RESULTS
from .dataset import write_json


def load_active_model(model_root):
    root=Path(model_root);active=json.loads((root/'active.json').read_text())
    payload=publication_artifact(root,active['model_path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=active['model_sha256']:raise ValueError('Active model hash mismatch')
    # Only local, self-produced, hash-checked model artifacts are accepted here.
    return pickle.loads(payload)


def run_daily_update(data_root=DATA,model_root=MODELS,results_root=RESULTS):
    data_root,model_root,results_root=map(Path,[data_root,model_root,results_root])
    manifest=json.loads((data_root/'feature_manifest.json').read_text())
    if manifest.get('status')!='complete':raise ValueError('Feature dataset is incomplete')
    model=load_active_model(model_root)
    frame=pd.read_parquet(data_root/'latest_inputs.parquet')
    day=pd.Timestamp(manifest['data_as_of'])
    if frame.empty or not frame.date.eq(day).all():raise ValueError('Current feature dates do not match manifest')
    if day<model.as_of:raise ValueError('Cannot apply a future-trained model to historical dates')
    output=model.predict(make_examples(frame,model.feature_columns),explain=True)
    directory=results_root/'daily_candidates'/str(day.date());directory.mkdir(parents=True,exist_ok=True)
    path=directory/'forecasts.parquet';output.to_parquet(path,index=False)
    result=dict(status='candidate_only',published=False,market='CN',currency='CNY',model_version=model.model_version,
        model_trained_as_of=model.as_of.isoformat(),data_as_of=str(day.date()),forecast_path=str(path),
        forecast_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reason='Daily refresh creates a reviewable candidate without changing active snapshot or weights')
    write_json(directory/'status.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--model-root',type=Path,default=MODELS);p.add_argument('--results-root',type=Path,default=RESULTS)
    a=p.parse_args();print(json.dumps(run_daily_update(a.data_root,a.model_root,a.results_root),ensure_ascii=False))
