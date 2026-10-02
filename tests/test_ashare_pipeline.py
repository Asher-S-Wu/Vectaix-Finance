import importlib.util
import pytest


def test_pipeline_fails_closed_without_complete_source_calendar(tmp_path):
    assert importlib.util.find_spec('ashare_quant.pipeline') is not None,'pipeline missing'
    from ashare_quant.pipeline import run_pipeline
    with pytest.raises((ValueError,RuntimeError),match='calendar'):
        run_pipeline(tmp_path/'data',tmp_path/'results',tmp_path/'models',skip_collection=True)


def test_selected_action_collection_requires_finished_core_run(tmp_path, monkeypatch):
    import inspect,json
    from ashare_quant.pipeline import run_backtests
    from ashare_quant import corporate_actions
    from tests.test_ashare_end_to_end import _synthetic_rebalance_pipeline
    assert 'collect_actions' in inspect.signature(run_backtests).parameters,'optional action enrichment missing'
    _synthetic_rebalance_pipeline(tmp_path)
    data=tmp_path/'SYNTHETIC_data';results=tmp_path/'SYNTHETIC_results'
    calls=[]
    def capture(*args,**kwargs):
        calls.append((args,kwargs));return {'collection_status':'synthetic_test'}
    monkeypatch.setattr(corporate_actions,'collect_selected_dividends',capture)
    run_backtests(data,results,collect_actions=True)
    assert calls==[]
    assert json.loads((results/'corporate_actions_status.json').read_text())['collection_status']=='not_collected_core_completion_unverified'
    raw=data/'raw';raw.mkdir()
    import pandas as pd
    n=int(pd.read_parquet(data/'calendar.parquet').is_open.eq(1).sum())
    (raw/'collection_run.json').write_text(json.dumps({'status':'complete','requested_dates':n,'end':'20250204'}))
    run_backtests(data,results,collect_actions=True)
    assert len(calls)==1
    assert calls[0][1]['model_frozen'] is True and calls[0][1]['core_collection_stopped'] is True
    assert len(calls[0][0][1])<=700


def test_reference_receives_dated_quality_breaks_without_mutating_raw_bars(tmp_path,monkeypatch):
    import json,hashlib,pandas as pd
    from tests.test_ashare_end_to_end import _synthetic_rebalance_pipeline
    from ashare_quant.pipeline import run_backtests
    from ashare_quant import replay
    _synthetic_rebalance_pipeline(tmp_path)
    data=tmp_path/'SYNTHETIC_data';results=tmp_path/'SYNTHETIC_results'
    path=data/'feature_manifest.json';manifest=json.loads(path.read_text())
    manifest['adjustment_reference_issues']=[{'security_id':'600000.SH','date':'2025-01-02'}]
    path.write_text(json.dumps(manifest))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (data/'bars').glob('*.parquet')}
    original=replay.run_reference_replay;observed=[]
    def capture(forecasts,bars,*args,**kwargs):
        assert 'reference_continuity_break' in bars,'dated quality flags missing from reference replay'
        flagged=bars[bars.reference_continuity_break]
        observed.extend(flagged[['security_id','date']].to_dict('records'))
        return original(forecasts,bars,*args,**kwargs)
    monkeypatch.setattr(replay,'run_reference_replay',capture)
    run_backtests(data,results)
    assert len(observed)==3 and all(r['security_id']=='600000.SH' and r['date']==pd.Timestamp('2025-01-02') for r in observed)
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (data/'bars').glob('*.parquet')}
