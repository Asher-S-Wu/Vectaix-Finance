import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from hk_quant.prediction_tasks import TASK_STATUS_COLUMNS, TASK_COLUMNS, PREDICTION_SCHEMA


def complete_forecast(sid, horizon, day='2024-06-03'):
    return dict(date=day, data_as_of=day, model_trained_as_of='2024-05-31', model_version='cn-test',
                security_id=sid, horizon=horizon, prediction_schema=PREDICTION_SCHEMA,
                score=float(horizon), probability_up=.62, baseline_probability=.5,
                expected_return=.1, q10=-.08, q50=.04, q90=.12,
                baseline_q10=-.1, baseline_q50=0., baseline_q90=.1,
                status='ok', **{key:'ok' for key in TASK_STATUS_COLUMNS.values()}, reasons='',
                return_basis='CNY source-adjusted price return',
                explanations=[dict(feature='momentum_20', contribution=.1, task='score')])


@pytest.fixture
def cn_service(tmp_path):
    from ashare_quant.service import AShareQuantService
    data, models = tmp_path/'data/cn/universal', tmp_path/'models/cn/universal'
    (data/'bars').mkdir(parents=True)
    snapshot = models/'snapshots/cn-test'; snapshot.mkdir(parents=True)
    day='2024-06-03'; ids=['600000.SH','000001.SZ','430047.BJ']
    pd.DataFrame([dict(security_id=sid, exchange_code=sid, name=sid, list_date='2000-01-01',
                       delist_date=None, asset_type='equity', currency='CNY', lot_size=100,
                       identity_status='verified') for sid in ids]).to_parquet(data/'securities.parquet')
    pd.DataFrame([dict(date=day,security_id=sid,raw_open=10.,raw_close=10.,high=10.1,low=9.9,
                       volume=1_000_000.,amount=10_000_000.,adj_close=10.,up_limit=11.,down_limit=9.,
                       quote_present=True) for sid in ids]).to_parquet(data/'bars/2024.parquet')
    pd.DataFrame([dict(date=day,security_id=sid,adv20_amount=10_000_000.,volatility_20=.02)
                  for sid in ids]).to_parquet(data/'latest_features.parquet')
    history=pd.DataFrame(np.random.default_rng(71).normal(0,.01,(80,3)),columns=ids)
    history.insert(0,'date',pd.bdate_range('2024-01-01',periods=80))
    history.to_parquet(data/'risk_returns.parquet')
    pd.DataFrame([complete_forecast(sid,h) for sid in ids[:2] for h in (1,5,20,60)]).to_parquet(snapshot/'forecasts.parquet')
    acceptance=dict(model_version='cn-test',data_as_of=day,research_ready=True,eligible=False,
                    execution_validated=False,limitations=['Research sample; no production validation'])
    (snapshot/'acceptance.json').write_text(json.dumps(acceptance))
    (snapshot/'metadata.json').write_text(json.dumps(dict(model_version='cn-test',data_as_of=day,
        model_trained_as_of='2024-05-31',market='CN',currency='CNY',features=['momentum_20'],
        data_lineage={'source':'test fixture'})))
    (models/'active.json').write_text(json.dumps(dict(schema_version='ashare-v1',market='CN',currency='CNY',
        model_version='cn-test',data_as_of=day,forecast_path='snapshots/cn-test/forecasts.parquet',
        metadata_path='snapshots/cn-test/metadata.json',acceptance_path='snapshots/cn-test/acceptance.json',
        research_ready=True,eligible=False)))
    service = AShareQuantService(data=data,models=models)
    refresh_hashes(service)
    return service


def refresh_hashes(service):
    path=service.models/'active.json';active=json.loads(path.read_text())
    active['artifact_sha256']={field:hashlib.sha256((service.models/active[field]).read_bytes()).hexdigest()
        for field in ('forecast_path','metadata_path','acceptance_path')}
    path.write_text(json.dumps(active))


def test_missing_cn_snapshot_never_falls_back_to_hk(tmp_path):
    from ashare_quant.service import AShareQuantService, ModelNotPublished
    with pytest.raises(ModelNotPublished):
        AShareQuantService(data=tmp_path,models=tmp_path).status()


def test_research_snapshot_discloses_readiness_and_scope(cn_service):
    status=cn_service.status()
    assert status['research_ready'] is True and status['eligible'] is False
    assert status['execution_validated'] is False and status['currency']=='CNY'
    assert status['market']=='CN' and status['limitations']
    assert status['publication_mode']=='research'


def test_rank_stock_unscored_and_explanations_parity(cn_service):
    ranking=cn_service.rank_market(20)
    assert [r['security_id'] for r in ranking['items']]==['000001.SZ','600000.SH','430047.BJ']
    assert ranking['currency']=='CNY' and ranking['research_ready'] is True
    stock=cn_service.forecast_stock('600000')
    assert [r['horizon'] for r in stock['forecasts']]==[1,5,20,60]
    assert stock['forecasts'][0]['explanations'][0]['feature']=='momentum_20'
    missing=cn_service.forecast_stock('430047.BJ')['forecasts']
    assert all(r['status']=='not_scored' and r['expected_return'] is None for r in missing)


@pytest.mark.parametrize('defect',['escape_path','wrong_market','wrong_currency','unready','mismatched_certificate',
    'future_training','wrong_return_basis','wrong_security','missing_task','partial_invalid','duplicate'])
def test_invalid_snapshot_fails_closed(cn_service, defect):
    from ashare_quant.service import ModelNotPublished
    active_path=cn_service.models/'active.json'; active=json.loads(active_path.read_text())
    frame_path=cn_service.models/active['forecast_path']; frame=pd.read_parquet(frame_path)
    if defect=='escape_path': active['forecast_path']='../forecasts.parquet'
    elif defect=='wrong_market': active['market']='HK'
    elif defect=='wrong_currency': active['currency']='HKD'
    elif defect=='unready': active['research_ready']=False
    elif defect=='mismatched_certificate':
        path=cn_service.models/active['acceptance_path']; cert=json.loads(path.read_text()); cert['model_version']='other'; path.write_text(json.dumps(cert))
    elif defect=='future_training': frame['model_trained_as_of']='2025-01-01'
    elif defect=='wrong_return_basis': frame['return_basis']='HKD source-adjusted price return'
    elif defect=='wrong_security': frame.loc[0,'security_id']='00001.HK'
    elif defect=='missing_task': frame=frame.drop(columns=['probability_status'])
    elif defect=='partial_invalid': frame.loc[0,'probability_up']=1.5
    elif defect=='duplicate': frame=pd.concat([frame,frame.iloc[[0]]])
    active_path.write_text(json.dumps(active)); frame.to_parquet(frame_path)
    if defect != 'escape_path': refresh_hashes(cn_service)
    with pytest.raises(ModelNotPublished): cn_service.status()


def test_partial_task_status_keeps_other_task_values(cn_service):
    path=cn_service.models/'snapshots/cn-test/forecasts.parquet'; frame=pd.read_parquet(path)
    selected=frame.security_id.eq('600000.SH') & frame.horizon.eq(20)
    frame.loc[selected,['status','interval_status']]='invalid_quantile_order'
    frame.loc[selected,TASK_COLUMNS['intervals']]=None; frame.to_parquet(path)
    refresh_hashes(cn_service)
    item=next(r for r in cn_service.forecast_stock('600000.SH')['forecasts'] if r['horizon']==20)
    assert item['interval_status']=='invalid_quantile_order' and item['q10'] is None
    assert item['score']==20 and item['probability_up']==.62


def test_csv_holding_dates_and_input_rejection(cn_service):
    from ashare_quant.service import parse_holdings_csv, RequestError
    frame=parse_holdings_csv('security_id,quantity,average_cost,acquired_date\n600000.SH,200,10,2024-06-03\n')
    assert frame.iloc[0].security_id=='600000.SH' and frame.iloc[0].acquired_date=='2024-06-03'
    for csv in ('security_id,quantity\n00001.HK,100\n','security_id,quantity\n600000.SH,1.2\n',
                'security_id,quantity,sellable_quantity\n600000.SH,100,200\n'):
        with pytest.raises(ValueError): parse_holdings_csv(csv)
    with pytest.raises(RequestError): cn_service.rank_market(1.5)
    with pytest.raises(RequestError): cn_service.advise(pd.DataFrame([dict(security_id='600000.SH',quantity=-1)]),1000)


def test_service_advice_is_cny_and_has_research_disclosure(cn_service):
    result=cn_service.advise(pd.DataFrame(columns=['security_id','quantity']),100000.)
    assert result['status']=='ok'
    assert result['currency']=='CNY' and result['publication_mode']=='research'
    assert all(r['currency']=='CNY' for r in result['recommendations'])
    assert 'hkd' not in json.dumps(result).lower()


@pytest.mark.parametrize('field',['forecast_path','metadata_path','acceptance_path'])
def test_artifact_hash_corruption_fails_closed(cn_service,field):
    from ashare_quant.service import ModelNotPublished
    active=json.loads((cn_service.models/'active.json').read_text())
    path=cn_service.models/active[field]
    if field=='forecast_path':
        frame=pd.read_parquet(path);frame['score']+=1.;frame.to_parquet(path)
    else: path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ModelNotPublished): cn_service.status()


def test_missing_hash_manifest_fails_closed(cn_service):
    from ashare_quant.service import ModelNotPublished
    path=cn_service.models/'active.json';active=json.loads(path.read_text())
    del active['artifact_sha256'];path.write_text(json.dumps(active))
    with pytest.raises(ModelNotPublished): cn_service.status()


@pytest.mark.parametrize('field',['data_lineage','features','model_trained_as_of'])
def test_snapshot_requires_training_and_source_metadata(cn_service,field):
    from ashare_quant.service import ModelNotPublished
    active=json.loads((cn_service.models/'active.json').read_text())
    path=cn_service.models/active['metadata_path'];metadata=json.loads(path.read_text())
    del metadata[field];path.write_text(json.dumps(metadata));refresh_hashes(cn_service)
    with pytest.raises(ModelNotPublished): cn_service.status()


def test_request_date_validation_and_snapshot_date(cn_service):
    from ashare_quant.service import RequestError, ModelNotPublished
    assert cn_service.status()['data_as_of'].startswith('2024-06-03')
    with pytest.raises(RequestError):cn_service.rank_market(20,as_of='not a date')
    with pytest.raises(ModelNotPublished):cn_service.rank_market(20,as_of='2024-06-04')


@pytest.mark.parametrize('task,status',[('intervals','invalid_quantile_order'),('probability_up','unrepresentable_prediction')])
def test_account_buy_gate_intentionally_requires_all_forecast_tasks(cn_service,task,status):
    path=cn_service.models/'snapshots/cn-test/forecasts.parquet';frame=pd.read_parquet(path)
    chosen=frame.security_id.eq('600000.SH') & frame.horizon.eq(20)
    frame.loc[chosen,['status',TASK_STATUS_COLUMNS[task]]]=status
    frame.loc[chosen,TASK_COLUMNS[task]]=None;frame.to_parquet(path);refresh_hashes(cn_service)
    item=next(row for row in cn_service.forecast_stock('600000.SH')['forecasts'] if row['horizon']==20)
    assert item['expected_return_status']=='ok' and item['expected_return']==.1
    result=cn_service.advise(pd.DataFrame(columns=['security_id','quantity']),100000.)
    assert result['status']=='ok'
    assert '600000.SH' not in {row['security_id'] for row in result['recommendations']}
    assert '000001.SZ' in {row['security_id'] for row in result['recommendations']}
