from fastapi.testclient import TestClient
import pytest
from hk_quant.api import create_app
from test_service import service
from test_service import partial_forecasts, TASK_STATES, PREDICTION_SCHEMA

def test_auth_and_request_validation(service):
    app=create_app(service,'key'); client=TestClient(app)
    assert client.get('/health').status_code==200
    assert client.get('/v1/rankings').status_code==401
    assert client.get('/v1/rankings',headers={'X-API-Key':'key'}).status_code==200
    assert client.get('/v1/rankings?horizon=2',headers={'X-API-Key':'key'}).status_code==422
    assert client.get('/v1/rankings?limit=-1',headers={'X-API-Key':'key'}).status_code==422
    assert client.post('/v1/portfolio/advice',headers={'X-API-Key':'key'},json={'holdings':[],'cash':100}).status_code in (200,422)
    assert client.post('/v1/portfolio/advice',headers={'X-API-Key':'key'},json={'holdings':[{'security_id':'A','quantity':-1}],'cash':100}).status_code==422

def test_empty_key_rejected(service):
    with pytest.raises(ValueError): create_app(service,'')


def test_csv_account_input_reaches_same_advice_service(service,monkeypatch):
    captured={}
    def advice(holdings,cash,risk,day):
        captured.update(holdings=holdings,cash=cash,risk=risk,day=day)
        return {'status':'ok'}
    monkeypatch.setattr(service,'advise',advice)
    client=TestClient(create_app(service,'key'))
    body={'holdings_csv':'security_id,quantity,average_cost\nA,200,10.5\n','cash':10000,'risk_profile':{'horizon':20}}
    response=client.post('/v1/portfolio/advice/csv',json=body,headers={'X-API-Key':'key'})
    assert response.status_code==200
    assert captured['holdings'].iloc[0].to_dict()=={'security_id':'A','quantity':200,'average_cost':10.5}
    assert captured['cash']==10000 and captured['risk']=={'horizon':20}
    assert client.post('/v1/portfolio/advice/csv',json=body).status_code==401


@pytest.mark.parametrize('csv',['security_id,quantity\nA,abc\n','security_id,quantity\nA,1\nA,2\n','security_id,quantity\n   ,1\n'])
def test_invalid_csv_is_a_controlled_input_error(service,csv):
    client=TestClient(create_app(service,'key'))
    response=client.post('/v1/portfolio/advice/csv',json={'holdings_csv':csv,'cash':100},headers={'X-API-Key':'key'})
    assert response.status_code==422


def test_partial_forecast_tasks_and_missing_rows_survive_json_endpoints(service):
    partial_forecasts(service)
    client=TestClient(create_app(service,'key'))
    headers={'X-API-Key':'key'}
    ranking=client.get('/v1/rankings?horizon=20',headers=headers)
    assert ranking.status_code==200
    payload=ranking.json()
    assert payload['prediction_schema']==PREDICTION_SCHEMA
    assert [item['security_id'] for item in payload['items']]==['A','B']
    a=payload['items'][0]
    assert a['interval_status']=='invalid_quantile_order' and a['q10'] is None
    assert a['score_status']=='ok' and a['score']==20.
    assert a['probability_status']=='ok' and a['probability_up']==.62
    response=client.get('/v1/stocks/A/forecast',headers=headers)
    assert response.status_code==200
    twenty=next(item for item in response.json()['forecasts'] if item['horizon']==20)
    assert twenty['probability_up']==.62 and twenty['interval_status']=='invalid_quantile_order'
    missing=client.get('/v1/stocks/B/forecast',headers=headers).json()['forecasts'][0]
    assert all(missing[field]=='not_scored' for field in TASK_STATES)
    assert missing['prediction_schema']==PREDICTION_SCHEMA and missing['probability_up'] is None


def test_old_published_schema_is_a_controlled_unavailable_model_response(service):
    import pandas as pd
    path=service.models/'forecast.parquet'
    pd.read_parquet(path).drop(columns='prediction_schema').to_parquet(path,index=False)
    response=TestClient(create_app(service,'key')).get('/v1/rankings',headers={'X-API-Key':'key'})
    assert response.status_code==503
