from fastapi.testclient import TestClient
import pytest
from test_ashare_service import cn_service


def client(service):
    from ashare_quant.api import create_app
    return TestClient(create_app(service,'test-key'))


def test_auth_endpoints_and_no_key(cn_service):
    from ashare_quant.api import create_app
    c=client(cn_service); h={'X-API-Key':'test-key'}
    assert c.get('/health').json()=={'status':'ok','market':'CN','currency':'CNY'}
    assert c.get('/v1/model/status').status_code==401
    assert c.get('/v1/model/status',headers=h).json()['publication_mode']=='research'
    assert c.get('/v1/rankings?horizon=20',headers=h).status_code==200
    assert c.get('/v1/stocks/600000.SH/forecast',headers=h).status_code==200
    assert c.get('/v1/stocks/00001.HK/forecast',headers=h).status_code in (404,422)
    with pytest.raises(ValueError): create_app(cn_service,'')


def test_json_and_csv_advice_have_same_output(cn_service):
    c=client(cn_service); h={'X-API-Key':'test-key'}
    record=dict(security_id='600000.SH',quantity=100,acquired_date='2024-06-03')
    a=c.post('/v1/portfolio/advice',headers=h,json=dict(holdings=[record],cash=100000.))
    b=c.post('/v1/portfolio/advice/csv',headers=h,json=dict(holdings_csv='security_id,quantity,acquired_date\n600000.SH,100,2024-06-03\n',cash=100000.))
    assert a.status_code==b.status_code==200
    assert a.json()==b.json()


@pytest.mark.parametrize('body',[
    dict(holdings=[dict(security_id='00001.HK',quantity=100)],cash=1000),
    dict(holdings=[dict(security_id='600000.SH',quantity=1.5)],cash=1000),
    dict(holdings=[dict(security_id='600000.SH',quantity=100,acquired_date='bad')],cash=1000),
    dict(holdings=[dict(security_id='600000.SH',quantity=100,sellable_quantity=200)],cash=1000),
    dict(holdings=[],cash=1000,risk_profile=dict(unknown=.1)),
])
def test_invalid_advice_inputs_are_422(cn_service,body):
    response=client(cn_service).post('/v1/portfolio/advice',headers={'X-API-Key':'test-key'},json=body)
    assert response.status_code==422


def test_invalid_csv_and_missing_snapshot_are_controlled(cn_service):
    c=client(cn_service); h={'X-API-Key':'test-key'}
    assert c.post('/v1/portfolio/advice/csv',headers=h,json=dict(holdings_csv='security_id,quantity\n600000.SH,no\n',cash=1)).status_code==422
    (cn_service.models/'active.json').unlink()
    assert c.get('/v1/rankings',headers=h).status_code==503
