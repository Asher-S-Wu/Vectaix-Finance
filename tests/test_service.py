import json
import pandas as pd
import pytest

from hk_quant.service import HKQuantService, ModelNotPublished, parse_holdings_csv


PREDICTION_SCHEMA = 'independent-task-availability-v1'
TASK_STATES = ('score_status', 'probability_status', 'interval_status', 'expected_return_status')
INTERVAL_FIELDS = ('q10', 'q50', 'q90', 'baseline_q10', 'baseline_q50', 'baseline_q90')
PREDICTION_FIELDS = ('score', 'probability_up', 'baseline_probability', 'expected_return', *INTERVAL_FIELDS)


def complete_forecast(security_id, horizon, day='2024-06-01'):
    return {'date':day, 'data_as_of':day, 'model_version':'v1', 'security_id':security_id, 'horizon':horizon,
            'prediction_schema':PREDICTION_SCHEMA, 'score':float(horizon), 'probability_up':.62,
            'baseline_probability':.5, 'expected_return':.1, 'q10':-.08, 'q50':.04, 'q90':.12,
            'baseline_q10':-.1, 'baseline_q50':0., 'baseline_q90':.1, 'status':'ok',
            **{field:'ok' for field in TASK_STATES}, 'reasons':'',
            'return_basis':'HKD source-adjusted price return',
            'explanations':[{'feature':'momentum_20','contribution':.1,'task':'score'}]}


def unavailable_forecast(security_id, horizon, day='2024-06-01'):
    return {**complete_forecast(security_id, horizon, day), **{field:None for field in PREDICTION_FIELDS},
            **{field:'insufficient_model_inputs' for field in TASK_STATES},
            'status':'insufficient_model_inputs', 'reasons':'缺数据', 'explanations':[]}


def partial_forecasts(service):
    path = service.models / 'forecast.parquet'
    frame = pd.read_parquet(path)
    selected = frame.security_id.eq('A') & frame.horizon.eq(20)
    frame.loc[selected, ['status', 'interval_status']] = 'invalid_quantile_order'
    frame.loc[selected, list(INTERVAL_FIELDS)] = None
    frame = frame.loc[~(frame.security_id.eq('B') & frame.horizon.eq(20))]
    competitor = complete_forecast('B', 20)
    competitor['score'] = 10.
    frame = pd.concat([frame, pd.DataFrame([competitor])], ignore_index=True)
    frame.to_parquet(path, index=False)
    return frame


@pytest.fixture
def service(tmp_path):
    data, models = tmp_path / 'data', tmp_path / 'models'
    (data / 'bars').mkdir(parents=True); models.mkdir()
    day = '2024-06-01'
    pd.DataFrame([
        {'security_id':'A','exchange_code':'00001.HK','name':'A','asset_type':'stock','identity_status':'verified','currency':'HKD','lot_size':100,'list_date':'2020-01-01','delist_date':None},
        {'security_id':'B','exchange_code':'00002.HK','name':'B','asset_type':'stock','identity_status':'unknown','currency':'HKD','lot_size':100,'list_date':'2020-01-01','delist_date':None},
    ]).to_parquet(data / 'securities.parquet')
    pd.DataFrame([{'date':day,'security_id':'A','raw_close':10.,'volume':10000,'quote_present':True,'fx_to_hkd':1.},{'date':day,'security_id':'B','raw_close':10.,'volume':10000,'quote_present':True,'fx_to_hkd':1.}]).to_parquet(data / 'bars/2024.parquet')
    pd.DataFrame([{'date':day,'security_id':'A','adv20_amount':100000.},{'date':day,'security_id':'B','adv20_amount':100000.}]).to_parquet(data / 'latest_features.parquet')
    pd.DataFrame({'date':pd.date_range('2024-01-01', periods=30),'A':[.01]*30,'B':[.01]*30}).to_parquet(data / 'risk_returns.parquet')
    forecast = models / 'forecast.parquet'
    pd.DataFrame([complete_forecast('A', h, day) for h in (1,5,20,60)] + [unavailable_forecast('B',20,day)]).to_parquet(forecast)
    acceptance = models / 'acceptance.json'; acceptance.write_text(json.dumps({'eligible':True,'model_version':'v1','data_as_of':day}))
    (models / 'active.json').write_text(json.dumps({'model_version':'v1','data_as_of':day,'forecast_path':forecast.name,'eligible':True,'acceptance_path':acceptance.name}))
    return HKQuantService(data=data, models=models)


def test_unpublished_never_falls_back(tmp_path):
    with pytest.raises(ModelNotPublished): HKQuantService(data=tmp_path, models=tmp_path).rank_market(20)


def test_rank_keeps_unscored_and_stock_has_four_horizons(service):
    rows = service.rank_market(20)['items']
    assert [row['security_id'] for row in rows] == ['A', 'B']
    assert service.forecast_stock('00001.HK')['forecasts'][0]['horizon'] == 1


def test_date_and_csv_are_strict(service):
    with pytest.raises(ModelNotPublished): service.rank_market(20, as_of='2024-06-02')
    with pytest.raises(ValueError): parse_holdings_csv('security_id,quantity\nA,-1\n')


def test_exact_security_identity_is_not_confused_with_reused_exchange_code(service):
    path=service.data/'securities.parquet'
    securities=pd.read_parquet(path)
    securities.loc[securities.security_id.eq('B'),'exchange_code']='A'
    securities.to_parquet(path,index=False)
    result=service.forecast_stock('A')
    assert result['security']['security_id']=='A'
    assert len(result['forecasts'])==4


def test_missing_forecasts_keep_four_explicit_unavailable_horizons(service):
    result = service.forecast_stock('B')
    assert [item['horizon'] for item in result['forecasts']] == [1, 5, 20, 60]
    for item in result['forecasts']:
        assert item['security_id'] == 'B'
        assert item['model_version'] == 'v1'
        assert item['data_as_of'].startswith('2024-06-01')
        assert item['expected_return'] is None
        if item['horizon'] == 20:
            assert item['status'] == 'insufficient_model_inputs'
            assert item['reasons'] == '缺数据'
            assert all(item[field] == 'insufficient_model_inputs' for field in TASK_STATES)
        else:
            assert item['status'] == 'not_scored'
            assert item['reasons'] == '该证券在此日期和期限没有已发布预测'
            assert all(item[field] == 'not_scored' for field in TASK_STATES)
            assert item['prediction_schema'] == PREDICTION_SCHEMA
            for field in (*PREDICTION_FIELDS, 'explanations'):
                assert item[field] is None


def test_market_unavailable_rows_have_date_horizon_and_reason(service):
    rows = service.rank_market(5)['items']
    missing = next(row for row in rows if row['security_id'] == 'B')
    assert missing['horizon'] == 5
    assert missing['model_version'] == 'v1'
    assert missing['data_as_of'].startswith('2024-06-01')
    assert missing['status'] == 'not_scored'
    assert missing['reasons'] == '该证券在此日期和期限没有已发布预测'
    assert missing['score'] is None
    assert all(missing[field] == 'not_scored' for field in TASK_STATES)
    assert missing['prediction_schema'] == PREDICTION_SCHEMA


def test_partial_interval_failure_preserves_legal_rank_probability_and_expected_return(service):
    partial_forecasts(service)
    ranking = service.rank_market(20)
    assert ranking['prediction_schema'] == PREDICTION_SCHEMA
    assert [row['security_id'] for row in ranking['items']] == ['A', 'B']
    a = ranking['items'][0]
    assert a['status'] == a['interval_status'] == 'invalid_quantile_order'
    assert a['score_status'] == a['probability_status'] == a['expected_return_status'] == 'ok'
    assert a['score'] == 20. and a['probability_up'] == .62 and a['expected_return'] == .1
    assert all(a[field] is None for field in INTERVAL_FIELDS)
    stock = service.forecast_stock('A')
    assert stock['prediction_schema'] == PREDICTION_SCHEMA
    item = next(item for item in stock['forecasts'] if item['horizon'] == 20)
    assert item['probability_up'] == .62 and item['score'] == 20.
    assert item['interval_status'] == 'invalid_quantile_order'


def test_unavailable_rank_does_not_hide_legal_probability(service):
    frame = partial_forecasts(service)
    selected = frame.security_id.eq('A') & frame.horizon.eq(5)
    frame.loc[selected, ['status', 'score_status']] = 'unrepresentable_prediction'
    frame.loc[selected, 'score'] = None
    frame.to_parquet(service.models/'forecast.parquet',index=False)
    item = next(item for item in service.forecast_stock('A')['forecasts'] if item['horizon']==5)
    assert item['score'] is None and item['score_status']=='unrepresentable_prediction'
    assert item['probability_up']==.62 and item['probability_status']=='ok'


@pytest.mark.parametrize('defect', ['absent_schema', 'old_schema', 'missing_baseline', 'missing_task_status',
                                   'false_aggregate', 'invalid_probability', 'failed_task_with_values',
                                   'duplicate_key', 'unsupported_horizon'])
def test_published_forecast_schema_and_task_contract_are_required(service, defect):
    path=service.models/'forecast.parquet'
    frame=pd.read_parquet(path)
    if defect=='absent_schema': frame=frame.drop(columns='prediction_schema')
    elif defect=='old_schema': frame['prediction_schema']='legacy-row-status'
    elif defect=='missing_baseline': frame=frame.drop(columns='baseline_probability')
    elif defect=='missing_task_status': frame=frame.drop(columns='probability_status')
    elif defect=='false_aggregate': frame.loc[0,'status']='invalid_quantile_order'
    elif defect=='invalid_probability': frame.loc[0,'probability_up']=1.1
    elif defect=='failed_task_with_values': frame.loc[0,['status','interval_status']]='invalid_quantile_order'
    elif defect=='duplicate_key': frame=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
    else: frame.loc[0,'horizon']=2
    frame.to_parquet(path,index=False)
    with pytest.raises(ModelNotPublished): service.status()


def test_partial_tasks_do_not_open_the_account_buy_gate(service):
    path=service.data/'securities.parquet'
    securities=pd.read_parquet(path)
    securities['lot_valid_from']='2020-01-01'
    securities['lot_valid_to']=None
    securities.to_parquet(path,index=False)
    holdings=pd.DataFrame(columns=['security_id','quantity'])
    # 完整预测能够进入风险检查，先确认本测试的账户与交易单位未提前阻挡候选。
    before=service.advise(holdings,10000)
    assert '市场风险因子' in before['reason']
    frame=pd.read_parquet(service.models/'forecast.parquet')
    selected=frame.security_id.eq('A') & frame.horizon.eq(20)
    frame.loc[selected,['status','interval_status']]='invalid_quantile_order'
    frame.loc[selected,list(INTERVAL_FIELDS)]=None
    frame.to_parquet(service.models/'forecast.parquet',index=False)
    result=service.advise(holdings,10000)
    assert result['status']=='data_error'
    assert result['reason']=='没有具备可交易证据、交易单位及币种汇率的候选证券'
    assert result['recommendations']==[]


def test_model_contribution_lists_from_parquet_remain_readable_json(service):
    path=service.models/'forecast.parquet'
    frame=pd.read_parquet(path)
    contributions=[{'feature':'momentum_20','contribution':.1,'task':'score'},
                   {'feature':'volatility_60','contribution':-.02,'task':'expected_return'}]
    frame['explanations']=[contributions if sid=='A' else [] for sid in frame.security_id]
    frame.to_parquet(path,index=False)
    result=service.forecast_stock('A')
    assert result['forecasts'][0]['explanations']==contributions
