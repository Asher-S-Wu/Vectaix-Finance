import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hk_quant.contracts import RiskProfile
from hk_quant.strategy import SavedForecastPolicy, run_strategy_backtest
from test_portfolio import inputs


def files(tmp_path):
    root=tmp_path/'data';root.mkdir();(root/'bars').mkdir();(root/'features').mkdir();(root/'references').mkdir()
    forecasts=tmp_path/'forecasts';forecasts.mkdir()
    a=inputs();dates=pd.bdate_range('2024-06-03',periods=5)
    bars=pd.concat([a[3].assign(date=day,vwap=a[3].raw_close,amount=a[3].raw_close*a[3].volume,data_valid=True) for day in dates],ignore_index=True)
    bars=bars.drop(columns='adv20_amount');bars.to_parquet(root/'bars'/'2024.parquet',index=False)
    factors=bars[['date','security_id']].assign(adv20_amount=10000000.)
    factors.to_parquet(root/'features'/'panel.parquet',index=False)
    master=a[4].assign(isin=['ISIN_A','ISIN_B'])
    master.to_parquet(root/'securities.parquet',index=False)
    lots=master[['security_id','lot_size','lot_valid_from','lot_valid_to']].assign(verified=True,source_url='https://www.hkexnews.hk/lot.pdf')
    lots.to_parquet(root/'historical_lots.parquet',index=False)
    actions=pd.DataFrame(columns=['security_id','action_type','effective_date','payment_date','verified','source_url','payment_basis'])
    actions.to_parquet(root/'corporate_actions.parquet',index=False)
    a[5].rename_axis('date').reset_index().to_parquet(root/'risk_returns.parquet',index=False)
    pd.DataFrame({'cal_date':dates,'is_open':1}).to_parquet(root/'references'/'calendar.parquet',index=False)
    predictions=pd.concat([a[0].assign(date=day,data_as_of=day.strftime('%Y-%m-%d'),model_trained_as_of='2024-05-31',fwd_return=999.,label_end=pd.Timestamp('2030-01-01')) for day in dates],ignore_index=True)
    predictions.to_parquet(forecasts/'2024.parquet',index=False)
    evidence=dict(verified=True,start_date='2024-06-03',end_date='2024-06-07',security_ids=['A','B'],source_urls=['https://www.hkexnews.hk/coverage.pdf'],historical_lot_coverage_complete=True,corporate_action_cash_coverage_complete=True)
    (root/'execution_coverage.json').write_text(json.dumps(evidence))
    return root,forecasts,dates,master,lots,a[5],factors,bars


def policy_and_account(tmp_path):
    root,forecast_dir,dates,master,lots,returns,factors,bars=files(tmp_path)
    policy=SavedForecastPolicy(forecast_dir,master,lots,returns,factors)
    account=dict(cash=100000.,equity=110000.,receivables=10000.,holdings=pd.DataFrame(columns=['security_id','quantity']),bars=bars[bars.date.eq(dates[0])])
    return policy,account,dates,root,forecast_dir


def test_policy_removes_labels_and_uses_total_nav_without_spending_receivables(tmp_path,monkeypatch):
    import hk_quant.strategy as strategy
    policy,account,dates,_,_=policy_and_account(tmp_path)
    actual=strategy.advise_portfolio
    def observed(forecasts,holdings,cash,*args,**kwargs):
        assert not any(c.startswith(('fwd_','label_')) for c in forecasts.columns)
        assert cash==100000.
        return actual(forecasts,holdings,cash,*args,**kwargs)
    monkeypatch.setattr(strategy,'advise_portfolio',observed)
    targets=policy(dates[0],account)
    report=policy.reports[-1]
    assert report['summary']['equity']==100000
    assert np.isclose(targets.target_weight.sum()*110000,sum(report['recommendations'].target_value_hkd))
    assert targets.target_weight.sum()<.2


@pytest.mark.parametrize('field,value',[('data_as_of','2024-06-04'),('model_trained_as_of','2024-06-04')])
def test_future_forecast_metadata_is_rejected(tmp_path,field,value):
    _,account,dates,root,forecast_dir=policy_and_account(tmp_path)
    path=forecast_dir/'2024.parquet';frame=pd.read_parquet(path);frame.loc[frame.date.eq(dates[0]),field]=value;frame.to_parquet(path,index=False)
    policy=SavedForecastPolicy(forecast_dir,pd.read_parquet(root/'securities.parquet'),pd.read_parquet(root/'historical_lots.parquet'),pd.read_parquet(root/'risk_returns.parquet').set_index('date'),pd.read_parquet(root/'features'))
    with pytest.raises(ValueError,match='时间'):
        policy(dates[0],account)


def test_later_predictions_and_returns_cannot_change_earlier_targets(tmp_path):
    policy,account,dates,root,forecast_dir=policy_and_account(tmp_path)
    before=policy(dates[0],account)
    path=forecast_dir/'2024.parquet';frame=pd.read_parquet(path);frame.loc[frame.date.gt(dates[0]),'expected_return']=100.;frame.to_parquet(path,index=False)
    history=policy.returns.copy();history.loc[pd.Timestamp('2030-01-01')]=999.
    after=SavedForecastPolicy(forecast_dir,policy.master,policy.historical_lots,history,policy.daily_factors)(dates[0],account)
    pd.testing.assert_frame_equal(before,after)


def test_four_cash_replays_share_pool_and_evaluate_with_evidence(tmp_path):
    root,forecast_dir,dates,*_=files(tmp_path)
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1],initial_cash=100000.)
    assert result['status']=='ok'
    assert result['portfolio_metrics']['portfolio_valid'] is True
    assert set(result['runs'])=={'model','equal_weight','stress_model','stress_equal_weight'}
    for run in result['runs'].values():
        assert run['daily'].cash.ge(0).all()
        assert run['trades'].date.min()==dates[1]
        assert run['summary']['gap_count']==0
    assert result['runs']['stress_equal_weight']['summary']['fees']>result['runs']['equal_weight']['summary']['fees']
    assert set(result['policies']['model'][0]['recommendations'].security_id)==set(result['policies']['equal_weight'][0]['recommendations'].security_id)


def test_empty_actions_without_coverage_do_not_prove_no_events(tmp_path):
    root,forecast_dir,dates,*_=files(tmp_path)
    (root/'execution_coverage.json').unlink()
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1])
    assert result['status']=='incomplete' and result['portfolio_metrics'] is None


def test_today_lot_cannot_authorize_past_execution(tmp_path):
    root,forecast_dir,dates,*_=files(tmp_path)
    lotfile=root/'historical_lots.parquet';lots=pd.read_parquet(lotfile);lots['lot_valid_from']='2026-01-01';lots.to_parquet(lotfile,index=False)
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1])
    assert result['status']=='incomplete'
    assert all(run['trades'].empty for run in result['runs'].values())
    assert result['portfolio_metrics']['portfolio_valid'] is False


def test_dividend_receivable_is_accounted_in_policy_nav_and_paid_only_on_date(tmp_path):
    root,forecast_dir,dates,*_=files(tmp_path)
    actions=pd.DataFrame([dict(security_id='A',action_type='cash_dividend',effective_date=dates[2],payment_date=dates[4],cash_per_share_hkd=1.,verified=True,source_url='https://www.hkexnews.hk/final.pdf',payment_basis='issuer_final_schedule_simulated')])
    actions.to_parquet(root/'corporate_actions.parquet',index=False)
    path=root/'bars'/'2024.parquet';bars=pd.read_parquet(path)
    bars.loc[bars.security_id.eq('A') & bars.date.ge(dates[2]),['raw_close','vwap']]=9.
    bars.to_parquet(path,index=False)
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1],initial_cash=100000.)
    assert result['status']=='ok'
    replay=result['runs']['model'];day=replay['daily'].set_index('date').loc[dates[2]]
    report=next(row for row in result['policies']['model'] if row['date']==dates[2])
    entitled_quantity=replay['events'].query("event_type == 'entitlement'").entitled_quantity.iloc[0]
    assert day.receivables==entitled_quantity
    assert np.isclose(report['summary']['equity']+day.receivables,day.equity)
    planned=replay['targets'].query('signal_date == @dates[2]')
    assert np.isclose(planned.target_weight.sum()*day.equity,report['recommendations'].target_value_hkd.sum())
    payments=replay['events'].query("event_type == 'payment'")
    assert payments.date.tolist()==[dates[4]] and payments.cash_flow.iloc[0]==entitled_quantity


def test_partial_historical_lots_cannot_claim_full_execution_coverage(tmp_path):
    root,forecast_dir,dates,*_=files(tmp_path)
    path=root/'historical_lots.parquet';lots=pd.read_parquet(path);lots=lots[lots.security_id.eq('A')];lots.to_parquet(path,index=False)
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1],initial_cash=100000.)
    assert result['status']=='incomplete'
    assert result['portfolio_metrics']['portfolio_valid'] is False
    assert result['execution_summary']['historical_lot_coverage_complete'] is False


def missing_lot_day(tmp_path,present,volume,amount,valid):
    root,forecast_dir,dates,*_=files(tmp_path)
    path=root/'historical_lots.parquet';lots=pd.read_parquet(path)
    before=lots.loc[lots.security_id.eq('A')].assign(lot_valid_to=dates[1].strftime('%Y-%m-%d'))
    after=lots.loc[lots.security_id.eq('A')].assign(lot_valid_from=dates[3].strftime('%Y-%m-%d'))
    pd.concat([lots.loc[lots.security_id.ne('A')],before,after],ignore_index=True).to_parquet(path,index=False)
    path=root/'bars'/'2024.parquet';bars=pd.read_parquet(path).astype({'volume':float})
    bars.loc[bars.security_id.eq('A')&bars.date.ge(dates[2]),['raw_close','vwap']]=9.
    bars.loc[bars.security_id.eq('A')&bars.date.eq(dates[2]),['quote_present','volume','amount','data_valid']]=[present,volume,amount,valid]
    bars.to_parquet(path,index=False)
    return root,forecast_dir,dates


@pytest.mark.parametrize('amount',[0.,np.nan])
def test_lot_coverage_exempts_only_confirmed_zero_volume_dates(tmp_path,amount):
    root,forecast_dir,dates=missing_lot_day(tmp_path,False,0.,amount,True)
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1],initial_cash=100000.)
    assert result['status']=='ok'
    assert result['execution_summary']['historical_lot_coverage_complete'] is True
    assert result['execution_summary']['historical_lot_gaps']==[]
    for run in result['runs'].values():
        assert run['gaps'].empty
        assert run['trades'].loc[run['trades'].security_id.eq('A')&run['trades'].date.eq(dates[2])].empty
        before=run['daily'].set_index('date').loc[dates[1]]
        day=run['daily'].set_index('date').loc[dates[2]]
        assert next(p['quantity'] for p in before.positions if p['security_id']=='A')==next(p['quantity'] for p in day.positions if p['security_id']=='A')


@pytest.mark.parametrize('present,volume,amount,valid',[
    (False,np.nan,0.,True),
    (False,100.,1000.,True),
    (True,0.,0.,True),
    (False,0.,100.,True),
    (False,0.,0.,False),
])
def test_unknown_or_conflicting_dates_still_require_lot_coverage(tmp_path,present,volume,amount,valid):
    root,forecast_dir,dates=missing_lot_day(tmp_path,present,volume,amount,valid)
    result=run_strategy_backtest(root,forecast_dir,start=dates[0],end=dates[-1],initial_cash=100000.)
    assert result['status']=='incomplete'
    assert result['execution_summary']['historical_lot_coverage_complete'] is False
    assert result['execution_summary']['historical_lot_gaps']==[dict(security_id='A',dates_without_unique_lot=1,first_date=dates[2].isoformat(),last_date=dates[2].isoformat())]
