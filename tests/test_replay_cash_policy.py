import pandas as pd
import pytest
from test_replay import inputs
from hk_quant.replay import replay_account


def cash_action(kind='cash_dividend', **changes):
    row=dict(security_id='A',action_type=kind,effective_date='2024-01-03',payment_date='2024-01-05',cash_per_share=1,cash_currency='HKD',verified=True,source_url='https://www.hkexnews.hk/final.pdf',payment_basis='issuer_final_schedule_simulated')
    row.update(changes)
    frame=pd.DataFrame([row]);frame.attrs['coverage_complete']=True
    return frame


@pytest.mark.parametrize('field,value',[('source_url',None),('source_url',''),('payment_basis',None),('payment_basis','announcement_date'),('payment_basis','cheque_despatch_deadline'),('verified',False)])
def test_cash_requires_evidence_and_explicit_basis(field,value):
    a=inputs();a[4]=cash_action(**{field:value})
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['daily'].iloc[-1]['cash']==89975


@pytest.mark.parametrize('basis',['actual_account_receipt','issuer_final_schedule_simulated'])
def test_basis_disclosed_in_summary_and_payment_event(basis):
    a=inputs();a[4]=cash_action(payment_basis=basis)
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['cash_payment_bases']==[basis]
    event=r['events'].query('cash_flow > 0').iloc[0]
    assert event.payment_basis==basis and event.source_url.endswith('final.pdf')


def test_non_session_payment_posts_on_exact_date_without_inventing_quote():
    a=inputs();a[2]=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-01-02','2024-01-03','2024-01-05'])})
    a[1]=a[1].loc[a[1].date!='2024-01-04'];a[4]=cash_action(payment_date='2024-01-04')
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='ok'
    assert r['daily'].iloc[-1]['cash']==90975
    event=r['events'].query('cash_flow > 0').iloc[0]
    assert event.date==pd.Timestamp('2024-01-04') and event.cash_after==90975
    assert pd.Timestamp('2024-01-04') not in set(r['daily'].date)


@pytest.mark.parametrize('kind',['privatisation_cash','acquisition','delisting'])
def test_cash_exit_replaces_shares_with_receivable_on_effective_date(kind):
    a=inputs();a[4]=cash_action(kind,cash_per_share=12)
    a[1]=a[1].loc[a[1].date<'2024-01-03']
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='ok'
    day=r['daily'].set_index('date').loc['2024-01-03']
    assert day.holdings_value==0 and day.receivables==12000 and day.equity==101975
    assert r['holdings'].empty
    assert r['daily'].iloc[-1]['cash']==101975


def test_non_session_split_adjusts_next_session_order():
    a=inputs();a[2]=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-01-03','2024-01-04','2024-01-05'])})
    a[1]=a[1].loc[a[1].date!='2024-01-02'];a[1].loc[a[1].date>'2024-01-01',['raw_close','vwap']]=5.;a[1]['volume']=1000000
    a[4]=pd.DataFrame([dict(security_id='A',action_type='split',effective_date='2024-01-02',share_multiplier=2,verified=True)])
    a[4].attrs['coverage_complete']=True
    r=replay_account(*a,initial_cash=100000)
    assert r['trades'].iloc[0]['quantity']==2000
    assert r['daily'].iloc[1]['holdings_value']==10000

def test_cash_exit_without_payment_date_still_cancels_verified_shares():
    a=inputs();a[4]=cash_action('privatisation_cash',cash_per_share=12,payment_date=None)
    a[1]=a[1].loc[a[1].date<'2024-01-03']
    r=replay_account(*a,initial_cash=100000)
    assert r['summary']['status']=='incomplete'
    assert r['holdings'].empty
    assert r['daily'].iloc[-1]['receivables']==12000
    assert r['daily'].iloc[-1]['equity']==101975
    assert r['daily'].iloc[-1]['cash']==89975
    assert not r['gaps'].reason.str.contains('未复权估值').any()
