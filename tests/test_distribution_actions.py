import pandas as pd
import pytest

from hk_quant.distributions import outcome_cash_actions


def inputs():
    records=pd.DataFrame([dict(issuer_reference='00778',status='matched_issuer_records',cash_per_unit_hkd=.1,
        ex_date=pd.Timestamp(day),payment_date=pd.Timestamp(day)+pd.Timedelta(days=20),cash_currency='HKD',
        announced_at=pd.NaT,usable_for_prediction_features=False,payment_basis='issuer_final_schedule_simulated',
        source_urls=['https://www.fortunereit.com/dividend']) for day in ('2010-03-01','2010-08-01')])
    identities=pd.DataFrame([dict(issuer_reference='00778',security_id='00778.HK',valid_from='2010-04-20',
        valid_to='2026-09-11',verified=True,source_url='https://www.hkex.com.hk/engC001.pdf')])
    return records,identities


def test_outcome_loader_preserves_cash_dates_and_excludes_pre_hk_listing():
    events,gaps=outcome_cash_actions(*inputs())
    assert len(events)==1 and len(gaps)==1
    assert events.iloc[0].effective_date==pd.Timestamp('2010-08-01')
    assert events.iloc[0].payment_date==pd.Timestamp('2010-08-21')
    assert events.announced_at.isna().all()
    assert not events.usable_for_prediction_features.any()
    assert not events.attrs['coverage_complete']
    assert gaps.iloc[0].reason=='no_verified_hk_identity_at_ex_date'


def test_unresolved_distribution_and_identity_conflict_cannot_be_imported():
    records,identities=inputs()
    records.loc[1,'status']='source_conflict'
    events,gaps=outcome_cash_actions(records,identities)
    assert events.empty and len(gaps)==2
    records,identities=inputs()
    with pytest.raises(ValueError,match='身份'):
        outcome_cash_actions(records,pd.concat([identities,identities]))


def test_outcome_loader_rejects_actual_receipt_claim_and_invalid_amount():
    records,identities=inputs()
    records.loc[1,'payment_basis']='actual_account_receipt'
    with pytest.raises(ValueError,match='模拟'):
        outcome_cash_actions(records,identities)
    records,identities=inputs();records.loc[1,'cash_per_unit_hkd']=-1
    with pytest.raises(ValueError,match='金额'):
        outcome_cash_actions(records,identities)
