import pandas as pd

from hk_quant.distribution_versions import versioned_cash_actions


def records():
    base=dict(security_id='A',event_id='A:final',published_at='2026-03-01T18:00:00+08:00',
        ex_date='2026-03-10',payment_date='2026-03-20',cash_per_unit_decimal='0.12',
        cash_currency='HKD',conditional_cash_amount=True,verified=True,
        record_date_source_conflict=False,final_schedule_usable=False,
        payment_date_type='specified_payment_date',scrip_optional=False,
        scrip_default_option=None,source_url='https://www.hkexnews.hk/first.pdf')
    final={**base,'published_at':'2026-03-12T20:00:00+08:00',
           'cash_per_unit_decimal':'0.11','conditional_cash_amount':False,
           'final_schedule_usable':True,'source_url':'https://www.hkexnews.hk/final.pdf'}
    return pd.DataFrame([base,final])


def test_final_cash_preserves_late_publication_and_native_currency():
    p=records();p['cash_currency']='CNY'
    actions,gaps=versioned_cash_actions(p,'2026-03-21T19:00:00+08:00')
    assert len(actions)==1 and gaps.empty
    a=actions.iloc[0]
    assert a.cash_per_share==0.11 and a.cash_currency=='CNY'
    assert 'cash_per_share_hkd' not in actions
    assert a.amount_known_at==pd.Timestamp('2026-03-12T20:00:00+08:00')
    assert a.effective_date==pd.Timestamp('2026-03-10')
    assert actions.attrs['coverage_complete'] is False


def test_unresolved_or_deadline_does_not_produce_payment():
    for column,value in [('record_date_source_conflict',True),
                         ('payment_date_type','on_or_before'),
                         ('conditional_cash_amount',True)]:
        p=records();p.loc[1,column]=value
        actions,gaps=versioned_cash_actions(p,'2026-03-21T19:00:00+08:00')
        assert actions.empty and len(gaps)==1


def test_cannot_use_final_revision_before_publication():
    actions,gaps=versioned_cash_actions(records(),'2026-03-11T19:00:00+08:00')
    assert actions.empty and len(gaps)==1


def test_ex_date_changed_after_entitlement_requires_review():
    p=records();p.loc[1,'ex_date']='2026-03-09'
    actions,gaps=versioned_cash_actions(p,'2026-03-21T19:00:00+08:00')
    assert actions.empty
    assert gaps.iloc[0].reason=='ex_date_not_known_before_entitlement'
