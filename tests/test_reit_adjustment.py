from decimal import Decimal

import pandas as pd
import pytest

from hk_quant.reit_adjustment import adjusted_history_asof


CALENDAR = pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-07', '2026-04-08'])


def bars():
    return pd.DataFrame({'security_id': ['00405.HK'] * 4, 'date': CALENDAR,
                         'raw_close': [10., 9., 9.2, 9.4]})


def event(**changes):
    row = dict(security_id='00405.HK', event_id='final',
               published_at='2026-03-20T19:00:00+08:00', ex_date='2026-04-02',
               cash_per_unit_decimal='1', cash_currency='HKD',
               conditional_cash_amount=False, cash_amount_status='declared_payment_currency_amount',
               event_type='cash_distribution', announcement_status='New announcement')
    row.update(changes)
    return row


def run(events, raw=None, cutoff='2026-04-08T19:00:00+08:00'):
    return adjusted_history_asof(bars() if raw is None else raw, CALENDAR,
                                pd.DataFrame(events), cutoff, 'HKD')


def test_post_ex_revision_does_not_change_earlier_snapshot_or_raw():
    raw = bars()
    original = raw.copy(deep=True)
    first = event(is_latest_discovered_version=False)
    revised = event(published_at='2026-04-07T20:00:00+08:00', cash_per_unit_decimal='2',
                    is_latest_discovered_version=True)
    old = run([first, revised], raw, '2026-04-07T19:00:00+08:00')
    later = run([first, revised], raw)
    assert old.history.adjusted_close.tolist() == [9., 9., 9.2]
    assert later.history.adjusted_close.tolist() == [8., 9., 9.2, 9.4]
    assert old.history.adjusted_close.iloc[0] == 9.
    pd.testing.assert_frame_equal(raw, original)
    assert 'is_latest_discovered_version' not in old.selected_events
    assert not later.coverage_complete
    assert later.return_basis == 'corporate_action_adjusted_price_return_not_cash_wealth'


def test_same_day_cash_is_summed_before_factor():
    result = run([event(cash_per_unit_decimal='1'), event(event_id='special', cash_per_unit_decimal='2')])
    assert result.history.adjusted_close.iloc[0] == 7.
    assert result.factors.iloc[0].cash_total_decimal == '3'
    assert Decimal(result.factors.iloc[0].factor_decimal) == Decimal('0.7')


def test_rights_adjustment_requires_public_terms_and_ex_reference():
    raw=bars();raw['prevClose']=[9.9,9.333,9.,9.2]
    rights=event(event_type='rights_issue',rights_new_units='1',rights_old_units='5',
        rights_subscription_price='6',rights_currency='HKD',rights_terms_public=True,
        conditional_cash_amount=True)
    result=run([rights],raw)
    assert result.status=='resolved'
    assert abs(result.history.adjusted_close.iloc[0]-28/3)<1e-10
    assert result.factors.iloc[0].adjustment_kind=='rights_issue'
    bad=raw.copy();bad.loc[1,'prevClose']=10.
    assert run([rights],bad).status=='unresolved'
    unknown=raw.drop(columns='prevClose')
    assert run([rights],unknown).status=='unresolved'


def test_missing_immediately_prior_session_cannot_use_older_close():
    result = run([event(ex_date='2026-04-07')], bars().drop(index=1))
    assert result.status == 'unresolved'
    assert result.gaps.reason.tolist() == ['missing_prior_session_close']
    assert pd.isna(result.history.adjusted_close.iloc[0])
    assert result.history.date.tolist() == list(CALENDAR[[0, 2, 3]])


@pytest.mark.parametrize('change,reason', [
    ({'cash_currency': 'CNY'}, 'cash_currency_mismatch'),
    ({'conditional_cash_amount': True}, 'cash_amount_not_final'),
    ({'cash_amount_status': 'conflicted'}, 'cash_amount_not_final'),
    ({'cash_per_unit_decimal': '10'}, 'nonpositive_theoretical_factor'),
    ({'cash_per_unit_decimal': None}, 'invalid_cash_amount'),
])
def test_unresolved_cash_never_silently_uses_factor_one(change, reason):
    result = run([event(**change)])
    assert result.status == 'unresolved'
    assert reason in result.gaps.reason.tolist()
    assert pd.isna(result.history.adjusted_close.iloc[0])
    assert pd.isna(result.history.adjustment_factor.iloc[0])


def test_pre_ex_conditional_does_not_invalidate_history():
    result = run([event(conditional_cash_amount=True)], cutoff='2026-04-01T19:00:00+08:00')
    assert result.status == 'resolved'
    assert result.history.adjusted_close.tolist() == [10.]


def test_final_publication_resolves_only_later_cutoff():
    events = [event(conditional_cash_amount=True),
              event(published_at='2026-04-07T20:00:00+08:00', cash_per_unit_decimal='0.5')]
    old = run(events, cutoff='2026-04-07T19:00:00+08:00')
    assert old.status == 'unresolved'
    assert run(events).history.adjusted_close.iloc[0] == 9.5
    assert pd.isna(old.history.adjusted_close.iloc[0])


def test_cutoff_requires_timezone_and_excludes_unfinished_daily_bar():
    with pytest.raises(ValueError, match='时区'):
        run([event()], cutoff='2026-04-08')
    result = run([event()], cutoff='2026-04-02T10:00:00+08:00')
    assert result.history.date.tolist() == [CALENDAR[0]]
    assert result.history.adjusted_close.tolist() == [9.]


def test_unknown_conditional_flag_is_unresolved():
    result = run([event(conditional_cash_amount=pd.NA)])
    assert result.status == 'unresolved'
    assert result.gaps.reason.tolist() == ['cash_amount_not_final']


def test_nullable_prior_close_is_not_replaced():
    raw = bars()
    raw['raw_close'] = raw.raw_close.astype('Float64')
    raw.loc[0, 'raw_close'] = pd.NA
    result = run([event()], raw)
    assert result.gaps.reason.tolist() == ['missing_prior_session_close']
    assert pd.isna(result.history.raw_close.iloc[0])


def test_unsupported_ex_dated_rights_and_partial_cash_group_are_not_ignored():
    result = run([event(), event(event_id='rights', event_type='rights_issue')])
    assert result.status == 'unresolved'
    assert result.gaps.reason.tolist() == ['missing_public_rights_terms']
    assert result.factors.empty
    assert pd.isna(result.history.adjusted_close.iloc[0])


def test_unknown_ex_date_invalidates_history_without_guessing():
    result = run([event(ex_date=None)])
    assert result.gaps.reason.tolist() == ['unknown_ex_date']
    assert result.history.adjusted_close.isna().all()
