import pandas as pd
import pytest

from hk_quant.distribution_versions import distributions_asof


def versions():
    return pd.DataFrame([
        dict(security_id='00405.HK', event_id='final', published_at='2026-03-11T21:19:00+08:00',
             ex_date='2026-04-07', cash_per_unit_decimal='0.0214', conditional_cash_amount=True,
             is_latest_discovered_version=False, version_status='superseded_preserved'),
        dict(security_id='00405.HK', event_id='final', published_at='2026-03-12T12:01:00+08:00',
             ex_date='2026-04-02', cash_per_unit_decimal='0.0214', conditional_cash_amount=True,
             is_latest_discovered_version=False, version_status='superseded_preserved'),
        dict(security_id='00405.HK', event_id='final', published_at='2026-04-09T19:44:00+08:00',
             ex_date='2026-04-02', cash_per_unit_decimal='0.0211', conditional_cash_amount=False,
             is_latest_discovered_version=True, version_status='latest_discovered'),
    ])


def test_publication_time_and_revisions_not_retroactive():
    frame = versions()
    assert distributions_asof(frame, '2026-03-11T19:00:00+08:00').empty
    first = distributions_asof(frame, '2026-03-12T10:00:00+08:00').iloc[0]
    assert first.ex_date == '2026-04-07'
    old = distributions_asof(frame, '2026-04-09T19:00:00+08:00').iloc[0]
    assert old.ex_date == '2026-04-02'
    assert old.cash_per_unit_decimal == '0.0214'
    assert old.conditional_cash_amount
    new = distributions_asof(frame, '2026-04-09T19:44:00+08:00').iloc[0]
    assert new.cash_per_unit_decimal == '0.0211'
    assert not new.conditional_cash_amount


def test_snapshot_metadata_cannot_leak_future_revision():
    actual = distributions_asof(versions(), '2026-04-02T19:00:00+08:00')
    assert 'is_latest_discovered_version' not in actual
    assert 'version_status' not in actual
    assert actual.iloc[0].cash_per_unit_decimal == '0.0214'


def test_unknown_or_naive_publication_is_rejected():
    for invalid in [None, '2026-03-11']:
        frame = versions()
        frame.loc[0, 'published_at'] = invalid
        with pytest.raises(ValueError, match='公开时间'):
            distributions_asof(frame, '2026-04-02T19:00:00+08:00')


def test_ambiguous_version_and_naive_cutoff_rejected():
    with pytest.raises(ValueError, match='重复'):
        distributions_asof(pd.concat([versions(), versions().iloc[[0]]]), '2026-04-02T19:00:00+08:00')
    with pytest.raises(ValueError, match='时区'):
        distributions_asof(versions(), '2026-04-02')


def test_future_rows_do_not_change_historical_snapshot():
    old = versions().iloc[:2].copy()
    cutoff = '2026-04-02T19:00:00+08:00'
    pd.testing.assert_frame_equal(distributions_asof(old, cutoff), distributions_asof(versions(), cutoff))
