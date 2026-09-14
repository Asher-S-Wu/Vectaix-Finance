import json

import pandas as pd
import pytest

from tmp.import_reviewed_cash_actions import import_reviewed_actions, merge_reviewed_actions


def reviewed():
    return pd.DataFrame([dict(security_id='00405.HK', action_type='cash_dividend',
        event_id='00405.HK:Final:2025', effective_date='2026-04-02', payment_date='2026-05-22',
        cash_per_share=.0211, cash_per_share_decimal='0.0211', cash_currency='HKD', verified=True,
        source_url='https://issuer.test/final.pdf', source_document='final.pdf',
        amount_known_at='2026-04-09T19:44:00+08:00', entitlement_known_at='2026-03-12T12:01:00+08:00',
        payment_basis='issuer_final_schedule_simulated', identity_issue_id=4690)])


def candidate(security_id='00778.HK'):
    return pd.DataFrame([dict(security_id=security_id, action_type='cash_dividend',
        effective_date='2026-04-02', payment_date='2026-05-22', cash_per_share=.0211,
        cash_per_share_hkd=None, cash_currency='HKD', verified=False,
        source_url='https://archive.test/event', source_import_id=123,
        announced_at=None, formal_replay_eligible=False)])


def test_append_preserves_unknown_candidate_time_and_is_idempotent():
    actual, audit = merge_reviewed_actions(candidate(), reviewed())
    assert audit['appended'] == 1 and audit['replaced'] == 0
    assert len(actual) == 2
    old = actual.loc[actual.security_id.eq('00778.HK')].iloc[0]
    new = actual.loc[actual.security_id.eq('00405.HK')].iloc[0]
    assert pd.isna(old.amount_known_at) and not old.verified
    assert new.amount_known_at == pd.Timestamp('2026-04-09T11:44:00Z')
    assert new.entitlement_known_at == pd.Timestamp('2026-03-12T04:01:00Z')
    assert new.payment_date == pd.Timestamp('2026-05-22')
    assert new.verified and not new.formal_replay_eligible
    repeated, repeat_audit = merge_reviewed_actions(actual, reviewed())
    pd.testing.assert_frame_equal(actual, repeated)
    assert repeat_audit['appended'] == 0


def test_unique_candidate_is_replaced_and_archive_source_is_retained():
    actual, audit = merge_reviewed_actions(candidate('00405.HK'), reviewed())
    assert len(actual) == 1 and audit['replaced'] == 1
    row = actual.iloc[0]
    assert row.source_url == 'https://issuer.test/final.pdf'
    assert row.provider_source_url == 'https://archive.test/event'
    assert row.source_import_id == 123
    assert row.verified


@pytest.mark.parametrize('problem', ['duplicate', 'amount', 'currency'])
def test_conflicting_candidate_prevents_import_without_overwriting_files(tmp_path, problem):
    existing = candidate('00405.HK')
    if problem == 'duplicate':
        existing = pd.concat([existing, existing], ignore_index=True)
    elif problem == 'amount':
        existing.loc[0, 'cash_per_share'] = .0214
    else:
        existing.loc[0, 'cash_currency'] = 'CNY'
    actions_path = tmp_path / 'corporate_actions.parquet'
    source_path = tmp_path / 'reviewed.parquet'
    audit_path = tmp_path / 'partial_replay_inputs_audit.json'
    existing.to_parquet(actions_path, index=False)
    reviewed().to_parquet(source_path, index=False)
    audit_path.write_text(json.dumps({'historical_lots_rows': 99, 'corporate_actions_rows': len(existing)}))
    before = actions_path.read_bytes()
    before_audit = audit_path.read_bytes()
    with pytest.raises(ValueError, match='候选重复|金额或币种冲突'):
        import_reviewed_actions(actions_path, source_path, audit_path)
    assert actions_path.read_bytes() == before
    assert audit_path.read_bytes() == before_audit


def test_file_import_updates_cash_counts_only(tmp_path):
    actions = tmp_path / 'corporate_actions.parquet'
    source = tmp_path / 'reviewed.parquet'
    audit = tmp_path / 'partial_replay_inputs_audit.json'
    candidate().to_parquet(actions, index=False)
    reviewed().to_parquet(source, index=False)
    audit.write_text(json.dumps(dict(historical_lots_rows=999, historical_lots_verified_rows=888,
        formal_replay_eligible=False, coverage_complete=False)))
    import_reviewed_actions(actions, source, audit)
    counts = json.loads(audit.read_text())
    assert counts['corporate_actions_rows'] == 2
    assert counts['corporate_actions_securities'] == 2
    assert counts['corporate_actions_verified_rows'] == 1
    assert counts['historical_lots_rows'] == 999 and counts['historical_lots_verified_rows'] == 888
    assert not counts['formal_replay_eligible'] and not counts['coverage_complete']
