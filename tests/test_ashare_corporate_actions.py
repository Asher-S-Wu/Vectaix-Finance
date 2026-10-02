"""Offline contracts for selected-security, provider-schedule dividend evidence."""
import json

import pandas as pd
import pytest

from ashare_quant import corporate_actions as ca
from ashare_quant.collect import CollectionError, PermanentAPIError


def dividend(**changes):
    row = dict(ts_code='600000.SH', end_date='20231231', ann_date='20240320',
               div_proc='实施', stk_div=0., stk_bo_rate=0., stk_co_rate=0.,
               cash_div=.09, cash_div_tax=.10, record_date='20240620',
               ex_date='20240621', pay_date='20240621', div_listdate=None,
               imp_ann_date='20240614', base_date=None, base_share=1000.)
    return dict(row, **changes)


def normalize(*rows, **kwargs):
    return ca.normalize_dividends(pd.DataFrame(rows), **kwargs)


def test_cash_schedule_uses_final_publication_gross_and_explicit_research_basis():
    actions, rejected, summary = normalize(dividend())
    assert rejected.empty
    a = actions.iloc[0]
    assert a.security_id == '600000.SH'
    assert a.ann_date == '2024-06-14'
    assert a.source_ann_date == '20240320'
    assert a.available_from == '2024-06-15'
    assert a.cash_per_share_pre_tax == .10
    assert a.provider_cash_per_share_after_tax == .09
    assert a.share_multiplier == 1.
    assert a.verified and not a.issuer_verified
    assert a.evidence_basis == 'provider_final_schedule_simulated'
    assert a.payment_basis == 'gross_cash_with_20pct_conservative_tax_reserve'
    assert a.tax_reserve_rate == .20
    assert not a.investor_net_cash_verified and not a.point_in_time_complete
    assert summary['accepted_events'] == 1
    assert summary['actual_accounting_verified'] is False


@pytest.mark.parametrize(('change', 'reason'), [
    ({'imp_ann_date': None}, 'missing_implementation_publication'),
    ({'imp_ann_date': '20240620'}, 'publication_not_before_record'),
    ({'imp_ann_date': '20240622'}, 'publication_not_before_record'),
    ({'record_date': '20240621'}, 'invalid_event_dates'),
    ({'pay_date': None}, 'missing_payment_date'),
    ({'pay_date': '20240619'}, 'invalid_event_dates'),
    ({'ex_date': '20240230'}, 'invalid_ex_date'),
    ({'cash_div_tax': None}, 'invalid_gross_cash'),
    ({'cash_div_tax': -.1}, 'invalid_gross_cash'),
    ({'cash_div_tax': float('inf')}, 'invalid_gross_cash'),
    ({'cash_div': .20}, 'inconsistent_provider_after_tax_cash'),
    ({'stk_div': .1, 'stk_bo_rate': .1}, 'unsupported_stock_distribution'),
    ({'stk_div': .1}, 'inconsistent_stock_distribution'),
    ({'stk_div': None}, 'missing_stock_distribution_fields'),
    ({'rights_ratio': .1}, 'unsupported_rights_or_differential_event'),
    ({'differential_dividend': True}, 'unsupported_rights_or_differential_event'),
    ({'distribution_note': '回购专户不参与分红'}, 'unsupported_rights_or_differential_event'),
    ({'div_proc': '预案'}, 'not_implemented'),
])
def test_incomplete_unsafe_or_nonfinal_event_is_retained_but_not_accepted(change, reason):
    actions, rejected, _ = normalize(dividend(**change))
    assert actions.empty
    assert len(rejected) == 1
    assert reason in rejected.iloc[0].rejection_reasons
    assert not rejected.iloc[0].verified
    assert json.loads(rejected.iloc[0].raw_record_json)['ts_code'] == '600000.SH'


def test_null_provider_net_is_not_invented_and_cannot_replace_gross():
    actions, rejected, _ = normalize(dividend(cash_div=None))
    assert len(actions) == 1 and rejected.empty
    assert pd.isna(actions.iloc[0].provider_cash_per_share_after_tax)
    actions, rejected, _ = normalize(dividend(cash_div_tax=None, cash_div=.08))
    assert actions.empty and 'invalid_gross_cash' in rejected.iloc[0].rejection_reasons


def test_exact_alias_history_is_one_economic_event_with_all_sources_preserved():
    actions, rejected, summary = normalize(
        dividend(ts_code='000022.SZ'), dividend(ts_code='001872.SZ'),
        aliases={'000022.SZ': '001872.SZ'})
    assert len(actions) == 1 and rejected.empty
    a = actions.iloc[0]
    assert a.security_id == '001872.SZ'
    assert json.loads(a.source_ts_codes_json) == ['000022.SZ', '001872.SZ']
    assert len(json.loads(a.raw_records_json)) == 2
    assert summary['equivalent_duplicate_rows'] == 1


def test_conflicting_same_publication_alias_records_fail_closed():
    actions, rejected, _ = normalize(
        dividend(ts_code='000022.SZ'), dividend(ts_code='001872.SZ', cash_div_tax=.2),
        aliases={'000022.SZ': '001872.SZ'})
    assert actions.empty and len(rejected) == 2
    assert all('conflicting_final_versions' in x for x in rejected.rejection_reasons)


def test_explicitly_dated_revision_uses_latest_before_entitlement_and_retains_prior():
    actions, rejected, _ = normalize(dividend(imp_ann_date='20240612'),
                                     dividend(imp_ann_date='20240618', cash_div_tax=.2))
    assert len(actions) == 1
    assert actions.iloc[0].ann_date == '2024-06-18'
    assert actions.iloc[0].cash_per_share_pre_tax == .2
    assert len(rejected) == 1
    assert 'superseded_provider_version' in rejected.iloc[0].rejection_reasons


def test_asof_never_backfills_later_revision_or_announcement_date_only():
    rows = [dividend(imp_ann_date='20240612'),
            dividend(imp_ann_date='20240618', cash_div_tax=.2)]
    actions, rejected, _ = normalize(*rows, asof_date='2024-06-18')
    assert len(actions) == 1 and actions.iloc[0].cash_per_share_pre_tax == .1
    assert 'not_yet_visible' in rejected.iloc[0].rejection_reasons
    actions, _, _ = normalize(dividend(), asof_date='2024-06-14')
    assert actions.empty


def test_late_correction_or_unidentifiable_version_invalidates_whole_event():
    for changes in ({'imp_ann_date': '20240625'}, {'imp_ann_date': None}):
        actions, rejected, _ = normalize(dividend(), dividend(cash_div_tax=.2, **changes))
        assert actions.empty and len(rejected) == 2
        assert all('conflicting_final_versions' in x for x in rejected.rejection_reasons)


class DividendClient:
    """Transport substitute: no credentials, subprocess, sockets or real API."""
    def __init__(self, records=None, error=None, ignore_offset=False):
        self.records = records or {}
        self.error = error
        self.ignore_offset = ignore_offset
        self.calls = []

    def query(self, endpoint, params, fields):
        self.calls.append((endpoint, dict(params), fields))
        if self.error:
            raise self.error
        offset = 0 if self.ignore_offset else params['offset']
        rows = self.records.get(params['ts_code'], [])[offset:offset+params['limit']]
        return {'data': {'fields': fields.split(','),
                         'items': [[r.get(c) for c in fields.split(',')] for r in rows]}}


def collect(tmp_path, client, ids=('600000.SH',), **kwargs):
    return ca.collect_selected_dividends(tmp_path, ids, client=client,
                                        model_frozen=True, core_collection_stopped=True,
                                        **kwargs)


def test_collection_requires_frozen_selection_and_stopped_core_before_network(tmp_path):
    client = DividendClient()
    with pytest.raises(ValueError, match='frozen'):
        ca.collect_selected_dividends(tmp_path, ['600000.SH'], client=client,
                                     core_collection_stopped=True)
    with pytest.raises(ValueError, match='stopped'):
        ca.collect_selected_dividends(tmp_path, ['600000.SH'], client=client,
                                     model_frozen=True)
    assert not client.calls


def test_collection_is_narrow_cached_and_expands_only_selected_known_aliases(tmp_path):
    refs = tmp_path/'references'; refs.mkdir()
    pd.DataFrame([dict(o_code='000022.SZ', n_code='001872.SZ', verified=True),
                  dict(o_code='000043.SZ', n_code='001914.SZ', verified=True)]).to_parquet(refs/'verified_code_changes.parquet')
    pd.DataFrame([dict(o_code='839729.BJ', n_code='920729.BJ')]).to_parquet(refs/'bse_mapping.parquet')
    client = DividendClient({'000022.SZ': [dividend(ts_code='000022.SZ')],
                             '001872.SZ': [dividend(ts_code='001872.SZ')]})
    summary = collect(tmp_path, client, ['001872.SZ'])
    assert summary['selected_canonical_count'] == 1
    assert summary['queried_source_codes'] == ['000022.SZ', '001872.SZ']
    assert len(client.calls) == 2
    assert all(c[0] == 'dividend' and set(c[1]) == {'ts_code','limit','offset'} for c in client.calls)
    assert all(c[1]['limit'] == 2000 for c in client.calls)
    assert summary['collection_status'] == 'complete'
    assert summary['actual_accounting_verified'] is False
    assert len(pd.read_parquet(refs/'verified_corporate_actions.parquet')) == 1
    assert pd.read_parquet(refs/'unverified_corporate_actions.parquet').empty
    assert json.loads((refs/'corporate_actions_provenance.json').read_text())['requests_per_minute'] == 30
    collect(tmp_path, client, ['001872.SZ'])
    assert len(client.calls) == 2


def test_empty_or_permission_failure_saves_incomplete_evidence_without_aborting(tmp_path):
    for label, client in [('empty', DividendClient()),
                          ('denied', DividendClient(error=PermanentAPIError('dividend: api rejected')))]:
        root = tmp_path/label
        summary = collect(root, client)
        assert summary['collection_status'] == 'incomplete'
        assert summary['affected_security_ids'] == ['600000.SH']
        assert summary['accepted_events'] == 0
        assert pd.read_parquet(root/'references'/'verified_corporate_actions.parquet').empty
        assert (root/'references'/'corporate_actions_provenance.json').exists()


def test_cap_requires_terminal_page_and_repeated_page_is_incomplete(tmp_path):
    rows = [dividend(end_date=f'{2000+i//365:04d}{1+i%12:02d}01', ann_date=f'{1900+i:04d}0101') for i in range(2000)]
    # Collection audits raw-page completeness even when historical fields later fail normalization.
    client = DividendClient({'600000.SH': rows})
    summary = collect(tmp_path/'ok', client)
    assert [c[1]['offset'] for c in client.calls] == [0, 2000]
    assert summary['collection_status'] == 'complete'
    repeating = DividendClient({'600000.SH': rows}, ignore_offset=True)
    summary = collect(tmp_path/'overlap', repeating)
    assert summary['collection_status'] == 'incomplete'
    assert summary['partition_errors'][0]['error_type'] == 'CollectionError'


def test_scope_limit_and_unknown_non_share_are_rejected_without_queries(tmp_path):
    client = DividendClient()
    summary = collect(tmp_path, client, [f'60{i:04d}.SH' for i in range(701)])
    assert summary['collection_status'] == 'incomplete'
    assert summary['scope_blocker'] == 'selected_identity_limit'
    assert len(summary['affected_security_ids']) == 701
    with pytest.raises(ValueError, match='A-share'):
        collect(tmp_path, client, ['00001.HK'])
    assert not client.calls


def test_empty_input_produces_typed_replay_compatible_tables():
    actions, rejected, summary = ca.normalize_dividends(pd.DataFrame())
    assert actions.empty and rejected.empty
    assert {'security_id','ann_date','record_date','ex_date','pay_date','share_list_date',
            'cash_per_share_pre_tax','share_multiplier','verified','source_endpoint','payment_basis'} <= set(actions)
    assert summary['accepted_events'] == 0


def test_permanent_endpoint_rejection_stops_remaining_requests(tmp_path):
    client = DividendClient(error=PermanentAPIError('redacted permission error'))
    summary = collect(tmp_path, client, ['600000.SH', '600001.SH', '600002.SH'])
    assert len(client.calls) == 1
    assert summary['unattempted_source_codes'] == ['600001.SH', '600002.SH']
    assert summary['affected_security_ids'] == ['600000.SH', '600001.SH', '600002.SH']


def test_invalid_asof_is_rejected_before_collection(tmp_path):
    client = DividendClient()
    with pytest.raises(ValueError, match='date'):
        collect(tmp_path, client, asof_date='2024-14-01')
    assert not client.calls


def test_code_change_csv_fallback_expands_selected_alias_only(tmp_path):
    client = DividendClient()
    summary = collect(tmp_path, client, ['302132.SZ'])
    assert summary['queried_source_codes'] == ['300114.SZ', '302132.SZ']
    assert any(r.get('format') == 'csv' for r in summary['alias_references'])


def test_inconsistent_base_share_or_basis_date_is_unhandled():
    for changes in ({'base_share': -1}, {'base_share': 'bad'}, {'base_date': '20240624'}):
        actions, rejected, _ = normalize(dividend(**changes))
        assert actions.empty
        assert 'invalid_distribution_basis' in rejected.iloc[0].rejection_reasons


def test_extreme_publication_date_does_not_crash_normalization():
    actions, rejected, _ = normalize(dividend(imp_ann_date='99991231'))
    assert actions.empty and len(rejected) == 1
    assert 'invalid_imp_ann_date' in rejected.iloc[0].rejection_reasons


def test_missing_record_version_cannot_supersede_valid_final_schedule():
    actions, rejected, _ = normalize(dividend(), dividend(imp_ann_date='20240618', record_date=None))
    assert actions.empty and len(rejected) == 2


def test_alias_cycles_ambiguous_references_and_nonshares_fail_closed(tmp_path):
    with pytest.raises(ValueError, match='cyclic'):
        normalize(dividend(), aliases={'600000.SH': '600001.SH', '600001.SH': '600000.SH'})
    with pytest.raises(ValueError, match='non-A-share'):
        normalize(dividend(), aliases={'600000.SH': '00001.HK'})
    refs = tmp_path/'references'; refs.mkdir()
    pd.DataFrame([dict(o_code='600000.SH', n_code='600001.SH'),
                  dict(o_code='600000.SH', n_code='600002.SH')]).to_parquet(refs/'bse_mapping.parquet')
    client = DividendClient()
    with pytest.raises(ValueError, match='ambiguous'):
        collect(tmp_path, client)
    assert not client.calls


def test_all_null_scalars_and_date_types_are_handled():
    row = dividend(ann_date=pd.Timestamp('2024-03-20'), imp_ann_date=pd.Timestamp('2024-06-14'),
                   cash_div=pd.NA, div_listdate=pd.NaT, base_date=pd.NA)
    actions, rejected, _ = normalize(row)
    assert len(actions) == 1 and rejected.empty
    assert actions.iloc[0].ann_date == '2024-06-14'


def test_unknown_source_identity_is_never_used_as_selected_evidence(tmp_path):
    client = DividendClient({'600000.SH': [dividend(ts_code='600001.SH')]})
    summary = collect(tmp_path, client)
    assert summary['collection_status'] == 'incomplete'
    assert summary['accepted_events'] == 0
    assert summary['affected_security_ids'] == ['600000.SH']


def test_cache_corruption_remains_incomplete_without_network_retry(tmp_path):
    client = DividendClient({'600000.SH': [dividend()]})
    summary = collect(tmp_path, client)
    artifact = tmp_path/'raw'/summary['partitions'][0]['artifact']
    artifact.write_bytes(b'corrupt')
    result = collect(tmp_path, client)
    assert len(client.calls) == 1
    assert result['collection_status'] == 'incomplete'
    assert result['accepted_events'] == 0


def test_normalized_schedule_replays_gross_cash_reserve_without_share_creation():
    from ashare_quant.replay import run_replay
    days = ['2024-05-31', '2024-06-03', '2024-06-14', '2024-06-20', '2024-06-21', '2024-06-24']
    bars = pd.DataFrame([dict(date=d, security_id='600000.SH', raw_open=10., raw_close=10.,
                              high=10., low=10., adj_close=10., adj_factor=1.,
                              volume=1_000_000., amount=10_000_000.,
                              adv20_amount=10_000_000., up_limit=11., down_limit=9.,
                              quote_present=True) for d in days])
    bars.loc[bars.date.ge('2024-06-21'), ['raw_open','raw_close','high','low']] = 9.9
    bars.loc[bars.date.ge('2024-06-21'), 'adj_factor'] = 10/9.9
    forecasts = pd.DataFrame([dict(date='2024-05-31', security_id='600000.SH', horizon=20,
                                   score=1., score_status='ok')])
    actions, _, _ = normalize(dividend(pay_date='20240624'))
    result = run_replay(forecasts, bars, days, initial_cash=100_000, top_k=1, actions=actions)
    bought = result['trades'][0]['quantity']
    assert result['summary']['cash_dividends_paid'] == pytest.approx(bought*.1*.8)
    assert result['summary']['dividend_tax_reserve'] == pytest.approx(bought*.1*.2)
    assert not result['summary']['exact_investor_tax_known']
    assert all(p['quantity'] == bought for p in result['positions'])
    assert [c['date'] for c in result['cash'] if c['event'] == 'cash_dividend'] == ['2024-06-24']


def test_configured_selection_limit_never_silently_trims(tmp_path):
    client = DividendClient()
    summary = collect(tmp_path, client, ['600000.SH', '600001.SH'], max_selected_securities=1)
    assert summary['collection_status'] == 'incomplete'
    assert summary['scope_blocker'] == 'selected_identity_limit'
    assert summary['selected_canonical_count'] == 2
    assert not client.calls


def test_query_code_hard_limit_counts_aliases_and_stops_before_network(tmp_path):
    refs = tmp_path/'references'; refs.mkdir()
    pd.DataFrame([dict(o_code=f'60{i:04d}.SH', n_code='600999.SH') for i in range(1000) if i != 999] +
                 [dict(o_code='000001.SZ', n_code='600999.SH')]).to_parquet(refs/'bse_mapping.parquet')
    client = DividendClient()
    summary = collect(tmp_path, client, ['600999.SH'])
    assert summary['collection_status'] == 'incomplete'
    assert summary['scope_blocker'] == 'source_code_hard_limit'
    assert not client.calls


def test_changed_ex_date_same_fiscal_period_is_not_double_counted_as_two_events():
    actions, rejected, _ = normalize(dividend(), dividend(imp_ann_date='20240618',
                            record_date='20240621', ex_date='20240624', pay_date='20240624'))
    assert actions.empty and len(rejected) == 2
    assert all('ambiguous_period_schedule' in r for r in rejected.rejection_reasons)


def test_malformed_final_ex_date_does_not_hide_a_conflicting_period_revision():
    actions, rejected, _ = normalize(dividend(), dividend(imp_ann_date='20240618', ex_date=None))
    assert actions.empty and len(rejected) == 2
    assert all('ambiguous_period_schedule' in r for r in rejected.rejection_reasons)


def test_cash_only_with_red_share_listing_date_is_ambiguous():
    actions, rejected, _ = normalize(dividend(div_listdate='20240621'))
    assert actions.empty
    assert 'ambiguous_stock_distribution' in rejected.iloc[0].rejection_reasons


def test_saved_output_checksums_and_actual_attempt_scope_are_auditable(tmp_path):
    import hashlib
    client = DividendClient(error=PermanentAPIError('denied'))
    summary = collect(tmp_path, client, ['600000.SH', '600001.SH'])
    assert summary['requested_source_codes'] == ['600000.SH', '600001.SH']
    assert summary['queried_source_codes'] == ['600000.SH']
    for filename, digest in summary['output_sha256'].items():
        assert hashlib.sha256((tmp_path/'references'/filename).read_bytes()).hexdigest() == digest


def test_revision_cannot_retroactively_move_an_earlier_entitlement_date():
    actions, rejected, _ = normalize(dividend(record_date='20240617'),
                                     dividend(imp_ann_date='20240618'))
    assert actions.empty and len(rejected) == 2
    assert all('conflicting_final_versions' in r for r in rejected.rejection_reasons)


@pytest.mark.parametrize(('bonus', 'conversion'), [(None, None), (None, float('nan')),
                                                    (0., None), (None, 0.), (pd.NA, pd.NA)])
def test_explicit_zero_aggregate_allows_absent_stock_decomposition(bonus, conversion):
    actions, rejected, _ = normalize(dividend(stk_div=0., stk_bo_rate=bonus, stk_co_rate=conversion))
    assert len(actions) == 1 and rejected.empty
    assert actions.iloc[0].share_multiplier == 1.
    assert actions.iloc[0].cash_per_share_pre_tax == .1
    assert not actions.iloc[0].issuer_verified
    raw = json.loads(actions.iloc[0].raw_records_json)[0]
    assert raw['stk_div'] == 0.
    if pd.isna(bonus):
        assert raw['stk_bo_rate'] is None
    if pd.isna(conversion):
        assert raw['stk_co_rate'] is None


@pytest.mark.parametrize('field', ['stk_bo_rate', 'stk_co_rate'])
@pytest.mark.parametrize('component', [.1, -.1, float('inf'), -float('inf'), 'bad', True, False])
def test_explicit_zero_aggregate_rejects_contradictory_stock_component(field, component):
    row = dividend(stk_div=0., stk_bo_rate=None, stk_co_rate=None)
    row[field] = component
    actions, rejected, _ = normalize(row)
    assert actions.empty and len(rejected) == 1
    assert 'inconsistent_stock_distribution' in rejected.iloc[0].rejection_reasons


@pytest.mark.parametrize('aggregate', [None, float('nan'), float('inf'), -float('inf'), 'bad', True, False])
def test_zero_or_absent_components_never_substitute_for_missing_valid_aggregate(aggregate):
    actions, rejected, _ = normalize(dividend(stk_div=aggregate, stk_bo_rate=0., stk_co_rate=0.))
    assert actions.empty and len(rejected) == 1


def test_zero_aggregate_missing_and_explicit_zero_components_are_equivalent_aliases():
    actions, rejected, summary = normalize(
        dividend(ts_code='000022.SZ', stk_bo_rate=None, stk_co_rate=None),
        dividend(ts_code='001872.SZ'), aliases={'000022.SZ': '001872.SZ'})
    assert len(actions) == 1 and rejected.empty
    assert summary['equivalent_duplicate_rows'] == 1
    assert len(json.loads(actions.iloc[0].raw_records_json)) == 2
