"""Cash-account invariants for monthly, next-session A-share execution."""
import pandas as pd
import pytest

from ashare_quant.replay import run_replay


def inputs(dates=('2024-01-30', '2024-01-31', '2024-02-01', '2024-02-02'),
           securities=('600000.SH',), cash=100_000):
    bars = pd.DataFrame([dict(date=d, security_id=s, raw_open=10., raw_close=10.,
                             high=10., low=10., adj_close=10., adj_factor=1.,
                             volume=1_000_000., amount=10_000_000.,
                             adv20_amount=10_000_000., up_limit=11., down_limit=9.,
                             quote_present=True) for d in dates for s in securities])
    forecasts = pd.DataFrame([dict(date='2024-01-31', security_id=s, horizon=20,
                                   score=10-i, score_status='ok')
                              for i,s in enumerate(securities)])
    return forecasts, bars, list(dates), dict(initial_cash=cash, top_k=1)


def test_month_end_signal_fills_next_session_and_reconciles_cash():
    f,b,c,k = inputs()
    r = run_replay(f,b,c,**k)
    assert len(r['trades']) == 1
    t = r['trades'][0]
    assert t['signal_date'] == '2024-01-31' and t['date'] == '2024-02-01'
    assert t['side'] == 'buy' and t['quantity'] == 9900
    assert t['price'] == 10.
    last = r['equity'][-1]
    assert last['cash'] == pytest.approx(100_000 - t['notional'] - t['fee'])
    assert last['equity'] == pytest.approx(100_000 - t['fee'])
    assert r['summary']['execution_valid'] is True


def test_non_month_end_and_unavailable_scores_never_trade():
    f,b,c,k = inputs()
    f.loc[0, 'date'] = '2024-01-30'
    assert run_replay(f,b,c,**k)['trades'] == []
    f.loc[0, 'date'] = '2024-01-31'; f.loc[0, 'score_status'] = 'unavailable'
    assert run_replay(f,b,c,**k)['trades'] == []


def test_independent_interval_failure_does_not_block_valid_score():
    f,b,c,k = inputs()
    f['status'] = 'invalid_interval'; f['interval_status'] = 'unavailable'
    assert len(run_replay(f,b,c,**k)['trades']) == 1


def test_participation_is_bounded_by_both_signal_adv_and_execution_amount():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-01-31'), 'adv20_amount'] = 1_000_000.
    b.loc[b.date.eq('2024-02-01'), 'amount'] = 500_000.
    b.loc[b.date.eq('2024-02-01'), 'adv20_amount'] = 100_000_000.
    r = run_replay(f,b,c,**k)
    assert r['trades'][0]['notional'] == 5_000.
    assert r['trades'][0]['participation_limit_notional'] == 5_000.
    assert r['trades'][0]['signal_adv20_amount'] == 1_000_000.


def test_missing_adv_does_not_use_future_adv_or_fill():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-01-31'), 'adv20_amount'] = float('nan')
    r = run_replay(f,b,c,**k)
    assert r['trades'] == []
    assert any(a['reason'] == 'missing_signal_adv20' for a in r['audit'])


def test_unfilled_buy_is_not_retried_using_stale_signal():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-01'), 'quote_present'] = False
    r = run_replay(f,b,c,**k)
    assert not r['trades']
    assert any(a['reason'] == 'missing_quote' for a in r['audit'])


def test_delayed_lower_limit_sell_retries_and_cannot_create_short_position():
    dates = ('2024-01-31','2024-02-01','2024-02-29','2024-03-01','2024-03-04')
    f,b,c,k = inputs(dates, ('600000.SH','000001.SZ'))
    f = pd.DataFrame([dict(date=d, security_id=s, horizon=20, score=score,
                           score_status='ok') for d, pairs in [
        ('2024-01-31',[('600000.SH',2),('000001.SZ',1)]),
        ('2024-02-29',[('600000.SH',1),('000001.SZ',2)])] for s,score in pairs])
    b.loc[(b.date.eq('2024-03-01')) & b.security_id.eq('600000.SH'),
          ['raw_open','raw_close','high','low']] = 9.
    r = run_replay(f,b,c,**k)
    sells = [t for t in r['trades'] if t['side'] == 'sell']
    assert len(sells) == 1 and sells[0]['date'] == '2024-03-04'
    assert any(a['reason'] == 'at_lower_limit' for a in r['audit'])
    assert all(p['quantity'] >= 0 for p in r['positions'])
    assert all(e['cash'] >= 0 for e in r['equity'])


def test_missing_authoritative_limits_is_audited_not_assumed_st_rule():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-01'), 'up_limit'] = float('nan')
    r = run_replay(f,b,c,**k)
    assert not r['trades']
    assert any(a['reason'] == 'missing_price_limit' for a in r['audit'])
    assert r['summary']['execution_data_complete'] is False


def test_adjustment_jump_never_changes_actual_shares_and_invalidates_accounting():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-02'), ['raw_open','raw_close','high','low']] = 5.
    b.loc[b.date.eq('2024-02-02'), 'adj_factor'] = 2.
    r = run_replay(f,b,c,**k)
    held = [p for p in r['positions'] if p['date']=='2024-02-02'][0]
    assert held['quantity'] == r['trades'][0]['quantity']
    assert held['valuation_status'] == 'unresolved_corporate_action'
    assert r['equity'][-1]['equity'] is None
    assert r['equity'][-1]['reference_equity'] > r['equity'][-1]['raw_mark_equity']
    assert r['summary']['valuation_valid'] is False
    assert r['summary']['total_return'] is None


def test_action_jump_before_buy_prevents_opening():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-01'), 'adj_factor'] = 1.01
    r = run_replay(f,b,c,**k)
    assert not r['trades']
    assert any(a['reason']=='unresolved_corporate_action' for a in r['audit'])


def test_final_partial_month_is_not_assumed_month_end():
    f,b,c,k = inputs(('2024-01-29','2024-01-30'))
    f['date'] = '2024-01-29'
    assert not run_replay(f,b,c,**k)['trades']


def test_default_selects_thirty_with_tie_break_stable_and_no_lookahead():
    securities = tuple(f'{i:06d}.SZ' for i in range(1, 33))
    f,b,c,k = inputs(securities=securities, cash=10_000_000)
    f['score'] = 1.; f['fwd_return'] = list(range(32))
    r = run_replay(f.sample(frac=1,random_state=2),b,c,initial_cash=10_000_000)
    assert len(r['trades']) == 30
    assert {t['security_id'] for t in r['trades']} == set(securities[:30])


def test_duplicate_price_identity_is_rejected():
    f,b,c,k = inputs()
    with pytest.raises(ValueError, match='duplicate'):
        run_replay(f,pd.concat([b,b.iloc[:1]]),c,**k)


def test_execution_validity_is_not_claimed_with_missing_execution_evidence():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-01'), 'up_limit'] = float('nan')
    assert run_replay(f,b,c,**k)['summary']['execution_valid'] is False


def test_reporting_start_retains_leading_month_end_signal_without_prior_equity():
    f,b,c,k = inputs()
    r = run_replay(f,b,c,start_date='2024-02-01',**k)
    assert r['trades'][0]['signal_date'] == '2024-01-31'
    assert [e['date'] for e in r['equity']] == ['2024-02-01','2024-02-02']


def test_stale_held_quote_is_flagged_without_fabricated_account_equity():
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-02'), 'quote_present'] = False
    r = run_replay(f,b,c,**k)
    assert r['equity'][-1]['equity'] is None
    assert r['positions'][-1]['valuation_status'] == 'stale_or_missing_quote'
    assert r['summary']['valuation_valid'] is False


def dividend_action(**changes):
    action = dict(security_id='600000.SH', ann_date='2024-01-30',
                  record_date='2024-02-01', ex_date='2024-02-02',
                  pay_date='2024-02-05', cash_per_share_pre_tax=.5,
                  share_multiplier=1., source_url='https://issuer.example/dividend',
                  verified=True, payment_basis='issuer_final_schedule')
    action.update(changes)
    return action


def action_inputs():
    f,b,c,k = inputs(('2024-01-31','2024-02-01','2024-02-02','2024-02-05'))
    b.loc[b.date.ge('2024-02-02'), ['raw_open','raw_close','high','low']] = 9.5
    b.loc[b.date.ge('2024-02-02'), 'adj_factor'] = 10 / 9.5
    return f,b,c,k


def test_verified_cash_dividend_uses_record_shares_and_pay_date_without_split():
    f,b,c,k = action_inputs()
    r = run_replay(f,b,c,actions=pd.DataFrame([dividend_action()]),**k)
    count = r['trades'][0]['quantity']
    assert all(p['quantity'] == count for p in r['positions'])
    daily = {d['date']:d for d in r['equity']}
    assert daily['2024-02-02']['receivables'] == pytest.approx(count*.4)
    assert daily['2024-02-05']['receivables'] == 0
    assert daily['2024-02-05']['cash'] - daily['2024-02-02']['cash'] == pytest.approx(count*.4)
    assert r['summary']['valuation_valid'] is True
    assert r['summary']['dividend_tax_policy'] == 'conservative_20_percent_reserve'
    assert r['summary']['dividend_tax_reserve'] == pytest.approx(count*.1)
    assert any(e['event_type']=='entitlement' and e['entitled_quantity']==count for e in r['events'])


def test_unverified_or_economically_inconsistent_event_never_clears_factor_guard():
    f,b,c,k = action_inputs()
    for action in (dividend_action(verified=False),
                   dividend_action(cash_per_share_pre_tax=.1),
                   dividend_action(record_date=None)):
        r = run_replay(f,b,c,actions=[action],**k)
        assert r['summary']['valuation_valid'] is False
        assert r['equity'][-1]['equity'] is None


def test_unknown_payment_date_preserves_receivable_and_flags_cash_timing():
    f,b,c,k = action_inputs()
    r = run_replay(f,b,c,actions=[dividend_action(pay_date=None)],**k)
    assert r['equity'][-1]['receivables'] > 0
    assert r['summary']['valuation_valid'] is True
    assert r['summary']['cash_timing_complete'] is False
    assert not any(e['event_type']=='cash_payment' for e in r['events'])


def test_verified_bonus_adds_only_entitled_shares_and_respects_listing_date():
    f,b,c,k = action_inputs()
    b.loc[b.date.ge('2024-02-02'), ['raw_open','raw_close','high','low']] = 10/1.1
    b.loc[b.date.ge('2024-02-02'), 'adj_factor'] = 1.1
    action = dividend_action(cash_per_share_pre_tax=0., share_multiplier=1.1,
                             share_list_date='2024-02-05', pay_date=None)
    r = run_replay(f,b,c,actions=[action],**k)
    count = r['trades'][0]['quantity']
    positions = {p['date']:p for p in r['positions']}
    assert positions['2024-02-02']['quantity'] == round(count*1.1)
    assert positions['2024-02-02']['sellable_quantity'] == count
    assert positions['2024-02-05']['sellable_quantity'] == round(count*1.1)
    assert r['summary']['valuation_valid'] is True


def test_record_date_before_acquisition_has_no_dividend_entitlement():
    f,b,c,k = action_inputs()
    b.loc[b.date.ge('2024-02-01'), ['raw_open','raw_close','high','low']] = 9.5
    b.loc[b.date.ge('2024-02-01'), 'adj_factor'] = 10/9.5
    action = dividend_action(record_date='2024-01-31', ex_date='2024-02-01')
    r = run_replay(f,b,c,actions=[action],**k)
    assert r['summary']['cash_dividends_paid'] == 0
    assert not any(e.get('net_amount',0)>0 for e in r['events'])


def test_unheld_unselected_adjustment_does_not_invalidate_verified_account():
    f,b,c,k = inputs(securities=('600000.SH','000001.SZ'))
    b.loc[b.date.eq('2024-02-02') & b.security_id.eq('000001.SZ'),'adj_factor'] = 2.
    r = run_replay(f,b,c,**k)
    assert r['summary']['execution_valid'] is True
    assert r['summary']['valuation_valid'] is True
    assert '000001.SZ' in r['summary']['unresolved_corporate_actions']


def test_nan_source_is_not_corporate_action_evidence():
    f,b,c,k = action_inputs()
    r = run_replay(f,b,c,actions=[dividend_action(source_url=float('nan'))],**k)
    assert r['summary']['valuation_valid'] is False


def test_reference_replay_is_labeled_fractional_research_and_fee_stress_monotone():
    from ashare_quant.replay import run_reference_replay
    f,b,c,k = inputs(cash=100_000)
    low = run_reference_replay(f,b,c,fee=.001,initial_cash=100_000,top_k=1)
    high = run_reference_replay(f,b,c,fee=.005,initial_cash=100_000,top_k=1)
    assert low['summary']['execution_validated'] is False
    assert low['summary']['methodology'] == 'research_only_adjusted_unit_monthly_replay'
    assert low['summary']['ending_equity'] > high['summary']['ending_equity']
    assert low['trades'][0]['date'] == '2024-02-01'
    assert low['trades'][0]['units'] % 1 != 0


def test_reference_uses_side_specific_limit_not_blanket_tradeability():
    from ashare_quant.replay import run_reference_replay
    f,b,c,k = inputs(('2024-01-31','2024-02-01','2024-02-29','2024-03-01'))
    f = pd.concat([f,f.assign(date='2024-02-29')])
    b.loc[b.date.eq('2024-03-01'), ['raw_open','raw_close','high','low']] = 11.
    b.loc[b.date.eq('2024-03-01'), 'adj_close'] = 11.
    b[['amount','adv20_amount']] = 20_000_000.
    r = run_reference_replay(f,b,c,top_k=1,initial_cash=100_000)
    march = [t for t in r['trades'] if t['date']=='2024-03-01']
    assert [t['side'] for t in march] == ['sell']
    assert any(a['reason']=='at_upper_limit' for a in r['audit'])


def test_reference_missing_terminal_mark_remains_incomplete():
    from ashare_quant.replay import run_reference_replay
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-02-02'), ['raw_close','adj_close']] = float('nan')
    r = run_reference_replay(f,b,c,top_k=1)
    assert r['summary']['status'] == 'incomplete_valuation'
    assert r['summary']['ending_equity'] is None
    assert r['summary']['total_return'] is None


def test_reference_capacity_uses_signal_adv_and_execution_amount():
    from ashare_quant.replay import run_reference_replay
    f,b,c,k = inputs()
    b.loc[b.date.eq('2024-01-31'), 'adv20_amount'] = 1_000_000.
    b.loc[b.date.eq('2024-02-01'), 'amount'] = 500_000.
    r = run_reference_replay(f,b,c,top_k=1)
    assert r['trades'][0]['notional'] == 5_000.


def test_requested_reference_end_is_not_silently_shortened_to_last_quote():
    from ashare_quant.replay import run_reference_replay
    f,b,c,k = inputs()
    b = b.loc[~b.date.eq('2024-02-02')]
    r = run_reference_replay(f,b,c,end_date='2024-02-02',top_k=1)
    assert r['equity'][-1]['date'] == '2024-02-02'
    assert r['summary']['ending_equity'] is None


def test_reference_default_grid_begins_first_execution_not_signal():
    from ashare_quant.replay import run_reference_replay
    f,b,c,k = inputs()
    r = run_reference_replay(f,b,c,top_k=1)
    assert r['equity'][0]['date'] == '2024-02-01'


def test_requested_actual_end_is_not_silently_shortened_to_last_quote():
    f,b,c,k = inputs()
    b = b.loc[~b.date.eq('2024-02-02')]
    r = run_replay(f,b,c,end_date='2024-02-02',**k)
    assert r['equity'][-1]['date'] == '2024-02-02'
    assert r['summary']['ending_equity'] is None


def test_corporate_action_events_preserve_verifiable_source():
    f,b,c,k = action_inputs()
    action = dividend_action()
    r = run_replay(f,b,c,actions=[action],**k)
    event = next(e for e in r['events'] if e['event_type']=='ex_entitlement')
    assert event['source_url'] == action['source_url']
    assert event['payment_basis'] == action['payment_basis']


def test_old_unheld_adjustment_does_not_block_later_new_raw_share_purchase():
    dates = ('2024-01-02','2024-01-03','2024-01-31','2024-02-01','2024-02-29','2024-03-01')
    f,b,c,k = inputs(dates, ('600000.SH','000001.SZ'))
    f = pd.DataFrame([dict(date=day, security_id=sid, horizon=20, score=score,
                           score_status='ok') for day, pairs in [
        ('2024-01-31',[('600000.SH',2),('000001.SZ',1)]),
        ('2024-02-29',[('600000.SH',1),('000001.SZ',2)])] for sid,score in pairs])
    b.loc[b.security_id.eq('000001.SZ') & b.date.ge('2024-01-03'),'adj_factor'] = 1.05
    r = run_replay(f,b,c,start_date='2024-01-02',**k)
    buys = [t for t in r['trades'] if t['side']=='buy' and t['security_id']=='000001.SZ']
    assert len(buys) == 1 and buys[0]['date']=='2024-03-01'
    assert r['summary']['valuation_valid'] is True
    assert any(e['event_type']=='unresolved_unheld_action' and e['date']=='2024-01-03'
               for e in r['events'])


@pytest.mark.parametrize('source', [None, '', ' ', '\t\n', float('nan'), 'NaN', 'nan', 'None', 'null', '<NA>'])
def test_blank_action_sources_never_resolve_owned_entitlement(source):
    f,b,c,k = action_inputs()
    r = run_replay(f,b,c,actions=[dividend_action(source_url=source)],**k)
    assert r['summary']['valuation_valid'] is False
    assert r['positions'][-1]['valuation_status']=='unresolved_corporate_action'


def test_held_reference_continuity_break_never_creates_wealth_or_sale_proceeds():
    from ashare_quant.replay import run_reference_replay
    dates = ('2024-01-31','2024-02-01','2024-02-02','2024-02-29','2024-03-01')
    f,b,c,k = inputs(dates)
    f = pd.concat([f,f.assign(date='2024-02-29')])
    b['reference_continuity_break'] = b.date.eq('2024-02-02')
    b.loc[b.date.ge('2024-02-02'),'adj_close'] = 1000.
    original = b.copy(deep=True)
    r = run_reference_replay(f,b,c,initial_cash=100_000,top_k=1)
    assert [t['side'] for t in r['trades']] == ['buy']
    assert r['equity'][-1]['cash'] == pytest.approx(r['trades'][0]['cash_after'])
    assert all(e['equity'] is None for e in r['equity'] if e['date']>='2024-02-02')
    assert r['positions'][-1]['valuation_status']=='unverified_reference_continuity'
    assert r['summary']['ending_equity'] is None
    assert r['summary']['ending_reference_equity'] is None
    assert r['summary']['total_return'] is None
    assert r['summary']['cagr'] is None
    assert r['summary']['unresolved_reference_continuity']==['600000.SH']
    assert any(a['reason']=='unverified_reference_continuity' for a in r['audit'])
    pd.testing.assert_frame_equal(b,original)


def test_reference_continuity_break_blocks_sale_before_same_day_rebalance():
    from ashare_quant.replay import run_reference_replay
    dates = ('2024-01-31','2024-02-01','2024-02-29','2024-03-01')
    f,b,c,k = inputs(dates)
    f = pd.concat([f,f.assign(date='2024-02-29')])
    b['reference_continuity_break'] = b.date.eq('2024-03-01')
    b.loc[b.date.eq('2024-03-01'),'adj_close'] = 1000.
    r = run_reference_replay(f,b,c,initial_cash=100_000,top_k=1)
    assert not [t for t in r['trades'] if t['date']=='2024-03-01']
    assert r['summary']['ending_equity'] is None


def test_selected_break_invalidates_performance_even_if_later_clean_entry_occurs():
    from ashare_quant.replay import run_reference_replay
    dates = ('2024-01-31','2024-02-01','2024-02-29','2024-03-01')
    f,b,c,k = inputs(dates)
    f = pd.concat([f,f.assign(date='2024-02-29')])
    b['reference_continuity_break'] = b.date.eq('2024-02-01')
    b.loc[b.date.ge('2024-02-01'),'adj_close'] = 1000.
    r = run_reference_replay(f,b,c,initial_cash=100_000,top_k=1)
    assert [t['date'] for t in r['trades']] == ['2024-03-01']
    assert r['summary']['ending_equity'] is None
    assert r['summary']['total_return'] is None
    assert r['summary']['cagr'] is None
    assert r['summary']['max_drawdown'] is None
    assert r['summary']['unverified_entry_outcomes']==['600000.SH']
    assert all(e['equity'] is None for e in r['equity'])
    assert any(a['reason']=='reference_continuity_break' for a in r['audit'])


def test_flat_unselected_reference_break_does_not_poison_later_clean_entry():
    from ashare_quant.replay import run_reference_replay
    dates = ('2024-01-31','2024-02-01','2024-02-02','2024-02-29','2024-03-01')
    f,b,c,k = inputs(dates, ('600000.SH','000001.SZ'))
    f = pd.DataFrame([dict(date=d, security_id=s, horizon=20, score=score,
                           score_status='ok') for d,pairs in [
        ('2024-01-31',[('600000.SH',2),('000001.SZ',1)]),
        ('2024-02-29',[('600000.SH',1),('000001.SZ',2)])] for s,score in pairs])
    b['reference_continuity_break'] = b.date.eq('2024-02-02') & b.security_id.eq('000001.SZ')
    b.loc[b.date.ge('2024-02-02') & b.security_id.eq('000001.SZ'),'adj_close'] = 1000.
    r = run_reference_replay(f,b,c,initial_cash=100_000,top_k=1)
    assert any(t['side']=='buy' and t['security_id']=='000001.SZ' and t['date']=='2024-03-01' for t in r['trades'])
    assert r['summary']['ending_equity'] is not None
    assert r['summary']['unresolved_reference_continuity']==[]
