"""Opt-in SYNTHETIC integration validation, never authentic market evidence.

Run: ASHARE_SYNTHETIC_E2E=1 .venv/bin/python -m pytest -q -s \
    tests/test_ashare_end_to_end.py \
    --basetemp=/tmp/ashare-integration-smoke/pytest

All raw inputs are deterministic inventions. Weekdays are a synthetic exchange
calendar, not the historical SSE calendar. Full daily factor/prediction paths
are exercised; estimator training uses a declared one-in-21-session fixture
sample so this test remains practical. No network or production artifacts.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

SYNTHETIC_NOTICE = (
    'SYNTHETIC INTEGRATION TEST ONLY: invented prices, securities and weekday '
    'calendar; not authentic Tushare training, performance evidence, or investment advice.'
)


def _partition(raw, endpoint, name, frame, params):
    directory = raw / endpoint
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / (name + '.parquet')
    frame.to_parquet(artifact, index=False)
    manifest = dict(
        endpoint=endpoint, status='complete' if len(frame) else 'empty',
        params=params, rows=len(frame), artifact=str(artifact.relative_to(raw)),
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        provenance=SYNTHETIC_NOTICE, completed_at='2026-09-30T00:00:00Z',
    )
    (directory / (name + '.manifest.json')).write_text(json.dumps(manifest))


def _make_synthetic_raw(root):
    """40 invented identities, chronological prices, dated daily-source schemas."""
    raw = root / 'raw'
    calendar = pd.bdate_range('2016-01-01', '2026-09-30')
    ids = ([f'{600000+i:06}.SH' for i in range(32)] +
           ['000001.SZ', '000002.SZ', '300001.SZ', '300002.SZ',
            '688001.SH', '688002.SH', '430001.BJ', '920001.BJ'])
    securities = pd.DataFrame([
        dict(ts_code=sid, name=f'SYNTHETIC {i}', list_date='20211115' if sid.endswith('.BJ') else '20000101',
             delist_date=None, list_status='L', industry='Synthetic fixture')
        for i, sid in enumerate(ids)
    ])
    _partition(raw, 'stock_basic', 'synthetic', securities, {'list_status': 'L'})
    _partition(raw, 'trade_cal', 'synthetic', pd.DataFrame(dict(
        exchange='SSE', cal_date=calendar.strftime('%Y%m%d'), is_open=1)), {'exchange': 'SSE'})
    _partition(raw, 'bse_mapping', 'synthetic', pd.DataFrame(columns=['o_code', 'n_code']), {})
    rng = np.random.default_rng(20260930)
    n, k = len(calendar), len(ids)
    market = rng.normal(.0001, .004, (n, 1))
    innovation = market + rng.normal(0, .013, (n, k))
    # Bounded recurrent individual patterns create both directions at all horizons.
    innovation += .0015 * np.sin(np.arange(n)[:, None] / 31 + np.arange(k)[None, :])
    closes = (8 + np.arange(k)[None, :] * .4) * np.exp(np.cumsum(innovation, axis=0))
    previous = np.vstack([closes[0] / np.exp(innovation[0]), closes[:-1]])
    opens = previous * np.exp(rng.normal(0, .002, (n, k)))
    highs = np.maximum(opens, closes) * 1.004
    lows = np.minimum(opens, closes) * .996
    amount_cny = rng.uniform(80_000_000, 140_000_000, (n, k))
    for t, day in enumerate(calendar):
        key = day.strftime('%Y%m%d')
        base = dict(ts_code=ids, trade_date=key)
        frames = {
            'daily': pd.DataFrame({**base, 'open': opens[t], 'close': closes[t],
                'high': highs[t], 'low': lows[t], 'pre_close': previous[t],
                'vol': amount_cny[t] / closes[t] / 100, 'amount': amount_cny[t] / 1000}),
            'daily_basic': pd.DataFrame({**base, 'total_mv': closes[t] * 100_000,
                'circ_mv': closes[t] * 70_000, 'total_share': 100_000.,
                'float_share': 70_000., 'free_share': 60_000.,
                'turnover_rate': amount_cny[t] / closes[t] / 600_000_000 * 100}),
            'adj_factor': pd.DataFrame({**base, 'adj_factor': 1.}),
            'stk_limit': pd.DataFrame({**base, 'up_limit': previous[t] * 1.10,
                'down_limit': previous[t] * .90}),
        }
        for endpoint, frame in frames.items():
            if day < pd.Timestamp('2021-11-15'):
                frame = frame.loc[~frame.ts_code.str.endswith('.BJ')].copy()
            _partition(raw, endpoint, key, frame, {'trade_date': key})
        if t % 500 == 0:
            print(f'SYNTHETIC raw fixture progress: {t}/{n} sessions', flush=True)
    benchmark_frames = []
    for i, code in enumerate(['000300.SH', '000905.SH', '000852.SH', '399006.SZ']):
        benchmark_frames.append(pd.DataFrame(dict(ts_code=code,
            trade_date=calendar.strftime('%Y%m%d'),
            close=1000 * np.exp(np.cumsum(market[:, 0] + (i - 1) * .00002)))))
    _partition(raw, 'index_daily', 'synthetic', pd.concat(benchmark_frames), {})
    (root / 'SYNTHETIC_ONLY.txt').write_text(SYNTHETIC_NOTICE)
    return calendar, ids


def _bound_synthetic_samples(root, manifest):
    """Declared fixture sampling only, preserving legal dates and honest hashes."""
    from ashare_quant.dataset import write_json
    before = after = 0
    for path in sorted((root / 'training_samples').glob('*.parquet')):
        frame = pd.read_parquet(path)
        before += len(frame)
        dates = pd.DatetimeIndex(frame.date.unique()).sort_values()[::21]
        selected = frame.loc[frame.date.isin(dates)].copy()
        after += len(selected)
        selected.to_parquet(path, index=False)
    manifest['synthetic_integration_only'] = True
    manifest['limitations'].insert(0, SYNTHETIC_NOTICE)
    manifest['synthetic_training_sampling'] = dict(
        rule='first eligible then every 21st session within each calendar year',
        eligible_rows_before=before, training_sample_rows=after)
    manifest['training_sample_sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / 'training_samples').glob('*.parquet'))
    }
    write_json(root / 'feature_manifest.json', manifest)
    return after


@pytest.mark.skipif(os.environ.get('ASHARE_SYNTHETIC_E2E') != '1',
                    reason='Opt-in full synthetic four-model integration smoke')
def test_synthetic_source_to_models_snapshot_api_replays_and_report(tmp_path):
    """Actual glue, four real estimator kinds, no model/prediction/replay mocks."""
    from ashare_quant.collect import audit_collection
    from ashare_quant.dataset import prepare_data, build_features, write_json
    from ashare_quant.training import run_training, split_window, KINDS, PROTOCOL
    from ashare_quant.pipeline import run_backtests
    from ashare_quant.report import build_report
    from ashare_quant.service import AShareQuantService, ModelNotPublished
    from ashare_quant.api import create_app
    from fastapi.testclient import TestClient
    import resource

    started = time.monotonic()
    root = tmp_path / 'SYNTHETIC_ONLY'
    data, models, results = (root / name for name in ('data', 'models', 'results'))
    data.mkdir(parents=True)
    root.joinpath('SYNTHETIC_ONLY.txt').write_text(SYNTHETIC_NOTICE)
    print(SYNTHETIC_NOTICE, flush=True)
    calendar, ids = _make_synthetic_raw(data)
    readiness = audit_collection(data / 'raw')
    assert readiness['core_status'] == 'complete'
    audit = prepare_data(data)
    assert audit['status'] == 'complete'
    assert audit['securities'] == 40 and len(audit['years']) == 11
    assert all(y['outside_master_or_listing_period'] == 0 for y in audit['years'])
    manifest = build_features(data)
    assert len(manifest['features']) == 37
    expected_rows = len(calendar) * (len(ids) - 2) + int((calendar >= pd.Timestamp('2021-11-15')).sum()) * 2
    assert manifest['rows'] == expected_rows
    assert manifest['data_as_of'] == '2026-09-30'
    sample_rows = _bound_synthetic_samples(data, manifest)
    samples = pd.concat([pd.read_parquet(p) for p in (data / 'training_samples').glob('*.parquet')])
    train, calibration = split_window(samples, manifest['features'], PROTOCOL['training_as_of'],
        calibration_start=PROTOCOL['calibration_start'])
    assert train.label_end.max() < pd.Timestamp('2023-01-01')
    assert calibration.label_end.max() <= pd.Timestamp('2023-12-29')
    for frame in (train, calibration):
        assert set(frame.horizon) == {1, 5, 20, 60}
        for _, part in frame.groupby('horizon'):
            assert len(part) >= 30 and part.fwd_return.gt(0).nunique() == 2
    print(f'SYNTHETIC 37-factor fixture ready: {manifest["rows"]} rows; {sample_rows} training observations', flush=True)
    status = run_training(data, results, models)
    assert status['status'] == 'trained'
    assert status['data_as_of'] == '2026-09-30'
    development = json.loads((results / 'development_summary.json').read_text())
    assert set(development) == set(KINDS)
    for kind in KINDS:
        metadata = json.loads((models / 'frozen' / f'{kind}.json').read_text())
        assert metadata['market'] == 'CN' and metadata['currency'] == 'CNY'
        assert pd.Timestamp(metadata['as_of']) == pd.Timestamp('2023-12-29')
        assert pd.Timestamp(metadata['latest_label_end']) <= pd.Timestamp('2023-12-29')
        assert metadata['sample_counts']['train']['used'] >= 120
        assert metadata['sample_counts']['calibration']['used'] >= 120
        assert development[kind]['as_of'] == '2024-12-31'
        assert development[kind]['horizons']['20']['ic_dates'] >= 20
        if kind.startswith('lightgbm'):
            assert metadata['parameters']['n_jobs'] <= 8
    frozen = json.loads((results / 'frozen_architecture.json').read_text())
    assert frozen['confirmation_used'] is False
    common = json.loads((results / 'development_common_universe.json').read_text())
    assert set(common) == set(KINDS)
    assert len({x['common_score_rows'] for x in common.values()}) == 1
    assert len({x['common_mature_rows'] for x in common.values()}) == 1
    assert all(x['ic_dates'] >= 20 for x in common.values())
    assert frozen['selected_kind'] == max(KINDS, key=lambda k: (
        common[k]['rank_ic_mean'], -list(KINDS).index(k)))
    assert frozen['selection_statistic'] == common[frozen['selected_kind']]['rank_ic_mean']
    confirmation = json.loads((results / 'confirmation_summary.json').read_text())
    for horizon in ('1', '5', '20', '60'):
        assert confirmation['horizons'][horizon]['mature_labels'] > 30
    service = AShareQuantService(data=data, models=models)
    visible = service.status()
    assert visible['forecast_count'] == 160 and visible['forecast_security_count'] == 40
    assert visible['research_ready'] is True and visible['eligible'] is False
    assert visible['execution_validated'] is False
    assert any('SYNTHETIC' in line for line in visible['limitations'])
    active = json.loads((models / 'active.json').read_text())
    assert hashlib.sha256((models / active['model_path']).read_bytes()).hexdigest() == active['model_sha256']
    current_forecasts = pd.read_parquet(models / active['forecast_path'])
    assert current_forecasts.fwd_return.isna().all()
    assert current_forecasts.label_end.isna().all()
    assert current_forecasts.score_status.eq('ok').all()
    assert pd.to_datetime(current_forecasts.model_trained_as_of).eq(pd.Timestamp('2026-09-30')).all()
    assert status['latest_model_version'] != status['frozen_model_version']
    from ashare_quant.daily_update import run_daily_update
    published_before = (models / 'active.json').read_bytes()
    refreshed = run_daily_update(data, models, results)
    assert refreshed['status'] == 'candidate_only' and refreshed['published'] is False
    assert (models / 'active.json').read_bytes() == published_before
    pd.testing.assert_frame_equal(current_forecasts, pd.read_parquet(refreshed['forecast_path']))
    ranking = service.rank_market(20)
    assert len(ranking['items']) == 40
    stock = service.forecast_stock(ids[0])
    assert [x['horizon'] for x in stock['forecasts']] == [1, 5, 20, 60]
    assert any(x['explanations'] for x in stock['forecasts'])
    with pytest.raises(ModelNotPublished):
        service.rank_market(20, as_of='2025-01-01')
    client = TestClient(create_app(service, api_key='synthetic-test-key-not-a-credential'))
    headers = {'x-api-key': 'synthetic-test-key-not-a-credential'}
    assert client.get('/health').json()['market'] == 'CN'
    assert client.get('/v1/model/status').status_code == 401
    assert client.get('/v1/model/status', headers=headers).status_code == 200
    assert len(client.get('/v1/rankings?horizon=20&limit=5', headers=headers).json()['items']) == 5
    assert client.get(f'/v1/stocks/{ids[0]}/forecast', headers=headers).status_code == 200
    assert client.get('/v1/rankings?horizon=2', headers=headers).status_code == 422
    advice = client.post('/v1/portfolio/advice', headers=headers,
        json={'holdings': [{'security_id': ids[0], 'quantity': 100, 'acquired_date': '2026-09-29'}],
              'cash': 100_000})
    assert advice.status_code == 200, advice.text
    csv_advice = client.post('/v1/portfolio/advice/csv', headers=headers,
        json={'holdings_csv': f'security_id,quantity,acquired_date\n{ids[0]},100,2026-09-29\n', 'cash': 100_000})
    assert csv_advice.status_code == 200, csv_advice.text
    print('SYNTHETIC training, frozen selection, holdout, snapshot and authenticated API passed', flush=True)
    actual, references = run_backtests(data, results)
    assert actual['currency'] == 'CNY' and actual['trade_count'] > 0
    assert actual['valuation_valid'] is True
    assert actual['audit_counts'].get('missing_signal_adv20', 0) == 0
    assert set(references) == {'fee_0.0010', 'fee_0.0025', 'fee_0.0050'}
    for reference in references.values():
        assert reference['trade_count'] > 0 and reference['actual_shares'] is False
        assert reference['execution_validated'] is False
    benchmarks = json.loads((results / 'benchmark_summary.json').read_text())
    assert len(benchmarks) == 4 and all(x['status'] == 'complete' for x in benchmarks.values())
    report = build_report(data, results, models)
    report_path = Path(report['report'])
    # Keep the generated fixture report unambiguously labeled at its first line.
    report_path.write_text('# SYNTHETIC INTEGRATION TEST ONLY\n\n' + SYNTHETIC_NOTICE + '\n\n' + report_path.read_text())
    assert report['status'] == 'complete' and report['performance_accepted'] is False
    assert 'SYNTHETIC' in report_path.read_text()
    summary = dict(synthetic_only=True, notice=SYNTHETIC_NOTICE,
        elapsed_seconds=round(time.monotonic()-started, 2),
        peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        source_sessions=len(calendar), source_securities=len(ids), feature_count=37,
        feature_rows=manifest['rows'], training_sample_observations=sample_rows,
        selected_kind=frozen['selected_kind'], forecast_rows=visible['forecast_count'],
        kinds=list(KINDS), actual_trade_count=actual['trade_count'],
        reference_trade_counts={k: v['trade_count'] for k, v in references.items()},
        benchmark_count=len(benchmarks), report_path=str(report_path))
    assert summary['peak_rss_mib'] < 9 * 1024
    write_json(root / 'synthetic_validation_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def _synthetic_rebalance_pipeline(tmp_path, score_unavailable=False, corporate_action=False):
    """Realistic contract: raw bars exclude ADV; dated model inputs contain it."""
    from ashare_quant.dataset import write_json
    from ashare_quant.pipeline import run_backtests
    data, results = tmp_path / 'SYNTHETIC_data', tmp_path / 'SYNTHETIC_results'
    for directory in (data / 'bars', data / 'normalized', data / 'references',
                      results / 'confirmation' / 'linear'):
        directory.mkdir(parents=True)
    days = pd.to_datetime(['2024-12-30', '2024-12-31', '2025-01-01', '2025-01-02',
                           '2025-01-30', '2025-01-31', '2025-02-03', '2025-02-04'])
    ids = [f'{600000+i:06}.SH' for i in range(40)]
    bars = pd.DataFrame([dict(date=d, security_id=s, raw_open=10., raw_close=10.,
        high=10.1, low=9.9, adj_close=10., adj_factor=1., volume=10_000_000.,
        amount=100_000_000., up_limit=11., down_limit=9., quote_present=True)
        for d in days for s in ids])
    if corporate_action:
        affected = bars.security_id.eq(ids[0]) & bars.date.ge(pd.Timestamp('2025-01-02'))
        bars.loc[affected, ['raw_open', 'raw_close', 'high', 'low']] = [9.9, 9.9, 10., 9.8]
        bars.loc[affected, 'adj_factor'] = 10 / 9.9
        action = dict(action_id='SYNTHETIC_DIVIDEND', security_id=ids[0], verified=True,
            ann_date=pd.Timestamp('2024-12-30'), record_date=pd.Timestamp('2025-01-01'),
            ex_date=pd.Timestamp('2025-01-02'), pay_date=pd.Timestamp('2025-01-30'),
            cash_per_share_pre_tax=.1, share_multiplier=1.,
            source_endpoint='SYNTHETIC_ONLY', payment_basis='synthetic fixture payment date')
        pd.DataFrame([action]).to_parquet(data / 'references' / 'verified_corporate_actions.parquet', index=False)
    pd.DataFrame(dict(cal_date=days, is_open=1)).to_parquet(data / 'calendar.parquet', index=False)
    for year, group in bars.groupby(bars.date.dt.year):
        group.to_parquet(data / 'bars' / f'{year}.parquet', index=False)
        group[['date', 'security_id']].assign(adv20_amount=100_000_000.).to_parquet(
            data / 'normalized' / f'{year}.parquet', index=False)
    for signal_day in pd.to_datetime(['2024-12-31', '2025-01-31']):
        reverse = signal_day.year == 2025
        forecasts = pd.DataFrame([dict(date=signal_day, security_id=s, horizon=20,
            score=float(i if reverse else 40-i), score_status='ok', adv20_amount=100_000_000.)
            for i, s in enumerate(ids)])
        if reverse and score_unavailable:
            forecasts.loc[forecasts.security_id.eq(ids[0]), ['score', 'score_status']] = [np.nan, 'insufficient_model_inputs']
        forecasts.to_parquet(results / 'confirmation' / 'linear' / f'{signal_day:%Y-%m}.parquet', index=False)
    pd.DataFrame(dict(ts_code='000300.SH', trade_date=days.strftime('%Y%m%d'), close=100.)).to_parquet(
        data / 'references' / 'benchmarks.parquet', index=False)
    write_json(data / 'feature_manifest.json', dict(data_as_of='2025-02-04', synthetic_only=True))
    write_json(results / 'frozen_architecture.json', dict(selected_kind='linear'))
    write_json(results / 'training_status.json', dict(status='trained', synthetic_only=True))
    actual, reference = run_backtests(data, results)
    return actual, reference, results


def test_backtest_pipeline_supplies_signal_adv_for_actual_sells(tmp_path):
    actual, _, results = _synthetic_rebalance_pipeline(tmp_path)
    assert actual['audit_counts'].get('missing_signal_adv20', 0) == 0, actual
    trades = pd.read_parquet(results / 'execution' / 'trades.parquet')
    assert trades.side.eq('sell').any()
    assert actual['open_positions'] == 30


def test_backtest_pipeline_supplies_signal_adv_for_reference_sells(tmp_path):
    _, references, results = _synthetic_rebalance_pipeline(tmp_path)
    for fee, summary in references.items():
        assert summary['completed_positions'] >= 30, summary
        trades = pd.read_parquet(results / 'reference' / fee / 'trades.parquet')
        assert trades.side.eq('sell').any()
        assert summary['open_positions'] == 30


def test_backtest_pipeline_keeps_liquidity_for_held_name_without_new_score(tmp_path):
    actual, references, _ = _synthetic_rebalance_pipeline(tmp_path, score_unavailable=True)
    assert actual['audit_counts'].get('missing_signal_adv20', 0) == 0, actual
    assert actual['open_positions'] == 30
    assert all(summary['completed_positions'] >= 30 for summary in references.values())


def test_backtest_pipeline_serializes_dated_corporate_action_event_ledger(tmp_path):
    actual, _, results = _synthetic_rebalance_pipeline(tmp_path, corporate_action=True)
    assert actual['cash_dividends_paid'] > 0
    assert actual['valuation_valid'] is True
    events = pd.read_parquet(results / 'execution' / 'events.parquet')
    assert {'entitlement', 'ex_entitlement', 'cash_payment'} <= set(events.event_type)
    assert events.loc[events.event_type.eq('cash_payment'), 'date'].iloc[0] == '2025-01-30'
    cash = pd.read_parquet(results / 'execution' / 'cash.parquet')
    assert cash.loc[cash.event.eq('cash_dividend'), 'amount'].sum() == actual['cash_dividends_paid']


def test_frozen_selection_rejects_edited_architecture_with_unchanged_development_hash(tmp_path):
    from ashare_quant.training import freeze_architecture
    development = {kind: {'horizons': {'20': {'rank_ic_mean': score}}}
                   for kind, score in [('factor', .1), ('linear', .2),
                                       ('lightgbm_small', .3), ('lightgbm_large', .4)]}
    frozen = freeze_architecture(development, tmp_path)
    path = tmp_path / 'frozen_architecture.json'
    frozen['selected_kind'] = 'factor'
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='frozen|Architecture|selection|integrity'):
        freeze_architecture(development, tmp_path)


def test_development_ic_is_daily_equal_weight_and_keeps_unmatured_prediction_denominator(tmp_path):
    from ashare_quant.training import evaluate_development
    from tests.test_ashare_service import complete_forecast
    rows = []
    for day, count, direction in [('2024-02-01', 20, 1), ('2024-02-02', 40, -1)]:
        for i in range(count):
            row = complete_forecast(f'{600000+i:06}.SH', 20, day)
            row.update(score=float(i), fwd_return=direction * (i - count/2) / 1000,
                       label_end=pd.Timestamp('2024-03-15'))
            rows.append(row)
    immature = complete_forecast('600000.SH', 20, '2024-12-31')
    immature.update(score=1., fwd_return=.99, label_end=pd.Timestamp('2025-01-28'))
    rows.append(immature)
    path = tmp_path / 'SYNTHETIC_predictions.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    summary = evaluate_development([path])['horizons']['20']
    assert summary['predictions'] == 61
    assert summary['mature_labels'] == summary['score_available'] == 60
    assert summary['ic_dates'] == 2
    assert summary['rank_ic_mean'] == pytest.approx(0., abs=1e-14)
    assert summary['probability_available'] == 60


def test_deterministic_training_sample_does_not_filter_unknown_future_outcomes():
    from ashare_quant.features import sample_observations
    rows = pd.DataFrame(dict(date=pd.Timestamp('2024-01-02'),
        security_id=[f'{600000+i:06}.SH' for i in range(300)], status='ok',
        fwd_return_20=np.nan))
    sampled = sample_observations(rows, max_stocks_per_date=256)
    assert len(sampled) == 256 and sampled.fwd_return_20.isna().all()
    pd.testing.assert_frame_equal(sampled, sample_observations(rows, max_stocks_per_date=256))
    realized = rows.assign(fwd_return_20=np.linspace(-1, 1, len(rows)))
    assert sampled.security_id.tolist() == sample_observations(realized, max_stocks_per_date=256).security_id.tolist()


def test_development_comparison_uses_common_scores_before_label_maturity(tmp_path):
    from ashare_quant import training
    assert hasattr(training, 'compare_common_score_universe'), 'Common-score development comparison missing'
    paths = {}
    for kind in training.KINDS:
        rows = []
        for date, label_end in [('2024-06-03', '2024-07-01'), ('2024-12-20', '2025-01-17')]:
            for i in range(40):
                available = not (kind == 'factor' and date == '2024-06-03' and i < 10)
                direction = 1 if kind in ('factor', 'linear') else -1
                rows.append(dict(date=pd.Timestamp(date), security_id=f'{600000+i:06}.SH', horizon=20,
                    score=float(direction*i) if available else np.nan,
                    score_status='ok' if available else 'insufficient_model_inputs',
                    fwd_return=np.nan if (date == '2024-06-03' and i == 10) else (i - 20)/1000,
                    label_end=pd.Timestamp(label_end)))
        # Another horizon cannot enlarge the 20-session comparison universe.
        rows.append(dict(rows[-1], horizon=1))
        path = tmp_path / f'SYNTHETIC_{kind}.parquet'
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths[kind] = [path]
    compared = training.compare_common_score_universe(paths, cutoff='2024-12-31')
    assert set(compared) == set(training.KINDS)
    for kind, result in compared.items():
        # 30 common-score rows in June plus 40 still-unmatured December rows.
        assert result['common_score_rows'] == 70
        # One observed-period outcome is missing; all December labels are late.
        assert result['common_mature_rows'] == 29
        assert result['ic_dates'] == 1
        assert result['rank_ic_mean'] == pytest.approx(1. if kind in ('factor', 'linear') else -1.)


def test_quarantine_preserves_past_normalized_inputs_and_missing_label_denominators(tmp_path):
    """A future source break changes crossing outcomes, never earlier inputs."""
    from ashare_quant.dataset import build_features
    from ashare_quant.data import normalize_securities
    from tests.test_ashare_core import bars_fixture
    bars, calendar = bars_fixture()
    bars['pre_close'] = bars.raw_close.shift().fillna(bars.raw_close)
    broken = bars.copy()
    broken.loc[250:, 'adj_factor'] = 2.
    broken['adj_close'] = broken.raw_close * broken.adj_factor
    frames = {}
    manifests = {}
    roots = {}
    for name, source in [('clean', bars), ('quarantined', broken)]:
        root = tmp_path / ('SYNTHETIC_' + name)
        roots[name] = root
        directory = root / 'bars_by_security' / '600000.SH'
        directory.mkdir(parents=True)
        source_hashes = {}
        for year, group in source.groupby(source.date.dt.year):
            path = directory / f'{year}.parquet'
            group.to_parquet(path, index=False)
            source_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        pd.DataFrame(dict(cal_date=calendar, is_open=1)).to_parquet(root / 'calendar.parquet', index=False)
        normalize_securities(pd.DataFrame([dict(ts_code='600000.SH', name='SYNTHETIC',
            list_date='19910101', delist_date=None)])).to_parquet(root / 'securities.parquet', index=False)
        manifests[name] = build_features(root)
        frames[name] = pd.read_parquet(root / 'normalized').sort_values('date').reset_index(drop=True)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
    assert manifests['clean']['features'] == manifests['quarantined']['features']
    assert len(manifests['quarantined']['features']) == 37
    assert manifests['quarantined']['rows'] == len(bars)
    assert len(manifests['quarantined']['adjustment_reference_issues']) == 1
    before_break = frames['clean'].date.lt(calendar[250])
    input_columns = [c for c in frames['clean'] if not c.startswith(('fwd_return_', 'label_end_'))]
    pd.testing.assert_frame_equal(frames['clean'].loc[before_break, input_columns],
                                  frames['quarantined'].loc[before_break, input_columns])
    samples = pd.read_parquet(roots['quarantined'] / 'training_samples')
    for horizon in (1, 5, 20, 60):
        crossing_dates = calendar[250-horizon:250]
        crossing = frames['quarantined'].loc[frames['quarantined'].date.isin(crossing_dates)]
        assert len(crossing) == horizon and crossing.status.eq('ok').all()
        assert crossing[f'fwd_return_{horizon}'].isna().all()
        assert crossing[f'label_end_{horizon}'].notna().all()
        assert set(crossing_dates) <= set(samples.date)
        coverage = manifests['quarantined']['label_coverage'][str(horizon)]
        assert coverage['eligible_mature_missing'] == horizon
        assert coverage['eligible_mature'] == coverage['eligible_mature_observed'] + horizon
    assert frames['quarantined'].status.iloc[250] == 'unresolved_adjustment'
    assert frames['quarantined'].status.iloc[251:309].eq('insufficient_history').all()
    assert frames['quarantined'].status.iloc[309] == 'ok'


@pytest.mark.parametrize('tail_case', ['still_listed', 'pre_delist', 'no_source_directory', 'no_scored_year'])
def test_dated_master_retains_unavailable_quote_tails_and_absent_identities(tmp_path, tail_case):
    from ashare_quant.data import normalize_securities
    from ashare_quant.dataset import build_features
    from ashare_quant.features import security_features
    from tests.test_ashare_core import bars_fixture
    bars, calendar = bars_fixture()
    root = tmp_path / ('SYNTHETIC_' + tail_case)
    target, anchor = '600000.SH', '600001.SH'
    target_source = bars.iloc[:350].copy()
    source_hashes = {}
    anchor_source = bars.loc[bars.date.dt.year.eq(2019)] if tail_case == 'no_scored_year' else bars
    sources = [(anchor, anchor_source.assign(security_id=anchor))]
    absent_source = tail_case in ('no_source_directory', 'no_scored_year')
    if not absent_source:
        sources.append((target, target_source))
    for sid, source in sources:
        directory = root / 'bars_by_security' / sid
        directory.mkdir(parents=True)
        for year, group in source.groupby(source.date.dt.year):
            path = directory / f'{year}.parquet'
            group.to_parquet(path, index=False)
            source_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    delist_date = calendar[380] if tail_case == 'pre_delist' else None
    master = normalize_securities(pd.DataFrame([
        dict(ts_code=target, name='SYNTHETIC missing quotes', list_date='19910101',
             delist_date=delist_date.strftime('%Y%m%d') if delist_date is not None else None),
        dict(ts_code=anchor, name='SYNTHETIC observed', list_date='19910101', delist_date=None)]))
    master.to_parquet(root / 'securities.parquet', index=False)
    pd.DataFrame(dict(cal_date=calendar, is_open=1)).to_parquet(root / 'calendar.parquet', index=False)
    manifest = build_features(root)
    normalized = pd.read_parquet(root / 'normalized')
    target_rows = normalized.loc[normalized.security_id.eq(target)].sort_values('date').reset_index(drop=True)
    expected_dates = calendar[:380] if delist_date is not None else calendar
    assert target_rows.date.tolist() == list(expected_dates)
    assert len(manifest['features']) == 37
    assert manifest['rows'] == len(calendar) + len(expected_dates)
    assert manifest['securities'] == 2
    unavailable_dates = expected_dates if absent_source else expected_dates[350:]
    unavailable = target_rows.loc[target_rows.date.isin(unavailable_dates)]
    assert unavailable.status.eq('no_trade_quote').all()
    assert unavailable.quote_present.eq(False).all()
    individual_inputs = [c for c in manifest['features'] if not c.startswith('market_')]
    assert unavailable[individual_inputs + ['raw_close', 'adj_close_cny', 'adv20_amount', 'sigma_daily']].isna().all().all()
    for h in (1, 5, 20, 60):
        assert unavailable[f'fwd_return_{h}'].isna().all()
        expected_end = pd.Series(calendar, index=calendar).shift(-h).reindex(unavailable_dates).reset_index(drop=True)
        pd.testing.assert_series_equal(unavailable[f'label_end_{h}'].reset_index(drop=True), expected_end,
                                       check_names=False)
    latest = pd.read_parquet(root / 'latest_inputs.parquet')
    assert (target in set(latest.security_id)) is (delist_date is None)
    assert manifest['input_status_counts']['no_trade_quote'] == len(unavailable_dates) + len(calendar) - len(anchor_source)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in source_hashes.items())
    if not absent_source:
        original = security_features(target_source, calendar).reset_index(drop=True)
        saved = pd.read_parquet(root / 'features' / f'{target}.parquet')
        observed = saved.loc[saved.date.le(target_source.date.max()), list(original)].reset_index(drop=True)
        original.attrs = {}; observed.attrs = {}
        pd.testing.assert_frame_equal(original, observed, check_dtype=False)
        assert manifest['label_coverage']['20']['eligible_mature_missing'] == 20
    else:
        assert manifest['identities_without_source_bars'] == 1


def test_bounded_feature_reader_preserves_pandas_values_order_nulls_and_sample_ids(tmp_path):
    """Resource-only reader substitution cannot change the 256-name draw."""
    from ashare_quant import dataset
    from ashare_quant.features import security_features, feature_columns, cross_sectional_inputs, sample_observations
    from tests.test_ashare_core import bars_fixture
    assert hasattr(dataset, 'read_feature_window'), 'Bounded Arrow feature-window reader missing'
    bars, calendar = bars_fixture()
    base = security_features(bars, calendar)
    base.attrs = {'adjustment_issues': [], 'synthetic_only': True,
                  'provenance_note': 'SYNTHETIC bounded reader equivalence fixture'}
    base['quote_present'] = True
    base['is_st'] = pd.Series([pd.NA if i % 17 == 0 else False for i in range(len(base))], dtype='boolean')
    for name, value in [('market_breadth_60', .5), ('market_momentum_20', .01),
                        ('market_momentum_60', .03), ('market_momentum_252', .12),
                        ('market_volatility_60', .15)]:
        base[name] = value
    features = feature_columns(base)
    assert len(features) == 37
    directory = tmp_path / 'SYNTHETIC_features'
    directory.mkdir()
    for i in range(300):
        frame = base.copy()
        frame['security_id'] = f'{600000+i:06}.SH'
        frame['momentum_20'] = frame.momentum_20 + i / 1_000_000
        frame['log_amount_20'] = frame.log_amount_20 + i / 1000
        frame.to_parquet(directory / f'{600000+i:06}.SH.parquet', index=False, row_group_size=63)
    windows = [('2018-01-01', '2019-01-01', None),
               ('2018-03-01', '2018-04-01', None),
               ('2018-12-01', '2019-02-01', ['date', 'security_id', 'is_st', 'fwd_return_20', 'label_end_20']),
               ('2019-06-01', '2019-08-01', ['date', 'security_id', 'is_st', 'fwd_return_20', 'label_end_20']),
               ('2030-01-01', '2030-02-01', ['date', 'security_id', 'is_st', 'fwd_return_20', 'label_end_20'])]
    for start, end, columns in windows:
        filters = [('date', '>=', pd.Timestamp(start)), ('date', '<', pd.Timestamp(end))]
        expected = pd.read_parquet(directory, columns=columns, filters=filters)
        actual = dataset.read_feature_window(directory, pd.Timestamp(start), pd.Timestamp(end), columns=columns)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        assert actual.attrs == expected.attrs
        if columns is None:
            expected[features] = expected[features].astype(np.float32)
            actual[features] = actual[features].astype(np.float32)
            expected_inputs = cross_sectional_inputs(expected, features)
            actual_inputs = cross_sectional_inputs(actual, features)
            pd.testing.assert_frame_equal(expected_inputs, actual_inputs, check_exact=True)
            expected_samples = sample_observations(expected_inputs.loc[expected_inputs.status.eq('ok')])
            actual_samples = sample_observations(actual_inputs.loc[actual_inputs.status.eq('ok')])
            assert expected_samples.groupby('date').size().eq(256).all()
            pd.testing.assert_frame_equal(expected_samples, actual_samples, check_exact=True)
