import json

import pandas as pd
import pytest

from hk_quant.fixed_backtest import run_fixed_backtest


@pytest.fixture
def fixed_inputs(tmp_path):
    forecasts, data, source, results = [tmp_path / name for name in ('forecasts', 'data', 'source', 'results')]
    forecasts.mkdir()
    data.mkdir()
    records = {}
    for kind in ('factor', 'linear', 'lightgbm_small', 'lightgbm_large'):
        model = tmp_path / f'{kind}.pkl'
        model.write_bytes(b'Frozen model: replay must never load, train or change this file.')
        records[kind] = dict(path=str(model), model_version=kind, trained_as_of='2023-12-29')
        folder = forecasts / kind
        folder.mkdir()
        pd.DataFrame([dict(date=pd.Timestamp(day), security_id=sid, horizon=20, score=score,
                           score_status='ok', adv20_amount=2_000_000., model_version=kind,
                           model_trained_as_of='2023-12-29', label_end=pd.Timestamp('2024-02-05'),
                           probability_up=.7, probability_status='ok', fwd_return=.1)
                      for day in ('2024-01-01', '2024-01-31')
                      for sid, score in [('A', 2.), ('T', 1.)]]).to_parquet(folder / 'predictions.parquet')
    (forecasts / 'protocol.json').write_text(json.dumps(dict(models=records, data_as_of='2024-02-05')))
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-31', '2024-02-01', '2024-02-05'])
    pd.DataFrame(dict(cal_date=dates, is_open=1)).to_parquet(data / 'calendar.parquet')
    pd.DataFrame(dict(security_id=['A', 'T', 'B'], currency=['HKD'] * 3,
                      identity_status=['verified'] * 3,
                      identity_valid_from=pd.to_datetime(['2020-01-01', '2020-01-01', '2024-02-01']),
                      identity_valid_to=pd.to_datetime(['2024-02-01', '2024-02-01', None])))\
        .to_parquet(data / 'securities.parquet')
    quotes, factors = [], []
    for day in dates:
        for sid in (['A', 'T'] if day < pd.Timestamp('2024-02-01') else ['B']):
            price, factor = (20., 2.) if sid == 'B' else (10., 1.)
            quotes.append(dict(ts_code=sid, trade_date=day, open=price * factor,
                               high=price * factor, low=price * factor, close=price * factor,
                               vol=100000., amount=100000. * price, vwap=price))
            factors.append(dict(ts_code=sid, trade_date=day, close_price=price, cum_adjfactor=factor))
    for api, rows in [('hk_daily_adj', quotes), ('hk_adjfactor', factors)]:
        folder = source / 'source' / api
        folder.mkdir(parents=True)
        frame = pd.DataFrame(rows)
        for month, part in frame.groupby(frame.trade_date.dt.strftime('%Y%m')):
            part.to_parquet(folder / f'{month}.parquet')
    actions = tmp_path / 'actions.csv'
    pd.DataFrame([dict(security_id='A', effective_date='2024-02-01', payment_date='2024-02-05',
                       known_date='2024-01-31', factor_date='2024-01-31', cash_per_share_hkd=12.,
                       source_url='https://issuer.example/cash.pdf',
                       payment_basis='issuer_final_schedule_simulated')]).to_csv(actions, index=False)
    transfers = tmp_path / 'transfers.csv'
    pd.DataFrame([dict(security_id='T', successor_id='B', effective_date='2024-02-01',
                       known_date='2024-01-31', old_factor_date='2024-01-31',
                       new_factor_date='2024-02-01', share_multiplier=1.,
                       source_url='https://issuer.example/transfer.pdf')]).to_csv(transfers, index=False)
    return forecasts, data, source, results, actions, transfers, records


def test_saved_predictions_replay_cash_events_and_transfers_without_training(fixed_inputs, monkeypatch):
    forecasts, data, source, results, actions, transfers, records = fixed_inputs
    from pathlib import Path
    import hk_quant.research as research

    def forbidden_training(*args, **kwargs):
        pytest.fail('A fixed backtest must not train')

    monkeypatch.setattr(research, 'fit_at', forbidden_training)
    states = {r['path']: (Path(r['path']).read_bytes(), Path(r['path']).stat().st_mtime_ns)
              for r in records.values()}
    result = run_fixed_backtest(forecasts, data, source, results, initial_cash=100.,
                                terminal_actions_path=actions, transfers_path=transfers)
    for kind in records:
        normal, stress = result[kind]['normal'], result[kind]['stress']
        assert normal['training_performed'] is False
        assert normal['ending_equity'] > 100.
        assert normal['open_positions'] == 0
        assert normal['missing_valuation_days'] == 0
        assert stress['ending_equity'] < normal['ending_equity']
        trades = pd.read_csv(results / kind / 'normal/trades.csv')
        original_units = trades.loc[trades.security_id.eq('T'), 'units'].iloc[0]
        converted = trades.loc[trades.security_id.eq('B')].iloc[0]
        assert converted.units == pytest.approx(original_units / 2.)
        daily = pd.read_csv(results / kind / 'normal/daily.csv').set_index('date')
        assert daily.index.is_unique
        assert daily.loc['2024-02-01', 'receivables'] > 0
        assert daily.loc['2024-02-05', 'receivables'] == 0
    for path, state in states.items():
        assert (Path(path).read_bytes(), Path(path).stat().st_mtime_ns) == state
    status = json.loads((results / 'run_status.json').read_text())
    assert status == dict(status='complete', training_performed=False)


def test_mismatched_saved_model_version_is_rejected_before_replay(fixed_inputs):
    forecasts, data, source, results, actions, transfers, _ = fixed_inputs
    path = forecasts / 'factor/predictions.parquet'
    frame = pd.read_parquet(path)
    frame['model_version'] = 'different-model'
    frame.to_parquet(path)
    with pytest.raises(ValueError, match='模型版本'):
        run_fixed_backtest(forecasts, data, source, results,
                           terminal_actions_path=actions, transfers_path=transfers)
    assert not results.exists()
