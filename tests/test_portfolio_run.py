import json
from pathlib import Path

import pandas as pd

from hk_quant.portfolio_run import export_backtest
from test_strategy import files


def test_export_records_cash_replay_and_confirmation_input(tmp_path):
    root,forecasts,dates,*_=files(tmp_path)
    output=tmp_path/'result'
    result=export_backtest(root,forecasts,output,start=dates[0],end=dates[-1],initial_cash=100000.)
    assert result['status']=='ok'
    metrics=json.loads((output/'development_portfolio_metrics.json').read_text())
    assert metrics['portfolio_valid'] is True
    replay=Path(result['replay_directory'])
    for name in ('model','equal_weight','stress_model','stress_equal_weight'):
        trades=pd.read_parquet(replay/name/'trades.parquet')
        assert trades.date.min()==dates[1]
        assert (replay/name/'events.parquet').exists()
        assert (replay/name/'policy_reports.json').exists()


def test_incomplete_export_cannot_leave_old_passing_metrics(tmp_path):
    root,forecasts,dates,*_=files(tmp_path)
    output=tmp_path/'result';output.mkdir()
    (output/'development_portfolio_metrics.json').write_text('{"portfolio_valid":true}')
    (root/'execution_coverage.json').unlink()
    result=export_backtest(root,forecasts,output,start=dates[0],end=dates[-1])
    assert result['status']=='incomplete'
    metrics=json.loads((output/'development_portfolio_metrics.json').read_text())
    assert metrics['portfolio_valid'] is False
    assert '缺少' in metrics['reason']
    assert not (output/'portfolio_replay').exists()
