"""直接回放已保存的港股模型预测，不读取训练样本或更新模型。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import write_json
from .data import join_market_sources
from .dated_identity import identity_row_mask
from .price_replay import replay_prices
from .research import common_signal_universe, prediction_accuracy


def run_fixed_backtest(forecast_root, data_root, source_root, results_root, *, initial_cash=1_000_000.,
                       terminal_actions_path=None, transfers_path=None):
    forecast_root, data_root, source_root, results_root = map(
        Path, (forecast_root, data_root, source_root, results_root))
    original = json.loads((forecast_root / 'protocol.json').read_text(encoding='utf-8'))
    model_records = original['models']
    as_of = pd.Timestamp(original['data_as_of'])
    candidates = {}
    model_files = {}
    for kind, record in model_records.items():
        path = Path(record['path'])
        model_files[path] = (path.stat().st_size, path.stat().st_mtime_ns)
        frame = pd.read_parquet(forecast_root / kind / 'predictions.parquet')
        trained = pd.to_datetime(frame.model_trained_as_of)
        if (not frame.model_version.eq(record['model_version']).all()
                or not trained.eq(pd.Timestamp(record['trained_as_of'])).all()
                or not trained.lt(frame.date).all()):
            raise ValueError('保存的预测与模型版本或历史训练截止日不一致')
        candidates[kind] = frame
    common = common_signal_universe(list(candidates.values()), 1_000_000.)
    rankings = {}
    for kind, frame in candidates.items():
        rank = frame.loc[frame.horizon.eq(20)]
        rank = rank.loc[pd.MultiIndex.from_frame(rank[['date', 'security_id']]).isin(common)]
        rankings[kind] = rank[['date', 'security_id', 'score', 'adv20_amount']].copy()
    selected = pd.concat([rank.sort_values(['date', 'score', 'security_id'], ascending=[True, False, True])
                          .groupby('date').head(30) for rank in rankings.values()], ignore_index=True)
    ids = set(selected.security_id)
    transfers = None
    if transfers_path is not None:
        transfers = pd.read_csv(transfers_path, parse_dates=['effective_date', 'known_date',
                                                            'old_factor_date', 'new_factor_date'])
        ids.update(transfers.successor_id)
        ids.update(transfers.security_id)
    start = selected.date.min()
    master = pd.read_parquet(data_root / 'securities.parquet').set_index('security_id')
    parts = []
    for month in pd.period_range(start, as_of, freq='M'):
        filename = month.strftime('%Y%m') + '.parquet'
        quote = pd.read_parquet(source_root / 'source/hk_daily_adj' / filename)
        factor = pd.read_parquet(source_root / 'source/hk_adjfactor' / filename)
        part, _ = join_market_sources(quote.loc[quote.ts_code.isin(ids)], factor.loc[factor.ts_code.isin(ids)])
        valid = identity_row_mask(part, master) & part.security_id.map(master.currency).eq('HKD')
        part['data_valid'] &= valid
        parts.append(part.loc[part.date.between(start, as_of)])
    bars = pd.concat(parts, ignore_index=True)
    calendar_frame = pd.read_parquet(data_root / 'calendar.parquet')
    calendar = pd.DatetimeIndex(calendar_frame.loc[calendar_frame.is_open.eq(1), 'cal_date']).sort_values()
    calendar = calendar[calendar <= as_of]

    def adjustment_at(sid, day):
        factors = pd.read_parquet(source_root / 'source/hk_adjfactor' / f'{day:%Y%m}.parquet')
        evidence = factors.loc[factors.ts_code.eq(sid) & factors.trade_date.eq(day), 'cum_adjfactor']
        if len(evidence) != 1 or not np.isfinite(evidence.iloc[0]) or evidence.iloc[0] <= 0:
            raise ValueError('公司行动缺少明确的历史复权单位换算证据')
        return float(evidence.iloc[0])

    if transfers is not None:
        transfers['unit_multiplier'] = np.nan
        for index, transfer in transfers.iterrows():
            if (transfer.old_factor_date >= transfer.effective_date
                    or transfer.new_factor_date != transfer.effective_date
                    or not np.isfinite(transfer.share_multiplier) or transfer.share_multiplier <= 0):
                raise ValueError('转板股份比例或换算日期无效')
            old_factor = adjustment_at(transfer.security_id, transfer.old_factor_date)
            new_factor = adjustment_at(transfer.successor_id, transfer.new_factor_date)
            transfers.loc[index, 'unit_multiplier'] = old_factor * transfer.share_multiplier / new_factor
    actions = None
    if terminal_actions_path is not None:
        actions = pd.read_csv(terminal_actions_path, parse_dates=['effective_date', 'payment_date',
                                                                'known_date', 'factor_date'])
        actions['adjusted_cash_per_unit'] = np.nan
        for index, action in actions.iterrows():
            if (not np.isfinite(action.cash_per_share_hkd)
                    or action.cash_per_share_hkd < 0 or action.factor_date > action.effective_date):
                raise ValueError('终止结算缺少明确的历史复权单位换算证据')
            actions.loc[index, 'adjusted_cash_per_unit'] = (
                action.cash_per_share_hkd * adjustment_at(action.security_id, action.factor_date))
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / 'run_status.json', dict(status='running', training_performed=False))
    protocol = dict(training_performed=False, model_records=model_records,
                    data_as_of=str(as_of.date()), initial_cash=initial_cash,
                    selection='Month-end 20-session scores; common available pool, top 30; no future-outcome selection.',
                    execution='Next-session positive-volume close; unfilled buy stays cash; pending sale attempts on later sessions.',
                    sizing='95% of cash available after sales, divided into 30 fixed slots; 1% of signal ADV and actual daily turnover.',
                    return_basis='HKD source-adjusted synthetic units',
                    valuation='Explicit same-day reference quotes; missing values remain missing. No forward filling.',
                    terminal_actions=str(Path(terminal_actions_path).resolve()) if terminal_actions_path else None,
                    transfers=str(Path(transfers_path).resolve()) if transfers_path else None,
                    event_recognition='Terminal effective_date is account recognition after the final notice; legal effective date is retained in source evidence.',
                    liquidation='Last completed month-end schedules liquidation; pending sales continue to data cutoff.',
                    limitations=['Synthetic adjusted units, not historical raw-share or board-lot account returns.',
                                 'Historical candidate comparison; recent results may have been viewed before, so not a pristine holdout.',
                                 'Cash dividends and capital events embedded in source adjustments; exact cash payment timing is not reconstructed.',
                                 'Daily closing-price execution and turnover caps are simulation assumptions, not guaranteed fills.',
                                 'Terminal cash uses explicitly cited issuer schedules as simulated payment dates.',
                                 'Historical identity and security universe completeness remain limited to source evidence.'])
    write_json(results_root / 'protocol.json', protocol)
    results, summaries = {}, []
    for kind, rank in rankings.items():
        results[kind] = {}
        for scenario, fee in [('normal', .0025), ('stress', .005)]:
            print(f'fixed replay {kind} {scenario}', flush=True)
            replay = replay_prices(rank, bars, calendar, initial_cash=initial_cash, fee=fee,
                                   terminal_actions=actions, transfers=transfers, end=as_of)
            folder = results_root / kind / scenario
            folder.mkdir(parents=True, exist_ok=True)
            for name, value in replay.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(folder / f'{name}.csv', index=False)
            summary = replay['summary']
            summary['unfilled_buy_records'] = int(replay['orders'].status.str.startswith('buy_unfilled').sum())
            summary['delayed_sale_records'] = int(replay['orders'].status.str.startswith('sell_pending').sum())
            write_json(folder / 'metrics.json', summary)
            results[kind][scenario] = summary
            summaries.append(dict(model=kind, scenario=scenario, **summary))
        results[kind]['direction'] = prediction_accuracy(candidates[kind], as_of)
    for path, state in model_files.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != state:
            raise ValueError('回测期间模型文件发生变化')
    write_json(results_root / 'metrics.json', dict(protocol=protocol, results=results, saved_models_unchanged=True))
    pd.DataFrame(summaries).to_csv(results_root / 'summary.csv', index=False)
    write_json(results_root / 'run_status.json', dict(status='complete', training_performed=False))
    return results


def main():
    parser = argparse.ArgumentParser(description='读取固定模型预测进行逐日回测，不训练模型')
    for name in ['forecast-root', 'data-root', 'source-root', 'results-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--initial-cash', type=float, default=1_000_000.)
    parser.add_argument('--terminal-actions', type=Path)
    parser.add_argument('--transfers', type=Path)
    args = parser.parse_args()
    run_fixed_backtest(args.forecast_root, args.data_root, args.source_root, args.results_root,
                       initial_cash=args.initial_cash, terminal_actions_path=args.terminal_actions,
                       transfers_path=args.transfers)


if __name__ == '__main__':
    main()
