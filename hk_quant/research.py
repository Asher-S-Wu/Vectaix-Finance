"""新版模型的月度量价研究；结果不作为正式账户回测或发布凭据。"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import write_json
from .contracts import FORECAST_RETURN_BASIS
from .data import join_market_sources, security_master
from .features import identity_period_features, write_security_features
from .market_context import market_context_from_inputs
from .training import KINDS, MODEL_REVISION, cross_sectional_inputs, fit_at, predict_frame, sample_observations


def observed_quote_periods(bars):
    dates = bars[['security_id', 'date']].copy()
    dates.loc[~bars.quote_present, 'date'] = pd.NaT
    return dates.groupby('security_id').date.agg(first_quote='min', last_quote='max').reset_index()


def interval_returns(prices, calendar, signal, next_signal, securities):
    sessions = pd.DatetimeIndex(calendar)
    entry = sessions[sessions.get_loc(pd.Timestamp(signal)) + 1]
    exit_day = sessions[sessions.get_loc(pd.Timestamp(next_signal)) + 1]
    values = prices.reindex(index=[entry, exit_day], columns=securities)
    gross = values.iloc[1] / values.iloc[0].where(values.iloc[0].gt(0)) - 1
    return pd.DataFrame({'security_id': list(securities), 'entry_date': entry,
                         'exit_date': exit_day, 'gross_return': gross.to_numpy()})


def summarize_intervals(intervals, fee=.0025, equity_weight=.95, hit_rate_column='profitable_hit_rate'):
    result = {'status': 'complete_price_research', 'months': len(intervals),
              'fee_per_side': fee, 'equity_weight': equity_weight,
              'missing_selected_outcomes': int(intervals.missing.sum()),
              'total_return': None, 'cagr': None, 'max_drawdown_monthly': None,
              'profitable_hit_rate': None, 'median_hit_rate': None,
              'monthly_win_vs_pool': None, 'benchmark_total_return': None}
    if intervals.empty or intervals.selected.eq(0).any():
        result['status'] = 'insufficient_signals'
        return result
    if intervals.missing.gt(0).any() or intervals.gross_return.isna().any():
        result['status'] = 'incomplete_selected_returns'
        return result
    # 每期完整卖出并重新等权买入，费用对应真实买卖金额；5%现金不计利息。
    gross_factor = 1 + intervals.gross_return.to_numpy(float)
    factors = 1 - equity_weight + equity_weight * gross_factor * (1 - fee) / (1 + fee)
    wealth = np.r_[1., factors.cumprod()]
    days = (intervals.exit_date.iloc[-1] - intervals.entry_date.iloc[0]).days
    result.update(total_return=float(wealth[-1] - 1),
                  cagr=float(wealth[-1] ** (365.25 / days) - 1),
                  max_drawdown_monthly=float((wealth / np.maximum.accumulate(wealth) - 1).min()),
                  profitable_hit_rate=float(intervals[hit_rate_column].mean()),
                  start_date=str(intervals.entry_date.iloc[0].date()),
                  end_date=str(intervals.exit_date.iloc[-1].date()))
    if intervals.benchmark_gross_return.notna().all():
        benchmark = (1 - equity_weight + equity_weight * (1 + intervals.benchmark_gross_return)
                     * (1 - fee) / (1 + fee))
        result.update(benchmark_total_return=float(benchmark.prod() - 1),
                      monthly_win_vs_pool=float((factors > benchmark.to_numpy()).mean()),
                      median_hit_rate=float(intervals.median_hit_rate.mean()))
    return result


def prediction_accuracy(predictions, as_of):
    results = {}
    for horizon, frame in predictions.groupby('horizon', sort=True):
        matured = pd.to_datetime(frame.label_end).le(pd.Timestamp(as_of))
        usable = (matured & frame.probability_status.eq('ok') & frame.fwd_return.notna()
                  & np.isfinite(frame.probability_up) & np.isfinite(frame.fwd_return))
        observed = frame.loc[usable]
        count = len(observed)
        results[str(horizon)] = {'samples': count, 'total_predictions': len(frame),
                                'direction_accuracy': (float(((observed.probability_up >= .5)
                                                              == (observed.fwd_return > 0)).mean()) if count else None),
                                'actual_up_ratio': float(observed.fwd_return.gt(0).mean()) if count else None}
    return results


def common_signal_universe(candidates, minimum_amount):
    pools = []
    for frame in candidates:
        usable = frame.horizon.eq(20) & frame.score_status.eq('ok') & frame.adv20_amount.ge(minimum_amount)
        pools.append(pd.MultiIndex.from_frame(frame.loc[usable, ['date', 'security_id']]))
    common = pools[0]
    for pool in pools[1:]:
        common = common.intersection(pool)
    return common


def prepare_research(source_root, output):
    source_root, output = Path(source_root), Path(output)
    collection = json.loads((source_root / 'collection_status.json').read_text(encoding='utf-8'))
    if collection['status'] != 'complete':
        raise ValueError('行情采集尚未完成，不能开始本轮回测')
    output.mkdir(parents=True, exist_ok=True)
    calendar_frame = pd.read_parquet(source_root / 'references/calendar.parquet')
    calendar = pd.DatetimeIndex(calendar_frame.loc[calendar_frame.is_open.eq(1), 'cal_date']).sort_values()
    calendar = calendar[(calendar >= pd.Timestamp(collection['requested_start']))
                        & (calendar <= pd.Timestamp(collection['requested_end']))]
    basic = pd.read_parquet(source_root / 'references/hk_basic.parquet')
    reference = source_root / 'references/reit_research'
    hkex = pd.read_csv(reference / 'equities_and_reits.csv', dtype={'Stock Code': str})
    source_date = json.loads((reference / 'sources.json').read_text(encoding='utf-8'))['source_file_updated_at']
    partitions, audits = [], []
    for quote_file in sorted((source_root / 'source/hk_daily_adj').glob('*.parquet')):
        bars, audit = join_market_sources(pd.read_parquet(quote_file),
                                         pd.read_parquet(source_root / 'source/hk_adjfactor' / quote_file.name))
        partitions.append(bars.loc[bars.date.isin(calendar)])
        audits.append({'month': quote_file.stem, **audit})
    bars = pd.concat(partitions, ignore_index=True)
    del partitions
    observed = observed_quote_periods(bars)
    master = security_master(basic, observed, hkex, source_date)
    master.to_parquet(output / 'securities.parquet', index=False)
    metadata = master.set_index('security_id')
    bars['fx_to_hkd'] = np.where(bars.security_id.map(metadata.currency).eq('HKD'), 1., np.nan)
    bars['adj_close_hkd'] = bars.adj_close * bars.fx_to_hkd
    bars['amount_hkd'] = bars.amount * bars.fx_to_hkd
    features_dir = output / 'features'
    features_dir.mkdir(exist_ok=True)
    rows = eligible = 0
    paths = []
    for number, (security, group) in enumerate(bars.groupby('security_id', sort=True), 1):
        identity = metadata.loc[security].copy()
        identity['security_id'] = security
        frame = identity_period_features(group, calendar, identity)
        if identity.asset_type != 'equity' or identity.currency != 'HKD':
            frame['status'] = 'outside_price_research_scope'
        paths.append(write_security_features(frame, features_dir))
        rows += len(frame)
        eligible += int(frame.status.eq('ok').sum())
        if number % 250 == 0:
            print(f'research features {number}/{len(metadata)} securities', flush=True)
    price_inputs = pd.concat([pd.read_parquet(path, columns=['date', 'security_id', 'return_1', 'bias_60', 'status'])
                              for path in paths], ignore_index=True)
    price_inputs.loc[price_inputs.status.eq('outside_price_research_scope'), ['return_1', 'bias_60']] = np.nan
    price_inputs = price_inputs.merge(bars[['date', 'security_id', 'quote_present']],
                                     on=['date', 'security_id'], how='left', validate='one_to_one')
    market = market_context_from_inputs(price_inputs, calendar)
    del price_inputs, bars
    normalized_dir = output / 'normalized'
    sampled_dir = output / 'training_samples'
    normalized_dir.mkdir(exist_ok=True)
    sampled_dir.mkdir(exist_ok=True)
    feature_columns = None
    for year in sorted(set(calendar.year)):
        frame = pd.read_parquet(features_dir, filters=[('date', '>=', pd.Timestamp(year, 1, 1)),
                                                       ('date', '<', pd.Timestamp(year + 1, 1, 1))])
        frame = frame.merge(market, on='date', validate='many_to_one')
        excluded = {'date', 'security_id', 'raw_close', 'fx_to_hkd', 'adj_close_hkd', 'adv20_amount', 'status'}
        feature_columns = [c for c in frame if c not in excluded and not c.startswith(('fwd_return_', 'label_end_'))]
        frame[feature_columns] = frame[feature_columns].astype(np.float32)
        normalized = cross_sectional_inputs(frame, feature_columns)
        normalized.to_parquet(normalized_dir / f'{year}.parquet', index=False)
        sample_observations(normalized.loc[normalized.status.eq('ok')]).to_parquet(sampled_dir / f'{year}.parquet', index=False)
        print(f'research normalized {year}: {len(frame)} rows', flush=True)
    manifest = {'status': 'complete', 'research_only': True, 'start_date': str(calendar[0].date()),
                'data_as_of': str(calendar[-1].date()), 'rows': rows, 'eligible_rows': eligible,
                'features': feature_columns, 'securities': len(master), 'source_month_audits': audits,
                'scope': 'HKD provider equity identities within verified listing periods; price, volume, capitalization and cross-sectional market factors',
                'return_basis': FORECAST_RETURN_BASIS,
                'limitations': ['Does not establish full-market historical identity or terminal-return coverage.',
                                'Does not include financial statement, offer, index or HIBOR factors, or REIT models.',
                                'Does not provide lot, participation, cash dividend or executable-account validation.']}
    write_json(output / 'research_manifest.json', manifest)
    calendar_frame.to_parquet(output / 'calendar.parquet', index=False)
    return manifest


def run_research(data_root, results_root, model_root, start_year=2016, top_n=30, minimum_amount=1_000_000.):
    data_root, results_root, model_root = map(Path, (data_root, results_root, model_root))
    manifest = json.loads((data_root / 'research_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('研究因子尚未完成')
    results_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(manifest['data_as_of'])
    calendar_frame = pd.read_parquet(data_root / 'calendar.parquet')
    calendar = pd.DatetimeIndex(calendar_frame.loc[calendar_frame.is_open.eq(1), 'cal_date']).sort_values()
    calendar = calendar[calendar <= as_of]
    month_ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period('M')).max()
    # 未结束的当月不计作月末信号。
    signals = pd.DatetimeIndex(month_ends.loc[month_ends.index < as_of.to_period('M')])
    signals = signals[signals.year >= start_year]
    if len(signals) < 2:
        raise ValueError('至少需要两个已结束月份的信号')
    protocol = {'model_revision': MODEL_REVISION, 'research_only': True, 'kinds': list(KINDS),
                'data_as_of': str(as_of.date()), 'start_year': start_year,
                'training': 'Annual expanding-window refit; previous 12 months reserved for calibration; only matured labels enter training.',
                'signals': 'Completed calendar month ends, four prediction horizons; next trading session closing price for entry and exit.',
                'selection': {'horizon': 20, 'top_n': top_n, 'minimum_signal_adv20_hkd': minimum_amount,
                              'pool': 'Same signal-date score-available universe across all four candidates; future outcomes never determine membership.'},
                'portfolio': {'equity_weight': .95, 'cash_weight': .05, 'fee_per_side': .0025,
                              'stress_fee_per_side': .005, 'rebalance': 'liquidate fully and repurchase equal weights every month'},
                'evaluation_periods': {'development': '2016-2023', 'confirmation': '2024 onward'},
                'model_selection': 'All four predefined candidates reported separately; no selection by confirmation results.',
                'return_basis': FORECAST_RETURN_BASIS, 'limitations': manifest['limitations']}
    protocol_path = results_root / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('结果目录已有不同研究方案，请使用新的结果目录')
    write_json(protocol_path, protocol)
    samples = pd.concat([pd.read_parquet(p) for p in sorted((data_root / 'training_samples').glob('*.parquet'))], ignore_index=True)
    for year in sorted(set(signals.year)):
        training = samples.loc[samples.date.lt(pd.Timestamp(year, 1, 1))]
        cutoff = training.date.max()
        test = pd.read_parquet(data_root / 'normalized' / f'{year}.parquet')
        test = test.loc[test.date.isin(signals)]
        for kind in KINDS:
            prediction_path = results_root / kind / f'{year}.parquet'
            if prediction_path.exists():
                continue
            write_json(results_root / 'run_status.json', {'status': 'training', 'kind': kind, 'year': int(year)})
            print(f'research training {kind} for {year}, as_of={cutoff.date()}', flush=True)
            version = f'{MODEL_REVISION}-price-research-{kind}-{year}'
            model = fit_at(training, kind, cutoff, version, manifest['features'])
            predictions = predict_frame(model, test)
            predictions = predictions.merge(test[['date', 'security_id', 'adv20_amount']],
                                            on=['date', 'security_id'], how='left', validate='many_to_one')
            prediction_path.parent.mkdir(exist_ok=True)
            predictions.to_parquet(prediction_path, index=False)
            with (model_root / f'{kind}-{year}.pkl').open('wb') as stream:
                pickle.dump(model, stream)
            print(f'research completed {kind} {year}: {len(predictions)} forecasts', flush=True)
            del model, predictions
    del samples
    features = pd.read_parquet(data_root / 'features', columns=['date', 'security_id', 'adj_close_hkd'])
    prices = features.pivot(index='date', columns='security_id', values='adj_close_hkd')
    candidates = [pd.concat([pd.read_parquet(p, columns=['date', 'security_id', 'horizon', 'score_status', 'adv20_amount'])
                             for p in sorted((results_root / kind).glob('*.parquet'))], ignore_index=True)
                  for kind in KINDS]
    common = common_signal_universe(candidates, minimum_amount)
    del candidates
    results = {}
    for kind in KINDS:
        predictions = pd.concat([pd.read_parquet(p) for p in sorted((results_root / kind).glob('*.parquet'))], ignore_index=True)
        ranking = predictions.loc[predictions.horizon.eq(20)]
        ranking = ranking.loc[pd.MultiIndex.from_frame(ranking[['date', 'security_id']]).isin(common)]
        intervals, selections = [], []
        for signal, following in zip(signals[:-1], signals[1:]):
            pool = ranking.loc[ranking.date.eq(signal) & ranking.score_status.eq('ok')
                               & ranking.adv20_amount.ge(minimum_amount)]
            selected = pool.sort_values(['score', 'security_id'], ascending=[False, True]).head(top_n)
            realized = interval_returns(prices, calendar, signal, following, pool.security_id.tolist())
            held = selected[['security_id', 'score']].merge(realized, on='security_id', validate='one_to_one')
            held['signal_date'] = signal
            held['net_return'] = (1 + held.gross_return) * .9975 / 1.0025 - 1
            held['stress_net_return'] = (1 + held.gross_return) * .995 / 1.005 - 1
            selections.append(held)
            known = held.gross_return.notna().all() and not held.empty
            benchmark_known = realized.gross_return.notna().all() and not realized.empty
            entry = calendar[calendar.get_loc(signal) + 1]
            exit_day = calendar[calendar.get_loc(following) + 1]
            intervals.append({'signal_date': signal, 'entry_date': entry, 'exit_date': exit_day,
                              'selected': len(held), 'pool_stocks': len(pool),
                              'missing': int(held.gross_return.isna().sum()),
                              'benchmark_missing': int(realized.gross_return.isna().sum()),
                              'gross_return': float(held.gross_return.mean()) if known else np.nan,
                              'benchmark_gross_return': float(realized.gross_return.mean()) if benchmark_known else np.nan,
                              'profitable_hit_rate': float(held.net_return.gt(0).mean()) if known else np.nan,
                              'stress_profitable_hit_rate': float(held.stress_net_return.gt(0).mean()) if known else np.nan,
                              'median_hit_rate': float(held.gross_return.gt(realized.gross_return.median()).mean())
                              if known and benchmark_known else np.nan})
        monthly = pd.DataFrame(intervals)
        monthly.to_csv(results_root / kind / 'monthly_returns.csv', index=False)
        pd.concat(selections, ignore_index=True).to_csv(results_root / kind / 'selections.csv', index=False)
        results[kind] = {}
        for period, start, end in [('all', pd.Timestamp(start_year, 1, 1), as_of),
                                   ('development', pd.Timestamp('2016-01-01'), pd.Timestamp('2023-12-31')),
                                   ('confirmation', pd.Timestamp('2024-01-01'), as_of)]:
            months = monthly.loc[monthly.signal_date.ge(start) & monthly.exit_date.le(end)]
            forecasts = predictions.loc[predictions.date.between(start, end)]
            results[kind][period] = {'portfolio': summarize_intervals(months),
                                     'stress_portfolio': summarize_intervals(months, fee=.005,
                                                                              hit_rate_column='stress_profitable_hit_rate'),
                                     'direction': prediction_accuracy(forecasts, end)}
        write_json(results_root / kind / 'metrics.json', results[kind])
    write_json(results_root / 'metrics.json', {'protocol': protocol, 'results': results})
    write_json(results_root / 'run_status.json', {'status': 'complete', 'research_only': True,
                                                'completed_at': datetime.now().astimezone().isoformat()})
    return results


def main():
    parser = argparse.ArgumentParser(description='运行新版四种模型的量价研究回测')
    subparsers = parser.add_subparsers(dest='stage', required=True)
    prepare = subparsers.add_parser('prepare')
    prepare.add_argument('--source-root', type=Path, required=True)
    prepare.add_argument('--data-root', type=Path, required=True)
    run = subparsers.add_parser('run')
    run.add_argument('--data-root', type=Path, required=True)
    run.add_argument('--results-root', type=Path, required=True)
    run.add_argument('--model-root', type=Path, required=True)
    run.add_argument('--start-year', type=int, default=2016)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare_research(args.source_root, args.data_root)
    else:
        run_research(args.data_root, args.results_root, args.model_root, args.start_year)


if __name__ == '__main__':
    main()
