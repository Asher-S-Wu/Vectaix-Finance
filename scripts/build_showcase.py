"""从冻结模型与已完成的回测生成 README 图表；不训练、不发单。"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL, BLUE, GOLD, RED = '#172C3C', '#60717C', '#00866F', '#467CB8', '#BC8B2D', '#CE6651'
PAPER, GRID = '#FAFAF6', '#E5E9E7'
MODEL_NAMES = {'factor': 'Factor score', 'linear': 'Linear',
               'lightgbm_small': 'Small LightGBM', 'lightgbm_large': 'Large LightGBM'}
FEATURE_NAMES = {
    'near_low_252': 'Distance from 252-day low', 'bias_60': '60-day price deviation',
    'volatility_252': '252-day volatility', 'near_high_252': 'Distance from 252-day high',
    'volatility_60': '60-day volatility', 'bias_20': '20-day price deviation',
    'log_amount_20': '20-day trading value', 'log_market_cap': 'Market capitalization',
    'market_breadth_60': '60-day market breadth', 'momentum_60': '60-day momentum',
}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.parse_math': False,
                         'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.facecolor': PAPER, 'figure.facecolor': PAPER,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'savefig.facecolor': PAPER})


def header(fig, title, subtitle):
    fig.text(.065, .955, 'VECTAIX FINANCE  /  HISTORICAL BACKTEST', fontsize=10, color=TEAL, weight='bold')
    fig.text(.065, .899, title, fontsize=23, weight='bold')
    fig.text(.065, .86, subtitle, fontsize=10.5, color=MUTED)


def save(fig, destination, footer):
    fig.text(.065, .032, footer, fontsize=9, color=MUTED)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def line_panel(ax):
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)


def read_case(results, forecast_root, data_root):
    folder = results / 'lightgbm_small/normal'
    trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
    first = trades.loc[trades.side.eq('buy'), 'date'].min()
    entries = trades.loc[trades.date.eq(first) & trades.side.eq('buy')]
    positions = pd.read_csv(folder / 'completed.csv', parse_dates=['entry_date', 'exit_date'])
    positions = positions.loc[positions.position_id.isin(entries.position_id)].sort_values('position_id')
    assert set(positions.position_id) == set(entries.position_id), '第一期必须全部结束'
    fills = trades.loc[trades.position_id.isin(entries.position_id)].copy()
    holdings = pd.read_csv(folder / 'holdings.csv', parse_dates=['date'])
    holdings = holdings.loc[holdings.date.eq(first)].copy()
    daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date']).set_index('date')
    predictions = pd.read_parquet(forecast_root / 'lightgbm_small/predictions.parquet')
    signal = predictions.loc[predictions.date.lt(first), 'date'].max()
    next_signal = predictions.loc[predictions.date.gt(signal), 'date'].min()
    calendar = pd.read_parquet(data_root / 'calendar.parquet')
    scheduled_exit = calendar.loc[calendar.is_open.eq(1) & calendar.cal_date.gt(next_signal), 'cal_date'].min()
    selection = predictions.loc[predictions.date.eq(signal) & predictions.horizon.eq(20)
                                & predictions.security_id.isin(entries.security_id),
                                ['date', 'security_id', 'score', 'adv20_amount']].copy()
    selection = selection.sort_values(['score', 'security_id'], ascending=[False, True])
    selection.insert(0, 'rank', np.arange(1, len(selection) + 1))
    names = pd.read_parquet(data_root / 'securities.parquet').set_index('security_id')['name']
    for frame in (selection, positions, fills, holdings):
        frame.insert(1, 'name', frame.security_id.map(names))
    initial_cash = json.loads((folder / 'metrics.json').read_text())['initial_cash']
    signed = np.where(fills.side.eq('buy'), -fills.value - fills.fee, fills.value - fills.fee)
    fills['cohort_cash_after'] = initial_cash + pd.Series(signed, index=fills.index).cumsum()
    fills = fills.drop(columns=['cash_after', 'day_amount']).rename(
        columns={'units': 'adjusted_units', 'price': 'adjusted_price', 'value': 'notional_hkd', 'fee': 'fee_hkd'})
    holdings = holdings.rename(columns={'units': 'adjusted_units', 'mark': 'adjusted_price', 'value': 'value_hkd'})
    holdings['account_weight'] = holdings.value_hkd / daily.loc[first, 'equity']
    pnl = float((positions.proceeds - positions.cost).sum())
    case = dict(selection_rule='First entry cohort in the completed backtest, not selected by profit.',
                signal_date=str(signal.date()), entry_date=str(first.date()),
                scheduled_exit_date=str(scheduled_exit.date()), final_exit_date=str(positions.exit_date.max().date()),
                positions=len(positions), fills=len(fills), winners=int(positions.net_return.gt(0).sum()),
                entry_cost_hkd=float(positions.cost.sum()), net_proceeds_hkd=float(positions.proceeds.sum()),
                net_pnl_hkd=pnl, return_on_entry_cost=pnl / positions.cost.sum(),
                contribution_to_initial_capital=pnl / initial_cash,
                total_fees_hkd=float(fills.fee_hkd.sum()), initial_cash_hkd=initial_cash,
                entry_cash_hkd=float(daily.loc[first, 'cash']), entry_equity_hkd=float(daily.loc[first, 'equity']),
                cohort_final_cash_hkd=float(fills.cohort_cash_after.iloc[-1]),
                accounting='Only the first entry cohort; subsequent rebalances excluded from cohort cash.',
                unit_basis='Fractional source-adjusted research units, not raw shares or board-lot fills.')
    assert np.isclose(case['cohort_final_cash_hkd'], initial_cash + pnl)
    assert np.isclose(holdings.value_hkd.sum() + case['entry_cash_hkd'], case['entry_equity_hkd'])
    return case, selection, positions, fills, holdings


def performance_chart(curve, summary, destination):
    small = summary.loc[summary.model.eq('lightgbm_small') & summary.scenario.eq('normal')].iloc[0]
    fig = plt.figure(figsize=(14, 9))
    header(fig, f'HK${small.initial_cash / 1e6:g}m becomes HK${small.ending_equity / 1e6:.2f}m',
           f'Frozen small LightGBM  |  {small.start_date} to {small.end_date}  |  Net of simulated fees')
    for x, label, value in [(.065, 'ANNUALIZED RETURN', f'{small.cagr:.2%}'),
                            (.32, 'TOTAL RETURN', f'+{small.total_return:.2%}'),
                            (.565, 'MAXIMUM DRAWDOWN', f'{small.max_drawdown_daily:.2%}'),
                            (.79, 'PROFITABLE POSITIONS', f'{small.profitable_hit_rate:.2%}')]:
        fig.text(x, .785, value, fontsize=25, weight='bold', color=TEAL if x < .5 else INK)
        fig.text(x, .751, label, fontsize=9, color=MUTED)
    ax = fig.add_axes([.08, .26, .77, .41])
    for key, label, color, dash in [('lightgbm_small', 'Small model', TEAL, '-'),
                                   ('HSI', 'Hang Seng', BLUE, '--'), ('HKTECH', 'Hang Seng Tech', GOLD, ':')]:
        y = curve[key]
        ax.plot(curve.date, y, lw=2.2, color=color, linestyle=dash, label=label)
        ax.annotate(f'{label}\n{y.iloc[-1]:.2f}x', (curve.date.iloc[-1], y.iloc[-1]),
                    xytext=(10, 0), textcoords='offset points', va='center', color=color, fontsize=10)
    line_panel(ax)
    ax.set_ylabel('Wealth / initial capital')
    ax.set_xlim(curve.date.iloc[0], curve.date.iloc[-1])
    ax.tick_params(labelbottom=False)
    ax.legend(loc='upper left', frameon=False, ncol=3, fontsize=10)
    dd = fig.add_axes([.08, .105, .77, .12], sharex=ax)
    y = curve.lightgbm_small / curve.lightgbm_small.cummax().clip(lower=1.) - 1
    dd.fill_between(curve.date, y, 0, color=TEAL, alpha=.16)
    dd.plot(curve.date, y, color=TEAL, lw=1)
    dd.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    dd.set_ylabel('Drawdown')
    dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    dd.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    line_panel(dd)
    save(fig, destination, 'Source: Tushare + frozen-model replay. Benchmarks are price indices (no dividends or fees). Ending value includes unsold positions.')


def factor_chart(importance, trained_as_of, destination):
    top = importance.nlargest(10, 'gain').iloc[::-1]
    fig = plt.figure(figsize=(14, 7.7))
    header(fig, 'What the ranking model uses', f'Top 10 inputs by training split gain  |  Model trained through {trained_as_of}')
    ax = fig.add_axes([.31, .14, .59, .64])
    labels = [FEATURE_NAMES[name] for name in top.feature]
    ax.barh(labels, top.gain_share, color=[TEAL if v > .1 else BLUE for v in top.gain_share], height=.6)
    for i, share in enumerate(top.gain_share):
        ax.text(share + .002, i, f'{share:.1%}', va='center', fontsize=11)
    ax.set_xlim(0, .19)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Share of split gain across all model inputs')
    ax.grid(axis='x', color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=9)
    save(fig, destination, 'Source: saved small LightGBM ranker. Gain describes model usage; it is not an isolated factor return or a causal attribution of profit.')


def candle_chart(source, selection, positions, fills, case, destination):
    first_two = selection.head(2).security_id.tolist()
    start = pd.Timestamp(case['signal_date']) - pd.Timedelta(days=30)
    end = pd.Timestamp(case['final_exit_date'])
    parts = [pd.read_parquet(source / 'source/hk_daily_adj' / (month.strftime('%Y%m') + '.parquet'))
             for month in pd.period_range(start, end, freq='M')]
    bars = pd.concat(parts, ignore_index=True)
    bars = bars.loc[bars.ts_code.isin(first_two) & bars.trade_date.between(start, end)
                    & bars.vol.gt(0) & bars.amount.gt(0)]
    fig = plt.figure(figsize=(14, 10))
    header(fig, 'Two selections. Two different outcomes.',
           f'Signal: {case["signal_date"]}  |  First two selected stocks by score  |  Simulated buy / sell markers')
    for n, sid in enumerate(first_two):
        ax = fig.add_axes([.085, .495 - n * .36, .82, .255])
        q = bars.loc[bars.ts_code.eq(sid)].sort_values('trade_date')
        xs = mdates.date2num(q.trade_date)
        for x, row in zip(xs, q.itertuples()):
            color = TEAL if row.close >= row.open else RED
            ax.vlines(x, row.low, row.high, color=color, lw=1)
            bottom, height = min(row.open, row.close), abs(row.open - row.close)
            if height == 0:
                ax.hlines(row.close, x - .32, x + .32, color=color, lw=1.2)
            else:
                ax.add_patch(Rectangle((x - .32, bottom), .64, height, facecolor=color, edgecolor=color))
        trade = fills.loc[fills.security_id.eq(sid)]
        for side, marker, color in [('buy', '^', BLUE), ('sell', 'v', GOLD)]:
            selected = trade.loc[trade.side.eq(side)]
            ax.scatter(selected.date, selected.adjusted_price, marker=marker, color=color,
                       s=110, edgecolors=PAPER, linewidths=1, zorder=5, label='Buy' if side == 'buy' else 'Sell')
        result = positions.set_index('security_id').loc[sid]
        rank = int(selection.set_index('security_id').loc[sid, 'rank'])
        ax.set_title(f'#{rank}  {sid}     |     Net position return {result.net_return:+.2%}',
                     loc='left', pad=14, fontsize=13, weight='bold')
        ax.axvline(pd.Timestamp(case['signal_date']), color=MUTED, linestyle=':', lw=1, label='Signal')
        ax.set_ylabel('Adjusted price')
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.legend(loc='upper right', ncol=3, frameon=False, fontsize=9)
        ax.margins(x=.025, y=.2)
        ax.set_xlim(start, end + pd.Timedelta(days=1))
        line_panel(ax)
    save(fig, destination, 'Historical backtest, not broker fills. Source-adjusted OHLC; trade units are fractional research units. Both gains and losses are shown.')


def holdings_chart(holdings, case, destination):
    h = holdings.assign(display_value=holdings.value_hkd.round(2)).sort_values(
        ['display_value', 'security_id'], ascending=[False, True])
    top = h.head(4)
    values = list(top.value_hkd) + [float(h.iloc[4:].value_hkd.sum()), case['entry_cash_hkd']]
    labels = list(top.security_id) + [f'Other {len(h) - 4} stocks', 'Cash']
    colors = [TEAL, BLUE, GOLD, RED, '#9FBCB8', '#DAE0E0']
    fig = plt.figure(figsize=(14, 7.5))
    header(fig, 'The first portfolio, including cash',
           f'{case["entry_date"]} close  |  Simulated fills after trading-value limits  |  {len(h)} positions')
    ax = fig.add_axes([.055, .115, .49, .66])
    ax.pie(values, colors=colors, startangle=90, counterclock=False,
           wedgeprops={'width': .30, 'edgecolor': PAPER, 'linewidth': 3})
    ax.text(0, .13, f'HK${case["entry_equity_hkd"]:,.0f}', ha='center', fontsize=21, weight='bold')
    ax.text(0, -.11, 'Account value after entry fees', ha='center', fontsize=10, color=MUTED)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = .72 - i * .085
        fig.text(.59, y, '●', color=color, fontsize=18)
        fig.text(.62, y, label, fontsize=12)
        fig.text(.905, y, f'{value / case["entry_equity_hkd"]:.2%}', ha='right', fontsize=12, weight='bold')
    save(fig, destination, 'Weights use total account value, including cash. Top four ties are ordered by stock code; all 30 positions are listed in the linked CSV.')


def outcome_chart(positions, case, destination):
    p = positions.assign(pnl=positions.proceeds - positions.cost).sort_values('pnl')
    fig = plt.figure(figsize=(14, 10))
    header(fig, f'The complete first cohort: HK${case["net_pnl_hkd"]:+,.0f}',
           f'{case["positions"]} positions  |  {case["winners"]} profitable  |  {case["fills"]} fills'
           f'  |  HK${case["total_fees_hkd"]:,.0f} total fees  |  Final exit: {case["final_exit_date"]}')
    ax = fig.add_axes([.14, .1, .72, .7])
    ax.barh(p.security_id, p.pnl, height=.66, color=np.where(p.pnl.ge(0), TEAL, RED))
    ax.axvline(0, color=MUTED, lw=.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:,.0f}k'))
    ax.set_xlabel('Net profit / loss per completed position (HKD)')
    ax.grid(axis='x', color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    save(fig, destination, 'First entry cohort only; later rebalances excluded. Net P&L includes both-side fees and delayed exits. This is historical simulation.')


def build(results, forecasts, source, data, destination):
    style()
    evidence, assets = destination / 'showcase', destination / 'assets'
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    assert json.loads((results / 'run_status.json').read_text())['status'] == 'complete'
    summary = pd.read_csv(results / 'summary.csv')
    protocol = json.loads((forecasts / 'protocol.json').read_text())
    model_path = Path(protocol['models']['lightgbm_small']['path'])
    before = (model_path.stat().st_size, model_path.stat().st_mtime_ns)
    with model_path.open('rb') as stream:
        model = pickle.load(stream)
    gain = model.ranker.booster_.feature_importance(importance_type='gain')
    importance = pd.DataFrame({'feature': model.ranker.booster_.feature_name(), 'gain': gain,
                              'gain_share': gain / gain.sum()})
    case, selection, positions, fills, holdings = read_case(results, forecasts, data)
    for name, frame in [('model_comparison', summary), ('factor_importance', importance),
                         ('first_cycle_selection', selection), ('first_cycle_positions', positions),
                         ('first_cycle_trades', fills), ('first_cycle_holdings', holdings)]:
        frame.to_csv(evidence / f'{name}.csv', index=False, float_format='%.10g')
    normal = summary.loc[summary.scenario.eq('normal')].set_index('model')
    start, end = pd.Timestamp(normal.start_date.iloc[0]), pd.Timestamp(normal.end_date.iloc[0])
    curve = pd.DataFrame()
    for kind in MODEL_NAMES:
        daily = pd.read_csv(results / kind / 'normal/daily.csv', parse_dates=['date']).set_index('date')
        curve[kind] = daily.loc[start:end, 'equity'] / normal.loc[kind, 'initial_cash']
    for code in ('HSI', 'HKTECH'):
        index = pd.read_csv(source / 'references/benchmarks_20260916' / f'{code}.csv', dtype={'trade_date': str})
        index['date'] = pd.to_datetime(index.trade_date, format='%Y%m%d')
        price = index.set_index('date').close
        curve[code] = price.reindex(curve.index) / price.loc[start]
    assert curve.notna().all().all()
    curve = curve.rename_axis('date').reset_index()
    curve.to_csv(evidence / 'performance_curve.csv', index=False, float_format='%.10g')
    benchmark = json.loads((results / 'benchmark_comparison.json').read_text())['benchmarks']
    public = dict(status='historical_backtest', live_trading=False, training_performed=False,
                  model='lightgbm_small', model_version=model.model_version,
                  trained_as_of=str(pd.Timestamp(model.as_of).date()), feature_count=len(model.feature_columns),
                  start_date=str(start.date()), end_date=str(end.date()), case=case, benchmarks=benchmark,
                  sources={'market_api': ['hk_daily_adj', 'hk_adjfactor'], 'index_api': 'index_global',
                           'provider': 'Tushare relay', 'results': str(results.relative_to(ROOT)).replace('\\', '/')},
                  methodology={'signal': 'Month-end 20-session score, common score-available universe, top 30.',
                               'fees_per_side': .0025, 'participation_cap': .01,
                               'initial_cash_hkd': 1000000, 'budget': '95% of available cash, 30 fixed slots.',
                               'execution': 'Next-session adjusted close, positive volume required; delayed sales retain positions.'},
                  limitations=['Not live trading or a broker statement.',
                               'Synthetic adjusted units; historical board lots and exact dividend payment timing are not reconstructed.',
                               'Benchmarks are price indices excluding dividends and fees.',
                               'End value includes unsold positions; reference marks may not be executable.',
                               'Candidates have been compared using this history; it is not a pristine holdout.',
                               'Feature gain measures training usage, not independent factor performance.'])
    (evidence / 'summary.json').write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    performance_chart(curve, summary, assets / 'performance.png')
    factor_chart(importance, public['trained_as_of'], assets / 'factor_importance.png')
    candle_chart(source, selection, positions, fills, case, assets / 'trade_candles.png')
    holdings_chart(holdings, case, assets / 'holdings.png')
    outcome_chart(positions, case, assets / 'first_cycle_pnl.png')
    assert (model_path.stat().st_size, model_path.stat().st_mtime_ns) == before
    print(json.dumps({'charts': 5, 'cohort_positions': len(positions), 'cohort_fills': len(fills),
                      'cohort_net_pnl_hkd': case['net_pnl_hkd'], 'training_performed': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'backtests/hk/fixed_execution_20260916')
    parser.add_argument('--forecasts', type=Path, default=ROOT / 'backtests/hk/fixed_models_20260916')
    parser.add_argument('--source', type=Path, default=ROOT / 'data/hk/universal')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/hk/research_20260916')
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs')
    args = parser.parse_args()
    build(args.results, args.forecasts, args.source, args.data, args.destination)
