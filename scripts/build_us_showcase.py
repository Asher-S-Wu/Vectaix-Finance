"""Render independent US research charts from saved aggregate evidence only.

No vendor inputs, pickle loading, fitting, tuning, or replay is performed here.
All output values and source hashes come from the completed frozen evaluation.
"""
import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = 'backtests/us/oef2015'
SOURCE_FILES = tuple(f'{RESULTS}/{name}.json' for name in (
    'protocol', 'frozen_architecture', 'development_common_universe',
    'confirmation_summary', 'source_audit', 'training_status', 'latest_refit_summary'))
MODEL_NAMES = {'factor': 'Factor score', 'linear': 'Ridge',
               'lightgbm_small': 'Small LightGBM', 'lightgbm_large': 'Large LightGBM'}
METRICS = ('rank_ic_mean', 'ic_ci_lower', 'ic_ci_upper', 'ic_dates', 'ic_calendar_slots',
           'predictions', 'mature_labels', 'missing_or_immature_outcomes', 'score_available',
           'probability_available', 'brier', 'baseline_brier', 'brier_skill',
           'interval_available', 'interval_covered', 'interval_coverage')
INK, MUTED, TEAL, BLUE, RED = '#172C3C', '#60717C', '#00866F', '#467CB8', '#CE6651'
PAPER, GRID = '#FAFAF6', '#E5E9E7'
REPLAY_NAMES = {'selected_15bps': 'Factor score / 15 bps',
                'selected_30bps': 'Factor score / 30 bps stress',
                'cohort_101slots_15bps': '101-slot cohort / 15 bps',
                'spy_15bps': 'SPY adjusted proxy / 15 bps'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(root=ROOT):
    """Refuse unfinished training, inconsistent selection, or stale evidence."""
    root = Path(root)
    protocol, frozen, common, confirmation, source, status, latest = [
        json.loads((root/name).read_text()) for name in SOURCE_FILES]
    if frozen['selection_set'] != 'development' or frozen['confirmation_used'] is not False:
        raise ValueError('Charts require development-only model selection')
    if status['status'] != 'complete' or status['confirmation_used_for_selection'] is not False:
        raise ValueError('Charts require complete training with independent confirmation')
    if latest['out_of_sample_evaluated'] is not False:
        raise ValueError('The separate latest refit must remain labelled unevaluated')
    selected = frozen['selected_kind']
    if set(common) != set(MODEL_NAMES) or selected not in MODEL_NAMES:
        raise ValueError('All four declared candidate families are required')
    common_hash = hashlib.sha256(json.dumps(common, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if common_hash != frozen['development_sha256']:
        raise ValueError('Development common-universe metrics do not match the freeze hash')
    samples = {(r['ic_dates'], r['date_sha256'], r['common_score_rows'], r['common_mature_rows'])
               for r in common.values()}
    if len(samples) != 1:
        raise ValueError('Candidate comparison must use the same rows and dates')
    if not all(math.isfinite(row['rank_ic_mean']) for row in common.values()):
        raise ValueError('Non-finite candidate ranking metric')
    if not math.isclose(frozen['selection_statistic'], common[selected]['rank_ic_mean'], abs_tol=1e-12):
        raise ValueError('Selection statistic does not match its saved candidate metric')
    winner = max(MODEL_NAMES, key=lambda kind: common[kind]['rank_ic_mean'])
    if selected != winner:
        raise ValueError('Selected model does not follow maximum IC with simple-first ties')
    if source['available_original_identities'] + source['unavailable_original_identities'] != source['declared_cohort']:
        raise ValueError('Source cohort counts must include every declared identity')
    if source['declared_cohort'] != protocol['declared_cohort_count'] or source['declared_cohort'] != 101:
        raise ValueError('Expected the fixed 101-share-class historical cohort')
    if (protocol['market'], protocol['currency']) != ('US', 'USD'):
        raise ValueError('US charts require independent USD model evidence')
    if (protocol['survivorship_free'] is not False or protocol['full_us_market'] is not False
            or source['cohort_survivorship_free'] is not False or source['full_us_market'] is not False):
        raise ValueError('Bounded-cohort limitations must remain explicit')
    if protocol['bootstrap']['block_sessions'] != 60 or protocol['bootstrap']['repetitions'] != 2000:
        raise ValueError('IC chart requires saved 60-session, 2000-draw bootstrap protocol')
    if confirmation['as_of'] != protocol['confirmation_end']:
        raise ValueError('Confirmation cutoff differs from the saved protocol')
    if status['selected_kind'] != selected or latest['kind'] != selected:
        raise ValueError('Completed training and latest refit must match the frozen architecture')
    if any(record['execution_validated'] is not False or record['eligible'] is not False
           for record in [status, latest]):
        raise ValueError('Execution validation and eligibility must remain false')
    selection = [dict(model=kind, label=MODEL_NAMES[kind], **common[kind]) for kind in MODEL_NAMES]
    rows = []
    for horizon in protocol['horizons']:
        actual = confirmation['horizons'][str(horizon)]
        row = dict(horizon=horizon, **{name: actual[name] for name in METRICS})
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in row.values()):
            raise ValueError(f'Non-finite confirmation metric at horizon {horizon}')
        if not -1 <= row['ic_ci_lower'] <= row['rank_ic_mean'] <= row['ic_ci_upper'] <= 1:
            raise ValueError('Saved IC estimate or confidence limits are inconsistent')
        if not 0 < row['ic_dates'] <= row['ic_calendar_slots']:
            raise ValueError('IC dates must be a nonempty subset of the calendar slots')
        if row['mature_labels'] + row['missing_or_immature_outcomes'] != row['predictions']:
            raise ValueError('Prediction outcome counts do not reconcile')
        if any(not 0 < row[field] <= row['mature_labels'] for field in
               ['score_available', 'probability_available', 'interval_available']):
            raise ValueError('Task denominators must be mature available outcomes')
        if not row['baseline_brier'] > 0 or not math.isclose(
                row['brier_skill'], 1-row['brier']/row['baseline_brier'], abs_tol=1e-12):
            raise ValueError('Brier skill does not match the saved scores')
        if not 0 <= row['interval_covered'] <= row['interval_available'] or not math.isclose(
                row['interval_coverage'], row['interval_covered']/row['interval_available'], abs_tol=1e-12):
            raise ValueError('Interval coverage does not match the saved counts')
        rows.append(row)
    model_path = f'models/us/oef2015/frozen/{selected}.json'
    model = json.loads((root/model_path).read_text())
    if (model['kind'], model['market'], model['currency']) != (selected, 'US', 'USD'):
        raise ValueError('Selected-model metadata belongs to a different model or market')
    for name in ['protocol', 'source_audit']:
        if model[name+'_sha256'] != digest(root/f'{RESULTS}/{name}.json'):
            raise ValueError(f'Selected model has a stale {name} hash')
    return dict(selected_model=selected, selected_label=MODEL_NAMES[selected], model_version=model['model_version'],
                development_start=protocol['development_start'], development_end=protocol['development_end'],
                confirmation_start=protocol['confirmation_start'], confirmation_end=confirmation['as_of'],
                universe_date=protocol['universe_date'], universe_known_by=protocol['universe_known_by'],
                declared_cohort=source['declared_cohort'],
                available_original_identities=source['available_original_identities'],
                unavailable_original_identities=source['unavailable_original_identities'],
                history_start=source['start'], history_end=source['end'],
                latest_model_version=latest['model_version'], latest_as_of=latest['as_of'],
                selection=selection, confirmation=rows,
                source_sha256={name: digest(root/name) for name in (*SOURCE_FILES, model_path)})


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'text.parse_math': False,
                        'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                        'ytick.color': INK, 'axes.facecolor': PAPER, 'figure.facecolor': PAPER,
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'axes.spines.left': False, 'axes.spines.bottom': False,
                        'savefig.facecolor': PAPER})


def frame(title, subtitle):
    fig = plt.figure(figsize=(14, 8))
    fig.text(.065, .95, 'VECTAIX FINANCE  /  US EXPERIMENT 1: INITIAL PUBLIC-DATA RUN', fontsize=11, color=TEAL, weight='bold')
    fig.text(.065, .885, title, fontsize=25, weight='bold')
    fig.text(.065, .837, subtitle, fontsize=12, color=MUTED)
    return fig


def finish(fig, destination, note, evidence, source):
    fig.text(.065, .122, note, fontsize=10.5, color=MUTED)
    fig.text(.065, .085,
             f'Fixed 2015 cohort: {evidence["available_original_identities"]}/{evidence["declared_cohort"]} original share classes have accepted histories; '
             f'{evidence["unavailable_original_identities"]} excluded or unavailable. Conditional observed outcomes.',
             fontsize=10.5, color=MUTED)
    fig.text(.065, .048, 'Source: '+source+'  |  Latest refit untested; execution not validated.', fontsize=9.5, color=MUTED)
    fig.savefig(destination, dpi=180, metadata={'Software': 'Vectaix Finance / build_us_showcase'})
    plt.close(fig)


def panel(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.45, -.55)
    ax.grid(axis='x', color=GRID, linewidth=.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=10)


def window(evidence):
    return f'Frozen {evidence["selected_label"]}  |  Confirmation: {evidence["confirmation_start"]} to {evidence["confirmation_end"]}'


def horizon_labels(rows):
    return [f'{row["horizon"]} session'+('s' if row['horizon'] != 1 else '') for row in rows]


def limits(values, margin=.25):
    low, high = min(0., min(values)), max(0., max(values))
    span = max(high-low, .001)
    return low-span*margin, high+span*margin


def confirmation_chart(evidence, destination):
    rows = evidence['confirmation']
    title = ('Ranking IC is negative at every horizon' if all(r['rank_ic_mean'] < 0 for r in rows)
             else 'Frozen-model ranking quality')
    fig = frame(title, window(evidence))
    ax = fig.add_axes([.16, .26, .60, .46])
    panel(ax, horizon_labels(rows))
    for i, row in enumerate(rows):
        mean, lower, upper = row['rank_ic_mean'], row['ic_ci_lower'], row['ic_ci_upper']
        color = TEAL if row['horizon'] == 20 else BLUE
        ax.errorbar(mean, i, xerr=[[mean-lower], [upper-mean]], fmt='o', color=color,
                    markersize=9, elinewidth=3, capsize=6, capthick=2)
        ax.text(mean, i-.23, f'{mean:+.4f}', ha='center', fontsize=14, weight='bold', color=color)
        ax.text(1.04, i, f'{row["ic_dates"]} IC dates', transform=ax.get_yaxis_transform(), va='center', fontsize=12)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(*limits([r[k] for r in rows for k in ['ic_ci_lower', 'ic_ci_upper']], .12))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel('Mean daily cross-sectional rank IC', labelpad=12)
    fig.text(.16, .765, 'Dots: mean IC     Lines: 95% confidence intervals', fontsize=12, color=MUTED)
    finish(fig, destination, '60-session circular block bootstrap, 2,000 draws; missing-day slots preserved. Mature outcomes only.',
           evidence, 'backtests/us/oef2015/confirmation_summary.json')


def selection_chart(evidence, destination):
    rows = sorted(evidence['selection'], key=lambda row: row['rank_ic_mean'], reverse=True)
    fig = frame('Development-set model comparison',
                f'{evidence["development_start"]} to {evidence["development_end"]}  |  20-session horizon  |  Common score-available universe')
    ax = fig.add_axes([.27, .26, .62, .46])
    panel(ax, [row['label']+('  (selected)' if row['model'] == evidence['selected_model'] else '') for row in rows])
    bounds = limits([r['rank_ic_mean'] for r in rows])
    offset = (bounds[1]-bounds[0])*.018
    for i, row in enumerate(rows):
        value = row['rank_ic_mean']
        color = TEAL if row['model'] == evidence['selected_model'] else BLUE
        ax.barh(i, value, height=.49, color=color)
        ax.text(value+(offset if value >= 0 else -offset), i, f'{value:+.4f}',
                ha='left' if value >= 0 else 'right', va='center', fontsize=14, weight='bold')
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(*bounds)
    ax.set_xlabel('20-session mean daily rank IC', labelpad=12)
    sample = rows[0]
    fig.text(.27, .765, f'{sample["ic_dates"]} shared IC dates  |  {sample["common_mature_rows"]:,} mature stock-date outcomes', fontsize=12, color=MUTED)
    finish(fig, destination, 'Four candidates share the same rows and dates. Only this development statistic selects the frozen architecture.',
           evidence, 'development_common_universe.json + frozen_architecture.json')


def probability_chart(evidence, destination):
    rows = evidence['confirmation']
    fig = frame('Probability skill against the historical baseline', window(evidence))
    ax = fig.add_axes([.16, .26, .70, .46])
    panel(ax, horizon_labels(rows))
    bounds = limits([r['brier_skill'] for r in rows], .3)
    offset = (bounds[1]-bounds[0])*.018
    for i, row in enumerate(rows):
        value = row['brier_skill']
        ax.barh(i, value, height=.49, color=RED if value < 0 else TEAL)
        ax.text(value+(offset if value >= 0 else -offset), i, f'{value:+.2%}',
                ha='left' if value >= 0 else 'right', va='center', fontsize=14, weight='bold')
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(*bounds)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_xlabel('Brier skill = 1 − model Brier / historical-frequency baseline Brier', labelpad=12)
    fig.text(.16, .765, 'Zero matches the baseline; negative values mean higher probability error', fontsize=12, color=MUTED)
    finish(fig, destination, 'Mature, probability-available outcomes only. Ranking IC and probability skill measure different tasks.',
           evidence, 'backtests/us/oef2015/confirmation_summary.json')


def coverage_chart(evidence, destination):
    rows = evidence['confirmation']
    title = ('Return intervals fall short of 80% coverage' if all(r['interval_coverage'] < .8 for r in rows)
             else 'Observed return-interval coverage')
    fig = frame(title, window(evidence))
    ax = fig.add_axes([.16, .26, .70, .46])
    panel(ax, horizon_labels(rows))
    for i, row in enumerate(rows):
        value = row['interval_coverage']
        ax.barh(i, value, height=.49, color=BLUE)
        label_at = max(value+.018, .825)
        ax.text(label_at if value < .90 else value-.018, i, f'{value:.2%}',
                ha='left' if value < .90 else 'right', va='center', fontsize=14,
                color=INK if value < .90 else 'white', weight='bold')
        if value > .30:
            ax.text(.02, i, f'n = {row["interval_available"]:,}', va='center', color='white', fontsize=12)
        else:
            ax.text(.45, i, f'n = {row["interval_available"]:,}', va='center', fontsize=12)
    ax.axvline(.8, color=INK, linestyle='--', lw=1.5)
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Observed coverage of the predicted q10–q90 return interval', labelpad=12)
    fig.text(.16, .765, 'Dashed line: nominal 80% coverage  |  n: mature, interval-available stock-date outcomes', fontsize=12, color=MUTED)
    finish(fig, destination, 'Coverage is an empirical hit rate, not an IC confidence interval. Overlapping outcomes are not independent samples.',
           evidence, 'backtests/us/oef2015/confirmation_summary.json')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_replay(root, evidence):
    """Load optional saved research replay, never infer executable performance."""
    root = Path(root)
    summary_path, curves_path = root/RESULTS/'replay_summary.json', root/RESULTS/'replay_curves.csv'
    if not summary_path.exists() and not curves_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if digest(curves_path) != summary['curves_sha256']:
        raise ValueError('Replay curve hash does not match its saved summary')
    if (summary['selected_kind'] != evidence['selected_model'] or summary['execution_validated'] is not False
            or summary['conditional_observable_cohort'] is not True
            or summary['survivorship_free'] is not False or summary['full_us_market'] is not False):
        raise ValueError('Replay must match the frozen model and disclose its bounded research scope')
    source = json.loads((root/RESULTS/'source_audit.json').read_text())
    if summary['price_sha256'] != source['prices_sha256'] or summary['calendar_sha256'] != source['calendar_sha256']:
        raise ValueError('Replay source hashes do not match the frozen source audit')
    raw = list(csv.DictReader(curves_path.open()))
    if set(row['series'] for row in raw) != set(REPLAY_NAMES) or set(summary['strategies']) != set(REPLAY_NAMES):
        raise ValueError('Replay requires the model, fee stress and both reference series')
    metrics, curves, shared_dates = [], [], None
    for name, label in REPLAY_NAMES.items():
        record = summary['strategies'][name]
        if (record['execution_validated'] is not False or record['performance_scope'] != 'research_valuation_only'
                or record['performance_validated'] is not True or record['valuation_gap_sessions'] != 0
                or record['terminal_unknown_count'] != 0):
            raise ValueError('Unresolved replay valuations cannot produce an equity chart')
        rows = [row for row in raw if row['series'] == name]
        dates = [row['date'] for row in rows]
        if (dates != sorted(set(dates)) or dates[0] != summary['start'] or dates[-1] != summary['end']
                or len(dates) != record['return_metrics']['sessions']):
            raise ValueError('Replay dates must match the complete saved window')
        if shared_dates is None:
            shared_dates = dates
        elif dates != shared_dates:
            raise ValueError('Replay comparisons require the same dates')
        nav = [float(row['nav']) for row in rows]
        if any(not math.isfinite(value) or value <= 0 for value in nav) or any(
                row['valuation_status'] != 'observed' or int(row['unresolved_holdings_count']) != 0 for row in rows):
            raise ValueError('Replay needs finite observed valuations for every session')
        total = nav[-1]/record['initial_cash']-1
        if not math.isclose(nav[-1], record['final_nav'], rel_tol=1e-12) or not math.isclose(
                total, record['return_metrics']['total_return'], abs_tol=1e-12):
            raise ValueError('Replay endpoint does not reconcile to its saved return')
        metrics.append(dict(series=name, label=label, **record['return_metrics'],
                            fee_per_side=record['fee_per_side'], initial_cash=record['initial_cash'],
                            final_nav=record['final_nav'], execution_validated=False,
                            mean_equity_exposure=sum(float(row['equity'])/float(row['nav']) for row in rows)/len(rows)))
        curves.extend(dict(series=name, date=row['date'], nav=float(row['nav']),
                           initial_capital_index=100*float(row['nav'])/record['initial_cash']) for row in rows)
    return dict(start=summary['start'], end=summary['end'], metrics=metrics, curves=curves,
                source_sha256={f'{RESULTS}/{path.name}': digest(path) for path in [summary_path, curves_path]})


def replay_chart(evidence, replay, destination):
    totals = {row['series']: row['total_return'] for row in replay['metrics']}
    title = ('Conditional replay trails both references' if totals['selected_15bps'] < min(
             totals['cohort_101slots_15bps'], totals['spy_15bps']) else 'Conditional adjusted-price research replay')
    fig = frame(title, f'{replay["start"]} to {replay["end"]}  |  Source-adjusted research units; execution unvalidated')
    ax = fig.add_axes([.10, .26, .78, .43])
    colors = [TEAL, RED, BLUE, INK]
    for name, color in zip(REPLAY_NAMES, colors):
        rows = [row for row in replay['curves'] if row['series'] == name]
        dates = [date.fromisoformat(row['date']) for row in rows]
        ax.plot(dates, [row['initial_capital_index'] for row in rows], color=color,
                linestyle='--' if name == 'selected_30bps' else '-', linewidth=2.1,
                label=REPLAY_NAMES[name]+f'  ({totals[name]:+.2%})')
    ax.axhline(100, color=MUTED, lw=.8)
    ax.grid(axis='y', color=GRID, linewidth=.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)
    ax.set_ylabel('Initial capital = 100', labelpad=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='lower left', bbox_to_anchor=(-.01, 1.01), ncol=2, frameon=False, fontsize=11)
    exposure = {row['series']: row['mean_equity_exposure'] for row in replay['metrics']}
    fig.text(.10, .18, f'Mean equity exposure: model {exposure["selected_15bps"]:.2%}; cohort {exposure["cohort_101slots_15bps"]:.2%}. '
             'References are not exposure-matched.', fontsize=11, color=MUTED)
    finish(fig, destination,
           'Monthly, next-session close; top 10, 95% target. Cohort has 101 fixed slots; missing slots stay cash. Costs per side.',
           evidence, 'replay_summary.json + replay_curves.csv | Conditional research valuation only')


def build(root=ROOT, destination=None):
    evidence = load_evidence(root)
    replay = load_replay(root, evidence)
    destination = Path(root)/'docs' if destination is None else Path(destination)
    assets, tables = destination/'assets/us', destination/'showcase/us'
    assets.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        style()
        for name, renderer in [('confirmation_ic', confirmation_chart), ('model_selection', selection_chart),
                               ('probability_skill', probability_chart), ('interval_coverage', coverage_chart)]:
            renderer(evidence, assets/(name+'.png'))
        if replay is not None:
            replay_chart(evidence, replay, assets/'conditional_replay.png')
    write_csv(tables/'model_selection.csv', evidence['selection'])
    write_csv(tables/'confirmation.csv', evidence['confirmation'])
    summary = {key: value for key, value in evidence.items() if key not in ['selection', 'confirmation']}
    summary.update(schema='us-aggregate-charts-v1', experiment='Experiment 1: initial public-data run',
                   evaluation_context='Retrospective current-vintage public-price research; confirmation is evaluated after development-only architecture freeze.',
                   cohort='101 original equity share classes in the 2015-06-30 OEF holdings, published 2015-09-02',
                   survivorship_free=False, full_us_market=False, point_in_time_vendor_vintages=False,
                   source_identity_limit='Ticker/provider-history review is not permanent-identifier or complete corporate-action verification.',
                   ic_interval_method='95% circular 60-session block bootstrap; 2,000 draws, seed 42; missing-day slots preserved',
                   interval_target='q10–q90 predicted return interval; nominal 80% coverage',
                   sample_policy='Conditional on observed mature task-available outcomes; unavailable original identities remain in prediction counts.',
                   source_policy='Yahoo Finance public history; AKShare is a secondary source audit, not a training input.',
                   latest_refit_evaluated_out_of_sample=False, execution_validated=False, eligible=False,
                   portfolio_limit='These four charts evaluate forecast tasks, not validated portfolio or execution performance.')
    if replay is not None:
        write_csv(tables/'conditional_replay.csv', replay['curves'])
        write_csv(tables/'conditional_replay_summary.csv', replay['metrics'])
        summary['replay'] = {key: value for key, value in replay.items() if key not in ['curves', 'source_sha256']}
        summary['replay_scope'] = 'Conditional bounded-cohort adjusted-unit illustration; research valuations reconcile, execution and actual-share action ledger remain unvalidated.'
        summary['source_sha256'] = dict(summary['source_sha256'], **replay['source_sha256'])
    (tables/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'docs', help='Output directory containing assets/us and showcase/us')
    args = parser.parse_args()
    build(destination=args.destination)
    print(f'US charts and aggregate tables written to {args.destination}')


if __name__ == '__main__':
    main()
