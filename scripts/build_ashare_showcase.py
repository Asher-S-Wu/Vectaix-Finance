"""Render public A-share evaluation charts from committed aggregate evidence.

Static Matplotlib PNGs follow the Hong Kong showcase's paper/teal/blue style.
Direct labels and CSV companions keep values accessible in a narrow README.
This renderer reads JSON only: it never fits, tunes, replays or loads a pickle.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'backtests/cn/universal/protocol.json',
    'backtests/cn/universal/frozen_architecture.json',
    'backtests/cn/universal/development_common_universe.json',
    'backtests/cn/universal/confirmation_summary.json',
    'models/cn/universal/frozen/linear.json',
    'backtests/cn/universal/development_summary.json',
)
INK, MUTED, TEAL, BLUE, RED = '#172C3C', '#60717C', '#00866F', '#467CB8', '#CE6651'
PAPER, GRID = '#FAFAF6', '#E5E9E7'
MODEL_NAMES = {'factor': 'Factor score', 'linear': 'Ridge',
               'lightgbm_small': 'Small LightGBM', 'lightgbm_large': 'Large LightGBM'}
METRICS = ('rank_ic_mean', 'ic_ci_lower', 'ic_ci_upper', 'ic_dates', 'predictions',
           'mature_labels', 'score_available', 'probability_available', 'brier',
           'baseline_brier', 'brier_skill', 'interval_available', 'interval_covered',
           'interval_coverage')


def load_evidence(root=ROOT):
    """Load the saved selection and confirmation; reject mismatched evidence."""
    documents = [json.loads((root / name).read_text()) for name in SOURCE_FILES]
    protocol, frozen, common, confirmation, model, development = documents
    if frozen['selection_set'] != 'development' or frozen['confirmation_used'] is not False:
        raise ValueError('Charts require development-only model selection')
    if frozen['selected_kind'] != model['kind'] or model['kind'] != 'linear':
        raise ValueError('Expected the saved frozen Ridge model')
    development_hash = hashlib.sha256(json.dumps(development, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if frozen['development_sha256'] != development_hash:
        raise ValueError('Development evidence hash does not match the freeze')
    if common != {kind: record['common_score_universe'] for kind, record in development.items()}:
        raise ValueError('Common-universe selection metrics differ from the frozen development record')
    if set(common) != set(MODEL_NAMES):
        raise ValueError('All four common-universe candidates are required')
    if not math.isclose(frozen['selection_statistic'], common['linear']['rank_ic_mean'], abs_tol=1e-12):
        raise ValueError('Selection statistic does not match the common-universe metric')
    if len({(r['ic_dates'], r['common_score_rows'], r['common_mature_rows']) for r in common.values()}) != 1:
        raise ValueError('Candidate comparison must use the same observed sample')
    selection = [dict(model=kind, label=MODEL_NAMES[kind], **common[kind]) for kind in MODEL_NAMES]
    rows = []
    for horizon in protocol['horizons']:
        source = confirmation['horizons'][str(horizon)]
        row = dict(horizon=horizon, **{name: source[name] for name in METRICS})
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f'Non-finite confirmation metric at horizon {horizon}')
        if not row['ic_ci_lower'] <= row['rank_ic_mean'] <= row['ic_ci_upper']:
            raise ValueError('IC estimate must lie within its saved interval')
        if row['baseline_brier'] <= 0 or not math.isclose(
                row['brier_skill'], 1 - row['brier'] / row['baseline_brier'], abs_tol=1e-12):
            raise ValueError('Brier skill does not match the saved scores')
        if row['interval_available'] <= 0 or not math.isclose(
                row['interval_coverage'], row['interval_covered'] / row['interval_available'], abs_tol=1e-12):
            raise ValueError('Interval coverage does not match the saved counts')
        rows.append(row)
    return dict(selected_model=frozen['selected_kind'], model_version=model['model_version'],
                development_start=protocol['development_start'], development_end=protocol['development_end'],
                confirmation_start=protocol['confirmation_start'], confirmation_end=confirmation['as_of'],
                selection=selection, confirmation=rows,
                source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES})


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'text.parse_math': False,
                        'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                        'ytick.color': INK, 'axes.facecolor': PAPER, 'figure.facecolor': PAPER,
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'axes.spines.left': False, 'axes.spines.bottom': False,
                        'savefig.facecolor': PAPER})


def frame(title, subtitle):
    fig = plt.figure(figsize=(14, 7.7))
    fig.text(.065, .95, 'VECTAIX FINANCE  /  A-SHARE EVALUATION', fontsize=11, color=TEAL, weight='bold')
    fig.text(.065, .885, title, fontsize=25, weight='bold')
    fig.text(.065, .837, subtitle, fontsize=12, color=MUTED)
    return fig


def finish(fig, destination, note, source):
    fig.text(.065, .075, note, fontsize=11, color=MUTED)
    fig.text(.065, .038, 'Source: ' + source, fontsize=10, color=MUTED)
    fig.savefig(destination, dpi=180, metadata={'Software': 'Vectaix Finance / build_ashare_showcase'})
    plt.close(fig)


def panel(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .45, -.55)
    ax.grid(axis='x', color=GRID, linewidth=.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=10)


def confirmation_window(evidence):
    return f'Frozen Ridge  |  Confirmation: {evidence["confirmation_start"]} to {evidence["confirmation_end"]}'


def confirmation_chart(evidence, destination):
    rows = evidence['confirmation']
    fig = frame('Ranking quality across four horizons', confirmation_window(evidence))
    ax = fig.add_axes([.16, .21, .60, .50])
    panel(ax, [f'{row["horizon"]} session' + ('s' if row['horizon'] != 1 else '') for row in rows])
    for i, row in enumerate(rows):
        mean = row['rank_ic_mean']
        color = TEAL if row['horizon'] == 20 else BLUE
        ax.errorbar(mean, i, xerr=[[mean - row['ic_ci_lower']], [row['ic_ci_upper'] - mean]],
                    fmt='o', color=color, ecolor=color, markersize=9, elinewidth=3, capsize=6, capthick=2)
        ax.text(mean, i - .23, f'{mean:.4f}', ha='center', fontsize=14, weight='bold', color=color)
        ax.text(1.04, i, f'{row["ic_dates"]} IC dates', transform=ax.get_yaxis_transform(), va='center', fontsize=12)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-.01, .27)
    ax.set_xticks([0, .05, .10, .15, .20, .25])
    ax.set_xlabel('Mean daily cross-sectional rank IC', labelpad=12)
    fig.text(.16, .757, 'Dots: mean IC     Lines: 95% confidence intervals', fontsize=12, color=MUTED)
    finish(fig, destination,
           '60-session circular block bootstrap, 2,000 draws. Mature outcomes only; longer horizons have fewer dates.',
           'backtests/cn/universal/confirmation_summary.json | Latest refit has no out-of-sample evaluation.')


def selection_chart(evidence, destination):
    rows = sorted(evidence['selection'], key=lambda row: row['rank_ic_mean'], reverse=True)
    fig = frame('Ridge leads the development comparison',
                f'{evidence["development_start"]} to {evidence["development_end"]}  |  20-session horizon  |  Common score-available universe')
    ax = fig.add_axes([.24, .21, .64, .50])
    panel(ax, [row['label'] + ('  (selected)' if row['model'] == evidence['selected_model'] else '') for row in rows])
    for i, row in enumerate(rows):
        value = row['rank_ic_mean']
        color = TEAL if row['model'] == evidence['selected_model'] else BLUE
        ax.barh(i, value, height=.49, color=color)
        ax.text(value + (.0018 if value >= 0 else -.0018), i, f'{value:+.4f}',
                ha='left' if value >= 0 else 'right', va='center', fontsize=14, weight='bold')
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-.025, .068)
    ax.set_xlabel('20-session mean daily rank IC', labelpad=12)
    sample = rows[0]
    fig.text(.24, .757, f'{sample["ic_dates"]} IC dates  |  {sample["common_mature_rows"]:,} mature stock-date outcomes', fontsize=12, color=MUTED)
    finish(fig, destination,
           'Selection follows the saved 2024 rule. These estimates are used for model choice; confirmation is evaluated separately.',
           'backtests/cn/universal/development_common_universe.json + frozen_architecture.json')


def probability_chart(evidence, destination):
    rows = evidence['confirmation']
    fig = frame('Probability skill is below the baseline', confirmation_window(evidence))
    ax = fig.add_axes([.16, .21, .70, .50])
    panel(ax, [f'{row["horizon"]} session' + ('s' if row['horizon'] != 1 else '') for row in rows])
    for i, row in enumerate(rows):
        value = row['brier_skill']
        ax.barh(i, value, height=.49, color=RED if value < 0 else TEAL)
        ax.text(value - .00035 if value < 0 else value + .00035, i, f'{value:+.2%}',
                ha='right' if value < 0 else 'left', va='center', fontsize=14, weight='bold')
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(min(-.014, min(row['brier_skill'] for row in rows) * 1.25),
                max(.001, max(row['brier_skill'] for row in rows) * 1.25))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_xlabel('Brier skill = 1 − model Brier / historical-frequency baseline Brier', labelpad=12)
    fig.text(.16, .757, 'Zero matches the baseline; negative values mean higher probability error', fontsize=12, color=MUTED)
    finish(fig, destination,
           'All four horizons have negative skill on mature, probability-available outcomes. Ranking IC measures a different task.',
           'backtests/cn/universal/confirmation_summary.json | Latest refit has no out-of-sample evaluation.')


def coverage_chart(evidence, destination):
    rows = evidence['confirmation']
    fig = frame('How often the return interval contains the outcome', confirmation_window(evidence))
    ax = fig.add_axes([.16, .21, .70, .50])
    panel(ax, [f'{row["horizon"]} session' + ('s' if row['horizon'] != 1 else '') for row in rows])
    for i, row in enumerate(rows):
        ax.barh(i, row['interval_coverage'], height=.49, color=BLUE)
        ax.text(max(row['interval_coverage'] + .016, .835), i, f'{row["interval_coverage"]:.2%}', va='center', fontsize=14, weight='bold')
        ax.text(.02, i, f'n = {row["interval_available"]:,}', va='center', color='white', fontsize=12)
    ax.axvline(.8, color=INK, linestyle='--', lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Observed coverage of the predicted q10–q90 return interval', labelpad=12)
    fig.text(.16, .757, 'Dashed line: nominal 80% coverage  |  n: mature, interval-available stock-date outcomes', fontsize=12, color=MUTED)
    finish(fig, destination,
           'Coverage is an empirical hit rate, not an IC confidence interval. Overlapping outcomes are not independent samples.',
           'backtests/cn/universal/confirmation_summary.json | Latest refit has no out-of-sample evaluation.')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def build(root=ROOT, destination=None):
    evidence = load_evidence(root)
    destination = root / 'docs' if destination is None else Path(destination)
    assets, tables = destination / 'assets/cn', destination / 'showcase/cn'
    assets.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        style()
        for name, render in [('confirmation_ic', confirmation_chart), ('model_selection', selection_chart),
                             ('probability_skill', probability_chart), ('interval_coverage', coverage_chart)]:
            render(evidence, assets / (name + '.png'))
    write_csv(tables / 'model_selection.csv', evidence['selection'])
    write_csv(tables / 'confirmation.csv', evidence['confirmation'])
    summary = {key: value for key, value in evidence.items() if key not in ('selection', 'confirmation')}
    summary.update(schema='ashare-aggregate-charts-v1',
                   evaluation_context='Retrospective chronological research with frozen parameters; confirmation did not select the model.',
                   ic_interval_method='95% circular 60-session block bootstrap; 2,000 draws',
                   interval_target='q10-q90 predicted return interval; nominal 80% coverage',
                   sample_policy='Only mature task-available outcomes enter each metric; counts are not independent sample sizes.',
                   latest_refit_evaluated_out_of_sample=False,
                   portfolio_returns_verified=False,
                   portfolio_limit='Unresolved corporate actions and terminal valuations prevent verification of equity, CAGR and drawdown.')
    (tables / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs', help='Output directory containing assets/cn and showcase/cn')
    args = parser.parse_args()
    build(destination=args.destination)
    print(f'A-share charts and aggregate tables written to {args.destination}')


if __name__ == '__main__':
    main()
