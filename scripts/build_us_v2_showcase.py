"""Rebuild bounded US v2 research figures from saved aggregate evidence only.

No vendor downloads, per-security predictions, model execution, or retraining.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'vectaix-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'vectaix-cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter, MaxNLocator
import matplotlib.dates as mdates
import numpy as np
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
RESULTS = 'backtests/us/oef2015-rank-v2'
SOURCE_FILES = tuple(f'{RESULTS}/{name}.json' for name in (
    'protocol', 'source_audit', 'annual_common_ic', 'development_selection',
    'nested_selection', 'frozen_architecture', 'reused_diagnostic_summary',
    'training_status', 'latest_refit_summary'))
MODEL_PATH = 'models/us/oef2015-rank-v2/frozen/model.json'
MODEL_NAMES = {'factor_reference': 'Factor reference', 'momentum_12_1': '12–1 momentum',
               'ridge_rank_3y': 'Ridge rank / 3y', 'ridge_rank_5y': 'Ridge rank / 5y',
               'gbm_rank_3y': 'LightGBM rank / 3y', 'gbm_rank_5y': 'LightGBM rank / 5y'}
METRICS = ('rank_ic_mean', 'ic_ci_lower', 'ic_ci_upper', 'ic_dates', 'ic_calendar_slots',
           'predictions', 'mature_labels', 'missing_or_immature_outcomes', 'score_available',
           'probability_available', 'brier', 'baseline_brier', 'brier_skill',
           'interval_available', 'interval_covered', 'interval_coverage')
INK, MUTED, TEAL, BLUE, RED = '#172C3C', '#60717C', '#00866F', '#467CB8', '#BE5747'
PAPER, GRID = '#FAFAF6', '#E5E9E7'
FIGURES = ('rolling_selection', 'nested_selection', 'reused_diagnostic_ic', 'calibration_checks', 'source_recovery')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, message):
    if not isinstance(actual, (float, int)) or not math.isfinite(actual):
        raise ValueError('Non-finite '+message)
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(message)


def load_evidence(root=ROOT):
    """Check cohort counts, chronology, shared samples, arithmetic, and source hashes."""
    root = Path(root)
    protocol, source, annual, selection, nested, frozen, diagnostic, status, latest = [
        json.loads((root/name).read_text()) for name in SOURCE_FILES]
    if status['status'] != 'complete':
        raise ValueError('Charts require complete training')
    if (status['recent_period_used_for_selection'] is not False
            or status['untouched_recent_holdout'] is not False
            or frozen['recent_diagnostic_used_for_selection'] is not False
            or diagnostic['untouched_holdout'] is not False
            or protocol['recent_diagnostic_is_untouched'] is not False
            or diagnostic['evaluation_status'] != 'reused_diagnostic_after_experiment1'):
        raise ValueError('The recent period must stay labelled reused diagnostic, not a new holdout')
    if latest['out_of_sample_evaluated'] is not False:
        raise ValueError('The latest refit must remain unevaluated out of sample')
    if any(x['eligible'] is not False or x['execution_validated'] is not False for x in (status, latest)):
        raise ValueError('Eligibility and execution validation must remain false')
    if (source['identities_with_some_history'] + source['fully_unavailable_identities'] != source['declared_cohort']
            or source['declared_cohort'] != 101 or len(source['coverage']) != 101
            or len({r['security_id'] for r in source['coverage']}) != 101
            or sum(r['valid_quote_rows'] > 0 for r in source['coverage']) != source['identities_with_some_history']
            or len(source['recovered']) != source['recovered_identity_count']):
        raise ValueError('Source cohort counts do not reconcile')
    for key in ('vendor_vintages_point_in_time', 'survivorship_free', 'terminal_outcomes_complete'):
        if source[key] is not False:
            raise ValueError('Bounded source limitations must remain explicit')
    years, order = protocol['walk_forward_years'], list(protocol['candidates'])
    if years != [2020, 2021, 2022, 2023, 2024] or set(order) != set(MODEL_NAMES):
        raise ValueError('Require all six predeclared candidates and five annual folds')
    if len(protocol['features']) != 24 or protocol['final_heads'] != [1, 5, 20, 60]:
        raise ValueError('Unexpected feature or horizon contract')
    for key, name in [('annual_common_ic_sha256', 'annual_common_ic'),
                      ('selection_sha256', 'development_selection'), ('nested_selection_sha256', 'nested_selection')]:
        if frozen[key] != digest(root/f'{RESULTS}/{name}.json'):
            raise ValueError('Saved aggregate evidence does not match its freeze hash')
    annual_rows = []
    for year in years:
        members = annual[str(year)]
        if set(members) != set(order):
            raise ValueError('Missing declared annual candidate')
        samples = {(r['ic_dates'], r['date_sha256'], r['common_score_rows'], r['common_mature_rows']) for r in members.values()}
        if len(samples) != 1:
            raise ValueError('Candidate comparisons must use the same rows and dates within each year')
        for kind in order:
            row = members[kind]
            if not math.isfinite(row['rank_ic_mean']):
                raise ValueError('Non-finite annual metric')
            if not 0 < row['ic_dates'] <= row['common_mature_rows'] <= row['common_score_rows']:
                raise ValueError('Annual sample counts do not reconcile')
            annual_rows.append(dict(year=year, model=kind, **row))
    for kind in order:
        values = [annual[str(y)][kind]['rank_ic_mean'] for y in years]
        saved = selection[kind]
        if saved['years'] != years or saved['annual_ic'] != values:
            raise ValueError('Saved selection differs from annual metrics')
        for key, expected in [('mean_annual_ic', np.mean(values)), ('median_annual_ic', np.median(values)),
                              ('worst_annual_ic', min(values)), ('positive_years', sum(x > 0 for x in values))]:
            close(saved[key], expected, 'Selection summary does not reconcile with annual metrics')
    selected = max(order, key=lambda kind: selection[kind]['mean_annual_ic'])
    if selected != frozen['selected_kind'] or selected != status['selected_kind'] or selected != latest['kind']:
        raise ValueError('Selected architecture does not follow predeclared mean-IC rule')
    close(frozen['selection_statistic'], selection[selected]['mean_annual_ic'], 'Selection statistic differs from saved mean')
    if status['candidate_fold_fits'] != len(order)*len(years):
        raise ValueError('The candidate-fold fit count is incomplete')
    path = nested['outer_path']
    if [r['outer_year'] for r in path] != protocol['nested_walk_forward']['outer_evaluation_years']:
        raise ValueError('Nested outer evaluation years differ')
    for row in path:
        outer, inner = row['outer_year'], row['inner_years']
        if inner != list(range(2020, outer)) or row['own_outer_year_used_for_selection'] is not False:
            raise ValueError('Nested architecture selection must use strictly earlier inner years')
        chosen = max(order, key=lambda k: np.mean([annual[str(y)][k]['rank_ic_mean'] for y in inner]))
        if chosen != row['selected_kind']:
            raise ValueError('Nested selection differs from earlier-fold mean-IC rule')
        close(row['inner_mean_ic'], np.mean([annual[str(y)][chosen]['rank_ic_mean'] for y in inner]), 'Nested inner mean differs')
        close(row['outer_ic'], annual[str(outer)][chosen]['rank_ic_mean'], 'Nested outer IC differs')
    close(nested['mean_outer_ic'], np.mean([r['outer_ic'] for r in path]), 'Nested outer mean differs')
    rows = []
    if diagnostic['as_of'] != protocol['data_end']:
        raise ValueError('Diagnostic cutoff differs from the frozen protocol')
    for horizon in protocol['final_heads']:
        row = dict(horizon=horizon, **{name: diagnostic['horizons'][str(horizon)][name] for name in METRICS})
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row.values()):
            raise ValueError('Non-finite diagnostic metric')
        if not -1 <= row['ic_ci_lower'] <= row['rank_ic_mean'] <= row['ic_ci_upper'] <= 1:
            raise ValueError('IC confidence interval does not contain its point estimate')
        if not 0 < row['ic_dates'] <= row['ic_calendar_slots']:
            raise ValueError('IC dates do not reconcile')
        if row['mature_labels'] + row['missing_or_immature_outcomes'] != row['predictions']:
            raise ValueError('Diagnostic outcome counts do not reconcile')
        if any(not 0 < row[k] <= row['mature_labels'] for k in ('score_available', 'probability_available', 'interval_available')):
            raise ValueError('Task counts must be mature and available')
        if row['baseline_brier'] <= 0:
            raise ValueError('Brier skill requires a positive baseline error')
        close(row['brier_skill'], 1-row['brier']/row['baseline_brier'], 'Brier skill differs from saved errors')
        if not 0 <= row['interval_covered'] <= row['interval_available']:
            raise ValueError('Interval coverage count is invalid')
        close(row['interval_coverage'], row['interval_covered']/row['interval_available'], 'Interval coverage differs from saved counts')
        rows.append(row)
    model = json.loads((root/MODEL_PATH).read_text())
    if (model['kind'], model['market'], model['currency']) != (selected, 'US', 'USD'):
        raise ValueError('Saved model is not the selected independent USD architecture')
    for name in ('protocol', 'source_audit'):
        if model[name+'_sha256'] != digest(root/f'{RESULTS}/{name}.json'):
            raise ValueError('Model has stale '+name+' hash')
    if protocol['source_audit_sha256'] != digest(root/f'{RESULTS}/source_audit.json'):
        raise ValueError('Protocol source audit hash differs')
    recovered = []
    coverage = {r['security_id']: r for r in source['coverage']}
    for r in source['recovered']:
        if r['unsupported_terminal_labels'] is not True or r['valid_quote_rows'] != coverage[r['security_id']]['valid_quote_rows']:
            raise ValueError('Recovered source counts or unresolved-terminal flags differ')
        if not r['valid_start'] <= r['supported_data_end'] <= r['last_original_trade_date']:
            raise ValueError('Recovered source dates are inconsistent')
        recovered.append({**r, 'has_source_gap_before_terminal': r['supported_data_end'] < r['last_original_trade_date']})
    evidence = dict(selected_model=selected, selected_label=MODEL_NAMES[selected], model_version=model['model_version'],
                selection_statistic=frozen['selection_statistic'], mean_outer_ic=nested['mean_outer_ic'],
                years=years, annual=annual_rows, selection=[dict(model=k, label=MODEL_NAMES[k], **selection[k]) for k in order],
                nested=path, diagnostic=rows, diagnostic_start='2025-01-01', diagnostic_end=diagnostic['as_of'],
                declared_cohort=source['declared_cohort'], identities_with_some_history=source['identities_with_some_history'],
                recovered=recovered, recovered_quote_rows=sum(r['valid_quote_rows'] for r in recovered),
                coverage=source['coverage'], untouched_holdout=False, latest_as_of=latest['as_of'],
                latest_model_version=latest['model_version'], bootstrap=diagnostic['bootstrap'],
                source_sha256={name: digest(root/name) for name in (*SOURCE_FILES, MODEL_PATH)})
    load_optional_evidence(root, evidence)
    return evidence


def load_optional_evidence(root, evidence):
    """Load paired diagnostics and terminal-aware replay aggregates when present."""
    paired_file = root/RESULTS/'paired_recent_summary.json'
    if paired_file.exists():
        paired = json.loads(paired_file.read_text())
        if paired['untouched_holdout'] is not False:
            raise ValueError('Paired comparison is a reused diagnostic')
        rows = []
        for h in (1, 5, 20, 60):
            row = dict(horizon=h, **paired['horizons'][str(h)])
            for key in ('first_run_ic','second_run_ic','paired_ic_change','change_ci_lower','change_ci_upper'):
                if not math.isfinite(row[key]):
                    raise ValueError('Non-finite paired diagnostic')
            close(row['paired_ic_change'], row['second_run_ic']-row['first_run_ic'], 'Paired change differs from common-sample means')
            if not row['change_ci_lower'] <= row['paired_ic_change'] <= row['change_ci_upper']:
                raise ValueError('Paired interval does not contain its estimate')
            if not 0 < row['ic_dates'] <= row['common_mature_rows'] <= row['common_score_rows']:
                raise ValueError('Paired common sample counts do not reconcile')
            rows.append(row)
        evidence['paired'] = rows
        evidence['source_sha256'][str(paired_file.relative_to(root))] = digest(paired_file)
    replay_file = root/RESULTS/'replay_summary.json'
    if replay_file.exists():
        replay = json.loads(replay_file.read_text())
        if (replay['untouched_holdout'] is not False or replay['reused_diagnostic'] is not True
                or replay['execution_validated'] is not False or replay['selected_kind'] != evidence['selected_model']):
            raise ValueError('Replay must retain reused, unvalidated scope')
        curve_file = root/RESULTS/'replay_curves.csv'
        if replay['curves_sha256'] != digest(curve_file):
            raise ValueError('Replay curve hash differs')
        rows=[]
        for name, row in replay['strategies'].items():
            if row['valuation_gap_sessions'] or row['terminal_unknown_count']:
                if (row['performance_scope'] != 'reference_only' or row['performance_validated'] is not False
                    or any(row[k] is not None for k in ('return_metrics','total_return','annualized_return','max_drawdown','final_nav'))):
                    raise ValueError('Unresolved portfolio must not promote stale-reference values to validated returns')
            rows.append(dict(strategy=name, **{k:row[k] for k in (
                'performance_scope','performance_validated','execution_validated','valuation_gap_sessions',
                'terminal_unknown_count','total_return','annualized_return','max_drawdown','fill_ratio',
                'average_equity_exposure','total_fees','rebalance_count')}))
        evidence['replay'] = rows
        for file in (replay_file, curve_file):
            evidence['source_sha256'][str(file.relative_to(root))] = digest(file)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.parse_math': False,
        'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
        'axes.facecolor': PAPER, 'figure.facecolor': PAPER, 'savefig.facecolor': PAPER,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.spines.bottom': False})


def frame(title, subtitle, *, reused=False, height=8):
    fig = plt.figure(figsize=(14, height))
    fig.text(.06, .951, 'VECTAIX FINANCE  /  US EXPERIMENT 2',
             fontsize=10.5, color=TEAL, weight='bold')
    fig.text(.06, .889, title, fontsize=24, weight='bold')
    fig.text(.06, .839, subtitle, fontsize=11.5, color=MUTED)
    if reused:
        fig.text(.94, .951, 'REUSED DIAGNOSTIC · NOT A NEW HOLDOUT', ha='right', fontsize=10.5, color=RED, weight='bold')
    return fig


def finish(fig, path, note, evidence, source):
    fig.text(.06, .126, note, fontsize=10.2, color=MUTED, linespacing=1.45)
    fig.text(.06, .078, 'Fixed 101-share-class 2015 cohort; historical coverage is not complete terminal wealth or point-in-time vendor data.',
             fontsize=10.1, color=MUTED)
    fig.text(.06, .039, 'Sources: '+source+'  |  Latest refit untested. Execution unvalidated. See summary.json for SHA-256 hashes.',
             fontsize=9.2, color=MUTED)
    fig.savefig(path, dpi=180, metadata={'Software': 'Vectaix Finance / build_us_v2_showcase'})
    plt.close(fig)


def panel(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.45, -.55)
    ax.tick_params(length=0, pad=9)
    ax.grid(axis='x', color=GRID, lw=.9)
    ax.set_axisbelow(True)


def limits(values, margin=.15):
    low, high = min(0., min(values)), max(0., max(values))
    span = max(high-low, .001)
    return low-margin*span, high+margin*span


def rolling_chart(e, path):
    selected = e['selected_label']
    fig = frame('Six candidates across five rolling annual folds',
                f'20-session common-cohort mean daily rank IC · Selection: {selected} ({e["selection_statistic"]:+.4f} mean)')
    rows = e['selection']
    values = np.array([r['annual_ic']+[r['mean_annual_ic']] for r in rows])
    bound = max(abs(values.min()), abs(values.max()), .001)
    cmap = LinearSegmentedColormap.from_list('honest_ic', [RED, PAPER, TEAL])
    ax = fig.add_axes([.235, .28, .685, .485])
    ax.imshow(values, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    labels = [r['label']+('  *' if r['model'] == e['selected_model'] else '') for r in rows]
    ax.set_yticks(range(6), labels)
    counts = [next(r['ic_dates'] for r in e['annual'] if r['year'] == y) for y in e['years']]
    ax.set_xticks(range(6), [f'{y}\nn={n} dates' for y,n in zip(e['years'],counts)]+['5-year\nequal-weight mean'])
    ax.tick_params(length=0, pad=10)
    ax.axvline(4.5, color=PAPER, lw=6)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{values[i,j]:+.4f}', ha='center', va='center', fontsize=13,
                    weight='bold' if j == 5 else 'normal', color='white' if abs(values[i,j]) > bound*.60 else INK)
    fig.text(.235, .184, 'Negative IC     ←     0     →     Positive IC       * Selected by the fixed mean-IC rule', color=MUTED, fontsize=11)
    finish(fig, path, 'Development comparison, not an independent test of the final winner. Every candidate uses identical rows and dates within each year.\n3y and 5y windows coincide in 2020 because all fitting is truncated at 2016.', e,
           'annual_common_ic.json; development_selection.json')


def nested_chart(e, path):
    fig = frame('Chronological selection: inner and outer results',
                f'Architecture chosen before each outer year · Mean outer IC {e["mean_outer_ic"]:+.4f} · '
                f'{sum(r["outer_ic"]>0 for r in e["nested"])}/{len(e["nested"])} positive outer years')
    ax = fig.add_axes([.16, .30, .59, .43]);rows=e['nested']
    panel(ax, [str(r['outer_year']) for r in rows])
    for i,r in enumerate(rows):
        ax.plot([r['inner_mean_ic'], r['outer_ic']], [i,i], color=GRID, lw=6, zorder=1)
        ax.scatter(r['inner_mean_ic'], i, marker='o', s=80, facecolor=PAPER, edgecolor=BLUE, lw=2, zorder=2)
        ax.scatter(r['outer_ic'], i, marker='D', s=75, color=TEAL if r['outer_ic']>=0 else RED, zorder=3)
        ax.text(r['inner_mean_ic'], i-.24, f'{r["inner_mean_ic"]:+.4f}', ha='center', fontsize=12, color=BLUE)
        ax.text(r['outer_ic'], i+.29, f'{r["outer_ic"]:+.4f}', ha='center', fontsize=12, color=INK, weight='bold')
        ax.text(1.05,i,f'{MODEL_NAMES[r["selected_kind"]]}\nInner years: {r["inner_years"][0]}–{r["inner_years"][-1]}',
                transform=ax.get_yaxis_transform(), va='center', fontsize=11)
    ax.axvline(0,color=MUTED,lw=1)
    ax.set_xlim(*limits([r[k] for r in rows for k in ('inner_mean_ic','outer_ic')],.2))
    ax.set_xlabel('20-session mean daily rank IC',labelpad=12)
    fig.text(.16,.77,'○ Earlier inner-fold mean       ◆ Subsequent outer-year IC',fontsize=12,color=MUTED)
    finish(fig,path,'Each choice uses only strictly earlier annual folds; the outer year never selects its own architecture.\nThis nested path remains retrospective research on current-vintage, bounded historical data.',e,'nested_selection.json; annual_common_ic.json')


def diagnostic_title(rows):
    if all(r['rank_ic_mean'] > 0 and r['ic_ci_lower'] < 0 < r['ic_ci_upper'] for r in rows):
        return 'Positive point estimates; all intervals cross zero'
    return 'Frozen ranking on the reused diagnostic period'


def diagnostic_chart(e,path):
    fig=frame(diagnostic_title(e['diagnostic']),
              f'{e["selected_label"]} · Frozen through {e["diagnostic_end"]} · {e["diagnostic_start"]}–{e["diagnostic_end"]}',reused=True)
    ax=fig.add_axes([.16,.285,.60,.465]);rows=e['diagnostic']
    panel(ax,[f'{r["horizon"]} session'+('s' if r['horizon']!=1 else '') for r in rows])
    for i,r in enumerate(rows):
        m,lo,hi=(r[k] for k in ('rank_ic_mean','ic_ci_lower','ic_ci_upper'))
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=TEAL,markersize=9,elinewidth=3,capsize=6,capthick=2)
        ax.text(m,i-.25,f'{m:+.4f}',ha='center',fontsize=14,weight='bold',color=TEAL)
        ax.text(1.04,i,f'{r["ic_dates"]} IC dates\n{r["score_available"]:,} mature rows',transform=ax.get_yaxis_transform(),va='center',fontsize=11)
    ax.axvline(0,color=MUTED,lw=1)
    ax.set_xlim(*limits([r[k] for r in rows for k in ('ic_ci_lower','ic_ci_upper')],.12))
    ax.xaxis.set_major_locator(MaxNLocator(6));ax.set_xlabel('Mean daily cross-sectional rank IC',labelpad=12)
    fig.text(.16,.781,'Dots: mean IC    Lines: 95% confidence intervals',fontsize=12,color=MUTED)
    if e.get('paired'):
        pair = next(r for r in e['paired'] if r['horizon'] == 20)
        fig.text(.16,.185, f'Paired 20-session change vs v1: {pair["paired_ic_change"]:+.4f}  '
                 f'[{pair["change_ci_lower"]:+.4f}, {pair["change_ci_upper"]:+.4f}] · '
                 f'{pair["common_mature_rows"]:,} common mature rows · Posthoc, joint method/data change', fontsize=10.5, color=MUTED)
    finish(fig,path,'95% circular 60-session block bootstrap, 2,000 draws; missing-day slots preserved. Outcomes overlap and are conditional on availability.\n2025–2026 was already examined in Experiment 1. Reuse cannot establish fresh independent confirmation.',e,'reused_diagnostic_summary.json; paired_recent_summary.json' if e.get('paired') else 'reused_diagnostic_summary.json')


def calibration_chart(e,path):
    rows=e['diagnostic'];fig=frame('Probability skill and return-interval coverage',
        f'{e["selected_label"]} · {e["diagnostic_start"]}–{e["diagnostic_end"]} · Mature task-available outcomes only',reused=True)
    left=fig.add_axes([.15,.29,.31,.45]);right=fig.add_axes([.61,.29,.30,.45])
    labels=[f'{r["horizon"]} session'+('s' if r['horizon']!=1 else '') for r in rows]
    for ax in (left,right):panel(ax,labels)
    left.set_title('Probability Brier skill',loc='left',fontsize=16,pad=24,weight='bold')
    skills=[r['brier_skill'] for r in rows];left.barh(range(4),skills,height=.38,color=[RED if x<0 else TEAL for x in skills])
    low,high=limits(skills,.32);left.set_xlim(low,high);left.axvline(0,color=MUTED,lw=1)
    for i,s in enumerate(skills):left.text(1.01,i,f'{s:+.3%}',transform=left.get_yaxis_transform(),ha='left',va='center',fontsize=12,weight='bold')
    left.xaxis.set_major_formatter(PercentFormatter(1,decimals=1));left.xaxis.set_major_locator(MaxNLocator(4))
    left.set_xlabel('0 = historical-frequency baseline',labelpad=12)
    right.set_title('q10–q90 interval coverage',loc='left',fontsize=16,pad=24,weight='bold')
    right.set_xlim(0,1);right.axvline(.8,color=MUTED,ls='--',lw=1.6)
    for i,r in enumerate(rows):
        right.scatter(r['interval_coverage'],i,s=85,color=TEAL,zorder=3)
        right.text(r['interval_coverage']+.025,i-.15,f'{r["interval_coverage"]:.2%}',fontsize=12,weight='bold')
        right.text(.035,i+.26,f'{r["interval_covered"]:,} / {r["interval_available"]:,}',fontsize=10.5,color=MUTED)
    right.xaxis.set_major_formatter(PercentFormatter(1));right.set_xticks([0,.2,.4,.6,.8,1])
    right.set_xlabel('Dashed line = nominal 80% target',labelpad=12)
    finish(fig,path,'Negative skill means higher Brier error than the historical-frequency baseline. Interval coverage alone does not establish useful forecasts.\nCounts are stock-date outcomes, not independent samples; latest 2026-09-30 refit has no out-of-sample test.',e,'reused_diagnostic_summary.json')


def source_chart(e,path):
    rows=sorted(e['recovered'],key=lambda r:(r['supported_data_end'],r['security_id']))
    fig=frame(f'{len(rows)} recovered histories, with explicit end boundaries',
        f'{e["recovered_quote_rows"]:,} valid quote rows restored · {e["identities_with_some_history"]}/{e["declared_cohort"]} identities now have some history · Terminal outcomes remain unresolved',height=10)
    ax=fig.add_axes([.13,.265,.61,.50]);panel(ax,[r['security_id'] for r in rows])
    for i,r in enumerate(rows):
        start,end,last=[mdates.date2num(date.fromisoformat(r[k])) for k in ('valid_start','supported_data_end','last_original_trade_date')]
        ax.plot([start,end],[i,i],color=TEAL,lw=5,solid_capstyle='butt')
        ax.scatter(end,i,color=TEAL,s=25,zorder=3)
        if r['has_source_gap_before_terminal']:
            ax.plot([end,last],[i,i],color=RED,lw=2,ls='--');ax.scatter(last,i,marker='|',s=100,color=RED)
        ax.text(1.025,i,f'{r["supported_data_end"]}  ·  {r["valid_quote_rows"]:,} rows',transform=ax.get_yaxis_transform(),va='center',fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(mdates.date2num(date(2013,10,1)),mdates.date2num(date(2026,10,1)))
    fig.text(.13,.791,'Solid teal: accepted history window*     Dashed red: unfilled source gap before original termination',fontsize=11,color=MUTED)
    fig.text(.765,.791,'Last supported quote · Valid rows',fontsize=10.5,color=MUTED)
    ax.set_xlabel('Calendar date; gaps within each accepted window remain missing',labelpad=12)
    finish(fig,path,'* Windows show boundaries, not complete daily coverage. FOXA quotes stop 2018-03-27; original termination is 2019-03-19.\nDD begins 2015-07-02 after an unmodeled noncash spinoff. No assumed cash settlement, acquirer substitution, or CVR value.',e,'source_audit.json')


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def build(root=ROOT,destination=None):
    root=Path(root);destination=Path(destination) if destination is not None else root/'docs'
    e=load_evidence(root);assets=destination/'assets/us/v2';tables=destination/'showcase/us/v2'
    assets.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True);style()
    for name,renderer in zip(FIGURES,(rolling_chart,nested_chart,diagnostic_chart,calibration_chart,source_chart)):
        renderer(e,assets/f'{name}.png')
    write_csv(tables/'rolling_selection.csv',e['annual'],('year','model','rank_ic_mean','ic_dates','common_score_rows','common_mature_rows','date_sha256','universe','outcome_policy'))
    write_csv(tables/'development_selection.csv',e['selection'],('model','mean_annual_ic','median_annual_ic','worst_annual_ic','positive_years'))
    write_csv(tables/'nested_selection.csv',e['nested'],('outer_year','inner_years','selected_kind','inner_mean_ic','outer_ic','own_outer_year_used_for_selection'))
    write_csv(tables/'reused_diagnostic.csv',e['diagnostic'],('horizon',*METRICS))
    write_csv(tables/'source_recovery.csv',e['recovered'],('security_id','valid_start','supported_data_end','last_original_trade_date','valid_quote_rows','has_source_gap_before_terminal','unsupported_terminal_labels','event_verification_scope','raw_export_sha256','events_export_sha256'))
    write_csv(tables/'cohort_coverage.csv',e['coverage'],('security_id','valid_quote_rows','eligible_rows','first_valid_quote','last_valid_quote','source_status'))
    if e.get('paired'):
        write_csv(tables/'paired_recent.csv', e['paired'], tuple(e['paired'][0]))
    if e.get('replay'):
        write_csv(tables/'replay_status.csv', e['replay'], tuple(e['replay'][0]))
    summary={k:e[k] for k in ('selected_model','model_version','selection_statistic','mean_outer_ic','declared_cohort',
        'identities_with_some_history','recovered_quote_rows','diagnostic_start','diagnostic_end','latest_model_version','latest_as_of','source_sha256')}
    summary.update(version='us-v2-aggregate-showcase-1',untouched_holdout=False,survivorship_free=False,full_us_market=False,
        terminal_outcomes_complete=False,execution_validated=False,eligible=False,latest_refit_evaluated_out_of_sample=False,
        evaluation_status='REUSED DIAGNOSTIC, NOT A NEW HOLDOUT',bootstrap=e['bootstrap'],
        definitions={'rank_ic':'Equally weighted daily cross-sectional Spearman score/adjusted-price-return correlation',
            'selection':'Highest equal-weight mean of five 2020–2024 annual common-row/common-date ICs; six predeclared candidates',
            'nested':'Each outer-year architecture uses only earlier inner-year mean IC; outer ICs are not winner-selection scores',
            'brier_skill':'1 - model Brier / historical-frequency baseline Brier',
            'interval_coverage':'Count within q10–q90 / mature interval-available count; nominal 80%',
            'recovery':'Some bounded original-identity history, not complete action-adjusted terminal wealth or point-in-time source vintages'},
        figures={name:{'path':f'assets/us/v2/{name}.png','sha256':digest(assets/f'{name}.png')} for name in FIGURES})
    if e.get('paired'):
        summary['paired_comparison'] = e['paired']
        summary['paired_comparison_scope'] = 'Same original-source common stock-date outcomes; posthoc joint method/data change, not causal attribution or fresh confirmation'
    if e.get('replay'):
        summary['replay_status'] = e['replay']
    (tables/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--destination',type=Path);args=parser.parse_args()
    print(json.dumps(build(args.root,args.destination),indent=2))
