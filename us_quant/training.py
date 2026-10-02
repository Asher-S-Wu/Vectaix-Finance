"""Immutable US chronological train/calibration/development/confirmation protocol."""
from __future__ import annotations
import numpy as np
import pandas as pd
from hk_quant.training import make_examples


def split_examples(frame,features,calibration_start='2023-01-01',as_of='2023-12-29',train_start='2016-01-01'):
    examples=make_examples(frame.loc[frame.date.ge(pd.Timestamp(train_start))&frame.status.eq('ok')],features)
    mature=examples.fwd_return.notna()&examples.label_end.notna()&examples.sigma_daily.gt(0)
    boundary=pd.Timestamp(calibration_start)
    train=examples.loc[mature&examples.date.lt(boundary)&examples.label_end.lt(boundary)].reset_index(drop=True)
    cal=examples.loc[mature&examples.date.ge(boundary)&examples.label_end.le(pd.Timestamp(as_of))].reset_index(drop=True)
    return train,cal


def protocol(universe_sha256):
    from .features import ALL_FEATURES
    from hk_quant.training import FACTOR_FEATURES,LINEAR_FEATURES,KINDS
    return dict(version='us-oef2015-yahoo-chronological-v1',market='US',currency='USD',
        declared_cohort_count=101,universe_sha256=universe_sha256,universe_date='2015-06-30',universe_known_by='2015-09-02',
        history_start='2014-01-01',training_start='2016-01-01',calibration_start='2023-01-01',training_as_of='2023-12-29',
        development_start='2024-01-01',development_end='2024-12-31',confirmation_start='2025-01-01',confirmation_end='2026-09-30',
        source='Yahoo Finance daily public chart, current-vintage retrospective data',point_in_time_vendor_vintages=False,
        latest_refit=dict(train_start='2016-01-01',calibration_start='2025-10-01',as_of='2026-09-30',architecture='frozen development-selected kind',out_of_sample_evaluated=False),
        survivorship_free=False,full_us_market=False,horizons=[1,5,20,60],
        price_basis='split-and-distribution-adjusted close returns; split-adjusted OHLC intraday ratios',
        volume_basis='source split-adjusted historical volume; split-adjusted close times volume USD trading-value proxy',
        corporate_action_limit='Provider adjustments not a complete independently verified historical action or terminal-value ledger',
        identity_policy='Original fixed cohort; no successor substitution; unresolved complex lineages and missing histories retained in denominator',
        features=ALL_FEATURES,factor_features=list(FACTOR_FEATURES),linear_features=list(LINEAR_FEATURES),
        feature_availability_rule='For tree inputs, >=20% nonmissing on eligible 2016–2022 training dates only',
        contemporaneous_eligibility=dict(min_history_sessions=60,min_adv20_usd_proxy=10_000_000,no_future_endpoint_requirement=True),
        preprocessing='same-session cross-sectional percentile ranks except market context; unranked sigma; train-only Ridge scaler',
        candidates=list(KINDS),candidate_parameters=dict(factor_weights=[.25,-.25,-.25,.25],ridge_alpha=10.,
            small=dict(num_leaves=15,n_estimators=160,min_child_samples=100,reg_lambda=5.,learning_rate=.05,n_jobs=8,random_state=42),
            large=dict(num_leaves=31,n_estimators=240,min_child_samples=100,reg_lambda=5.,learning_rate=.04,n_jobs=8,random_state=42)),
        score_heads='factor score; Ridge regression; LightGBM 20-session LambdaRank head plus regression scores at other horizons',
        target='arithmetic forward return / sqrt(horizon); no label winsorization',
        purge='train labels end before 2023-01-01; calibration labels end by 2023-12-29; development labels end by 2024-12-31',
        selection_rule='largest 20-session mean daily rank IC on identical score-available common rows and shared dates',
        minimum_ic_cross_section=20,constant_prediction_policy='zero ranking skill on otherwise eligible shared day',
        tie_break=list(KINDS),bootstrap=dict(method='circular session-block bootstrap, preserve missing-day slots',block_sessions=60,repetitions=2000,seed=42),
        replay=dict(signal='last calendar session monthly',execution='next shared session close',top_n=10,tie_break='security_id ascending',
            invested_fraction=.95,initial_cash_usd=1_000_000,fee_slippage_per_side=.0015,stress_fee_slippage_per_side=.003,
            max_participation=.01,cash_return=0.,leverage=False,shorts=False,units='fractional source-adjusted research units',
            missing_fill='no hindsight replacement',terminal_policy='retain unknown holdings; stale marks reference-only; affected path metrics invalid',
            missing_daily_value='invalidates path statistics, even when a later quote reappears'))


def compare_candidates(predictions,cutoff):
    import hashlib
    frames={};common=None
    for kind,frame in predictions.items():
        f=frame.loc[frame.horizon.eq(20)&frame.date.le(pd.Timestamp(cutoff))].set_index(['date','security_id'])
        if f.index.has_duplicates:raise ValueError('Duplicate prediction keys')
        frames[kind]=f
        available=f.index[f.score_status.eq('ok')&np.isfinite(f.score)]
        common=available if common is None else common.intersection(available)
    if common is None or not len(common):raise ValueError('No common predictions')
    common=common.sort_values();reference=next(iter(frames.values())).reindex(common)
    mature=reference.label_end.le(pd.Timestamp(cutoff))&np.isfinite(reference.fwd_return)
    labels=reference.loc[mature,'fwd_return']
    daykeys=[]
    for date,values in labels.groupby(level='date'):
        if len(values)>=20 and values.nunique()>1:daykeys.append(date)
    dates_digest=hashlib.sha256('|'.join(str(d.date()) for d in daykeys).encode()).hexdigest()
    if not daykeys:raise ValueError('No comparable IC dates')
    results={}
    for kind,frame in frames.items():
        f=frame.reindex(common)
        if not np.allclose(f.fwd_return,reference.fwd_return,equal_nan=True) or not f.label_end.equals(reference.label_end):raise ValueError('Candidate labels disagree')
        observed=f.loc[mature];ics=[];constants=0
        for date in daykeys:
            day=observed.xs(date,level='date')
            constant=day.score.nunique()<2;constants+=int(constant)
            ics.append(0. if constant else float(day.score.corr(day.fwd_return,method='spearman')))
        results[kind]=dict(rank_ic_mean=float(np.mean(ics)),ic_dates=len(daykeys),date_sha256=dates_digest,constant_score_days=constants,
            common_score_rows=len(common),common_mature_rows=int(mature.sum()),universe='common contemporaneously score-available original-cohort identities',
            outcome_policy='conditional observed mature endpoints; missing outcomes remain disclosed')
    return results


def freeze_selection(scores,path):
    import json,hashlib
    from hk_quant.training import KINDS
    from .data import write_immutable_json
    candidates=[(float(scores[k]['rank_ic_mean']),-i,k) for i,k in enumerate(KINDS) if np.isfinite(scores[k]['rank_ic_mean'])]
    if not candidates:raise ValueError('No candidate metrics')
    selected=max(candidates)
    result=dict(selected_kind=selected[2],selection_statistic=selected[0],selection_set='development',confirmation_used=False,
        selection_rule='maximum common-row/common-date 20-session mean daily rank IC',
        development_sha256=hashlib.sha256(json.dumps(scores,sort_keys=True,allow_nan=False).encode()).hexdigest())
    write_immutable_json(path,result);return result
