"""CNY-specific adapter for tested market-neutral rolling factor mathematics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from hk_quant.features import security_features as _rolling_features
from hk_quant.market_context import market_context_from_inputs
from hk_quant.training import cross_sectional_inputs, sample_observations


def security_features(bars,calendar,*,quarantine_unresolved=True):
    """Compute the same price/volume factors without exporting HKD semantics.

    The legacy numeric routine requires three HK-named columns. Their ephemeral
    input values are CNY with unity conversion; no FX operation is performed.
    Internal names are removed immediately and never serialized.
    """
    if bars.empty or bars.security_id.nunique()!=1:raise ValueError('A single A-share identity is required')
    if bars.duplicated('date').any():raise ValueError('duplicate dates')
    frame=bars.sort_values('date').copy()
    if 'pre_close' in frame:
        implied=frame.raw_close.shift()*frame.adj_factor.shift()/frame.adj_factor
        discrepancy=(implied-frame.pre_close).abs()
        tolerance=np.maximum(.02,frame.pre_close.abs()*.003)
        bad=implied.notna() & frame.pre_close.notna() & discrepancy.gt(tolerance)
        if bad.any():
            dates=','.join(frame.loc[bad,'date'].dt.strftime('%Y-%m-%d').head(8))
            if not quarantine_unresolved:
                raise ValueError(f'Unreconciled adjustment reference for {frame.security_id.iloc[0]} on {dates}; review source before training')
            # Never invent a repaired factor. Each continuity segment has its
            # own observed source-price scale. Rolling inputs restart and
            # forward labels cannot cross an unexplained source discontinuity.
            issues=[dict(security_id=str(row.security_id),date=str(row.date.date()),
                         source_pre_close=float(row.pre_close),factor_implied_pre_close=float(implied.loc[index]))
                    for index,row in frame.loc[bad].iterrows()]
            pieces=[security_features(group,calendar,quarantine_unresolved=False)
                    for _,group in frame.groupby(bad.cumsum(),sort=False)]
            result=pd.concat(pieces,ignore_index=True).set_index('date')
            dates_index=pd.DatetimeIndex(calendar)
            dates_index=dates_index[(dates_index>=frame.date.min())&(dates_index<=frame.date.max())]
            result=result.reindex(dates_index)
            result['date']=dates_index;result['security_id']=str(frame.security_id.iloc[0])
            result['status']=result.status.fillna('no_trade_quote')
            result.loc[pd.DatetimeIndex(frame.loc[bad,'date']),'status']='unresolved_adjustment'
            result=result.reset_index(drop=True)
            result.attrs['adjustment_issues']=issues
            result.attrs['adjustment_policy']='quarantine discontinuities; reset history; no cross-break labels; raw source unchanged'
            return result
    frame['fx_to_hkd']=1.
    frame['adj_close_hkd']=frame.adj_close
    frame['amount_hkd']=frame.amount
    frame['open_adj']=frame.raw_open*frame.adj_factor
    frame['high_adj']=frame.high*frame.adj_factor
    frame['low_adj']=frame.low*frame.adj_factor
    result=_rolling_features(frame,calendar).drop(columns='fx_to_hkd').rename(columns={'adj_close_hkd':'adj_close_cny'})
    result.attrs['adjustment_issues']=[]
    return result


def feature_columns(frame):
    excluded={'date','security_id','raw_close','adj_close_cny','adv20_amount','status','quote_present','is_st'}
    return [c for c in frame if c not in excluded and not c.startswith(('fwd_return_','label_end_'))]
