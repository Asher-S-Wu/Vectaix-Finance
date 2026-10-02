"""Reconstruct source-separated original identities from raw OHLCV and actions."""
from __future__ import annotations
import numpy as np
import pandas as pd


def reconstruct_original_bars(raw,events,calendar,security_id,valid_start=None,valid_end=None):
    dates=pd.DatetimeIndex(calendar);f=raw.copy();f['date']=pd.to_datetime(f.date)
    if f.date.duplicated().any():raise ValueError('Duplicate original-identity quote dates')
    if not f.date.isin(dates).all():raise ValueError('Raw quote outside shared calendar')
    if valid_start is not None:f=f.loc[f.date.ge(pd.Timestamp(valid_start))]
    if valid_end is not None:f=f.loc[f.date.le(pd.Timestamp(valid_end))]
    f=f.set_index('date').reindex(dates)
    cols=['open','high','low','close','volume'];f[cols]=f[cols].apply(pd.to_numeric,errors='coerce')
    finite=np.isfinite(f[cols]).all(axis=1)
    o,h,l,c=[f[x] for x in ['open','high','low','close']]
    valid=finite&f[cols].gt(0).all(axis=1)&(h>=np.maximum(o,c))&(l<=np.minimum(o,c))&(h>=l)
    disagreement=pd.to_numeric(f.get('crosscheck_close_difference_usd',pd.Series(0.,index=dates)),errors='coerce').abs()
    valid&=~disagreement.gt(.001*c.abs())
    act=events.copy()
    if len(act):
        act['ex_date']=pd.to_datetime(act.ex_date)
        if act.ex_date.duplicated().any():raise ValueError('Combine separately verified same-day actions before reconstruction')
        if not act.ex_date.isin(dates).all():raise ValueError('Action outside shared calendar')
        if valid_start is not None:act=act.loc[act.ex_date.ge(pd.Timestamp(valid_start))]
        if valid_end is not None:act=act.loc[act.ex_date.le(pd.Timestamp(valid_end))]
        act=act.set_index('ex_date').reindex(dates)
        splits=act.split_ratio_new_per_old.fillna(1.).astype(float)
        cash=act.cash_usd_per_pre_event_share.fillna(0.).astype(float)
    else:
        splits=pd.Series(1.,index=dates);cash=pd.Series(0.,index=dates)
    if (~np.isfinite(splits)|splits.le(0)|~np.isfinite(cash)|cash.lt(0)).any():raise ValueError('Invalid split/cash action')
    action=splits.ne(1)|cash.ne(0);previous=c.shift();previous_valid=valid.shift(fill_value=False)
    if (action&(~previous_valid|previous.le(cash))).any():raise ValueError('Action requires a valid prior-session raw close and positive ex-distribution reference')
    multipliers=pd.Series(1.,index=dates)
    multipliers.loc[action]=splits.loc[action]*previous.loc[action]/(previous.loc[action]-cash.loc[action])
    adjustment=multipliers.cumprod();split_level=splits.cumprod();split_factor=split_level/split_level.iloc[-1]
    out=pd.DataFrame({'date':dates,'security_id':security_id},index=dates)
    for column in ['open','high','low','close']:out['split_adjusted_'+column]=(f[column]*split_factor).where(valid)
    out['volume']=(f.volume/split_factor).where(valid)
    out['adj_close']=(c*adjustment).where(valid)
    out['dollar_volume_proxy']=(c*f.volume).where(valid)
    out['quote_present']=valid;out['data_valid']=valid
    out['adjustment_factor']=adjustment
    return out.reset_index(drop=True)
