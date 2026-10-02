"""Backward-only US price/volume features on one shared session calendar."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import HORIZONS

PRICE_FEATURES=['return_1','history_sessions',*[f'momentum_{d}' for d in (5,20,60,120,252)],'momentum_252_skip20',*[f'volatility_{d}' for d in (20,60,252)],'downside_volatility_60','max_return_20','return_skew_60','near_high_252','near_low_252','bias_20','bias_60','log_amount_20','amount_ratio_20_120','amount_cv_20','volume_cv_20','amihud_20','intraday_range','intraday_return','sessions_since_quote']
MARKET_FEATURES=['market_momentum_20','market_momentum_60','market_momentum_252','market_volatility_60','market_breadth_60']
FEATURE_COLUMNS=PRICE_FEATURES
ALL_FEATURES=PRICE_FEATURES+MARKET_FEATURES


def security_features(bars,calendar):
    if bars.empty or bars.security_id.nunique()!=1:raise ValueError('One security required')
    if bars.date.duplicated().any():raise ValueError('Duplicate dates')
    dates=pd.DatetimeIndex(calendar)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:raise ValueError('Invalid shared calendar')
    frame=bars.set_index('date').reindex(dates)
    quoted=frame.quote_present.eq(True)&frame.data_valid.eq(True)
    price=frame.adj_close.where(quoted);volume=frame.volume.where(quoted)
    amount=frame.dollar_volume_proxy.where(quoted)
    returns=price.pct_change(fill_method=None);log_return=np.log(price/price.shift())
    out=pd.DataFrame({'date':dates,'security_id':bars.security_id.iloc[0]},index=dates)
    out['adj_close']=price;out['quote_present']=quoted;out['return_1']=returns
    out['history_sessions']=quoted.cumsum();out['adv20_amount']=amount.rolling(20,min_periods=12).mean()
    for d in (5,20,60,120,252):out[f'momentum_{d}']=price/price.shift(d)-1
    out['momentum_252_skip20']=price.shift(20)/price.shift(252)-1
    for d in (20,60,252):out[f'volatility_{d}']=log_return.rolling(d,min_periods=max(12,d//2)).std()*np.sqrt(252)
    out['downside_volatility_60']=log_return.clip(upper=0).pow(2).rolling(60,min_periods=30).mean().pow(.5)*np.sqrt(252)
    out['max_return_20']=returns.rolling(20,min_periods=12).max();out['return_skew_60']=returns.rolling(60,min_periods=30).skew()
    out['near_high_252']=price/price.rolling(252,min_periods=60).max()-1
    out['near_low_252']=price/price.rolling(252,min_periods=60).min()-1
    for d in (20,60):out[f'bias_{d}']=price/price.rolling(d,min_periods=max(12,d//2)).mean()-1
    out['log_amount_20']=np.log(out.adv20_amount.where(out.adv20_amount>0))
    out['amount_ratio_20_120']=out.adv20_amount/amount.rolling(120,min_periods=60).mean()
    out['amount_cv_20']=amount.rolling(20,min_periods=12).std()/out.adv20_amount
    out['volume_cv_20']=volume.rolling(20,min_periods=12).std()/volume.rolling(20,min_periods=12).mean()
    out['amihud_20']=(returns.abs()/amount).rolling(20,min_periods=12).mean()*1e8
    out['intraday_range']=(frame.split_adjusted_high-frame.split_adjusted_low)/frame.split_adjusted_close
    out['intraday_return']=frame.split_adjusted_close/frame.split_adjusted_open-1
    positions=np.arange(len(frame));last=np.maximum.accumulate(np.where(quoted,positions,-1))
    out['sessions_since_quote']=positions-last
    out['status']='ok';out.loc[out.history_sessions<60,'status']='insufficient_history'
    out.loc[out.adv20_amount<10_000_000,'status']='below_liquidity_floor'
    out.loc[~quoted,'status']='no_trade_quote'
    for h in HORIZONS:
        out[f'fwd_return_{h}']=price.shift(-h)/price-1
        out[f'label_end_{h}']=pd.Series(dates,index=dates).shift(-h)
    out[out.select_dtypes('number').columns]=out.select_dtypes('number').replace([np.inf,-np.inf],np.nan)
    return out.reset_index(drop=True)


def add_market_context(frame,calendar):
    from hk_quant.market_context import market_context_from_inputs
    # The fixed cohort is intentionally the market-context scope, not all US.
    context=market_context_from_inputs(frame,calendar)
    return frame.merge(context,on='date',how='left',validate='many_to_one')


def normalize_features(frame):
    features=[c for c in ALL_FEATURES if c in frame]
    from hk_quant.training import cross_sectional_inputs
    typed=frame.copy()
    typed[features]=typed[features].astype(float)
    return cross_sectional_inputs(typed,features)
