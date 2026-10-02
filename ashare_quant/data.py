"""A-share source normalization with explicit units and time availability."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _dated(frame, name):
    out=frame.copy()
    if out.empty:
        return pd.DataFrame(columns=['ts_code','trade_date'])
    if not {'ts_code','trade_date'} <= set(out):
        raise ValueError(f'{name}: missing source keys')
    out['trade_date']=pd.to_datetime(out.trade_date,format='%Y%m%d',errors='raise')
    if out.duplicated(['ts_code','trade_date']).any():
        raise ValueError(f'{name}: duplicate security/date keys')
    return out


def normalize_daily(daily,basic,factors,limits):
    """Tushare hands/thousands/ten-thousands -> actual shares and CNY.

    No missing adjustment, valuation, limit or quote is silently filled.
    circ_mv is not free-float cap; true free float uses free_share * close.
    """
    source=_dated(daily,'daily')
    for name,other in [('basic',basic),('factors',factors),('limits',limits)]:
        right=_dated(other,name)
        overlap=(set(source)&set(right))-{'ts_code','trade_date'}
        right=right.drop(columns=list(overlap))
        source=source.merge(right,on=['ts_code','trade_date'],how='left',validate='one_to_one')
    get=lambda col: pd.to_numeric(source[col],errors='coerce') if col in source else pd.Series(np.nan,index=source.index)
    out=source[['ts_code','trade_date']].rename(columns={'ts_code':'security_id','trade_date':'date'})
    for s,t in [('open','raw_open'),('close','raw_close'),('high','high'),('low','low'),('pre_close','pre_close'),('adj_factor','adj_factor'),('up_limit','up_limit'),('down_limit','down_limit')]:out[t]=get(s)
    out['volume']=get('vol')*100.
    out['amount']=get('amount')*1000.
    for col in ['total_mv','circ_mv','total_share','float_share','free_share']:out[col]=get(col)*10000.
    out['free_mv']=out.free_share*out.raw_close
    out['turnover_ratio']=get('turnover_rate')/100.
    out['adj_close']=out.raw_close*out.adj_factor
    out['quote_present']=out.volume.gt(0)&out.amount.gt(0)&out.raw_close.gt(0)
    price_valid=out[['raw_open','raw_close','high','low']].gt(0).all(axis=1)
    out['data_valid']=price_valid&out.adj_factor.gt(0)&np.isfinite(out.adj_factor)&out.high.ge(out.low)
    audit=dict(rows=len(out),securities=int(out.security_id.nunique()),quote_rows=int(out.quote_present.sum()),
               missing_adjustment_rows=int(out.adj_factor.isna().sum()),missing_limit_rows=int(out[['up_limit','down_limit']].isna().any(axis=1).sum()),
               invalid_rows=int((~out.data_valid).sum()),source_units='daily: hands, CNY thousands; basic: shares/cap ten-thousands; turnover percent',
               output_units='shares and CNY; ratios decimal')
    return out.sort_values(['security_id','date']).reset_index(drop=True),audit


def normalize_securities(raw):
    out=raw.copy()
    valid=out.ts_code.astype(str).str.match(r'^(?:60\d{4}\.SH|68\d{4}\.SH|00\d{4}\.SZ|30\d{4}\.SZ|\d{6}\.BJ)$')
    out=out.loc[valid].rename(columns={'ts_code':'security_id'})
    if out.security_id.duplicated().any():
        raise ValueError('duplicate A-share identity requires explicit source review')
    out['exchange_code']=out.security_id.str[:6]
    for c in ['list_date','delist_date']:
        out[c]=pd.to_datetime(out[c] if c in out else pd.Series(pd.NaT,index=out.index),errors='coerce',format='%Y%m%d')
    out['source_list_date']=out.list_date
    bj=out.security_id.str.endswith('.BJ') & out.list_date.notna()
    out.loc[bj,'list_date']=out.loc[bj,'list_date'].clip(lower=pd.Timestamp('2021-11-15'))
    out['currency']='CNY';out['asset_type']='equity';out['identity_status']='verified'
    out['identity_valid_from']=out.list_date;out['identity_valid_to']=out.delist_date
    out['lot_size']=np.where(out.security_id.str.startswith(('688','689')),200,100)
    out['metadata_policy']='Current name/industry are display-only, never historical model inputs'
    return out.reset_index(drop=True)


def universe_asof(securities,day):
    day=pd.Timestamp(day)
    return securities.loc[securities.list_date.notna()&securities.list_date.le(day)&(securities.delist_date.isna()|securities.delist_date.gt(day))].copy()


def fundamentals_asof(observations,records,metrics):
    """Conservative date-only release lag; later restatements never backfill.

    Records must retain all known versions. This join cannot establish source
    version completeness and is not used by the baseline price-only model.
    """
    output=observations[['date','security_id']].copy().reset_index(drop=True)
    for c in metrics:output[c]=np.nan
    if records.empty:return output
    data=records.copy()
    date_column=data['f_ann_date'].where(data.f_ann_date.notna(),data.ann_date) if 'f_ann_date' in data else data.ann_date
    data['available_at']=pd.to_datetime(date_column,format='%Y%m%d',errors='coerce')+pd.Timedelta(days=1)
    data['period_end']=pd.to_datetime(data.end_date,format='%Y%m%d',errors='coerce')
    if data.duplicated(['security_id','available_at','period_end']).any():
        raise ValueError('Ambiguous same-release fundamental versions')
    for security,group in data.groupby('security_id',sort=False):
        latest=pd.Timestamp.min;events=[]
        for _,row in group.dropna(subset=['available_at','period_end']).sort_values(['available_at','period_end']).iterrows():
            if row.period_end>=latest:
                latest=row.period_end;events.append(row)
        if not events:continue
        events=pd.DataFrame(events).drop_duplicates('available_at',keep='last')
        idx=output.index[output.security_id.eq(security)]
        q=output.loc[idx,['date']].copy();q['original_index']=idx
        if q.empty:continue
        matched=pd.merge_asof(q.sort_values('date'),events[['available_at',*metrics]].sort_values('available_at'),left_on='date',right_on='available_at',direction='backward')
        output.loc[matched.original_index,metrics]=matched[metrics].to_numpy()
    return output
