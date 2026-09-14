"""逐日因子与四期限标签，所有滚动计算只使用过去的数据。"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import HORIZONS
from .contracts import FORECAST_RETURN_BASIS, CASH_DIVIDEND_ACCOUNTING
from .collect import write_json
from .data import financial_asof
from .dated_identity import identity_row_mask
from .paths import DATA, ROOT

FINANCIAL_METRICS = ('eps_hkd', 'bps_hkd', 'roe', 'net_margin', 'revenue_growth',
                     'profit_growth', 'operating_cashflow_to_assets', 'debt_to_assets')


def write_security_features(frame, directory):
    """原身份保留在数据中；文件名编码适配Windows而不合并不同证券。"""
    if frame.empty or frame.security_id.isna().any() or frame.security_id.nunique()!=1:
        raise ValueError('因子分区必须包含单一证券身份')
    security=frame.security_id.iloc[0]
    if not isinstance(security,str) or not re.fullmatch(r'(?:\d{5}(?:![A-Z0-9]*)?\.HK|HKREIT:[1-9]\d*)',security):
        raise ValueError(f'证券标识不符合已注册数据源身份格式: {security}')
    path=Path(directory)/f'{quote(security,safe=".!-")}.parquet'
    frame.to_parquet(path,index=False,row_group_size=252)
    return path


def reit_features_asof(bars, calendar, event_versions, cutoff, quote_currency):
    """从原始量价和当时公告构建单日输入；收益标签必须另行按成熟时间生成。"""
    from .reit_adjustment import adjusted_history_asof
    result=adjusted_history_asof(bars,calendar,event_versions,cutoff,quote_currency)
    frame=_reit_features_for_state(result,calendar).iloc[[-1]].copy()
    frame.attrs.update(as_of=pd.Timestamp(cutoff).isoformat(),coverage_complete=False,
                       approved_for_training=False,return_basis=result.return_basis,
                       adjustment_gaps=json.loads(result.gaps.to_json(orient='records',date_format='iso')))
    return frame.reset_index(drop=True)


def _reit_features_for_state(result,calendar):
    """内部计算：调用方仅可取该公告状态实际有效日期内的行。"""
    history=result.history.copy()
    if history.empty:
        raise ValueError('截止时点没有已完成日线，无法构建因子')
    history['adj_close']=history.adjusted_close
    history['adj_close_hkd']=history.adj_close*history.fx_to_hkd
    for source,target in [('raw_open','open_adj'),('high','high_adj'),('low','low_adj')]:
        history[target]=history[source]*history.adjustment_factor
    history['data_valid']=history.data_valid.eq(True)&history.adjustment_status.eq('resolved')
    frame=security_features(history,calendar)
    frame=frame.drop(columns=[c for c in frame if c.startswith(('fwd_return_','label_end_'))])
    # Longest price dependency is a 252-session return/volatility window,
    # including its starting reference price. Older gaps remain in the audit.
    sessions=pd.DatetimeIndex(calendar)
    flags=history.set_index('date').adjustment_status.eq('unresolved').reindex(sessions).eq(True)
    affected=flags.rolling(253,min_periods=1).max()
    frame.loc[frame.date.map(affected).eq(1),'status']='adjustment_unresolved'
    return frame


def reit_forward_labels(bars, calendar, event_versions, signal_date, cutoff, quote_currency):
    """每个期限在自身结束日19:00的信息集上定义收益，不回填后续修订。"""
    from .reit_adjustment import adjusted_history_asof
    cutoff=pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError('标签截止时间必须包含时区')
    sessions=pd.DatetimeIndex(calendar)
    signal=pd.Timestamp(signal_date)
    if signal not in sessions or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError('信号日期必须属于有序且唯一的交易日历')
    if bars.security_id.nunique()!=1:raise ValueError('标签必须属于单一证券')
    row=dict(date=signal,security_id=bars.security_id.iloc[0]);states={}
    position=sessions.get_loc(signal)
    for horizon in HORIZONS:
        end=sessions[position+horizon] if position+horizon<len(sessions) else pd.NaT
        row[f'label_end_{horizon}']=end
        row[f'fwd_return_{horizon}']=np.nan
        if pd.isna(end):
            states[str(horizon)]='no_completed_horizon';continue
        known=end.tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19)
        if known>cutoff:
            states[str(horizon)]='not_mature';continue
        snapshot=adjusted_history_asof(bars,sessions,event_versions,known,quote_currency)
        prices=snapshot.history.set_index('date')
        if signal not in prices.index or end not in prices.index:
            states[str(horizon)]='missing_endpoint';continue
        endpoints=prices.loc[[signal,end]]
        legal=(endpoints.data_valid.eq(True)&endpoints.adjustment_status.eq('resolved')
               &endpoints.fx_to_hkd.gt(0)&np.isfinite(endpoints.fx_to_hkd)
               &endpoints.adjusted_close.gt(0)&np.isfinite(endpoints.adjusted_close))
        if not legal.all():
            states[str(horizon)]='unresolved_price_adjustment_or_fx';continue
        values=endpoints.adjusted_close*endpoints.fx_to_hkd
        row[f'fwd_return_{horizon}']=float(values.iloc[1]/values.iloc[0]-1)
        states[str(horizon)]='available_at_horizon_end'
    output=pd.DataFrame([row])
    output.attrs.update(label_states=states,approved_for_training=False,coverage_complete=False,
                        label_publication_policy='Each horizon uses its own end-day 19:00 information set; later revisions are not backfilled.')
    return output


def security_features(bars, calendar):
    security = bars.security_id.iloc[0]
    original = bars.sort_values('date').set_index('date')
    dates = pd.DatetimeIndex(calendar)
    dates = dates[(dates >= original.index.min()) & (dates <= original.index.max())]
    frame = original.reindex(dates)
    price = frame.adj_close_hkd.where(frame.data_valid.eq(True))
    quoted = frame.quote_present.isin([True])
    # 已有参考估值可反映复牌跳变，但绝不代表存在成交。
    returns = price.pct_change(fill_method=None)
    log_return = np.log(price / price.shift(1))
    amount = frame.amount_hkd.where(quoted)
    volume = frame.volume.where(quoted)
    out = pd.DataFrame(index=dates)
    out['date'] = dates
    out['security_id'] = security
    out['raw_close'] = frame.raw_close
    out['fx_to_hkd'] = frame.fx_to_hkd
    out['adj_close_hkd'] = price
    out['return_1'] = returns
    out['history_sessions'] = quoted.cumsum()
    out['adv20_amount'] = amount.rolling(20, min_periods=12).mean() / frame.fx_to_hkd
    for days in (5, 20, 60, 120, 252):
        out[f'momentum_{days}'] = price / price.shift(days) - 1
    out['momentum_252_skip20'] = price.shift(20) / price.shift(252) - 1
    for days in (20, 60, 252):
        minimum = max(12, days // 2)
        out[f'volatility_{days}'] = log_return.rolling(days, min_periods=minimum).std() * np.sqrt(252)
    out['downside_volatility_60'] = log_return.clip(upper=0).pow(2).rolling(60, min_periods=30).mean().pow(.5) * np.sqrt(252)
    out['max_return_20'] = returns.rolling(20, min_periods=12).max()
    out['return_skew_60'] = returns.rolling(60, min_periods=30).skew()
    out['near_high_252'] = price / price.rolling(252, min_periods=60).max() - 1
    out['near_low_252'] = price / price.rolling(252, min_periods=60).min() - 1
    out['bias_20'] = price / price.rolling(20, min_periods=12).mean() - 1
    out['bias_60'] = price / price.rolling(60, min_periods=30).mean() - 1
    out['log_amount_20'] = np.log(amount.rolling(20, min_periods=12).mean().where(lambda x:x>0))
    out['amount_ratio_20_120'] = amount.rolling(20, min_periods=12).mean() / amount.rolling(120, min_periods=60).mean()
    out['amount_cv_20'] = amount.rolling(20, min_periods=12).std() / amount.rolling(20, min_periods=12).mean()
    out['volume_cv_20'] = volume.rolling(20, min_periods=12).std() / volume.rolling(20, min_periods=12).mean()
    out['amihud_20'] = (returns.abs() / amount).rolling(20, min_periods=12).mean() * 1e8
    out['intraday_range'] = (frame.high_adj - frame.low_adj) / frame.adj_close
    out['intraday_return'] = frame.adj_close / frame.open_adj - 1
    out['log_market_cap'] = np.log((frame.total_mv * frame.fx_to_hkd).where(lambda x:x>0))
    out['log_free_market_cap'] = np.log((frame.free_mv * frame.fx_to_hkd).where(lambda x:x>0))
    out['free_share_ratio'] = frame.free_share / frame.total_share.where(lambda x:x>0)
    out['turnover_20'] = frame.turnover_ratio.rolling(20, min_periods=12).mean()
    out['share_change_60'] = frame.total_share / frame.total_share.shift(60) - 1
    out['share_change_252'] = frame.total_share / frame.total_share.shift(252) - 1
    positions = np.arange(len(frame))
    last_quote = np.maximum.accumulate(np.where(quoted, positions, -1))
    out['sessions_since_quote'] = positions - last_quote
    out['status'] = 'ok'
    out.loc[out.history_sessions < 60, 'status'] = 'insufficient_history'
    out.loc[~quoted, 'status'] = 'no_trade_quote'
    out.loc[~frame.data_valid.eq(True) | price.isna(), 'status'] = 'data_issue'
    global_dates = pd.Series(pd.DatetimeIndex(calendar), index=pd.DatetimeIndex(calendar))
    for horizon in HORIZONS:
        out[f'fwd_return_{horizon}'] = price.shift(-horizon) / price - 1
        out[f'label_end_{horizon}'] = global_dates.shift(-horizon).reindex(dates)
    numeric = out.select_dtypes('number').columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


def index_rate_features(directory, calendar):
    result = pd.DataFrame({'date': calendar})
    for code, prefix, first_public in [('HSI','hsi','1969-11-24'), ('HKTECH','hktech','2020-07-27')]:
        frames = [pd.read_parquet(path) for path in sorted(directory.glob(f'{code}_*.parquet'))]
        quotes = pd.concat(frames, ignore_index=True)
        quotes['date'] = pd.to_datetime(quotes.trade_date, format='%Y%m%d')
        if quotes.date.duplicated().any():
            raise ValueError(f'{code}指数日期重复')
        close = quotes.set_index('date').close.reindex(calendar)
        close = close.where(close.index >= pd.Timestamp(first_public))
        for length in (20,60,252):
            result[f'market_{prefix}_momentum_{length}'] = (close/close.shift(length)-1).to_numpy()
        result[f'market_{prefix}_volatility_60'] = (np.log(close/close.shift(1)).rolling(60,min_periods=30).std()*np.sqrt(252)).to_numpy()
    rates = pd.concat([pd.read_parquet(path) for path in sorted(directory.glob('hibor_*.parquet'))], ignore_index=True)
    rates['date'] = pd.to_datetime(rates.date, format='%Y%m%d')
    if rates.date.duplicated().any():
        raise ValueError('HIBOR日期重复')
    rates = rates.set_index('date').reindex(calendar)
    result['market_hibor_3m'] = (rates['3m']/100).to_numpy()
    result['market_hibor_change_20'] = ((rates['3m']-rates['3m'].shift(20))/100).to_numpy()
    result['market_hibor_curve'] = ((rates['12m']-rates['on'])/100).to_numpy()
    return result


def offer_features(frame, offers, exchange_code, listing_date=None, delisting_date=None):
    relevant = offers[offers.code.eq(exchange_code)]
    if pd.notna(listing_date):
        relevant = relevant[relevant.offer_start.ge(listing_date)]
    if pd.notna(delisting_date):
        relevant = relevant[relevant.offer_start.lt(delisting_date)]
    frame['offer_active'] = np.nan
    frame['offer_started_60d'] = np.nan
    covered = frame.date.ge('2017-01-01')
    frame.loc[covered, ['offer_active','offer_started_60d']] = 0.
    for row in relevant.itertuples():
        known = frame.date.gt(row.offer_start)
        active = known & (pd.isna(row.offer_end) | frame.date.le(row.offer_end))
        recent = known & frame.date.le(row.offer_start + pd.Timedelta(days=60))
        frame.loc[active, 'offer_active'] = 1.
        frame.loc[recent, 'offer_started_60d'] = 1.
    return frame


def isolate_identity_periods(bars, master):
    """保留源日期和证券行，只让已核验期间的量价进入计算。"""
    scoped = bars.copy()
    valid = identity_row_mask(scoped, master)
    numeric = scoped.select_dtypes(include='number').columns.difference(['identity_source_issue_id'])
    scoped[numeric] = scoped[numeric].where(valid, axis=0)
    scoped.loc[~valid, ['quote_present','data_valid']] = False
    return scoped


def identity_period_features(group, calendar, metadata):
    """先隔离身份再计算滚动量价和未来标签，期间外保留缺口行。"""
    master = pd.DataFrame([metadata]).set_index('security_id')
    frame = security_features(isolate_identity_periods(group, master), calendar)
    proof = pd.Series(identity_row_mask(group, master).to_numpy(), index=group.date)
    valid = frame.date.map(proof).eq(True)
    columns = frame.columns.difference(['date','security_id','status'])
    frame[columns] = frame[columns].where(valid, axis=0)
    frame.loc[~valid, 'status'] = ('identity_period_unresolved' if metadata.identity_status in ('verified', 'partially_verified')
                                 else 'identity_unresolved')
    return frame


def build_features(data_root=DATA):
    audit = json.loads((data_root / 'data_audit.json').read_text(encoding='utf-8'))
    master = pd.read_parquet(data_root / 'securities.parquet').set_index('security_id')
    calendar_frame = pd.read_parquet(data_root / 'references/calendar.parquet')
    calendar = pd.DatetimeIndex(calendar_frame.loc[calendar_frame.is_open.eq(1), 'cal_date'])
    calendar = calendar[(calendar>=pd.Timestamp(audit['start'])) & (calendar<=pd.Timestamp(audit['end']))]
    bars = pd.concat([pd.read_parquet(path) for path in sorted((data_root / 'bars').glob('*.parquet'))], ignore_index=True)
    bars = isolate_identity_periods(bars[bars.date.isin(calendar)], master)
    offers = pd.read_csv(ROOT / 'data/hk/raw/corporate_events_research/sfc_offer_periods_all.csv',
                         parse_dates=['offer_start','offer_end'])
    financials = pd.read_parquet(data_root / 'financial_versions.parquet')
    output = data_root / 'features'
    output.mkdir(parents=True, exist_ok=True)
    latest, availability, rows, training_rows = [], [], 0, 0
    written=[]
    feature_columns = None
    for number, (security, group) in enumerate(bars.groupby('security_id', sort=True), 1):
        metadata = master.loc[security].copy()
        metadata['security_id'] = security
        if metadata.asset_type=='reit':
            frame=pd.read_parquet(data_root/'reit_price_features'/f'{quote(security,safe=".!-")}.parquet')
            frame=frame.loc[frame.date.isin(calendar)].copy()
            if frame.empty or not frame.security_id.eq(security).all() or frame.date.duplicated().any():
                raise ValueError(f'REIT时点因子身份或日期无效: {security}')
            expected=calendar[(calendar>=group.date.min())&(calendar<=group.date.max())]
            if not expected.isin(frame.date).all():raise ValueError(f'REIT时点因子缺交易日记录: {security}')
            valid=(metadata.identity_status=='verified')&frame.date.ge(metadata.identity_valid_from)
            if pd.notna(metadata.identity_valid_to):valid &= frame.date.lt(metadata.identity_valid_to)
            numeric=frame.select_dtypes('number').columns
            frame[numeric]=frame[numeric].where(valid,axis=0)
            frame.loc[~valid,'status']='identity_period_unresolved'
        else:
            frame = identity_period_features(group, calendar, metadata)
        if metadata.asset_type not in ('equity','reit'):
            frame['status']='outside_model_asset_scope' if metadata.asset_type!='unresolved' else 'asset_type_unresolved'
        frame = offer_features(frame, offers, metadata.exchange_code, metadata.list_date, metadata.delist_date)
        values = financial_asof(frame, financials)
        for metric in FINANCIAL_METRICS:
            frame[metric] = values[metric] if metric in values else np.nan
        close_hkd = frame.raw_close * frame.fx_to_hkd
        frame['earnings_yield'] = frame.eps_hkd / close_hkd
        frame['book_to_price'] = frame.bps_hkd / close_hkd
        excluded = {'date','security_id','raw_close','fx_to_hkd','adj_close_hkd','adv20_amount','status'}
        feature_columns = [c for c in frame if c not in excluded and not c.startswith(('fwd_return_','label_end_'))]
        frame[feature_columns] = frame[feature_columns].astype(np.float32)
        written.append(write_security_features(frame,output))
        rows += len(frame)
        training_rows += int(frame.status.eq('ok').sum())
        if number % 100 == 0:
            print(f'features {number}/{bars.security_id.nunique()} securities; {rows} rows', flush=True)
    from .market_context import market_context_from_inputs
    price_inputs=pd.concat([pd.read_parquet(path,columns=['date','security_id','return_1','bias_60']) for path in written],ignore_index=True)
    price_inputs=price_inputs.merge(bars[['date','security_id','quote_present']],on=['date','security_id'],how='left',validate='one_to_one')
    in_scope=price_inputs.security_id.map(master.asset_type).isin(['equity','reit'])
    price_inputs.loc[~in_scope,['return_1','bias_60']]=np.nan
    market=market_context_from_inputs(price_inputs,calendar)
    market=market.merge(index_rate_features(data_root/'references/market',calendar),on='date',validate='one_to_one')
    feature_columns+=list(market.columns.difference(['date']))
    for path in written:
        frame=pd.read_parquet(path).merge(market,on='date',validate='one_to_one')
        frame[feature_columns]=frame[feature_columns].astype(np.float32)
        write_security_features(frame,output)
        availability.append(frame[feature_columns].notna().sum())
        latest.append(frame.iloc[[-1]])
    pd.concat(latest, ignore_index=True).to_parquet(data_root / 'latest_features.parquet', index=False)
    coverage = pd.DataFrame(availability).sum() / rows
    write_json(data_root / 'feature_manifest.json', {
        'data_as_of': audit['end'], 'start_date': audit['start'], 'securities': len(latest),
        'rows': rows, 'eligible_rows': training_rows, 'features': feature_columns,
        'coverage': coverage.to_dict(), 'horizons': list(HORIZONS),
        'return_basis': FORECAST_RETURN_BASIS, 'cash_dividend_accounting': CASH_DIVIDEND_ACCOUNTING,
        'market_return_source':'daily_information_set_return_1_and_bias_60',
        'valuation_return_policy': 'Use explicitly supplied, valid source valuations to retain resumption jumps; reference prices do not establish tradability. Missing prices remain missing.',
        'financial_rows_verified': int(financials.verified.sum()),
        'signal_cutoff_local_time': '19:00 Asia/Hong_Kong'})


if __name__ == '__main__':
    build_features()
