"""市场、证券身份和带公开时间的数据整理。"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from .collect import write_json
from .paths import DATA


def join_market_sources(quotes, factors):
    quotes, factors = quotes.copy(), factors.copy()
    for frame in (quotes, factors):
        frame['trade_date'] = pd.to_datetime(frame.trade_date)
        if frame.duplicated(['ts_code', 'trade_date']).any():
            raise ValueError('证券日期重复，不能连接行情')
    joined = quotes.merge(factors, on=['ts_code', 'trade_date'], how='outer', validate='one_to_one', indicator=True)
    joined = joined.rename(columns={'ts_code': 'security_id', 'trade_date': 'date',
                                    'close_price': 'raw_close', 'vol': 'volume',
                                    'open': 'open_adj', 'high': 'high_adj', 'low': 'low_adj'})
    joined['adj_close'] = joined.raw_close * joined.cum_adjfactor
    joined['source_vwap'] = joined.vwap
    joined['vwap'] = joined.amount / joined.volume.where(joined.volume.gt(0))
    joined['quote_present'] = joined.volume.gt(0) & joined.amount.gt(0) & joined.vwap.gt(0)
    price_difference = (joined['close'] - joined.adj_close).abs()
    bad_adjustment = price_difference.gt(np.maximum(.02, joined.adj_close.abs() * .001))
    bad_vwap = (joined.source_vwap - joined.vwap).abs().gt(.005001)
    joined['data_valid'] = (joined.raw_close.gt(0) & joined.cum_adjfactor.gt(0)
                            & ~bad_adjustment & ~bad_vwap)
    audit = {'rows': len(joined), 'securities': int(joined.security_id.nunique()),
             'reference_only_rows': int(joined['_merge'].eq('right_only').sum()),
             'quote_rows': int(joined.quote_present.sum()),
             'missing_raw_price_rows': int(joined.raw_close.isna().sum()),
             'adjustment_mismatch_rows': int(bad_adjustment.sum()),
             'vwap_unit_mismatch_rows': int(bad_vwap.sum()),
             'nonpositive_factor_rows': int(joined.cum_adjfactor.le(0).sum())}
    joined = joined.drop(columns=['close', '_merge'])
    return joined.sort_values(['security_id', 'date']).reset_index(drop=True), audit


def financial_asof(observations, records):
    output = observations[['date', 'security_id']].copy().reset_index(drop=True)
    for metric in records.metric.unique():
        output[metric] = np.nan
    valid = records[records.verified & records.source_url.notna() & records.source_url.ne('')].copy()
    if valid.empty:
        return output
    valid['period_end'] = pd.to_datetime(valid.period_end)
    valid['published_at'] = pd.to_datetime(valid.published_at).dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    if valid.duplicated(['security_id', 'metric', 'period_end', 'published_at']).any():
        raise ValueError('同一财务版本包含冲突记录')
    for (security, metric), group in valid.groupby(['security_id', 'metric'], sort=False):
        indexes = output.index[output.security_id.eq(security)]
        if indexes.empty:
            continue
        timeline, latest_period = [], pd.Timestamp.min
        for row in group.sort_values(['published_at', 'period_end']).itertuples():
            if row.period_end >= latest_period:
                latest_period = row.period_end
                timeline.append({'available_at': row.published_at, 'value': row.value})
        events = pd.DataFrame(timeline).drop_duplicates('available_at', keep='last')
        query = output.loc[indexes, ['date']].copy()
        query['query_index'] = indexes
        query['cutoff'] = (query.date + pd.Timedelta(hours=19)).dt.tz_localize('Asia/Hong_Kong').dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
        events['available_at'] = events.available_at.astype('datetime64[ns, UTC]')
        matched = pd.merge_asof(query.sort_values('cutoff'), events.sort_values('available_at'),
                               left_on='cutoff', right_on='available_at', direction='backward')
        output.loc[matched.query_index, metric] = matched.value.to_numpy()
    return output


def canonical_currency(value):
    """HKEX人民币展示名RMB规范为ISO币种CNY；原展示名另存。"""
    return 'CNY' if isinstance(value,str) and value=='RMB' else value


def apply_asset_type_evidence(master,evidence):
    """按同一ISIN的证券类型证据细分供应商大类，所有身份行均保留。"""
    result=master.copy()
    result['provider_asset_type']=result.asset_type
    result['asset_type_evidence_status']='provider_category_only'
    result['asset_type_source_url']=None
    if evidence.security_id.duplicated().any():raise ValueError('证券类型证据身份重复')
    source=evidence.set_index('security_id')
    kinds={'ordinary_share':'equity','reit':'reit','preference':'preference',
           'depositary_receipt':'depositary_receipt','other_security_type':'other_security_type'}
    for index,row in result.iterrows():
        if row.security_id not in source.index:continue
        fact=source.loc[row.security_id]
        if fact.mapping_status!='matched_unique_isin':continue
        if pd.isna(row['isin']) or pd.isna(fact.matched_isin) or row['isin']!=fact.matched_isin:
            result.loc[index,'asset_type_evidence_status']='isin_mismatch';continue
        if fact.asset_class not in kinds:continue
        if not isinstance(fact.source_url,str) or not fact.source_url.startswith('https://'):
            raise ValueError('证券类型证据缺少来源')
        result.loc[index,'asset_type']=kinds[fact.asset_class]
        result.loc[index,'asset_type_evidence_status']='isin_matched_security_type'
        result.loc[index,'asset_type_source_url']=fact.source_url
    return result


def security_master(basic, observed, hkex, source_date):
    basic = basic.copy()
    basic['exchange_code'] = basic.ts_code.str.extract(r'^(\d{5})', expand=False) + '.HK'
    basic['list_date'] = pd.to_datetime(basic.list_date, format='%Y%m%d')
    basic['delist_date'] = pd.to_datetime(basic.delist_date, format='%Y%m%d')
    hkex = hkex.copy()
    hkex['exchange_code'] = hkex['Stock Code'].astype(str).str.zfill(5) + '.HK'
    hkex = hkex.set_index('exchange_code')
    records = []
    identifiers = set(observed.security_id) | set(basic.loc[basic.list_status.eq('L'), 'ts_code']) | set(hkex.index)
    intervals = observed.set_index('security_id')
    for identifier in sorted(identifiers):
        match = re.match(r'^(\d{5})', identifier)
        if not match:
            raise ValueError(f'未识别的证券标识：{identifier}')
        exchange_code = match.group(1) + '.HK'
        exact = basic[basic.ts_code.eq(identifier)]
        candidates = exact
        if identifier in intervals.index:
            first, last = intervals.loc[identifier, ['first_quote', 'last_quote']]
            lifecycle = basic.exchange_code.eq(exchange_code) & basic.list_date.le(first)
            lifecycle &= basic.delist_date.isna() | basic.delist_date.ge(last)
            candidates = basic[lifecycle]
        is_current_identifier = identifier == exchange_code
        current_conflict = False
        # 当前代码只确认自己的有效上市/转板期间，不继承代码下更早的源行情。
        if is_current_identifier and len(exact) == 1 and exact.iloc[0].list_status == 'L' and exchange_code in hkex.index:
            current = hkex.loc[exchange_code]
            basic_current = exact.iloc[0]
            same_identity = (isinstance(basic_current['isin'], str) and bool(basic_current['isin'].strip())
                             and basic_current['isin'] == current['ISIN']
                             and canonical_currency(basic_current.curr_type) == canonical_currency(current['Trading Currency']))
            if same_identity:
                candidates = exact
            else:
                candidates = basic.iloc[:0]
                current_conflict = True
        row = {'security_id': identifier, 'exchange_code': exchange_code, 'name': None,
               'isin': None, 'list_date': pd.NaT, 'delist_date': pd.NaT,
               'currency': None, 'asset_type': 'unresolved', 'lot_size': np.nan,
               'lot_valid_from': pd.NaT, 'lot_valid_to': pd.NaT,
               'identity_status': 'unresolved', 'metadata_source_date': source_date,
               'identity_valid_from': pd.NaT, 'identity_valid_to': pd.NaT,
               'identity_period_status': 'unresolved',
               'identity_period_basis': 'security identifier listing/transfer lifecycle; not issuer original IPO',
               'identity_period_gap_reason': '证券身份尚未核验',
               'source_first_quote': intervals.loc[identifier, 'first_quote'] if identifier in intervals.index else pd.NaT,
               'source_last_quote': intervals.loc[identifier, 'last_quote'] if identifier in intervals.index else pd.NaT}
        if len(candidates) == 1:
            source = candidates.iloc[0]
            row.update(name=source['name'], isin=source['isin'], list_date=source.list_date,
                       delist_date=source.delist_date, currency=source.curr_type,
                       asset_type='equity', identity_status='verified')
        can_use_current = identifier not in intervals.index or (len(candidates) == 1 and candidates.iloc[0].list_status == 'L')
        if is_current_identifier and can_use_current and not current_conflict and exchange_code in hkex.index:
            source = hkex.loc[exchange_code]
            row.update(name=source['Name of Securities'], isin=source['ISIN'],
                       currency=source['Trading Currency'], lot_size=float(str(source['Board Lot']).replace(',', '')),
                       lot_valid_from=pd.Timestamp(source_date), identity_status='verified',
                       asset_type='reit' if source['Category'] == 'Real Estate Investment Trusts' else 'equity')
        if row['identity_status'] == 'verified':
            if pd.notna(row['list_date']):
                row['identity_valid_from'] = row['list_date']
            else:
                row['identity_valid_from'] = pd.Timestamp(source_date)
                row['identity_period_basis'] = 'current HKEX snapshot only; historical identifier lifecycle unverified'
            row['identity_valid_to'] = row['delist_date']
            early = pd.notna(row['source_first_quote']) and row['source_first_quote'] < row['identity_valid_from']
            late = pd.notna(row['identity_valid_to']) and pd.notna(row['source_last_quote']) and row['source_last_quote'] >= row['identity_valid_to']
            row['identity_period_status'] = 'source_outside_verified_period' if early or late else 'verified_period'
            row['identity_period_gap_reason'] = '源行情含已核验证券期间之外的数据，未认定为该实体历史' if early or late else ''
        elif current_conflict:
            row['identity_period_gap_reason'] = '当前证券主表与官方HKEX的ISIN或币种冲突'
        row['currency_source_label']=row['currency']
        row['currency']=canonical_currency(row['currency'])
        records.append(row)
    return pd.DataFrame(records)


def prepare(data_root=DATA, start='20100101', end='20260909'):
    basic = pd.read_parquet(data_root / 'references/hk_basic.parquet')
    calendar = pd.read_parquet(data_root / 'references/calendar.parquet')
    open_dates = pd.DatetimeIndex(calendar.loc[calendar.is_open.eq(1), 'cal_date'])
    hkex_path = data_root / 'references/reit_research/equities_and_reits.csv'
    hkex = pd.read_csv(hkex_path, dtype={'Stock Code': str})
    hkex_source = json.loads((hkex_path.parent / 'sources.json').read_text(encoding='utf-8'))
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq='M')
    all_audits, intervals, year_parts = [], [], []
    output = data_root / 'bars'
    output.mkdir(parents=True, exist_ok=True)
    for period in periods:
        key = period.strftime('%Y%m')
        quotes = pd.read_parquet(data_root / f'source/hk_daily_adj/{key}.parquet')
        factors = pd.read_parquet(data_root / f'source/hk_adjfactor/{key}.parquet')
        bars, audit = join_market_sources(quotes, factors)
        audit['month'] = key
        outside_calendar = ~bars.date.isin(open_dates)
        audit['outside_calendar_rows'] = int(outside_calendar.sum())
        bars = bars[~outside_calendar & bars.date.between(pd.Timestamp(start), pd.Timestamp(end))]
        all_audits.append(audit)
        traded = bars[bars.quote_present]
        interval = traded.groupby('security_id').date.agg(first_quote='min', last_quote='max').reset_index()
        intervals.append(interval)
        year_parts.append(bars)
        if period.month == 12 or period == periods[-1]:
            year = pd.concat(year_parts, ignore_index=True).sort_values(['security_id', 'date'])
            year.to_parquet(output / f'{period.year}.parquet', index=False)
            print(f'prepared {period.year}: {len(year)} rows', flush=True)
            year_parts = []
    observed = pd.concat(intervals).groupby('security_id').agg(first_quote=('first_quote', 'min'),
                                                              last_quote=('last_quote', 'max')).reset_index()
    master = security_master(basic, observed, hkex, hkex_source['source_file_updated_at'])
    type_evidence=pd.read_parquet(data_root/'references/webb_archive/identity_mapping.parquet')
    master=apply_asset_type_evidence(master,type_evidence)
    master['primary_security_id'] = master.security_id
    for isin, group in master[master['isin'].notna()].groupby('isin'):
        current_hkd = group[group.currency.eq('HKD') & group.delist_date.isna()]
        if len(current_hkd) == 1:
            master.loc[group.index, 'primary_security_id'] = current_hkd.security_id.iloc[0]
    master.to_parquet(data_root / 'securities.parquet', index=False)
    fx_records = json.loads((data_root / 'references/fx_research/hkma_usd_cny_hkd_2010_present.json').read_text(encoding='utf-8'))['records']
    fx = pd.DataFrame(fx_records)
    fx['date'] = pd.to_datetime(fx.end_of_day)
    fx = fx.set_index('date')
    fx_gaps = 0
    for year in range(int(start[:4]), int(end[:4]) + 1):
        path = output / f'{year}.parquet'
        frame = pd.read_parquet(path).merge(master[['security_id', 'currency']], on='security_id', validate='many_to_one')
        frame['fx_to_hkd'] = np.nan
        frame.loc[frame.currency.eq('HKD'), 'fx_to_hkd'] = 1.
        for currency, field in [('USD', 'usd'), ('CNY', 'cny')]:
            mask = frame.currency.eq(currency)
            frame.loc[mask, 'fx_to_hkd'] = frame.loc[mask, 'date'].map(fx[field])
        fx_gaps += int((frame.quote_present & frame.fx_to_hkd.isna()).sum())
        frame['adj_close_hkd'] = frame.adj_close * frame.fx_to_hkd
        frame['amount_hkd'] = frame.amount * frame.fx_to_hkd
        frame.to_parquet(path, index=False)
    problems = master[master.identity_status.ne('verified') | master.identity_period_status.eq('source_outside_verified_period')]
    problems.to_csv(data_root / 'identity_gaps.csv', index=False)
    write_json(data_root / 'data_audit.json', {
        'start': start, 'end': end, 'monthly': all_audits,
        'rows': sum(a['rows'] for a in all_audits), 'observed_securities': len(observed),
        'registered_securities': len(master), 'unresolved_identities': int(master.identity_status.ne('verified').sum()),
        'identity_period_gap_securities': int(master.identity_period_status.eq('source_outside_verified_period').sum()),
        'identity_period_policy': 'Only verified security identifier periods are usable; earlier source prices are unresolved, not restored prior-board history or original IPO history',
        'quote_rows_missing_hkd_conversion': fx_gaps,
        'critical_gap_count': sum(a['adjustment_mismatch_rows'] + a['vwap_unit_mismatch_rows'] for a in all_audits),
        'future_leakage_detected': False,
        'current_lot_valid_from': hkex_source['source_file_updated_at'],
        'historical_lot_coverage_complete': False,
        'financial_point_in_time_coverage': 'requires verified publication timestamps and versions',
        'corporate_action_cash_coverage_complete': False,
        'reit_source_coverage': 'reported separately; not supplied by the Tushare equity endpoints'})
    return master


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='20100101')
    parser.add_argument('--end', required=True)
    args = parser.parse_args()
    prepare(start=args.start, end=args.end)
