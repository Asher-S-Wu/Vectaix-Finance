"""固定分钟来源段的REIT日线研究与CCASS对照；不注入训练数据。"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'data/hk/universal'
CCASS_CUTOFF=pd.Timestamp('2025-12-24')
NUMERIC=('open','high','low','close','vol','amount')


def frequency_for_date(day):
    day=pd.Timestamp(day)
    if pd.Timestamp('2011-01-01')<=day<=pd.Timestamp('2017-06-30'):return '1min'
    if pd.Timestamp('2017-07-01')<=day:return '60min'
    return None


def aggregate_minutes(frame,frequency,source_file):
    data=frame.copy();data['_time']=pd.to_datetime(data.trade_time,errors='coerce')
    if data['_time'].isna().any():raise ValueError('分钟时间无法解析')
    data['_day']=data['_time'].dt.normalize()
    rows=[]
    for (code,day),group in data.groupby(['ts_code','_day'],sort=True):
        group=group.sort_values('_time',kind='stable')
        values=group[list(NUMERIC)].apply(pd.to_numeric,errors='coerce')
        missing=[name for name in NUMERIC if not np.isfinite(values[name].to_numpy(float)).all()]
        volume=None
        nonintegral=bool(values.vol.dropna().mod(1).ne(0).any())
        if 'vol' not in missing and not nonintegral:volume=sum(int(v) for v in values.vol)
        amount=None if 'amount' in missing else sum((Decimal(str(v)) for v in values.amount),Decimal(0))
        first,last=values.iloc[0],values.iloc[-1]
        positive=values.vol.gt(0)&values.amount.gt(0)
        known_zero=values.vol.eq(0)&values.amount.eq(0)
        unknown_flow=~(positive|known_zero)
        positions=np.flatnonzero(positive.to_numpy())
        first_trade=int(positions[0]) if len(positions) else None
        opening=values.iloc[first_trade].open if first_trade is not None else np.nan
        prior_unknown=bool(unknown_flow.iloc[:first_trade].any()) if first_trade is not None else bool(unknown_flow.any())
        negative=bool(values.vol.lt(0).any() or values.amount.lt(0).any())
        duplicates=int(group.duplicated(['_time']).sum())
        invalid_trade_prices={name:int((positive&(~np.isfinite(values[name])|values[name].le(0))).sum()) for name in ('open','high','low','close')}
        high_valid=bool(len(positions) and not unknown_flow.any() and invalid_trade_prices['high']==0)
        low_valid=bool(len(positions) and not unknown_flow.any() and invalid_trade_prices['low']==0)
        open_valid=bool(first_trade is not None and np.isfinite(opening) and opening>0 and not prior_unknown and not negative and not nonintegral and not duplicates)
        rows.append({'ts_code':code,'date':day,'frequency':frequency,'source_files':[str(source_file)],'minute_rows':len(group),
            'first_time':str(group._time.iloc[0]),'last_time':str(group._time.iloc[-1]),
            'open':float(opening) if np.isfinite(opening) else None,
            'reported_first_open':float(first.open) if np.isfinite(first.open) else None,
            'reported_first_volume':float(first.vol) if np.isfinite(first.vol) else None,
            'reported_first_amount':float(first.amount) if np.isfinite(first.amount) else None,
            'first_positive_trade_time':str(group._time.iloc[first_trade]) if first_trade is not None else None,
            'first_positive_trade_position':first_trade,'unknown_flow_before_first_trade':prior_unknown,
            'unknown_flow_rows':int(unknown_flow.sum()),'open_input_valid':open_valid,
            'positive_trade_rows':int(positive.sum()),
            **{'positive_trade_invalid_'+name+'_rows':count for name,count in invalid_trade_prices.items()},
            'high_input_valid':high_valid,'low_input_valid':low_valid,
            'reported_high_all_rows':float(values.high.max()) if 'high' not in missing else None,
            'reported_low_all_rows':float(values.low.min()) if 'low' not in missing else None,
            'close':float(last.close) if np.isfinite(last.close) else None,
            'high':float(values.loc[positive,'high'].max()) if high_valid else None,
            'low':float(values.loc[positive,'low'].min()) if low_valid else None,
            'volume':volume,'amount':float(amount) if amount is not None else None,'amount_decimal':str(amount) if amount is not None else None,
            'negative_volume_rows':int(values.vol.lt(0).sum()),'negative_amount_rows':int(values.amount.lt(0).sum()),
            'has_missing_numeric':bool(missing),'missing_numeric_fields':missing,'nonintegral_volume':nonintegral,
            'nonpositive_price_rows':int(values[['open','high','low','close']].le(0).any(axis=1).sum()),
            'duplicate_timestamp_rows':duplicates,
            'opening_row_positive_trade':bool(first.vol>0 and first.amount>0)})
    result=pd.DataFrame(rows)
    if rows:result['volume']=pd.array([row['volume'] for row in rows],dtype='Int64')
    return result


def map_minute_identity(daily,master,stocklistings,listings):
    boards={int(row.stockExID) for row in listings.itertuples() if str(row.longName).startswith('SEHK')}
    periods=stocklistings[stocklistings.StockExID.isin(boards)].copy()
    periods['_code']=periods.StockCode.map(lambda value:str(value).zfill(5)+'.HK' if pd.notna(value) and re.fullmatch(r'\d{1,5}',str(value)) else None)
    periods['_start']=pd.to_datetime(periods.FirstTradeDate,errors='coerce');periods['_end']=pd.to_datetime(periods.DelistDate,errors='coerce')
    grouped={code:group.to_dict('records') for code,group in periods.groupby('_code')}
    identities=master.set_index('issueID').security_id.to_dict();annotations=[]
    for row in daily.itertuples():
        candidates=[p for p in grouped.get(row.ts_code,[]) if pd.notna(p['_start']) and p['_start']<=row.date
                    and (pd.isna(p['DelistDate']) or (pd.notna(p['_end']) and row.date<p['_end']))]
        item={'issueID':None,'security_id':None,'exchange_code':None,'source_listing_ID':None,'candidate_listing_ids':[int(p['ID']) for p in candidates]}
        if len(candidates)!=1:item['identity_status']='ambiguous_or_missing_listing_period'
        elif int(candidates[0]['IssueID']) not in identities:item['identity_status']='not_selected_reit_issue'
        else:
            source=candidates[0];issue=int(source['IssueID'])
            item.update(issueID=issue,security_id=identities[issue],exchange_code=row.ts_code,source_listing_ID=int(source['ID']),identity_status='matched_source_period')
        annotations.append(item)
    out=pd.concat([daily.reset_index(drop=True),pd.DataFrame(annotations)],axis=1)
    out['issueID']=pd.array([row['issueID'] for row in annotations],dtype='Int64')
    return out


def _known(value):return pd.notna(value)


def compare_with_ccass(daily,ccass):
    references={}
    for row in ccass.to_dict('records'):references.setdefault((int(row['issueID']),pd.Timestamp(row['date'])),[]).append(row)
    duplicates=daily.duplicated(['issueID','date'],keep=False) & daily.issueID.notna()
    additions=[]
    for index,row in enumerate(daily.to_dict('records')):
        info={'comparison_status':None,'close_match':None,'volume_match':None,'amount_match':None,'amount_matches_reported_precision':None,
              'ccass_raw_close':None,'ccass_volume':None,'ccass_amount':None,'amount_delta_decimal':None,'amount_difference_scale':'unavailable',
              'full_day_corroboration':'not_established','source_open_usable':False}
        if row['identity_status']!='matched_source_period':info['comparison_status']='identity_gap'
        elif frequency_for_date(row['date'])!=row['frequency']:info['comparison_status']='outside_fixed_source_segment'
        elif duplicates.iloc[index]:info['comparison_status']='duplicate_daily_sources'
        elif row['date']>CCASS_CUTOFF:info['comparison_status']='minute_only_after_ccass_cutoff'
        else:
            matches=references.get((int(row['issueID']),pd.Timestamp(row['date'])),[])
            if len(matches)!=1:info['comparison_status']='missing_or_ambiguous_ccass_reference'
            else:
                ref=matches[0]
                info.update(ccass_raw_close=ref['raw_close'],ccass_volume=ref['volume'],ccass_amount=ref['amount'])
                info['close_match']=bool(_known(row['close']) and _known(ref['raw_close']) and np.float32(row['close'])==np.float32(ref['raw_close']))
                info['volume_match']=bool(_known(row['volume']) and _known(ref['volume']) and int(row['volume'])==int(ref['volume']))
                info['amount_match']=bool(_known(row['amount_decimal']) and _known(ref['amount']) and Decimal(row['amount_decimal'])==Decimal(int(ref['amount'])))
                if _known(row['amount_decimal']) and _known(ref['amount']):
                    delta=Decimal(row['amount_decimal'])-Decimal(int(ref['amount']))
                    info['amount_delta_decimal']=str(delta)
                    info['amount_difference_scale']='exact' if delta==0 else ('below_one_reported_unit' if abs(delta)<1 else 'at_least_one_reported_unit')
                    info['amount_matches_reported_precision']=abs(delta)<1
                code_match=ref['exchange_code']==row['exchange_code']
                info['comparison_status']='matched_daily_totals' if code_match and all(info[name] for name in ('close_match','volume_match','amount_match')) else 'daily_mismatch'
                if code_match and info['close_match'] and info['volume_match'] and info['amount_difference_scale']=='below_one_reported_unit':
                    info['comparison_status']='matched_at_reported_amount_precision'
                    info['full_day_corroboration']='close_volume_agree_amount_consistent_with_whole_unit_display_not_exact'
                if info['comparison_status']=='matched_daily_totals':info['full_day_corroboration']='close_volume_amount_agree_with_ccass'
        price_inputs_ok=row['open_input_valid']
        info['minute_open_input_valid']=bool(price_inputs_ok)
        info['source_open_usable']=bool(info['comparison_status'] in ('matched_daily_totals','matched_at_reported_amount_precision') and price_inputs_ok)
        additions.append(info)
    return pd.concat([daily.reset_index(drop=True),pd.DataFrame(additions)],axis=1)


def build_panel_candidate(ccass,minutes):
    lookup={}
    for row in minutes.to_dict('records'):
        if pd.notna(row['issueID']):lookup.setdefault((int(row['issueID']),pd.Timestamp(row['date'])),[]).append(row)
    records=[]
    for source in ccass.to_dict('records'):
        row=dict(source);row.update(raw_open=None,minute_reported_open=None,price_source='ccass_quotes',volume_amount_source='ccass_quotes',
            open_source=None,open_evidence='no_usable_minute_open',minute_comparison_status='not_requested_2010' if row['date']<pd.Timestamp('2011-01-01') else 'no_minute_source',
            approved_for_training=False,adjustment_applied=False,minute_source_files=[],negative_volume_rows=None,negative_amount_rows=None,has_missing_numeric=None)
        matches=lookup.get((int(row['issueID']),pd.Timestamp(row['date'])),[])
        if len(matches)==1:
            minute=matches[0];row.update(minute_reported_open=minute['open'],minute_comparison_status=minute['comparison_status'],minute_source_files=minute['source_files'],negative_volume_rows=minute['negative_volume_rows'],negative_amount_rows=minute['negative_amount_rows'],has_missing_numeric=minute['has_missing_numeric'])
            if minute['source_open_usable']:
                row.update(raw_open=minute['open'],open_source='tushare_'+minute['frequency'],open_evidence='source_open_with_ccass_daily_totals_corroboration')
        elif len(matches)>1:row['minute_comparison_status']='duplicate_daily_sources'
        records.append(row)
    for minute in minutes.to_dict('records'):
        if minute['comparison_status']!='minute_only_after_ccass_cutoff':continue
        close=minute['close'] if _known(minute['close']) and minute['close']>0 else None
        opening=minute['open'] if minute['minute_open_input_valid'] and _known(minute['open']) and minute['open']>0 else None
        volume=minute['volume'];amount=minute['amount']
        records.append({'issueID':int(minute['issueID']),'security_id':minute['security_id'],'date':minute['date'],'exchange_code':minute['exchange_code'],
            'raw_close':close,'reported_close':minute['close'],'raw_open':opening,'minute_reported_open':minute['open'],'volume':volume,'amount':amount,
            'high':minute['high'] if minute['high_input_valid'] else None,
            'low':minute['low'] if minute['low_input_valid'] else None,
            'high_low_source':'tushare_'+minute['frequency'],
            'high_input_valid':minute['high_input_valid'],'low_input_valid':minute['low_input_valid'],
            'vwap':float(amount/volume) if _known(volume) and _known(amount) and volume>0 and amount>0 else None,
            'susp':None,'newsusp':None,'noclose':None,'price_source':'tushare_minutes_only','volume_amount_source':'tushare_minutes_only',
            'open_source':'tushare_'+minute['frequency'],'open_evidence':'minute_source_only_without_ccass_verification',
            'minute_comparison_status':minute['comparison_status'],'minute_source_files':minute['source_files'],
            'negative_volume_rows':minute['negative_volume_rows'],'negative_amount_rows':minute['negative_amount_rows'],
            'has_missing_numeric':minute['has_missing_numeric'],'approved_for_training':False,'adjustment_applied':False})
    result=pd.DataFrame(records)
    result['volume']=pd.array([row['volume'] for row in records],dtype='Int64')
    for column in ('vol','turn'):
        if column in result:result[column]=pd.array([row.get(column) for row in records],dtype='UInt64')
    return result.sort_values(['date','security_id']).reset_index(drop=True)


def read_fixed_minute_sources(base):
    pieces=[];audit=[]
    for folder,frequency,left,right in [('reit_1min_raw','1min','2011-01-01','2017-06-30'),('reit_minutes_raw','60min','2017-07-01','2026-09-10')]:
        root=Path(base)/folder;state=json.loads((root/'collection_status.json').read_text(encoding='utf-8'))
        if state['status']!='finished' or state['errors'] or state['completed_partitions']!=state['planned_partitions'] or state['frequency']!=frequency:
            raise ValueError('指定分钟来源采集尚未完整结束: '+folder)
        stats={'root':str(root),'frequency':frequency,'start':left,'end':right,'data_partitions':0,'api_empty_partitions':0,'rows_read':0}
        for record in state['partitions']:
            params=record['request']['params']
            if params['freq']!=frequency or record['request']['api_name']!='hk_mins':raise ValueError('分钟缓存请求频率不匹配')
            start=pd.Timestamp(params['start_date']);end=pd.Timestamp(params['end_date'])
            if end.normalize()<pd.Timestamp(left) or start.normalize()>pd.Timestamp(right):continue
            if record['status']=='api_empty':stats['api_empty_partitions']+=1;continue
            if record['status']!='downloaded':raise ValueError('分钟分区不是已完成的原始数据')
            if start.time()!=pd.Timestamp('00:00:00').time() or end.time()!=pd.Timestamp('23:59:59').time():raise ValueError('分钟请求未覆盖完整日边界')
            path=Path(record['parquet_file'])
            if path.suffix!='.parquet':raise ValueError('不读取.partial分钟文件')
            frame=pd.read_parquet(path)
            times=pd.to_datetime(frame.trade_time,errors='coerce')
            if len(frame)!=record['rows'] or times.isna().any() or times.lt(start).any() or times.gt(end).any() or not frame.ts_code.eq(params['ts_code']).all():raise ValueError('分钟原分区计数、代码或日期不匹配')
            frame=frame.loc[times.dt.normalize().between(left,right)]
            if frame.empty:continue
            pieces.append(aggregate_minutes(frame,frequency,path.resolve()));stats['data_partitions']+=1;stats['rows_read']+=len(frame)
        audit.append(stats);print(f'aggregated {folder}: {stats}',flush=True)
    if not pieces:raise ValueError('固定来源段没有分钟记录；未改频或填充')
    return pd.concat(pieces,ignore_index=True),audit


def main():
    parser=argparse.ArgumentParser(description='固定REIT分钟来源段聚合与CCASS对照，仅独立研究输出')
    parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args();root=args.base/'references/webb_archive';ccass=root/'ccass'
    state=json.loads((ccass/'data_inventory.json').read_text(encoding='utf-8'))
    if state['status']!='finished' or state['tables']['quotes']['status']!='parsed':raise ValueError('最终CCASS报价未完成')
    master=pd.read_parquet(ccass/'reit_master.parquet')
    minute,source_audit=read_fixed_minute_sources(args.base)
    minute=map_minute_identity(minute,master,pd.read_parquet(root/'parquet/stocklistings.parquet'),pd.read_parquet(root/'parquet/listings.parquet'))
    raw=pd.read_parquet(ccass/'reit_daily_raw.parquet')
    write_research_outputs(ccass,minute,raw,source_audit)


def write_research_outputs(ccass,minute,raw,source_audit):
    minute=compare_with_ccass(minute,raw)
    panel=build_panel_candidate(raw,minute)
    minute.to_parquet(ccass/'reit_minute_daily.parquet',index=False)
    comparison_columns=['date','ts_code','security_id','issueID','frequency','identity_status','comparison_status','close_match','volume_match','amount_match','ccass_raw_close','ccass_volume','ccass_amount','close','volume','amount','amount_decimal','amount_delta_decimal','amount_difference_scale','source_files','source_open_usable','full_day_corroboration']
    comparison_columns.append('amount_matches_reported_precision')
    minute[comparison_columns].to_parquet(ccass/'reit_minute_comparison.parquet',index=False)
    panel.to_parquet(ccass/'reit_raw_panel_candidate.parquet',index=False)
    audit={'status':'unapproved_raw_panel_research','minute_daily_rows':len(minute),'panel_rows':len(panel),'source_segments':source_audit,
        'comparison_counts':{str(k):int(v) for k,v in minute.comparison_status.value_counts().items()},
        'amount_difference_scale_counts':{str(k):int(v) for k,v in minute.amount_difference_scale.value_counts().items()},
        'field_nonexact_counts':{name:int(minute[name].eq(False).sum()) for name in ('close_match','volume_match','amount_match')},
        'amount_precision_evidence':{'path':'references/hkex_dqs/vwap_range_reconciliation.json','observations':['00435 2010-06-21: sales 3387225.32; displayed turn 3387225','02778 2010-08-12: sales 19851084.46; displayed turn 19851084'],
            'interpretation':'When close and volume agree, an absolute amount difference below one reported unit is consistent with whole-unit display precision, not exact equality. The original values and exact comparison are retained. No particular rounding direction is inferred.','acceptance_absolute_difference_strictly_less_than':1},
        'negative_volume_days':int(minute.negative_volume_rows.gt(0).sum()),'negative_amount_days':int(minute.negative_amount_rows.gt(0).sum()),
        'missing_numeric_days':int(minute.has_missing_numeric.sum()),'source_open_usable_days':int(minute.source_open_usable.sum()),
        'open_from_later_first_trade_days':int(minute.first_positive_trade_position.gt(0).sum()),
        'open_input_valid_days':int(minute.open_input_valid.sum()),
        'positive_trade_invalid_high_days':int(minute.positive_trade_invalid_high_rows.gt(0).sum()),
        'positive_trade_invalid_low_days':int(minute.positive_trade_invalid_low_rows.gt(0).sum()),
        'ccass_cutoff':'2025-12-24','source_only_start':'2025-12-25','approved_for_training':False,
        'rules':['Fixed 1min/60min date segments, no automatic source replacement.','All returned rows included in totals, including negative corrections; any missing field makes that field aggregate unknown.',
                 'Open is the original open of the first strictly positive volume-and-amount row, never a later replacement for missing or nonpositive open. Unknown preceding flow, negative corrections or duplicate timestamps block its use.',
                 'Later invalid high/low on positive trades invalidate the respective daily field, not an already observed opening price. Original first-row values and all-row quality flags are retained.',
                 'Whole-day API request boundaries are checked; these alone do not establish complete market-session coverage.','CCASS closing compared at FLOAT32 precision; volume exactly. Decimal amount differences are preserved; sub-unit differences qualify only as whole-unit display-precision agreement.',
                 'Before cutoff, CCASS prices/quantities are retained; mismatching minute open is retained only as minute_reported_open, not raw_open.',
                 'After cutoff, rows are explicit Tushare-only observations without invented suspension flags or CCASS verification.',
                 '2010 has no invented open. No corporate-action adjustment or event coverage is generated. pquotes is not read or merged.']}
    (ccass/'reit_minute_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
