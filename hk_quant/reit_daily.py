"""REIT来源日线研究：保留IssueID/柜台/原始标志，不复权、不补开盘价。"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'data/hk/universal'


def _text(value):
    if pd.isna(value):return None
    return str(value).strip() or None


def _hk_boards(listings):
    return set(int(row.stockExID) for row in listings.itertuples() if str(row.longName).startswith('SEHK'))


def build_reit_master(stocklistings,issues,current,listings):
    hk=stocklistings[stocklistings.StockExID.isin(_hk_boards(listings))]
    historic_ids=set(int(x) for x in stocklistings.loc[stocklistings.StockExID.eq(23),'IssueID'])
    current_rows=current[current.Category.eq('Real Estate Investment Trusts')]
    by_isin={key:sorted(set(int(x) for x in group.IssueID)) for key,group in stocklistings.assign(_isin=stocklistings['isin'].map(_text)).dropna(subset=['_isin']).groupby('_isin')}
    current_ids={};gaps=[]
    for row in current_rows.to_dict('records'):
        isin=_text(row['ISIN']);matches=by_isin.get(isin,[]) if isin else []
        if len(matches)!=1:
            gaps.append({'scope':'master','current_code':str(row['Stock Code']),'reason':'当前REIT的ISIN缺失、未匹配或多义','candidate_issue_ids':matches});continue
        current_ids.setdefault(matches[0],[]).append(row)
    universe=historic_ids|set(current_ids);records=[]
    source=issues.set_index('ID1')
    for issue in sorted(universe):
        periods=hk[hk.IssueID.eq(issue)];current_matches=current_ids.get(issue,[])
        identity=None;name=None;isin=None
        if len(current_matches)==1:
            row=current_matches[0];identity=str(row['Stock Code']).zfill(5)+'.HK';name=row['Name of Securities'];isin=_text(row['ISIN']);scope='current_hkex_reit_by_isin'
        elif len(current_matches)>1:
            scope='unresolved_current_codes';gaps.append({'scope':'master','issueID':issue,'reason':'同一REIT对应多个当前代码，未选取其一'})
        elif not periods.empty and periods.DelistDate.notna().all():
            identity='HKREIT:'+str(issue);scope='historical_reit_issue_namespace'
        else:
            scope='unresolved_current_or_historical';gaps.append({'scope':'master','issueID':issue,'reason':'源REIT没有已闭合上市期，也未唯一对应当前HKEX REIT'})
        if issue not in source.index:raise ValueError('REIT源IssueID缺少issue记录')
        records.append({'issueID':issue,'security_id':identity,'name':name,'current_hkex_isin':isin,'record_scope':scope,
                        'source_typeID':int(source.loc[issue,'typeID']),'source_issuerID':int(source.loc[issue,'issuer']),
                        'source_listing_ids':[int(x) for x in periods.ID],'source_stock_codes':periods.StockCode.map(_text).tolist(),
                        'evidence_basis':'Historical StockExID23 union current HKEX REIT exact unique ISIN; no recycled-code/name matching'})
    return pd.DataFrame(records),gaps


def transform_reit_quotes(quotes,master,stocklistings,listings,start='2010-01-01',quote_table='quotes'):
    if quote_table not in ('quotes','pquotes'):raise ValueError('只能处理quotes或pquotes')
    q=quotes.loc[quotes.issueID.isin(master.issueID)].copy()
    dates=pd.to_datetime(q.atDate,errors='coerce')
    gaps=[{'scope':quote_table,'issueID':int(row.issueID),'atDate':str(row.atDate),'reason':'无法解析源日期'} for row in q.loc[dates.isna()].itertuples()]
    q=q.loc[dates.notna() & dates.ge(pd.Timestamp(start))].reset_index(drop=True)
    q['date']=pd.to_datetime(q.atDate)
    q=q.merge(master[['issueID','security_id','record_scope']],on='issueID',how='left',validate='many_to_one')
    q['quote_table']=quote_table;q['reported_close']=q.closing
    meaningful=q.noclose.eq(0) & q.closing.gt(0) & np.isfinite(q.closing)
    q['raw_close']=q.closing.where(meaningful)
    q['volume']=q.vol.astype('UInt64');q['amount']=q.turn.astype('UInt64')
    traded=q.vol.gt(0) & q.turn.gt(0)
    q['positive_trade_totals']=traded;q['vwap']=np.nan
    q.loc[traded,'vwap']=q.loc[traded,'turn'].astype(float)/q.loc[traded,'vol'].astype(float)
    q['price_status']=np.where(meaningful,'meaningful_source_close','missing_meaningful_close')
    periods=stocklistings.loc[stocklistings.StockExID.isin(_hk_boards(listings))].copy()
    periods['_start']=pd.to_datetime(periods.FirstTradeDate,errors='coerce');periods['_end']=pd.to_datetime(periods.DelistDate,errors='coerce')
    grouped={int(key):group.to_dict('records') for key,group in periods.groupby('IssueID')}
    annotations=[]
    for row in q.itertuples():
        annotation={'StockCode':None,'exchange_code':None,'source_listing_ID':None,'source_StockExID':None,'listing_candidate_ids':[]}
        if quote_table=='pquotes':
            annotation['identity_period_status']='parallel_counter_code_unresolved'
            annotations.append(annotation);continue
        candidates=[period for period in grouped.get(int(row.issueID),[]) if pd.notna(period['_start']) and period['_start']<=row.date
                    and (pd.isna(period['DelistDate']) or (pd.notna(period['_end']) and row.date<period['_end']))]
        annotation['listing_candidate_ids']=[int(p['ID']) for p in candidates]
        if len(candidates)!=1:
            annotation['identity_period_status']='ambiguous_listing_period' if candidates else 'outside_source_listing_period'
        else:
            source=candidates[0];code=_text(source['StockCode'])
            if code and re.fullmatch(r'\d{1,5}',code):
                annotation.update(StockCode=code,exchange_code=code.zfill(5)+'.HK',source_listing_ID=int(source['ID']),
                                  source_StockExID=int(source['StockExID']),identity_period_status='matched_source_listing_period')
            else:annotation['identity_period_status']='missing_source_stock_code'
        if annotation['identity_period_status']!='matched_source_listing_period':
            gaps.append({'scope':quote_table,'issueID':int(row.issueID),'atDate':row.atDate,
                         'reason':annotation['identity_period_status'],'listing_candidate_ids':annotation['listing_candidate_ids']})
        annotations.append(annotation)
    fields=['StockCode','exchange_code','source_listing_ID','source_StockExID','listing_candidate_ids','identity_period_status']
    q=pd.concat([q.reset_index(drop=True),pd.DataFrame(annotations,columns=fields)],axis=1)
    q['source_listing_ID']=q.source_listing_ID.astype('Int64');q['source_StockExID']=q.source_StockExID.astype('Int64')
    q['counter_scope']='normal_source_quotes' if quote_table=='quotes' else 'parallel_quotes_not_mapped_to_normal_counter'
    return q,gaps


def _value(value):return None if pd.isna(value) else value


def compare_dqs_2010(daily,dqs):
    raw=daily.loc[daily.date.dt.year.eq(2010) & daily.exchange_code.notna()]
    official=dqs.loc[pd.to_datetime(dqs.date).dt.year.eq(2010)]
    left={};right={}
    for row in raw.to_dict('records'):left.setdefault((pd.Timestamp(row['date']),row['exchange_code']),[]).append(row)
    for row in official.to_dict('records'):right.setdefault((pd.Timestamp(row['date']),row['exchange_code']),[]).append(row)
    comparisons=[]
    for key in sorted(set(left)|set(right)):
        a=left.get(key,[]);b=right.get(key,[])
        row={'date':key[0],'exchange_code':key[1],'ccass_rows':len(a),'dqs_rows':len(b),
             'comparison_status':'paired' if len(a)==len(b)==1 else 'missing_or_ambiguous_key'}
        if row['comparison_status']!='paired':comparisons.append(row);continue
        a=a[0];b=b[0];closing=_value(a['raw_close']);reference=_value(b['raw_close'])
        price_equal=(closing is None and reference is None) or (closing is not None and reference is not None and bool(np.float32(closing)==np.float32(reference)))
        volume=_value(a['volume']);dqs_volume=_value(b['volume']);amount=_value(a['amount']);dqs_amount=_value(b['amount'])
        explicit_suspension=b['status']=='exchange_reported_suspension'
        row.update(security_id=a['security_id'],issueID=int(a['issueID']),reported_ccass_close=float(a['reported_close']),
                   ccass_raw_close=closing,dqs_raw_close=reference,ccass_volume=volume,dqs_volume=dqs_volume,
                   ccass_amount=amount,dqs_amount=dqs_amount,ccass_susp=int(a['susp']),ccass_newsusp=int(a['newsusp']),
                   ccass_noclose=int(a['noclose']),dqs_status=b['status'],
                   close_equal_float32=bool(price_equal),volume_equal=volume==dqs_volume,amount_equal=amount==dqs_amount,
                   suspension_marker_equal=bool(a['susp']==int(explicit_suspension)),dqs_source_file=b.get('source_file'),
                   dqs_quote_marker=b.get('quote_marker'))
        comparisons.append(row)
    frame=pd.DataFrame(comparisons)
    paired=frame[frame.comparison_status.eq('paired')]
    differences={name:int(paired[column].eq(False).sum()) for name,column in [('close','close_equal_float32'),('volume','volume_equal'),('amount','amount_equal'),('suspension_marker','suspension_marker_equal')]} if not paired.empty else {}
    audit={'dqs_input_rows':len(official),'ccass_2010_rows_with_matched_code':len(raw),'paired_keys':len(paired),
           'unpaired_or_ambiguous_keys':int(frame.comparison_status.ne('paired').sum()),'difference_counts':differences,
           'price_comparison':'Exact equality after converting both nonmissing closing values to source FLOAT32 precision; both missing equal.',
           'null_comparison':'Missing DQS quantity/amount is not silently equated to source zero.',
           'suspension_comparison':'Source susp flag versus explicit printed DQS suspension marker. Numeric DQS rows without the marker do not independently rule out intraday suspension.',
           'differences_corrected':False}
    return frame,audit


def main():
    parser=argparse.ArgumentParser(description='REIT原始日线研究整理与DQS核对，不写正式行情')
    parser.add_argument('--webb-root',type=Path,default=BASE/'references/webb_archive')
    parser.add_argument('--hkex-csv',type=Path,default=BASE/'references/reit_research/equities_and_reits.csv')
    parser.add_argument('--dqs',type=Path,default=BASE/'references/hkex_dqs/reit_2010_daily_observations.parquet')
    parser.add_argument('--start',default='2010-01-01')
    args=parser.parse_args();root=args.webb_root;ccass=root/'ccass';source=root/'parquet'
    state=json.loads((ccass/'data_inventory.json').read_text(encoding='utf-8'))
    if state['status']!='finished' or any(state['tables'][table]['status']!='parsed' for table in ('quotes','pquotes')):
        raise ValueError('CCASS正常解析尚未完成；不读取.partial或另起解析')
    stocklistings=pd.read_parquet(source/'stocklistings.parquet');issues=pd.read_parquet(source/'issue.parquet');listings=pd.read_parquet(source/'listings.parquet')
    current=pd.read_csv(args.hkex_csv,dtype={'Stock Code':str})
    master,master_gaps=build_reit_master(stocklistings,issues,current,listings)
    ids=master.issueID.tolist();start=pd.Timestamp(args.start).strftime('%Y-%m-%d')
    outputs={};gaps={}
    for table in ('quotes','pquotes'):
        file=ccass/'parquet'/f'{table}.parquet'
        if not file.is_file():raise ValueError('缺少最终Parquet: '+str(file))
        quotes=pd.read_parquet(file,filters=[('issueID','in',ids),('atDate','>=',start)])
        outputs[table],gaps[table]=transform_reit_quotes(quotes,master,stocklistings,listings,start,table)
    provenance=json.loads((ccass/'provenance.json').read_text(encoding='utf-8'))
    for table,frame in outputs.items():frame['source_archive_url']=provenance['archive_url'];frame['source_table']=table
    dqs_state=json.loads((args.dqs.parent/'import_status.json').read_text(encoding='utf-8'))
    if dqs_state['status']!='complete':raise ValueError('DQS最新导入未完成，不能比较旧的或写入中的数据')
    comparison,dqs_audit=compare_dqs_2010(outputs['quotes'],pd.read_parquet(args.dqs))
    dqs_audit['import_updated_at']=dqs_state['updated_at']
    outputs['quotes'].to_parquet(ccass/'reit_daily_raw.parquet',index=False)
    outputs['pquotes'].to_parquet(ccass/'reit_parallel_quotes.parquet',index=False)
    master.to_parquet(ccass/'reit_master.parquet',index=False)
    comparison.to_parquet(ccass/'reit_dqs_2010_comparison.parquet',index=False)
    counts=[]
    for issue,group in outputs['quotes'].groupby('issueID'):
        counts.append({'issueID':int(issue),'rows':len(group),'first_date':group.date.min().date().isoformat(),'last_date':group.date.max().date().isoformat(),
                       'missing_meaningful_close':int(group.raw_close.isna().sum()),'source_susp_rows':int(group.susp.eq(1).sum()),
                       'positive_trade_rows':int(group.positive_trade_totals.sum())})
    audit={'status':'raw_source_research_only','start':start,'master_rows':len(master),'normal_rows':len(outputs['quotes']),
           'parallel_rows':len(outputs['pquotes']),'master_gaps':master_gaps,'identity_period_gaps':gaps['quotes'],
           'parallel_counter_note':'pquotes is retained separately with no claimed actual temporary counter code.',
           'securities':counts,'dqs_comparison':dqs_audit,'source':provenance,
           'value_rules':'Source integers and susp/newsusp/noclose unchanged; closing<=0 or noclose implies missing raw_close. VWAP calculated only when both source vol and turn are positive.',
           'no_adjustment_or_event_coverage_created':True,'no_open_price_created':True,'v4_modified':False}
    (ccass/'reit_daily_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('master_rows','normal_rows','parallel_rows','dqs_comparison')},ensure_ascii=False))


if __name__=='__main__':main()
