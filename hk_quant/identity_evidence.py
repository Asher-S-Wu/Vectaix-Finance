"""基于ISIN与证券级IssueID的独立身份研究，不修改正式证券主表。"""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'data/hk'
PREFERENCE_TYPES={'Cpref','Pref','CRP','PrtPrf','Prefrd','RedPrf','NVPrfd','HDRP'}
DEPOSITARY_TYPES={'IDR','HDRO','HDRP','ADS','GDR'}
ORDINARY_TYPES={'O','H','A','B','ONV','O(GBP)','C ord','X ord','Y ord','Z ord'}


def _text(value):
    if pd.isna(value):return None
    value=str(value).strip()
    return value if value else None


def _numeric_code(value):
    text=_text(value)
    return text.zfill(5) if text and re.fullmatch(r'\d{1,5}',text) else None


def compare_identity_evidence(master,current,issues,stocklistings,sectypes,listings):
    for name,frame,key in [('V4',master,'security_id'),('issue',issues,'ID1'),('stocklistings',stocklistings,'ID'),('sectypes',sectypes,'typeID'),('listings',listings,'stockExID')]:
        if frame[key].duplicated().any():raise ValueError(name+'存在重复主键')
    official=current.copy();official['_code']=official['Stock Code'].map(_numeric_code)
    if official['_code'].isna().any() or official['_code'].duplicated().any():raise ValueError('当前HKEX代码缺失或重复')
    official=official.set_index('_code')
    source=stocklistings.copy();source['_isin']=source['isin'].map(_text);source['_code']=source.StockCode.map(_numeric_code)
    isin_issues={key:sorted(set(int(v) for v in group.IssueID)) for key,group in source[source['_isin'].notna()].groupby('_isin')}
    issue_index=issues.set_index('ID1');type_index=sectypes.set_index('typeID');board_index=listings.set_index('stockExID')
    hk_boards=set(int(row.stockExID) for row in listings.itertuples() if str(row.longName).startswith('SEHK'))
    code_issues={key:sorted(set(int(v) for v in group.IssueID)) for key,group in source[source.StockExID.isin(hk_boards) & source['_code'].notna()].groupby('_code')}
    records=master.to_dict('records');existing=set(master.security_id)
    records.extend({'security_id':code+'.HK','isin':None,'asset_type':None} for code in official.index if code+'.HK' not in existing)
    mapping=[]
    for item in records:
        security=item['security_id'];match=re.match(r'^(\d{5})',security);code=match.group(1) if match else None
        registry=official.loc[code] if code in official.index and security==code+'.HK' else None
        old_isin=_text(item.get('isin'));current_isin=_text(registry['ISIN']) if registry is not None else None
        conflict=old_isin is not None and current_isin is not None and old_isin!=current_isin
        selected=None if conflict else current_isin if current_isin is not None else old_isin
        candidates=isin_issues.get(selected,[]) if selected else []
        if conflict:status='isin_conflict';basis='V4 and current HKEX ISIN disagree; neither selected'
        elif selected is None:status='missing_isin';basis='No ISIN anchor; code/name matching prohibited'
        elif not candidates:status='isin_not_in_source';basis='ISIN not present in extracted source stocklistings'
        elif len(candidates)>1:status='ambiguous_isin';basis='One ISIN maps to multiple source IssueIDs'
        elif candidates[0] not in issue_index.index:status='missing_issue_record';basis='Source listing references an absent issue record'
        else:status='matched_unique_isin';basis='Exact ISIN to unique stocklistings.IssueID; issue FK and source listing periods retained'
        issue_id=candidates[0] if status=='matched_unique_isin' else None
        type_id=None;short=None;long=None;issuer=None
        if issue_id is not None:
            issue=issue_index.loc[issue_id];type_id=int(issue.typeID);issuer=int(issue.issuer)
            if type_id in type_index.index:
                short=_text(type_index.loc[type_id,'typeShort']);long=_text(type_index.loc[type_id,'typeLong'])
        concordant=not conflict and selected is not None and selected==current_isin
        category=_text(registry['Category']) if registry is not None else None
        subcategory=_text(registry['Sub-Category']) if registry is not None and 'Sub-Category' in registry.index else None
        if concordant and category=='Real Estate Investment Trusts':asset='reit';asset_basis='Current HKEX REIT category with identical ISIN; not inferred from source board ID'
        elif short in DEPOSITARY_TYPES:asset='depositary_receipt';asset_basis='Source sectypes dictionary'
        elif short in PREFERENCE_TYPES:asset='preference';asset_basis='Source sectypes dictionary'
        elif short in ORDINARY_TYPES:asset='ordinary_share';asset_basis='Source sectypes dictionary'
        elif short is not None:asset='other_security_type';asset_basis='Source sectypes retained without broader eligibility inference'
        else:asset='unknown';asset_basis='No unique source security type'
        mapping.append({'security_id':security,'input_presence':'v4' if security in existing else 'current_hkex_only',
            'v4_isin':old_isin,'current_hkex_isin':current_isin,'matched_isin':selected,'IssueID':issue_id,
            'source_issuerID':issuer,'source_typeID':type_id,'source_typeShort':short,'source_typeLong':long,
            'candidate_issue_ids':candidates,'same_code_other_issue_ids':[value for value in code_issues.get(code,[]) if value!=issue_id],
            'mapping_status':status,'evidence_basis':basis,'asset_class':asset,'asset_class_basis':asset_basis,
            'is_ordinary_share':asset=='ordinary_share','is_preference_security':short in PREFERENCE_TYPES,
            'v4_asset_type':item.get('asset_type'),'current_hkex_category':category,'current_hkex_sub_category':subcategory,
            'source_tables':['stocklistings','issue','sectypes','listings'],'source_scope':'Webb Enigma source mapping, not full historical identity verification'})
    mapped=pd.DataFrame(mapping)
    for column in ('IssueID','source_issuerID','source_typeID'):mapped[column]=mapped[column].astype('Int64')
    matched=mapped[mapped.mapping_status.eq('matched_unique_isin')]
    associations={int(key):sorted(group.security_id.tolist()) for key,group in matched.groupby('IssueID')}
    periods=[];transitions=[]
    for row in source[source.IssueID.isin(associations)].to_dict('records'):
        issue_id=int(row['IssueID']);board=int(row['StockExID'])
        start=pd.to_datetime(row['FirstTradeDate'],errors='coerce');end=pd.to_datetime(row['DelistDate'],errors='coerce')
        status='source_interval'
        if pd.isna(start):status='missing_or_invalid_start'
        elif _text(row['DelistDate']) is not None and (pd.isna(end) or end<=start):status='invalid_end'
        periods.append({key:row[key] for key in stocklistings.columns} | {
            'mapped_security_ids':associations[issue_id],
            'source_board_short':_text(board_index.loc[board,'shortName']) if board in board_index.index else None,
            'source_board_long':_text(board_index.loc[board,'longName']) if board in board_index.index else None,
            'period_status':status,'evidence_basis':'Original stocklistings row of the unique ISIN-linked IssueID; no interval filling',
            'source_table':'stocklistings'})
    period_frame=pd.DataFrame(periods,columns=list(stocklistings.columns)+['mapped_security_ids','source_board_short','source_board_long','period_status','evidence_basis','source_table'])
    for issue_id,group in source[source.IssueID.isin(associations) & source.StockExID.isin(hk_boards) & source['2ndCtr'].eq(0)].groupby('IssueID'):
        group=group.assign(_start=pd.to_datetime(group.FirstTradeDate,errors='coerce')).dropna(subset=['_start']).sort_values(['_start','ID'])
        rows=list(group.to_dict('records'))
        for left,right in zip(rows,rows[1:]):
            end=pd.to_datetime(left['DelistDate'],errors='coerce');begin=right['_start']
            relation=('open_end_overlap_or_unknown' if pd.isna(end) else 'matching_transfer_boundary' if end==begin else 'gap' if end<begin else 'overlap')
            transitions.append({'IssueID':int(issue_id),'left_listing_ID':int(left['ID']),'right_listing_ID':int(right['ID']),
                'left_stock_code':_text(left['StockCode']),'right_stock_code':_text(right['StockCode']),
                'left_board_ID':int(left['StockExID']),'right_board_ID':int(right['StockExID']),
                'left_end':_text(left['DelistDate']),'left_final_trade':_text(left['FinalTradeDate']),
                'right_start':_text(right['FirstTradeDate']),'relation':relation})
    def counts(column):return {('_missing' if pd.isna(k) else str(k)):int(v) for k,v in mapped[column].value_counts(dropna=False).items()}
    audit={'status':'research_mapping_only','input_v4_rows':len(master),'current_hkex_rows':len(current),'mapping_rows':len(mapped),
        'source_period_rows':len(period_frame),'mapping_status_counts':counts('mapping_status'),
        'conflict_counts':{'isin_conflict':int(mapped.mapping_status.eq('isin_conflict').sum()),
                           'ambiguous_isin':int(mapped.mapping_status.eq('ambiguous_isin').sum()),
                           'matched_rows_with_recycled_code':sum(row['mapping_status']=='matched_unique_isin' and bool(row['same_code_other_issue_ids']) for row in mapping),
                           'disjoint_primary_hk_transitions':sum(row['relation']=='gap' for row in transitions)},
        'asset_class_counts':counts('asset_class'),
        'source_type_counts':counts('source_typeLong'),'current_hkex_subcategory_counts':counts('current_hkex_sub_category'),
        'gap_rows':[{field:row[field] for field in ('security_id','v4_isin','current_hkex_isin','mapping_status','candidate_issue_ids')} for row in mapping if row['mapping_status']!='matched_unique_isin'],
        'type_dictionary_gap_security_ids':matched.loc[matched.source_typeLong.isna(),'security_id'].tolist(),
        'transitions':transitions,'all_historical_identity_verified':False,'early_quote_attribution_verified':False,
        'limits':['ISIN absence or ambiguity is a gap; no matching by issuer name or recycled code.',
                  'Every stocklistings row is preserved; no min/max envelope fills discontinuous listing periods.',
                  'FinalTradeDate, DelistDate and FirstTradeDate retain distinct source meanings; matching boundary dates are evidence, not a new merged period.',
                  'Same-IssueID transfer history alone does not establish that vendor early prices labelled with the later code are genuine earlier-counter quotes.',
                  'Open-ended source periods are not automatically extended beyond the archive snapshot; source snapshot and current HKEX observations are separate.',
                  'Current REIT classification uses the current HKEX ISIN/category, including a REIT listed under a non-23 source board.',
                  'Asset classes are research taxonomy from explicit sectypes and HKEX evidence; they do not grant strategy eligibility.']}
    return mapped,period_frame,audit


def main():
    parser=argparse.ArgumentParser(description='输出独立证券身份及类型对照，不修改正式主表')
    parser.add_argument('--v4-root',type=Path,default=BASE/'universal_v4')
    parser.add_argument('--webb-root',type=Path,default=BASE/'universal/references/webb_archive')
    parser.add_argument('--hkex-csv',type=Path,default=BASE/'universal/references/reit_research/equities_and_reits.csv')
    args=parser.parse_args();source=args.webb_root/'parquet'
    mapping,periods,audit=compare_identity_evidence(pd.read_parquet(args.v4_root/'securities.parquet'),
        pd.read_csv(args.hkex_csv,dtype={'Stock Code':str}),pd.read_parquet(source/'issue.parquet'),
        pd.read_parquet(source/'stocklistings.parquet'),pd.read_parquet(source/'sectypes.parquet'),pd.read_parquet(source/'listings.parquet'))
    provenance=json.loads((args.webb_root/'provenance.json').read_text(encoding='utf-8'))
    audit['sources']={'webb':provenance,'v4_master':str((args.v4_root/'securities.parquet').resolve()),
        'current_hkex_file':str(args.hkex_csv.resolve()),'current_hkex_url':'https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx'}
    mapping['source_archive_date']=provenance['archive_snapshot_date'];periods['source_archive_date']=provenance['archive_snapshot_date']
    mapping['source_url']=provenance['archive_url'];periods['source_url']=provenance['archive_url']
    mapping.to_parquet(args.webb_root/'identity_mapping.parquet',index=False)
    periods.to_parquet(args.webb_root/'identity_periods.parquet',index=False)
    (args.webb_root/'identity_mapping_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:audit[key] for key in ('mapping_rows','source_period_rows','mapping_status_counts','asset_class_counts')},ensure_ascii=False))


if __name__=='__main__':main()
