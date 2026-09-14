"""读取发行人分派历史；历史结果证据不冒充当时已公开的预测因子。"""
import argparse
import json
import math
import re
from io import StringIO
from pathlib import Path

import pandas as pd


def _date(value):
    value=str(value).strip()
    chinese=re.fullmatch(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日',value)
    if chinese:return pd.Timestamp(year=int(chinese[1]),month=int(chinese[2]),day=int(chinese[3]))
    return pd.to_datetime(value,format='%d %b %Y')


def parse_history(table):
    if table.shape[1]!=8:raise ValueError('分派表应为八列；结构已变化，不能按旧列位解析')
    rows=[]
    for number,row in enumerate(table.itertuples(index=False,name=None)):
        year=re.match(r'^(20\d{2})',str(row[0]))
        if not year:continue
        for kind,offset in (('interim',2),('final',5)):
            raw_amount=str(row[offset]).strip()
            if raw_amount in ('-','–','nan','None'):continue
            amount=re.match(r'^([0-9]+(?:\.[0-9]+)?)',raw_amount)
            if not amount:raise ValueError('无法识别每单位分派金额: '+raw_amount)
            rows.append({'period_year':int(year[1]),'distribution_type':kind,
                         'cash_per_unit_hkd':float(amount[1]),'ex_date':_date(row[offset+1]),
                         'payment_date':_date(row[offset+2]),'source_row':number})
    result=pd.DataFrame(rows)
    if result.duplicated(['period_year','distribution_type']).any():raise ValueError('重复分派期间')
    if (result.payment_date<result.ex_date).any():raise ValueError('派付日早于除息日')
    return result


def reconcile_histories(english,chinese,issuer_reference,source_urls):
    keys=['period_year','distribution_type']
    joined=english.merge(chinese,on=keys,how='outer',suffixes=('_en','_zh'),validate='one_to_one',indicator=True)
    records=[]
    for row in joined.to_dict('records'):
        complete=row['_merge']=='both'
        agrees=complete and all(row[column+'_en']==row[column+'_zh'] for column in ('cash_per_unit_hkd','ex_date','payment_date'))
        records.append({**{key:row[key] for key in keys},'issuer_reference':issuer_reference,
            'status':'matched_issuer_records' if agrees else 'source_conflict',
            'cash_per_unit_hkd':row['cash_per_unit_hkd_en'] if agrees else None,
            'ex_date':row['ex_date_en'] if agrees else pd.NaT,
            'payment_date':row['payment_date_en'] if agrees else pd.NaT,
            'cash_currency':'HKD','announced_at':pd.NaT,'usable_for_prediction_features':False,
            'payment_basis':'issuer_final_schedule_simulated','source_urls':list(source_urls),
            'verification_scope':'issuer-recorded distribution outcome/schedule; not investor receipt or historical publication time',
            'english_evidence':{column:row[column+'_en'] for column in ('cash_per_unit_hkd','ex_date','payment_date')},
            'chinese_evidence':{column:row[column+'_zh'] for column in ('cash_per_unit_hkd','ex_date','payment_date')}})
    return pd.DataFrame(records)


def outcome_cash_actions(records,identity_periods):
    """将已核验的分派结果关联到当时港股身份；不提供预测可用时间或完整性证明。"""
    periods=identity_periods.copy()
    periods['valid_from']=pd.to_datetime(periods.valid_from)
    periods['valid_to']=pd.to_datetime(periods.valid_to)
    if periods.valid_from.isna().any() or periods.valid_to.isna().any() or periods.valid_to.lt(periods.valid_from).any():
        raise ValueError('证券身份期间必须有明确有效日期')
    events=[];gaps=[]
    for row in records.to_dict('records'):
        day=pd.Timestamp(row['ex_date'])
        matches=periods.loc[periods.issuer_reference.eq(row['issuer_reference']) & periods.verified.eq(True)
                            & periods.valid_from.le(day) & periods.valid_to.ge(day)]
        if len(matches)>1:raise ValueError('除息日对应多个已核验身份期间')
        if len(matches)==0:
            gaps.append({**row,'reason':'no_verified_hk_identity_at_ex_date'});continue
        if row['status'] not in ('matched_issuer_records','resolved_by_issuer_report'):
            gaps.append({**row,'reason':'unresolved_distribution_source'});continue
        if row['payment_basis']!='issuer_final_schedule_simulated':raise ValueError('发行人分派记录只能标记模拟到账')
        amount=float(row['cash_per_unit_hkd']);payment=pd.Timestamp(row['payment_date'])
        if not math.isfinite(amount) or amount<0 or row['cash_currency']!='HKD':raise ValueError('现金分派金额或币种无效')
        if pd.isna(day) or pd.isna(payment) or payment<day:raise ValueError('分派除息日或派付日无效')
        sources=list(row['source_urls']);identity=matches.iloc[0]
        if not sources or not all(isinstance(url,str) and url.startswith('https://') for url in sources):raise ValueError('分派缺少来源')
        if not isinstance(identity.source_url,str) or not identity.source_url.startswith('https://'):raise ValueError('身份缺少来源')
        events.append(dict(security_id=identity.security_id,action_type='cash_dividend',effective_date=day,payment_date=payment,
            cash_per_share_hkd=amount,cash_currency='HKD',verified=True,source_url=sources[-1],source_urls=sources,
            identity_source_url=identity.source_url,payment_basis='issuer_final_schedule_simulated',
            announced_at=pd.NaT,usable_for_prediction_features=False,evidence_role='historical_cash_outcome_only'))
    columns=['security_id','action_type','effective_date','payment_date','cash_per_share_hkd','cash_currency','verified',
             'source_url','source_urls','identity_source_url','payment_basis','announced_at','usable_for_prediction_features','evidence_role']
    result=pd.DataFrame(events,columns=columns)
    if result.duplicated(['security_id','effective_date','payment_date']).any():raise ValueError('同一现金分派重复，不能重复入账')
    result.attrs['coverage_complete']=False
    return result,pd.DataFrame(gaps)


def collect_cached_records(directory):
    directory=Path(directory)
    outputs=[]
    for code in ('00823','00778'):
        frames=[];urls=[]
        for suffix in ('','_zh'):
            tables=pd.read_html(StringIO((directory/f'{code}{suffix}.html').read_text(encoding='utf-8')))
            if len(tables)!=1:raise ValueError('发行人页面表格结构发生变化')
            frames.append(parse_history(tables[0]))
            urls.append(json.loads((directory/f'{code}{suffix}_source.json').read_text())['source_url'])
        records=reconcile_histories(*frames,code,urls)
        outputs.append(records)
    combined=pd.concat(outputs,ignore_index=True)
    combined.to_parquet(directory/'distribution_records.parquet',index=False)
    from .service import jsonable
    (directory/'distribution_records.json').write_text(json.dumps(jsonable(combined),ensure_ascii=False,indent=2),encoding='utf-8')
    return {'rows':len(combined),'matched':int(combined.status.eq('matched_issuer_records').sum()),
            'conflicts':int(combined.status.eq('source_conflict').sum()),'usable_for_prediction_features':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    print(json.dumps(collect_cached_records(parser.parse_args().directory),ensure_ascii=False))
