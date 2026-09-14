"""导入具备公开时间与来源证据的财务历史版本；不推测、不换算财务值。"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from .paths import DATA

COLUMNS = ('security_id','metric','value','period_end','published_at','version',
           'verified','source_url','unit','currency')
RATIO_METRICS = ('roe','net_margin','revenue_growth','profit_growth',
                 'operating_cashflow_to_assets','debt_to_assets')
METRIC_UNITS = {name: ('ratio','N/A') for name in RATIO_METRICS}
METRIC_UNITS.update({name: ('HKD/share','HKD') for name in ('eps_hkd','bps_hkd')})
PUBLICATION_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})')


def _validate_versions(frame):
    """完全相同的重导入保持幂等；同版本或同公开时刻的矛盾拒绝。"""
    keys = ['security_id','metric','period_end','version']
    for _, group in frame.groupby(keys,dropna=False,sort=False):
        if len(group[list(COLUMNS)].drop_duplicates()) > 1:
            raise ValueError('同一财务版本包含冲突记录')
    result = frame.drop_duplicates(keys,keep='first')
    if result.duplicated(['security_id','metric','period_end','published_at']).any():
        raise ValueError('同一财务期间公开时刻包含冲突版本')
    return result.sort_values(['security_id','metric','period_end','published_at']).reset_index(drop=True)


def import_financial_csv(path, securities):
    """校验CSV，返回可训练版本和未经核验的缺口；此函数不写文件。"""
    path = Path(path)
    frame = pd.read_csv(path,dtype=str,keep_default_na=False)
    missing = set(COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError('财务CSV缺少字段: '+','.join(sorted(missing)))
    if frame.empty:
        raise ValueError('财务CSV不能为空')
    frame = frame[list(COLUMNS)].copy()
    if frame.apply(lambda col: col.str.strip().eq('')).any().any():
        raise ValueError('财务CSV字段不能为空，不能补齐公开时间或其他证据')
    if frame.apply(lambda col: col.ne(col.str.strip())).any().any():
        raise ValueError('财务CSV字段不能带首尾空白')
    if not {'security_id','identity_status'}.issubset(securities.columns):
        raise ValueError('证券主表缺少身份核验字段')
    if securities.security_id.duplicated().any():
        raise ValueError('证券主表身份标识重复')
    known = set(securities.loc[securities.identity_status.eq('verified'),'security_id'])
    if not frame.security_id.isin(known).all():
        raise ValueError('财务CSV证券身份必须匹配已核验证券主表security_id')
    booleans = frame.verified.str.lower()
    if not booleans.isin(['true','false']).all():
        raise ValueError('verified必须明确为true或false')
    frame['verified'] = booleans.eq('true')
    frame['value'] = pd.to_numeric(frame.value,errors='coerce')
    if not np.isfinite(frame.value).all():
        raise ValueError('财务值必须为明确有限数值')
    for row in frame.itertuples():
        if row.metric not in METRIC_UNITS or (row.unit,row.currency)!=METRIC_UNITS[row.metric]:
            raise ValueError('财务指标单位币种不匹配：eps_hkd/bps_hkd用HKD/share与HKD；比例用ratio与N/A')
        source = urlsplit(row.source_url)
        if source.scheme not in ('https','http') or not source.hostname or source.username or source.password:
            raise ValueError('source_url必须为公开公告的HTTP或HTTPS来源地址')
        if not PUBLICATION_PATTERN.fullmatch(row.published_at):
            raise ValueError('published_at必须包含真实时分秒和明确时区的ISO时间')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',row.period_end):
            raise ValueError('period_end必须为YYYY-MM-DD报告期末日期')
    frame['published_at_original'] = frame.published_at
    try:
        frame['published_at'] = pd.to_datetime(frame.published_at,format='ISO8601',utc=True,errors='raise')
        frame['period_end'] = pd.to_datetime(frame.period_end,format='%Y-%m-%d',errors='raise')
    except (ValueError,TypeError) as exc:
        raise ValueError('财务日期或公开时间无效') from exc
    local_date = frame.published_at.dt.tz_convert('Asia/Hong_Kong').dt.tz_localize(None).dt.normalize()
    if frame.period_end.gt(local_date).any():
        raise ValueError('报告期末不能晚于实际公开日期')
    frame['source_file'] = str(path.resolve())
    frame['source_row'] = np.arange(2,len(frame)+2)
    frame['imported_at'] = pd.Timestamp.now(tz='UTC')
    frame = _validate_versions(frame)
    records = frame.loc[frame.verified].copy().reset_index(drop=True)
    gaps = frame.loc[~frame.verified].copy().reset_index(drop=True)
    gaps['reason'] = '财务记录未经证据核验，不进入训练'
    return {'status':'complete' if gaps.empty else 'incomplete','records':records,'gaps':gaps}


def write_financial_versions(csv_path, master_path, output_path):
    """追加核验后的版本，冲突时保留原文件；缺口写入单独CSV。"""
    output_path = Path(output_path)
    result = import_financial_csv(csv_path,pd.read_parquet(master_path))
    records = result['records']
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if not existing.empty:
            required = {*COLUMNS,'published_at_original','source_file','source_row','imported_at'}
            if not required.issubset(existing.columns):
                raise ValueError('已有财务版本缺少严格导入的来源证据字段')
            if not existing.verified.eq(True).all():
                raise ValueError('已有财务版本包含未经核验记录')
            records = _validate_versions(pd.concat([existing,records],ignore_index=True))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    records.to_parquet(output_path,index=False)
    gaps_path = output_path.with_suffix('.gaps.csv')
    result['gaps'].to_csv(gaps_path,index=False)
    return {'status':result['status'],'stored_versions':len(records),
            'verified_input_versions':len(result['records']),'gap_count':len(result['gaps']),
            'output':str(output_path),'gaps_output':str(gaps_path)}


def main():
    parser = argparse.ArgumentParser(description='导入已核验公告财务历史版本，保留实际公开时间和来源')
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--master',type=Path,default=DATA/'securities.parquet')
    parser.add_argument('--output',type=Path,default=DATA/'financial_versions.parquet')
    args = parser.parse_args()
    print(json.dumps(write_financial_versions(args.csv,args.master,args.output),ensure_ascii=False))


if __name__ == '__main__':
    main()
