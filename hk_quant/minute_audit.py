"""检查已收取分钟数据的日级覆盖及量价关系；不生成可交易日线。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_daily_observations(minutes, calendar):
    frame=minutes.copy()
    frame['timestamp']=pd.to_datetime(frame.trade_time)
    frame['date']=frame.timestamp.dt.normalize()
    if frame.duplicated(['ts_code','timestamp']).any():raise ValueError('分钟证券时间重复')
    fields=['open','close','high','low','vol','amount']
    values=frame[fields].apply(pd.to_numeric,errors='coerce')
    frame['invalid_numeric']=~np.isfinite(values).all(axis=1)
    frame['negative_flow']=(values[['vol','amount']]<0).any(axis=1)
    frame[fields]=values
    days=set(pd.to_datetime(calendar))
    rows=[]
    for (code,date),group in frame.sort_values('timestamp').groupby(['ts_code','date']):
        volume=group.vol.sum(min_count=len(group))
        amount=group.amount.sum(min_count=len(group))
        ratio=amount/volume if pd.notna(volume) and volume>0 else np.nan
        low,high=group.low.min(),group.high.max()
        rows.append({'ts_code':code,'date':date,'bars':len(group),'first_timestamp':group.timestamp.iloc[0],
            'last_timestamp':group.timestamp.iloc[-1],'last_reported_close':group.close.iloc[-1],
            'volume_sum':volume,'amount_sum':amount,'amount_volume_ratio':ratio,
            'reported_low':low,'reported_high':high,'is_exchange_session':date in days,
            'invalid_numeric_bars':int(group.invalid_numeric.sum()),'negative_flow_bars':int(group.negative_flow.sum()),
            'ratio_within_reported_day_range':bool(pd.notna(ratio) and low-1e-8<=ratio<=high+1e-8),
            'approved_for_daily_training':False})
    return pd.DataFrame(rows)


def audit_collected(root,calendar_path):
    root=Path(root)
    status=json.loads((root/'collection_status.json').read_text(encoding='utf-8'))
    paths=[Path(part['parquet_file']) for part in status['partitions'] if part.get('parquet_file')]
    if not paths:raise ValueError('尚无非空分钟分区')
    minutes=pd.concat([pd.read_parquet(path) for path in paths],ignore_index=True)
    calendar=pd.read_parquet(calendar_path)
    daily=audit_daily_observations(minutes,calendar.loc[calendar.is_open.eq(1),'cal_date'])
    daily.to_parquet(root/'minute_daily_audit.parquet',index=False)
    summary={'collected_partition_count':status['completed_partitions'],'planned_partitions':status['planned_partitions'],
        'raw_minute_rows':len(minutes),'observed_security_days':len(daily),
        'non_session_security_days':int((~daily.is_exchange_session).sum()),
        'invalid_numeric_bars':int(daily.invalid_numeric_bars.sum()),
        'negative_flow_bars':int(daily.negative_flow_bars.sum()),
        'daily_ratio_outside_range':int((daily.volume_sum.gt(0)&~daily.ratio_within_reported_day_range).sum()),
        'approved_for_daily_training':False,
        'interpretation':'Coverage and arithmetic diagnostics only; no inference of suspension, raw-price adjustment basis, currency, complete intraday coverage or a tradable VWAP.'}
    (root/'minute_daily_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--calendar',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit_collected(args.root,args.calendar),ensure_ascii=False))
