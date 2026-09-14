"""保留REIT原始价格和币种；复权由独立时点事件流程处理。"""
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def align_partition_schemas(directory):
    """统一分区字段类型；null提升为明确类型，不补写任何字段值。"""
    paths=sorted(Path(directory).glob('*.parquet'))
    if not paths:raise ValueError('没有行情分区')
    schemas=[pq.read_schema(path) for path in paths]
    common=pa.unify_schemas(schemas,promote_options='permissive')
    changed=[]
    for path,schema in zip(paths,schemas):
        if schema.equals(common,check_metadata=False):continue
        table=pq.read_table(path)
        converted=table.cast(common.with_metadata(table.schema.metadata),safe=True)
        if table.num_rows!=converted.num_rows:raise ValueError('字段类型调整改变了行数')
        pq.write_table(converted,path)
        changed.append(path.name)
    return dict(partitions=len(paths),changed_partitions=changed,fields=len(common))


def normalize_reit_raw(panel,currencies,fx):
    result=panel.copy()
    if result.duplicated(['security_id','date']).any():raise ValueError('REIT原始日线重复')
    result['currency']=result.issueID.map(currencies)
    if not result.currency.isin(['HKD','CNY','USD']).all():raise ValueError('REIT报价币种未确认')
    result['fx_to_hkd']=np.nan
    result.loc[result.currency.eq('HKD'),'fx_to_hkd']=1.
    for currency,column in [('CNY','cny'),('USD','usd')]:
        selected=result.currency.eq(currency)
        result.loc[selected,'fx_to_hkd']=result.loc[selected,'date'].map(fx[column])
    result['amount_hkd']=result.amount*result.fx_to_hkd
    result['quote_present']=result.volume.gt(0)
    result['data_valid']=result.raw_close.gt(0)&np.isfinite(result.raw_close)
    for column in ['adj_close','adj_close_hkd','cum_adjfactor','open_adj','high_adj','low_adj',
                   'total_mv','free_mv','total_share','free_share','turnover_ratio']:
        result[column]=np.nan
    result['price_adjustment_basis']='daily_public_event_versions_required'
    result['identity_date_verified']=False
    result['approved_for_training']=False
    return result
