import numpy as np
import pandas as pd
from hk_quant.reit_data import normalize_reit_raw
from hk_quant.reit_data import align_partition_schemas


def test_raw_reference_is_not_trade_and_adjustment_is_not_invented():
    p=pd.DataFrame(dict(security_id='HKREIT:4875',issueID=4875,date=pd.date_range('2010-01-04',periods=3),
        raw_close=[10.,10.,np.nan],raw_open=[10.,np.nan,np.nan],high=[10.,np.nan,np.nan],low=[10.,np.nan,np.nan],
        volume=[1000,0,0],amount=[10000.,0.,0.],vwap=[10.,np.nan,np.nan]))
    result=normalize_reit_raw(p,pd.Series({4875:'HKD'}),pd.DataFrame(columns=['cny','usd']))
    assert result.quote_present.tolist()==[True,False,False]
    assert result.data_valid.tolist()==[True,True,False]
    assert result.raw_close.iloc[1]==10.
    assert result.adj_close_hkd.isna().all() and result.open_adj.isna().all()
    assert result.fx_to_hkd.eq(1.).all()
    assert result.security_id.eq('HKREIT:4875').all()


def test_native_currency_missing_fx_stays_missing():
    p=pd.DataFrame(dict(security_id=['87001.HK'],issueID=[6809],date=[pd.Timestamp('2026-09-11')],
        raw_close=[1.],raw_open=[1.],high=[1.],low=[1.],volume=[1000],amount=[1000.],vwap=[1.]))
    fx=pd.DataFrame({'cny':[1.1],'usd':[7.8]},index=[pd.Timestamp('2026-09-10')])
    result=normalize_reit_raw(p,pd.Series({6809:'CNY'}),fx)
    assert pd.isna(result.fx_to_hkd.iloc[0]) and pd.isna(result.amount_hkd.iloc[0])
    assert result.raw_close.iloc[0]==1.


def test_partition_schema_alignment_preserves_nulls_and_large_integers(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    large=2**53+1
    pq.write_table(pa.table({'source':[None],'quantity':pa.array([large],type=pa.int64())}),tmp_path/'2010.parquet')
    pq.write_table(pa.table({'source':['minutes'],'quantity':pa.array([None],type=pa.int64())}),tmp_path/'2026.parquet')
    audit=align_partition_schemas(tmp_path)
    actual=pq.read_table(tmp_path)
    assert actual.column('source').to_pylist()==[None,'minutes']
    assert actual.column('quantity').to_pylist()==[large,None]
    assert audit['partitions']==2
