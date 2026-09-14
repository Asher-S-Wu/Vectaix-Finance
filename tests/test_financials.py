import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hk_quant.financials import import_financial_csv, write_financial_versions
from hk_quant.data import financial_asof


def master():
    return pd.DataFrame([{'security_id':'00001.HK','identity_status':'verified'}])


def row(**changes):
    record=dict(security_id='00001.HK',metric='eps_hkd',value='1.25',period_end='2023-12-31',published_at='2024-03-20T18:59:59+08:00',version='1',verified='true',source_url='https://www.hkexnews.hk/final.pdf',unit='HKD/share',currency='HKD')
    record.update(changes)
    return record


def csv(tmp_path,records):
    path=tmp_path/'financial.csv'
    pd.DataFrame(records).to_csv(path,index=False)
    return path


@pytest.mark.parametrize('changes',[
    {'published_at':'2024-03-20'}, {'published_at':'2024-03-20T18:00:00'},
    {'published_at':'2024-03-20T18:00+08:00'}, {'published_at':''},
    {'source_url':''}, {'source_url':'file:///announcement.pdf'},
    {'unit':'HKD'}, {'currency':'USD'}, {'value':'inf'}, {'value':''},
    {'version':''}, {'verified':'unknown'}, {'metric':'made_up'},
    {'security_id':'99999.HK'}, {'period_end':'2024-12-31'},
    {'metric':'roe','unit':'percent','currency':'N/A'},
    {'metric':'roe','unit':'ratio','currency':'HKD'},
])
def test_rejects_incomplete_or_unusable_evidence(tmp_path,changes):
    with pytest.raises(ValueError):
        import_financial_csv(csv(tmp_path,[row(**changes)]),master())


def test_identity_must_be_verified(tmp_path):
    securities=master();securities['identity_status']='unresolved'
    with pytest.raises(ValueError,match='身份'):
        import_financial_csv(csv(tmp_path,[row()]),securities)


def test_verified_values_keep_source_and_exact_time(tmp_path):
    source=csv(tmp_path,[row(),row(metric='roe',unit='ratio',currency='N/A',value='.12')])
    result=import_financial_csv(source,master())
    assert result['status']=='complete' and result['gaps'].empty
    records=result['records']
    assert records.published_at.iloc[0]==pd.Timestamp('2024-03-20T10:59:59Z')
    assert records.published_at_original.iloc[0]=='2024-03-20T18:59:59+08:00'
    assert records.source_file.iloc[0]==str(source.resolve())
    assert records.source_row.tolist()==[2,3]
    assert records.source_url.str.startswith('https://www.hkexnews.hk/').all()
    assert records.loc[records.metric=='roe','value'].iloc[0]==.12


def test_unverified_values_are_gaps_and_never_training_records(tmp_path):
    result=import_financial_csv(csv(tmp_path,[row(verified='false')]),master())
    assert result['status']=='incomplete' and result['records'].empty
    assert len(result['gaps'])==1 and result['gaps'].reason.iloc[0]


@pytest.mark.parametrize('field,value',[('value','2.0'),('published_at','2024-03-21T18:00:00+08:00'),('source_url','https://www.hkexnews.hk/changed.pdf')])
def test_conflicting_same_version_is_rejected(tmp_path,field,value):
    with pytest.raises(ValueError,match='冲突'):
        import_financial_csv(csv(tmp_path,[row(),row(**{field:value})]),master())


def test_same_instant_different_versions_cannot_create_ambiguous_value(tmp_path):
    with pytest.raises(ValueError,match='冲突'):
        import_financial_csv(csv(tmp_path,[row(),row(version='2',value='2')]),master())


def test_revision_is_visible_only_after_its_actual_publication(tmp_path):
    source=csv(tmp_path,[row(),row(version='2',value='2',published_at='2024-03-21T19:00:01+08:00')])
    records=import_financial_csv(source,master())['records']
    observations=pd.DataFrame({'date':pd.to_datetime(['2024-03-19','2024-03-20','2024-03-21','2024-03-22']),'security_id':'00001.HK'})
    result=financial_asof(observations,records)
    assert np.isnan(result.eps_hkd.iloc[0])
    assert result.eps_hkd.iloc[1:].tolist()==[1.25,1.25,2.0]


@pytest.mark.parametrize('timestamp,expected',[('2024-03-20T11:00:00Z',1.25),('2024-03-20T11:00:01Z',np.nan)])
def test_hong_kong_19_cutoff_uses_explicit_timezone(tmp_path,timestamp,expected):
    records=import_financial_csv(csv(tmp_path,[row(published_at=timestamp)]),master())['records']
    observations=pd.DataFrame({'date':pd.to_datetime(['2024-03-20']),'security_id':'00001.HK'})
    value=financial_asof(observations,records).eps_hkd.iloc[0]
    assert (np.isnan(value) if np.isnan(expected) else value==expected)


def test_reimports_preserve_history_and_reject_conflicts_before_writing(tmp_path):
    securities=tmp_path/'master.parquet';master().to_parquet(securities,index=False)
    output=tmp_path/'financial_versions.parquet'
    first=csv(tmp_path,[row()]);write_financial_versions(first,securities,output)
    second=csv(tmp_path,[row(version='2',value='2',published_at='2024-03-22T18:00:00+08:00')])
    write_financial_versions(second,securities,output)
    write_financial_versions(second,securities,output)
    saved=pd.read_parquet(output)
    assert saved.version.tolist()==['1','2'] and saved.value.tolist()==[1.25,2.0]
    conflicting=csv(tmp_path,[row(version='2',value='3',published_at='2024-03-22T18:00:00+08:00')])
    with pytest.raises(ValueError,match='冲突'):
        write_financial_versions(conflicting,securities,output)
    pd.testing.assert_frame_equal(saved,pd.read_parquet(output))


def test_cli_writes_only_verified_records_and_gap_report(tmp_path):
    securities=tmp_path/'master.parquet';master().to_parquet(securities,index=False)
    output=tmp_path/'financial_versions.parquet'
    source=csv(tmp_path,[row(),row(metric='roe',unit='ratio',currency='N/A',value='.2',verified='false')])
    completed=subprocess.run([sys.executable,'-m','hk_quant.financials','--csv',str(source),'--master',str(securities),'--output',str(output)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert pd.read_parquet(output).metric.tolist()==['eps_hkd']
    assert len(pd.read_csv(output.with_suffix('.gaps.csv')))==1

def test_fractional_second_after_cutoff_remains_unavailable(tmp_path):
    records=import_financial_csv(csv(tmp_path,[row(published_at='2024-03-20T19:00:00.001+08:00')]),master())['records']
    observations=pd.DataFrame({'date':pd.to_datetime(['2024-03-20','2024-03-21']),'security_id':'00001.HK'})
    values=financial_asof(observations,records).eps_hkd
    assert np.isnan(values.iloc[0]) and values.iloc[1]==1.25


def test_new_revision_of_old_period_does_not_replace_newer_period(tmp_path):
    records=import_financial_csv(csv(tmp_path,[row(),row(period_end='2024-06-30',published_at='2024-08-20T18:00:00+08:00',value='2'),row(version='2',published_at='2025-03-20T18:00:00+08:00',value='9')]),master())['records']
    observations=pd.DataFrame({'date':pd.to_datetime(['2024-03-21','2024-08-21','2025-03-21']),'security_id':'00001.HK'})
    assert financial_asof(observations,records).eps_hkd.tolist()==[1.25,2.,2.]
