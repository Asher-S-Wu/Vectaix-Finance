import pandas as pd
import pytest

from hk_quant.features import write_security_features


def test_retired_reit_and_reused_current_code_stay_separate(tmp_path):
    identities=['HKREIT:4875','00625.HK','00625!A.HK']
    for identity in identities:
        write_security_features(pd.DataFrame(dict(security_id=[identity],date=[pd.Timestamp('2020-01-02')],status=['data_issue'])),tmp_path)
    files=list(tmp_path.glob('*.parquet'))
    assert len(files)==3
    assert all(':' not in p.name for p in files)
    restored=pd.read_parquet(tmp_path)
    assert set(restored.security_id)==set(identities)
    assert len(restored)==3


def test_mixed_security_partition_rejected(tmp_path):
    with pytest.raises(ValueError,match='单一'):
        write_security_features(pd.DataFrame({'security_id':['HKREIT:4875','00625.HK']}),tmp_path)


def test_unrecognized_identity_cannot_create_arbitrary_path(tmp_path):
    with pytest.raises(ValueError,match='证券标识'):
        write_security_features(pd.DataFrame({'security_id':['../escape.HK']}),tmp_path)
    assert list(tmp_path.iterdir())==[]
