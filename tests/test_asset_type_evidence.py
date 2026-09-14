import pandas as pd
from hk_quant.data import apply_asset_type_evidence


def test_positive_preference_evidence_overrides_broad_equity_without_dropping_row():
    master=pd.DataFrame(dict(security_id=['04621.HK','09626.HK','NEW.HK'],isin=['PREF','Z','NEW'],asset_type=['equity']*3))
    evidence=pd.DataFrame([dict(security_id='04621.HK',matched_isin='PREF',mapping_status='matched_unique_isin',asset_class='preference',source_url='https://example.test/source'),
        dict(security_id='09626.HK',matched_isin='Z',mapping_status='matched_unique_isin',asset_class='ordinary_share',source_url='https://example.test/source')])
    result=apply_asset_type_evidence(master,evidence).set_index('security_id')
    assert len(result)==3
    assert result.loc['04621.HK','asset_type']=='preference'
    assert result.loc['04621.HK','provider_asset_type']=='equity'
    assert result.loc['09626.HK','asset_type']=='equity'
    assert result.loc['NEW.HK','asset_type_evidence_status']=='provider_category_only'


def test_type_evidence_cannot_cross_isin_change():
    master=pd.DataFrame(dict(security_id=['00001.HK'],isin=['NEW'],asset_type=['equity']))
    evidence=pd.DataFrame([dict(security_id='00001.HK',matched_isin='OLD',mapping_status='matched_unique_isin',asset_class='preference',source_url='https://example.test/source')])
    result=apply_asset_type_evidence(master,evidence).iloc[0]
    assert result.asset_type=='equity'
    assert result.asset_type_evidence_status=='isin_mismatch'
