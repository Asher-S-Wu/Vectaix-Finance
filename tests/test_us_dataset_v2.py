import pandas as pd
import numpy as np
import pytest
from us_quant.dataset_v2 import apply_gap_fills, assemble_panel
from tests.test_us_data_features import bars


def test_gap_fills_are_explicit_non_overwriting_rows():
    a=pd.DataFrame({'date':['2020-01-02'],'close':[10.]});b=pd.DataFrame({'date':['2020-01-03'],'close':[11.]})
    result=apply_gap_fills(a,b);assert len(result)==2
    with pytest.raises(ValueError,match='overwrite'):apply_gap_fills(a,a)


def test_v2_panel_keeps_all_cohort_identities_and_retired_gaps():
    f=bars();f.security_id='A';r=f.iloc[:300].copy();r.security_id='RETIRED'
    features,normalized,prices,audit=assemble_panel({'A':f,'RETIRED':r},['A','RETIRED','UNRESOLVED'],f.date)
    assert len(features)==1200 and len(normalized)==1200 and len(prices)==1200
    retired=features.loc[features.security_id.eq('RETIRED')].reset_index(drop=True)
    assert retired.loc[300:,'fwd_return_1'].isna().all()
    assert pd.isna(retired.loc[299,'fwd_return_1'])
    assert features.loc[features.security_id.eq('UNRESOLVED'),'status'].eq('source_history_unavailable').all()
    assert len(audit)==3
