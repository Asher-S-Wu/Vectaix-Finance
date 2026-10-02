import pandas as pd
from scripts.run_us_replay import cohort_benchmark_scores


def test_benchmark_keeps_original_identity_cash_slots():
    f=pd.DataFrame({'date':pd.to_datetime(['2025-01-02']*2),'security_id':['A','MISSING'],'status':['ok','source_history_unavailable']})
    p=cohort_benchmark_scores(f)
    assert len(p)==2
    assert p.loc[p.security_id.eq('MISSING'),'score_status'].iloc[0]!='ok'
    assert p.horizon.eq(20).all()
