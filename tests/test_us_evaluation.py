import numpy as np
import pandas as pd
from us_quant.evaluation import block_bootstrap, evaluate


def test_block_bootstrap_preserves_missing_slot_and_is_deterministic():
    a=block_bootstrap([.1,np.nan,.2,.3],block=2,repetitions=100)
    assert a==block_bootstrap([.1,np.nan,.2,.3],block=2,repetitions=100)
    assert a[0]<=.2<=a[1]


def test_missing_outcomes_remain_in_prediction_denominator():
    rows=[]
    for h in (1,5,20,60):
        for s in range(22):rows.append(dict(date=pd.Timestamp('2025-01-02'),security_id=str(s),horizon=h,label_end=pd.Timestamp('2025-04-01'),fwd_return=s/100 if s<20 else np.nan,score=s,score_status='ok',probability_status='ok',probability_up=.6,baseline_probability=.5,interval_status='ok',q10=-.1,q50=.1,q90=.3,baseline_q10=-.1,baseline_q50=0,baseline_q90=.1,expected_return_status='ok',expected_return=.1))
    result,daily=evaluate(pd.DataFrame(rows),'2025-12-31')
    assert result['horizons']['20']['predictions']==22
    assert result['horizons']['20']['mature_labels']==20
    assert result['horizons']['20']['missing_or_immature_outcomes']==2
    assert result['horizons']['20']['rank_ic_mean']==1.


def test_default_bootstrap_declines_underpowered_session_history():
    assert block_bootstrap([.1,.2])==[None,None]
    assert block_bootstrap([.1]*59+[np.nan])==[None,None]
