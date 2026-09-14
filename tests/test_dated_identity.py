import numpy as np
import pandas as pd
import pytest

from hk_quant.dated_identity import SCOPE, apply_bar_evidence, validate_evidence
from hk_quant.features import isolate_identity_periods, identity_period_features
from hk_quant.market_context import risk_returns_from_inputs
import test_features as feature_fixtures


def fixture():
    bars = feature_fixtures.FeatureTests().bars()
    bars['identity_date_verified'] = True
    bars['identity_source_issue_id'] = 123
    bars.loc[[2, 5], 'identity_date_verified'] = False
    bars.loc[3, 'identity_source_issue_id'] = 456
    master = pd.DataFrame([dict(security_id='A', identity_status='partially_verified',
        identity_valid_from=pd.NaT, identity_valid_to=pd.NaT,
        identity_source_issue_id=123, identity_evidence_scope=SCOPE, asset_type='equity')])
    return bars, master


def test_dated_proofs_mask_gaps_and_wrong_issue_and_do_not_become_factors():
    bars, master = fixture()
    isolated = isolate_identity_periods(bars, master.set_index('security_id'))
    assert isolated.adj_close_hkd.iloc[[2,3,5]].isna().all()
    assert isolated.adj_close_hkd.iloc[1] == bars.adj_close_hkd.iloc[1]
    calendar = pd.DatetimeIndex(bars.date)
    frame = identity_period_features(bars, calendar, master.iloc[0])
    assert frame.return_1.iloc[[2,3,4,5,6]].isna().all()
    assert pd.notna(frame.return_1.iloc[1])
    assert not any(c.startswith('identity_') for c in frame.columns)
    assert frame.loc[2,'status']=='identity_period_unresolved'
    risk = risk_returns_from_inputs(frame[['security_id','date','return_1']],master)
    assert pd.notna(risk.A.iloc[1]) and risk.A.iloc[[2,3,4,5,6]].isna().all()


def test_whitelist_does_not_extend_to_missing_sessions_or_next_year():
    bars, master=fixture()
    bars['date']=pd.bdate_range('2025-12-23',periods=len(bars))
    bars['identity_date_verified']=bars.date.lt('2026-01-01')
    missing=bars.date.iloc[1]
    frame=identity_period_features(bars.drop(index=1),pd.DatetimeIndex(bars.date),master.iloc[0])
    assert frame.loc[frame.date.eq(missing),'raw_close'].isna().all()
    assert frame.loc[frame.date.ge('2026-01-01'),'raw_close'].isna().all()


def test_conversion_uses_dated_fx_without_altering_quotes_or_missing_values():
    bars, _=fixture()
    bars['volume']=100.; bars['amount']=1000.; bars['currency']=None
    evidence=bars.iloc[:2][['security_id','date','raw_close','volume','amount']].copy()
    evidence['issueID']=123
    records=pd.DataFrame([dict(security_id='A',source_currency='CNY')]).set_index('security_id')
    fx=pd.DataFrame({'usd':[7.8], 'cny':[1.1]},index=[bars.date.iloc[0]])
    result=apply_bar_evidence(bars,evidence,records,fx)
    assert result.adj_close_hkd.iloc[0]==bars.adj_close.iloc[0]*1.1
    assert np.isnan(result.adj_close_hkd.iloc[1])
    assert not result.identity_date_verified.iloc[2]
    pd.testing.assert_frame_equal(result[['raw_close','volume','amount','open_adj']],bars[['raw_close','volume','amount','open_adj']])
    bars.loc[0,'raw_close']+=1
    with pytest.raises(ValueError,match='raw_close'):apply_bar_evidence(bars,evidence,records,fx)
    bars.loc[0,'raw_close']=np.nan
    with pytest.raises(ValueError,match='raw_close'):apply_bar_evidence(bars,evidence,records,fx)


def test_duplicate_evidence_is_rejected():
    evidence=pd.DataFrame([dict(security_id='A',date=pd.Timestamp('2025-01-01'))]*2)
    with pytest.raises(ValueError,match='逐日身份证据重复'):
        validate_evidence(pd.DataFrame({'security_id':['A']}),pd.DataFrame({'candidate_id':[1]}),evidence)


def test_missing_proof_columns_never_enable_partial_identity():
    bars, master=fixture()
    frame=isolate_identity_periods(bars.drop(columns=['identity_date_verified','identity_source_issue_id']),master.set_index('security_id'))
    assert frame.adj_close_hkd.isna().all()


def test_issue_consistency_and_original_quote_source_are_required():
    review=pd.DataFrame([dict(security_id='A',IssueID=123)])
    candidates=pd.DataFrame({'candidate_id':[1]})
    evidence=pd.DataFrame([dict(security_id='A',date=pd.Timestamp('2025-01-01'),
        issueID=456,all_match=True,quote_source='quotes',quote_present=True,volume=100,
        close_match=True,volume_match=True,amount_match=True,observed_conflict=False)])
    with pytest.raises(ValueError,match='IssueID不一致'):validate_evidence(review,candidates,evidence)
    evidence['quote_source']='pquotes'
    with pytest.raises(ValueError,match='原始正成交'):validate_evidence(review,candidates,evidence)
